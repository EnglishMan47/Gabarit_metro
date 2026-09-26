"""Model management: evaluation runs and detector versions in MLflow.

The detector has no trained weights - its "model" is the code plus the
parameter set (``config/gabarit.yaml``). Every parameter set is registered in
the MLflow Model Registry as a version of ``gabarit-detector`` (a pyfunc model
that runs the pipeline over a bag), and every evaluation of a recording is an
MLflow run with the parameters and quality metrics (events, detection range,
latency). Without MLflow installed the runs are kept in a local JSON journal,
so the admin panel works either way.

Tracking server (local):  python -m gabarit mlflow   ->  http://localhost:5000
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from gabarit.config import Config
from gabarit.pipeline import Pipeline
from gabarit.rosbag import BagInfo, BagReader

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "artifacts" / "mlflow"
LOCAL_RUNS = ROOT / "artifacts" / "admin" / "runs.json"
MODEL_NAME = "gabarit-detector"
EXPERIMENT = "gabarit-evaluation"


def tracking_uri() -> str:
    return os.environ.get("GABARIT_MLFLOW_TRACKING_URI") or f"sqlite:///{(STORE / 'mlflow.db').as_posix()}"


def ui_url() -> str:
    return os.environ.get("GABARIT_MLFLOW_URL", "http://localhost:5000")


def _mlflow():
    try:
        import mlflow  # noqa: F401
        return mlflow
    except ImportError:
        return None


def available() -> bool:
    return _mlflow() is not None


def flat_params(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(prefix: str, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                walk(f"{prefix}{k}.", v)
            else:
                out[f"{prefix}{k}"] = v
    walk("", cfg.to_dict())
    return out


# ---------------------------------------------------------------- evaluation
def evaluate_bag(info: BagInfo, cfg: Config, progress: Callable[[int, int], None] | None = None,
                 stop: Callable[[], bool] | None = None) -> dict:
    """Run the detector over a recording; quality and speed metrics."""
    reader = BagReader(info)
    pipe = Pipeline(cfg)
    n = len(reader)
    total, alarm, check = [], 0, 0
    events: dict[int, dict] = {}
    vmax = 0.0
    try:
        for i, ts, raw in reader.iterate():
            if stop is not None and stop():
                break
            d = pipe.process_raw(raw, ts * 1e-9, frame_index=i).data
            total.append(d["timing_ms"]["total"])
            alarm += int(d["obstacle"])
            check += int(d["status"] == "CHECK")
            vmax = max(vmax, d["speed_kmh"])
            for ob in d["obstacles"]:
                e = events.setdefault(ob["track_id"], {"track_id": ob["track_id"], "first_frame": i,
                                                       "first_distance_m": ob["distance_m"], "frames": 0})
                e["frames"] += 1
            if progress is not None and (i % 10 == 0 or i == n - 1):
                progress(i + 1, n)
    finally:
        reader.close()
    t = np.array(total) if total else np.zeros(1)
    ev = sorted(events.values(), key=lambda e: e["first_frame"])
    return {
        "bag": info.id, "bag_name": info.name, "frames": len(total),
        "metrics": {
            "alarm_frames": alarm, "check_frames": check, "events": len(ev),
            "first_detection_m": max((e["first_distance_m"] for e in ev), default=0.0),
            "latency_p50_ms": round(float(np.percentile(t, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(t, 95)), 2),
            "max_speed_kmh": round(vmax, 1),
        },
        "events": ev,
    }


# ---------------------------------------------------------------- journal
def _local_runs() -> list[dict]:
    try:
        return json.loads(LOCAL_RUNS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _save_local(entry: dict) -> None:
    LOCAL_RUNS.parent.mkdir(parents=True, exist_ok=True)
    runs = _local_runs()
    runs.append(entry)
    LOCAL_RUNS.write_text(json.dumps(runs[-200:], ensure_ascii=False, indent=1), encoding="utf-8")


def _prepare(ml) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    ml.set_tracking_uri(tracking_uri())
    if ml.get_experiment_by_name(EXPERIMENT) is None:
        loc = None if tracking_uri().startswith("http") else (STORE / "artifacts").resolve().as_uri()
        ml.create_experiment(EXPERIMENT, artifact_location=loc)
    ml.set_experiment(EXPERIMENT)


def log_evaluation(result: dict, cfg: Config, config_version: str | None = None) -> dict:
    """Log one evaluation; returns the journal entry (with the MLflow run id if any)."""
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "bag": result["bag_name"], "frames": result["frames"],
             "metrics": result["metrics"], "config_version": config_version, "run_id": None}
    ml = _mlflow()
    if ml is not None:
        try:
            _prepare(ml)
            with ml.start_run(run_name=f"eval {result['bag_name']}") as run:
                ml.set_tags({"bag": result["bag_name"], "config_version": str(config_version or "")})
                ml.log_params({k: str(v)[:250] for k, v in flat_params(cfg).items()})
                ml.log_metrics({k: float(v) for k, v in result["metrics"].items()})
                ml.log_dict(result["events"], "events.json")
                ml.log_text(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), "config.yaml")
                entry["run_id"] = run.info.run_id
        except Exception as err:  # the evaluation itself must not fail because of the tracker
            entry["mlflow_error"] = str(err)[:300]
    _save_local(entry)
    return entry


def recent_runs(n: int = 20) -> list[dict]:
    return list(reversed(_local_runs()))[:n]


# ---------------------------------------------------------------- registry
def _pyfunc_class():
    """The pyfunc model class (defined lazily: mlflow is an optional dependency)."""
    import mlflow.pyfunc

    class GabaritDetector(mlflow.pyfunc.PythonModel):
        """Runs the obstacle detector over ROS 2 bags.

        Input: DataFrame with a column ``bag_path`` (bag directory or .db3).
        Output: one row per frame - bag, frame, status, obstacle, distance_m, free_m.
        """

        def load_context(self, context):
            from gabarit.config import Config as _Cfg
            self.cfg = _Cfg.load(context.artifacts["config"])

        def predict(self, context, model_input, params=None):
            import pandas as pd
            from gabarit.pipeline import Pipeline as _Pipe
            from gabarit.rosbag import BagReader as _Reader, open_bag as _open
            rows = []
            for path in model_input["bag_path"]:
                reader = _Reader(_open(path))
                pipe = _Pipe(self.cfg)
                try:
                    for i, ts, raw in reader.iterate():
                        d = pipe.process_raw(raw, ts * 1e-9, frame_index=i).data
                        rows.append({"bag": str(path), "frame": i, "status": d["status"], "obstacle": d["obstacle"],
                                     "distance_m": d["nearest_distance_m"], "free_m": d["free_distance_m"]})
                finally:
                    reader.close()
            return pd.DataFrame(rows)

    return GabaritDetector


def register_detector(cfg: Config, note: str = "") -> dict:
    """Register the parameter set as a new version of the detector model."""
    ml = _mlflow()
    snap = ROOT / "artifacts" / "admin" / "configs" / f"config-{time.strftime('%Y%m%d-%H%M%S')}.yaml"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    entry: dict[str, Any] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "config_file": snap.name, "note": note,
                             "model": MODEL_NAME, "version": None}
    if ml is None:
        entry["mlflow_error"] = "MLflow не установлен: версия сохранена только как файл параметров"
        return entry
    try:
        _prepare(ml)
        with ml.start_run(run_name=f"register {snap.stem}"):
            ml.log_params({k: str(v)[:250] for k, v in flat_params(cfg).items()})
            if note:
                ml.set_tag("note", note)
            info = ml.pyfunc.log_model(
                name="detector", python_model=_pyfunc_class()(), artifacts={"config": str(snap)},
                code_paths=[str(ROOT / "gabarit")],
                pip_requirements=["numpy>=1.25,<3", "PyYAML>=6,<7", "threadpoolctl>=3,<4", "pandas"],
                registered_model_name=MODEL_NAME,
            )
            entry["version"] = getattr(info, "registered_model_version", None)
            entry["model_uri"] = info.model_uri
    except Exception as err:
        entry["mlflow_error"] = str(err)[:300]
    return entry


def serve_ui(host: str = "127.0.0.1", port: int = 5000) -> int:
    """Start the MLflow tracking server with the project store."""
    import subprocess
    import sys
    STORE.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlflow", "server", "--backend-store-uri", tracking_uri(),
           "--default-artifact-root", (STORE / "artifacts").resolve().as_uri(),
           "--host", host, "--port", str(port)]
    print(f"MLflow: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    return subprocess.call(cmd)
