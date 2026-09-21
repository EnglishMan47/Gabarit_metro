"""Run the browser-based lidar obstacle detection application."""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import struct
import threading
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from webapp.bag_store import BagStore
from webapp.bev import build_bev
from webapp.config import PROJECT_ROOT, load_config
from webapp.detector import detect_obstacles, visualization_sample
from webapp.inference import HybridInferenceEngine
from webapp.scene_model import NormalSceneModel
from webapp.tracking import TrackManager


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"


class Application:
    def __init__(self, data_root: Path, feedback_log: Path):
        self.store = BagStore(data_root)
        self.feedback_log = feedback_log.resolve()
        self.default_config = load_config()
        model_path = (PROJECT_ROOT / self.default_config.background_model_path).resolve()
        self.scene_model = NormalSceneModel.load(model_path) if model_path.exists() else None
        self.inference = HybridInferenceEngine(PROJECT_ROOT / "models" / "bev_segmentation.onnx")
        self.tracker = TrackManager(self.default_config)
        self.tracker_context: tuple[str, str] | None = None
        self.tracker_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.sessions: OrderedDict[str, tuple[Any, TrackManager]] = OrderedDict()

    def frame(self, query: dict[str, list[str]]) -> dict[str, Any]:
        # Bound memory use and keep decoding and tracking in request order.
        with self.frame_lock:
            return self._frame(query)

    def _frame(self, query: dict[str, list[str]]) -> dict[str, Any]:
        bag_id = query.get("bag", [""])[0]
        index = int(query.get("index", ["0"])[0])
        config = self.default_config.with_overrides(
            corridor_width=float(
                query.get("corridor_width", [str(self.default_config.corridor_width)])[0]
            ),
            corridor_center_x=float(
                query.get("corridor_center_x", [str(self.default_config.corridor_center_x)])[0]
            ),
            fallback_floor_z=float(
                query.get("floor_z", [str(self.default_config.fallback_floor_z)])[0]
            ),
            max_forward=float(
                query.get("max_forward", [str(self.default_config.max_forward)])[0]
            ),
        )
        frame = self.store.read_frame(bag_id, index)
        detection = detect_obstacles(frame["points"], config, self.scene_model)
        inference_mode = query.get("inference_mode", ["auto"])[0]
        bev = None
        if self.inference.should_build_bev(inference_mode):
            bev = build_bev(
                frame["points"], frame["intensity"], detection, config, self.scene_model
            )
        inference, _ = self.inference.infer(bev, inference_mode)
        detection["inference"] = inference
        context = (bag_id, json.dumps(config.public(), sort_keys=True))
        session_id = query.get("session", ["default"])[0][:100]
        with self.tracker_lock:
            previous = self.sessions.pop(session_id, None)
            tracker = previous[1] if previous and previous[0] == context else TrackManager(config)
            tracked = tracker.update(index, detection["objects"])
            self.sessions[session_id] = (context, tracker)
            while len(self.sessions) > 16:
                self.sessions.popitem(last=False)
        detection["objects"] = tracked["objects"]
        detection["event_detected"] = tracked["event_detected"]
        detection["event_nearest_distance_m"] = tracked["event_nearest_distance_m"]
        detection["obstacle_detected"] = tracked["event_detected"]
        detection["active_tracks"] = tracked["active_tracks"]
        detection["completed_events"] = tracked["completed_events"]
        return {
            "bag": frame["bag"].public(),
            "frame_index": index,
            "bag_timestamp_ns": frame["bag_timestamp_ns"],
            "header_timestamp_ns": frame["header_timestamp_ns"],
            "frame_id": frame["frame_id"],
            "raw_point_count": frame["raw_point_count"],
            "valid_point_count": frame["valid_point_count"],
            "decode_ms": frame["decode_ms"],
            **detection,
            "points": visualization_sample(
                frame["points"],
                frame["intensity"],
                max_points=config.visualization_max_points,
                forward_limit=config.max_forward,
            ),
        }

    def frame_packet(self, query: dict[str, list[str]]) -> bytes:
        """Encode frame metadata plus little-endian float32 XYZI points."""
        payload = self.frame(query)
        points = np.asarray(payload.pop("points"), dtype="<f4")
        if points.ndim != 2 or points.shape[1] != 4:
            raise RuntimeError("visualization points must have shape (N, 4)")
        payload["point_count"] = int(len(points))
        payload["point_stride"] = 4
        payload["point_dtype"] = "float32-le"
        header = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        padding = b"\0" * ((-(4 + len(header))) % 4)
        return struct.pack("<I", len(header)) + header + padding + points.tobytes(order="C")

    def save_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = {"bag", "frame_index", "decision"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"missing feedback fields: {sorted(missing)}")
        if payload["decision"] not in {"confirmed", "dismissed"}:
            raise ValueError("decision must be confirmed or dismissed")
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "bag": str(payload["bag"]),
            "frame_index": int(payload["frame_index"]),
            "decision": payload["decision"],
            "nearest_distance_m": payload.get("nearest_distance_m"),
            "comment": str(payload.get("comment", ""))[:500],
        }
        self.feedback_log.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def feedback_records(self) -> list[dict[str, Any]]:
        if not self.feedback_log.exists():
            return []
        records = []
        with self.feedback_log.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def feedback_export(self, output_format: str) -> tuple[bytes, str, str]:
        records = self.feedback_records()
        if output_format == "json":
            body = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            return body, "application/json; charset=utf-8", "metro-lidar-feedback.json"
        if output_format == "csv":
            columns = ["recorded_at", "bag", "frame_index", "decision", "nearest_distance_m", "comment"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
            return stream.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8", "metro-lidar-feedback.csv"
        raise ValueError("format must be json or csv")


def make_handler(app: Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MetroLidarWeb/0.1"

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {format_string % args}")

        def json_response(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def bytes_response(self, body: bytes, content_type: str, filename: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def packet_response(self, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/vnd.metro-lidar.frame")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def error_response(self, error: Exception, status: HTTPStatus) -> None:
            self.json_response({"error": str(error)}, status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self.json_response({
                        "status": "ok",
                        "bags": len(app.store.list_bags()),
                        "scene_model_loaded": app.scene_model is not None,
                        "detector_version": app.default_config.version,
                        "inference": app.inference.capabilities(),
                    })
                    return
                if parsed.path == "/api/capabilities":
                    self.json_response(app.inference.capabilities())
                    return
                if parsed.path == "/api/bags":
                    self.json_response({"bags": app.store.list_bags()})
                    return
                if parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    if query.get("response", ["json"])[0] == "packet":
                        self.packet_response(app.frame_packet(query))
                    else:
                        self.json_response(app.frame(query))
                    return
                if parsed.path == "/api/feedback":
                    self.json_response({"records": app.feedback_records()})
                    return
                if parsed.path == "/api/export":
                    output_format = parse_qs(parsed.query).get("format", ["json"])[0]
                    self.bytes_response(*app.feedback_export(output_format))
                    return
                self.serve_static(parsed.path)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return  # A cancelled frame request is normal browser navigation.
            except (KeyError, IndexError) as error:
                self.error_response(error, HTTPStatus.NOT_FOUND)
            except (TypeError, ValueError) as error:
                self.error_response(error, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - last-resort server boundary
                traceback.print_exc()
                self.error_response(error, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/feedback":
                self.error_response(ValueError("unknown endpoint"), HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                self.json_response(app.save_feedback(payload), HTTPStatus.CREATED)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self.error_response(error, HTTPStatus.BAD_REQUEST)

        def serve_static(self, url_path: str) -> None:
            relative = "index.html" if url_path in {"", "/"} else unquote(url_path.lstrip("/"))
            path = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT.resolve() not in path.parents and path != STATIC_ROOT.resolve():
                self.error_response(ValueError("invalid static path"), HTTPStatus.NOT_FOUND)
                return
            if not path.is_file():
                self.error_response(ValueError("not found"), HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted/for_hackathon"))
    parser.add_argument("--feedback-log", type=Path, default=Path("artifacts/operator_feedback.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = Application(args.data_root, args.feedback_log)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Metro lidar web app: http://{args.host}:{args.port}")
    print(f"Data root: {app.store.data_root} ({len(app.store.list_bags())} bag files)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
