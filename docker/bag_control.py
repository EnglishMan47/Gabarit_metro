"""Manage one ROS bag player inside a long-lived container.

No shell interpolation, daemon, network listener, or third-party dependency is
needed. A process-start token prevents a stale PID file from killing a reused
PID. ROS imports are lazy so process/path logic can be tested without ROS.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import tempfile


def state_dir() -> Path:
    path = Path(os.environ.get("GABARIT_PLAYER_STATE", "/tmp/gabarit-player"))
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def player_lock():
    import fcntl  # Linux container; imported lazily for host-side unit tests.
    with (state_dir() / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def process_token(pid: int) -> str | None:
    """Return Linux process start ticks, excluding zombies and other groups."""
    if pid <= 1:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z" or os.getpgid(pid) != pid:
            return None
        return fields[19]
    except (OSError, IndexError, ValueError):
        return None


def read_state() -> dict:
    try:
        data = json.loads((state_dir() / "state.json").read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(data: dict) -> None:
    target = state_dir() / "state.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(target)


def is_running(data: dict) -> bool:
    pid, token = data.get("pid"), data.get("start_token")
    return isinstance(pid, int) and token is not None and process_token(pid) == token


def stop_player() -> dict:
    """Caller holds player_lock; stop and bound cleanup of the entire group."""
    data = read_state()
    if is_running(data):
        pid = data["pid"]
        for sig, seconds in ((signal.SIGINT, 5.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 1.0)):
            if not is_running(data):
                break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            until = time.monotonic() + seconds
            while is_running(data) and time.monotonic() < until:
                time.sleep(0.05)
        if is_running(data):
            raise RuntimeError(f"could not stop player process group {pid}")
    data.update(running=False, paused=False)
    write_state(data)
    return data


def ros_call(name: str, service_type: str, values: dict | None = None, timeout: float = 8.0):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rosidl_runtime_py.utilities import get_service
    context = rclpy.context.Context()
    rclpy.init(context=context)
    node = rclpy.create_node(f"gabarit_bag_control_{os.getpid()}", context=context)
    # Humble's implicit executor belongs to the default context. This helper
    # owns an independent context, including when called by the supervisor.
    executor = SingleThreadedExecutor(context=context)
    try:
        srv = get_service(service_type)
        client = node.create_client(srv, name)
        deadline = time.monotonic() + timeout
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"ROS service unavailable: {name}")
        request = srv.Request()
        for key, value in (values or {}).items():
            setattr(request, key, value)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, executor=executor,
                                         timeout_sec=max(0.0, deadline - time.monotonic()))
        if not future.done():
            raise RuntimeError(f"ROS service timed out: {name}")
        response = future.result()
        if response is None:
            raise RuntimeError(f"ROS service returned no result: {name}")
        if hasattr(response, "success") and not response.success:
            raise RuntimeError(f"ROS service rejected request: {name}: {getattr(response, 'message', '')}")
        return response
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def set_config(config: str) -> None:
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    param = Parameter(name="config", value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=config))
    response = ros_call("/gabarit/set_parameters", "rcl_interfaces/srv/SetParameters", {"parameters": [param]})
    if not response.results or not all(result.successful for result in response.results):
        raise RuntimeError("detector rejected the config parameter")


def _bag_directory(path: Path) -> bool:
    return (path.is_dir() and (path / "metadata.yaml").is_file()
            and any(file.is_file() for pattern in ("*.db3", "*.mcap", "*.db3.zstd", "*.mcap.zstd")
                    for file in path.glob(pattern)))


def list_bags(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    bags = {path.parent for path in root.rglob("metadata.yaml") if _bag_directory(path.parent)}
    bags.update(path for extension in ("*.db3", "*.mcap") for path in root.rglob(extension)
                if not (path.parent / "metadata.yaml").is_file())
    return sorted(bags, key=str)


def resolve_bag(value: str, root: Path) -> Path:
    candidates = [Path(value), root / value]
    for path in candidates:
        if _bag_directory(path):
            return path.resolve()
        if path.is_file() and path.suffix in (".db3", ".mcap"):
            return path.resolve()
    matches = [path for path in list_bags(root) if path.name == value]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"bag name is ambiguous; use its /data/path: {value}")
    raise ValueError(f"bag not found: {value}; run gabarit bag list")


def positive_rate(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("rate must be a finite number greater than zero")
    return number


def cache_bag(bag: Path) -> Path:
    """Copy only the selected bag to Linux storage, then reuse by source version."""
    import yaml
    bag = bag.resolve()
    if bag.is_dir():
        metadata = bag / "metadata.yaml"
        info = (yaml.safe_load(metadata.read_text(encoding="utf-8")) or {}).get("rosbag2_bagfile_information", {})
        names = info.get("relative_file_paths") or [p.name for p in bag.iterdir()
                                                    if p.suffix in (".db3", ".mcap")]
        files = [metadata]
        for name in names:
            path = (bag / name).resolve()
            if not path.is_relative_to(bag):
                raise ValueError("bag metadata references storage outside its directory")
            if not path.is_file():
                raise ValueError(f"missing bag storage: {path}")
            files.append(path)
        base = bag
    else:
        files, base = [bag], bag.parent
    version = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    key = hashlib.sha256(json.dumps(version).encode()).hexdigest()[:24]
    root = Path(os.environ.get("GABARIT_CACHE_DIR", "/var/cache/gabarit")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    if (target / "complete.json").is_file():
        return target if bag.is_dir() else target / bag.name
    required = sum(row[1] for row in version)
    max_gb = float(os.environ.get("GABARIT_CACHE_MAX_GB", "16"))
    if not math.isfinite(max_gb) or max_gb <= 0:
        raise ValueError("GABARIT_CACHE_MAX_GB must be finite and positive")
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if used + required > max_gb * 1024**3:
        raise RuntimeError(f"Linux cache budget {max_gb:g} GiB exceeded; omit --cache, use Linux storage, "
                           "or raise GABARIT_CACHE_MAX_GB after checking host disk space")
    if shutil.disk_usage(root).free < required + 512 * 1024**2:
        raise RuntimeError(f"insufficient Linux cache space: need {required / 1024**3:.1f} GiB plus 0.5 GiB reserve; "
                           "omit --cache or mount the recording from a Linux filesystem")
    print(f"Caching {bag.name}: {required / 1024**3:.1f} GiB to Linux storage (once per source version)",
          file=sys.stderr, flush=True)
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=root))
    try:
        for source, (_, size, modified) in zip(files, version):
            dest = staging / source.relative_to(base)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            if dest.stat().st_size != size or (source.stat().st_size, source.stat().st_mtime_ns) != (size, modified):
                raise RuntimeError(f"incomplete cache copy: {source}")
        (staging / "complete.json").write_text(json.dumps(version))
        staging.rename(target)
    except Exception:
        # Only this newly allocated staging directory is disposable; sources
        # and previous completed cache entries are never removed.
        shutil.rmtree(staging)
        raise
    return target if bag.is_dir() else target / bag.name


def start_player(bag: Path, rate: float, paused: bool, extra: list[str], config: str | None = None,
                 cache: bool = False) -> dict:
    """Replace the old bag, reset detector, then release the new paused player."""
    if config and not Path(config).is_file():
        raise ValueError(f"config does not exist inside the container: {config}")
    stop_player()
    playback_bag = cache_bag(bag) if cache else bag
    if config:
        set_config(config)
    storage_config = Path(os.environ.get("GABARIT_STORAGE_CONFIG", "/opt/gabarit/config/rosbag_read.yaml"))
    sqlite = bag.suffix == ".db3" or (bag.is_dir() and any(bag.glob("*.db3")))
    storage_args = (["--storage-config-file", str(storage_config)]
                    if sqlite and storage_config.is_file()
                    and not any(arg.split("=", 1)[0] == "--storage-config-file" for arg in extra) else [])
    argv = ["ros2", "bag", "play", str(playback_bag), "--rate", str(rate),
            "--start-paused", "--disable-keyboard-controls", "--clock", "100",
            "--read-ahead-queue-size", "5", *storage_args, *extra]
    with (state_dir() / "player.log").open("w") as log:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    data = {"pid": process.pid, "start_token": process_token(process.pid),
            "bag": str(bag), "rate": rate, "paused": True, "running": True,
            "playback_bag": str(playback_bag), "cached": cache,
            "argv": argv, "log": str(state_dir() / "player.log")}
    write_state(data)
    try:
        ros_call("/rosbag2_player/is_paused", "rosbag2_interfaces/srv/IsPaused", timeout=25.0)
        if not is_running(data):
            raise RuntimeError("player exited; inspect gabarit bag log")
        # A paused player advertises its input topics before we discover and reset.
        # No messages from the new recording may reach an old detector state.
        ros_call("/gabarit/reset", "std_srvs/srv/Trigger", timeout=25.0)
        if not paused:
            ros_call("/rosbag2_player/resume", "rosbag2_interfaces/srv/Resume")
            data["paused"] = False
        write_state(data)
        return data
    except Exception:
        stop_player()
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage the ROS bag player without restarting the container")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list bags under /data")
    play = commands.add_parser("play", aliases=["switch"], help="stop previous bag, reset detector, play new bag")
    play.add_argument("bag")
    play.add_argument("--rate", type=positive_rate, default=1.0)
    play.add_argument("--paused", action="store_true")
    play.add_argument("--cache", action="store_true", help="copy selected bag once to fast Linux cache (Docker Desktop)")
    play.add_argument("--config", help="detector YAML path inside the container; applied at reset")
    for command in ("stop", "pause", "resume", "next", "status", "log"):
        commands.add_parser(command)
    rate = commands.add_parser("rate", help="set rate during playback, e.g. 0.25")
    rate.add_argument("value", type=positive_rate)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    split = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser().parse_args(arguments[:split])
    extra = arguments[split + 1:] if split < len(arguments) else []
    if extra and args.command not in ("play", "switch"):
        raise ValueError("extra ros2 bag arguments are only valid for play/switch")
    root = Path(os.environ.get("GABARIT_DATA", "/data"))
    if args.command == "list":
        for bag in list_bags(root):
            print(bag)
        return 0
    if args.command == "log":
        log = state_dir() / "player.log"
        print(log.read_text(errors="replace")[-16000:] if log.exists() else "No bag has been played.")
        return 0
    # Resolve an invalid name before stopping a currently playing valid bag.
    bag = resolve_bag(args.bag, root) if args.command in ("play", "switch") else None
    with player_lock():
        if bag is not None:
            data = start_player(bag, args.rate, args.paused, extra, args.config, args.cache)
        elif args.command == "stop":
            data = stop_player()
        else:
            data = read_state()
            data["running"] = is_running(data)
            if not data["running"]:
                if args.command != "status":
                    raise RuntimeError("no managed bag is playing; use gabarit bag play BAG")
                data["paused"] = False
            else:
                if args.command in ("pause", "resume", "next"):
                    service = {"next": "play_next"}.get(args.command, args.command)
                    typ = {"pause": "Pause", "resume": "Resume", "next": "PlayNext"}[args.command]
                    ros_call(f"/rosbag2_player/{service}", f"rosbag2_interfaces/srv/{typ}")
                elif args.command == "rate":
                    ros_call("/rosbag2_player/set_rate", "rosbag2_interfaces/srv/SetRate", {"rate": args.value})
                data["paused"] = bool(ros_call("/rosbag2_player/is_paused", "rosbag2_interfaces/srv/IsPaused").paused)
                data["rate"] = float(ros_call("/rosbag2_player/get_rate", "rosbag2_interfaces/srv/GetRate").rate)
            write_state(data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"gabarit bag: {error}", file=sys.stderr)
        raise SystemExit(1)
