"""Supervise the detector, RViz and optional portable browser desktop."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bag_control import player_lock, start_player, stop_player, resolve_bag, ros_call


class Workspace:
    def __init__(self):
        self.children: list[tuple[str, subprocess.Popen]] = []
        self.stopping = False

    def spawn(self, name: str, argv: list[str]) -> subprocess.Popen:
        process = subprocess.Popen(argv, start_new_session=True)
        self.children.append((name, process))
        return process

    def stop(self, *_):
        self.stopping = True

    def close(self):
        try:
            with player_lock():
                stop_player()
        except Exception as error:
            # A failed player cleanup must not skip cleanup of ROS and the GUI.
            print(f"player cleanup failed: {error}", file=sys.stderr)
        def group_alive(process):
            # A launch process may already have exited while its node children
            # still live in the same process group. Reap leaders, check groups.
            process.poll()
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False

        for sig, timeout in ((signal.SIGINT, 6), (signal.SIGTERM, 2), (signal.SIGKILL, 1)):
            active = [process for _, process in self.children if group_alive(process)]
            for process in reversed(active):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and any(group_alive(process) for process in active):
                time.sleep(0.05)

    def browser_desktop(self):
        os.environ["DISPLAY"] = ":99"
        self.spawn("Xvfb", ["Xvfb", ":99", "-screen", "0", "1600x900x24", "-nolisten", "tcp"])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.stopping:
                raise RuntimeError("workspace interrupted during display startup")
            probe = subprocess.run(["xdpyinfo", "-display", ":99"], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=2)
            if probe.returncode == 0:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Xvfb failed to start")
        # Avoid the first-run wallpaper helper dialog on a minimal image.
        wm_init = Path("/tmp/gabarit-fluxbox/init")
        wm_init.parent.mkdir(parents=True, exist_ok=True)
        defaults = Path("/etc/X11/fluxbox/init")
        settings = defaults.read_text() if defaults.exists() else ""
        settings = "\n".join(line for line in settings.splitlines()
                             if not line.startswith(("session.screen0.rootCommand:", "session.screen0.toolbar.visible:")))
        wm_init.write_text(settings + "\nsession.screen0.rootCommand: true\nsession.screen0.toolbar.visible: false\n")
        self.spawn("window manager", ["fluxbox", "-rc", str(wm_init)])
        self.spawn("VNC", ["x11vnc", "-display", ":99", "-rfbport", "5900", "-forever",
                           "-shared", "-localhost", "-nopw", "-quiet"])
        self.spawn("browser GUI", ["websockify", "--web=/usr/share/novnc", "0.0.0.0:6080", "localhost:5900"])
        print("RViz desktop: http://localhost:6080/vnc.html?autoconnect=1&resize=scale", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent ROS/RViz workspace; switch recordings with gabarit bag")
    parser.add_argument("--bag")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--config", default=os.environ.get("GABARIT_CONFIG", "/opt/gabarit/config/gabarit.yaml"))
    parser.add_argument("--input-topic", default="auto")
    parser.add_argument("--input-qos", choices=("auto", "reliable", "best_effort"),
                        default=os.environ.get("GABARIT_INPUT_QOS", "auto"))
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--gui", choices=("web", "x11", "off"), default=os.environ.get("GABARIT_GUI", "web"))
    arguments = sys.argv[1:]
    split = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:split])
    extra = arguments[split + 1:] if split < len(arguments) else []
    from bag_control import positive_rate
    positive_rate(str(args.rate))
    if not Path(args.config).is_file():
        parser.error(f"config not found: {args.config}")
    bag = resolve_bag(args.bag, Path(os.environ.get("GABARIT_DATA", "/data"))) if args.bag else None
    workspace = Workspace()
    signal.signal(signal.SIGTERM, workspace.stop)
    signal.signal(signal.SIGINT, workspace.stop)
    try:
        if args.gui == "web" and not args.no_rviz:
            workspace.browser_desktop()
        if args.gui == "x11" and not os.environ.get("DISPLAY"):
            raise RuntimeError("X11 mode requires DISPLAY and an authorized X socket; see docs/DOCKER_GUI.md")
        rviz = not args.no_rviz and args.gui != "off"
        workspace.spawn("ROS launch", ["ros2", "launch", "gabarit_ros", "gabarit.launch.py",
                                        f"config:={args.config}", f"input_topic:={args.input_topic}",
                                        f"input_qos:={args.input_qos}",
                                        f"rviz:={str(rviz).lower()}", "use_sim_time:=true"])
        ros_call("/gabarit/reset", "std_srvs/srv/Trigger", timeout=25.0)
        if workspace.stopping:
            return 0
        if bag is not None:
            with player_lock():
                start_player(bag, args.rate, args.paused, extra)
        print("Workspace ready. Use: docker compose exec gabarit gabarit bag list", flush=True)
        print("Live detector UI: http://localhost:8080  |  Ctrl+C stops the workspace", flush=True)
        while not workspace.stopping:
            for name, process in workspace.children:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited with status {process.returncode}")
            time.sleep(0.2)
        return 0
    finally:
        workspace.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"gabarit workspace: {error}", file=sys.stderr)
        raise SystemExit(1)
