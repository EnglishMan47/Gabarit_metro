"""ROS 2 node: PointCloud2 in, obstacle decision out.

Subscribed:
  <input_topic>   sensor_msgs/PointCloud2   ("auto" = every PointCloud2 topic)

Published:
  /gabarit/obstacle            std_msgs/Bool      path blocked (confirmed obstacle)
  /gabarit/distance            std_msgs/Float32   distance to the nearest obstacle, m (NaN when clear)
  /gabarit/free_distance       std_msgs/Float32   verified free distance ahead, m
  /gabarit/status              std_msgs/String    full JSON result of the frame
  /gabarit/markers             visualization_msgs/MarkerArray  envelope, path, obstacle boxes (RViz2)
  /gabarit/obstacle_points     sensor_msgs/PointCloud2  points of confirmed obstacles
All outputs are in the lidar frame of the input message.
"""

from __future__ import annotations

import gabarit  # noqa: F401  (single-threaded BLAS before NumPy is loaded)

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float32, String, Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from gabarit.config import Config
from gabarit.pipeline import Pipeline
from gabarit.rosbag import RawCloud

_ROS_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def msg_to_raw(msg: PointCloud2) -> RawCloud:
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [order + _ROS_TYPES[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    n = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return RawCloud(stamp, msg.header.frame_id, msg.height, msg.width, arr)


class GabaritNode(Node):
    def __init__(self) -> None:
        super().__init__("gabarit")
        self.declare_parameter("input_topic", "auto")
        self.declare_parameter("config", "")
        self.declare_parameter("web", True)
        self.declare_parameter("web_port", 8080)
        self.declare_parameter("publish_markers", True)
        cfg_path = self.get_parameter("config").value
        self.cfg = Config.load(cfg_path) if cfg_path else Config()
        self.pipe = Pipeline(self.cfg)
        self.lock = threading.Lock()
        self.busy = False
        self.dropped = 0
        self.processed = 0
        self.lat_hist: list[float] = []

        out_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.VOLATILE)
        self.pub_obst = self.create_publisher(Bool, "/gabarit/obstacle", out_qos)
        self.pub_dist = self.create_publisher(Float32, "/gabarit/distance", out_qos)
        self.pub_free = self.create_publisher(Float32, "/gabarit/free_distance", out_qos)
        self.pub_status = self.create_publisher(String, "/gabarit/status", out_qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/gabarit/markers", out_qos)
        self.pub_points = self.create_publisher(PointCloud2, "/gabarit/obstacle_points", out_qos)

        # reliable + keep-last-1: huge clouds are not fragmented away, and a slow
        # frame is replaced by the newest one instead of building a queue
        self.declare_parameter("reliable", True)
        rel = ReliabilityPolicy.RELIABLE if self.get_parameter("reliable").value else ReliabilityPolicy.BEST_EFFORT
        self.in_qos = QoSProfile(depth=1, reliability=rel, history=HistoryPolicy.KEEP_LAST)
        self.subs = {}
        topic = self.get_parameter("input_topic").value
        if topic and topic != "auto":
            self._subscribe(topic)
        else:
            self.create_timer(1.0, self._discover)
        self.hub = None
        if self.get_parameter("web").value:
            self._start_web(int(self.get_parameter("web_port").value))
        self.create_timer(5.0, self._report)
        self.get_logger().info("gabarit node ready (input: %s)" % topic)

    # ---------------------------------------------------------------- inputs
    def _discover(self) -> None:
        for name, types in self.get_topic_names_and_types():
            if "sensor_msgs/msg/PointCloud2" in types and name not in self.subs and not name.startswith("/gabarit"):
                self._subscribe(name)

    def _subscribe(self, name: str) -> None:
        self.subs[name] = self.create_subscription(PointCloud2, name, self._on_cloud, self.in_qos)
        self.get_logger().info(f"subscribed to {name}")

    def _on_cloud(self, msg: PointCloud2) -> None:
        # the subscription keeps only the newest frames (depth 2): when the
        # detector is slower than the sensor, stale clouds are skipped
        if self.busy:
            self.dropped += 1
            return
        self.busy = True
        try:
            t0 = time.perf_counter()
            raw = msg_to_raw(msg)
            with self.lock:
                res = self.pipe.process_raw(raw, raw.stamp)
            self.processed += 1
            d = res.data
            d["input_topic"] = msg.header.frame_id
            d["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self.lat_hist.append(d["latency_ms"])
            del self.lat_hist[:-200]
            self._publish(msg.header, res)
            if self.hub is not None:
                from web.session import pack_frame
                meta = dict(d)
                meta.update({"type": "frame", "bag": "ROS 2", "synthetic": []})
                self.hub.publish_threadsafe(pack_frame(res, meta))
        except Exception as err:  # pragma: no cover - keep the node alive
            self.get_logger().error(f"frame failed: {err}")
        finally:
            self.busy = False

    # ---------------------------------------------------------------- outputs
    def _publish(self, header: Header, res) -> None:
        d = res.data
        self.pub_obst.publish(Bool(data=bool(d["obstacle"])))
        self.pub_dist.publish(Float32(data=float(d["nearest_distance_m"]) if d["nearest_distance_m"] is not None else math.nan))
        self.pub_free.publish(Float32(data=float(d["free_distance_m"])))
        self.pub_status.publish(String(data=json.dumps(d, ensure_ascii=False, default=float)))
        if self.get_parameter("publish_markers").value:
            self.pub_markers.publish(self._markers(header, res))
        idx = np.flatnonzero(res.labels == 4)
        if len(idx):
            self.pub_points.publish(self._cloud(header, res.cloud.xyz[idx]))
        if d["obstacle"]:
            self.get_logger().warn(f"OBSTACLE {d['nearest_distance_m']:.1f} m ahead", throttle_duration_sec=1.0)

    def _to_sensor(self, lat: float, fwd: float, up: float, res) -> Point:
        """Levelled track coordinates -> original sensor frame (inverse levelling)."""
        cal = self.pipe.track.calib
        a, b = math.atan(cal.pitch), math.atan(cal.roll)
        z2 = up - cal.height
        x = lat * math.cos(b) - z2 * math.sin(b)
        z1 = lat * math.sin(b) + z2 * math.cos(b)
        f = fwd * math.cos(a) - z1 * math.sin(a)
        z = fwd * math.sin(a) + z1 * math.cos(a)
        axis = self.cfg.sensor.forward_axis
        if axis == "-y":
            return Point(x=float(x), y=float(-f), z=float(z))
        if axis == "+y":
            return Point(x=float(-x), y=float(f), z=float(z))
        if axis == "-x":
            return Point(x=float(-f), y=float(-x), z=float(z))
        return Point(x=float(f), y=float(x), z=float(z))

    def _markers(self, header: Header, res) -> MarkerArray:
        arr = MarkerArray()
        tr = res.track
        d = res.data
        W, H = self.cfg.envelope.width, self.cfg.envelope.height
        color = (0.94, 0.27, 0.27) if d["obstacle"] else (0.13, 0.77, 0.37)
        env = Marker(header=header, ns="envelope", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        env.scale.x = 0.05
        env.color.r, env.color.g, env.color.b, env.color.a = (*color, 0.9)
        s_end = min(tr.observed_until, 200.0)
        ss = np.arange(3.0, max(s_end, 4.0), 2.0)
        cs = np.interp(ss, tr.s, tr.c)
        zs = np.interp(ss, tr.s, tr.z)
        for k in range(len(ss) - 1):
            for lat, up in ((W / 2, 0), (-W / 2, 0), (W / 2, H), (-W / 2, H)):
                env.points.append(self._to_sensor(cs[k] + lat, ss[k], zs[k] + up, res))
                env.points.append(self._to_sensor(cs[k + 1] + lat, ss[k + 1], zs[k + 1] + up, res))
        for k in range(0, len(ss), 12):
            ring = [(W / 2, 0), (W / 2, H), (-W / 2, H), (-W / 2, 0), (W / 2, 0)]
            for (l0, u0), (l1, u1) in zip(ring[:-1], ring[1:]):
                env.points.append(self._to_sensor(cs[k] + l0, ss[k], zs[k] + u0, res))
                env.points.append(self._to_sensor(cs[k] + l1, ss[k], zs[k] + u1, res))
        arr.markers.append(env)
        path = Marker(header=header, ns="path", id=1, type=Marker.LINE_STRIP, action=Marker.ADD)
        path.scale.x = 0.04
        path.color.r, path.color.g, path.color.b, path.color.a = 0.22, 0.74, 0.97, 1.0
        for s, c, z in zip(ss, cs, zs):
            path.points.append(self._to_sensor(c, s, z, res))
        arr.markers.append(path)
        clear = Marker(header=header, ns="obstacles", action=Marker.DELETEALL)
        arr.markers.append(clear)
        for k, ob in enumerate(d["obstacles"] + d["pending"]):
            b = ob["box"]
            s0, s1 = b["s"][0], max(b["s"][1], b["s"][0] + 0.3)
            c = float(np.interp(s0, tr.s, tr.c))
            z = float(np.interp(s0, tr.s, tr.z))
            m = Marker(header=header, ns="obstacles", id=100 + k, type=Marker.CUBE, action=Marker.ADD)
            m.pose.position = self._to_sensor(c + 0.5 * (b["d"][0] + b["d"][1]), 0.5 * (s0 + s1), z + 0.5 * (b["h"][0] + b["h"][1]), res)
            m.pose.orientation.w = 1.0
            m.scale.x = max(b["d"][1] - b["d"][0], 0.2) + 0.2
            m.scale.y = s1 - s0 + 0.2
            m.scale.z = max(b["h"][1] - b["h"][0], 0.2) + 0.2
            conf = ob.get("confirmed", False)
            m.color.r, m.color.g, m.color.b, m.color.a = (0.94, 0.27, 0.27, 0.6) if conf else (0.96, 0.62, 0.04, 0.5)
            arr.markers.append(m)
            t = Marker(header=header, ns="obstacles", id=200 + k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.pose.position = self._to_sensor(c, s0, z + b["h"][1] + 1.0, res)
            t.pose.orientation.w = 1.0
            t.scale.z = 1.2
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{ob['distance_m']:.1f} m"
            arr.markers.append(t)
        return arr

    def _cloud(self, header: Header, xyz: np.ndarray) -> PointCloud2:
        """Sensor working frame (lateral, forward, up) -> axes of the input cloud."""
        lat, fwd, up = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        axis = self.cfg.sensor.forward_axis
        if axis == "-y":
            pts = np.column_stack((lat, -fwd, up))
        elif axis == "+y":
            pts = np.column_stack((-lat, fwd, up))
        elif axis == "-x":
            pts = np.column_stack((-fwd, -lat, up))
        else:
            pts = np.column_stack((fwd, lat, up))
        pts = pts.astype(np.float32)
        msg = PointCloud2(header=header, height=1, width=len(pts), is_bigendian=False, point_step=12,
                          row_step=12 * len(pts), is_dense=True)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        msg.data = pts.tobytes()
        return msg

    def _report(self) -> None:
        if self.lat_hist:
            a = np.array(self.lat_hist)
            self.get_logger().info(f"frames {self.processed}, dropped {self.dropped}, latency p50 {np.percentile(a, 50):.1f} ms, "
                                   f"p95 {np.percentile(a, 95):.1f} ms")

    # ---------------------------------------------------------------- web
    def _start_web(self, port: int) -> None:
        try:
            import asyncio
            from aiohttp import web
            from web.server import App
        except ImportError as err:  # pragma: no cover
            self.get_logger().warn(f"web interface disabled: {err}")
            return
        app = App(None, None, live=True)
        self.hub = app.live

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app.live.loop = loop
            runner = web.AppRunner(app.make())
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "0.0.0.0", port).start())
            loop.run_forever()

        threading.Thread(target=run, daemon=True).start()
        self.get_logger().info(f"web interface: http://localhost:{port}")


def main() -> None:
    rclpy.init()
    node = GabaritNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
