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
  /gabarit/view_cloud          sensor_msgs/PointCloud2  levelled working cloud for RViz
  /gabarit/view_markers        visualization_msgs/MarkerArray  levelled RViz overlays
  /gabarit/reset               std_srvs/Trigger  reset temporal state and reload config parameter
Decision outputs retain the input sensor frame. view_* share gabarit_view,
with REP-103 x forward, y left, z up and the rail head plane at z=0.
"""

from __future__ import annotations

import gabarit  # noqa: F401  (single-threaded BLAS before NumPy is loaded)

import json
import array
import math
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from builtin_interfaces.msg import Time
from std_msgs.msg import Bool, Float32, String, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from gabarit.config import Config
from gabarit.cloud import axis_transform
from gabarit.pipeline import Pipeline
from gabarit.rosbag import RawCloud, pointcloud2_array, decode_pointcloud2
from gabarit_ros.render_worker import LatestRenderer


def msg_to_raw(msg: PointCloud2) -> RawCloud:
    fields = [(field.name, field.offset, field.datatype, field.count) for field in msg.fields]
    arr = pointcloud2_array(bytes(msg.data), fields, msg.height, msg.width,
                            msg.point_step, msg.row_step, msg.is_bigendian)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return RawCloud(stamp, msg.header.frame_id, msg.height, msg.width, arr)


def input_to_raw(msg):
    if isinstance(msg, (bytes, bytearray, memoryview)):
        raw = decode_pointcloud2(msg)
        endian = "<" if msg[:2] == b"\x00\x01" else ">"
        sec, nsec = struct.unpack_from(endian + "iI", msg, 4)
        header = Header(stamp=Time(sec=sec, nanosec=nsec), frame_id=raw.frame_id)
        return raw, header
    return msg_to_raw(msg), msg.header


def xyz_cloud_cdr(header, pts, intensity=None) -> bytes:
    """Serialize the viewer cloud directly, avoiding per-byte Python/C conversion."""
    if intensity is not None:
        pts = np.column_stack((pts, intensity))
    pts = np.ascontiguousarray(pts, dtype="<f4")
    names = ("x", "y", "z", "intensity") if intensity is not None else ("x", "y", "z")
    out = bytearray(b"\x00\x01\x00\x00")
    def u32(value):
        out.extend(b"\x00" * ((-(len(out) - 4)) % 4))
        out.extend(struct.pack("<I", value))
    def string(value):
        data = value.encode("utf-8") + b"\x00"
        u32(len(data)); out.extend(data)
    out.extend(struct.pack("<iI", header.stamp.sec, header.stamp.nanosec))
    string(header.frame_id)
    u32(1); u32(len(pts)); u32(len(names))
    for index, name in enumerate(names):
        string(name); u32(index * 4); out.append(7); u32(1)
    out.append(0)
    step = len(names) * 4
    u32(step); u32(len(pts) * step)
    data = pts.tobytes()
    u32(len(data)); out.extend(data); out.append(1)
    return bytes(out)


def view_indices(labels, limit=30000):
    """Bound background traffic to RViz while retaining all obstacle evidence."""
    if len(labels) <= limit:
        return np.arange(len(labels))
    important = np.flatnonzero(labels >= 3)
    background = np.flatnonzero(labels < 3)
    take = max(limit - len(important), 0)
    selected = (background[np.linspace(0, len(background) - 1, take).astype(np.int64)]
                if take and len(background) else background[:0])
    return np.concatenate((selected, important))


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
        self.sensor_axes = axis_transform(self.cfg.sensor.forward_axis, self.cfg.sensor.up_axis)
        self.pipe = Pipeline(self.cfg)
        self.lock = threading.Lock()
        self.busy = False
        self.dropped = 0
        self.processed = 0
        self.lat_hist: list[float] = []
        self._last_marker_counts: dict[str, int] = {}

        out_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.VOLATILE)
        self.pub_obst = self.create_publisher(Bool, "/gabarit/obstacle", out_qos)
        self.pub_dist = self.create_publisher(Float32, "/gabarit/distance", out_qos)
        self.pub_free = self.create_publisher(Float32, "/gabarit/free_distance", out_qos)
        self.pub_status = self.create_publisher(String, "/gabarit/status", out_qos)
        self.pub_markers = self.create_publisher(MarkerArray, "/gabarit/markers", out_qos)
        self.pub_points = self.create_publisher(PointCloud2, "/gabarit/obstacle_points", out_qos)
        self.pub_view_cloud = self.create_publisher(PointCloud2, "/gabarit/view_cloud", self._view_qos())
        self.pub_view_markers = self.create_publisher(MarkerArray, "/gabarit/view_markers", out_qos)
        self._view_marker_counts = {}
        self._view_worker = LatestRenderer(self._build_view, self._send_view,
            lambda error:self.get_logger().error(f"visualization failed: {error}"))
        self.create_service(Trigger, "~/reset", self._reset)

        # Prefer reliable for large clouds when offered; a best-effort bag must
        # not be silently rejected by a reliable-only subscription.
        self.declare_parameter("reliable", True)
        self.declare_parameter("input_qos", "auto")
        if self.get_parameter("input_qos").value not in ("auto", "reliable", "best_effort"):
            raise ValueError("input_qos must be auto, reliable, or best_effort")
        self.subs = {}
        self._subscription_reliability = {}
        topic = self.get_parameter("input_topic").value
        if topic and topic != "auto":
            self._subscribe(topic)
        # Discovery and QoS changes must work while sim time is paused.
        self.create_timer(0.25, self._discover, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.hub = None
        self.web_app = None
        if self.get_parameter("web").value:
            self._start_web(int(self.get_parameter("web_port").value))
        self.create_timer(5.0, self._report, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info("gabarit node ready (input: %s)" % topic)

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _view_qos():
        return QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)

    def _reset(self, request, response):
        try:
            path = self.get_parameter("config").value
            cfg = Config.load(path) if path else Config()
            pipeline = Pipeline(cfg)
            sensor_axes = axis_transform(cfg.sensor.forward_axis, cfg.sensor.up_axis)
            with self.lock:
                self.cfg, self.pipe = cfg, pipeline
                self.sensor_axes = sensor_axes
                self.processed = self.dropped = 0
                self.lat_hist.clear()
                self._last_marker_counts.clear()
            def clear_markers():
                self._view_marker_counts.clear()
                for publisher in (self.pub_markers, self.pub_view_markers):
                    publisher.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
            self._view_worker.reset(clear_markers)
            self.pub_obst.publish(Bool(data=False))
            self.pub_dist.publish(Float32(data=math.nan))
            self.pub_free.publish(Float32(data=0.0))
            if self.web_app is not None:
                self.web_app.base_cfg = cfg
            if self.hub is not None:
                self.hub.clear()
            self._discover()
            response.success = True
            response.message = f"pipeline reset; config: {path or 'defaults'}"
            self.get_logger().info(response.message)
        except Exception as error:
            response.success = False
            response.message = f"reset failed; previous pipeline retained: {error}"
        return response

    def _discover(self) -> None:
        topic = self.get_parameter("input_topic").value
        if topic and topic != "auto":
            self._subscribe(topic)
            return
        for name, types in self.get_topic_names_and_types():
            if "sensor_msgs/msg/PointCloud2" in types and not name.startswith("/gabarit"):
                self._subscribe(name)

    def _input_reliability(self, name: str):
        mode = self.get_parameter("input_qos").value
        if mode == "reliable":
            return ReliabilityPolicy.RELIABLE
        if mode == "best_effort" or not self.get_parameter("reliable").value:
            return ReliabilityPolicy.BEST_EFFORT
        endpoints = self.get_publishers_info_by_topic(name)
        if not endpoints:
            return self._subscription_reliability.get(name, ReliabilityPolicy.BEST_EFFORT)
        if all(info.qos_profile.reliability == ReliabilityPolicy.RELIABLE for info in endpoints):
            return ReliabilityPolicy.RELIABLE
        return ReliabilityPolicy.BEST_EFFORT

    def _subscribe(self, name: str) -> None:
        reliability = self._input_reliability(name)
        if name in self.subs:
            if self._subscription_reliability[name] == reliability:
                return
            self.destroy_subscription(self.subs.pop(name))
        qos = QoSProfile(depth=4, reliability=reliability, history=HistoryPolicy.KEEP_LAST)
        # A generated Python PointCloud2 conversion walks the large byte
        # sequence before entering the callback. Decode CDR directly instead.
        self.subs[name] = self.create_subscription(PointCloud2, name, lambda msg: self._on_cloud(msg, name),
                                                    qos, raw=True)
        self._subscription_reliability[name] = reliability
        self.get_logger().info(f"subscribed to {name} ({reliability.name}, depth=4)")

    def _on_cloud(self, msg: PointCloud2, topic: str = "") -> None:
        # A small bounded history absorbs transport bursts. Sustained overload
        # is still counted by the external source-stamp benchmark.
        if self.busy:
            self.dropped += 1
            return
        self.busy = True
        try:
            t0 = time.perf_counter()
            raw, header = input_to_raw(msg)
            t_input = time.perf_counter()
            with self.lock:
                res = self.pipe.process_raw(raw, raw.stamp)
            self.processed += 1
            t_core = time.perf_counter()
            d = res.data
            d["input_topic"] = topic
            d["input_frame"] = header.frame_id
            d["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            self._publish(header, res)
            t_outputs = time.perf_counter()
            if self.hub is not None:
                meta = dict(d)
                meta.update({"type": "frame", "bag": "ROS 2", "synthetic": []})
                self.hub.publish_frame_threadsafe(res, meta)
            end = time.perf_counter()
            d["latency_ms"] = round((end - t0) * 1000, 2)
            d["ros_timing_ms"] = {"input_decode": round((t_input - t0) * 1000, 2),
                                  "pipeline": round((t_core - t_input) * 1000, 2),
                                  "outputs": round((t_outputs - t_core) * 1000, 2),
                                  "web_pack": round((end - t_outputs) * 1000, 2),
                                  "total": d["latency_ms"]}
            self.lat_hist.append(d["latency_ms"])
            del self.lat_hist[:-200]
            self.pub_status.publish(String(data=json.dumps(d, ensure_ascii=False, default=float)))
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
        if self.get_parameter("publish_markers").value and self.pub_markers.get_subscription_count():
            self.pub_markers.publish(self._markers(header, res))
        if self.pub_points.get_subscription_count():
            idx = np.flatnonzero(res.labels == 4)
            self.pub_points.publish(self._cloud(header, res.cloud.xyz[idx]))
        # The viewer uses one frame for every lidar: REP-103 x=forward,
        # y=left, z=up, with the rail plane at z=0. Raw-frame outputs above
        # retain their original frame_id for ROS consumers.
        view_header = Header(stamp=header.stamp, frame_id="gabarit_view")
        markers = self.get_parameter("publish_markers").value and self.pub_view_markers.get_subscription_count()
        cloud = bool(self.pub_view_cloud.get_subscription_count())
        if markers or cloud:
            self._view_worker.submit((view_header,res,self.cfg,bool(markers),cloud))
        if d["obstacle"]:
            self.get_logger().warn(f"OBSTACLE {d['nearest_distance_m']:.1f} m ahead", throttle_duration_sec=1.0)

    def _build_view(self, job):
        header,res,cfg,draw_markers,draw_cloud=job
        context=SimpleNamespace(cfg=cfg,_last_marker_counts=self._view_marker_counts)
        markers=GabaritNode._markers(context,header,res,view=True) if draw_markers else None
        cloud=None
        if draw_cloud:
            view_idx = view_indices(res.labels)
            points = res.lev[view_idx][:, [1, 0, 2]]
            cloud=xyz_cloud_cdr(header,points,res.cloud.intensity[view_idx])
        return markers,cloud

    def _send_view(self,payload):
        markers,cloud=payload
        if markers is not None:self.pub_view_markers.publish(markers)
        if cloud is not None:self.pub_view_cloud.publish(cloud)

    def destroy_node(self):
        self._view_worker.close()
        return super().destroy_node()

    def _to_sensor(self, lat: float, fwd: float, up: float, res) -> Point:
        """Levelled track coordinates -> original sensor frame (inverse levelling)."""
        cal = self.pipe.track.calib
        a, b = math.atan(cal.pitch), math.atan(cal.roll)
        z2 = up - cal.height
        x = lat * math.cos(b) - z2 * math.sin(b)
        z1 = lat * math.sin(b) + z2 * math.cos(b)
        f = fwd * math.cos(a) - z1 * math.sin(a)
        z = fwd * math.sin(a) + z1 * math.cos(a)
        point = np.array([x, f, z]) @ self.sensor_axes
        return Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))

    def _markers(self, header: Header, res, view: bool = False) -> MarkerArray:
        arr = MarkerArray()
        point = (lambda lat, fwd, up: Point(x=float(fwd), y=float(lat), z=float(up))) if view else (
            lambda lat, fwd, up: self._to_sensor(lat, fwd, up, res))
        tr = res.track
        d = res.data
        W, H = self.cfg.envelope.width, self.cfg.envelope.height
        color = (0.94, 0.27, 0.27) if d["obstacle"] else (0.13, 0.77, 0.37)
        env = Marker(header=header, ns="envelope", id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        env.pose.orientation.w = 1.0
        env.scale.x = 0.05
        env.color.r, env.color.g, env.color.b, env.color.a = (*color, 0.9)
        s_end = min(tr.observed_until, self.cfg.detection.max_range, self.cfg.track.horizon)
        ss = np.arange(3.0, max(s_end, 4.0), 2.0)
        cs = np.interp(ss, tr.s, tr.c)
        zs = np.interp(ss, tr.s, tr.z)
        for k in range(len(ss) - 1):
            for lat, up in ((W / 2, 0), (-W / 2, 0), (W / 2, H), (-W / 2, H)):
                env.points.append(point(cs[k] + lat, ss[k], zs[k] + up))
                env.points.append(point(cs[k + 1] + lat, ss[k + 1], zs[k + 1] + up))
        for k in range(0, len(ss), 12):
            ring = [(W / 2, 0), (W / 2, H), (-W / 2, H), (-W / 2, 0), (W / 2, 0)]
            for (l0, u0), (l1, u1) in zip(ring[:-1], ring[1:]):
                env.points.append(point(cs[k] + l0, ss[k], zs[k] + u0))
                env.points.append(point(cs[k] + l1, ss[k], zs[k] + u1))
        arr.markers.append(env)
        path = Marker(header=header, ns="path", id=1, type=Marker.LINE_STRIP, action=Marker.ADD)
        path.pose.orientation.w = 1.0
        path.scale.x = 0.04
        path.color.r, path.color.g, path.color.b, path.color.a = 0.22, 0.74, 0.97, 1.0
        for s, c, z in zip(ss, cs, zs):
            path.points.append(point(c, s, z))
        arr.markers.append(path)
        obstacles = d["obstacles"] + d["pending"]
        # Delete only obsolete IDs; DELETEALL here also deleted the envelope
        # just added earlier in the array and made it flicker/disappear in RViz.
        key = "view" if view else "raw"
        for k in range(len(obstacles), self._last_marker_counts.get(key, 0)):
            for marker_id in (100 + k, 200 + k):
                arr.markers.append(Marker(header=header, ns="obstacles", id=marker_id, action=Marker.DELETE))
        self._last_marker_counts[key] = len(obstacles)
        for k, ob in enumerate(obstacles):
            b = ob["box"]
            s0, s1 = b["s"][0], max(b["s"][1], b["s"][0] + 0.3)
            c = float(np.interp(s0, tr.s, tr.c))
            z = float(np.interp(s0, tr.s, tr.z))
            m = Marker(header=header, ns="obstacles", id=100 + k, type=Marker.LINE_LIST, action=Marker.ADD)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08
            # Explicit corner transforms keep the box aligned for all sensor
            # axis conventions and calibrated pitch/roll, including Falcon.
            corners = [(c + lat, forward, z + height)
                       for forward in (s0, s1) for lat in b["d"] for height in b["h"]]
            for i, j in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
                m.points.extend((point(*corners[i]), point(*corners[j])))
            conf = ob.get("confirmed", False)
            m.color.r, m.color.g, m.color.b, m.color.a = (0.94, 0.27, 0.27, 0.6) if conf else (0.96, 0.62, 0.04, 0.5)
            arr.markers.append(m)
            t = Marker(header=header, ns="obstacles", id=200 + k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.pose.position = point(c, s0, z + b["h"][1] + 1.0)
            t.pose.orientation.w = 1.0
            t.scale.z = 1.2
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{ob['distance_m']:.1f} m"
            arr.markers.append(t)
        return arr

    def _cloud(self, header: Header, xyz: np.ndarray) -> PointCloud2:
        """Sensor working frame (lateral, forward, up) -> axes of the input cloud."""
        pts = xyz @ self.sensor_axes
        return self._xyz_cloud(header, pts)

    @staticmethod
    def _xyz_cloud(header: Header, pts: np.ndarray, intensity=None) -> PointCloud2:
        if intensity is not None:
            pts = np.column_stack((pts, intensity))
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        step = 4 * pts.shape[1]
        msg = PointCloud2(header=header, height=1, width=len(pts), is_bigendian=False, point_step=12,
                          row_step=step * len(pts), is_dense=True)
        msg.point_step = step
        names = ["x", "y", "z", "intensity"] if intensity is not None else ["x", "y", "z"]
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)]
        # The generated Humble setter validates every element of bytes twice
        # in Python. A typed uint8 buffer takes its constant-time fast path.
        msg.data = array.array("B", pts.tobytes())
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
        app.base_cfg = self.cfg
        self.web_app = app
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
