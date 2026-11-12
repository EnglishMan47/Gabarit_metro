"""Generate a test oracle using ROS 2's serializer, not our CDR decoder."""
import json
from pathlib import Path
from geometry_msgs.msg import Point
from rclpy.serialization import serialize_message
from visualization_msgs.msg import Marker, MarkerArray

root = Path(__file__).resolve().parent
a = Marker()
a.header.frame_id = 'test_sensor'
a.header.stamp.sec, a.header.stamp.nanosec = 123, 456
a.ns, a.id, a.type, a.action = 'test_objects', 7, Marker.LINE_LIST, Marker.ADD
a.pose.position.x, a.pose.position.y, a.pose.position.z = 1., 2., 3.
a.pose.orientation.w = 1.
a.scale.x, a.scale.y, a.scale.z = 1., 2., 3.
a.color.r, a.color.g, a.color.b, a.color.a = .25, .5, .75, 1.
a.lifetime.sec, a.lifetime.nanosec = 2, 500
a.points = [Point(x=-.5,y=-1.,z=-1.5), Point(x=.5,y=1.,z=1.5)]
a.text = 'independent fixture'
b = Marker()
b.header.frame_id, b.ns, b.id, b.action = 'test_sensor', 'test_objects', 7, Marker.DELETE
b.pose.orientation.w = 1.
(root/'marker_array_ros2_humble.cdr').write_bytes(serialize_message(MarkerArray(markers=[a,b])))
(root/'marker_array_ros2_humble.json').write_text(json.dumps(dict(
    generated_by='rclpy.serialization.serialize_message(MarkerArray)',
    marker_count=2, first=dict(id=7, ns='test_objects', type=5, action=0,
    position=[1.,2.,3.], quaternion=[0.,0.,0.,1.],scale=[1.,2.,3.],
    color=[.25,.5,.75,1.],points=[[-.5,-1.,-1.5],[.5,1.,1.5]],text='independent fixture'),
    second_action=2),indent=2))
