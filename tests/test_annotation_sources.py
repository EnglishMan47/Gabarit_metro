import json
from pathlib import Path

import numpy as np
import pytest

from gabarit.annotation_sources import decode_marker_array, MarkerCdrError, point_membership

FIXTURE = Path(__file__).resolve().parent/'fixtures/marker_array_ros2_humble.cdr'


def test_cdr_matches_independent_ros_serializer_fixture():
    expected = json.loads(FIXTURE.with_suffix('.json').read_text())
    actual = decode_marker_array(FIXTURE.read_bytes())
    assert len(actual) == expected['marker_count']
    for key,value in expected['first'].items():
        assert actual[0][key] == value
    assert actual[1]['action'] == expected['second_action']
    assert actual[0]['lifetime_s'] == pytest.approx(2.0000005)


@pytest.mark.parametrize('cut', [0, 7, 40, 100, 300, -1])
def test_partial_marker_payload_is_rejected(cut):
    with pytest.raises((MarkerCdrError, UnicodeDecodeError)):
        decode_marker_array(FIXTURE.read_bytes()[:cut])


def test_membership_uses_recorded_marker_pose_without_a_detector_path():
    marker = decode_marker_array(FIXTURE.read_bytes())[0]
    points = np.array([[1.,2.,3.], [1.4,2.9,4.4], [3.,2.,3.]])
    assert point_membership(points, marker).tolist() == [True,True,False]
    marker['quaternion'] = [0.,0.,np.sin(np.pi/4),np.cos(np.pi/4)]
    assert point_membership(np.array([[1.9,2.,3.]]),marker).item()
    marker['action'] = 2
    with pytest.raises(ValueError):
        point_membership(points,marker)
