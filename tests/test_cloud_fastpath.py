import numpy as np
import pytest

from gabarit.cloud import from_raw, working_set, dual_return_split, _selected_return_split
from gabarit.rosbag import RawCloud


@pytest.mark.parametrize('columns', [1, 2, 7, 24])
@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('forward,up,pattern', [('-y','+z','mechanical'), ('+x','+z','mechanical'),
                                                ('+z','+x','mirror')])
def test_roi_conversion_matches_full_scan_followed_by_public_working_mask(columns, dual, forward, up, pattern):
    rings = 16
    rng = np.random.default_rng(20261110 + columns)
    dtype = [('x','f4'), ('y','f4'), ('z','f4'), ('ring','u2'), ('intensity','f4')]
    points = np.zeros(columns*rings, dtype=dtype)
    for name in 'xyz':
        points[name] = rng.uniform(-70,70,len(points))
    points['z'] *= .06
    points['ring'] = np.tile(np.arange(rings),columns)
    points['intensity'] = rng.uniform(0,255,len(points))
    points['x'][0] = np.nan
    raw = RawCloud(0., 'sensor', rings, columns, points)
    options = dict(up_axis=up, scan_pattern=pattern, dual_returns=dual, preserve_source_index=True)
    full = from_raw(raw, forward, 1., **options)
    expected = full.subset(working_set(full, 65.))
    actual = from_raw(raw, forward, 1., working_range=65., **options)
    for field in ('xyz','ring','column','intensity','split','source_index'):
        np.testing.assert_array_equal(getattr(actual,field),getattr(expected,field))


def test_selected_echoes_match_original_full_scan_even_when_partner_outside_roi():
    rng = np.random.default_rng(11)
    rings,columns = 8,17
    ranges = rng.uniform(1,400,rings*columns).astype('f4')
    ranges[:2*rings] = np.tile([50.,51.5,80.,100.,20.,22.,200.,210.],2)
    ranges[rings] = 300.  # retained 50m first echo, partner beyond working range
    squared = ranges*ranges
    valid = rng.random(len(ranges)) > .1
    valid[0] = valid[rings] = True
    selected = np.flatnonzero(valid & (ranges < 205.))
    expected = dual_return_split(squared,valid,rings,columns)[selected]
    actual = _selected_return_split(squared,valid,selected,rings,columns)
    np.testing.assert_array_equal(actual,expected)
    assert actual[np.flatnonzero(selected==0)[0]]
