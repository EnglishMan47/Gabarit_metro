"""An unobserved vertical tail cannot invent curvature and feed it back."""
from pathlib import Path
import numpy as np
import pytest
from gabarit.track import _smooth_profile, _vertical_tail


def test_recorded_sparse_tail_does_not_recycle_quadratic_height():
    fixture=Path(__file__).parent/'fixtures/vertical_fit_failure.npz'
    with np.load(fixture,allow_pickle=False) as data:
        s=data['x'];observed=data['supported'];old=data['fit']
        fitted,sigma,support=_smooth_profile(s,data['meas'].tolist(),data['prior'],data['prior_sig'],
                                            data['fallback'],3e7,2.,False)
    np.testing.assert_allclose(fitted,old,atol=1e-6)
    last=np.flatnonzero(support)[-1]
    repaired=_vertical_tail(s,fitted,support)
    np.testing.assert_array_equal(repaired[:last+1],fitted[:last+1])
    assert np.max(abs(repaired))<2.
    assert np.max(abs(np.diff(np.gradient(repaired[last+1:],s[last+1:]))))<1e-10
    # Physical floor-noise estimates and current-evidence flags stay separate.
    assert np.array_equal(observed,support) and np.isfinite(sigma).all()


@pytest.mark.parametrize('slope',[-.025,.025])
def test_real_heading_survives_a_missing_tail_without_claiming_new_measurements(slope):
    s=np.arange(3.,101.)
    z=.2+slope*(s-3.)
    known=s<40.
    z[~known]+=.01*(s[~known]-40.)**2
    actual=_vertical_tail(s,z,known)
    np.testing.assert_allclose(actual,.2+slope*(s-3.))
    assert not known[s>=40.].any()


def test_all_missing_frame_cannot_propagate_quadratic_tail():
    s=np.arange(3.,101.)
    z=.002*(s-3.)**2
    actual=_vertical_tail(s,z,np.zeros(len(s),bool))
    assert np.max(abs(np.diff(np.gradient(actual,s))))<1e-12
