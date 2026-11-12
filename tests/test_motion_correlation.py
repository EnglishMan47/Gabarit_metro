import numpy as np
import pytest
from gabarit.egomotion import _score, NF, NX


def reference(prev,cur,shifts):
    result=[]
    for shift in shifts:
        a,b=(prev[shift:],cur[:NF-shift]) if shift>=0 else (prev[:NF+shift],cur[-shift:])
        if len(a)<NF//3:
            result.append(-np.inf);continue
        a,b=a-a.mean(),b-b.mean()
        result.append((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-6))
    return np.array(result)


@pytest.mark.parametrize('kind',['random','sparse','periodic','constant','large_offset'])
@pytest.mark.parametrize('displacement',[-5,0,21])
def test_fft_keeps_overlap_normalisation_and_registration_peak(kind,displacement):
    rng=np.random.default_rng(881)
    a=rng.normal(size=(NF,NX))
    if kind=='sparse':a*=rng.random(a.shape)<.04
    if kind=='periodic':a=np.tile(rng.normal(size=(10,NX)),(NF//10,1))
    if kind=='constant':a[:]=1.
    if kind=='large_offset':a+=1e9
    b=np.roll(a,-displacement,axis=0)
    shifts=np.r_[[-NF, -NF//2],np.arange(-7,58),[NF//2,NF]]
    before,after=reference(a,b,shifts),_score(a,b,shifts)
    np.testing.assert_allclose(after,before,atol=2e-12,rtol=2e-12)
    if kind not in ('constant','periodic'):
        assert shifts[np.argmax(after)]==shifts[np.argmax(before)]==displacement
