import json
from pathlib import Path
from types import SimpleNamespace as NS

from scripts import annotation_review
from gabarit.evaluation import load_annotations, score_frame_labels
from tests.sim import tunnel_scan


def test_review_starts_unknown_and_preserves_source_frame_identity(tmp_path,monkeypatch):
    raw=tunnel_scan()
    class Reader:
        def __init__(self,info):pass
        def __len__(self):return 6
        def read(self,index):return 1000000000+index,raw
        def close(self):pass
    monkeypatch.setattr(annotation_review,'BagReader',Reader)
    monkeypatch.setattr(annotation_review,'discover_bags',lambda _: [NS(id='independent/bag')])
    out=tmp_path/'review'
    assert annotation_review.main(['--data',str(tmp_path),'--out',str(out),'--every','2','--max-points','100'])==0
    data=json.loads((out/'annotations.json').read_text())
    frames=data['bags']['independent/bag']['frames']
    assert [r['frame'] for r in frames]==[0,2,4]
    assert all(r['obstacle'] is None and not r['reviewed'] for r in frames)
    labels=load_annotations(out/'annotations.json')
    result=score_frame_labels([dict(frame=0,obstacle=True)],labels.bags['independent/bag'])
    assert result['excluded_unknown_frames']==1 and result['fp']==result['fn']==0
    assert 'Pipeline' not in (out/'review.html').read_text(encoding='utf-8')
