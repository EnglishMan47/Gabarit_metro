import importlib.util
from pathlib import Path
import threading
import time

spec=importlib.util.spec_from_file_location('render_worker',Path(__file__).parents[1]/'ros2/gabarit_ros/gabarit_ros/render_worker.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def wait(predicate):
    deadline=time.monotonic()+3
    while not predicate() and time.monotonic()<deadline:time.sleep(.005)
    assert predicate()


def test_busy_rendering_keeps_one_latest_frame_without_blocking_decisions():
    entered=threading.Event();release=threading.Event();seen=[];built=[]
    def build(job):
        built.append(job)
        if job==0:entered.set();assert release.wait(3)
        return job
    worker=module.LatestRenderer(build,seen.append,lambda e:None)
    try:
        worker.submit(0);assert entered.wait(3)
        for i in range(1,101):worker.submit(i)
        assert built==[0] and worker._pending[1]==100
        release.set();wait(lambda:len(seen)==2)
        assert seen==[0,100] and built==[0,100]
    finally:release.set();worker.close()


def test_switch_cannot_publish_a_late_old_frame_after_deleteall():
    entered=threading.Event();release=threading.Event();seen=[]
    def build(job):
        if job=='old':entered.set();assert release.wait(3)
        return job
    worker=module.LatestRenderer(build,seen.append,lambda e:None)
    try:
        worker.submit('old');assert entered.wait(3)
        worker.reset(lambda:seen.append('DELETEALL'))
        worker.submit('new');release.set();wait(lambda:'new' in seen)
        assert seen==['DELETEALL','new']
    finally:release.set();worker.close()


def test_render_error_does_not_kill_worker_and_close_prevents_late_publication():
    seen=[];errors=[]
    def build(job):
        if job=='bad':raise ValueError('bad frame')
        return job
    worker=module.LatestRenderer(build,seen.append,errors.append)
    worker.submit('bad');wait(lambda:bool(errors))
    worker.submit('good');wait(lambda:bool(seen));worker.close()
    worker.submit('late')
    assert seen==['good'] and isinstance(errors[0],ValueError)
