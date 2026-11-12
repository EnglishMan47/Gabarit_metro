import asyncio
from web.server import LiveHub
import web.session
import threading


def test_slow_viewer_cannot_block_fast_viewer_or_accumulate_unbounded_frames():
    async def scenario():
        release=asyncio.Event()
        class Viewer:
            def __init__(self,slow=False):self.frames=[];self.slow=slow
            async def send_bytes(self,payload):
                self.frames.append(payload)
                if self.slow:await release.wait()
        fast,slow=Viewer(),Viewer(True)
        hub=LiveHub();hub.loop=asyncio.get_running_loop();hub.clients.update([fast,slow])
        await hub._broadcast(b'first');await asyncio.sleep(.02)
        for i in range(100):await hub._broadcast(str(i).encode())
        await asyncio.sleep(.02)
        assert fast.frames[-1]==b'99' and slow.frames==[b'first']
        assert len(hub._senders)<=2 and len(hub._pending)<=2
        release.set();await asyncio.sleep(.02)
        assert slow.frames[-1]==b'99'
    asyncio.run(scenario())


def test_ros_publish_never_waits_for_web_encoding_and_keeps_only_latest(monkeypatch):
    async def scenario():
        entered=threading.Event();release=threading.Event();calls=[]
        def encode(result,meta):
            calls.append(result)
            if result==0:
                entered.set()
                assert release.wait(3)
            return str(result).encode()
        monkeypatch.setattr(web.session,'pack_frame',encode)
        class Viewer:
            def __init__(self):self.frames=[]
            async def send_bytes(self,payload):self.frames.append(payload)
        viewer=Viewer();hub=LiveHub();hub.loop=asyncio.get_running_loop();hub.clients.add(viewer)
        hub.publish_frame_threadsafe(0,{})
        for _ in range(100):
            if entered.is_set():break
            await asyncio.sleep(.01)
        assert entered.is_set()
        for i in range(1,101):hub.publish_frame_threadsafe(i,{})
        assert calls==[0] and hub._latest_frame[0]==100
        release.set()
        for _ in range(100):
            if viewer.frames:break
            await asyncio.sleep(.01)
        assert viewer.frames==[b'100'] and calls==[0,100]
        assert not hub._packing
    asyncio.run(scenario())


def test_paused_stream_keeps_lazy_frame_without_packing_every_ros_callback(monkeypatch):
    calls=[]
    monkeypatch.setattr(web.session,'pack_frame',lambda result,meta:calls.append(result) or str(result).encode())
    hub=LiveHub()
    for i in range(30):hub.publish_frame_threadsafe(i,{'frame':i})
    assert calls==[]
    assert hub.latest_payload()==b'29' and calls==[29]
    hub.clear()
    assert hub.latest_payload() is None


def test_late_encoded_frame_cannot_overwrite_newer_cache_or_revive_cleared_stream():
    async def scenario():
        hub=LiveHub();hub.publish_frame_threadsafe(1,{'frame':1})
        generation=hub._generation
        hub.publish_frame_threadsafe(2,{'frame':2})
        await hub._broadcast(b'old',generation)
        assert hub.last is None
        hub.clear();await hub._broadcast(b'old',generation)
        assert hub.latest_payload() is None
    asyncio.run(scenario())
