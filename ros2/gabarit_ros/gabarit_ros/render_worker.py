"""Bounded background rendering; reset cannot publish an old recording late."""
import threading


class LatestRenderer:
    def __init__(self, build, deliver, error):
        self.build,self.deliver,self.error=build,deliver,error
        self._condition=threading.Condition()
        self._publication=threading.Lock()
        self._pending=None;self._generation=0;self._closed=False
        self._thread=threading.Thread(target=self._run,name='gabarit-visualization',daemon=True)
        self._thread.start()

    def submit(self, job):
        with self._condition:
            if not self._closed:
                self._pending=(self._generation,job)
                self._condition.notify()

    def reset(self, clear):
        # Building/encoding never holds this lock. Only publication is ordered
        # with DELETEALL, so an old in-flight build cannot revive old boxes.
        with self._publication:
            with self._condition:
                self._generation+=1;self._pending=None
            clear()

    def close(self):
        with self._condition:
            self._closed=True;self._generation+=1;self._pending=None
            self._condition.notify()
        self._thread.join(timeout=2)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._closed or self._pending is not None)
                if self._closed:return
                generation,job=self._pending;self._pending=None
            try:
                payload=self.build(job)
                with self._publication:
                    with self._condition:
                        valid=not self._closed and generation==self._generation
                    if valid:self.deliver(payload)
            except Exception as error:
                self.error(error)
