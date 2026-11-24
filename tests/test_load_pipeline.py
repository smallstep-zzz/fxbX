from unittest import TestCase
import weakref
from fxbex.load_pipeline import WeightTransfers

class Value:pass

class PipelineTest(TestCase):
    def make(self,slots=2,budget=16):
        self.events=[];self.pending=[];self.live=[];self.reads=0
        class Event:
            def __init__(inner,ref):inner.ref=ref;inner.done=False
            def synchronize(inner):
                assert inner.ref() is not None,'Host source released before DMA completion'
                inner.done=True
        def upload(value,spec,asynchronous):
            self.pending.append(Event(weakref.ref(value)));return 'npu'
        def record():
            event=self.pending.pop();self.events.append(event);return event
        def sync():
            for event in [*self.pending,*self.events]:
                if not event.done:event.synchronize()
            self.pending.clear()
        def read():
            self.reads+=1
            value=Value();self.live.append(weakref.ref(value));return value
        return WeightTransfers(upload=upload,synchronize=sync,record=record,slots=slots,max_bytes=budget),read
    def test_two_slots_hold_sources_until_completion(self):
        window,read=self.make()
        with window:
            for _ in range(6):
                self.assertEqual(window.transfer(read,{},8),'npu')
                self.assertLessEqual(sum(v() is not None for v in self.live),2)
                self.assertLessEqual(window.bytes,16)
            self.assertTrue(all(e.done for e in self.events[:-2]))
        self.assertTrue(all(e.done for e in self.events));self.assertFalse(window.pending)
    def test_byte_budget_and_oversize_fallback(self):
        window,read=self.make(budget=10)
        with window:
            window.transfer(read,{},8);window.transfer(read,{},8)
            self.assertTrue(self.events[0].done)
            window.transfer(read,{},20)
            self.assertTrue(all(e.done for e in self.events))
        self.assertEqual(window.stats['oversize_serial'],1)
    def test_reader_failure_drains_inflight(self):
        window,read=self.make()
        def fail():raise ValueError('checkpoint changed')
        with self.assertRaises(ValueError),window:
            window.transfer(read,{},8);window.transfer(fail,{},8)
        self.assertTrue(all(e.done for e in self.events));self.assertFalse(window.pending)
    def test_upload_failure_after_enqueue_retains_current_source(self):
        window,read=self.make();upload=window.upload
        def fail(value,spec,async_copy):
            upload(value,spec,async_copy);raise RuntimeError('device admission failure')
        window.upload=fail
        with self.assertRaises(RuntimeError),window:window.transfer(read,{},8)
        self.assertFalse(self.pending)
    def test_serial_and_missing_events(self):
        for record in (None,False):
            window,read=self.make();window.record=record;window.slots=1
            with window:
                for _ in range(3):window.transfer(read,{},8)
            self.assertFalse(window.pending);self.assertEqual(window.stats['count'],3)
    def test_sync_failure_keeps_sources(self):
        window,read=self.make()
        def fail():raise RuntimeError('sync failed')
        with self.assertRaises(RuntimeError):
            with window:
                window.transfer(read,{},8);window.synchronize=fail
        self.assertTrue(window.pending);self.assertTrue(self.live[0]() is not None)
    def test_failed_enqueue_and_failed_fence_hold_current_buffer(self):
        window,read=self.make()
        def upload(value,spec,async_copy):raise RuntimeError('enqueue failed after submission')
        def sync():raise RuntimeError('device unavailable')
        window.upload=upload;window.synchronize=sync
        with self.assertRaises(RuntimeError),window:window.transfer(read,{},8)
        self.assertIsNotNone(window.current);self.assertIsNotNone(self.live[0]())
