"""Bounded host staging for SDK-owned, event-fenced weight transfers."""
from collections import deque
from time import perf_counter


class WeightTransfers:
    def __init__(self, *, upload, synchronize, record=None, slots=2, max_bytes=256*1024**2):
        if type(slots) is not int or slots not in (1,2):raise ValueError('Weight transfer slots must be 1 or 2')
        if type(max_bytes) is not int or max_bytes<1:raise ValueError('Weight buffer budget must be positive')
        self.upload,self.synchronize,self.record=upload,synchronize,record
        self.started=perf_counter()
        self.slots=slots if callable(record) else 1;self.budget=max_bytes;self.pending=deque();self.current=None;self.bytes=0
        self.stats={'mode':'double_buffer' if self.slots==2 else 'serial','read_s':0.,'submit_s':0.,'wait_s':0.,'count':0,'peak_queued_bytes':0,'peak_staging_bytes':0,'oversize_serial':0}

    def _retire(self):
        value,event,size=self.pending[0]
        start=perf_counter()
        event.synchronize()  # Keep the source alive until the SDK completes its copy.
        self.stats['wait_s']+=perf_counter()-start
        self.pending.popleft();self.bytes-=size

    def transfer(self, read, spec, estimated_bytes):
        if type(estimated_bytes) is not int or estimated_bytes<0:raise ValueError('Unknown weight staging size')
        while self.pending and (len(self.pending)>=self.slots or self.bytes+estimated_bytes>self.budget):self._retire()
        async_copy=self.slots==2 and estimated_bytes<=self.budget
        if not async_copy:
            while self.pending:self._retire()
        start=perf_counter();value=read();self.stats['read_s']+=perf_counter()-start;self.current=value
        start=perf_counter()
        try:
            output=self.upload(value,spec,async_copy)
            if async_copy:
                event=self.record()
                self.pending.append((value,event,estimated_bytes));self.bytes+=estimated_bytes
            else:
                self.synchronize()
                if estimated_bytes>self.budget:self.stats['oversize_serial']+=1
        except BaseException:
            # An exception can follow a successful enqueue. Fence the current source too.
            self.synchronize();self.current=None
            raise
        self.current=None
        self.stats['submit_s']+=perf_counter()-start;self.stats['count']+=1
        self.stats['peak_queued_bytes']=max(self.stats['peak_queued_bytes'],self.bytes)
        self.stats['peak_staging_bytes']=max(self.stats['peak_staging_bytes'],self.bytes,estimated_bytes)
        return output

    def __enter__(self):return self
    def __exit__(self,*exc):
        # Keep references if synchronization fails; the session retains this window.
        start=perf_counter();self.synchronize();self.stats['wait_s']+=perf_counter()-start
        self.pending.clear();self.current=None;self.bytes=0;self.stats['wall_s']=perf_counter()-self.started
