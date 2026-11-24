# Bounded weight transfers

`ExecutionFXB` uses SDK-owned asynchronous H2D copies with at most two pending
host source tensors. The original checkpoint identity, shape, dtype, stride,
value-range and frozen-buffer checks still run. No compiler or custom DMA engine
is involved. CPU-resident dictionaries and header-first `SafeTensorWeights` use
the same transfer window; duplicated execution-plan targets upload once.

Before preparing another tensor, the window retires an old SDK completion event
if the slot or byte limit would be exceeded. A source stays referenced until that
event completes. Read/submit errors drain pending work; if completion itself
fails, the failed session retains the buffers. A partially loaded session is
never returned as ready.

```python
from fxbx import ExecutionFXB
with ExecutionFXB('model.fxbx', load_buffers=2,
                  load_buffer_bytes=256 * 1024**2) as model:
    print(model.load_stats)
```

`load_buffers=1` provides serial comparison/recovery. SDKs without the public
`Event` API use the serial path. The 256 MiB limit controls queued host staging,
not total process RSS, SDK allocation caches, resident HBM or executable memory.
Checkpoint estimates conservatively include the original tensor and its
cast/slice result. An individual larger tensor uses the existing one-at-a-time
synchronous path and is counted in `oversize_serial`; it cannot be split without
an SDK tensor-transfer contract. `peak_queued_bytes` reports the bounded asynchronous queue.
`peak_staging_bytes` may therefore exceed the
queue budget for that one serial tensor.

`load_stats` separates host read/cast time, submission time, completion waiting,
window wall time and staging high-water mark. It does not claim DMA engine time
or physical HBM measurements. Pageable source buffers are retained explicitly;
SDK pinning/staging is left to the SDK. We do not assume CUDA's pinning behavior.
Request inputs/outputs, recurrent state reset and LoRA parameter replacement keep
their existing synchronous paths. They need separate workload evidence before
changing their ordering or lifetimes.
