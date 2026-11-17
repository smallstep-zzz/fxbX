"""RNGD check: sequence replay must match the per-step loop bit for bit, and be timed.

Usage: python tests/hardware/compare_sequence_replay.py MODEL.fxbx [repeats]
Expects MODEL.fxbx.weights/, MODEL.fxbx.sample.safetensors (image0, image1...) and
MODEL.fxbx.reference.safetensors ("<frame>.<output>") next to the package.
"""
import json
import statistics
import sys
import time

import torch
from safetensors.torch import load_file
from fxbx import GraphFXB, SafeTensorWeights

path = sys.argv[1]
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 6
samples = load_file(path + '.sample.safetensors')
reference = load_file(path + '.reference.safetensors')
frames = sorted(samples)

with GraphFXB(path, weights=SafeTensorWeights(path + '.weights')) as session:
    model = session  # GraphFXB returns the execution-plan session itself
    sequences = dict(session._sequences)
    assert sequences, 'no stateless entry was prepared as a sequence'

    def run(use_sequence, frame):
        session._sequences = sequences if use_sequence else {}
        return model.predict((samples[frame],))

    for index, frame in enumerate(frames):
        fast, loop = run(True, frame), run(False, frame)
        assert all(torch.equal(a, b) for a, b in zip(fast, loop)), 'sequence differs from step loop'
        for output, value in enumerate(fast):
            torch.testing.assert_close(value, reference[f'{index}.{output}'], rtol=.02, atol=.02)
    times = {'sequence': [], 'loop': []}
    for k in range(repeats):
        for mode in (('sequence', 'loop') if k % 2 == 0 else ('loop', 'sequence')):
            start = time.perf_counter()
            run(mode == 'sequence', frames[k % len(frames)])
            times[mode].append(time.perf_counter() - start)
    session._sequences = sequences
    print(json.dumps({'package': path, 'bitwise_equal': True,
                      'steps': len(session.plan['entries']['forward']['steps']),
                      'folded_tensors': sum(len(f) for _, f in sequences.values()),
                      **{f'{m}_p50_ms': round(1000 * statistics.median(v), 2) for m, v in times.items()}}))
