"""Measure a package through fxbX with and without its .merged/ sidecar, alternating in one session.

Usage: bench_sidecar.py MODEL.fxbx WEIGHTS_DIR (--inputs GLOB [--expected GLOB] | --sample F --reference F) [--atol A] [--reps N] [--trace PREFIX]
Both variants run in the same session so host load affects them alike.
"""
import argparse, glob, json, os, statistics, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from fxbx import GraphFXB, SafeTensorWeights

a = argparse.ArgumentParser()
a.add_argument('pkg'); a.add_argument('weights'); a.add_argument('--inputs'); a.add_argument('--expected')
a.add_argument('--sample'); a.add_argument('--reference'); a.add_argument('--atol', type=float, default=.02)
a.add_argument('--reps', type=int, default=20); a.add_argument('--trace')
a = a.parse_args()
if a.sample:
    sample, ref = load_file(a.sample), load_file(a.reference)
    cases = [((sample[f],), tuple(ref[k] for k in sorted((k for k in ref if k.startswith(f'{i}.')), key=lambda k: int(k.split('.')[1]))))
             for i, f in enumerate(sorted(sample))]
else:
    one = lambda pattern: tuple(next(iter(load_file(p).values())) for p in sorted(glob.glob(pattern)))
    cases = [(one(a.inputs), one(a.expected) if a.expected else None)]

side = a.pkg + '.merged'
with GraphFXB(a.pkg, weights=SafeTensorWeights(a.weights)) as s:
    merged_seq, merged_status = dict(s._sequences), dict(s.merged_status)
    os.rename(side, side + '.off')
    try:
        s._build_sequences(); plain_seq, plain_status = dict(s._sequences), dict(s.merged_status)
    finally:
        os.rename(side + '.off', side)
    variants = {'merged': merged_seq, 'unmerged': plain_seq}
    def run(name, inputs):
        s._sequences = variants[name]; return s.predict(inputs)
    report = {}
    outs = {name: [run(name, c[0]) for c in cases] for name in variants}
    for name in variants:
        errs, ok = [], True
        for (inp, ref), out in zip(cases, outs[name]):
            if ref is None: continue
            errs.append([float((o.float() - r.float()).abs().max()) for o, r in zip(out, ref)])
            try:
                for o, r in zip(out, ref): torch.testing.assert_close(o, r, rtol=.02, atol=a.atol)
            except AssertionError: ok = False
        report[name] = {'reference_pass': ok if errs else None, 'max_abs_err_vs_reference': errs}
    times = {name: [] for name in variants}
    for k in range(a.reps):
        for name in (('merged', 'unmerged') if k % 2 == 0 else ('unmerged', 'merged')):
            t = time.perf_counter(); run(name, cases[k % len(cases)][0]); times[name].append(time.perf_counter() - t)
    for name in variants: report[name]['p50_ms'] = round(1000 * statistics.median(times[name]), 3)
    report['merged']['status'], report['unmerged']['status'] = merged_status.get('forward'), plain_status.get('forward')
    if a.trace:
        from furiosa.torch.profiler import Profiler
        for name in variants:
            path = f'{a.trace}.{name}.json'
            with Profiler(path): run(name, cases[0][0])
            ev = json.loads(Path(path).read_text())['traceEvents']
            tasks = [e for e in ev if e.get('pid') == 0 and '::Task' in e.get('name', '')]
            tid = max({e['tid'] for e in tasks}, key=lambda t: sum(e['tid'] == t for e in tasks)) if tasks else None
            iv = sorted((e['ts'], e['ts'] + e['dur']) for e in tasks if e['tid'] == tid); busy = 0; cs = ce = None
            for x, z in iv:
                if ce is None or x > ce:
                    busy += (ce - cs) if ce is not None else 0; cs, ce = x, z
                else: ce = max(ce, z)
            busy += (ce - cs) if ce is not None else 0
            report[name].update(npu_tasks=len(iv), npu_busy_ms=round(busy / 1000, 3),
                                cpu_routes=sum('run_by_cpu' in e.get('name', '') for e in ev))
    s._sequences = merged_seq
diff = [[float((x.float() - y.float()).abs().max()) for x, y in zip(m, b)] for m, b in zip(outs['merged'], outs['unmerged'])]
print(json.dumps({'pkg': a.pkg, **report, 'max_abs_diff_merged_vs_unmerged': diff,
                  'speedup': round(report['unmerged']['p50_ms'] / report['merged']['p50_ms'], 2)}))
