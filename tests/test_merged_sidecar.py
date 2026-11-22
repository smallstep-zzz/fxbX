import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fxbex.execution_fxb import ExecutionFXB
from fxbex.graph_fxb import _sha


class FakeRunnable:
    def __init__(self, ir):
        self._inner = SimpleNamespace(expected_num_fusion=8,
                                      input_dram_shapes=[SimpleNamespace(host_sizes=[2])],
                                      output_dram_shapes=[SimpleNamespace(host_sizes=[2])])

    def to(self, device):
        return self


def plan():
    # x -> a -> b -> c -> y ; two repeated two-step blocks: [a, b] and [c, d]
    tensors = {n: {'shape': [2], 'kind': 'input' if n == 'x' else 'intermediate'} for n in 'xabcd'}
    steps = [{'kernel': 'k1', 'inputs': ['x'], 'outputs': ['a']},
             {'kernel': 'k2', 'inputs': ['a'], 'outputs': ['b']},
             {'kernel': 'k1', 'inputs': ['b'], 'outputs': ['c']},
             {'kernel': 'k2', 'inputs': ['c'], 'outputs': ['d']}]
    return {'tensors': tensors, 'entries': {'forward': {'steps': steps, 'outputs': ['d']}}}


class MergedSidecarLoading(unittest.TestCase):
    def load(self, groups):
        p = plan()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / 'm.fxbx'
            side = Path(str(pkg) + '.merged'); side.mkdir()
            raw = b'same-runnable'
            (side / 'r1.bin').write_bytes(raw)
            for g in groups:
                g.setdefault('member', 'r1.bin'); g.setdefault('sha256', hashlib.sha256(raw).hexdigest())
            (side / 'manifest.json').write_text(json.dumps({
                'schema': 'fxbx.merged-runnables.v1', 'execution_plan_sha256': _sha(p),
                'packages': {'furiosa-torch': '1', 'furiosa-tcc': '1'}, 'entries': {'forward': groups}}))
            calls = []
            ir = SimpleNamespace(Runnable=SimpleNamespace(deserialize=lambda b: calls.append(b) or b))
            fake = SimpleNamespace(path=pkg, plan=p, resident={}, device='npu', _merged_programs={},
                                   sdk=SimpleNamespace(Runnable=FakeRunnable, ir=ir))
            with patch('importlib.metadata.version', return_value='1'):
                merged, covered, reason = ExecutionFXB._merged_groups(fake, 'forward', p['entries']['forward'])
            return merged, covered, reason, calls

    def test_identical_members_are_loaded_once_and_shared(self):
        merged, covered, reason, calls = self.load([
            {'steps': [0, 1], 'inputs': ['x'], 'outputs': ['b']},
            {'steps': [2, 3], 'inputs': ['b'], 'outputs': ['d']}])
        self.assertEqual(reason, 'used')
        self.assertEqual(covered, {0, 1, 2, 3})
        self.assertEqual(len(calls), 1)
        self.assertIs(merged[0][1], merged[2][1])

    def test_dataflow_check_still_rejects_wrong_outputs(self):
        # group 1 claims 'a' as an output, but nothing after step 1 reads it
        _, _, reason, _ = self.load([
            {'steps': [0, 1], 'inputs': ['x'], 'outputs': ['a', 'b']},
            {'steps': [2, 3], 'inputs': ['b'], 'outputs': ['d']}])
        self.assertEqual(reason, 'group differs from plan dataflow')


if __name__ == '__main__':
    unittest.main()
