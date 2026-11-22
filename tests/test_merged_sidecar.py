import hashlib
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fxbex.execution_fxb import ExecutionFXB, ExecutionPlanError
from fxbex.graph_fxb import _sha
from fxbex.merged import MergedError, prunable_kernels, pruned_kernels

RAW = b'same-runnable'


class FakeRunnable:
    def __init__(self, ir):
        self._inner = SimpleNamespace(expected_num_fusion=8,
                                      input_dram_shapes=[SimpleNamespace(host_sizes=[2])],
                                      output_dram_shapes=[SimpleNamespace(host_sizes=[2])])

    def to(self, device):
        return self


def plan():
    # x -> a -> b -> c -> d ; two repeated two-step blocks: [a, b] and [c, d]
    tensors = {n: {'shape': [2], 'kind': 'input' if n == 'x' else 'intermediate'} for n in 'xabcd'}
    steps = [{'kernel': 'k1', 'inputs': ['x'], 'outputs': ['a']},
             {'kernel': 'k2', 'inputs': ['a'], 'outputs': ['b']},
             {'kernel': 'k1', 'inputs': ['b'], 'outputs': ['c']},
             {'kernel': 'k2', 'inputs': ['c'], 'outputs': ['d']}]
    return {'tensors': tensors, 'kernels': {'k1': {}, 'k2': {}},
            'entries': {'forward': {'steps': steps, 'outputs': ['d'], 'inputs': ['x'], 'state_updates': []}}}


GROUPS = [{'steps': [0, 1], 'inputs': ['x'], 'outputs': ['b']},
          {'steps': [2, 3], 'inputs': ['b'], 'outputs': ['d']}]


def manifest(p, groups, packages=None, pruned=()):
    for g in groups:
        g.setdefault('member', 'r1.bin'); g.setdefault('sha256', hashlib.sha256(RAW).hexdigest())
    return {'schema': 'fxbx.merged-runnables.v1', 'execution_plan_sha256': _sha(p),
            'packages': packages or {'furiosa-torch': '1', 'furiosa-tcc': '1'},
            'entries': {'forward': groups}, 'pruned_kernels': list(pruned)}


class MergedLoading(unittest.TestCase):
    def load(self, groups, *, embedded=False, packages=None, pruned=(), installed='1', fail_load=False):
        p = plan()
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / 'm.fxbx'
            m = manifest(p, groups, packages, pruned)
            with zipfile.ZipFile(pkg, 'w') as z:
                z.writestr('manifest.json', '{}')
                if embedded:
                    z.writestr('merged/manifest.json', json.dumps(m)); z.writestr('merged/r1.bin', RAW)
            if not embedded:
                side = Path(str(pkg) + '.merged'); side.mkdir()
                (side / 'r1.bin').write_bytes(RAW); (side / 'manifest.json').write_text(json.dumps(m))
            calls = []

            def deserialize(b):
                if fail_load:
                    raise RuntimeError('format changed')
                calls.append(b); return b
            ir = SimpleNamespace(Runnable=SimpleNamespace(deserialize=deserialize))
            fake = SimpleNamespace(path=pkg, plan=p, resident={}, device='npu', _merged_programs={},
                                   pruned=set(pruned), sdk=SimpleNamespace(Runnable=FakeRunnable, ir=ir))
            fake._merged_groups_checked = lambda e, r: ExecutionFXB._merged_groups_checked(fake, e, r)
            with patch('importlib.metadata.version', return_value=installed), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                merged, covered, reason = ExecutionFXB._merged_groups(fake, 'forward', p['entries']['forward'])
            return merged, covered, reason, calls, [str(x.message) for x in w]

    def test_identical_members_are_loaded_once_and_shared(self):
        merged, covered, reason, calls, _ = self.load([dict(g) for g in GROUPS])
        self.assertEqual(reason, 'used')
        self.assertEqual(covered, {0, 1, 2, 3})
        self.assertEqual(len(calls), 1)
        self.assertIs(merged[0][1], merged[2][1])

    def test_dataflow_check_still_rejects_wrong_outputs(self):
        groups = [dict(GROUPS[0], outputs=['a', 'b']), dict(GROUPS[1])]
        _, _, reason, _, warned = self.load(groups)
        self.assertEqual(reason, 'group differs from plan dataflow')
        self.assertTrue(any('running per step' in m for m in warned))

    def test_embedded_members_are_read_from_the_package(self):
        merged, covered, reason, calls, _ = self.load([dict(g) for g in GROUPS], embedded=True)
        self.assertEqual(reason, 'used')
        self.assertEqual(covered, {0, 1, 2, 3})

    def test_sdk_version_difference_warns_and_runs(self):
        _, covered, reason, _, warned = self.load([dict(g) for g in GROUPS], installed='2')
        self.assertTrue(reason.startswith('used'))
        self.assertEqual(covered, {0, 1, 2, 3})
        self.assertTrue(any('another SDK version' in m for m in warned))

    def test_unusable_groups_with_pruned_edfs_are_an_error(self):
        with self.assertRaisesRegex(ExecutionPlanError, 'omits 2 EDFs'):
            self.load([dict(g) for g in GROUPS], embedded=True, pruned=('k1', 'k2'), fail_load=True)


class Pruning(unittest.TestCase):
    def test_only_kernels_whose_every_use_is_merged_can_be_pruned(self):
        p = plan()
        self.assertEqual(prunable_kernels(p, manifest(p, [dict(g) for g in GROUPS])), {'k1', 'k2'})
        # only the first block merged: k1/k2 are still used by the per-step second block
        self.assertEqual(prunable_kernels(p, manifest(p, [dict(GROUPS[0])])), set())

    def test_declared_pruned_kernels_are_checked(self):
        p = plan()
        with self.assertRaisesRegex(MergedError, 'still need'):
            pruned_kernels(p, manifest(p, [dict(GROUPS[0])], pruned=('k1',)))
        self.assertEqual(pruned_kernels(p, manifest(p, [dict(g) for g in GROUPS], pruned=('k2',))), {'k2'})

    def test_folded_steps_keep_their_kernels(self):
        p = plan()
        p['tensors']['x']['kind'] = 'parameter'  # step 0 reads only a parameter: folded at load
        self.assertNotIn('k1', prunable_kernels(p, manifest(p, [dict(g) for g in GROUPS])))


if __name__ == '__main__':
    unittest.main()
