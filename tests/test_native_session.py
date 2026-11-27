from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock,patch
import torch
from safetensors.torch import save_file,load_file
from fxbex.native_session import NativeSession
from fxbex.web_demo import WebDemo
from fxbex.execution_fxb import _entry_input, ExecutionPlanError, validate_execution_plan


class CheckedIndexTest(TestCase):
    def test_narrowing_requires_versioned_external_input_contract(self):
        index = {'kind': 'user_input', 'shape': [2], 'stride': [1], 'dtype': 'int32',
                 'index_input_dtype': 'int64', 'value_range': {'min': 0, 'max_exclusive': 8}}
        plan = {
            'schema': 'tclmachine.execution-plan.v3',
            'source_metadata': {'architecture': 'Fixture'},
            'compatibility': dict.fromkeys(('torch', 'furiosa-torch', 'furiosa-tcc'), 'fixture'),
            'compiler_identity': {'version': 'fixture', 'revision': 'fixture'},
            'device_config': {'fusion': 8, 'mesh': 'single'},
            'tensors': {'ids': index, 'y': {'kind': 'intermediate', 'shape': [2],
                                           'stride': [1], 'dtype': 'float32'}},
            'kernels': {'lookup': {'edf_path': 'fixture.edf', 'edf_sha256': '0' * 64,
                                   'expected_num_fusion': 8, 'abi': {'inputs': ['i32'], 'outputs': ['f32']}}},
            'entries': {'forward': {'inputs': ['ids'], 'outputs': ['y'], 'state_updates': {},
                                    'steps': [{'kernel': 'lookup', 'inputs': ['ids'], 'outputs': ['y']}]}},
        }
        validate_execution_plan(plan)
        with self.assertRaisesRegex(ExecutionPlanError, 'require schema v3'):
            validate_execution_plan(dict(plan, schema='tclmachine.execution-plan.v1'))
        for field, value in (('kind', 'state'), ('index_input_dtype', 'float32')):
            with self.subTest(field=field), self.assertRaises(ExecutionPlanError):
                validate_execution_plan(dict(plan, tensors=plan['tensors'] | {'ids': dict(index, **{field: value})}))

    def test_original_width_is_checked_before_narrowing(self):
        spec = {'shape': [2], 'stride': [1], 'dtype': 'int32',
                'index_input_dtype': 'int64', 'value_range': {'min': 0, 'max_exclusive': 8}}
        ids = torch.tensor([0, 7], dtype=torch.int64)
        result = _entry_input(ids, spec, torch, 'ids')
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.tolist(), ids.tolist())
        for bad in (-2**63, -1, 8, 2**32 + 1, 2**63 - 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ExecutionPlanError, 'value_range'):
                _entry_input(torch.tensor([0, bad], dtype=torch.int64), spec, torch, 'ids')
        with self.assertRaisesRegex(ExecutionPlanError, 'metadata'):
            _entry_input(ids.int(), spec, torch, 'ids')

class NativeTransportTest(TestCase):
    def setUp(self):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.package=self.root/'native.fxbx';self.source=self.root/'input.safetensors'
        save_file({'x':torch.arange(6,dtype=torch.float32).reshape(2,3)},str(self.source))
        self.model=SimpleNamespace(plan={'entries':{'forward':{'inputs':['x'],'outputs':['y']}},'tensors':{}},memory={},failed=False,
            __enter__=Mock(),close=Mock(),predict=lambda values:(values[0]*2,))
    def test_native_web_sample_uses_tensor_file_and_saves_all_outputs(self):
        with patch('fxbex.execution_fxb.ExecutionFXB',return_value=self.model):
            with NativeSession(self.package,report_dir=self.root/'reports') as session:
                demo=WebDemo(session);self.assertFalse(demo.config['lora']['supported'])
                response=demo.run({'tensors':str(self.source)},lambda callback:callback())
                result=response['result'];self.assertTrue(Path(result['result_file']).is_relative_to(session.report_dir))
                self.assertTrue(torch.equal(load_file(result['result_file'])['y'],torch.arange(6,dtype=torch.float32).reshape(2,3)*2))
                self.assertEqual(result['outputs']['y']['shape'],[2,3])
                with self.assertRaises(ValueError):session.predict({'tensors':str(self.source),'loras':[{'id':'bad'}]})
        self.model.close.assert_called_once()
    def test_stateful_native_plan_is_not_silently_shared_across_requests(self):
        self.model.plan['tensors']['cache']={'kind':'state'}
        with patch('fxbex.execution_fxb.ExecutionFXB',return_value=self.model):
            with self.assertRaisesRegex(ValueError,'lifecycle adapter'):NativeSession(self.package,report_dir=self.root/'reports').__enter__()
        self.model.__enter__.assert_not_called()
