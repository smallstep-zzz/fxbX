from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock,patch
import torch
from safetensors.torch import save_file,load_file
from fxbex.native_session import NativeSession
from fxbex.web_demo import WebDemo

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
