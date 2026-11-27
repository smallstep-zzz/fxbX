"""CPU output copies need no plan scan; NPU persistent outputs still need clones."""
from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from fxbex.execution_fxb import ExecutionFXB


class Tensor:
    def __init__(self, device='furiosa'):
        self.device=SimpleNamespace(type=device)
        self.copies=self.clones=0
    def cpu(self):
        self.copies+=1
        return Tensor('cpu')
    def clone(self):
        self.clones+=1
        return Tensor()


class OutputAliasTests(TestCase):
    def model(self, kind):
        tensor=Tensor()
        model=ExecutionFXB.__new__(ExecutionFXB)
        model._lock=RLock();model.stack=object();model.failed=False;model.programs={'k':object()}
        model.torch=SimpleNamespace(no_grad=nullcontext)
        model._requests=[];model._default_ready=True;model._sync=lambda:None
        model._admit_requests=lambda *args,**kwargs:{}
        recipe={'inputs':[],'outputs':['y'],'state_updates':{},
                'steps':[{'inputs':[] if kind=='zero_input' else ['x'],'outputs':['y']}]}
        model.plan={'entries':{'forward':recipe},'tensors':{'y':{}}}
        model.resident={'y':tensor} if kind=='resident' else {}
        model.states={'y':tensor} if kind=='state' else {}
        folded={'y':tensor} if kind=='folded' else {}
        model._sequences={'forward':(SimpleNamespace(run=lambda inputs:[tensor]),folded)}
        return model,tensor

    def test_cpu_copy_does_not_scan_any_kind_of_persistent_output(self):
        for kind in ('zero_input','resident','state','folded'):
            model,tensor=self.model(kind)
            with self.subTest(kind=kind), patch.dict('os.environ',{'FXBX_TRACE_STEPS':'0'}), \
                 patch('fxbex.execution_fxb._tensor_matches',return_value=True), \
                 patch('fxbex.execution_fxb._zero_input_outputs',side_effect=AssertionError('unneeded plan scan')):
                output=model.run('forward',output_device='cpu')[0]
            self.assertEqual(output.device.type,'cpu')
            self.assertIsNot(output,tensor)
            self.assertEqual((tensor.copies,tensor.clones),(1,0))

    def test_npu_output_still_clones_each_persistent_kind(self):
        for kind in ('zero_input','resident','state','folded'):
            model,tensor=self.model(kind)
            with self.subTest(kind=kind), patch.dict('os.environ',{'FXBX_TRACE_STEPS':'0'}), \
                 patch('fxbex.execution_fxb._tensor_matches',return_value=True):
                output=model.run('forward',output_device='furiosa')[0]
            self.assertEqual(output.device.type,'furiosa')
            self.assertIsNot(output,tensor)
            self.assertEqual((tensor.copies,tensor.clones),(0,1))
