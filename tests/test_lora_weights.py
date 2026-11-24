import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
import torch
from safetensors.torch import save_file
from fxbex.lora_weights import BoundLoras

class BoundLoraTest(TestCase):
    def make(self):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);(root/'adapters').mkdir()
        a=torch.tensor([[1.,2.]]);b=torch.tensor([[.1],[.2]])
        save_file({'a':a,'b':b},str(root/'adapters/style.safetensors'))
        doc={'schema_version':1,'adapters':[{'id':'style','name':'Style','file':'adapters/style.safetensors','components':{'net':[{'target':'weight','a':'a','b':'b','alpha':1}]}}]}
        (root/'lora.json').write_text(json.dumps(doc));base=torch.tensor([[1.,2.],[3.,4.]])
        m=SimpleNamespace(plan={'tensors':{'w':{'kind':'parameter','target':'weight','dtype':'float32','shape':[2,2]}},'entries':{'forward':{'state_updates':{}}}},weights={'weight':base},updates=[])
        m.replace_parameters=lambda x,**kw:m.updates.append(x)
        return root,m,doc
    def test_merge_and_restore_without_accumulation(self):
        root,m,_=self.make();l=BoundLoras(root,{'net':m});base=m.weights['weight'].clone()
        l.apply([{'id':'style','scale':.5}]);want=base+torch.tensor([[.05,.1],[.1,.2]])
        self.assertTrue(torch.equal(m.updates[-1]['weight'],want))
        l.apply([{'id':'style','scale':.5}]);self.assertEqual(len(m.updates),1)
        l.apply([{'id':'style','scale':1.0}]);l.apply([])
        self.assertTrue(torch.equal(m.updates[-1]['weight'],base));self.assertTrue(torch.equal(m.weights['weight'],base))
    def test_bad_shapes_and_unbound_tensors_fail(self):
        root,m,_=self.make();save_file({'a':torch.ones(1,2),'b':torch.ones(3,1)},str(root/'adapters/style.safetensors'))
        with self.assertRaises(ValueError):BoundLoras(root,{'net':m})
    def test_device_failure_stops_further_updates(self):
        root,m,_=self.make();l=BoundLoras(root,{'net':m})
        def fail(values,**kw):raise RuntimeError('device error')
        m.replace_parameters=fail
        with self.assertRaises(RuntimeError):l.apply([{'id':'style','scale':1.0}])
        with self.assertRaises(RuntimeError):l.apply([])
    def test_guarded_parameter_and_path_escape_fail(self):
        root,m,doc=self.make();m.plan['frozen_buffer_guards']=[{'target':'weight'}]
        with self.assertRaises(ValueError):BoundLoras(root,{'net':m})
        m.plan.pop('frozen_buffer_guards');doc['adapters'][0]['file']='../outside.safetensors';(root/'lora.json').write_text(json.dumps(doc))
        with self.assertRaises(ValueError):BoundLoras(root,{'net':m})
