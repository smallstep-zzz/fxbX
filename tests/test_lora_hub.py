import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import torch
from safetensors.torch import save_file,load_file
from fxbex.lora_weights import BoundLoras
from fxbex.lora_hub import HfLoraLibrary


class HubLoraTest(TestCase):
    def setUp(self):
        tmp=TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.path=self.root/'weights.safetensors';save_file({'a':torch.ones(1,2),'b':torch.ones(2,1)},str(self.path))
        self.model=SimpleNamespace(plan={'tensors':{'w':{'kind':'parameter','target':'weight','dtype':'float32','shape':[2,2]}},'entries':{'f':{'state_updates':{}}}},weights={'weight':torch.zeros(2,2)})
        self.pairs={'net':[{'target':'weight','a':'a','b':'b'}]}
        self.info=SimpleNamespace(id='owner/style',sha='a'*40,card_data={'base_model':'owner/base','license':'apache-2.0','widget':[{'text':'a portrait'}]},siblings=[SimpleNamespace(rfilename='weights.safetensors',size=self.path.stat().st_size)])
        self.bound=BoundLoras(self.root,{'net':self.model})
        self.hub=self.make(self.bound)
    def make(self,bound):
        return HfLoraLibrary(bound,base_models=['owner/base'],convert=lambda path:(load_file(str(path)),self.pairs),cache_dir=self.root/'cache')
    def install(self):return self.hub.install(repo='owner/style',revision='a'*40,filename='weights.safetensors')
    def mocks(self):
        api=patch('huggingface_hub.HfApi');download=patch('huggingface_hub.hf_hub_download',return_value=str(self.path))
        a=api.start();d=download.start();self.addCleanup(api.stop);self.addCleanup(download.stop);a.return_value.model_info.return_value=self.info
        return a,d
    def test_import_is_data_only_persistent_and_idempotent(self):
        _,download=self.mocks();key=self.install()
        self.assertEqual(len(self.bound.rows),1);self.assertFalse(self.bound.active)
        self.assertEqual(self.install(),key);download.assert_called_once()
        restored=BoundLoras(self.root,{'net':self.model});self.make(restored)
        self.assertEqual(restored.catalog()[0]['id'],key)
        self.assertEqual(restored.catalog()[0]['example_prompt'],'a portrait')
        target=self.hub.root/(key+'.safetensors');target.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'checksum'):self.make(BoundLoras(self.root,{'net':self.model}))
    def test_mismatch_and_budget_reject_before_download(self):
        _,download=self.mocks()
        self.info.card_data['base_model']='other/base'
        with self.assertRaisesRegex(ValueError,'base model'):self.install()
        self.info.card_data['base_model']='owner/base';self.info.siblings[0].size=10**10
        with self.assertRaisesRegex(ValueError,'budget'):self.install()
        download.assert_not_called();self.assertFalse(self.bound.rows)
    def test_unknown_tensors_and_nonfinite_weights_do_not_register(self):
        self.mocks();save_file({'a':torch.ones(1,2),'b':torch.ones(2,1),'extra':torch.ones(1)},str(self.path));self.info.siblings[0].size=self.path.stat().st_size
        with self.assertRaisesRegex(ValueError,'Unbound'):self.install()
        save_file({'a':torch.full((1,2),float('nan')),'b':torch.ones(2,1)},str(self.path));self.info.siblings[0].size=self.path.stat().st_size
        with self.assertRaisesRegex(ValueError,'Non-finite'):self.install()
        self.assertFalse(self.hub.index.exists());self.assertFalse(self.bound.rows)
    def test_search_and_unsafe_repository(self):
        api,download=self.mocks();api.return_value.list_models.return_value=[self.info]
        result=self.hub.search('portrait');self.assertEqual(result[0]['repo'],'owner/style');self.assertTrue(result[0]['compatible'])
        self.assertEqual(self.hub.search('owner/style')[0]['revision'],'a'*40)
        with self.assertRaises(ValueError):self.hub.search('https://evil.test/x')
        with self.assertRaises(ValueError):self.hub.install(repo='owner/style',revision='main',filename='../x.pkl')
        download.assert_not_called()

    def test_hub_model_card_object(self):
        api,_=self.mocks()
        from huggingface_hub import ModelCardData
        self.info.card_data=ModelCardData(base_model='owner/base',license='apache-2.0')
        self.assertTrue(self.hub.search('owner/style')[0]['compatible'])
