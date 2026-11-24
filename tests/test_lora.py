import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from fxbex.lora import catalog,selection
from fxbex.runtime import Session

class LoraTest(TestCase):
    def adapter(self):
        return SimpleNamespace(lora_catalog=lambda:[{'id':'style','name':'Style'}],apply_loras=lambda chosen:None,max_active_loras=2)
    def test_validation(self):
        info=catalog(self.adapter())
        self.assertEqual(selection([{'id':'style','scale':.4}],info),[{'id':'style','scale':.4}])
        for value in ([{'id':'unknown'}],[{'id':'style','scale':float('nan')}],[{'id':'style','scale':True}],[{'id':'style','scale':1.1}],[{'id':'style'},{'id':'style'}],None):
            with self.assertRaises(ValueError):selection(value,info)
        self.assertEqual(selection([{'id':'style','scale':0}],info),[])
        with self.assertRaises(ValueError):selection([{'id':'style'}],catalog(SimpleNamespace()))
    def session(self):
        root=TemporaryDirectory();self.addCleanup(root.cleanup)
        s=Session('unused',report_dir=Path(root.name));s.adapter=self.adapter();s.adapter.active=[]
        def apply(chosen):s.adapter.active=chosen;time.sleep(.005)
        s.adapter.apply_loras=apply
        s.adapter.predict=lambda payload:{'actual':list(s.adapter.active),'payload':payload}
        s._call=lambda operation,fn:fn()
        return s
    def test_request_scope_and_base_restore(self):
        s=self.session();a=s.predict({'prompt':'one','loras':[{'id':'style','scale':.5}]});b=s.predict({'prompt':'two'})
        self.assertEqual(a['actual'],a['applied_loras']);self.assertNotIn('loras',a['payload']);self.assertEqual(b['actual'],[])
        self.assertEqual(b['applied_loras'],[])
    def test_concurrent_requests_do_not_mix_selection(self):
        s=self.session();results=[]
        def run(scale):results.append(s.predict({'loras':[{'id':'style','scale':scale}]}))
        threads=[threading.Thread(target=run,args=(scale,)) for scale in (.2,.7)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(len(results),2)
        for result in results:self.assertEqual(result['actual'],result['applied_loras'])
    def test_invalid_selection_does_not_poison_or_run_model(self):
        s=self.session();calls=[];s.adapter.predict=lambda payload:calls.append(payload)
        with self.assertRaises(ValueError):s.predict({'loras':[{'id':'bad'}]})
        self.assertFalse(s.failed);self.assertFalse(calls)

    def test_prompt_metadata_and_external_source_validation(self):
        adapter=self.adapter()
        adapter.lora_catalog=lambda:[{'id':'style','name':'Style','trigger_words':['toy_face'],
            'example_prompt':'toy_face, a portrait','source_url':'https://huggingface.co/example/style'}]
        info=catalog(adapter)
        self.assertEqual(info['adapters'][0]['trigger_words'],['toy_face'])
        self.assertEqual(info['adapters'][0]['example_prompt'],'toy_face, a portrait')
        for extra in ({'trigger_words':'not-a-list'},{'example_prompt':3},{'source_url':'javascript:alert(1)'},{'source_url':'https://huggingface.co.evil.test/owner/model'}):
            adapter.lora_catalog=lambda value=extra:[{'id':'style','name':'Style',**value}]
            with self.assertRaises(ValueError):catalog(adapter)

    def test_explicit_empty_overrides_command_line_default(self):
        s=self.session();s.default_loras=[{'id':'style','scale':.5}]
        self.assertEqual(s.predict({'prompt':'one'})['applied_loras'],s.default_loras)
        self.assertEqual(s.predict({'prompt':'two','loras':[]})['applied_loras'],[])

    def test_host_preflight_rejects_before_lora_mutation_without_poisoning(self):
        s=self.session();applied=[];s.adapter.apply_loras=lambda chosen:applied.append(chosen)
        def reject(payload):raise ValueError('Prompt exceeds context')
        s.adapter.validate_predict_inputs=reject
        with self.assertRaisesRegex(ValueError,'context'):s.predict({'prompt':'too long','loras':[{'id':'style'}]})
        self.assertFalse(s.failed);self.assertFalse(applied)
        del s.adapter.validate_predict_inputs
        self.assertEqual(s.predict({'prompt':'valid'})['applied_loras'],[])
