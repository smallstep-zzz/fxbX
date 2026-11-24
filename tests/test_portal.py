import io,json,shlex,threading,time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from fxbex.portal import ModelManager,handler,catalog_rows
from fxbex.service import PackageService,WorkerRequestError
from fxbex.lora import cli_selection

INFO={'supported':True,'adapters':[{'id':'style','default_scale':.8,'min_scale':0,'max_scale':1}], 'max_active':1}
CONFIG={'model':'test','method':'predict','example':{'prompt':'fox'},'lora':INFO,'default_loras':[]}
class FakeService:
    instances=[]
    def __init__(self,path,**options):self.path=path;self.options=options;self.loaded=False;self.process=None;self.context_limit=512;self.stopped=False;self.__class__.instances.append(self)
    def load(self,progress,*,loras=None):self.loaded=True;progress('loading','loading');return {'web_config':CONFIG}
    def stop(self):self.loaded=False;self.stopped=True
    def unload(self):self.stop()
    def cancel(self):self.stop()
    def web(self,operation,payload=None):return CONFIG if operation=='config' else {'result':payload}

class PortalTests(TestCase):
    def make(self):
        manager=ModelManager([{'id':'owner/a@v1','name':'a'},{'id':'owner/b@v1','name':'b'}],service_factory=FakeService,resolve=lambda value:'/tmp/'+value.split('/')[-1]+'.fxbx')
        self.addCleanup(manager.close);return manager
    def ready(self,manager,model='owner/a@v1'):
        manager.start(model);manager.thread.result(timeout=3);self.assertEqual(manager.snapshot()['phase'],'ready')
    def test_switch_unloads_only_owned_worker(self):
        manager=self.make();self.ready(manager);old=manager.service;self.ready(manager,'owner/b@v1')
        self.assertTrue(old.stopped);self.assertEqual(manager.current,'owner/b@v1')
        with self.assertRaises(ValueError):manager.start('/arbitrary/file')
    def test_active_request_prevents_switch_or_stop(self):
        manager=self.make();self.ready(manager);started=threading.Event();finish=threading.Event()
        def web(*args):started.set();finish.wait(3);return {}
        manager.service.web=web
        t=threading.Thread(target=lambda:manager.call('web','run',{}));t.start();self.assertTrue(started.wait(2))
        try:
            with self.assertRaises(ValueError):manager.start('owner/b@v1')
            with self.assertRaises(ValueError):manager.stop()
        finally:finish.set();t.join(3)
    def test_commands_restore_selection_and_use_resident_api(self):
        manager=self.make();self.ready(manager)
        result=manager.commands(manager.current,[{'id':'style','scale':.6}],{'prompt':"quote ' and 한글"},8000)
        self.assertEqual(shlex.split(result['serve']),['fxbx','serve','owner/a@v1','--port','8000','--lora','style=0.6'])
        compile(result['sample'],'<sample>','exec');self.assertIn('/demo/run',result['sample']);self.assertIn('0.6',result['sample'])
        with self.assertRaises(ValueError):manager.commands(manager.current,[{'id':'bad'}],{},8000)
        with self.assertRaises(ValueError):manager.commands('owner/b@v1',[{'id':'style'}],{},8000)
    def test_failed_start_can_retry(self):
        manager=self.make();manager.resolve=lambda value:(_ for _ in ()).throw(ValueError('bad package'))
        manager.start('owner/a@v1');manager.thread.result(timeout=3);self.assertEqual(manager.snapshot()['phase'],'error')
        manager.resolve=lambda value:'/tmp/model.fxbx';self.ready(manager)
    def test_same_origin_guard_and_portal_routes(self):
        manager=self.make();server=ThreadingHTTPServer(('127.0.0.1',0),handler(manager,0));t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        self.addCleanup(t.join,3);self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        with urlopen(base+'/') as response:self.assertIn(b'manager/serve',response.read())
        body=json.dumps({'model':'owner/a@v1'}).encode()
        with self.assertRaises(HTTPError) as denied:urlopen(Request(base+'/manager/serve',body,{'Origin':'https://evil.test','X-FXBX-Demo':'1'}))
        self.assertEqual(denied.exception.code,403)
        with urlopen(Request(base+'/manager/serve',body,{'X-FXBX-Demo':'1'})) as response:self.assertEqual(response.status,202)
        manager.thread.result(timeout=3)
        with urlopen(base+'/demo/config') as response:self.assertEqual(json.load(response)['model'],'test')
        with urlopen(base+'/playground') as response:self.assertIn("frame-ancestors 'self'",response.headers['Content-Security-Policy'])
    def test_latest_release_selection(self):
        def row(version):return {'model':{'name':'a','source_repo':'owner/a','category':'Image Generation'},'release':{'version':version,'author_sha256':'hash'}}
        doc={'models':[row('old'),row('new')],'release_order':['owner/a@old#hash','owner/a@new#hash']}
        self.assertEqual(catalog_rows(doc)[0]['id'],'owner/a@new')
    def test_cli_gallery_and_default_lora(self):
        from fxbex.cli import main
        with patch('fxbex.portal.launch') as launch:
            self.assertEqual(main(['serve','--port','8100']),0);launch.assert_called_once_with(8100,initial=None,loras=[],load_options={})
        self.assertEqual(cli_selection(['style=.4']),[{'id':'style','scale':.4}])
        for values in (['style=nan'],['a','a']):
            with self.assertRaises(ValueError):cli_selection(values)

class SoftWorkerErrors(TestCase):
    def test_invalid_hub_request_does_not_terminate_loaded_model(self):
        service=PackageService('/tmp/model.fxbx');service.loaded=True
        service.process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO('@@FXBEX_RESPONSE@@{"ok":false,"fatal":false,"error":"invalid repository"}\n'))
        service.log=io.StringIO()
        with patch.object(service,'stop') as stop:
            with self.assertRaises(WorkerRequestError):service.web('search',{'query':'bad'})
            stop.assert_not_called()
        service.process=None;service.log=None

class LinuxWorkerLifetime(TestCase):
    def test_spawning_thread_survives_load_job(self):
        import subprocess,sys
        if not sys.platform.startswith('linux'):self.skipTest('Linux parent-death signal')
        class Service(FakeService):
            def load(self,progress,*,loras=None):
                self.process=subprocess.Popen([sys.executable,'-u','-c',"import ctypes,signal,time;ctypes.CDLL(None).prctl(1,signal.SIGTERM);print('ready',flush=True);time.sleep(20)"],stdout=subprocess.PIPE,text=True)
                assert self.process.stdout.readline().strip()=='ready';self.loaded=True;return {'web_config':CONFIG}
            def stop(self):
                if self.process:
                    if self.process.poll() is None:self.process.terminate()
                    self.process.wait(timeout=3);self.process.stdout.close()
                self.loaded=False
        manager=ModelManager([{'id':'owner/a@v1','name':'a'}],service_factory=Service,resolve=lambda p:'/tmp/model.fxbx')
        self.addCleanup(manager.close);manager.start('owner/a@v1');manager.thread.result(timeout=3)
        time.sleep(.1)
        self.assertIsNone(manager.service.process.poll())
        self.assertEqual(manager.snapshot()['phase'],'ready')

class RequestModelBinding(TestCase):
    def test_stale_sample_is_rejected_before_entering_worker(self):
        manager=ModelManager([{'id':'owner/a@v1','name':'a'}],service_factory=FakeService,resolve=lambda p:'/tmp/model.fxbx')
        self.addCleanup(manager.close);manager.start('owner/a@v1');manager.thread.result(timeout=3)
        with patch.object(manager.service,'web') as web:
            with self.assertRaises(ValueError):manager.call('web','run',{},expected_key='wrong-model')
            web.assert_not_called()
        self.assertEqual(manager.snapshot()['phase'],'ready')
