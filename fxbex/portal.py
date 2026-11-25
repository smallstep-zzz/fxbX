"""Local model catalog and one isolated, SDK-owned model worker."""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import threading
import time
from urllib.request import urlopen
import zipfile

from .hub import CATALOG, catalog_matches, resolve_package, _local_package
from .diagnostics import describe_error, worker_exit
from .lora import scene_prompt
from .service import PackageService, WorkerRequestError
from .settings import cache_root


def read_catalog():
    cache=cache_root()/'portal-catalog.json'
    try:
        with urlopen(os.environ.get('FXBEX_CATALOG_URL',CATALOG),timeout=20) as response:raw=response.read(4_000_001)
        if len(raw)>4_000_000:raise ValueError('Catalog too large')
        doc=json.loads(raw)
        try:
            place_url=os.environ.get('FXBEX_CATALOG_URL',CATALOG).rsplit('/',1)[0]+'/place-catalog.json'
            with urlopen(place_url,timeout=10) as response:place_raw=response.read(4_000_001)
            if len(place_raw)>4_000_000:raise ValueError('Place catalog too large')
            doc['_place']=json.loads(place_raw)
        except (OSError,ValueError):pass
        try:
            lora_url=os.environ.get('FXBEX_CATALOG_URL',CATALOG).rsplit('/',1)[0]+'/lora-catalog.json'
            with urlopen(lora_url,timeout=10) as response:lora_raw=response.read(4_000_001)
            if len(lora_raw)>4_000_000:raise ValueError('LoRA catalog too large')
            lora_doc=json.loads(lora_raw)
            if lora_doc.get('schema_version')==1 and isinstance(lora_doc.get('adapters'),list):doc['_loras']=lora_doc
        except (OSError,ValueError):pass
        models=catalog_rows(doc)
        raw=json.dumps(doc,ensure_ascii=False).encode()
    except (OSError,ValueError):
        if not cache.is_file() or cache.stat().st_size>4_000_000:raise
        return catalog_rows(json.loads(cache.read_text()))
    cache.parent.mkdir(parents=True,exist_ok=True)
    temp=cache.with_name('portal-catalog-'+str(time.time_ns())+'.tmp');temp.write_bytes(raw);temp.replace(cache)
    return models


def catalog_rows(doc):
    if not isinstance(doc,dict) or not isinstance(doc.get('models'),list):raise ValueError('Invalid model catalog')
    rows=[]
    place={r['source']:r for r in doc.get('_place',{}).get('models',[]) if isinstance(r,dict) and 'source' in r}
    categories={'Image Generation':'image','image':'image','Computer Vision':'vision','Vision':'vision','3D':'3d','3D Generation':'3d','Decision Making':'decision','Embedding':'embedding','Embeddings':'embedding'}
    sources=sorted({p['model']['source_repo'] for p in doc['models']})
    for source in sources:
        matches=catalog_matches(doc,source)
        if len(matches)!=1:raise ValueError('Ambiguous catalog release')
        item=matches[0];model=item['model'];release=item['release']
        web=place.get(source,{})
        category=web.get('category',categories.get(model.get('category'),'other'))
        same=web.get('version')==release['version']
        rows.append({'id':source+'@'+release['version'],'name':source.split('/')[-1],'source':source,
                     'category':category,'description':web.get('description',model.get('short_description','')) if same else model.get('short_description',''),
                     'precision':release.get('precision',''),'sdk':release.get('sdk_version',''),
                     'version':release['version'],'license':model.get('source_license',''),
                     'recommendations':[r for r in doc.get('_loras',{}).get('adapters',[]) if source in r.get('models',[])],
                     'published':web.get('published_on','') if same else release.get('published_at',model.get('published_at',''))})
    return rows


def recommended_prompts(info, recommendations):
    """Add reviewed prompt metadata without changing available LoRAs or scales."""
    candidates = {
        (row.get('repo'), row.get('revision'), row.get('filename')): row
        for row in recommendations
        if isinstance(row, dict) and all(
            isinstance(row.get(field), str) and row[field]
            for field in ('repo', 'revision', 'filename')
        )
    }
    adapters = []
    for row in info['adapters']:
        identity = tuple(row.get(field) for field in ('source_repo', 'source_revision', 'source_file'))
        reviewed = candidates.get(identity, {})
        prompt = scene_prompt(reviewed.get('example_prompt'), reviewed.get('trigger'), row.get('name'))
        adapters.append({**row, 'example_prompt': prompt} if prompt else row)
    return {**info, 'adapters': adapters}


class ModelManager:
    def __init__(self,models,*,initial=None,loras=None,service_factory=PackageService,resolve=resolve_package,load_options=None):
        self.models=list(models);self.factory=service_factory;self.resolve=resolve;self.load_options=load_options or {}
        # Linux parent-death signals bind to the spawning thread; keep it alive.
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='fxbx-model-manager')
        self.lock=threading.RLock();self.service=None;self.current=None;self.thread=None;self.switching=False;self.busy=0
        self.closing=False;self.phase='idle';self.error='';self.started=None;self.config=None;self.default_loras=loras or []
        self.pending = None
        self.failure = None
        self.initial=None
        if initial:
            path=Path(initial).expanduser()
            if path.is_dir():path=_local_package(path,announce=False) or path
            if path.is_file():
                identifier=str(path.resolve());title=path.stem;category='Local'
                try:
                    with zipfile.ZipFile(path) as archive:
                        info=archive.getinfo('manifest.json')
                        if info.file_size<=4_000_000:
                            m=json.loads(archive.read(info));name=m.get('model',{}).get('name',title)
                            if isinstance(name,str):
                                title=name.split('/')[-1]
                                matches=[r for r in self.models if name in (r['source'],r['name'])]
                                match=matches[0] if len(matches)==1 else None
                                if match:category=match['category']
                except (OSError,ValueError,KeyError,zipfile.BadZipFile):pass
                recommendations=match.get('recommendations',[]) if 'match' in locals() and match else []
                self.models.insert(0,{'id':identifier,'name':title+' · 로컬','source':identifier,'category':category,'description':'서버의 로컬 패키지','precision':'','sdk':'','version':'local','license':'','local':True,'recommendations':recommendations})
            else:
                name,_,version=str(initial).partition('@')
                matches=[r for r in self.models if name in (r['source'],r['name']) and (not version or r['version']==version)]
                if len(matches)!=1:raise ValueError('Choose an exact catalog ID or existing local package')
                identifier=matches[0]['id']
            self.initial=identifier

    def snapshot(self):
        with self.lock:
            if self.phase=='ready' and self.service:
                process=self.service.process
                if not self.service.loaded or (process is not None and process.poll() is not None):
                    detail = worker_exit(process.poll() if process is not None else None,
                                         getattr(self.service, 'log_path', None))
                    self.phase='error';self.error=detail['message']
                    self.failure={**detail, 'stage':'runtime', 'model':self.current}
            return {'phase':self.phase,'model':self.current,'target':self.pending,'busy':self.busy,'error':self.error,
                    'diagnostic':self.failure,
                    'elapsed_s':round(time.monotonic()-self.started,1) if self.started else 0,
                    'context_limit':getattr(self.service,'context_limit',512),'config':self.config if self.phase=='ready' else None,'label':getattr(self,'label','')}

    def start(self,identifier,loras=None):
        with self.lock:
            self.snapshot()
            if identifier not in {r['id'] for r in self.models}:raise ValueError('Unknown catalog model')
            if self.closing or self.switching or self.busy:raise ValueError('현재 작업이 끝난 뒤 모델을 전환하세요.')
            if self.phase=='ready' and self.current==identifier:return self.snapshot()
            self.switching=True;self.phase='switching';self.error='';self.started=time.monotonic()
            self.failure=None
            self.pending=identifier
            self.thread=self.executor.submit(self._switch,identifier,loras or [])
        return self.snapshot()

    def stop(self):
        with self.lock:
            if self.closing or self.switching or self.busy:raise ValueError('현재 작업이 끝난 뒤 서빙을 중지하세요.')
            self.switching=True;self.phase='stopping';self.thread=self.executor.submit(self._switch,None,[])
        return self.snapshot()

    def _switch(self,identifier,loras):
        old = self.service
        previous = self.current
        previous_config = self.config
        restore = identifier is not None and old is not None and old.loaded and previous != identifier
        service = None
        stage = 'download' if identifier is not None else 'unload'
        try:
            if identifier is not None:
                self.progress('resolving','모델 파일 확인·다운로드 중')
                path=self.resolve(identifier)
                if self.closing:raise RuntimeError('Model manager closed')
                options=self.load_options if identifier==self.initial else {}
                service=self.factory(path,**options)
            stage = 'unload'
            if old:
                if old.loaded:old.unload()
                else:old.stop()
            with self.lock:
                self.service=service
                self.current=identifier
                self.config=None
            if identifier is None:
                with self.lock:self.phase='idle';self.started=None;self.error='';self.failure=None
                return
            stage = 'load'
            self._load(service, identifier, loras)
            print('NPU package ready: '+identifier,flush=True)
        except Exception as exc:
            detail = {**describe_error(exc), 'stage':stage, 'model':identifier}
            if stage == 'load' and getattr(service, 'log_path', None):
                detail.setdefault('log', str(service.log_path))
            if service is not None:
                service.stop()
            if restore and not self.closing:
                try:
                    if not old.loaded:
                        with self.lock:self.service=old;self.current=previous
                        self._load(old, previous, (previous_config or {}).get('default_loras', []))
                    with self.lock:
                        self.phase='ready';self.error=f'{identifier} 전환 실패 · {detail["message"]}'
                        self.failure=detail
                    return
                except Exception as recovery_error:
                    old.stop()
                    detail['message'] += ' · 이전 모델 복구 실패: ' + describe_error(recovery_error)['message']
            with self.lock:
                self.current=identifier;self.phase='error';self.error=detail['message'];self.failure=detail
        finally:
            with self.lock:self.switching=False;self.pending=None

    def _load(self, service, identifier, loras):
        with self.lock:
            if self.closing:raise RuntimeError('Model manager closed')
        loaded=service.load(self.progress,loras=loras)
        config=loaded.get('web_config') or service.web('config')
        config=self._with_recommended_prompts(config, identifier)
        with self.lock:
            if self.closing:raise RuntimeError('Model manager closed')
            self.config=config;self.phase='ready';self.error='';self.failure=None

    def progress(self,phase,label):
        with self.lock:self.phase='loading';self.error='';self.label=str(label)[:300]

    def _recommendations(self, identifier):
        return next(row.get('recommendations', []) for row in self.models if row['id'] == identifier)

    def _with_recommended_prompts(self, config, identifier):
        if 'lora' not in config:
            return config
        return {**config, 'lora': recommended_prompts(config['lora'], self._recommendations(identifier))}

    def call(self,method,*args,expected_key=None):
        with self.lock:
            if self.phase!='ready' or self.switching or self.service is None:raise ValueError('모델을 먼저 서빙하세요.')
            if expected_key and expected_key!=hashlib.sha256(self.current.encode()).hexdigest():raise ValueError('서빙 모델이 바뀌었습니다. 호출 샘플을 다시 생성하세요.')
            service=self.service;identifier=self.current;self.busy+=1
        try:
            value=getattr(service,method)(*args)
            if method=='web':
                operation = args[0]
                if operation == 'config':
                    value = {**self._with_recommended_prompts(value, identifier),
                             'model_key':hashlib.sha256(identifier.encode()).hexdigest()}
                elif operation in ('loras', 'import'):
                    info = value if operation == 'loras' else value['lora']
                    info = recommended_prompts(info, self._recommendations(identifier))
                    with self.lock:
                        self.config['lora'] = info
                    value = info if operation == 'loras' else {**value, 'lora': info}
            return value
        except WorkerRequestError as exc:
            if exc.fatal:
                detail={**describe_error(exc), 'stage':'inference', 'model':identifier}
                if getattr(service, 'log_path', None):
                    detail.setdefault('log', str(service.log_path))
                with self.lock:
                    self.phase='error';self.error=detail['message'];self.failure=detail
            raise
        finally:
            with self.lock:self.busy-=1

    def commands(self,identifier,loras,inputs,port):
        if identifier not in {r['id'] for r in self.models}:raise ValueError('Unknown model')
        from .lora import selection
        ready=self.phase=='ready' and identifier==self.current
        if loras and not ready:raise ValueError('LoRA 선택은 실행 중인 모델에만 적용할 수 있습니다.')
        info=(self.config or {}).get('lora',{'supported':False,'adapters':[],'max_active':0})
        chosen=selection(loras,info) if ready else []
        argv=['fxbx','serve',identifier,'--port',str(port)]
        if identifier==self.initial and self.load_options.get('weights_dir'):argv+=['--weights-dir',self.load_options['weights_dir']]
        for row in chosen:argv+=['--lora',row['id']+'='+str(row['scale'])]
        payload=inputs if isinstance(inputs,dict) else ((self.config or {}).get('example') or {})
        if ready and self.config.get('method')=='predict' and info.get('supported'):payload={**payload,'loras':chosen}
        base='http://127.0.0.1:'+str(port)
        headers={'Content-Type':'application/json','X-FXBX-Demo':'1','X-FXBX-Model-Key':hashlib.sha256(identifier.encode()).hexdigest()}
        sample=('import json\nfrom urllib.request import Request, urlopen, urlretrieve\n\n'
                f'base = {base!r}\npayload = {payload!r}\n'
                'request = Request(base + "/demo/run", data=json.dumps(payload).encode(),\n'
                f'                  headers={headers!r})\n'
                'with urlopen(request, timeout=600) as response:\n    result = json.load(response)\n'
                'print(json.dumps(result["result"], ensure_ascii=False, indent=2))\n'
                'if result.get("image_url"):\n    urlretrieve(base + result["image_url"], "result.png")\n    print("result.png")\n')
        return {'serve':shlex.join(argv),'sample':sample if ready else None}

    def close(self):
        with self.lock:self.closing=True;service=self.service;cancel=self.switching or self.busy>0
        if service:
            if cancel:service.cancel();service.stop()
            elif service.loaded:service.unload()
            else:service.stop()
        self.executor.shutdown(wait=True,cancel_futures=True)


def handler(manager,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_bytes(self,code,data,mime,*,embedded=False):
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; frame-src 'self'; connect-src 'self'; frame-ancestors "+("'self'" if embedded else "'none'")+"; base-uri 'none'")
            self.end_headers();self.wfile.write(data)
        def send(self,code,value):self.send_bytes(code,json.dumps(value,ensure_ascii=False).encode(),'application/json')
        def do_GET(self):
            try:
                if self.path=='/':self.send_bytes(200,Path(__file__).with_name('portal.html').read_bytes(),'text/html; charset=utf-8')
                elif self.path=='/playground':self.send_bytes(200,Path(__file__).with_name('web_demo.html').read_bytes(),'text/html; charset=utf-8',embedded=True)
                elif self.path=='/diagnostics.js':self.send_bytes(200,Path(__file__).with_name('diagnostics.js').read_bytes(),'text/javascript; charset=utf-8')
                elif self.path=='/manager/catalog':self.send(200,{'models':manager.models,'initial':manager.initial,'port':self.server.server_port})
                elif self.path=='/manager/status':self.send(200,manager.snapshot())
                elif self.path in ('/demo/config','/demo/loras'):self.send(200,manager.call('web','config' if self.path.endswith('config') else 'loras'))
                elif self.path.startswith('/demo/image/'):
                    value=manager.call('web','image',self.path.removeprefix('/demo/image/'))
                    data=base64.b64decode(value['base64'],validate=True)
                    if len(data)>32*1024**2 or value['mime'] not in ('image/png','image/jpeg','image/webp'):raise ValueError('Invalid image')
                    self.send_bytes(200,data,value['mime'])
                elif self.path in ('/status','/v1/models'):
                    snapshot=manager.snapshot();model=(snapshot['config'] or {}).get('model',snapshot['model'])
                    self.send(200,{'object':'list','data':[{'id':model,'object':'model'}] if model else [],'loaded':snapshot['phase']=='ready','context_limit':snapshot['context_limit']})
                else:self.send(404,{'error':{'message':'Unknown endpoint'}})
            except Exception as exc:self.send(400,{'error':{**describe_error(exc), 'fatal':getattr(exc,'fatal',False)}})
        def do_POST(self):
            try:
                host=self.headers.get('Host','')
                browser=self.path.startswith(('/manager/','/demo/'))
                if browser and (host.partition(':')[0] not in ('127.0.0.1','localhost') or self.headers.get('Origin','http://'+host)!='http://'+host or self.headers.get('X-FXBX-Demo')!='1'):
                    self.send(403,{'error':{'message':'Local browser requests only'}});return
                size=int(self.headers.get('Content-Length','0'));limit=max(65536,getattr(manager.service,'context_limit',512)*32)
                if not 0<size<=limit:raise ValueError('Request exceeds transport limit')
                data=json.loads(self.rfile.read(size))
                if self.path=='/manager/serve':self.send(202,manager.start(data['model']))
                elif self.path=='/manager/stop':self.send(202,manager.stop())
                elif self.path=='/manager/commands':self.send(200,manager.commands(data['model'],data.get('loras',[]),data.get('inputs'),self.server.server_port))
                elif self.path=='/demo/run':self.send(200,manager.call('web','run',data,expected_key=self.headers.get('X-FXBX-Model-Key')))
                elif self.path in ('/demo/hub/search','/demo/hub/import'):
                    self.send(200,manager.call('web',self.path.rsplit('/',1)[1],data,expected_key=self.headers.get('X-FXBX-Model-Key')))
                elif self.path=='/predict':
                    result=manager.call('predict',data,expected_key=self.headers.get('X-FXBX-Model-Key'));result.pop('ok',None);self.send(200,result)
                elif self.path=='/v1/chat/completions':
                    supported={'model','messages','max_tokens','max_completion_tokens','stream','temperature','n'}
                    if set(data)-supported or data.get('stream') or data.get('temperature',0)!=0 or data.get('n',1)!=1:raise ValueError('Unsupported chat options')
                    if 'max_tokens' in data and 'max_completion_tokens' in data:raise ValueError('Specify only one token limit')
                    config=manager.config or {}
                    if data.get('model',config.get('model'))!=config.get('model'):raise ValueError('Requested model is not loaded')
                    result=manager.call('chat',data['messages'],data.get('max_completion_tokens',data.get('max_tokens',32)))
                    self.send(200,{'id':f'chatcmpl-{time.time_ns()}','object':'chat.completion','created':int(time.time()),'model':config.get('model'),'choices':[{'index':0,'message':{'role':'assistant','content':result['text']},'finish_reason':'stop'}],'fxbex':result})
                else:self.send(404,{'error':{'message':'Unknown endpoint'}})
            except Exception as exc:self.send(400,{'error':{**describe_error(exc), 'fatal':getattr(exc,'fatal',False)}})
    return Handler


def launch(port=8000,*,initial=None,loras=None,load_options=None):
    try:models=read_catalog()
    except (OSError,ValueError):
        if not initial:raise
        models=[]
    manager=ModelManager(models,initial=initial,loras=loras,load_options=load_options)
    with ThreadingHTTPServer(('127.0.0.1',port),handler(manager,port)) as server:
        print(f'fxbX 모델 서빙: http://127.0.0.1:{server.server_port}/ (Ctrl-C로 종료)',flush=True)
        if manager.initial:manager.start(manager.initial,manager.default_loras)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:manager.close()
