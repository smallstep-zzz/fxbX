"""Persistent stdio package worker using the existing bounded service protocol."""
import contextlib
import ctypes
import json
import os
import signal
import sys
import traceback
from .runtime import Session


def main():
    parent=os.getppid()
    if sys.platform.startswith('linux'):
        if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError('parent-death signal failed')
        if os.getppid()!=parent or parent==1:raise RuntimeError('Parent exited')
    protocol=sys.stdout;session=None;demo=None
    try:
        for line in sys.stdin:
            request={}
            try:
                request=json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    if request['action']=='load':
                        if session:raise RuntimeError('Package already loaded')
                        print('@@FXBEX_PROGRESS@@'+json.dumps({'phase':'loading','label':'모델 검사·가중치 적재·NPU 실행 준비 중'}),file=protocol,flush=True)
                        options=request.get('options',{})
                        if not isinstance(options,dict) or set(options)-{'weights_dir','report_dir','allow_sdk_mismatch'}:raise ValueError('Invalid load options')
                        from .native_fxb import read_native_manifest
                        from .native_session import NativeSession
                        session_type=NativeSession if read_native_manifest(sys.argv[1]) is not None else Session
                        candidate=session_type(sys.argv[1],**options)
                        candidate.__enter__()
                        try:
                            from .lora import selection
                            from .web_demo import WebDemo
                            candidate.default_loras=selection(request.get('loras',[]),candidate.lora_catalog())
                            demo=WebDemo(candidate)
                        except BaseException:
                            candidate.__exit__(*sys.exc_info());raise
                        session=candidate
                        result={'ok':True,'metrics':getattr(session.adapter,'acceptance',{}),'report_dir':str(session.report_dir),'context_limit':getattr(getattr(session.adapter,'runtime',None),'capacity',512),'memory':session.memory,'web_config':demo.config}
                    elif request['action']=='web':
                        if not session:raise RuntimeError('Load package first')
                        operation=request.get('operation');payload=request.get('payload')
                        if operation=='config':value=demo.config
                        elif operation=='loras':value=demo.loras()
                        elif operation=='run':value=demo.run(payload,lambda callback:callback())
                        elif operation in ('search','import'):value=demo.hub_request(operation,payload)
                        elif operation=='image':
                            import base64
                            data,mime=demo.image(payload);value={'mime':mime,'base64':base64.b64encode(data).decode('ascii')}
                        else:raise ValueError('Unknown browser operation')
                        result={'ok':True,'value':value}
                    elif request['action']=='chat':
                        if not session:raise RuntimeError('Load package first')
                        result={'ok':True,**session.chat(request['messages'],request['max_new_tokens'])}
                        result['fxbex_execution']=session.execution.last
                        result['memory']=session.memory
                    elif request['action']=='predict':
                        if not session:raise RuntimeError('Load package first')
                        result={'ok':True,**session.predict(request['inputs'])}
                        result['fxbex_execution']=session.execution.last
                        result['memory']=session.memory
                    elif request['action']=='generate':
                        if not session:raise RuntimeError('Load package first')
                        def token(text,token_ids):
                            print('@@FXBEX_TOKEN@@'+json.dumps({'text':text,'token_ids':token_ids},ensure_ascii=False),file=protocol,flush=True)
                        result={'ok':True,**session.generate(request['prompt'],request['max_tokens'],on_token=token)}
                        result['fxbex_execution']=session.execution.last
                        result['memory']=session.memory
                    elif request['action']=='unload':
                        memory=None
                        if session:
                            session.__exit__(None,None,None);memory=session.memory;session=None
                        result={'ok':True,'memory':memory}
                    else:raise ValueError('Unknown worker action')
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                message = str(exc)
                from huggingface_hub.utils import GatedRepoError
                if isinstance(exc, GatedRepoError):
                    from urllib.parse import urlsplit
                    url = getattr(getattr(exc, 'response', None), 'url', '')
                    parts = urlsplit(str(url)).path.strip('/').split('/')
                    repo = '/'.join(parts[:2]) if len(parts) > 2 and parts[2] == 'resolve' else 'the original model'
                    message = (f'Hugging Face access required for {repo}. Accept the original model terms, '
                               'then run hf auth login as the account running fxbx on the server and retry.')
                result={'ok':False,'error':message,'fatal':session is None or session.failed}
            print('@@FXBEX_RESPONSE@@'+json.dumps(result,ensure_ascii=False),file=protocol,flush=True)
            if request.get('action')=='unload':break
    finally:
        if session:session.__exit__(None,None,None)

if __name__=='__main__':main()
