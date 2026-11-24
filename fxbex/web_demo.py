"""Browser interface for an already loaded, serialized local model session."""
import mimetypes
from pathlib import Path
from secrets import token_urlsafe

from .schema import resolve_sample


class WebDemo:
    def __init__(self, session):
        self.session = session
        self.images = {}
        contract = session.manifest.get('io_contract') or {}
        method = contract.get('method')
        if method not in ('chat', 'predict'):
            method = 'chat' if callable(getattr(session.adapter, 'chat', None)) else 'predict'
        sample = contract.get('example_input')
        if not isinstance(sample, dict):
            sample = {'messages': [{'role': 'user', 'content': ''}]} if method == 'chat' else {}
        if method=='predict' and (contract.get('inputs',{}).get('output') or {}).get('type')=='path':
            sample={k:v for k,v in sample.items() if k!='output'}
        from .lora import catalog
        self.config = {
            'model': session.manifest['model']['name'],
            'lora': catalog(session.adapter),
            'lora_hub': self.hub() is not None,
            'method': method,
            'inputs': contract.get('inputs') or {},
            'outputs': contract.get('outputs') or {},
            'example': sample,
            'expected_result': contract.get('expected_result'),
        }

    def hub(self):
        from inspect import getattr_static
        from .lora_hub import HfLoraLibrary
        adapter=self.session.adapter
        value=getattr(adapter,'lora_hub',None) if getattr_static(adapter,'lora_hub',None) is not None else None
        return value if isinstance(value,HfLoraLibrary) else None

    def hub_request(self, operation, request):
        hub=self.hub()
        if hub is None:raise ValueError('이 패키지는 Hugging Face LoRA 가져오기를 지원하지 않습니다.')
        if not isinstance(request,dict):raise ValueError('Expected a JSON object')
        if operation=='search':return {'results':hub.search(request.get('query',''))}
        if set(request)!={'repo','revision','filename'}:raise ValueError('Choose repository, revision and weight file')
        with self.session._predict_lock:
            key=hub.install(**request)
        return {'id':key,'lora':self.loras()}

    def loras(self):
        from .lora import catalog
        return catalog(self.session.adapter)

    def run(self, request, invoke):
        if not isinstance(request, dict):
            raise ValueError('입력은 JSON 객체여야 합니다.')
        inputs = resolve_sample({'example_input': request}, self.session.root, self.session.manifest)
        if self.config['method'] == 'chat':
            return {'result': invoke(lambda: self.session.chat(
                inputs['messages'], inputs.get('max_new_tokens', 32)))}
        contract=self.session.manifest.get('io_contract') or {}
        output=(contract.get('inputs') or {}).get('output') or {}
        if output.get('type')=='path':
            root=Path(self.session.report_dir).resolve()/'demo'
            root.mkdir(parents=True,exist_ok=True)
            suffix=Path(str((contract.get('example_input') or {}).get('output',''))).suffix
            inputs['output']=str(root/(token_urlsafe(18)+suffix))
        result = invoke(lambda: self.session.predict(inputs))
        response = {'result': result}
        if isinstance(result, dict) and isinstance(result.get('image_path'), str):
            root = Path(self.session.report_dir).resolve()
            path = Path(result['image_path'])
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            mime = mimetypes.guess_type(path.name)[0]
            if (path.is_relative_to(root) and path.is_file()
                    and mime in ('image/png', 'image/jpeg', 'image/webp')
                    and path.stat().st_size <= 32 * 1024 * 1024):
                key = token_urlsafe(18)
                if len(self.images) >= 16:
                    self.images.pop(next(iter(self.images)))
                self.images[key] = (path, mime)
                response['image_url'] = '/demo/image/' + key
        return response

    def image(self, key):
        path, mime = self.images[key]
        root = Path(self.session.report_dir).resolve()
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError('Invalid image path')
        with path.open('rb') as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError('Image exceeds the display size limit')
        return data, mime
