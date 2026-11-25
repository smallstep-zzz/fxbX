"""Data-only Hugging Face LoRA discovery; conversion is owned by the package.

No model code, pickle, compiler, or automatic application is invoked here.
"""
import fcntl
from functools import wraps
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from .lora import scene_prompt

_REPO = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*')


def _library_lock(method):
    @wraps(method)
    def guarded(self,*args,**kwargs):
        with (self.root/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self._reload()
            return method(self,*args,**kwargs)
    return guarded


class HfLoraLibrary:
    def __init__(self, bound, *, base_models, convert, cache_dir=None):
        self.bound, self.base_models, self.convert = bound, list(base_models), convert
        if not self.base_models or any(not _REPO.fullmatch(v) for v in self.base_models):
            raise ValueError('Declare compatible Hugging Face base model IDs')
        # Scope imported bindings to the exact saved execution contracts.
        identity = json.dumps({k:v.plan for k,v in bound.components.items()}, sort_keys=True)
        scope = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.root = Path(cache_dir or Path.home()/'.cache/fxbx/loras')/scope
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root/'library.json'
        self.rows=[]
        with (self.root/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            self._reload()

    def _reload(self):
        rows=json.loads(self.index.read_text()) if self.index.exists() else []
        if not isinstance(rows,list) or len(rows)>32:raise ValueError('Invalid imported LoRA library')
        known={r['id']:r for r in self.rows}
        for row in rows:
            if row['id'] not in known:self._attach(row)
            elif row!=known[row['id']]:raise ValueError('Cached LoRA registry changed unexpectedly')
        self.rows=rows

    def _attach(self, row):
        name=row['file']
        if not re.fullmatch(r'hf-[0-9a-f]{24}\.safetensors',name):raise ValueError('Invalid cached LoRA path')
        path=self.root/name
        if not 0<path.stat().st_size<=self.bound.limit:raise ValueError('Cached LoRA exceeds memory budget')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Cached LoRA checksum mismatch')
        self.bound.validate(row,path)
        self.bound.rows.append(row);self.bound.files[row['id']]=path

    @staticmethod
    def _repo(value):
        if not isinstance(value,str) or not _REPO.fullmatch(value):raise ValueError('Use a Hugging Face owner/model ID')
        return value

    def _describe(self, info):
        card=info.card_data.to_dict() if hasattr(info.card_data,'to_dict') else dict(info.card_data or {})
        bases=card.get('base_model',[])
        if isinstance(bases,str):bases=[bases]
        files=[{'name':s.rfilename,'size':getattr(s,'size',None)} for s in info.siblings or []
               if s.rfilename.endswith('.safetensors')]
        words=card.get('trigger_words',[])
        if isinstance(words,str):words=[words]
        words=[w for w in words if isinstance(w,str) and 0<len(w)<=128][:16] if isinstance(words,list) else []
        widgets = card.get('widget', [])
        prompt = ''
        if isinstance(widgets, list):
            for widget in widgets:
                if isinstance(widget, dict):
                    prompt = scene_prompt(widget.get('text'), info.id, info.id.rsplit('/', 1)[-1], *words)
                    if prompt:
                        break
        compatible=isinstance(bases,list) and bool(set(self.base_models)&set(v for v in bases if isinstance(v,str)))
        return {'repo':info.id,'revision':info.sha,'files':files,'license':str(card.get('license','unknown'))[:200],
                'base_models':bases,'compatible':compatible,'trigger_words':words,'example_prompt':prompt,
                'reason':'가중치 검사 전 · 품질 미검증' if compatible else '패키지가 허용한 베이스 모델과 일치하지 않습니다.'}

    def search(self, query=''):
        from huggingface_hub import HfApi
        if not isinstance(query,str) or len(query)>200:raise ValueError('Search query must be at most 200 characters')
        query=query.strip();api=HfApi()
        if '/' in query:
            return [self._describe(api.model_info(self._repo(query),files_metadata=True))]
        found={}
        for base in self.base_models:
            for info in api.list_models(search=query or None,filter=('lora','base_model:'+base),
                                        sort='downloads',limit=12,full=True,cardData=True):
                found[info.id]=self._describe(info)
        return list(found.values())[:24]

    @_library_lock
    def install(self, *, repo, revision, filename):
        from huggingface_hub import HfApi, hf_hub_download
        from safetensors import safe_open
        from safetensors.torch import save_file
        self._repo(repo)
        if not isinstance(revision,str) or not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('Use an immutable Hub revision')
        if not isinstance(filename,str):raise ValueError('Choose a safetensors file')
        part=PurePosixPath(filename)
        if part.is_absolute() or '..' in part.parts or part.suffix!='.safetensors':raise ValueError('Only safetensors data files are accepted')
        key='hf-'+hashlib.sha256((repo+'@'+revision+'/'+filename).encode()).hexdigest()[:24]
        if any(r['id']==key for r in self.rows):return key
        if len(self.rows)>=32 or len(self.bound.rows)>=64:raise ValueError('LoRA library is full')
        info=HfApi().model_info(repo,revision=revision,files_metadata=True)
        desc=self._describe(info)
        if info.sha!=revision or not desc['compatible']:raise ValueError('LoRA base model/revision does not match the package')
        file=next((f for f in desc['files'] if f['name']==filename),None)
        if file is None or type(file['size']) is not int or not 0<file['size']<=self.bound.limit:
            raise ValueError('LoRA file is missing, has unknown size, or exceeds the memory budget')
        if sum(p.stat().st_size for p in self.root.glob('*.safetensors'))+file['size']>2*1024**3:
            raise ValueError('Imported LoRA disk budget (2 GiB) exceeded')
        path=Path(hf_hub_download(repo,filename,revision=revision))
        if path.stat().st_size!=file['size']:raise ValueError('Downloaded LoRA size mismatch')
        # Check the header before the trusted package converter materializes tensors.
        with safe_open(str(path),framework='pt',device='cpu') as reader:
            if not reader.keys():raise ValueError('Empty LoRA')
            for name in reader.keys():
                if reader.get_slice(name).get_dtype() not in ('F16','BF16','F32'):raise ValueError('Unsupported LoRA tensor dtype')
        state,components=self.convert(path)
        if not isinstance(state,dict) or not state:raise ValueError('Converter returned no LoRA tensors')
        import torch
        if sum(v.numel()*v.element_size() for v in state.values())>self.bound.limit:raise ValueError('Converted LoRA exceeds memory budget')
        if any(not torch.isfinite(v).all().item() for v in state.values()):raise ValueError('Non-finite LoRA weights')
        row={'id':key,'name':repo,'file':key+'.safetensors','components':components,
             'source_repo':repo,'source_revision':revision,'source_file':filename,'source_url':'https://huggingface.co/'+repo,
             'default_scale':.8,'min_scale':0,'max_scale':1,'trigger_words':desc['trigger_words'],
             'example_prompt':desc['example_prompt'],'description':'가중치 호환 검사 통과 · 이미지 품질은 미검증 · 라이선스: '+desc['license']}
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            local=Path(tmp)/row['file'];save_file({k:v.contiguous() for k,v in state.items()},str(local))
            self.bound.validate(row,local)
            row['sha256']=hashlib.sha256(local.read_bytes()).hexdigest()
            local.replace(self.root/row['file'])
            index=Path(tmp)/'library.json';index.write_text(json.dumps([*self.rows,row],ensure_ascii=False,indent=2));index.replace(self.index)
        self.bound.rows.append(row);self.bound.files[key]=self.root/row['file'];self.rows.append(row)
        return key
