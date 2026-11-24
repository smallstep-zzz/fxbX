"""Load packaged linear LoRA pairs for explicit saved-plan parameter inputs.

FuryNo5/FHA owns conversion and authoring. No remote code, implicit name mapping,
new kernels, or changes to checkpoint integrity are performed by this loader.
"""
from collections import OrderedDict
import json
import math
from pathlib import Path,PurePosixPath


class BoundLoras:
    def __init__(self,root,components,*,max_cache_bytes=512*1024**2,max_update_bytes=2*1024**3):
        self.root=Path(root);self.components=components;self.limit=max_cache_bytes;self.update_budget=max_update_bytes
        self.cache=OrderedDict();self.cache_bytes=0;self.active=[];self.targets={};self.failed=False
        path=self.root/'lora.json'
        if not path.is_file():self.rows=[];return
        if path.stat().st_size>1024**2:raise ValueError('LoRA index exceeds 1 MiB')
        doc=json.loads(path.read_text())
        if doc.get('schema_version')!=1 or not isinstance(doc.get('adapters'),list):raise ValueError('Invalid LoRA index')
        self.rows=doc['adapters']
        if len(self.rows)>64:raise ValueError('Too many packaged LoRAs')
        seen=set()
        from safetensors import safe_open
        for row in self.rows:
            key=row.get('id')
            if not isinstance(key,str) or not key or key in seen:raise ValueError('Invalid LoRA identity')
            seen.add(key);file=PurePosixPath(row['file'])
            if file.is_absolute() or '..' in file.parts or len(file.parts)!=2 or file.parts[0]!='adapters' or file.suffix!='.safetensors':
                raise ValueError('LoRA weights must be packaged under adapters/')
            local=self.root/file
            if not local.is_file() or not 0<local.stat().st_size<=self.limit:raise ValueError('LoRA exceeds cache budget')
            bindings=row.get('components')
            if not isinstance(bindings,dict) or not bindings:raise ValueError('LoRA needs explicit component bindings')
            with safe_open(str(local),framework='pt',device='cpu') as reader:
                used=set()
                for component,pairs in bindings.items():
                    if component not in components or not isinstance(pairs,list) or not pairs:raise ValueError('Unknown LoRA component')
                    model=components[component];specs={s.get('target'):s for s in model.plan['tensors'].values() if s['kind']=='parameter'};targets=set()
                    if any(r['state_updates'] for r in model.plan['entries'].values()):raise ValueError('Stateful LoRA plans are not supported')
                    for pair in pairs:
                        target=pair['target'];spec=specs.get(target)
                        if target in targets or spec is None or 'checkpoint_slice' in spec:raise ValueError('Unsupported LoRA target')
                        targets.add(target)
                        if any(g['target']==target for g in model.plan.get('frozen_buffer_guards',[])):raise ValueError('LoRA targets a frozen constant')
                        a=reader.get_slice(pair['a']);b=reader.get_slice(pair['b']);ash=a.get_shape();bsh=b.get_shape()
                        if a.get_dtype() not in ('F32','F16','BF16') or b.get_dtype() not in ('F32','F16','BF16'):raise ValueError('LoRA needs floating tensors')
                        shape=spec['shape']
                        if (len(ash)!=2 or len(bsh)!=2 or len(shape)!=2 or not ash[0]>0 or
                                bsh[1]!=ash[0] or [bsh[0],ash[1]]!=shape):raise ValueError('Only shape-matched linear LoRA pairs are supported')
                        alpha=pair.get('alpha',ash[0])
                        if type(alpha) not in (int,float) or not math.isfinite(alpha):raise ValueError('Invalid LoRA alpha')
                        used.update((pair['a'],pair['b']))
                if used!=set(reader.keys()):raise ValueError('Unbound LoRA tensors; partial application is forbidden')

    def catalog(self):
        return [{k:r[k] for k in ('id','name','description','min_scale','max_scale','default_scale') if k in r} for r in self.rows]

    def _weights(self,row):
        key=row['id']
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        from safetensors.torch import load_file
        size=(self.root/row['file']).stat().st_size
        while self.cache and self.cache_bytes+size>self.limit:
            _,(_,old)=self.cache.popitem(last=False);self.cache_bytes-=old
        value=load_file(str(self.root/row['file']))
        self.cache[key]=(value,size);self.cache_bytes+=size
        return self.cache[key]

    def apply(self,chosen):
        if self.failed:raise RuntimeError('LoRA update failed; reload the package')
        if chosen==self.active:return
        import torch
        rows={r['id']:r for r in self.rows};next_targets={};contributions={}
        for selected in chosen:
            row=rows[selected['id']]
            for component,pairs in row['components'].items():
                for pair in pairs:
                    contributions.setdefault((component,pair['target']),[]).append((row,pair,selected['scale']))
                    next_targets.setdefault(component,set()).add(pair['target'])
        for component in set(self.targets)|set(next_targets):
            specs={s.get('target'):s for s in self.components[component].plan['tensors'].values() if s['kind']=='parameter'}
            names=self.targets.get(component,set())|next_targets.get(component,set())
            size=sum(math.prod(specs[t]['shape'])*torch.empty((),dtype=getattr(torch,specs[t]['dtype'])).element_size() for t in names)
            if size>self.update_budget:raise ValueError('LoRA replacement weights exceed staging memory budget')
        try:
            with torch.no_grad():
                for component in sorted(set(self.targets)|set(next_targets)):
                    model=self.components[component];values={}
                    for target in sorted(self.targets.get(component,set())|next_targets.get(component,set())):
                        value=model.weights[target].detach().float()
                        for row,pair,scale in contributions.get((component,target),[]):
                            weights,_=self._weights(row)
                            a=weights[pair['a']].float();b=weights[pair['b']].float()
                            value=value+(b@a)*(scale*pair.get('alpha',a.shape[0])/a.shape[0])
                        dtype=next(s['dtype'] for s in model.plan['tensors'].values() if s.get('target')==target)
                        values[target]=value.to(getattr(torch,dtype)).contiguous()
                    model.replace_parameters(values,max_bytes=self.update_budget)
            self.targets=next_targets;self.active=[dict(s) for s in chosen]
        except BaseException:
            self.failed=True
            raise
