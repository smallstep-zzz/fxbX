"""Request-scoped LoRA selection advertised by a trusted package adapter."""
import math
import re
from inspect import getattr_static


def scene_prompt(value, *labels):
    if not isinstance(value, str) or len(value) > 4000:
        return ''
    value = value.strip()
    words = re.findall(r'[^\W_]+', value.casefold())
    compact = ''.join(words)
    # Hub widgets often contain just a repository name or activation token.
    if len(value.split()) < 2 or any(
        compact == ''.join(re.findall(r'[^\W_]+', label.casefold()))
        for label in labels if isinstance(label, str)
    ):
        return ''
    return value


def catalog(adapter):
    describe = getattr(adapter, 'lora_catalog', None) if getattr_static(adapter,'lora_catalog',None) is not None else None
    apply = getattr(adapter, 'apply_loras', None) if getattr_static(adapter,'apply_loras',None) is not None else None
    if not callable(describe) or not callable(apply):
        return {'supported': False, 'adapters': [], 'max_active': 0,
                'reason': '이 패키지는 LoRA 적용을 지원하지 않습니다.'}
    rows = describe()
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('Invalid package LoRA catalog')
    seen = set(); result = []
    for row in rows:
        if not isinstance(row, dict):raise ValueError('Invalid LoRA catalog entry')
        key = row.get('id');name = row.get('name', key)
        if (not isinstance(key,str) or not 1 <= len(key) <= 128 or key in seen
                or not isinstance(name,str) or not 1 <= len(name) <= 256):
            raise ValueError('Invalid or duplicate LoRA identity')
        seen.add(key)
        low, high, default = (row.get(k,v) for k,v in [('min_scale',0.0),('max_scale',1.0),('default_scale',1.0)])
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (low,high,default)) or not low <= default <= high:
            raise ValueError('Invalid LoRA scale range')
        entry={'id':key,'name':name,'description':str(row.get('description',''))[:1000],
               'min_scale':low,'max_scale':high,'default_scale':default}
        words=row.get('trigger_words',[])
        if not isinstance(words,list) or len(words)>16 or any(not isinstance(w,str) or not 0<len(w)<=128 for w in words):
            raise ValueError('Invalid LoRA trigger words')
        entry['trigger_words']=words
        prompt=row.get('example_prompt','')
        if not isinstance(prompt,str) or len(prompt)>4000:raise ValueError('Invalid LoRA example prompt')
        entry['example_prompt']=scene_prompt(prompt,key,name,*words)
        for field in ('source_repo','source_revision','source_file'):
            value = row.get(field)
            if isinstance(value, str) and 0 < len(value) <= 1024:
                entry[field] = value
        url=row.get('source_url','')
        if not isinstance(url,str) or (url and not re.fullmatch(r'https://huggingface\.co/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',url)):
            raise ValueError('LoRA source must be a Hugging Face model URL')
        entry['source_url']=url
        result.append(entry)
    limit = getattr(adapter,'max_active_loras',1)
    if type(limit) is not int or not 1 <= limit <= 8:raise ValueError('Invalid LoRA selection limit')
    return {'supported':True,'adapters':result,'max_active':limit,
            'reason':'' if result else '이 모델에 등록된 LoRA가 없습니다.'}


def selection(value, info):
    if not isinstance(value,list):raise ValueError('loras must be a list')
    if not info['supported'] and value:raise ValueError('This package does not support LoRA')
    if len(value)>info['max_active']:raise ValueError('Too many active LoRAs')
    known={r['id']:r for r in info['adapters']};seen=set();result=[]
    for item in value:
        if not isinstance(item,dict) or set(item)-{'id','scale'}:raise ValueError('Expected LoRA id and scale')
        key=item.get('id')
        if not isinstance(key,str) or key not in known or key in seen:raise ValueError('Unknown or duplicate LoRA')
        row=known[key];scale=item.get('scale',row['default_scale'])
        if type(scale) not in (int,float) or not math.isfinite(scale) or not row['min_scale']<=scale<=row['max_scale']:
            raise ValueError('LoRA scale is outside the supported range')
        seen.add(key)
        if scale!=0:result.append({'id':key,'scale':float(scale)})
    return sorted(result,key=lambda r:r['id'])


def cli_selection(values):
    result=[]
    for value in values:
        key,sep,scale=value.rpartition('=')
        if not sep:key=value
        if not 0<len(key)<=128:raise ValueError('Expected --lora ID[=SCALE]')
        row={'id':key}
        if sep:
            try:row['scale']=float(scale)
            except ValueError:raise ValueError('Invalid LoRA scale') from None
            if not math.isfinite(row['scale']):raise ValueError('Invalid LoRA scale')
        result.append(row)
    if len(result)>8 or len({r['id'] for r in result})!=len(result):raise ValueError('Duplicate or excessive LoRAs')
    return result
