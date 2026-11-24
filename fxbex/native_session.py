"""Explicit native execution-plan transport for the process-isolated web worker."""
from pathlib import Path
from types import SimpleNamespace
import time

from .settings import runs_path


class NativeSession:
    def __init__(self,path,*,report_dir=None,weights_dir=None,allow_sdk_mismatch=False):
        self.path=Path(path).resolve();self.root=self.path.parent
        self.report_dir=Path(report_dir or runs_path()/f'native_{time.time_ns()}').resolve();self.report_dir.mkdir(parents=True,exist_ok=True)
        self.weights_dir=weights_dir;self.model=None;self.failed=False;self.memory=None
        import threading
        self._predict_lock=threading.RLock();self.execution=SimpleNamespace(last=None)
        self.adapter=SimpleNamespace(predict=self.predict)

    def __enter__(self):
        from .checkpoint import SafeTensorWeights
        from .execution_fxb import ExecutionFXB
        weights=Path(self.weights_dir) if self.weights_dir else self.path.with_name(self.path.name+'.weights')
        if self.weights_dir and not weights.exists():raise ValueError('External weights path does not exist')
        model=ExecutionFXB(self.path,weights=SafeTensorWeights(weights) if weights.exists() else None)
        entry=model.plan['entries'].get('forward')
        if entry is None:raise ValueError('Native web execution requires a forward entry')
        if any(s['kind']=='state' for s in model.plan['tensors'].values()):raise ValueError('Stateful native models require an explicit lifecycle adapter')
        model.__enter__();self.model=model;self.memory=model.memory
        sample=str(self.path)+'.sample.safetensors'
        self.manifest={'model':{'name':self.path.stem},'files':{},'io_contract':{
            'schema_version':1,'method':'predict','example_input':{'tensors':sample,'output':'result.safetensors'},
            'inputs':{'tensors':{'type':'path'},'output':{'type':'path'}},
            'outputs':{'result_file':{'type':'path'}},
            'expected_result':'입력은 이미지 파일이 아닌 실행 프로필의 safetensors 텐서 파일입니다. 결과 텐서는 safetensors로 저장합니다.'}}
        return self

    def lora_catalog(self):
        return {'supported':False,'adapters':[],'max_active':0,'reason':'이 원시 텐서 패키지는 LoRA 바인딩을 제공하지 않습니다.'}

    def predict(self,inputs):
        if not isinstance(inputs,dict) or set(inputs)-{'tensors','output','loras'}:raise ValueError('Use tensors and optional output paths')
        from .lora import selection
        selection(inputs.get('loras',getattr(self,'default_loras',[])),self.lora_catalog())
        source=inputs.get('tensors')
        if not isinstance(source,str) or not Path(source).is_file():raise ValueError('입력 safetensors 파일이 없습니다. 게시자의 샘플 또는 해당 입력 계약에 맞는 파일을 지정하세요.')
        if self.failed:raise RuntimeError('Native session failed; reload it')
        from safetensors.torch import load_file,save_file
        tensors=load_file(source)
        entry=self.model.plan['entries']['forward']
        if all(n in tensors for n in entry['inputs']):values=tuple(tensors[n] for n in entry['inputs'])
        elif len(entry['inputs'])==1 and 'image0' in tensors:values=(tensors['image0'],)
        else:raise ValueError('입력 텐서 이름이 일치하지 않습니다: '+', '.join(entry['inputs']))
        with self._predict_lock:
            start=time.perf_counter()
            try:outputs=self.model.predict(values)
            except BaseException:
                self.failed=self.model.failed;raise
            elapsed=time.perf_counter()-start
            path=Path(inputs.get('output',self.report_dir/f'output-{time.time_ns()}.safetensors')).resolve();path.parent.mkdir(parents=True,exist_ok=True)
            outputs=dict(zip(entry['outputs'],outputs,strict=True))
            save_file({k:v.contiguous().clone() for k,v in outputs.items()},str(path))
            self.execution.last={'operation':'predict','wall_seconds':elapsed,'scope':'host request including transfers and waits'}
            return {'result_file':str(path),'outputs':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in outputs.items()},'wall_s':elapsed}

    def __exit__(self,*exc):
        if self.model:self.model.close();self.model=None
