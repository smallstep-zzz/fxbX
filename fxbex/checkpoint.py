"""Header-first, tensor-at-a-time external weights for native execution plans.

Adapts furiosa-opt-models' Checkpoint/strict-load ownership using safetensors' public
API; no Rust memory ABI, model implementation, dtype conversion or quantization.
"""
from collections.abc import Mapping
import json
from pathlib import Path


TORCH_FILES = ('.ckpt', '.pt', '.pth', '.bin')
_DTYPES = {'BOOL': 'bool', 'U8': 'uint8', 'I8': 'int8', 'I16': 'int16',
           'I32': 'int32', 'I64': 'int64', 'F16': 'float16', 'BF16': 'bfloat16',
           'F32': 'float32', 'F64': 'float64', 'F8_E4M3': 'float8_e4m3fn',
           'F8_E5M2': 'float8_e5m2'}


class SafeTensorWeights(Mapping):
    """Local safetensors or sharded checkpoint, or one PyTorch file, with optional explicit target names.

    ``names`` maps execution-plan targets to checkpoint tensor names. Supplying it
    explicitly selects a subset; otherwise every checkpoint tensor must be bound.
    No tensor payload is read at construction or validate(). Keep files immutable
    while a session uses them. Values are never cached on the CPU by this object.
    ``extra`` maps further targets to small in-memory tensors (a package's derived
    buffers); plan specs may slice rows (checkpoint_slice) or cast (checkpoint_dtype).
    """
    def __init__(self, path, *, names=None, extra=None):
        from safetensors import safe_open
        path = Path(path).absolute()
        indexed = None
        self._torch = None
        if path.is_file() and path.suffix in TORCH_FILES:
            self._init_torch(path, names, extra)
            return
        if path.is_file():
            files = [path]
        else:
            index = path / 'model.safetensors.index.json'
            if index.exists():
                if index.stat().st_size > 16 * 1024**2:
                    raise ValueError('checkpoint index exceeds 16 MiB')
                indexed = json.loads(index.read_text())['weight_map']
                if not isinstance(indexed, dict) or not indexed:
                    raise ValueError('checkpoint index needs a nonempty weight_map')
                labels = list(indexed.values())
                if any(not isinstance(n, str) or Path(n).is_absolute() or '..' in Path(n).parts for n in labels):
                    raise ValueError('checkpoint shard path must stay relative')
                files = [path / n for n in sorted(set(labels))]
            else:
                files = sorted(path.glob('*.safetensors'))
        if not files or len(files) > 1024:
            raise ValueError('checkpoint needs 1..1024 safetensors files')
        self._specs = {}; self._files = {}; self._identities = {}
        for file in files:
            before = self._identity(file)
            with safe_open(str(file), framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    if name in self._files:
                        raise ValueError('duplicate checkpoint tensor: ' + name)
                    view = reader.get_slice(name)
                    self._specs[name] = {'shape': list(view.get_shape()), 'dtype': _DTYPES.get(view.get_dtype())}
                    self._files[name] = file
            if self._identity(file) != before:
                raise ValueError('checkpoint changed during metadata admission')
            self._identities[file] = before
        if indexed is not None and (set(indexed) != set(self._files) or any(self._files[k] != path / v for k, v in indexed.items())):
            raise ValueError('checkpoint headers disagree with weight_map')
        self.names = dict(names) if names is not None else {n: n for n in self._files}
        if (not self.names or any(not isinstance(k, str) or not k or not isinstance(v, str) or v not in self._files
                                  for k, v in self.names.items())):
            raise ValueError('explicit target mapping names an unknown checkpoint tensor')
        self.loaded_tensors = 0
        self._casts = {}; self._slices = {}
        self.extra = dict(extra or {})

    def _init_torch(self, path, names, extra):
        """One PyTorch checkpoint, read with weights_only=True (no code execution), memory-mapped."""
        import torch
        before = self._identity(path)
        try:
            state = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
        except RuntimeError:  # legacy (non-zip) serialization cannot be memory-mapped
            state = torch.load(str(path), map_location='cpu', weights_only=True)
        if isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
            state = state['state_dict']
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                                                for k, v in state.items()):
            raise ValueError('PyTorch checkpoint must hold a flat {name: tensor} state dict')
        if self._identity(path) != before:
            raise ValueError('checkpoint changed during metadata admission')
        self._torch = state
        self._specs = {k: {'shape': list(v.shape), 'dtype': str(v.dtype).removeprefix('torch.')} for k, v in state.items()}
        self._files = {k: path for k in state}; self._identities = {path: before}
        self.names = dict(names) if names is not None else {n: n for n in self._files}
        if (not self.names or any(not isinstance(k, str) or not k or not isinstance(v, str) or v not in self._files
                                  for k, v in self.names.items())):
            raise ValueError('explicit target mapping names an unknown checkpoint tensor')
        self.loaded_tensors = 0
        self._casts = {}; self._slices = {}
        self.extra = dict(extra or {})

    @staticmethod
    def _identity(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def __iter__(self): return iter({**self.names, **self.extra})
    def __len__(self): return len(set(self.names) | set(self.extra))

    def validate(self, targets):
        if set(targets) != set(self.names) | set(self.extra):
            raise ValueError('checkpoint targets differ; supply an explicit names mapping for a subset')
        for target, spec in targets.items():
            wanted = str(spec['dtype']).removeprefix('torch.')
            stride = []; size = 1
            for dim in reversed(spec['shape']):
                stride.insert(0, size); size *= max(dim, 1)
            if target in self.extra:
                value = self.extra[target]
                if (list(value.shape) != spec['shape'] or str(value.dtype).removeprefix('torch.') != wanted
                        or stride != spec['stride']):
                    raise ValueError('package tensor shape/dtype/contiguous layout differs: ' + target)
                continue
            declared = self._specs[self.names[target]]
            shape = list(declared['shape'])
            if 'checkpoint_slice' in spec:  # a row block of a fused original tensor
                start, stop = spec['checkpoint_slice']
                if not shape or not 0 <= start < stop <= shape[0]:
                    raise ValueError('checkpoint row slice is out of range: ' + target)
                shape[0] = stop - start
                self._slices[target] = (start, stop)
            stored = spec.get('checkpoint_dtype', wanted)  # original checkpoints are cast at load
            if declared['dtype'] != stored or shape != spec['shape'] or stride != spec['stride']:
                raise ValueError('checkpoint shape/dtype/contiguous layout differs: ' + target)
            if stored != wanted:
                self._casts[target] = wanted
        for file, identity in self._identities.items():
            if self._identity(file) != identity:
                raise ValueError('checkpoint changed after metadata admission')

    def __getitem__(self, target):
        from safetensors import safe_open
        if target in self.extra:
            return self.extra[target]
        name = self.names[target]; file = self._files[name]
        if self._identity(file) != self._identities[file]:
            raise ValueError('checkpoint changed after metadata admission')
        if self._torch is not None:
            tensor = self._torch[name].clone()  # a private copy, not a view of the mapped file
        else:
            with safe_open(str(file), framework='pt', device='cpu') as reader:
                tensor = reader.get_tensor(name)
        self.loaded_tensors += 1
        if target in self._slices:
            start, stop = self._slices[target]
            tensor = tensor[start:stop].contiguous()
        if target in self._casts:
            import torch
            tensor = tensor.to(getattr(torch, self._casts[target]))
        return tensor
