"""Validate and reconstruct declarative FXB return values."""
from math import prod

SCHEMA = 'tclmachine.execution-plan.v4'
MAX_ALLOCATION_BYTES = 16 << 20
MAX_RANK = 16


def is_metadata_only(plan):
    return (isinstance(plan, dict) and plan.get('schema') == SCHEMA
            and plan.get('kernels') == {})


def _storage_span(shape, stride):
    return 0 if 0 in shape else 1 + sum((n-1)*s for n,s in zip(shape,stride))


def allocation_bytes(records):
    from .memory import tensor_bytes
    return sum(tensor_bytes({'shape':[_storage_span(r['shape'],r['stride'])],'dtype':r['dtype']})
               for r in records if r['kind']=='allocation')


def validate_returns(entry, outputs, tensors, failure):
    from .memory import tensor_bytes

    records = entry.get('return_values')
    if not isinstance(records, list) or not 0 < len(records) <= 4096:
        failure('Return values require one or more descriptors')
    referenced = set()
    bytes_allocated = 0
    for record in records:
        if not isinstance(record, dict):
            failure('Invalid return value descriptor')
        kind = record.get('kind')
        if kind == 'scalar':
            if (set(record) != {'kind', 'value'} or type(record['value']) not in (bool, int)
                    or type(record['value']) is int and not -(1 << 63) <= record['value'] < 1 << 63):
                failure('Static scalar result must be a bool or signed int64')
            continue
        if kind in ('tensor','item','index64'):
            if set(record) != {'kind', 'source'} or not isinstance(record.get('source'),str) or record['source'] not in outputs:
                failure('Tensor return must name a declared entry output')
            if kind == 'item' and prod(tensors[record['source']]['shape']) != 1:
                failure('Scalar extraction requires exactly one element')
            if kind == 'index64' and tensors[record['source']]['dtype'] != 'int32':
                failure('Index widening needs a signed int32 source')
            referenced.add(record['source'])
            continue
        if kind not in ('view', 'allocation'):
            failure('Unknown return value kind')
        fields = {'kind','shape','stride'} | ({'source','offset'} if kind == 'view' else {'dtype'})
        shape, stride = record.get('shape'), record.get('stride')
        if (set(record) != fields or not isinstance(shape,list) or not isinstance(stride,list)
                or len(shape) != len(stride) or len(shape) > MAX_RANK
                or any(type(n) is not int or not 0 <= n < 2**31 for n in (*shape,*stride))):
            failure('Return value needs exact bounded shape and stride')
        span = _storage_span(shape,stride)
        if kind == 'allocation':
            dtype = record['dtype']
            if not isinstance(dtype,str):
                failure('Return allocation needs a dtype')
            size = tensor_bytes({'shape':[span],'dtype':dtype})
            if size is None:
                failure('Unknown return allocation dtype')
            bytes_allocated += size
            if bytes_allocated > MAX_ALLOCATION_BYTES:
                failure('Return allocation exceeds the byte limit')
        else:
            source, offset = record['source'], record['offset']
            if not isinstance(source,str) or source not in outputs or type(offset) is not int or offset < 0:
                failure('View needs a declared source and nonnegative offset')
            base = tensors[source]
            dense = 1
            for n,s in zip(reversed(base['shape']),reversed(base['stride'])):
                if n > 1 and s != dense:
                    failure('View source must use dense contiguous storage')
                dense *= n
            if offset+span > prod(base['shape']):
                failure('View exceeds the declared source storage')
            referenced.add(source)
    if referenced != set(outputs):
        failure('Return descriptors must cover the declared entry outputs')


def assemble_returns(records, bases, torch, device):
    result = []
    for record in records:
        kind = record['kind']
        if kind == 'scalar':
            value = record['value']
        elif kind == 'allocation':
            value = torch.empty_strided(record['shape'], record['stride'],
                                        dtype=getattr(torch,record['dtype'].removeprefix('torch.')), device=device)
        else:
            value = bases[record['source']]
            if kind == 'view':
                value = value.as_strided(record['shape'],record['stride'],value.storage_offset()+record['offset'])
            elif kind == 'item':
                value = value.cpu().item()
            elif kind == 'index64':
                value = value.to(torch.int64)
        result.append(value)
    return tuple(result)
