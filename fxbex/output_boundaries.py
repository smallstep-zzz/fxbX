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
               for r in records if r['kind'] in ('allocation','zero_tensor'))


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
        if kind=='writeback':
            source,target=record.get('source'),record.get('target')
            if (set(record)!={'kind','source','target'} or len(records)!=1
                    or not isinstance(source,str) or not isinstance(target,str)
                    or source not in outputs or target not in entry['inputs']
                    or tensors[source]['kind']!='intermediate' or 'index_input_dtype' in tensors[target]
                    or any(tensors[source][k]!=tensors[target][k] for k in ('shape','dtype','stride'))):
                failure('Writeback requires one exact user buffer and one produced result')
            referenced.add(source)
            continue
        if kind=='compact_indices':
            source,count=record.get('source'),record.get('count')
            if (set(record)!={'kind','source','count'} or not isinstance(source,str) or not isinstance(count,str)
                    or source==count or source not in outputs or count not in outputs):
                failure('Compact indices need declared coordinate and count outputs')
            rows,length=tensors[source],tensors[count]
            if (rows['dtype']!='int32' or length['dtype']!='int32' or length['shape']!=[]
                    or len(rows['shape'])!=2 or rows['kind']!='intermediate' or length['kind']!='intermediate'):
                failure('Compact indices need a signed int32 matrix and scalar count')
            referenced.update((source,count))
            continue
        if kind=='complex_view':
            source=record.get('source')
            if set(record)!={'kind','source'} or not isinstance(source,str) or source not in outputs:
                failure('Complex return needs a declared real-imaginary tensor')
            spec=tensors[source]
            if (spec['dtype']!='float32' or not spec['shape'] or spec['shape'][-1]!=2
                    or spec['stride'][-1]!=1 or any(n%2 for n in spec['stride'][:-1])):
                failure('Complex return requires interleaved FP32 real and imaginary lanes')
            referenced.add(source)
            continue
        if kind=='packed_int64':
            source=record.get('source')
            if set(record)!={'kind','source'} or not isinstance(source,str) or source not in outputs:
                failure('Packed int64 output must name a declared result')
            spec=tensors[source]
            if spec['kind']!='intermediate' or not is_int64_word_layout(spec):
                failure('Packed int64 needs a dense two-word int32 tensor')
            referenced.add(source)
            continue
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
        if kind not in ('view','allocation','zero_tensor'):
            failure('Unknown return value kind')
        fields = {'kind','shape','stride'} | ({'source','offset'} if kind == 'view' else {'dtype'})
        shape, stride = record.get('shape'), record.get('stride')
        if (set(record) != fields or not isinstance(shape,list) or not isinstance(stride,list)
                or len(shape) != len(stride) or len(shape) > MAX_RANK
                or any(type(n) is not int or not 0 <= n < 2**31 for n in (*shape,*stride))):
            failure('Return value needs exact bounded shape and stride')
        span = _storage_span(shape,stride)
        if kind in ('allocation','zero_tensor'):
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


def assemble_returns(records, bases, torch, device, *, inputs=None):
    result = []
    for record in records:
        kind = record['kind']
        if kind == 'scalar':
            value = record['value']
        elif kind in ('allocation','zero_tensor'):
            # ATen empty leaves values unspecified. Initialise the entire
            # backing span before returning memory to an external caller.
            span = _storage_span(record['shape'],record['stride'])
            storage = torch.zeros((span,),
                                  dtype=getattr(torch,record['dtype'].removeprefix('torch.')), device=device)
            value = storage.as_strided(record['shape'],record['stride'])
        else:
            value = bases[record['source']]
            if kind == 'view':
                value = value.as_strided(record['shape'],record['stride'],value.storage_offset()+record['offset'])
            elif kind == 'item':
                value = value.cpu().item()
            elif kind == 'index64':
                value = value.to(torch.int64)
            elif kind=='writeback':
                out=inputs[record['target']]
                out.copy_(value)
                value=out
            elif kind=='compact_indices':
                from .execution_fxb import ExecutionPlanError

                count=bases[record['count']].item()
                if not 0<=count<=value.shape[0]:
                    raise ExecutionPlanError('Native compact result count is outside the declared capacity')
                value=value[:count].to(torch.int64)
            elif kind=='complex_view':
                value=torch.view_as_complex(value)
            elif kind=='packed_int64':
                value=value.view(torch.int64).squeeze(-1)
        result.append(value)
    return tuple(result)


def is_int64_word_layout(spec):
    shape,stride=spec['shape'],spec['stride']
    return (spec['dtype']=='int32' and len(shape)>=2 and shape[-1]==2
            and tuple(stride)==tuple(prod(shape[i+1:]) for i in range(len(shape))))
