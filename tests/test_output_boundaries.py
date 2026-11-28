import unittest

from fxbex.output_boundaries import assemble_returns, validate_returns


class OutputBoundarySafety(unittest.TestCase):
    def validate(self, record):
        def fail(message):
            raise ValueError(message)
        return validate_returns({'return_values':[record]}, ['x'],
                                {'x':{'shape':[2,3],'stride':[3,1],'dtype':'float32'}},fail)

    def test_accepts_a_zero_stride_view_inside_its_source(self):
        self.validate({'kind':'view','source':'x','shape':[4,2,3],'stride':[0,3,1],'offset':0})

    def test_rejects_alias_beyond_storage_or_unverified_source(self):
        valid={'kind':'view','source':'x','shape':[2,3],'stride':[3,1],'offset':0}
        for record in (valid | {'offset':1},valid | {'stride':[4,1]},valid | {'source':'y'}):
            with self.subTest(record=record),self.assertRaises(ValueError):
                self.validate(record)

    def test_rejects_unbounded_allocation_and_unjustified_widening(self):
        for record in ({'kind':'allocation','shape':[2,3],'stride':[1,1<<30],'dtype':'float32'},
                       {'kind':'index64','source':'x'}):
            with self.subTest(record=record),self.assertRaises(ValueError):
                self.validate(record)

    def test_allocation_keeps_layout_with_known_backing_bytes(self):
        import torch

        record={'kind':'allocation','shape':[2,2],'stride':[3,1],'dtype':'float32'}
        (result,)=assemble_returns([record],{},torch,'cpu')
        self.assertEqual(result.stride(),(3,1))
        self.assertEqual(result.as_strided((5,),(1,)).tolist(),[0.]*5)

    def test_writeback_has_exact_user_destination_and_returns_its_identity(self):
        import torch

        def fail(message):
            raise ValueError(message)

        record={'kind':'writeback','source':'y','target':'out'}
        entry={'inputs':['out'],'return_values':[record]}
        spec={'shape':[2],'stride':[1],'dtype':'float32'}
        tensors={'out':spec | {'kind':'user_input'},'y':spec | {'kind':'intermediate'}}
        validate_returns(entry,['y'],tensors,fail)
        with self.assertRaisesRegex(ValueError,'exact user buffer'):
            validate_returns(entry | {'inputs':[]},['y'],tensors,fail)
        out=torch.zeros(2)
        (value,)=assemble_returns([record],{'y':torch.ones(2)},torch,'cpu',inputs={'out':out})
        self.assertIs(value,out)
        self.assertEqual(out.tolist(),[1.,1.])

    def test_compact_count_is_checked_before_trimming(self):
        import torch

        from fxbex.execution_fxb import ExecutionPlanError

        descriptor={'kind':'compact_indices','source':'rows','count':'count'}
        rows=torch.tensor([[1,3],[2,5]],dtype=torch.int32)
        (selected,)=assemble_returns([descriptor],{'rows':rows,'count':torch.tensor(1,dtype=torch.int32)},
                                     torch,'cpu')
        self.assertEqual(selected.dtype,torch.int64)
        self.assertEqual(selected.tolist(),[[1,3]])
        with self.assertRaisesRegex(ExecutionPlanError,'outside the declared capacity'):
            assemble_returns([descriptor],{'rows':rows,'count':torch.tensor(3,dtype=torch.int32)},torch,'cpu')

    def test_complex_view_reuses_interleaved_storage(self):
        import torch

        record={'kind':'complex_view','source':'x'}
        tensor=torch.tensor([[1.,2.],[3.,4.]])
        (value,)=assemble_returns([record],{'x':tensor},torch,'cpu')
        self.assertEqual(value.tolist(),[1+2j,3+4j])
        self.assertEqual(value.data_ptr(),tensor.data_ptr())
        with self.assertRaisesRegex(ValueError,'real and imaginary lanes'):
            self.validate(record)
