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
