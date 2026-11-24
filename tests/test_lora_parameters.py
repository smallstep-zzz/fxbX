import threading
from types import SimpleNamespace
from unittest import TestCase
import torch
from fxbex.execution_fxb import ExecutionFXB,ExecutionPlanError

class ParameterUpdateTest(TestCase):
    def model(self):
        m=ExecutionFXB.__new__(ExecutionFXB);m._lock=threading.RLock();m.stack=object();m.failed=False;m.torch=torch
        spec={'kind':'parameter','target':'weight','shape':[2,2],'stride':[2,1],'dtype':'float32'}
        m.plan={'entries':{'forward':{'state_updates':{}}},'tensors':{'w':spec,'alias':dict(spec)}}
        m.resident={'w':torch.eye(2),'alias':torch.eye(2)};m._sequences={'old':object()};m._sync=lambda:None
        m._upload=lambda value,spec:value.clone()
        def rebuild():m._sequences={'new':m.resident['w']}
        m._build_sequences=rebuild
        return m
    def test_aliases_and_sequences_use_new_parameter(self):
        m=self.model();old=m.resident['w'];v=torch.ones(2,2);m.replace_parameters({'weight':v})
        self.assertIs(m.resident['w'],m.resident['alias']);self.assertTrue(torch.equal(m._sequences['new'],v));self.assertTrue(torch.equal(old,torch.eye(2)))
    def test_bad_shape_guard_and_memory_budget_do_not_upload(self):
        for kind in ('shape','guard','budget','state'):
            m=self.model();calls=[];m._upload=lambda *args:calls.append(args)
            kwargs={};value=torch.ones(2,2)
            if kind=='shape':value=torch.ones(3,2)
            if kind=='guard':m.plan['frozen_buffer_guards']=[{'target':'weight'}]
            if kind=='budget':kwargs['max_bytes']=1
            if kind=='state':m.plan['entries']['forward']['state_updates']={'s':'next'}
            with self.assertRaises(ExecutionPlanError):m.replace_parameters({'weight':value},**kwargs)
            self.assertFalse(calls);self.assertFalse(m.failed)
    def test_failed_rebuild_requires_reload(self):
        m=self.model();old=m.resident
        def fail():raise RuntimeError('SDK rejected new sequence')
        m._build_sequences=fail
        with self.assertRaises(RuntimeError):m.replace_parameters({'weight':torch.ones(2,2)})
        self.assertIs(m.resident,old);self.assertTrue(m.failed)
        with self.assertRaises(ExecutionPlanError):m.replace_parameters({})
