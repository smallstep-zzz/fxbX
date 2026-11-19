import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fxbex.cli import main
from fxbex.package import PackageError, inspect


def _native(directory, manifest):
    path = Path(directory) / 'model.fxbx'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
    return path


class NativeRouting(unittest.TestCase):
    def test_native_fxb_gets_loader_guidance_not_missing_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _native(directory, {'format_version': 2, 'kernels': {}})
            with self.assertRaisesRegex(PackageError, 'NATIVE_FXB_REQUIRES_EXPLICIT_LOADER'):
                inspect(path)

    def test_inspect_official_fxb_without_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _native(directory, {'format_version': 2, 'model': {'architecture': 'official'}, 'versions': {'sdk': 'x'}})
            out = io.StringIO()
            with patch('fxbex.native_fxb.validate_native', return_value={'producer_metadata': None}), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(main(['inspect', str(path)]), 0)
            result = json.loads(out.getvalue()[out.getvalue().index('{'):])
            self.assertIsNone(result['execution_plan'])
            self.assertEqual(result['compatibility'], {'sdk': 'x'})

    def test_inspect_execution_plan_summary(self):
        plan = {'schema': 'tclmachine.execution-plan.v1', 'compatibility': {'torch': '2'},
                'entries': {'forward': {'inputs': ['x'], 'outputs': ['y'], 'steps': [{}, {}]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = _native(directory, {'format_version': 2, 'model': {'name': 'm'}})
            out = io.StringIO()
            with patch('fxbex.native_fxb.validate_native', return_value={'producer_metadata': {'execution_plan': {}}}), \
                    patch('fxbex.execution_fxb.read_execution_plan', return_value=plan), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(main(['inspect', str(path)]), 0)
            result = json.loads(out.getvalue()[out.getvalue().index('{'):])
            self.assertEqual(result['execution_plan']['entries']['forward']['steps'], 2)
            self.assertEqual(result['compatibility'], {'torch': '2'})


class CleanupKeepsOriginalError(unittest.TestCase):
    def test_graph_exit_reraises_body_error(self):
        from fxbex.graph_fxb import GraphFXB
        graph = object.__new__(GraphFXB)
        with patch.object(GraphFXB, 'close', side_effect=RuntimeError('cleanup')):
            with self.assertRaisesRegex(ValueError, 'body') as caught:
                graph.__exit__(ValueError, ValueError('body'), None)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)


if __name__ == '__main__':
    unittest.main()
