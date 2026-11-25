from types import SimpleNamespace
from unittest import TestCase

from fxbex.diagnostics import describe_error, worker_exit
from huggingface_hub import utils


def hub_error(name, code):
    error_type = getattr(utils, name)
    error = error_type.__new__(error_type)
    error.response = SimpleNamespace(
        status_code=code,
        url='https://huggingface.co/owner/model/resolve/main/file?signature=private',
    )
    return error


class FailureGuidanceTests(TestCase):
    def test_gated_source_survives_wrapping_and_hides_query(self):
        outer = OSError('transformers could not load')
        outer.__cause__ = hub_error('GatedRepoError',401)
        detail = describe_error(outer)
        self.assertEqual(detail['code'],'hf_access_required')
        self.assertEqual(detail['url'],'https://huggingface.co/owner/model')
        self.assertEqual(detail['command'],'hf auth login')
        self.assertNotIn('signature',str(detail))

    def test_hf_auth_is_separate_from_pinned_revision_failure(self):
        self.assertEqual(describe_error(hub_error('HfHubHTTPError',401))['code'],'hf_login_required')
        self.assertEqual(describe_error(hub_error('HfHubHTTPError',403))['command'],'hf auth whoami')
        missing = describe_error(hub_error('RevisionNotFoundError',404))
        self.assertEqual(missing['code'],'hf_file_missing')
        self.assertNotIn('command',missing)

    def test_worker_signals_report_observation(self):
        self.assertIn('SIGSEGV',worker_exit(-11,'worker.log')['hint'])
        self.assertIn('SIGKILL',worker_exit(-9,'worker.log')['hint'])
