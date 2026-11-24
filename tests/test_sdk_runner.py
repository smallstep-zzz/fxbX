import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

from fxbex import __version__
from fxbex.sdk_runner import worker_command


class SDKRunnerTests(unittest.TestCase):
    def test_old_worker_install_cannot_replace_active_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('fxbx','fxbex'):
                package=root/name;package.mkdir()
                (package/'__init__.py').write_text('raise RuntimeError("old worker package was loaded")')
            env=dict(os.environ,PYTHONPATH=directory)
            result=subprocess.run(worker_command(sys.executable,'fxbx','--version'),
                                  cwd=root,env=env,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),'fxbx '+__version__)
            result=subprocess.run(worker_command(sys.executable,'fxbex.worker','unused.fxbx'),
                                  input='{"action":"unload"}\n',cwd=root,env=env,
                                  capture_output=True,text=True,check=True)
            self.assertTrue(json.loads(result.stdout.removeprefix('@@FXBEX_RESPONSE@@'))['ok'])

    def test_worker_does_not_require_a_second_fxbx_install(self):
        with tempfile.TemporaryDirectory() as directory:
            venv.EnvBuilder(with_pip=False).create(directory)
            python=str(Path(directory)/'bin/python')
            env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
            result=subprocess.run(worker_command(python,'fxbx','--version'),cwd=directory,
                                  env=env,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),'fxbx '+__version__)

    def test_sdk_dependencies_are_not_loaded_from_launcher_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);launcher=root/'launcher';sdk=root/'sdk';sdk.mkdir()
            for name in ('fxbx','fxbex'):
                package=launcher/name;package.mkdir(parents=True)
                (package/'__init__.py').write_text('')
            from fxbex import sdk_runner
            bootstrap=launcher/'fxbex/sdk_runner.py';shutil.copyfile(sdk_runner.__file__,bootstrap)
            (launcher/'sdk_dependency.py').write_text('raise RuntimeError("foreign dependency")')
            (sdk/'sdk_dependency.py').write_text('VERSION="sdk-environment"')
            (launcher/'fxbx/__main__.py').write_text('import sdk_dependency; print(sdk_dependency.VERSION)')
            result=subprocess.run([sys.executable,str(bootstrap),'fxbx'],cwd=root,
                                  env=dict(os.environ,PYTHONPATH=str(sdk)),
                                  capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),'sdk-environment')

    def test_only_owned_entry_points_can_be_selected(self):
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            worker_command(sys.executable,'untrusted.adapter')
