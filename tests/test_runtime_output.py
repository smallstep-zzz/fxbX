import contextlib
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fxbex.cli import _runtime_call, main


class Output(unittest.TestCase):
    def test_cli_keeps_loader_log_alive_and_closes_it_after_unload(self):
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)

            def load():
                captured["handler"] = logging.StreamHandler(sys.stderr)

            def unload(*args):
                handler = captured["handler"]
                self.assertFalse(handler.stream.closed)
                handler.emit(logging.makeLogRecord({"msg": "worker shutdown"}))

            session = SimpleNamespace(
                report_dir=report, root=report, memory={},
                manifest={"runtime": {"adapter": "chat"}},
                __enter__=load, __exit__=unload,
                chat=lambda messages, limit: {"text": "answer"},
            )
            contract = {"method": "chat"}
            with patch("fxbex.settings.runtime_python", return_value=sys.executable), \
                    patch("fxbex.hub.resolve_package", return_value=report / "model.fxbx"), \
                    patch("fxbex.schema.describe", return_value=contract), \
                    patch("fxbex.schema.resolve_sample", return_value={"messages": []}), \
                    patch("fxbex.runtime.Session", return_value=session), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "Model", "--sample"]), 0)
            self.assertEqual((report / "runtime.log").read_text(), "worker shutdown\n")
            self.assertTrue(captured["handler"].stream.closed)

    def test_captured_logger_stays_writable_until_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            with path.open("a") as log:
                handler = _runtime_call(lambda: logging.StreamHandler(sys.stderr), log)
                record = logging.makeLogRecord({"msg": "worker shutdown"})
                _runtime_call(lambda: handler.emit(record), log)
            self.assertEqual(path.read_text(), "worker shutdown\n")

    def test_default_logs_and_verbose_prints(self):
        def model():
            print("NPU EXECUTE START")
            print("SDK diagnostic", file=sys.stderr)
            return {"text": "Hello"}

        log, out, err = io.StringIO(), io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(_runtime_call(model, log), {"text": "Hello"})
            self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))
            self.assertIn("SDK diagnostic", log.getvalue())
            _runtime_call(model, None)
        self.assertIn("NPU EXECUTE START", out.getvalue())
        self.assertIn("SDK diagnostic", err.getvalue())

    def test_errors_propagate(self):
        def fail():
            raise ValueError("inference failed")

        with self.assertRaisesRegex(ValueError, "inference failed"):
            _runtime_call(fail, io.StringIO())
