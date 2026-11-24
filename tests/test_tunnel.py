import unittest
from unittest.mock import patch
from fxbex.tunnel import command,launch


class TunnelTests(unittest.TestCase):
    def test_loopback_only(self):
        args=command('user@npu',18000,8000)
        self.assertEqual(args[-2:],['127.0.0.1:18000:127.0.0.1:8000','user@npu'])
        self.assertIn('ExitOnForwardFailure=yes',args)
        self.assertNotIn('StrictHostKeyChecking=no',args)

    def test_reject_option_and_invalid_ports(self):
        for host in ('-oProxyCommand=x','foo bar','foo\nbar',''):
            with self.assertRaises(ValueError):command(host)
        for port in (0,65536,True):
            with self.assertRaises(ValueError):command('npu',port)

    def test_print_does_not_start_ssh(self):
        with patch('subprocess.call') as call,patch('builtins.print') as output:
            self.assertEqual(launch('npu',print_only=True),0)
            call.assert_not_called();self.assertIn('ssh -N',output.call_args.args[0])

    def test_cli_no_sdk(self):
        from fxbex.cli import _main
        with patch('fxbex.tunnel.launch',return_value=0) as run:
            self.assertEqual(_main(['tunnel','npu','--local-port','18000','--print-only']),0)
            run.assert_called_once_with('npu',18000,8000,print_only=True)
