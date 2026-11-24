"""Client-side SSH forwarding; no SDK or device access is needed."""
import shlex
import shutil
import subprocess


def command(host, local_port=8000, remote_port=8000):
    if not isinstance(host, str) or not host or host.startswith('-') or any(c.isspace() or ord(c)<32 for c in host):
        raise ValueError('SSH host must be a host alias or user@hostname')
    for port in (local_port, remote_port):
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError('Ports must be between 1 and 65535')
    return ['ssh', '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', '-L', f'127.0.0.1:{local_port}:127.0.0.1:{remote_port}', host]


def launch(host, local_port=8000, remote_port=8000, *, print_only=False):
    args=command(host,local_port,remote_port)
    if print_only:
        print(shlex.join(args));return 0
    if not shutil.which('ssh'):raise ValueError('Install an OpenSSH client on the computer running your browser')
    print(f'브라우저를 쓰는 컴퓨터에서 연결합니다: http://127.0.0.1:{local_port}/\nSSH 인증 후 접속하세요. Ctrl-C로 터널만 종료합니다.',flush=True)
    try:return subprocess.call(args)
    except KeyboardInterrupt:return 130
