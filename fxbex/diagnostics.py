"""Actionable serving errors shared by the worker, console and CLI."""
import errno
import re
from urllib.error import URLError
from urllib.parse import urlsplit


def diagnostic(code, message, hint, **details):
    return {'code': code, 'message': message, 'hint': hint, **details}


def _hf_error(exc):
    if not type(exc).__module__.startswith('huggingface_hub.'):
        return None
    from huggingface_hub import utils

    # The Hub kept these classes in utils across our supported versions.
    def kind(name):
        return isinstance(exc, getattr(utils, name, ()))

    if not (kind('HfHubHTTPError') or kind('LocalTokenNotFoundError') or kind('LocalEntryNotFoundError')):
        return None
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    request = getattr(response, 'request', None)
    try:
        url = urlsplit(str(getattr(response, 'url', None) or getattr(request, 'url', '')))
    except ValueError:
        url = urlsplit('')
    path = url.path.removeprefix('/api/models/').strip('/').split('/')
    repo = '/'.join(path[:2])
    where = {}
    if url.scheme == 'https' and url.netloc == 'huggingface.co' and re.fullmatch(
            r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
        where['url'] = 'https://huggingface.co/' + repo
        where['repo'] = repo
    target = where.get('repo', '원본 저장소')
    if kind('GatedRepoError'):
        return diagnostic('hf_access_required', f'{target} 가중치에 접근할 수 없습니다.',
                          '원본 모델 페이지에서 약관에 동의하거나 접근을 요청하세요. 승인된 같은 계정으로 fxbX를 실행하는 NPU 서버에서 로그인한 뒤 다시 시도하세요.',
                          command='hf auth login', **where)
    if kind('RevisionNotFoundError') or (status == 404 and kind('EntryNotFoundError')):
        return diagnostic('hf_file_missing', f'{target}에서 릴리스가 지정한 파일 또는 리비전을 찾을 수 없습니다.',
                          '최신 FuryAwesome 릴리스를 선택하세요. 같은 오류가 나면 게시자에게 파일 이름과 리비전 확인을 요청하세요.', **where)
    if kind('LocalEntryNotFoundError'):
        return diagnostic('hf_cache_missing', '필요한 파일이 서버 캐시에 없습니다.',
                          'NPU 서버에서 Hugging Face에 연결할 수 있는지와 오프라인 설정을 확인한 뒤 다시 시도하세요.', **where)
    if kind('LocalTokenNotFoundError'):
        return diagnostic('hf_login_required', 'NPU 서버에 Hugging Face 로그인이 필요합니다.',
                          'fxbX를 실행하는 서버 계정에서 로그인한 뒤 다시 시도하세요.',
                          command='hf auth login', **where)
    if status in (401, 403, 404) and kind('RepositoryNotFoundError'):
        return diagnostic('hf_repo_unavailable', f'{target} 저장소를 열 수 없습니다.',
                          '주소를 확인하세요. 비공개 저장소라면 열람 권한이 있는 계정으로 NPU 서버에서 로그인하세요.',
                          command='hf auth whoami' if status == 403 else 'hf auth login', **where)
    if status in (401, 403):
        return diagnostic('hf_login_required' if status != 403 else 'hf_permission_denied',
                          f'Hugging Face가 {target} 요청을 거부했습니다.',
                          'NPU 서버의 로그인 계정과 토큰의 읽기 권한을 확인하세요. 필요하면 권한이 있는 계정으로 다시 로그인하세요.',
                          command='hf auth whoami' if status == 403 else 'hf auth login', **where)
    if status == 429:
        return diagnostic('hf_rate_limit', 'Hugging Face 요청 한도에 도달했습니다.',
                          '잠시 기다렸다가 다시 시도하세요. 서버에서 로그인하면 계정의 요청 한도가 적용됩니다.',
                          command='hf auth login', **where)
    return diagnostic('hf_unavailable', f'Hugging Face 파일 요청이 실패했습니다 (HTTP {status or "응답 없음"}).',
                      '서버의 네트워크 연결과 저장소 주소를 확인하고 다시 시도하세요.', **where)


def describe_error(exc):
    value = getattr(exc, 'diagnostic', None)
    if isinstance(value, dict):
        return value
    cause = exc
    for _ in range(5):
        hf = _hf_error(cause)
        if hf is not None:
            return hf
        cause = cause.__cause__ or cause.__context__
        if cause is None:
            break
    from .dependencies import MissingPackages, missing_import
    reason = str(exc).strip() or type(exc).__name__
    reason = re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[redacted]', reason)[:1500]
    if isinstance(exc, (MissingPackages, ModuleNotFoundError)):
        if isinstance(exc, MissingPackages) or missing_import(exc) is not None:
            return diagnostic('sdk_packages_missing', reason,
                              '패키지 요구사항에 맞는 SDK 환경에서 fxbX를 실행하거나 FXBEX_WORKER_PYTHON으로 해당 Python을 지정하세요.',
                              command='fxbx doctor')
    if reason.startswith('NPU busy:') or isinstance(exc, BlockingIOError):
        return diagnostic('npu_busy', '다른 작업이 NPU를 사용 중입니다.',
                          '다른 fxbX 콘솔에서 서빙을 중지하거나 기존 NPU 작업이 끝난 뒤 다시 시도하세요.')
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return diagnostic('disk_full', '서버 저장 공간이 부족합니다.',
                          '모델 캐시와 임시 폴더가 있는 디스크의 여유 공간을 확인하세요.', command='df -h')
    if isinstance(exc, MemoryError) or 'out of memory' in reason.lower():
        return diagnostic('memory_exhausted', reason,
                          '서버 RAM과 NPU 메모리 사용량을 확인하고, 필요한 경우 모델의 더 작은 실행 프로필을 선택하세요.')
    if isinstance(exc, (URLError, TimeoutError, ConnectionError)):
        return diagnostic('network_error', reason, 'NPU 서버의 네트워크 연결을 확인하고 다시 시도하세요.')
    if 'checksum mismatch' in reason.lower() or 'catalog checksum' in reason.lower():
        return diagnostic('checksum_mismatch', reason, '선택한 릴리스의 원본 해시와 캐시를 확인하세요. 해당 릴리스만 다시 받아 비교할 수 있습니다.')
    if 'SDK mismatch' in reason:
        return diagnostic('sdk_mismatch', reason, '해당 릴리스의 검증 환경과 현재 SDK 버전을 비교하세요.', command='fxbx doctor')
    return diagnostic('request_rejected' if isinstance(exc, (ValueError, TypeError)) else 'runtime_error',
                      reason, '위 원인을 확인한 뒤 다시 시도하세요. 같은 문제가 반복되면 작업자 로그를 확인하세요.')


def worker_exit(code, log_path):
    message = (f'FXBEX worker exited ({code}); see {log_path}' if code is not None
               else f'FXBEX worker connection closed; see {log_path}')
    if code in (-11, 139):
        hint = '작업자가 네이티브 실행 중 비정상 종료됐습니다 (SIGSEGV). 모델을 다시 적재하고, 반복되면 SDK 버전과 로그의 마지막 실행 단계를 확인하세요.'
    elif code in (-9, 137):
        hint = '작업자가 강제 종료됐습니다 (SIGKILL). 서버의 메모리 한도와 외부 종료 기록을 확인하세요.'
    else:
        hint = '모델을 다시 적재하세요. 같은 종료가 반복되면 작업자 로그를 확인하세요.'
    return diagnostic('worker_exited', message, hint, log=str(log_path), exit_code=code)
