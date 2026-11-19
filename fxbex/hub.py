"""Resolve catalog model IDs to immutable, checksum-verified HF packages."""
import hashlib,json,os,sys,shutil,tempfile,zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlsplit

CATALOG='https://raw.githubusercontent.com/smallstep-zzz/furyawesome/main/catalog.json'

PACKAGE_SUFFIXES = ('.fxbx', '.fxbex')


def _local_package(path, *, announce=True):
    if path.is_file():return path.resolve()
    if not path.is_dir():return None
    candidates=[p for suffix in PACKAGE_SUFFIXES for p in path.rglob('*'+suffix) if p.is_file()]
    if not candidates:return None
    selected=max(candidates,key=lambda p:(p.stat().st_mtime_ns,str(p)))
    if announce:print('Selected model: '+str(selected.resolve()),file=sys.stderr,flush=True)
    return selected.resolve()


def _fetch_merged(download, repo, member, revision, target, refresh):
    """Optional merged-Runnable sidecar stored beside the package at the same pinned commit.

    The commit pins its bytes like the package; ExecutionFXB still checks the plan hash,
    SDK versions, member hashes and dataflow before using it. Absent or invalid: skipped.
    """
    side=target.with_name(target.name+'.merged')
    if side.is_dir() and not refresh:return
    try:archive=download(repo,member+'.merged.zip',revision=revision,force_download=refresh)
    except Exception:return  # this release has no merged sidecar
    try:
        with zipfile.ZipFile(archive) as z:
            infos=z.infolist()
            if ('manifest.json' not in {i.filename for i in infos} or sum(i.file_size for i in infos)>(8<<30)
                    or any(i.filename.startswith('/') or '..' in Path(i.filename).parts for i in infos)):
                raise ValueError('unsafe merged sidecar')
            staged=Path(tempfile.mkdtemp(prefix='.merged-',dir=target.parent))
            z.extractall(staged)
        shutil.rmtree(side,ignore_errors=True);os.replace(staged,side)
        print('Merged runnables saved: '+str(side),file=sys.stderr,flush=True)
    except (zipfile.BadZipFile,OSError,ValueError) as exc:
        print('Merged sidecar skipped: '+str(exc),file=sys.stderr,flush=True)


def _fetch_companions(download, repo, member, revision, target, refresh):
    """Optional sample input stored beside the package at the same pinned commit.

    MEMBER.sample.safetensors -> TARGET.sample.safetensors (`fxbx run --sample` input).
    Weights are never republished: the plan names the original checkpoint (original_checkpoint).
    """
    local=target.with_name(target.name+'.sample.safetensors')
    if local.is_file() and not refresh:return
    try:path=download(repo,member+'.sample.safetensors',revision=revision,force_download=refresh)
    except Exception:return  # this release does not ship a sample
    fd,temp=tempfile.mkstemp(prefix='.sample-',dir=target.parent);os.close(fd)
    try:shutil.copyfile(path,temp);os.replace(temp,local)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    print('Sample saved: '+str(local),file=sys.stderr,flush=True)


def original_checkpoint(source, *, download=None):
    """Download the original HF checkpoint a plan names and verify every file's SHA-256.

    source = plan['source_metadata']['checkpoint'] = {'repo', 'revision' (40-hex commit),
    'files': {name: sha256}}. Files land in the normal Hugging Face cache, as for .fxb models.
    Returns the single safetensors file, or the snapshot folder for an indexed sharded checkpoint.
    """
    import re
    repo,revision,files=source.get('repo',''),source.get('revision',''),source.get('files') or {}
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo) or not re.fullmatch('[0-9a-f]{40}',revision) or not files
            or any(not re.fullmatch(r'([A-Za-z0-9_-][A-Za-z0-9_.-]*/)?[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors(\.index\.json)?',f)
                   or '..' in f.split('/') or not re.fullmatch('[0-9a-f]{64}',h)
                   for f,h in files.items())):
        raise ValueError('Invalid original checkpoint reference')
    if download is None:
        try:from huggingface_hub import hf_hub_download as download
        except ImportError:raise RuntimeError('Install download support: pip install huggingface_hub') from None
    print(f'Original weights: {repo}@{revision[:8]}',file=sys.stderr,flush=True)
    paths={}
    for name,expected in sorted(files.items()):
        path=Path(download(repo,name,revision=revision));h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Original checkpoint checksum mismatch: '+name)
        paths[name]=path
    shards=[n for n in paths if n.endswith('.safetensors')]
    if len(shards)==1:return paths[shards[0]]
    index=[n for n in paths if n.endswith('.index.json')]
    if len(index)!=1:raise ValueError('Sharded original checkpoint needs its index file')
    return paths[index[0]].parent


def catalog_matches(catalog,name,version=None):
    matches=[p for p in catalog['models'] if name in (p['model']['name'],p['model']['source_repo'],p['model']['source_repo'].split('/')[-1]) and (version is None or p['release']['version']==version)]
    sources={p['model']['source_repo'] for p in matches}
    if len(sources)>1:raise ValueError('Short model name is ambiguous; include the publisher/model ID')
    if len(matches)<=1:return matches
    order=catalog.get('release_order',[])
    def rank(p):
        key=p['model']['source_repo']+'@'+p['release']['version']+'#'+p['release']['author_sha256']
        return order.index(key) if key in order else -1
    if max(map(rank,matches))<0:raise ValueError('No catalog release order available; select MODEL@VERSION')
    return [max(matches,key=rank)]


def resolve_package(value, *, refresh=False):
    import re
    identifier=str(value)
    if identifier.startswith('./'):identifier=identifier[2:]
    name,sep,version=identifier.partition('@')
    if sep and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',version):raise ValueError('Invalid release version')
    path=Path(value).expanduser()
    explicit_local=isinstance(value,os.PathLike) or str(value).startswith(('.', '/', '~')) or identifier.endswith(PACKAGE_SUFFIXES) or path.is_file()
    inferred=None
    if '/' not in name and not path.exists():
        folders=[p for p in Path('.').glob('*/'+name) if p.is_dir()]
        if len(folders)==1:inferred=folders[0]
        elif len(folders)>1:raise ValueError('Short model name is ambiguous locally; include publisher/model ID')
    # An explicit version never reuses an unversioned legacy file.
    candidate=inferred if inferred else path
    local=_local_package(candidate.parent/candidate.name.split('@')[0]/version if sep else candidate,announce=False)
    if local and (explicit_local or sep) and not refresh:
        print('Selected model: '+str(local),file=sys.stderr,flush=True)
        return local
    if explicit_local and (isinstance(value,os.PathLike) or identifier.endswith(PACKAGE_SUFFIXES) or identifier.startswith(('.', '/', '~'))):raise FileNotFoundError(path)
    # Inferred publisher/model folders are caches, not explicit local selections.
    # A short catalog ID must not silently select an older, recently touched file.
    try:
        with urlopen(CATALOG,timeout=30) as response:
            raw=response.read(4_000_001)
    except OSError:
        if local:
            print('Catalog unavailable; using local package. Latest release is unknown.',file=sys.stderr,flush=True)
            print('Selected model: '+str(local),file=sys.stderr,flush=True)
            return local
        raise
    if len(raw)>4_000_000:raise ValueError('Catalog too large')
    catalog=json.loads(raw)
    name,sep,version=identifier.partition('@')
    matches=catalog_matches(catalog,name,version if sep else None)
    if not matches and local:
        print('Selected model: '+str(local),file=sys.stderr,flush=True)
        return local
    if len(matches)!=1:raise ValueError('Model ID is missing or ambiguous; use exact source_repo@version from FuryAwesome')
    release=matches[0]['release'];u=urlsplit(release['artifact_url']);parts=u.path.strip('/').split('/')
    import re
    if u.scheme!='https' or u.netloc!='huggingface.co' or u.query or u.fragment or len(parts)<5 or parts[2]!='resolve' or not re.fullmatch('[0-9a-f]{40}',parts[3]) or any(x in ('.','..','') for x in parts):raise ValueError('Expected immutable HF artifact URL')
    expected=release['author_sha256']
    if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Missing SHA256')
    source_id=matches[0]['model']['source_repo']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',source_id) or any(x in ('.','..') for x in source_id.split('/')):raise ValueError('Invalid source model ID')
    selected_version=release['version']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',selected_version):raise ValueError('Invalid release version')
    print(f'Selected release: {source_id}@{selected_version}',file=sys.stderr,flush=True)
    from .settings import cache_root
    destination=cache_root()/"models"/source_id/selected_version
    suffix = '.fxbx' if parts[-1].endswith('.fxbx') else '.fxbex'
    target=destination/(source_id.split('/')[-1]+suffix)
    if target.exists() and not refresh:
        local_hash=hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):local_hash.update(block)
        if local_hash.hexdigest()!=expected:raise ValueError('Installed release differs from catalog checksum; refusing to overwrite')
        print('Selected model: '+str(target.resolve()),file=sys.stderr,flush=True)
        return target.resolve()
    try:from huggingface_hub import hf_hub_download
    except ImportError:raise RuntimeError('Install download support: pip install huggingface_hub') from None
    print('Downloading model from Hugging Face (cached files are reused)...',file=sys.stderr,flush=True)
    path=Path(hf_hub_download('/'.join(parts[:2]),'/'.join(parts[4:]),revision=parts[3],force_download=refresh))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Downloaded package checksum mismatch')
    destination.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.download-',dir=destination);os.close(fd)
    try:
        from tqdm.auto import tqdm
        with path.open('rb') as source,open(temp,'wb') as output,tqdm(total=path.stat().st_size,unit='B',unit_scale=True,desc='Saving model',file=sys.stderr) as bar:
            for block in iter(lambda:source.read(1024*1024),b''):
                output.write(block);bar.update(len(block))
        os.replace(temp,target)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    print('Model saved: '+str(target.resolve()),file=sys.stderr,flush=True)
    _fetch_merged(hf_hub_download,'/'.join(parts[:2]),'/'.join(parts[4:]),parts[3],target,refresh)
    _fetch_companions(hf_hub_download,'/'.join(parts[:2]),'/'.join(parts[4:]),parts[3],target,refresh)
    return target.resolve()
