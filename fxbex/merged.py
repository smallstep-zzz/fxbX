"""Merged runnables embedded in a package (``merged/`` members) and the EDFs they make unnecessary.

A package may carry ``merged/manifest.json`` (schema ``fxbx.merged-runnables.v1``) and its
``merged/*.bin`` members: SDK-serialized Runnables that each replay consecutive plan steps. When
the manifest lists ``pruned_kernels``, those kernels' EDFs are absent from the archive. A kernel
may be pruned only if every step using it is covered by a merged group and none of those steps is
folded at load (folded steps run per kernel before the sequence is built) or belongs to an entry
that keeps per-step state. This module recomputes that from the plan instead of trusting the list.
"""
import json

MERGED_PREFIX = 'merged/'
MERGED_MANIFEST = MERGED_PREFIX + 'manifest.json'
SCHEMA = 'fxbx.merged-runnables.v1'
MAX_MANIFEST_SIZE = 64 * 1024 * 1024


class MergedError(ValueError):
    pass


def embedded_manifest(archive):
    """The embedded merged manifest of an open ZipFile, or None when the package has none."""
    try:
        info = archive.getinfo(MERGED_MANIFEST)
    except KeyError:
        return None
    if info.file_size > MAX_MANIFEST_SIZE:
        raise MergedError('Merged manifest exceeds size limit')
    try:
        manifest = json.loads(archive.read(MERGED_MANIFEST))
    except (ValueError, UnicodeError) as exc:
        raise MergedError('Invalid merged manifest') from exc
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise MergedError('Unsupported merged manifest schema')
    pruned = manifest.get('pruned_kernels', [])
    if not isinstance(pruned, list) or any(not isinstance(n, str) for n in pruned) or len(set(pruned)) != len(pruned):
        raise MergedError('pruned_kernels must be a list of unique kernel ids')
    return manifest


def stateless(plan, recipe):
    names = {n for step in recipe['steps'] for n in step['inputs'] + step['outputs']} | set(recipe['inputs'])
    return not recipe.get('state_updates') and all(plan['tensors'][n]['kind'] != 'state' for n in names)


def folded_positions(plan, recipe):
    """Steps that read only parameters, buffers or earlier folded outputs: they run at load."""
    const = {n for n, s in plan['tensors'].items() if s['kind'] in ('parameter', 'buffer')}
    folded = set()
    for position, step in enumerate(recipe['steps']):
        if all(n in const for n in step['inputs']):
            const.update(step['outputs']); folded.add(position)
    return folded


def prunable_kernels(plan, manifest):
    """Kernel ids whose EDFs no execution path can need while the embedded groups are used."""
    needed, used = set(), set()
    entries = manifest.get('entries') or {}
    for entry, recipe in plan['entries'].items():
        covered = {p for g in entries.get(entry) or [] for p in g.get('steps') or []}
        replayable = stateless(plan, recipe)
        folded = folded_positions(plan, recipe)
        for position, step in enumerate(recipe['steps']):
            used.add(step['kernel'])
            if not replayable or position in folded or position not in covered:
                needed.add(step['kernel'])
    return used - needed


def pruned_kernels(plan, manifest):
    """The declared pruned kernels after checking that each one is really unnecessary."""
    if manifest is None:
        return set()
    declared = set(manifest.get('pruned_kernels', []))
    unknown = declared - set(plan['kernels'])
    if unknown:
        raise MergedError('pruned_kernels names unknown kernels')
    extra = declared - prunable_kernels(plan, manifest)
    if extra:
        raise MergedError('pruned_kernels lists kernels that uncovered or folded steps still need')
    return declared
