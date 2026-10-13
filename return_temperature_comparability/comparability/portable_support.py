'Portable integrity and external-input contracts; no network or history lookup.'
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('Package paths must be safe relative POSIX paths')
    return path


def verify_hash_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'SHA256_MANIFEST.json').read_text())
    names = set()
    for item in manifest['files']:
        relative = relative_path(item['path'])
        if str(relative) in names:
            raise ValueError('Duplicate manifest path')
        names.add(str(relative))
        path = root / str(relative)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Missing or unsafe package file: ' + str(relative))
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('Package hash mismatch: ' + str(relative))
    return len(names)


def resolve_input_paths(manifest, manifest_path):
    base = Path(manifest_path).resolve().parent
    result = {}
    for item in manifest['inputs']:
        if item['id'] in result:
            raise ValueError('Duplicate input identity')
        path = Path(item['path'])
        result[item['id']] = path.resolve() if path.is_absolute() else (base / path).resolve()
    required = {'predist', 'xai', 'temporal_reader', 'weighted_reader', 'valid_day_carrier'}
    if set(result) != required:
        raise ValueError('Input manifest must contain exactly the five registered identities')
    return result


def missing_external_inputs(manifest, manifest_path):
    inputs = resolve_input_paths(manifest, manifest_path)
    return sorted(key for key in ('predist', 'xai') if not inputs[key].is_file())
