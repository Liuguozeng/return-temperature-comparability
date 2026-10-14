'Offline unit tests and packaged-derived verification, not raw replay.'
import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'


def deny_network(*args, **kwargs):
    raise RuntimeError('Offline validation: network access forbidden')


def run(output, dependencies=()):
    dependencies = [Path(p).resolve(strict=True) for p in dependencies]
    if any(not p.is_dir() for p in dependencies):
        raise ValueError('Dependencies must be explicitly selected local runtime directories')
    sys.path[:0] = [str(p) for p in dependencies]
    from portable_support import verify_hash_manifest
    output = Path(output).resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('Output must be a new directory, not the package or its ancestor')
    if output.exists():
        raise ValueError('Validation output must not already exist')
    output.mkdir(parents=True)
    os.chdir(ROOT)
    tempfile.tempdir = str(output)
    sys.pycache_prefix = str(output / 'pycache')
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    # Allow package files, the selected interpreter/dependencies, and OS resources.
    # Historical workspace paths are not permitted even if they happen to exist.
    allowed = [ROOT, output, *dependencies, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
               Path(sys.executable).resolve().parent.parent,
               Path('/System'), Path('/Library'), Path('/usr'), Path('/private/etc'),
               Path('/var/db/timezone').resolve()]
    # The figure tests share one helper with the package-root renderer.
    helper = ROOT.parent / 'panel_geometry.py'
    declarations = dict(line.split('  ', 1)[::-1] for line in (ROOT.parent / 'SHA256SUMS').read_text().splitlines())
    if declarations.get('panel_geometry.py') != hashlib.sha256(helper.read_bytes()).hexdigest():
        raise ValueError('Shared panel helper is missing or differs from the release')

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if path == helper and not writing:
                return
            if not any(path.is_relative_to(base) for base in allowed) and path != Path('/dev/null'):
                raise RuntimeError('Offline validation: access outside package/runtime forbidden: ' + str(path))
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('Offline validation: network access forbidden')

    sys.addaudithook(audit)
    started = time.monotonic()
    hashes = verify_hash_manifest(ROOT)
    with (output / 'UNIT_TESTS.txt').open('w') as stream:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Unit tests failed; inspect UNIT_TESTS.txt')
    import verify_results
    with (output / 'DERIVED_VERIFICATION.txt').open('w') as stream:
        with contextlib.redirect_stdout(stream):
            verify_results.verify(ROOT / 'results', output / 'derived')
    numerical = json.loads((output / 'derived' / 'VERIFICATION.json').read_text())
    status = {
        'status': 'PASS_PACKAGED_DERIVED_VERIFICATION_ONLY',
        'validation_scope': 'unit contracts, relative hashes, all Cartesian/disjoint values, registered aggregation, saved node draws/estimates and XAI omissions',
        'unit_tests_run': result.testsRun, 'package_hashes_verified': hashes,
        'network_access': 'disabled', 'historical_workspace_access': 'blocked',
        'raw_archive_replay': 'NOT_RUN_EXTERNAL_ARCHIVES_NOT_INCLUDED',
        'raw_replay_verified': False,
        'numerical_checks': numerical, 'elapsed_seconds': time.monotonic() - started,
    }
    (output / 'VALIDATION.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dependency', type=Path, action='append', default=[])
    args = parser.parse_args()
    run(args.output, args.dependency)
