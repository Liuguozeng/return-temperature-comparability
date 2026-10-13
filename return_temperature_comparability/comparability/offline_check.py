'Offline unit tests and packaged-derived verification, not raw replay.'
import argparse
import contextlib
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


def run(output):
    from portable_support import verify_hash_manifest
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Output must be under this extracted package')
    if output.exists():
        raise ValueError('Validation output must not already exist')
    output.mkdir(parents=True)
    os.chdir(ROOT)
    tempfile.tempdir = str(output)
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    # Allow package files, the selected interpreter/dependencies, and OS resources.
    # Historical workspace paths are not permitted even if they happen to exist.
    allowed = [ROOT, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
               Path(sys.executable).resolve().parent.parent,
               Path('/System'), Path('/Library'), Path('/usr'), Path('/private/etc'),
               Path('/var/db/timezone').resolve()]

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if not any(path.is_relative_to(base) for base in allowed) and path != Path('/dev/null'):
                raise RuntimeError('Offline validation: access outside package/runtime forbidden')
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
    args = parser.parse_args()
    run(args.output)
