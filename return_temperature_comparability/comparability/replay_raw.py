'Optional bounded raw replay. Lawful external archives are never bundled.'
import argparse
import json
import sys
from pathlib import Path

from analysis import (ROOT, Guard, compare, json_write, prepare, sha256,
                      summarize, validate_spec)
from portable_support import missing_external_inputs, resolve_input_paths


def run(manifest_path, spec_path, output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Output must be a new directory under this extracted package')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    spec = json.loads(Path(spec_path).read_text())
    validate_spec(spec)
    missing = missing_external_inputs(manifest, manifest_path)
    if missing:
        json_write(output / 'RAW_REPLAY_STATUS.json', {
            'status': 'NOT_RUN_MISSING_EXTERNAL_INPUTS',
            'raw_replay_verified': False, 'missing_input_ids': missing,
            'message': 'Supply lawfully obtained local archives matching the retained hashes.',
        })
        print('NOT_RUN_MISSING_EXTERNAL_INPUTS: ' + ', '.join(missing))
        return 3
    inputs = resolve_input_paths(manifest, manifest_path)
    for item in manifest['inputs']:
        if sha256(inputs[item['id']]) != item['sha256']:
            raise ValueError('Input hash mismatch: ' + item['id'])
    guard = Guard(output, spec)
    try:
        guard.stage('verify_and_prepare')
        periods, season_audit = prepare(inputs, output, guard)
        guard.stage('full_comparisons')
        primary, disjoint, audit = compare(periods, season_audit, output, guard)
        del periods
        guard.stage('node_bootstrap_and_xai_sensitivity')
        summarize(primary, disjoint, output, spec, guard)
        qualification = audit.groupby(['provider', 'status']).agg(
            pair_seasons=('pair_season_id', 'size'),
            synchronous_starts=('eligible_starts', 'sum'),
            asynchronous_starts=('async_combinations_per_duration', 'sum'),
            disjoint_starts=('disjoint_starts', 'sum')).reset_index()
        qualification.to_csv(output / 'QUALIFICATION_COUNTS.csv', index=False)
        for item in manifest['inputs']:
            if sha256(inputs[item['id']]) != item['sha256']:
                raise RuntimeError('Read-only input changed: ' + item['id'])
        guard.close()
        json_write(output / 'RAW_REPLAY_STATUS.json', {
            'status': 'RAW_COMPUTATION_COMPLETE_NOT_YET_INDEPENDENTLY_VERIFIED',
            'raw_replay_verified': False, 'figures_generated': False,
            'next_step': 'Compare regenerated derived values with the packaged results.',
        })
        return 0
    except BaseException as error:
        guard.checkpoint(type(error).__name__ + ': ' + str(error))
        raise
    finally:
        guard.stop.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--analysis-spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.input_manifest, args.analysis_spec, args.output))
