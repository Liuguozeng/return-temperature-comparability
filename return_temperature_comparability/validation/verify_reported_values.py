#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'return_temperature_comparability'
ANCHORS = json.loads('{"PROVIDER_EQUAL_SUMMARY.csv": [{"where": {"endpoint_family": "median", "duration_days": "7.0"}, "column": "provider_equal_estimate", "expected": 1.439, "tolerance": 0.001}, {"where": {"endpoint_family": "median", "duration_days": "90.0"}, "column": "provider_equal_estimate", "expected": 0.635, "tolerance": 0.001}, {"where": {"endpoint_family": "median", "comparison": "7_minus_90_days"}, "column": "provider_equal_estimate", "expected": 0.804, "tolerance": 0.001}, {"where": {"endpoint_family": "heat_weighted", "comparison": "7_minus_90_days"}, "column": "provider_equal_estimate", "expected": 0.847, "tolerance": 0.001}], "SECONDARY_ROBUSTNESS_CONTRASTS.csv": [{"where": {"split": "chronology_holdout", "target": "nonoverlap_complement", "short_days": "7", "long_days": "90"}, "column": "absolute_temperature_error_difference_C", "expected": 1.3487486095023513, "tolerance": 1e-12}, {"where": {"split": "chronology_holdout", "target": "nonoverlap_complement", "short_days": "14", "long_days": "90"}, "column": "absolute_temperature_error_difference_C", "expected": 1.1523780185579202, "tolerance": 1e-12}, {"where": {"split": "organization_holdout", "target": "nonoverlap_complement", "short_days": "7", "long_days": "90"}, "column": "absolute_temperature_error_difference_C", "expected": 1.1950033755532106, "tolerance": 1e-12}, {"where": {"split": "organization_holdout", "target": "nonoverlap_complement", "short_days": "14", "long_days": "90"}, "column": "absolute_temperature_error_difference_C", "expected": 1.0593958056769894, "tolerance": 1e-12}], "EQUAL_DAY_CONTRASTS.csv": [{"where": {"provider": "PreDist", "short_days": "7", "long_days": "90"}, "column": "contrast_c", "expected": 1.22909828921952, "tolerance": 1e-12}, {"where": {"provider": "XAI4HEAT", "short_days": "14", "long_days": "90"}, "column": "contrast_c", "expected": 0.63562738095238, "tolerance": 1e-12}], "NONOVERLAP_CONTRASTS.csv": [{"where": {"provider": "PreDist", "short_days": "7", "long_days": "90"}, "column": "contrast_c", "expected": 0.868108003451659, "tolerance": 1e-12}, {"where": {"provider": "XAI4HEAT", "short_days": "7", "long_days": "90"}, "column": "contrast_c", "expected": -0.0334469696969704, "tolerance": 1e-12}], "PHYSICAL_UNIT_CONTRASTS.csv": [{"where": {"provider": "XAI4HEAT", "physical_unit": "L22", "short_days": "7", "long_days": "90"}, "column": "contrast_c", "expected": 1.75047619047619, "tolerance": 1e-12}, {"where": {"provider": "Empa/NEST", "physical_unit": "Empa/NEST:UMAR:MTE_Net", "short_days": "14", "long_days": "90"}, "column": "contrast_c", "expected": 0.43047619047619, "tolerance": 1e-12}]}')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

checks = []
declared_lines = (ROOT / 'SHA256SUMS').read_text().splitlines()
declared_names = [line.split('  ', 1)[1] for line in declared_lines]
actual_names = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
checks.append((len(declared_names) == len(set(declared_names)), 'unique-hash-identities'))
checks.append((actual_names == set(declared_names), 'exact-payload-allowlist'))
checks.append((all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in declared_names), 'safe-relative-paths'))
for line in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, relative = line.split('  ', 1)
    path = ROOT / relative
    checks.append((path.exists() and sha256(path) == expected, f"hash:{relative}"))

for filename, specs in ANCHORS.items():
    with (ROOT / 'results' / filename).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for spec in specs:
        matches = [row for row in rows if all(row.get(key, '') == value for key, value in spec['where'].items())]
        ok = len(matches) == 1 and abs(float(matches[0][spec['column']]) - float(spec['expected'])) <= float(spec['tolerance'])
        checks.append((ok, f"anchor:{filename}:{spec['where']}:{spec['column']}"))

for contract in sorted((ROOT / 'figures').glob('Figure_*/FIGURE_NUMERIC_CONTRACT.json')):
    spec = json.loads(contract.read_text(encoding='utf-8'))
    plotted = contract.parent / 'plotted_values.csv'
    checks.append((plotted.exists() and sha256(plotted) == spec['plotted_values_sha256'], f"numeric-contract:{contract.parent.name}"))

scope_rows = json.loads((ROOT / 'REPRODUCTION_SCOPE_MAP.json').read_text(encoding='utf-8'))
required_scope_fields = {
    'artifact', 'scientific_role', 'source_version', 'verification_mode',
    'runner', 'required_inputs', 'expected_sha256', 'license_boundary',
}
allowed_modes = {'recomputed', 'hash_verified', 'presentation_derived', 'source_restricted'}
for row in scope_rows:
    artifact = ROOT / row['artifact']
    schema_ok = set(row) == required_scope_fields and row['verification_mode'] in allowed_modes
    hash_ok = artifact.is_file() and sha256(artifact) == row['expected_sha256']
    runner = ROOT / row['runner']
    runner_ok = runner.is_file()
    if row['verification_mode'] == 'recomputed':
        runner_ok = runner_ok and runner.name == 'rerun_submission_results.py'
    checks.append((schema_ok and hash_ok and runner_ok, f"reproduction-scope:{row['artifact']}"))

source_provenance_path = ROOT / 'source_provenance/SOURCE_ACQUISITION_MANIFEST.csv'
with source_provenance_path.open(encoding='utf-8-sig', newline='') as handle:
    source_rows = list(csv.DictReader(handle))
required_source_fields = set(['source_group', 'source_version_or_record', 'official_record_or_endpoint', 'carrier_id', 'access_record', 'source_or_manifest_sha256', 'hash_scope', 'license_or_terms', 'scientific_role', 'raw_redistributed', 'notes'])
private_path_markers = (
    '/' + 'Users' + '/',
    chr(92) + 'Users' + chr(92),
    '工作/' + 'Code' + 'X',
)
source_schema_ok = bool(source_rows) and set(source_rows[0]) == required_source_fields
source_values_ok = all(
    re.fullmatch('[0-9a-f]{64}', row['source_or_manifest_sha256'] or '')
    and row['raw_redistributed'] == 'no'
    and not any(marker in ' '.join(row.values()) for marker in private_path_markers)
    for row in source_rows
)
checks.append((source_schema_ok and source_values_ok, 'source-acquisition-provenance'))



failed = [label for ok, label in checks if not ok]
print(json.dumps({'paper': PAPER, 'checks': len(checks), 'failed': failed}, indent=2))
raise SystemExit(1 if failed else 0)
