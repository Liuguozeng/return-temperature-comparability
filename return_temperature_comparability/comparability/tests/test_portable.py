import json
import tempfile
import unittest
from pathlib import Path

from portable_support import (missing_external_inputs, relative_path,
                              resolve_input_paths, sha256, verify_hash_manifest)


class PortableContracts(unittest.TestCase):
    def test_relative_path_rejects_escape_and_absolute(self):
        for bad in ('../x', '/absolute/x', 'a/../../x', 'a\\b', ''):
            with self.assertRaises(ValueError):
                relative_path(bad)
        self.assertEqual(str(relative_path('results/table.csv')), 'results/table.csv')

    def test_hash_manifest_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'table.csv'
            path.write_text('a,b\n1,2\n')
            manifest = {'files': [{'path': path.name, 'bytes': path.stat().st_size,
                                   'sha256': sha256(path)}]}
            (root / 'SHA256_MANIFEST.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_hash_manifest(root), 1)
            path.write_text('a,b\n2,2\n')
            with self.assertRaises(ValueError):
                verify_hash_manifest(root)

    def test_missing_raw_is_not_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input_manifest.example.json'
            manifest = {'inputs': [{'id': name, 'path': 'absent/' + name}
                        for name in ('predist', 'xai', 'temporal_reader',
                                     'weighted_reader', 'valid_day_carrier')]}
            self.assertEqual(missing_external_inputs(manifest, path), ['predist', 'xai'])
            inputs = resolve_input_paths(manifest, path)
            self.assertTrue(all(p.is_relative_to(Path(folder).resolve()) for p in inputs.values()))

    def test_packaged_scientific_definitions_match_recorded_ast(self):
        import ast
        import hashlib
        root = Path(__file__).resolve().parents[1]
        if not (root / 'PROVENANCE.json').exists():
            self.skipTest('Prepared scientific AST proof is tested after packaging')
        proof = json.loads((root / 'PROVENANCE.json').read_text())['scientific_module']
        nodes = ast.parse((root / 'analysis.py').read_text()).body
        actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(actual, proof['canonical_definition_ast_sha256'])
        self.assertFalse(proof['scientific_algorithm_change'])

    def test_packaged_missing_raw_replay_returns_three(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / 'analysis.py').exists():
            self.skipTest('Scientific package not built; contract preflight only')
        from replay_raw import run
        with tempfile.TemporaryDirectory(dir=root / 'tests') as folder:
            output = Path(folder) / 'raw_check'
            self.assertEqual(run(root / 'input_manifest.example.json',
                                 root / 'analysis_spec.json', output), 3)
            status = json.loads((output / 'RAW_REPLAY_STATUS.json').read_text())
            self.assertFalse(status['raw_replay_verified'])
            self.assertEqual(status['status'], 'NOT_RUN_MISSING_EXTERNAL_INPUTS')
            self.assertNotIn('PASS', json.dumps(status))
