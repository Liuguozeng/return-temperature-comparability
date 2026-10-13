'Explicit, local release validation and current figure regeneration.'
import argparse
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path

def invoke(entry, args):
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *map(str, args)]
    runpy.run_path(str(entry), run_name='__main__')

def verify(root):
    declared = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ', 1)
        p = root / name
        if name in declared or not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError('Missing, duplicate or invalid member: ' + name)
        if hashlib.sha256(p.read_bytes()).hexdigest() != h:
            raise ValueError('Changed member: ' + name)
        declared[name] = h
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    if actual != set(declared):
        raise ValueError('Undeclared or missing package member')
    return {'identity_verified': len(actual), 'raw_replay': 'NOT_EXECUTED_REQUIRES_LEGAL_INPUTS'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'figures'])
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    root = a.package_root.resolve()
    identity = verify(root)
    contract = json.loads((root / 'RELEASE_INTERFACE.json').read_text())
    if contract['paper'] != 'return_temperature_comparability':
        raise ValueError('Cross-paper release root')
    if a.command == 'verify':
        print(json.dumps(identity, indent=2))
        return
    if a.output is None or a.output.exists():
        raise ValueError('An explicit new empty output directory is required')
    a.output.mkdir(parents=True)
    retained = a.output / 'retained'
    invoke(root / 'draw_current_figures.py', ['--package-root', root, '--output', retained])
    newer = a.output / 'comparison'
    invoke(root / 'comparability/render_comparability.py', ['figures', '--results', root / 'comparability/results', '--output', newer])
    old_main = 5
    outputs = []
    for path in newer.glob('*'):
        if path.suffix not in ('.pdf', '.png', '.svg'):
            continue
        name = re.sub('return_temperature_comparability_Figure_' + str(old_main), 'Figure_2', path.name)
        name = re.sub('return_temperature_comparability_Figure_S', 'Figure_S', name)
        target = a.output / name
        target.write_bytes(path.read_bytes())
        outputs.append(name)
    if not any(n.startswith('Figure_2.') for n in outputs):
        raise ValueError('New core Figure 2 was not generated')
    (a.output / 'REPRODUCTION_RECEIPT.json').write_text(json.dumps({
        'identity': identity, 'core_outputs': outputs, 'retained_directory': 'retained',
        'raw_replay': 'NOT_EXECUTED', 'model_fits': 0}, indent=2) + '\n')

if __name__ == '__main__':
    main()
