'Render retained evidence using explicit package inputs and current display numbers.'
import argparse
import json
import runpy
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root, output = a.package_root.resolve(), a.output.resolve()
    if output.exists():
        raise ValueError('An empty new output directory is required')
    contract = json.loads((root / 'MANUSCRIPT_INTERFACE.json').read_text())
    stage = output / 'stable_evidence_ids'
    if contract['paper'] != 'return_temperature_comparability':
        raise ValueError('Wrong manuscript interface for this figure entry point')
    entry = root / 'selected_presentation/scripts/build_mixed_evidence_figures.py'
    argv = [str(entry), '--package-root', str(root), '--output-root', str(stage), '--figures', 'return_temperature_comparability:2', 'return_temperature_comparability:3', 'return_temperature_comparability:4']
    sys.path.insert(0, str(entry.parent))
    sys.argv = argv
    runpy.run_path(str(entry), run_name='__main__')
    import re
    outputs = []
    for path in stage.rglob('*'):
        if not path.is_file() or path.suffix not in ('.pdf', '.png', '.svg'):
            continue
        match = re.search('(?:Figure_|fig)([2-5])(?=[_.])', path.name)
        if not match:
            continue
        old = match.group(1)
        new = contract['stable_evidence_to_display'][old]
        name = path.name[:match.start(1)] + str(new) + path.name[match.end(1):]
        target = output / name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('Conflicting display identity: ' + name)
        target.write_bytes(path.read_bytes())
        outputs.append({'stable_evidence_id': old, 'display_figure': new, 'file': name})
    if not outputs:
        raise ValueError('No retained result figures were rendered')
    (output / 'DISPLAY_RECEIPT.json').write_text(json.dumps({'outputs': outputs, 'interface': contract}, indent=2) + '\n')
    print(json.dumps(outputs, indent=2))

if __name__ == '__main__':
    main()
