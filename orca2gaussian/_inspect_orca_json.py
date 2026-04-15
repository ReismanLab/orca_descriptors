import json
from pathlib import Path

base = Path('/Users/julesschleinitz/Desktop/Code/orca_descriptors/library_pyox_2026_03_03/AATATIAROAAKIN-SNVBAGLBSA-N/conf_1')


def scan(path: Path):
    data = json.loads(path.read_text())
    print(f'\n=== {path.name} ===')
    if isinstance(data, dict):
        print('top:', list(data.keys())[:20])

    mol = data.get('Molecule', {}) if isinstance(data, dict) else {}
    if isinstance(mol, dict):
        print('Molecule keys:', list(mol.keys())[:30])
        if 'Atoms' in mol and isinstance(mol['Atoms'], list) and mol['Atoms']:
            print('Atom[0] keys:', list(mol['Atoms'][0].keys())[:40])

    geoms = data.get('Geometries') if isinstance(data, dict) else None
    if isinstance(geoms, list) and geoms:
        g0 = geoms[0]
        print('Geometries[0] keys:', list(g0.keys())[:50])

    out = []

    def walk(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f'{path}.{k}' if path else k
                lk = k.lower()
                if any(x in lk for x in ['homo','lumo','orbital','eigen','freq','mode','ir','hirsh','mulliken','dipole','polar']):
                    out.append((p, type(v).__name__, len(v) if isinstance(v, (list, dict)) else None))
                walk(v, p)
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], (dict, list)):
                walk(obj[0], f'{path}[0]')

    walk(data)
    seen = set()
    for p, t, n in out:
        if p in seen:
            continue
        seen.add(p)
        size = f' len={n}' if n is not None else ''
        print(f'- {p} ({t}{size})')


for fn in ['singlepoint.property.json', 'frequency.property.json', 'singlepoint.json', 'frequency.json', 'reduction.property.json', 'oxidation.property.json']:
    p = base / fn
    if p.exists():
        scan(p)
