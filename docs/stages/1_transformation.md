# Stage 1: `transformation`

Turns each input SMILES into the structure that is actually computed, for example an N,N ligand into
its L-NiH2 complex. Run by bceg at the start of every search task; its result (SMILES, charge,
multiplicity) is written to the method folder's `params.json` as `structure`, and that is where the
ORCA stage reads charge and multiplicity from. The block's `name` is the depth-1 folder of the
ensembles: `<ensembles_dir>/<name>/<InChIKey>/<method>/`. The InChIKey is always the one of the
**input** SMILES, so the same ligand has the same key under every transformation.

The ORCA stage records the product once more, per molecule, as soon as `orcadesc submit` sees it:
`$LIB/<InChIKey>/structure.mol` is the RDKit molecule the transformation built (hydrogens explicit,
atoms in the order of every `input.xyz` of the molecule, dative bonds kept; a start geometry returned by
the script is its 3D block), and `metadata.json` carries it as `structure` (`smiles`, `charge`,
`multiplicity`, `n_atoms`, `has_input_coords`, `mol_file`, `metadata`). `Chem.MolFromMolFile(path,
removeHs=False)` gives the object back (`sanitize=False` for a metal complex RDKit cannot sanitise);
`molecule_registry.json` lists the product SMILES as `transformed_smiles`.

## Keys

| Key | Default | Meaning | Hashed |
|---|---|---|---|
| `name` | `none` | folder name; required when a `script` is given | yes |
| `script` | none | Python file defining `transform(smiles, **params)`; a path relative to the YAML, or the bare name of a script shipped with bceg (`ni_x2.py`) | yes (its SHA-256) |
| `params` | `{}` | keyword arguments passed to `transform` | yes |
| `charge` | formal charge of the product | overrides what the script returns | yes |
| `multiplicity` | `1` | overrides what the script returns; checked against the electron count; used by every later stage (xtb, CREST, GOAT, UMA, ORCA) | yes |

The identity hashed into `params.json` is `{name, script, script_sha256, params, charge, multiplicity}`
(`bceg/transforms.py`, `Transformation.identity`). Editing a script after results exist changes its
SHA-256 and the next submit is refused: give the transformation a new `name`.

## Options

### `none`: the SMILES as it is

```yaml
transformation: none
```

or, to fix the spin state of the input molecule itself:

```yaml
transformation: {name: none, multiplicity: 3}
```

### `ni_x2.py`: N,N ligand to L-Ni(X)2 (shipped with bceg)

Recognises the pyox, box, biox, bilm and bipy scaffolds (checked in that order), binds both N donors
datively to Ni(+2) and adds two X(-1); net charge 0. A ligand matching no scaffold is skipped: the
molecule is recorded `not_applicable` and gets no ensemble and no ORCA job.

| `params` | Values |
|---|---|
| `x` | `H` (L-NiH2) or `F` (L-NiF2) |

```yaml
transformation:
  name: NiH2
  script: ni_x2.py
  params: {x: H}
  multiplicity: 1
```

```yaml
transformation:
  name: NiF2_triplet
  script: ni_x2.py
  params: {x: F}
  multiplicity: 3
```

### Your own script

A file next to the YAML (or anywhere, with an absolute path) defining:

```python
def transform(smiles: str, **params):
    ...
```

The argument is the canonical SMILES of the input molecule. Return one of:

- the product SMILES as a string (dative bonds allowed, e.g. `F[Ni]1(F)<-n2ccccc2-c2ccccn->12`);
- a dict with the fields of `bceg.structure.Structure`: `smiles` (required), `charge`, `multiplicity`,
  `coords` (an optional 3D start geometry in Angstrom, atom order of `AddHs(MolFromSmiles(smiles))`,
  used with `pre_refinement.start_geometry: input`), `metadata` (anything, recorded);
- `None` to skip the molecule (`not_applicable`).

`charge` and `multiplicity` given in the YAML override what the script returns.

```yaml
transformation:
  name: protonated
  script: protonate.py          # next to this YAML
  params: {site: N}
```

## Status

`transformation` in `orcadesc status` is `done` when the script accepted the molecule,
`n/a` when it returned `None` (every later stage is then `n/a` too) and `failed` when it raised; the
error is in the `reason` column.
