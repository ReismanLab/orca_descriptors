# Transformations

A transformation turns each (canonicalised) input SMILES into the structure that is actually
sampled, e.g. an N,N ligand `L` into the complex `LNiF2`. Its config name becomes the
depth-1 results folder (`results/NiF2/...`); the depth-2 folder stays the InChIKey of the
**input** SMILES, so the same ligand has the same key under every transformation.

A transformation is a Python file defining

```python
def transform(smiles: str):
    ...
```

which returns one of

- a product SMILES (`str`), dative bonds allowed, e.g. `F[Ni]1(F)<-n2ccccc2-c2ccccn->12`;
- a `dict` with keys of `bceg.structure.Structure`: `smiles` (required), `charge`,
  `multiplicity`, `coords` (optional 3D start geometry, atom order of
  `AddHs(MolFromSmiles(smiles))`), `metadata`;
- `None` to skip this molecule.

The SHA-256 of the script is part of every `params.json`: editing a script after results
exist makes the planner refuse to reuse them (rename the transformation instead).

> The interface will be finalised against the sample NiF2/NiH2 scripts.
