"""Input molecules: canonicalisation and InChIKey naming (depth-2 folders)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bceg.chem import canonical_smiles, inchikey


@dataclass(frozen=True)
class InputMolecule:
    input_smiles: str
    canonical_smiles: str
    inchikey: str
    name: str | None = None

    @classmethod
    def from_smiles(cls, smiles: str, name: str | None = None) -> InputMolecule:
        can = canonical_smiles(smiles)
        return cls(smiles, can, inchikey(can), name)


def load_inputs(path: str | Path) -> list[InputMolecule]:
    """CSV with a ``smiles`` column and optional ``name``; or a plain text file, one SMILES per line."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "smiles" not in df.columns:
            raise ValueError(f"{path} needs a 'smiles' column")
        names = df["name"] if "name" in df.columns else [None] * len(df)
        rows = [(s, None if pd.isna(n) else str(n)) for s, n in zip(df["smiles"], names)]
    else:
        rows = [(ln.split()[0], None) for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    mols: dict[str, InputMolecule] = {}
    for smi, name in rows:
        m = InputMolecule.from_smiles(smi, name)
        if m.inchikey in mols:
            warnings.warn(f"{smi} duplicates {mols[m.inchikey].input_smiles} (InChIKey {m.inchikey}); skipped")
            continue
        mols[m.inchikey] = m
    return list(mols.values())


def update_index(results_dir: Path, molecules: list[InputMolecule]) -> None:
    """Maintain ``results/molecules.csv`` mapping InChIKey to the input SMILES."""
    path = Path(results_dir) / "molecules.csv"
    new = pd.DataFrame([m.__dict__ for m in molecules], columns=["inchikey", "canonical_smiles", "input_smiles", "name"])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates("inchikey", keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values("inchikey").to_csv(path, index=False)
