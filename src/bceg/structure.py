"""The object handed to every conformer searcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rdkit import Chem

from bceg.chem import elements_of, is_organic, mol_from_smiles


@dataclass
class Structure:
    """A (possibly transformed) molecule to sample.

    Atom order is fixed by ``AddHs(MolFromSmiles(smiles))``; if ``coords`` are
    given they must follow that order. All output xyz files use this order.
    """

    smiles: str
    charge: int | None = None
    multiplicity: int = 1
    coords: np.ndarray | None = None  # optional 3D starting geometry (Angstrom)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mol = mol_from_smiles(self.smiles)
        if self.charge is None:
            self.charge = Chem.GetFormalCharge(self.mol)
        if self.coords is not None:
            self.coords = np.asarray(self.coords, dtype=float)
            if self.coords.shape != (self.mol.GetNumAtoms(), 3):
                raise ValueError(f"coords shape {self.coords.shape} does not match {self.mol.GetNumAtoms()} atoms")

    @property
    def elements(self) -> list[str]:
        return elements_of(self.mol)

    @property
    def uhf(self) -> int:
        return self.multiplicity - 1

    @property
    def is_organic(self) -> bool:
        return is_organic(self.elements)

    def identity(self) -> dict[str, Any]:
        """Fields that define the structure (part of the compared hyper-parameters)."""
        return {"smiles": self.smiles, "charge": self.charge, "multiplicity": self.multiplicity,
                "has_input_coords": self.coords is not None}
