"""Distance-bound floors around a metal (bceg.chem.metal_bounds_matrix): SmI2·L iodides must not embed on top
of each other (smI2_resample_m7, 2026-11: 15/131 molassembler replicates failed with I...I 0.0 or 0.8 A on
all 10 start attempts)."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from bceg import chem, geometry
from bceg.geometry import SEED_STRIDE
from bceg.search.base import START_DEFAULTS
from bceg.structure import Structure
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation

# replicate 1 of this ligand failed on every attempt (CXLUQCICTIHYBH-LQJZCPKCSA-N/masm_m7_rot30/run_1)
LIGAND = "CO[C@@H](CN(CCN(Cc1ccccc1)C[C@H](O)c1ccccc1)Cc1ccccc1)c1ccccc1"
DONORS = "[$([NX3;!a;!$(N[C,S]=O);!$(Nc)]),$([nX2]),$([OX2;!$(O[a]);$(O[CX4][CX4][NX3,nX2])])]"


def smi2_complex() -> Structure:
    return Transformation("SmI2", BUILTIN_TRANSFORMS / "metal_x.py",
                          {"metal": "Sm", "x": "I", "n_x": 2, "donors": DONORS}, multiplicity=7).apply(LIGAND)


def _bounds(bm: np.ndarray, i: int, j: int) -> tuple[float, float]:
    i, j = min(i, j), max(i, j)
    return bm[j][i], bm[i][j]


def test_organic_molecule_keeps_rdkit_bounds():
    assert chem.metal_bounds_matrix(Structure("CCCCO").mol) is None


def test_floors_around_the_metal():
    s = smi2_complex()
    bm = chem.metal_bounds_matrix(s.mol)
    assert bm is not None
    pt = Chem.GetPeriodicTable()
    sm = s.elements.index("Sm")
    i1, i2 = [i for i, e in enumerate(s.elements) if e == "I"]
    r_sm, r_i = pt.GetRcovalent("Sm"), pt.GetRcovalent("I")
    lo, hi = _bounds(bm, i1, i2)
    assert lo >= chem.METAL_13_FLOOR * 2 * r_i - 1e-6 and hi > lo  # RDKit's own lower bound: 1.0 A
    lo, hi = _bounds(bm, sm, i1)
    assert lo >= chem.METAL_BOND_FLOOR * (r_sm + r_i) - 1e-6 and hi > lo
    # the bound OH: Sm...H through Sm-O-H
    o = next(n.GetIdx() for n in s.mol.GetAtomWithIdx(sm).GetNeighbors()
             if n.GetSymbol() == "O" and n.GetTotalNumHs(includeNeighbors=True) == 1)
    h = next(n.GetIdx() for n in s.mol.GetAtomWithIdx(o).GetNeighbors() if n.GetSymbol() == "H")
    assert _bounds(bm, sm, h)[0] >= chem.METAL_13_FLOOR * (r_sm + pt.GetRcovalent("H")) - 1e-6


def test_smi2_iodides_do_not_collapse():
    """The 10 start-attempt seeds of the failing replicate: before the floors 10/10 had I...I < 1.1 A."""
    s = smi2_complex()
    i1, i2 = [i for i, e in enumerate(s.elements) if e == "I"]
    d = [np.linalg.norm(np.subtract(*chem.embed(s.mol, 1 + a * SEED_STRIDE)[[i1, i2]])) for a in range(10)]
    assert min(d) > 2.0, d
    assert sum(x > 2 * Chem.GetPeriodicTable().GetRcovalent("I") for x in d) >= 8, d  # past the clash check


def _iodide_on_sm(s: Structure, seed: int) -> np.ndarray:
    """A real embedding with one iodide moved onto Sm (distance geometry can trap it there: no gradient)."""
    x = chem.embed(s.mol, seed)
    x[s.elements.index("I")] = x[s.elements.index("Sm")] + 0.1
    return x


def test_collapsed_metal_bond_is_a_problem():
    s = smi2_complex()
    x = _iodide_on_sm(s, 1)
    assert any("metal-ligand bond collapsed" in p for p in geometry.coordination_problems(s.mol, x, 3.6))
    assert any("collapsed embedding" in p for p in geometry.collapsed_pairs(s.mol, x))
    assert geometry.collapsed_pairs(s.mol, chem.embed(s.mol, 1)) == []


def test_collapsed_embedding_is_rejected_before_the_preopt(tmp_path, monkeypatch):
    s = smi2_complex()
    real = chem.embed
    monkeypatch.setattr(geometry, "embed", lambda mol, seed, ff: _iodide_on_sm(s, seed) if seed == 1 else real(mol, seed, ff=ff))
    params = {**START_DEFAULTS, "start_preopt": "none", "max_metal_ligand_distance": 3.6,
              "start_allow_decoordination": True}
    _, info = geometry.make_start_geometry(s, 1, tmp_path, params)
    assert info["attempts"] == 2 and info["seed"] == 1 + SEED_STRIDE
    assert info["rejected"][0]["stage"] == "embedded"
    assert "collapsed embedding" in info["rejected"][0]["problems"][0]
