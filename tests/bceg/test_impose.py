"""Imposing constraint values on the start geometry (bceg.impose), and holding the imposed atoms fixed in the
pre-optimisation (xtb ``$fix`` with L-BFGS, ``pre_refinement: constrained``) and in the g-xTB post-optimisation."""

from __future__ import annotations

import numpy as np
import pytest

from bceg import constraints as cons
from bceg import geometry, impose
from bceg.chem import embed, mol_from_smiles
from bceg.search.crest import Crest
from bceg.structure import Structure
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation

NI_PLANE = {"planes": [{"element": "Ni", "neighbors": 1}]}
SM_LINEAR = {"freeze": [{"element": "Sm", "neighbors": 1}], "angles": [{"smarts": "[I:1]~[Sm:2]~[I:3]", "value": 180}]}


@pytest.fixture(scope="module")
def nih2():
    return Transformation("NiH2", BUILTIN_TRANSFORMS / "ni_x2.py", {"x": "H"}).apply("c1ccc(-c2ccccn2)nc1")


def _bond_lengths(mol, coords):
    return np.array([np.linalg.norm(coords[b.GetBeginAtomIdx()] - coords[b.GetEndAtomIdx()]) for b in mol.GetBonds()])


def test_planes_targets_and_held_atoms(nih2):
    mol = nih2.mol
    (ni,) = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Ni"]
    shell = sorted([ni, *(a.GetIdx() for a in mol.GetAtomWithIdx(ni).GetNeighbors())])
    assert cons.targets(NI_PLANE, mol) == {"planes": [shell], "bonds": [], "angles": [], "dihedrals": []}
    assert cons.held_atoms(NI_PLANE, mol) == shell
    resolved = cons.resolve(NI_PLANE, mol, embed(mol, 1, ff="none"))
    assert resolved["planes"] == [shell] and resolved["freeze"] == shell          # the search keeps the plane
    spec = {"angles": [{"smarts": "[#7:1]~[Ni:2]~[#7:3]", "value": 90}, "[#1]~[Ni]~[#1]"]}
    (angle,) = cons.targets(spec, mol)["angles"]                                  # only the entry with a value
    assert angle["value"] == 90.0 and angle["atoms"][1] == ni
    assert len(cons.held_atoms(spec, mol)) == 5                                   # but both name held atoms
    with pytest.raises(ValueError, match="at least 4 atoms"):
        cons.targets({"planes": ["[Ni]"]}, mol)
    with pytest.raises(ValueError, match="not planes"):
        cons.validate({"planes": [{"element": "Ni", "value": 0}]})


def test_ni_first_shell_made_planar_smoothly(nih2):
    mol = nih2.mol
    targets, held = cons.targets(NI_PLANE, mol), cons.held_atoms(NI_PLANE, mol)
    coords = embed(mol, 1, ff="none")
    assert impose.plane_rms(coords, targets["planes"][0]) > 0.3
    steps = []

    def check(c):
        steps.append(c.copy())
        return geometry.connectivity_problems(mol, c, 2.5)

    out, info = impose.impose(mol, coords, targets, held, steps=8, check=check)
    assert impose.plane_rms(out, targets["planes"][0]) < 0.01
    assert len(steps) == 9                                                        # every ramp step, and the end
    rms = [impose.plane_rms(c, targets["planes"][0]) for c in steps[:-1]]
    assert all(b <= a + 1e-6 for a, b in zip(rms, rms[1:]))                       # gradually, not in one jump
    np.testing.assert_allclose(_bond_lengths(mol, out), _bond_lengths(mol, coords), atol=0.15)
    assert set(info) == {"steps", "start", "after_ramp", "final"}


def test_i_sm_i_made_linear_at_constant_bond_lengths():
    mol = mol_from_smiles("I[Sm](I)(<-O1CCCC1)(<-O1CCCC1)")
    targets, held = cons.targets(SM_LINEAR, mol), cons.held_atoms(SM_LINEAR, mol)
    (t,) = targets["angles"]
    coords = embed(mol, 3, ff="none")
    assert cons.angle(coords, *t["atoms"]) < 120
    out, info = impose.impose(mol, coords, targets, held, check=geometry.new_problems(mol, coords, 4.0))
    assert cons.angle(out, *t["atoms"]) == pytest.approx(180.0, abs=0.5)
    assert list(info["after_ramp"].values())[0] > 177                             # the ramp got there, the snap finished
    i, sm, k = t["atoms"]
    for a in (i, k):
        assert np.linalg.norm(out[a] - out[sm]) == pytest.approx(np.linalg.norm(coords[a] - coords[sm]), abs=0.01)


def test_unreachable_targets_and_broken_bonds_are_rejected_not_forced(nih2, monkeypatch):
    mol = nih2.mol
    coords = embed(mol, 1, ff="none")
    targets, held = cons.targets(NI_PLANE, mol), cons.held_atoms(NI_PLANE, mol)
    # a ring C-C bond pulled to 3 A breaks the ring: the connectivity check stops the ramp
    ring = next(b for b in mol.GetBonds() if b.IsInRing() and b.GetBeginAtom().GetSymbol() == b.GetEndAtom().GetSymbol() == "C")
    far = {"planes": [], "angles": [], "dihedrals": [],
           "bonds": [{"atoms": [ring.GetBeginAtomIdx(), ring.GetEndAtomIdx()], "value": 3.0}]}
    with pytest.raises(impose.ImposeError, match="changed the connectivity.*bond broken"):
        impose.impose(mol, coords, far, held, check=lambda c: geometry.connectivity_problems(mol, c, 2.5))
    # a ramp that ends short of the snap tolerance is not snapped
    monkeypatch.setitem(impose.SNAP_TOLERANCE, "planes", 1e-9)
    with pytest.raises(impose.ImposeError, match="beyond the snap tolerance"):
        impose.impose(mol, coords, targets, held, steps=1)


def test_connectivity_problems_flag_an_inverted_stereocentre():
    s = Structure("C[C@H](N)O")
    coords = embed(s.mol, 1)
    assert geometry.connectivity_problems(s.mol, coords, 2.5) == []
    assert any("stereocentre" in p for p in geometry.connectivity_problems(s.mol, coords * [-1, 1, 1], 2.5))


def test_start_geometry_imposes_then_restrains_what_was_imposed_in_the_xtb_preopt(fake_xtb, nih2, tmp_path):
    """The pre-optimisation keeps the plane (dihedrals of 0/180 in xtb's $constrain) but not the atoms'
    positions: the bond lengths relax. Every dihedral avoids a near-linear triple (trans ligands)."""
    crest = Crest(constraints=NI_PLANE, start_constraints=True, start_impose_steps=5)
    coords, info = crest.start_geometry(nih2, seed=1, workdir=tmp_path)
    (group,) = cons.targets(NI_PLANE, nih2.mol)["planes"]
    assert impose.plane_rms(coords, group) < 0.01 and info["imposed"]["steps"] == 5
    assert [p["atoms"] for p in info["imposed"]["preopt_holds"]["planes"]] == [group]
    xcontrol = (tmp_path / "preopt_0" / "xcontrol").read_text().splitlines()
    assert xcontrol[:2] == ["$constrain", "   force constant=2.0"] and xcontrol[-1] == "$end"
    dihedrals = [line.split(":")[1].split(",") for line in xcontrol[2:-1]]
    assert len(dihedrals) == len(group) and all(d[4] in ("0.0", "180.0") for d in dihedrals)
    for d in dihedrals:
        i, j, k, l = (int(v) - 1 for v in d[:4])
        assert {i, j, k, l} <= set(group)
        assert 15 < cons.angle(coords, i, j, k) < 165 and 15 < cons.angle(coords, j, k, l) < 165
    assert (tmp_path / "preopt_0" / "args.txt").read_text().endswith("--input xcontrol")


def test_linear_angle_held_as_a_line_and_ase_constraints():
    mol = mol_from_smiles("I[Sm](I)(<-O1CCCC1)(<-O1CCCC1)")
    targets = cons.targets(SM_LINEAR, mol)
    coords = embed(mol, 3, ff="none")
    x, _ = impose.impose(mol, coords, targets, [], check=geometry.new_problems(mol, coords, 4.0))
    hold = impose.holds(x, targets)
    (line,) = hold["lines"]
    assert not hold["angles"] and line["atoms"] == targets["angles"][0]["atoms"]
    assert impose.xtb_restraints(x, hold) == [f"angle: {','.join(str(i + 1) for i in line['atoms'])},180.0"]

    from ase import Atoms
    from ase.constraints import FixAtoms, FixedLine, FixedPlane, FixInternals

    from bceg.refine.ase_opt import ase_constraints
    built = ase_constraints((), hold)
    assert [type(c) for c in built] == [FixedLine]
    atoms = Atoms(["I", "Sm", "I"], positions=x[line["atoms"]])
    atoms.set_constraint(FixedLine([0, 1, 2], line["direction"]))
    atoms.set_positions(atoms.get_positions() + np.array([[0.3, -0.2, 0.1]] * 3) * [[2], [0], [-1]])
    assert cons.angle(atoms.get_positions(), 0, 1, 2) == pytest.approx(180.0, abs=1e-6)   # moves along the line only
    # a plane plus a line across it, and an internal coordinate on a plane atom: those atoms are fixed outright
    mixed = {"planes": [{"atoms": [0, 1, 2, 3], "normal": [0, 0, 1.0]}], "lines": [{"atoms": [3, 4, 5], "direction": [0, 0.6, 0.8]}],
             "bonds": [{"atoms": [1, 6], "value": 1.5}], "angles": [], "dihedrals": []}
    built = ase_constraints([7], mixed)
    assert isinstance(built[0], FixAtoms) and sorted(built[0].index.tolist()) == [1, 3, 7]
    assert {type(c) for c in built[1:]} == {FixedLine, FixedPlane, FixInternals}


def test_start_constraints_params_and_hash():
    plain = Crest(constraints=NI_PLANE)
    assert "start_constraints" not in plain.compared_params() and "start_impose_steps" not in plain.compared_params()
    imposed = Crest(constraints=NI_PLANE, start_constraints=True)
    assert imposed.compared_params()["start_constraints"] is True and imposed.compared_params()["start_impose_steps"] == 10
    with pytest.raises(ValueError, match="no constraints to impose"):
        Crest(start_constraints=True)
    with pytest.raises(ValueError, match="at least 1"):
        Crest(constraints=NI_PLANE, start_impose_steps=0)
