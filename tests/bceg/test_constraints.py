"""Constraints on the conformer search: selection by SMARTS / element, resolution, CREST and GOAT inputs."""

from __future__ import annotations

import json
import math
import stat
import sys
import textwrap

import numpy as np
import pytest

from bceg import constraints as cons
from bceg.chem import embed
from bceg.search.crest import Crest, to_toml
from bceg.search.goat import Goat
from bceg.search.rdkit_etkdg import RDKitETKDG
from bceg.transforms import BUILTIN_TRANSFORMS, Transformation
from bceg.xyz import BOHR_TO_ANGSTROM

@pytest.fixture(scope="module")
def nih2():
    """bipy-NiH2, its start geometry and the indices of the Ni, its two N donors and two hydrides (read
    from the molecule: the atom order is the transformation's business)."""
    structure = Transformation("NiH2", BUILTIN_TRANSFORMS / "ni_x2.py", {"x": "H"}).apply("c1ccc(-c2ccccn2)nc1")
    (ni,) = [a.GetIdx() for a in structure.mol.GetAtoms() if a.GetSymbol() == "Ni"]
    shell = [a for a in structure.mol.GetAtomWithIdx(ni).GetNeighbors()]
    n = sorted(a.GetIdx() for a in shell if a.GetSymbol() == "N")
    h = sorted(a.GetIdx() for a in shell if a.GetSymbol() == "H")
    assert len(n) == 2 and len(h) == 2 and len(shell) == 4
    return structure, embed(structure.mol, 1, ff="none"), ni, n, h


def test_freeze_metal_and_first_shell(nih2):
    structure, coords, ni, n, h = nih2
    shell = sorted([ni, *n, *h])
    for entry in ({"element": "Ni", "neighbors": 1}, {"smarts": "[Ni]", "neighbors": 1}, {"element": ["Ni", "Pd"], "neighbors": 1}):
        r = cons.resolve({"freeze": [entry]}, structure.mol, coords)
        assert r["freeze"] == shell and not r["bonds"]
    assert cons.resolve({"freeze": ["[Ni]"]}, structure.mol, coords)["freeze"] == [ni]
    r = cons.resolve({"freeze": ["[Ni]~[#7]"]}, structure.mol, coords)   # every atom of every match
    assert r["freeze"] == sorted([ni, *n]) and r["labels"] == {str(i): f"{structure.elements[i]}{i}" for i in [ni, *n]}


def test_bonds_angles_dihedrals_one_per_match_at_the_start_value(nih2):
    structure, coords, ni, n, _ = nih2
    r = cons.resolve({"bonds": ["[Ni]~[#7]"], "angles": ["[#7]~[Ni]~[#7]"],
                      "dihedrals": [{"smarts": "[#7:1]~[c:2]~[c:3]~[#7:4]", "value": 0}]}, structure.mol, coords)
    assert sorted(sorted(b["atoms"]) for b in r["bonds"]) == sorted(sorted([ni, i]) for i in n)   # one per Ni-N
    for b in r["bonds"]:
        assert b["value"] == pytest.approx(np.linalg.norm(coords[b["atoms"][0]] - coords[b["atoms"][1]]), abs=1e-4)
    (a,) = r["angles"]                                                 # N-Ni-N and its mirror are one constraint
    assert a["atoms"][1] == ni and sorted(a["atoms"][::2]) == n
    assert a["value"] == pytest.approx(cons.angle(coords, *a["atoms"]), abs=1e-4)
    (d,) = r["dihedrals"]                                              # mapped atoms only, in map order
    assert {d["atoms"][0], d["atoms"][3]} == set(n) and d["value"] == 0.0 and d["force_constant"] is None


def test_force_constant_top_level_per_entry_and_restrained_groups(nih2):
    structure, coords, *_ = nih2
    r = cons.resolve({"force_constant": 0.5, "bonds": ["[Ni]~[#7]", {"smarts": "[Ni]~[#1]", "force_constant": 2}]},
                     structure.mol, coords)
    assert sorted(b["force_constant"] for b in r["bonds"]) == [0.5, 0.5, 2.0, 2.0]
    # a frozen group with a force constant: every distance within the group, nothing frozen exactly
    r = cons.resolve({"freeze": [{"element": "Ni", "neighbors": 1, "force_constant": 0.3}]}, structure.mol, coords)
    assert r["freeze"] == [] and len(r["bonds"]) == 10 and {b["force_constant"] for b in r["bonds"]} == {0.3}


def test_constraints_among_frozen_atoms_are_dropped(nih2):
    structure, coords, *_ = nih2
    r = cons.resolve({"freeze": [{"element": "Ni", "neighbors": 1}], "bonds": ["[Ni]~*", "[#7]~[#6]"],
                      "angles": ["[#7]~[Ni]~[#7]"]}, structure.mol, coords)
    assert r["bonds"] and all(not set(b["atoms"]) <= set(r["freeze"]) for b in r["bonds"]) and not r["angles"]


def test_selection_matching_nothing(nih2):
    structure, coords, *_ = nih2
    with pytest.raises(ValueError, match="matches no atom"):
        cons.resolve({"freeze": ["[Pd]"]}, structure.mol, coords)
    assert cons.is_empty(cons.resolve({"freeze": [{"smarts": "[Pd]", "optional": True}]}, structure.mol, coords))
    with pytest.raises(ValueError, match=r"matches with any bond type, '\[#28\]~\*'"):   # Ni-L bonds are dative
        cons.resolve({"freeze": ["[Ni]-*"]}, structure.mol, coords)
    with pytest.raises(ValueError, match="matches no atom") as info:                       # no bond to relax
        cons.resolve({"bonds": ["[Ni]-[Pd]"]}, structure.mol, coords)
    assert "any bond" not in str(info.value)


@pytest.mark.parametrize("spec, message", [
    ({"freez": []}, "unknown keys"),
    ({"freeze": "[Ni]"}, "expected a list"),
    ({"freeze": [{"smarts": "[Ni"}]}, "not a valid SMARTS"),
    ({"freeze": [{"element": "Xx"}]}, "unknown element"),
    ({"freeze": [{"element": "Ni", "smarts": "[Ni]"}]}, "exactly one of"),
    ({"freeze": [{"element": "Ni", "neighbors": -1}]}, "non-negative"),
    ({"freeze": [{"element": "Ni", "value": 1.0}]}, "not freeze"),
    ({"bonds": ["[Ni]"]}, "selecting 2 atoms"),
    ({"angles": ["[Ni:1]~[#7:2]"]}, "selecting 3 atoms"),
    ({"bonds": [{"smarts": "[Ni]~[#7]", "neighbors": 1}]}, "freeze only"),
    ({"force_constant": 0}, "positive number"),
    ({"bonds": [{"smarts": "[Ni]~[#7]", "force_constant": "strong"}]}, "positive number"),
])
def test_invalid_constraints_are_refused_when_the_searcher_is_built(spec, message):
    with pytest.raises(ValueError, match=message):
        Crest(constraints=spec)


def test_unconstrained_searchers_keep_their_hash():
    assert "constraints" not in Crest().compared_params() and "constraints" not in Goat().compared_params()
    assert Crest(constraints={"freeze": ["[Ni]"]}).compared_params()["constraints"] == {"freeze": ["[Ni]"]}
    assert not RDKitETKDG.supports_constraints() and Crest.supports_constraints() and Goat.supports_constraints()
    with pytest.raises(ValueError, match="unknown parameter"):
        RDKitETKDG(constraints={"freeze": ["[Ni]"]})


def test_crest_toml(nih2):
    structure, coords, ni, _, h = nih2
    crest = Crest(constraints={"freeze": [{"element": "Ni"}], "bonds": [{"smarts": "[Ni]~[#1]", "value": 1.5, "force_constant": 0.5}],
                               "angles": [{"smarts": "[#7]~[Ni]~[#7]", "value": 90}]})
    doc = crest.build_input(structure, crest.resolve_constraints(structure, coords))
    calc = doc["calculation"]
    assert calc["freeze"] == str(ni + 1)                                       # CREST counts from 1
    bonds = [c for c in calc["constraints"] if c["type"] == "bond"]
    assert [b["atoms"] for b in bonds] == [sorted([ni + 1, i + 1]) for i in h]   # i-j and j-i: the lower first
    assert bonds[0]["ref"] == pytest.approx(1.5 / BOHR_TO_ANGSTROM, abs=1e-6) and bonds[0]["fc"] == 0.5   # Bohr
    (angle,) = [c for c in calc["constraints"] if c["type"] == "angle"]
    assert angle["ref"] == pytest.approx(math.pi / 2, abs=1e-6) and "fc" not in angle                     # radians, CREST's k
    text = to_toml(doc)
    assert f'freeze = "{ni + 1}"' in text and text.count("[[calculation.constraints]]") == 3
    assert "constraints" not in Crest().build_input(structure)["calculation"]


def test_crest_md_settings_are_crests_unless_given(nih2):
    """shake, tstep and crossing: CREST's own behaviour by default, frozen atoms or not; out of the
    hash until set, so existing ensembles keep theirs."""
    structure, coords, *_ = nih2
    def doc(**params):
        crest = Crest(**params)
        return crest.build_input(structure, crest.resolve_constraints(structure, coords))
    assert "dynamics" not in doc() and "dynamics" not in doc(constraints={"freeze": ["[Ni]"]})
    assert Crest().command("crest") == Crest(constraints={"freeze": ["[Ni]"]}).command("crest") == ["crest", "--input", "input.toml"]
    assert not {"shake", "tstep", "crossing"} & set(Crest().compared_params())
    frozen = dict(constraints={"freeze": ["[Ni]"]}, shake=0, tstep=2, crossing=False)
    assert doc(**frozen)["dynamics"] == {"shake": 0, "tstep": 2.0}
    assert Crest(**frozen).command("crest")[-1] == "-nocross"
    assert {k: Crest(**frozen).compared_params()[k] for k in ("shake", "tstep", "crossing")} == {"shake": 0, "tstep": 2.0, "crossing": False}
    assert doc(shake=0, toml_extra={"dynamics": {"shake": 1}})["dynamics"] == {"shake": 1}   # toml_extra last


@pytest.mark.parametrize("params", [{"shake": 3}, {"shake": True}, {"tstep": 0}, {"tstep": "2"}, {"crossing": None}, {"crossing": "no"}])
def test_crest_md_settings_are_checked(params):
    with pytest.raises((ValueError, TypeError)):
        Crest(**params)


def test_goat_geom_block_and_exact_only(nih2):
    structure, coords, ni, _, h = nih2
    goat = Goat(constraints={"freeze": ["[Ni]"], "bonds": [{"smarts": "[Ni]~[#1]", "value": 1.45}]})
    text = goat.build_input(structure, goat.resolve_constraints(structure, coords))
    geom = text.split("%geom")[1].split("* xyzfile")[0]
    i, j = sorted([ni, h[0]])
    assert f"{{ C {ni} C }}" in geom and f"{{ B {i} {j} 1.4500 C }}" in geom   # ORCA counts from 0
    assert "%geom" not in Goat().build_input(structure)
    with pytest.raises(ValueError, match="exact"):
        Goat(constraints={"freeze": ["[Ni]"], "force_constant": 1.0})


def test_crest_run_writes_constraints_and_records_them(nih2, tmp_path, monkeypatch):
    """Through Crest.run with a stand-in crest that returns its input as the ensemble."""
    structure, _, ni, n, h = nih2
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "crest"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import shutil, sys
        if "--version" in sys.argv:
            print("Version 3.0.2"); sys.exit(0)
        open("argv.txt", "w").write(" ".join(sys.argv[1:]))
        shutil.copy("input.xyz", "crest_conformers.xyz")
        """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    crest = Crest(start_preopt="none", start_ff="none", threads=1, constraints={"freeze": [{"element": "Ni", "neighbors": 1}]},
                  crossing=False)
    workdir = tmp_path / "run"
    workdir.mkdir()
    result = crest.run(structure, 1, workdir)
    shell = sorted([ni, *n, *h])
    assert result.details["constraints"]["freeze"] == shell
    assert f'freeze = "{",".join(str(i + 1) for i in shell)}"' in (workdir / "input.toml").read_text()
    assert (workdir / "argv.txt").read_text() == "--input input.toml -nocross"
    json.dumps(result.details)   # recorded in run_<i>.json as it is
    free = tmp_path / "free"
    free.mkdir()
    Crest(start_preopt="none", start_ff="none", threads=1).run(structure, 1, free)
    assert (free / "argv.txt").read_text() == "--input input.toml"
