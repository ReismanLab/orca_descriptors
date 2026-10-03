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

NI, N1, N2, H1, H2 = 4, 3, 5, 16, 17  # bipy-NiH2: the Ni, its two N donors and two hydrides


@pytest.fixture(scope="module")
def nih2():
    structure = Transformation("NiH2", BUILTIN_TRANSFORMS / "ni_x2.py", {"x": "H"}).apply("c1ccc(-c2ccccn2)nc1")
    assert structure.elements[NI] == "Ni" and {structure.elements[i] for i in (N1, N2)} == {"N"}
    return structure, embed(structure.mol, 1, ff="none")


def test_freeze_metal_and_first_shell(nih2):
    structure, coords = nih2
    for entry in ({"element": "Ni", "neighbors": 1}, {"smarts": "[Ni]", "neighbors": 1}, {"element": ["Ni", "Pd"], "neighbors": 1}):
        r = cons.resolve({"freeze": [entry]}, structure.mol, coords)
        assert r["freeze"] == [N1, NI, N2, H1, H2] and not r["bonds"]
    assert cons.resolve({"freeze": ["[Ni]"]}, structure.mol, coords)["freeze"] == [NI]
    r = cons.resolve({"freeze": ["[Ni]~[#7]"]}, structure.mol, coords)   # every atom of every match
    assert r["freeze"] == [N1, NI, N2] and r["labels"] == {str(N1): f"N{N1}", str(NI): f"Ni{NI}", str(N2): f"N{N2}"}


def test_bonds_angles_dihedrals_one_per_match_at_the_start_value(nih2):
    structure, coords = nih2
    r = cons.resolve({"bonds": ["[Ni]~[#7]"], "angles": ["[#7]~[Ni]~[#7]"],
                      "dihedrals": [{"smarts": "[#7:1]~[c:2]~[c:3]~[#7:4]", "value": 0}]}, structure.mol, coords)
    assert [b["atoms"] for b in r["bonds"]] == [[N1, NI], [NI, N2]]   # Ni-N and N-Ni are one constraint
    assert r["bonds"][0]["value"] == pytest.approx(np.linalg.norm(coords[N1] - coords[NI]), abs=1e-4)
    assert [a["atoms"] for a in r["angles"]] == [[N1, NI, N2]]
    assert r["angles"][0]["value"] == pytest.approx(cons.angle(coords, N1, NI, N2), abs=1e-4)
    (d,) = r["dihedrals"]                                              # mapped atoms only, in map order
    assert {d["atoms"][0], d["atoms"][3]} == {N1, N2} and d["value"] == 0.0 and d["force_constant"] is None


def test_force_constant_top_level_per_entry_and_restrained_groups(nih2):
    structure, coords = nih2
    r = cons.resolve({"force_constant": 0.5, "bonds": ["[Ni]~[#7]", {"smarts": "[Ni]-[#1]", "force_constant": 2}]},
                     structure.mol, coords)
    assert sorted(b["force_constant"] for b in r["bonds"]) == [0.5, 0.5, 2.0, 2.0]
    # a frozen group with a force constant: every distance within the group, nothing frozen exactly
    r = cons.resolve({"freeze": [{"element": "Ni", "neighbors": 1, "force_constant": 0.3}]}, structure.mol, coords)
    assert r["freeze"] == [] and len(r["bonds"]) == 10 and {b["force_constant"] for b in r["bonds"]} == {0.3}


def test_constraints_among_frozen_atoms_are_dropped(nih2):
    structure, coords = nih2
    r = cons.resolve({"freeze": [{"element": "Ni", "neighbors": 1}], "bonds": ["[Ni]~*", "[#7]~[#6]"],
                      "angles": ["[#7]~[Ni]~[#7]"]}, structure.mol, coords)
    assert r["bonds"] and all(not set(b["atoms"]) <= set(r["freeze"]) for b in r["bonds"]) and not r["angles"]


def test_selection_matching_nothing(nih2):
    structure, coords = nih2
    with pytest.raises(ValueError, match="matches no atom"):
        cons.resolve({"freeze": ["[Pd]"]}, structure.mol, coords)
    assert cons.is_empty(cons.resolve({"freeze": [{"smarts": "[Pd]", "optional": True}]}, structure.mol, coords))


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
    structure, coords = nih2
    crest = Crest(constraints={"freeze": [{"element": "Ni"}], "bonds": [{"smarts": "[Ni]-[#1]", "value": 1.5, "force_constant": 0.5}],
                               "angles": [{"smarts": "[#7]~[Ni]~[#7]", "value": 90}]})
    doc = crest.build_input(structure, crest.resolve_constraints(structure, coords))
    calc = doc["calculation"]
    assert calc["freeze"] == str(NI + 1)                                       # CREST counts from 1
    bonds = [c for c in calc["constraints"] if c["type"] == "bond"]
    assert [b["atoms"] for b in bonds] == [[NI + 1, H1 + 1], [NI + 1, H2 + 1]]
    assert bonds[0]["ref"] == pytest.approx(1.5 / BOHR_TO_ANGSTROM, abs=1e-6) and bonds[0]["fc"] == 0.5   # Bohr
    (angle,) = [c for c in calc["constraints"] if c["type"] == "angle"]
    assert angle["ref"] == pytest.approx(math.pi / 2, abs=1e-6) and "fc" not in angle                     # radians, CREST's k
    text = to_toml(doc)
    assert 'freeze = "5"' in text and text.count("[[calculation.constraints]]") == 3
    assert "constraints" not in Crest().build_input(structure)["calculation"]


def test_goat_geom_block_and_exact_only(nih2):
    structure, coords = nih2
    goat = Goat(constraints={"freeze": ["[Ni]"], "bonds": [{"smarts": "[Ni]-[#1]", "value": 1.45}]})
    text = goat.build_input(structure, goat.resolve_constraints(structure, coords))
    geom = text.split("%geom")[1].split("* xyzfile")[0]
    assert f"{{ C {NI} C }}" in geom and f"{{ B {NI} {H1} 1.4500 C }}" in geom   # ORCA counts from 0
    assert "%geom" not in Goat().build_input(structure)
    with pytest.raises(ValueError, match="exact"):
        Goat(constraints={"freeze": ["[Ni]"], "force_constant": 1.0})


def test_crest_run_writes_constraints_and_records_them(nih2, tmp_path, monkeypatch):
    """Through Crest.run with a stand-in crest that returns its input as the ensemble."""
    structure, _ = nih2
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "crest"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import shutil, sys
        if "--version" in sys.argv:
            print("Version 3.0.2"); sys.exit(0)
        shutil.copy("input.xyz", "crest_conformers.xyz")
        """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    crest = Crest(start_preopt="none", start_ff="none", threads=1, constraints={"freeze": [{"element": "Ni", "neighbors": 1}]})
    workdir = tmp_path / "run"
    workdir.mkdir()
    result = crest.run(structure, 1, workdir)
    assert result.details["constraints"]["freeze"] == [N1, NI, N2, H1, H2]
    assert 'freeze = "4,5,6,17,18"' in (workdir / "input.toml").read_text()
    json.dumps(result.details)   # recorded in run_<i>.json as it is
