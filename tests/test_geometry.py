from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

from bceg import geometry
from bceg.chem import embed
from bceg.search.base import START_DEFAULTS
from bceg.search.crest import Crest
from bceg.structure import Structure
from bceg.transforms import Transformation

SCRIPT = Path(__file__).parents[1] / "transforms" / "ni_x2.py"


def nix2(x: str, mult: int = 1) -> Structure:
    return Transformation(f"Ni{x}2", SCRIPT, {"x": x}, multiplicity=mult).apply("c1ccc(-c2ccccn2)nc1")


def test_collapsed_hydrides_are_flagged():
    s = nix2("H")
    problems = geometry.coordination_problems(s.mol, embed(s.mol, 1, ff="UFF"), 2.5)
    assert any("clash" in p for p in problems)


def test_decoordinated_nitrogen_is_flagged():
    s = nix2("F")
    coords = embed(s.mol, 1, ff="none")
    ni = s.elements.index("Ni")
    n = next(nb.GetIdx() for nb in s.mol.GetAtomWithIdx(ni).GetNeighbors() if nb.GetSymbol() == "N")
    coords[n] = coords[ni] + 3.0 * (coords[n] - coords[ni]) / np.linalg.norm(coords[n] - coords[ni])
    assert any("decoordinated" in p for p in geometry.coordination_problems(s.mol, coords, 2.5))


def test_organic_geometry_passes():
    s = Structure("CCCCO")
    assert geometry.coordination_problems(s.mol, embed(s.mol, 1), 2.5) == []


def test_start_geometry_preopt_retry_and_spin(fake_xtb, tmp_path, monkeypatch):
    s = nix2("F", mult=3)
    calls = []
    real = geometry.coordination_problems

    def first_fails(mol, coords, cutoff):
        calls.append(coords.copy())
        return ["forced failure"] if len(calls) == 1 else real(mol, coords, 99.0)

    monkeypatch.setattr(geometry, "coordination_problems", first_fails)
    coords, info = Crest().start_geometry(s, seed=7, workdir=tmp_path)
    assert info["attempts"] == 2 and info["seed"] == 7 + geometry.SEED_STRIDE
    assert info["rejected"] == [{"seed": 7, "problems": ["forced failure"]}]
    assert info["preopt"] == "gfn2" and info["preopt_energy"] == -42.0
    assert info["preopt_program"] == str(fake_xtb) and "6.7.1" in info["preopt_version"]
    args = (tmp_path / "preopt_1" / "args.txt").read_text()
    assert "--gfn 2" in args and "--uhf 2" in args and "--chrg 0" in args
    assert (tmp_path / "start_geometry.xyz").exists()


def test_replicates_get_different_start_geometries(fake_xtb, tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, "coordination_problems", lambda *a: [])
    s = nix2("H")
    starts = [Crest().start_geometry(s, seed=r, workdir=tmp_path / str(r))[0] for r in (1, 2, 3)]
    assert not np.allclose(starts[0], starts[1]) and not np.allclose(starts[1], starts[2])


def test_gives_up_after_max_attempts(fake_xtb, tmp_path, monkeypatch):
    monkeypatch.setattr(geometry, "coordination_problems", lambda *a: ["bad"])
    params = {**START_DEFAULTS, "start_max_attempts": 3}
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        geometry.make_start_geometry(nix2("F"), 1, tmp_path, params)


def test_xtb_identity_is_not_cached_across_binaries(fake_xtb, tmp_path, monkeypatch):
    """The recorded path must be the xtb that actually ran, not one seen earlier in the process."""
    monkeypatch.setattr(geometry, "coordination_problems", lambda *a: [])
    _, info = Crest().start_geometry(nix2("H"), seed=1, workdir=tmp_path)
    assert info["preopt_program"] == str(fake_xtb)

    other = tmp_path / "other" / "xtb"          # a second build shadowing the first on PATH
    other.parent.mkdir()
    other.write_text(fake_xtb.read_text().replace("6.7.1 (fake)", "6.6.0 (other)"))
    other.chmod(other.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{other.parent}{os.pathsep}{os.environ['PATH']}")
    _, info = Crest().start_geometry(nix2("H"), seed=1, workdir=tmp_path / "b")
    assert info["preopt_program"] == str(other) and "6.6.0" in info["preopt_version"]


def test_no_preopt_records_no_xtb(tmp_path):
    _, info = Crest(start_preopt="none").start_geometry(nix2("H"), seed=1, workdir=tmp_path)
    assert info["preopt"] == "none" and info["preopt_program"] is None and info["preopt_version"] is None


def test_diversity_is_recorded(fake_xtb, tmp_path):
    _, info = Crest().start_geometry(nix2("F"), seed=1, workdir=tmp_path)
    assert info["preopt_level"] == "normal" and info["rmsd_preopt_vs_embedded"] == 0.0  # fake xtb: no change
    (ni,) = info["metal_geometry"].values()
    assert len(ni["angles_deg"]) == 6 and 0.0 <= ni["tau4"] <= 1.0


def test_preopt_level_is_passed_and_validated(fake_xtb, tmp_path):
    Crest(start_preopt_level="crude").start_geometry(nix2("F"), seed=1, workdir=tmp_path)
    assert "--opt crude" in (tmp_path / "preopt_0" / "args.txt").read_text()
    with pytest.raises(ValueError, match="--opt level"):
        Crest(start_preopt_level="fast").start_geometry(nix2("F"), seed=1, workdir=tmp_path / "x")


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb not installed")
@pytest.mark.parametrize("x", ["F", "H"])
def test_real_xtb_preopt_nix2_bpy(tmp_path, x):
    """Real GFN2-xTB pre-optimisation of L-NiX2 (bpy): coordination check passes; print diversity."""
    s = nix2(x)
    starts = []
    for rep in (1, 2, 3):
        coords, info = Crest().start_geometry(s, seed=rep, workdir=tmp_path / str(rep))
        assert info["attempts"] == 1 and info["rejected"] == []
        starts.append(coords)
        print(x, rep, "rmsd vs embedded", info["rmsd_preopt_vs_embedded"], info["metal_geometry"])
    rmsds = [geometry.heavy_atom_rmsd(s.mol, starts[i], starts[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    print(f"Ni{x}2/bpy replicate start RMSDs (A):", [round(r, 3) for r in rmsds])
