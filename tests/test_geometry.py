from __future__ import annotations

import os
import stat
import sys
import textwrap
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


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    """An `xtb` on PATH that returns the input geometry and logs its arguments."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "xtb"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import shutil, sys
        open("args.txt", "w").write(" ".join(sys.argv[1:]))
        shutil.copy(sys.argv[1], "xtbopt.xyz")
        print("          | TOTAL ENERGY               -42.000000000000 Eh   |")
    """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return exe


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
