"""The LoQI searcher: CLI flags (checked against upstream loqi 1.1.0), seeding, SDF reading, hints."""

from __future__ import annotations

import os
import stat
import sys
import textwrap

import numpy as np
import pytest

from bceg.external import ProgramNotFound
from bceg.search import SEARCHERS
from bceg.search.loqi import LoQI
from bceg.structure import Structure


@pytest.fixture
def fake_loqi(tmp_path, monkeypatch):
    """A ``loqi`` on PATH: embeds the SMILES with RDKit, atom j at (j, 0, 0) in the input order, and
    writes the records with the atom order reversed (so the searcher has to map them back).

    ``LOQI_FAKE_MODE``: ``ok`` (default), ``fail`` (exit 1, no SDF) or ``skip`` (empty SDF).
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "loqi"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import os, sys
        from rdkit import Chem
        from rdkit.Geometry import Point3D
        args = sys.argv[1:]
        open("args.txt", "w").write(" ".join(args))
        mode = os.environ.get("LOQI_FAKE_MODE", "ok")
        if mode == "fail":
            print("error: no such checkpoint", file=sys.stderr); raise SystemExit(1)
        get = lambda flag: args[args.index(flag) + 1]
        out, n = get("--output"), int(get("--n-confs"))
        with Chem.SDWriter(out) as w:
            if mode == "skip":
                print("skipping: bad smiles", file=sys.stderr); raise SystemExit(0)
            mol = Chem.AddHs(Chem.MolFromSmiles(get("--smiles")))
            mol = Chem.RenumberAtoms(mol, list(range(mol.GetNumAtoms()))[::-1])
            conf = Chem.Conformer(mol.GetNumAtoms())
            for a in mol.GetAtoms():  # the reversed index is where the atom was in the input order
                j = mol.GetNumAtoms() - 1 - a.GetIdx()
                conf.SetAtomPosition(a.GetIdx(), Point3D(float(j), 0.0, 0.0))
            mol.AddConformer(conf)
            mol.SetProp("loqi_model", get("--model")); mol.SetIntProp("loqi_failed", 1)
            for k in range(n):
                mol.SetIntProp("loqi_conformer_id", k); w.write(mol)
        print(f"wrote {{n}} conformers for 1 molecules to {{out}} (1 samples failed)", file=sys.stderr)
    """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("LOQI_FAKE_MODE", raising=False)
    monkeypatch.setenv("LOQI_CACHE_DIR", str(tmp_path / "loqi_cache"))
    return exe


def test_loqi_parameters_resources_and_applicability():
    s = SEARCHERS["loqi"]()
    assert isinstance(s, LoQI) and s.resource_class() == "gpu"
    assert LoQI(device="cpu").resource_class() == "cpu"
    assert set(s.compared_params()) == {"n_confs", "model", "config", "steps", "add_hs", "extra_args", "seed_offset"}
    assert set(s.recorded_params()) == {"device", "batch_atoms", "threads", "executable"}
    assert s.applicable(Structure("CCO")) is None
    assert "organic" in s.applicable(Structure("[Ni]"))
    with pytest.raises(ValueError, match="unknown parameter"):
        LoQI(n_conformers=5)


def test_loqi_passes_the_replicate_seed_and_maps_atoms_back(fake_loqi, tmp_path):
    structure = Structure("CCO")
    result = LoQI(n_confs=3, seed_offset=10).run(structure, 2, tmp_path)
    args = (tmp_path / "args.txt").read_text()
    assert args == "sample --smiles CCO --n-confs 3 --output confs.sdf --model loqi_flow_v1.1.0 --seed 12"
    assert len(result.conformers) == 3
    n = len(structure.elements)
    for c in result.conformers:  # template atom i is back at (i, 0, 0) despite the reversed SDF order
        np.testing.assert_allclose(c.coords, np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]))
        assert c.energy is None
    assert result.details == {"seed": 12, "model": "loqi_flow_v1.1.0", "n_sampled": 3, "n_failed_samples": 1,
                              "checkpoint": str(tmp_path / "loqi_cache" / "loqi_flow_v1.1.0.ckpt")}
    assert list(result.versions) == ["loqi"] and result.versions["loqi"]  # "unknown" here: the package is not installed
    assert result.log_files == [tmp_path / "loqi.out"] and "wrote 3 conformers" in (tmp_path / "loqi.out").read_text()


def test_loqi_optional_flags(fake_loqi, tmp_path):
    s = LoQI(n_confs=2, model="/ckpts/own.ckpt", config="loqi_flow.yaml", device="cuda", steps=20, batch_atoms=3000,
             add_hs=False, extra_args=["--verbose"])
    result = s.run(Structure("C"), 1, tmp_path)
    assert (tmp_path / "args.txt").read_text() == (
        "sample --smiles C --n-confs 2 --output confs.sdf --model /ckpts/own.ckpt --seed 1 --config loqi_flow.yaml "
        "--device cuda --steps 20 --batch-atoms 3000 --no-add-hs --verbose")
    assert result.details["checkpoint"] == "/ckpts/own.ckpt"   # a path: nothing to look up in the cache


def test_loqi_failure_points_at_the_weights_cache(fake_loqi, tmp_path, monkeypatch):
    monkeypatch.setenv("LOQI_FAKE_MODE", "fail")
    with pytest.raises(RuntimeError, match="no such checkpoint") as exc:
        LoQI().run(Structure("CCO"), 1, tmp_path)
    assert "loqi download --model loqi_flow_v1.1.0" in str(exc.value)
    assert f"LOQI_CACHE_DIR={tmp_path / 'loqi_cache'}" in str(exc.value)
    # with the checkpoint cached, the failure is something else: no download advice
    (tmp_path / "loqi_cache").mkdir()
    (tmp_path / "loqi_cache" / "loqi_flow_v1.1.0.ckpt").write_bytes(b"x")
    with pytest.raises(RuntimeError) as exc:
        LoQI().run(Structure("CCO"), 1, tmp_path)
    assert "loqi download" not in str(exc.value)


def test_loqi_empty_output_is_an_error(fake_loqi, tmp_path, monkeypatch):
    monkeypatch.setenv("LOQI_FAKE_MODE", "skip")
    with pytest.raises(RuntimeError, match=r"(?s)no readable conformers.*skipping: bad smiles"):
        LoQI().run(Structure("CCO"), 1, tmp_path)


def test_loqi_needs_the_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProgramNotFound, match="'loqi' not found on PATH.*pip install"):
        LoQI().run(Structure("CCO"), 1, tmp_path)
