"""ETKDG fallback without chirality enforcement (bceg.chem.embed) and CREST's trial-MD retry."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from rdkit import Chem

from bceg import chem
from bceg.geometry import SEED_STRIDE
from bceg.search import crest as crest_mod
from bceg.search.crest import Crest
from bceg.structure import Structure
from bceg.xyz import Conformer, write_xyz

# L-NiH2 of a C2 bis(oxazoline): RDKit's enforced embedding fails for every seed (nih2 library, 2026-10)
BIOX_NIH2 = "[H-]->[Ni+2]1(<-[H-])<-[N]2=C(C3=[N]->1[C@@H](C(C)C)CO3)OC[C@@H]2C(C)C"
PYOX_NIH2 = "[H-]->[Ni+2]1(<-[H-])<-[N]2=C(c3ccccn->13)OC[C@@H]2C(C)(C)C"


def test_biox_embeds_through_the_fallback_with_the_right_configuration():
    mol = chem.mol_from_smiles(BIOX_NIH2)
    for seed in (1, 2):
        coords = chem.embed(mol, seed, ff="none")
        assert coords.shape == (mol.GetNumAtoms(), 3)
        embedded = Chem.Mol(mol)
        embedded.RemoveAllConformers()
        c = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(coords):
            c.SetAtomPosition(i, [float(v) for v in xyz])
        embedded.AddConformer(c)
        assert chem.stereo_matches(embedded)
        assert chem.stereo_centers(mol) == [(7, "S"), (15, "S")]


def test_fallback_rejects_a_wrong_configuration(monkeypatch):
    mol = chem.mol_from_smiles(BIOX_NIH2)
    monkeypatch.setattr(chem, "stereo_matches", lambda m: False)
    with pytest.raises(RuntimeError, match="wrong configuration"):
        chem.embed(mol, 1, ff="none")


def test_enforced_embedding_still_used_when_it_works(monkeypatch):
    mol = chem.mol_from_smiles(PYOX_NIH2)
    monkeypatch.setattr(chem, "stereo_matches", lambda m: (_ for _ in ()).throw(AssertionError("fallback used")))
    chem.embed(mol, 1, ff="none")


def test_md_retries_is_a_resource_key_not_hashed():
    assert Crest(md_retries=5).compared_params() == Crest().compared_params()
    assert Crest(md_retries=5).recorded_params()["md_retries"] == 5


@pytest.fixture
def crest_harness(monkeypatch, tmp_path):
    """Fake start geometry, executable and CREST process; `calls` records every CREST invocation."""
    structure = Structure("[NH4+]", multiplicity=1)
    coords = np.zeros((structure.mol.GetNumAtoms(), 3))
    seeds = []

    def fake_start(self, structure, seed, workdir):
        seeds.append(seed)
        write_xyz(workdir / "start_geometry.xyz", structure.elements, [Conformer(coords, -1.0)])
        return coords, {"seed": seed, "preopt_energy": -1.0}

    monkeypatch.setattr(Crest, "start_geometry", fake_start)
    monkeypatch.setattr(crest_mod, "find_executable", lambda name: "/bin/crest")
    monkeypatch.setattr(crest_mod, "program_version", lambda cmd: "3.0.2")
    calls = []

    def make_run(outcomes):
        def fake_run(cmd, cwd, threads=1, log_name="crest.out"):
            calls.append(cwd)
            ok = outcomes[len(calls) - 1]
            text = "all good\n" if ok else f"...\n{crest_mod.TRIAL_MD_FAILED} 6 times!\n"
            (cwd / log_name).write_text(text)
            if ok:
                write_xyz(cwd / "crest_conformers.xyz", structure.elements, [Conformer(coords, -2.0)])
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, stdout=text)
        return fake_run

    return structure, seeds, calls, make_run


def test_trial_md_failure_is_retried_from_a_new_start(crest_harness, monkeypatch, tmp_path):
    structure, seeds, calls, make_run = crest_harness
    monkeypatch.setattr(crest_mod, "run", make_run([False, True]))
    result = Crest(threads=1).run(structure, 1, tmp_path)
    assert len(calls) == 2 and calls[1] == tmp_path / "md_retry_1"
    assert seeds == [1, 1 + SEED_STRIDE * Crest.defaults["start_max_attempts"]]
    assert result.details["seed"] == seeds[1]
    assert [r["error"] for r in result.details["md_retries"]] == ["trial MD failed"]
    assert (tmp_path / "crest_md_failed_1.out").exists()
    assert tmp_path / "crest_md_failed_1.out" in result.log_files
    assert tmp_path / "md_retry_1" / "crest.out" in result.log_files


def test_trial_md_failure_gives_up_after_md_retries(crest_harness, monkeypatch, tmp_path):
    structure, seeds, calls, make_run = crest_harness
    monkeypatch.setattr(crest_mod, "run", make_run([False, False, False]))
    with pytest.raises(RuntimeError, match="CREST failed"):
        Crest(threads=1).run(structure, 1, tmp_path)
    assert len(calls) == 3


def test_other_crest_failures_are_not_retried(crest_harness, monkeypatch, tmp_path):
    structure, seeds, calls, make_run = crest_harness

    def fake_run(cmd, cwd, threads=1, log_name="crest.out"):
        calls.append(cwd)
        (cwd / log_name).write_text("segfault\n")
        return subprocess.CompletedProcess(cmd, 139, stdout="segfault\n")

    monkeypatch.setattr(crest_mod, "run", fake_run)
    with pytest.raises(RuntimeError, match="CREST failed"):
        Crest(threads=1).run(structure, 1, tmp_path)
    assert len(calls) == 1
