"""Stage 5, ``pruning.method`` (desc_utils.submit.pruning): config validation, each method on small geometries,
and the prepare task recording what a method decided."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bceg.chem import embed
from bceg.inputs import InputMolecule
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_KCAL, Conformer

from desc_utils.submit import pruning, records
from desc_utils.submit.config import Config
from desc_utils.submit.tasks import prepare_molecule
from tests.test_submit import _ensemble, _frozen, _rewrite, fake_sbatch, tree  # noqa: F401  (fixtures)

KCAL = 1 / HARTREE_TO_KCAL


def _structure(smiles):
    s = Structure(InputMolecule.from_smiles(smiles).canonical_smiles)
    return s, s.elements


def _moved(x, seed=0):
    """The same geometry rotated and translated: every method must see it as identical."""
    return Rotation.random(random_state=seed).apply(x) + np.array([3.0, -1.0, 2.0])


def _confs(geoms, rel_kcal):
    return [Conformer(np.asarray(g, float), None if e is None else -100.0 + e * KCAL) for g, e in zip(geoms, rel_kcal)]


# --------------------------------------------------------------------------- config
def test_method_null_keeps_the_hash_and_a_method_fills_its_defaults(tree):
    for block in ({"energy_window_kcal_mol": 5.0}, {"energy_window_kcal_mol": 5.0, "method": None}):
        _rewrite(tree, pruning=block)
        cfg = Config.load(tree["cfg"])
        assert cfg.pruning_method is None and cfg.compared()["pruning"] == {"energy_window_kcal_mol": 5.0}
    _rewrite(tree, pruning={"energy_window_kcal_mol": 5.0, "method": "cregen", "cregen": {"rthr_angstrom": 0.2}})
    cfg = Config.load(tree["cfg"])
    assert cfg.pruning_method == "cregen"
    assert cfg.pruning_params == {**pruning.DEFAULTS["cregen"], "rthr_angstrom": 0.2}
    assert cfg.compared()["pruning"]["cregen"]["ethr_kcal_mol"] == 0.05
    _rewrite(tree, pruning={"method": "rmsd"})
    assert Config.load(tree["cfg"]).pruning_params["max_automorphisms"] == 10000


@pytest.mark.parametrize("block, match", [
    ({"method": "kmedoids"}, "pruning.method must be"),
    ({"rmsd": {"threshold_angstrom": 0.01}}, "without pruning.method"),
    ({"method": "rmsd", "tfd": {"threshold": 0.1}}, "is also given"),
    ({"method": "rmsd", "rmsd": {"cutoff": 0.1}}, "unknown keys"),
    ({"method": "rmsd", "rmsd": {"threshold_angstrom": 0.05}}, "set rmsd.energy_gate_kcal_mol explicitly"),
    ({"method": "tfd", "tfd": {"threshold": 1.5}}, "TFD"),
    ({"method": "tfd"}, "has no default"),
    ({"method": "marc", "marc": {"metric": "tanimoto"}}, "pruning.marc.metric"),
    ({"method": "marc", "marc": {"cluster": "affprop", "n_clusters": 5}}, "affprop"),
    ({"method": "cregen", "cregen": {"mirror": "yes"}}, "true or false"),
])
def test_bad_method_blocks_are_refused(tree, block, match):
    _rewrite(tree, pruning=block)
    with pytest.raises(ValueError, match=match):
        Config.load(tree["cfg"])


def test_a_wide_rmsd_threshold_is_accepted_with_an_explicit_gate(tree):
    _rewrite(tree, pruning={"method": "rmsd", "rmsd": {"threshold_angstrom": 0.1, "energy_gate_kcal_mol": None}})
    assert Config.load(tree["cfg"]).pruning_params["energy_gate_kcal_mol"] is None


# --------------------------------------------------------------------------- methods
def test_rmsd_collapses_a_moved_copy_onto_the_lower_conformer():
    s, el = _structure("CCCCO")
    a, b = embed(s.mol, 1), embed(s.mol, 7)
    confs = _confs([_moved(a), a, b], [0.05, 0.0, 1.0])
    out, summary = pruning.apply("rmsd", pruning.DEFAULTS["rmsd"], el, confs, names=["c1", "c2", "c3"])
    assert [o[0] for o in out] == ["skipped_duplicate", "prepared", "prepared"]
    assert out[0][1]["duplicate_of"] == "c2" and out[0][1]["rmsd_to_leader_angstrom"] < 1e-4
    assert summary["n_kept"] == 2 and summary["n_in"] == 3


def test_rmsd_without_energies_keeps_the_first_of_the_ensemble():
    s, el = _structure("CCO")
    a = embed(s.mol, 1)
    out, _ = pruning.apply("rmsd", pruning.DEFAULTS["rmsd"], el, _confs([a, _moved(a)], [None, None]), names=["c1", "c2"])
    assert [o[0] for o in out] == ["prepared", "skipped_duplicate"]


def test_bthrerf_runs_from_bthr_to_bthrmax_with_the_anisotropy():
    assert pruning.bthrerf(0.01, 0.0, 0.025, 0.5) == pytest.approx(0.01, abs=2e-5)
    assert pruning.bthrerf(0.01, 2.0, 0.025, 0.5) == pytest.approx(0.025, abs=2e-5)


def test_cregen_duplicates_rotamers_and_the_energy_threshold():
    """Ethanol: a moved copy within ethr is a duplicate; the same copy 1 kcal/mol higher is another conformer;
    two methyl hydrogens swapped (RMSD well above rthr, same rotational constants) is a rotamer."""
    s, el = _structure("CCO")
    a = embed(s.mol, 1)
    c_methyl = next(at.GetIdx() for at in s.mol.GetAtoms()
                    if at.GetSymbol() == "C" and sum(n.GetSymbol() == "H" for n in at.GetNeighbors()) == 3)
    h = [n.GetIdx() for n in s.mol.GetAtomWithIdx(c_methyl).GetNeighbors() if n.GetSymbol() == "H"]
    swapped = a.copy()
    swapped[[h[0], h[1], h[2]]] = a[[h[1], h[2], h[0]]]
    confs = _confs([a, _moved(a), _moved(a, 3), swapped], [0.0, 0.01, 1.0, 0.02])
    out, summary = pruning.apply("cregen", pruning.DEFAULTS["cregen"], el, confs, names=["c1", "c2", "c3", "c4"])
    assert [o[0] for o in out] == ["prepared", "skipped_duplicate", "prepared", "skipped_duplicate"]
    assert out[1][1] == {"duplicate_of": "c1", "cregen": "duplicate", "rmsd_to_leader_angstrom": pytest.approx(0, abs=1e-4)}
    assert out[3][1] == {"duplicate_of": "c1", "cregen": "rotamer"}
    assert summary["n_duplicates"] == 1 and summary["n_rotamers"] == 1
    keep_rot = {**pruning.DEFAULTS["cregen"], "drop_rotamers": False}
    assert pruning.apply("cregen", keep_rot, el, confs, names=["c1", "c2", "c3", "c4"])[0][3][0] == "prepared"


def test_cregen_mirror_images():
    s, el = _structure("C[C@H](N)CC(=O)O")
    a = embed(s.mol, 2)
    mirror = a * np.array([-1.0, 1.0, 1.0])
    confs = _confs([a, mirror], [0.0, 0.0])
    assert pruning.apply("cregen", pruning.DEFAULTS["cregen"], el, confs)[0][1][0] == "skipped_duplicate"
    no_mirror = {**pruning.DEFAULTS["cregen"], "mirror": False, "drop_rotamers": False}
    assert pruning.apply("cregen", no_mirror, el, confs)[0][1][0] == "prepared"


def test_cregen_needs_energies():
    s, el = _structure("CCO")
    a = embed(s.mol, 1)
    with pytest.raises(RuntimeError, match="needs an energy"):
        pruning.apply("cregen", pruning.DEFAULTS["cregen"], el, _confs([a, a], [0.0, None]))


def test_tfd_sees_torsions_not_orientation():
    """Butane: a moved copy has TFD 0; anti and gauche are far apart in TFD."""
    s, el = _structure("CCCC")
    geoms = [embed(s.mol, seed) for seed in range(1, 30)]
    from rdkit.Chem import rdMolTransforms
    from rdkit import Chem
    m = Chem.Mol(s.mol)

    def dihedral(x):
        conf = Chem.Conformer(m.GetNumAtoms())
        for i, v in enumerate(x):
            conf.SetAtomPosition(i, v.tolist())
        cs = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C"]
        return abs(rdMolTransforms.GetDihedralDeg(conf, *cs))
    anti = next(g for g in geoms if dihedral(g) > 150)
    gauche = next(g for g in geoms if dihedral(g) < 90)
    confs = _confs([anti, _moved(anti), gauche], [0.0, 0.0, 0.1])
    tfd = {**pruning.DEFAULTS["tfd"], "threshold": 0.02}
    out, summary = pruning.apply("tfd", tfd, el, confs, s.mol, names=["c1", "c2", "c3"])
    assert [o[0] for o in out] == ["prepared", "skipped_duplicate", "prepared"]
    assert out[1][1]["tfd_to_leader"] < 1e-3 and summary["n_torsions"] >= 1
    with pytest.raises(RuntimeError, match="needs the ensemble's structure"):
        pruning.apply("tfd", tfd, el, confs, None)


@pytest.mark.skipif(importlib.util.find_spec("navicat_marc") is None, reason="navicat_marc not installed")
def test_marc_keeps_one_conformer_per_cluster():
    s, el = _structure("CCCCO")
    geoms = [embed(s.mol, seed) for seed in range(1, 13)]
    confs = _confs(geoms, list(np.linspace(0, 3, 12)))
    params = {**pruning.DEFAULTS["marc"], "n_clusters": 3, "metric": "rmsd"}
    out, summary = pruning.apply("marc", params, el, confs, names=[f"c{i}" for i in range(1, 13)])
    kept = [i for i, o in enumerate(out) if o[0] == "prepared"]
    assert len(kept) == summary["n_clusters"] == 3
    for o in out:
        if o[0] != "prepared":
            assert o[0] == "skipped_not_representative" and o[1]["marc_representative"] in {f"c{i + 1}" for i in kept}
    # pick: lowest keeps the lowest-energy member of each cluster
    for i in kept:
        members = [j for j, o in enumerate(out) if o[1].get("marc_cluster") == out[i][1]["marc_cluster"]]
        assert i == min(members, key=lambda j: confs[j].energy)


def test_marc_missing_package_is_a_clear_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("navicat_marc"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    s, el = _structure("CCO")
    a = embed(s.mol, 1)
    with pytest.raises(RuntimeError, match="needs navicat_marc"):
        pruning.apply("marc", pruning.DEFAULTS["marc"], el, _confs([a, a, a], [0, 0, 0]))


# --------------------------------------------------------------------------- prepare
def test_prepare_records_what_the_method_decided(tree, fake_sbatch):
    """Window first (conf_4 at +6 is out), then rmsd on the three left: conf_2 is conf_1 moved."""
    s, _ = _structure("CCO")
    a, b = embed(s.mol, 1), embed(s.mol, 5)
    _ensemble(tree["results"], "CCO", rel_kcal=(0.0, 0.1, 1.0, 6.0), geometries=[a, _moved(a), b, a])
    _rewrite(tree, pruning={"energy_window_kcal_mol": 5.0, "method": "rmsd",
                            "rmsd": {"threshold_angstrom": 0.01}})
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    assert prepare_molecule(cfg, key, "CCO") == "prepared"
    meta = records.load_metadata(tree["lib"], key)
    confs = meta["conformers"]
    st = {k: v["status"] for k, v in confs.items()}
    assert st == {"conf_1": "submitted", "conf_2": "skipped_duplicate", "conf_3": "submitted",
                  "conf_4": "skipped_energy_filter"}
    assert confs["conf_2"]["duplicate_of"] == "conf_1"
    assert meta["orca_stage"]["n_to_run"] == 2
    assert meta["orca_stage"]["pruning_method"]["method"] == "rmsd" and meta["orca_stage"]["pruning_method"]["n_kept"] == 2
    assert meta["dft"]["pruning"]["method"] == "rmsd"
    assert not (tree["lib"] / key / "conf_2").exists()
    assert "skipped_duplicate" in records.SKIPPED_CONFORMER
