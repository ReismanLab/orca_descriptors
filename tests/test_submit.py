"""The ORCA stage (desc_utils.submit) against a fake bceg tree and a fake orca.

The tree has two molecules with a finished ensemble (CCO, CC(C)O) and one without (benzene); the
library config is written in the stage layout (transformation / pre_refinement / conformer_search /
post_optimization / pruning / dft) with ``run_bceg: false`` so nothing is computed.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from rdkit import Chem

from bceg.inputs import InputMolecule
from bceg.layout import method_dir
from bceg.provenance import make_params, read_record, write_record
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_KCAL, Conformer, write_xyz

from desc_utils.submit import records, slurm, template
from desc_utils.submit.cli import main
from desc_utils.submit.config import Config
from desc_utils.submit.source import Ensembles
from desc_utils.submit.status import stage_statuses, status_table
from desc_utils.submit.cost import cost_table
from desc_utils.submit.tasks import (allocated_cpus, energy_window, orca_outcome, prepare_molecule, run_orca_conformer,
                                     write_structure)

LOT = "uma-s-1p2"
LIGANDS = "CCO ethanol\nCC(C)O\nc1ccccc1 benzene\n"  # benzene gets no ensemble
LEVEL_OF_THEORY = {
    "opt_geom": True, "freq_calc": True, "nmr_calc": True, "vertical_redox": True,
    "opt_method": "wb97m-d4 def2-tzvp(-f)", "opt_solvent": "", "svp_method": "m06 d4 def2-tzvp",
    "property_solvent": "", "nmr_method": "m06 d4 def2-tzvp", "nbo_run": "NBO NPA", "extra_basis": {"Ni": "def2-tzvp"},
}


def _ensemble(results_dir: Path, smiles: str, rel_kcal=(0.0, 2.0, 10.0), replicates: int = 1,
              shift_kcal: float = 1.0) -> None:
    """Write a finished bceg method folder; replicate i has the energies ``rel_kcal + (i - 1) * shift_kcal``."""
    mol = InputMolecule.from_smiles(smiles)
    structure = Structure(mol.canonical_smiles)
    md = method_dir(results_dir, "none", mol.inchikey, "etkdg")
    md.path.mkdir(parents=True, exist_ok=True)
    n = len(structure.elements)
    sp = make_params({"searcher": "rdkit_etkdg", "params": {"n_confs": 3}, "seed_rule": "x",
                      "transformation": {"name": "none"}, "structure": structure.identity()})
    md.params().write_text(json.dumps(sp))
    md.params(LOT).write_text(json.dumps(make_params({"refiner": "uma", "lot": LOT, "search_params_sha256": sp["sha256"]})))
    for rep in range(1, replicates + 1):
        confs = [Conformer(coords=[[i * 0.1 + k + 10 * rep, 0.0, 0.0] for i in range(n)],
                           energy=-100.0 + (r + (rep - 1) * shift_kcal) / HARTREE_TO_KCAL)
                 for k, r in enumerate(rel_kcal)]
        write_xyz(md.raw(rep), structure.elements, confs)
        write_xyz(md.refined(rep, LOT), structure.elements, confs)
        write_record(md.run_record(rep), {"status": "done", "params_sha256": sp["sha256"], "phase": "search",
                                          "n_conformers": len(confs)})
        write_record(md.run_record(rep, LOT), {"status": "done", "versions": {"fairchem": "x"}, "n_conformers": len(confs)})


def _lib_cfg(**overrides) -> dict:
    raw = {
        "library_dir": "LIB",
        "smiles": "ligands.txt",
        "ensembles_dir": "DATA_bceg",
        "run_bceg": False,
        "transformation": {"name": "none", "multiplicity": 1},
        "pre_refinement": "none",
        "conformer_search": {"name": "etkdg", "searcher": "rdkit_etkdg", "params": {"n_confs": 3}},
        "post_optimization": {"refiner": "uma", "params": {"model": LOT}},
        "pruning": {"energy_window_kcal_mol": 5.0},
        "dft": {"level_of_theory": LEVEL_OF_THEORY},
        "slurm": {"setup": ["echo setup"], "orca": {"time": "01:00:00", "mem_per_cpu": "4G", "ntasks": 2},
                  "ensembles": {"cpu": {"time": "01:00:00"}}},
        "condense": False,
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def tree(tmp_path: Path) -> dict:
    (tmp_path / "ligands.txt").write_text(LIGANDS)
    results = tmp_path / "DATA_bceg"
    for smi in ("CCO", "CC(C)O"):
        _ensemble(results, smi)
    (tmp_path / "lib.yaml").write_text(yaml.safe_dump(_lib_cfg(), sort_keys=False))
    keys = {s: InputMolecule.from_smiles(s).inchikey for s in ("CCO", "CC(C)O", "c1ccccc1")}
    return {"root": tmp_path, "cfg": tmp_path / "lib.yaml", "lib": tmp_path / "LIB", "keys": keys, "results": results}


def _rewrite(tree, **overrides) -> Path:
    """Write lib.yaml again with top-level overrides (deleting a key: pass ``key=None`` via DELETE)."""
    raw = _lib_cfg()
    for k, v in overrides.items():
        if v is DELETE:
            raw.pop(k, None)
        else:
            raw[k] = v
    tree["cfg"].write_text(yaml.safe_dump(raw, sort_keys=False))
    return tree["cfg"]


DELETE = object()


@pytest.fixture
def fake_sbatch(monkeypatch):
    calls = []

    def _sbatch(script, dependency=None):
        calls.append({"script": Path(script), "dependency": dependency})
        return str(1000 + len(calls))

    monkeypatch.setattr(slurm, "sbatch", _sbatch)
    return calls


# --------------------------------------------------------------------------- config / template
def test_config_resolves_paths_and_defaults(tree):
    cfg = Config.load(tree["cfg"])
    assert cfg.library_dir == tree["lib"] and cfg.ensembles_dir == tree["results"] and cfg.smiles_path == tree["root"] / "ligands.txt"
    assert cfg.template_path.name == "calcParam.inp.tmpl" and cfg.template_path.exists()
    assert cfg.slurm["orca"]["ntasks"] == 2 and cfg.slurm["orca"]["nodes"] == 1  # override + default
    assert cfg.orca_on and not cfg.condense and not cfg.run_bceg
    assert cfg.transformation_name == "none" and cfg.method_name == "etkdg" and cfg.lot == LOT and not cfg.has_preopt
    assert cfg.pre_refinement == {"method": "none", "params": {}}
    assert cfg.ewin == 5.0
    compared = cfg.compared()
    assert set(compared) == {"transformation", "pre_refinement", "conformer_search", "post_optimization", "pruning", "dft", "replicates"}
    assert compared["conformer_search"]["params"]["n_confs"] == 3 and compared["conformer_search"]["params"]["prune_rms"] == -1.0
    assert compared["post_optimization"]["lot"] == LOT and compared["post_optimization"]["params"]["fmax"] == 0.05
    assert compared["transformation"]["multiplicity"] == 1 and compared["replicates"] == 1
    assert compared["dft"]["calc_param_template"]["sha256"] == cfg.template_sha256()
    assert cfg.recorded()["run_bceg"] is False and cfg.recorded()["paths"]["ensembles_dir"] == str(tree["results"])
    # bceg's view: one transformation, one method, replicate 1, the library's setup
    b = cfg.bceg_raw()
    assert b["replicates"] == 1 and b["inputs"] == str(tree["root"] / "ligands.txt") and b["results_dir"] == str(tree["results"])
    assert list(b["transformations"]) == ["none"] and list(b["methods"]) == ["etkdg"]
    assert b["methods"]["etkdg"] == {"searcher": "rdkit_etkdg", "params": {"n_confs": 3}, "slurm": {}}   # no start_* for ETKDG
    assert b["slurm"] == {"setup": ["echo setup"], "cpu": {"time": "01:00:00"}}


def test_stage_defaults_and_mapping_to_bceg(tree):
    # a searcher with a start geometry gets xtb GFN2 by default
    cfg = Config.load(_rewrite(tree, pre_refinement=DELETE, pruning=DELETE,
                               conformer_search={"name": "crest_gfn2", "searcher": "crest", "params": {"workhorse": "gfn2"}}))
    assert cfg.pre_refinement == {"method": "xtb", "params": {"gfn": "gfn2", "level": "normal"}} and cfg.has_preopt
    assert cfg.ewin is None and cfg.pruning == {"energy_window_kcal_mol": None}
    params = cfg.bceg_raw()["methods"]["crest_gfn2"]["params"]
    assert params["start_preopt"] == "gfn2" and params["start_preopt_level"] == "normal" and params["workhorse"] == "gfn2"
    assert "start_preopt" not in cfg.compared()["conformer_search"]["params"]      # the pre-refinement block owns it
    # uma pre-refinement and the embedding keys
    cfg = Config.load(_rewrite(tree, conformer_search={"name": "crest_uma", "searcher": "crest"},
                               pre_refinement={"method": "uma", "params": {"model": "uma-m-1p1", "fmax": 0.1},
                                               "max_metal_ligand_distance": 2.8, "max_attempts": 3}))
    params = cfg.bceg_raw()["methods"]["crest_uma"]["params"]
    assert params["start_preopt"] == "uma" and params["start_preopt_params"] == {"model": "uma-m-1p1", "fmax": 0.1}
    assert params["max_metal_ligand_distance"] == 2.8 and params["start_max_attempts"] == 3
    assert cfg.compared()["pre_refinement"]["max_attempts"] == 3
    # no post-optimisation: ORCA refines the raw search output
    cfg = Config.load(_rewrite(tree, post_optimization=None))
    assert cfg.lot is None and cfg.compared()["post_optimization"] is None
    ens = Ensembles(cfg).ensemble(InputMolecule.from_smiles("CCO"))
    assert ens.done and [p.name for p in ens.xyz_paths] == ["raw_1.xyz"]


@pytest.mark.parametrize("override, message", [
    ({"pre_refinement": {"method": "xtb"}}, "no start geometry"),                   # ETKDG has nothing to pre-optimise
    ({"pre_refinement": {"method": "dft"}}, "pre_refinement.method"),
    ({"conformer_search": {"name": "x", "searcher": "rdkit_etkdg", "params": {"n_conf": 3}}}, "unknown parameter"),
    ({"conformer_search": {"name": "x", "searcher": "magic"}}, "conformer_search.searcher"),
    ({"pruning": {"energy_window_kcal_mol": -1}}, "positive number"),
    ({"pruning": {"ewin": 5}}, "unknown keys"),
    ({"transformation": {"script": "ni_x2.py"}}, "`name` is required"),
    ({"post_optimization": {"refiner": "uma", "params": {"modle": "x"}}}, "unknown parameter"),
    ({"dft": {"level_of_theory": {**LEVEL_OF_THEORY, "conf_search": False}}, "prune": 1}, "unknown keys"),
])
def test_config_errors_are_caught_at_load(tree, capsys, override, message):
    path = _rewrite(tree, **override)
    with pytest.raises(ValueError, match=message):
        Config.load(path)
    assert main(["submit", str(path), "--dry-run"]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().err


def test_render_and_compound_steps():
    text = template.render(Config.load.__globals__["DEFAULT_TEMPLATE"], LEVEL_OF_THEORY, charge=0, multiplicity=1,
                           nprocs=8, maxcore_mb=3072)
    assert "$" not in text and "conf_search" not in text and "GOAT" not in text
    assert "%maxcore 3072" in text and "nprocs 8" in text
    assert text.count('NewGTO Ni "def2-tzvp" end') == 2
    opt = text.split('%base "optimization"')[1].split("StepEnd")[0]
    single = text.split('%base "singlepoint"')[1].split("StepEnd")[0]
    assert "%basis" in opt and "%basis" not in single
    steps = template.compound_steps(LEVEL_OF_THEORY, 0, 1)
    assert list(steps) == ["optimization", "frequency", "singlepoint", "nmr", "oxidation", "reduction"]
    assert steps["oxidation"] == {"runs": True, "method": "m06 d4 def2-tzvp", "solvent": "", "population_analysis": "NBO NPA",
                                  "charge": 1, "multiplicity": 2}
    assert steps["frequency"]["extra_basis"] == "Ni def2-tzvp"
    no_eb = dict(LEVEL_OF_THEORY, extra_basis={})
    assert "%basis" not in template.render(Config.load.__globals__["DEFAULT_TEMPLATE"], no_eb, 0, 1, 1, 1000)
    with pytest.raises(KeyError, match="unknown keys"):
        template.validate_level_of_theory({**LEVEL_OF_THEORY, "goat_method": "XTB"})


def test_maxcore():
    assert template.maxcore_mb("4G") == 3072
    assert template.maxcore_mb(None, "32G", ntasks=16) == 1536
    assert template.maxcore_mb("500") == 375
    assert template.maxcore_mb(None, None) == 2000


# --------------------------------------------------------------------------- submit
def test_submit_dry_run_writes_nothing(tree, capsys):
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 0
    assert not (tree["lib"] / "params.json").exists()
    out = capsys.readouterr().out
    assert "3 molecule(s) to prepare" in out and "orca: on" in out
    assert "pipeline: transformation none | pre_refinement none | conformer_search etkdg (rdkit_etkdg, 1 replicate) | " \
           f"post_optimization uma {LOT} | pruning 5 kcal/mol | dft on" in out


def test_submit_freezes_and_queues_prepare(tree, fake_sbatch):
    assert main(["submit", str(tree["cfg"])]) == 0
    lib = tree["lib"]
    params = json.loads((lib / "params.json").read_text())
    assert params["compared"]["pruning"]["energy_window_kcal_mol"] == 5.0 and params["recorded"]["run_bceg"] is False
    jobdirs = list((lib / "_jobs").iterdir())
    assert len(jobdirs) == 1
    jobdir = jobdirs[0]
    frozen = yaml.safe_load((jobdir / "config.yaml").read_text())
    assert frozen["_pipeline"]["repo"] and "git_commit" in frozen["_pipeline"]
    assert frozen["dft"]["calc_param_template"] == str(jobdir / "calcParam.inp.tmpl")
    assert Config.load(jobdir / "config.yaml").compared()["conformer_search"] == Config.load(tree["cfg"]).compared()["conformer_search"]
    tasks = [json.loads(l) for l in (jobdir / "prepare.jsonl").read_text().splitlines()]
    assert [t["inchikey"] for t in tasks] == list(tree["keys"].values())
    script = (jobdir / "prepare.sbatch").read_text()
    assert "#SBATCH --array=0-2%50" in script and "echo setup" in script and "prepare-task" in script
    assert fake_sbatch[0]["script"] == jobdir / "prepare.sbatch" and fake_sbatch[0]["dependency"] is None
    meta = records.load_metadata(lib, tree["keys"]["CCO"])
    assert meta["orca_stage"] == {"status": "queued", "prepare_job_id": "1001", "after_bceg_jobs": [],
                                  "submitted_at": meta["orca_stage"]["submitted_at"]}
    reg = records.load_registry(lib)
    assert reg[tree["keys"]["CCO"]]["smiles"] == "CCO" and reg[tree["keys"]["CCO"]]["smiles_line"] == 0
    assert reg[tree["keys"]["c1ccccc1"]]["name"] == "benzene"
    # a second submit skips everything queued
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 0


def test_changed_stage_setting_is_refused(tree, fake_sbatch, capsys):
    assert main(["submit", str(tree["cfg"])]) == 0
    raw = yaml.safe_load(tree["cfg"].read_text())
    raw["pruning"]["energy_window_kcal_mol"] = 8.0
    tree["cfg"].write_text(yaml.safe_dump(raw))
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "pruning.energy_window_kcal_mol: 5.0 -> 8.0" in err
    raw["pruning"]["energy_window_kcal_mol"] = 5.0
    raw["conformer_search"]["params"]["n_confs"] = 4
    tree["cfg"].write_text(yaml.safe_dump(raw))
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 2
    assert "conformer_search.params.n_confs: 3 -> 4" in capsys.readouterr().err
    # recorded settings may change freely
    raw["conformer_search"]["params"]["n_confs"] = 3
    raw["orca"] = "off"
    raw["slurm"]["orca"]["time"] = "99:00:00"
    raw["slurm"]["ensembles"]["cpu"]["time"] = "48:00:00"
    raw["conformer_search"]["slurm"] = {"cpus_per_task": 4}
    tree["cfg"].write_text(yaml.safe_dump(raw))
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 0


# --------------------------------------------------------------------------- prepare
def _frozen(tree, fake_sbatch) -> Config:
    assert main(["submit", str(tree["cfg"])]) == 0
    jobdir = next((tree["lib"] / "_jobs").iterdir())
    return Config.load(jobdir / "config.yaml")


def test_prepare_prunes_splits_and_submits(tree, fake_sbatch):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    assert prepare_molecule(cfg, key, "CCO") == "prepared"
    mol_dir = tree["lib"] / key
    assert sorted(p.name for p in mol_dir.iterdir() if p.is_dir()) == ["conf_1", "conf_2"]  # conf_3 is 10 kcal/mol up
    inp = (mol_dir / "conf_1" / "calcParam.inp").read_text()
    assert "nprocs 2" in inp and "%maxcore 3072" in inp and "* xyzfile" not in inp.split("%Compound")[0]
    xyz = (mol_dir / "conf_1" / "input.xyz").read_text().splitlines()
    assert xyz[0].strip() == "9" and xyz[1].startswith("-100.0000000000") and "energy=-100.0000000000" in xyz[1]
    from desc_utils.numbering.io import xyz_comment_energy
    assert xyz_comment_energy(mol_dir / "conf_1", "input.xyz", "energy", "hartree") == pytest.approx(-100.0)
    meta = records.load_metadata(tree["lib"], key)
    # the transformation's product: an RDKit mol file in the xyz atom order, and its record
    assert meta["structure"] == {"smiles": "CCO", "charge": 0, "multiplicity": 1, "n_atoms": 9, "has_input_coords": False,
                                 "mol_file": "structure.mol", "metadata": {}}
    saved = Chem.MolFromMolFile(str(mol_dir / "structure.mol"), removeHs=False)
    assert [a.GetSymbol() for a in saved.GetAtoms()] == [line.split()[0] for line in xyz[2:11]]
    assert Chem.MolToSmiles(Chem.RemoveHs(saved)) == "CCO"
    assert meta["dft"]["charge"] == 0 and meta["dft"]["multiplicity"] == 1
    assert meta["dft"]["pruning"] == {"energy_window_kcal_mol": 5.0}
    assert meta["dft"]["compound_steps"]["optimization"]["extra_basis"] == "Ni def2-tzvp"
    assert "conf_search" not in meta["dft"]["compound_steps"]
    assert meta["dft"]["level_of_theory"]["extra_basis"] == {"Ni": "def2-tzvp"}
    assert meta["ensemble"]["params_sha256"] and meta["ensemble"]["lot"] == LOT and meta["ensemble"]["replicates"] == 1
    assert meta["ensemble"]["runs"][0]["replicate"] == 1 and meta["ensemble"]["n_conformers_per_replicate"] == [3]
    assert meta["conformers"]["conf_1"]["replicate"] == 1 and meta["conformers"]["conf_3"]["index_in_replicate"] == 3
    assert meta["conformers"]["conf_3"]["status"] == "skipped_energy_filter"
    assert meta["conformers"]["conf_3"]["energy_above_min_kcal_mol"] == pytest.approx(10.0)
    assert meta["conformers"]["conf_1"]["status"] == "submitted" and meta["conformers"]["conf_1"]["slurm_array_job_id"] == "1002"
    assert meta["orca_stage"]["status"] == "prepared" and meta["orca_stage"]["n_to_run"] == 2
    orca_script = fake_sbatch[-1]["script"].read_text()
    assert "#SBATCH --array=0-1%50" in orca_script and "--signal=B:TERM@600" in orca_script and "orca-task" in orca_script
    assert "#SBATCH --mem-per-cpu=4G" in orca_script and "#SBATCH --ntasks=2" in orca_script
    assert records.molecule_state(meta) == ("in_progress", "2 ORCA job(s) submitted or running")
    # the registry now carries the ensemble provenance
    reg = records.load_registry(tree["lib"])[key]
    assert reg["source"]["method"] == "etkdg" and reg["source"]["params_sha256"] == meta["ensemble"]["params_sha256"]
    assert reg["transformed_smiles"] == "CCO"


def test_structure_mol_keeps_dative_bonds_and_start_geometry(tmp_path):
    from bceg.transforms import BUILTIN_TRANSFORMS, Transformation
    t = Transformation("NiF2", BUILTIN_TRANSFORMS / "ni_x2.py", {"x": "F"}, multiplicity=1)
    structure = t.apply(InputMolecule.from_smiles("c1ccc(-c2ccccn2)nc1").canonical_smiles)
    n = structure.mol.GetNumAtoms()
    structure = Structure(structure.smiles, charge=structure.charge, multiplicity=1,
                          coords=np.arange(3 * n, dtype=float).reshape(n, 3), metadata={"scaffold": "bipy"})
    rec = write_structure(tmp_path, structure)
    assert rec["charge"] == 0 and rec["n_atoms"] == n == 23 and rec["has_input_coords"] and rec["metadata"] == {"scaffold": "bipy"}
    saved = Chem.MolFromMolFile(str(tmp_path / rec["mol_file"]), removeHs=False, sanitize=False)
    assert [a.GetSymbol() for a in saved.GetAtoms()] == structure.elements
    assert sum(b.GetBondType() == Chem.BondType.DATIVE for b in saved.GetBonds()) == 4
    assert saved.GetConformer().GetAtomPosition(1).x == pytest.approx(3.0)


def test_no_pruning_keeps_every_conformer(tree, fake_sbatch):
    confs = [Conformer([[0.0, 0.0, 0.0]], e) for e in (-1.0, -1.0 + 10 / HARTREE_TO_KCAL, None)]
    assert [s for _, _, _, s in energy_window(confs, None)] == ["prepared", "prepared", "prepared"]
    assert [s for _, _, _, s in energy_window(confs, 5.0)] == ["prepared", "skipped_energy_filter", "skipped_no_energy"]
    _rewrite(tree, pruning=DELETE)
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    assert prepare_molecule(cfg, key, "CCO") == "prepared"
    meta = records.load_metadata(tree["lib"], key)
    assert meta["orca_stage"]["n_to_run"] == 3 and meta["dft"]["pruning"] == {"energy_window_kcal_mol": None}
    assert (tree["lib"] / key / "conf_3" / "input.xyz").exists()


def test_prepare_without_ensemble(tree, fake_sbatch):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["c1ccccc1"]
    assert prepare_molecule(cfg, key, "c1ccccc1") == "ensemble_missing"
    meta = records.load_metadata(tree["lib"], key)
    assert meta["orca_stage"]["status"] == "ensemble_missing" and "run_1_uma-s-1p2.json" in meta["orca_stage"]["reason"]
    assert records.molecule_state(meta)[0] == "incomplete"
    assert not (tree["lib"] / key / "conf_1").exists()
    # the product of stage 1 is recorded even before the ensemble exists
    assert meta["structure"]["smiles"] == "c1ccccc1" and (tree["lib"] / key / "structure.mol").exists()


def test_prepare_refuses_changed_ensemble(tree, fake_sbatch):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO")
    _ensemble(tree["results"], "CCO", rel_kcal=(0.0, 1.0))  # bceg re-run with other parameters -> new sha
    md = method_dir(tree["results"], "none", key, "etkdg")
    p = json.loads(md.params().read_text())
    p["sha256"] = "0" * 64
    md.params().write_text(json.dumps(p))
    from bceg.provenance import ParamsMismatch
    with pytest.raises(ParamsMismatch):
        prepare_molecule(cfg, key, "CCO", force=True)
    assert records.load_metadata(tree["lib"], key)["orca_stage"]["status"] == "ensemble_changed"


# --------------------------------------------------------------------------- replicates
def _two_replicates(tree) -> Path:
    search = {"name": "etkdg", "searcher": "rdkit_etkdg", "params": {"n_confs": 3}, "replicates": 2}
    return _rewrite(tree, conformer_search=search)


def test_replicates_are_pooled_in_order(tree, fake_sbatch):
    """conformer_search.replicates: bceg runs N searches; prepare pools raw_1..raw_N (in order) into one
    ensemble, prunes it as a whole and records where every conformer came from."""
    from desc_utils.submit.cli import pipeline_line
    cfg = Config.load(_two_replicates(tree))
    assert cfg.replicates == 2 and cfg.bceg_raw()["replicates"] == 2 and cfg.compared()["replicates"] == 2
    assert "replicates" not in cfg.compared()["conformer_search"]           # hashed once, at the top level
    assert "2 replicates" in pipeline_line(cfg)
    _ensemble(tree["results"], "CCO", replicates=2)                          # rep 2: 1, 3, 11 kcal/mol
    key = tree["keys"]["CCO"]
    ens = Ensembles(cfg).ensemble(InputMolecule.from_smiles("CCO"))
    assert ens.done and [p.name for p in ens.xyz_paths] == [f"raw_1_{LOT}.xyz", f"raw_2_{LOT}.xyz"]
    assert [(c.info["replicate"], c.info["index"]) for c in ens.conformers] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert ens.record["n_conformers"] == 6 and ens.record["n_conformers_per_replicate"] == [3, 3]

    cfg = _frozen(tree, fake_sbatch)
    assert prepare_molecule(cfg, key, "CCO") == "prepared"
    mol_dir = tree["lib"] / key
    assert sorted(p.name for p in mol_dir.iterdir() if p.is_dir()) == ["conf_1", "conf_2", "conf_4", "conf_5"]
    meta = records.load_metadata(tree["lib"], key)
    assert meta["orca_stage"]["n_conformers_in_ensemble"] == 6 and meta["orca_stage"]["n_replicates"] == 2
    assert meta["orca_stage"]["n_to_run"] == 4
    c4 = meta["conformers"]["conf_4"]
    assert c4["replicate"] == 2 and c4["index_in_replicate"] == 1 and c4["energy_above_min_kcal_mol"] == pytest.approx(1.0)
    assert meta["conformers"]["conf_6"]["status"] == "skipped_energy_filter"
    comment = (mol_dir / "conf_4" / "input.xyz").read_text().splitlines()[1]
    assert "replicate=2" in comment and "replicate_index=1" in comment
    assert len(meta["ensemble"]["runs"]) == 2 and meta["ensemble"]["runs"][1]["run_status"] == "done"
    # a library with one replicate and the same settings has a different hash: another library_dir
    one = Config.load(_rewrite(tree)).compared()
    two = Config.load(_two_replicates(tree)).compared()
    assert one != two and {k for k in one if one[k] != two[k]} == {"replicates"}


def test_every_replicate_must_be_finished(tree, fake_sbatch):
    """A replicate without a record is ensemble_missing, a failed one ensemble_failed: what a library
    refines never depends on which bceg jobs happened to be done."""
    cfg = Config.load(_two_replicates(tree))
    source = Ensembles(cfg)
    mol = InputMolecule.from_smiles("CC(C)O")                                 # the tree has replicate 1 only
    md = method_dir(tree["results"], "none", mol.inchikey, "etkdg")
    ens = source.ensemble(mol)
    assert ens.status == "ensemble_missing" and f"run_2_{LOT}.json" in ens.reason and "run_1_" not in ens.reason
    assert ens.record["runs"][1]["run_status"] is None
    write_record(md.run_record(2), {"status": "done", "n_conformers": 3})
    write_record(md.run_record(2, LOT), {"status": "failed", "error": "boom"})
    ens = source.ensemble(mol)
    assert ens.status == "ensemble_failed" and "run_2_" in ens.reason and "boom" in ens.reason
    # status folds the replicates' records into one: failed wins, then running, else done with the sum
    row = status_table(cfg).set_index("inchikey").loc[mol.inchikey]
    assert row["search"] == "done" and row["post_opt"] == "failed" and row["n_conf"] == 6
    write_record(md.run_record(2, LOT), {"status": "running", "phase": "refine"})
    row = status_table(cfg).set_index("inchikey").loc[mol.inchikey]
    assert row["post_opt"] == "running"
    md.run_record(2, LOT).unlink()
    row = status_table(cfg).set_index("inchikey").loc[mol.inchikey]
    assert row["post_opt"] == "-"                                             # one replicate's record is not the stage
    # the prepare task records the block too
    cfg = _frozen(tree, fake_sbatch)
    assert prepare_molecule(cfg, mol.inchikey, "CC(C)O") == "ensemble_missing"
    assert records.load_metadata(tree["lib"], mol.inchikey)["orca_stage"]["status"] == "ensemble_missing"


def test_aggregate_records_rules():
    from desc_utils.submit.source import aggregate_records
    done = {"status": "done", "n_conformers": 2}
    assert aggregate_records([None, None]) is None
    assert aggregate_records([done, None]) is None
    assert aggregate_records([done, {"status": "running", "phase": "search"}])["status"] == "running"
    assert aggregate_records([{"status": "failed", "error": "e"}, None])["status"] == "failed"
    assert aggregate_records([{"status": "running"}, {"status": "failed"}])["status"] == "running"
    assert aggregate_records([{"status": "not_applicable", "reason": "r"}, {"status": "running"}])["status"] == "not_applicable"
    assert aggregate_records([done, {"status": "done", "n_conformers": 5}]) == {"status": "done", "n_conformers": 7}
    assert aggregate_records([{"status": "done"}]) == {"status": "done"}


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_replicates_must_be_a_positive_integer(tree, value):
    search = {"name": "etkdg", "searcher": "rdkit_etkdg", "params": {"n_confs": 3}, "replicates": value}
    with pytest.raises(ValueError, match="replicates must be a positive integer"):
        Config.load(_rewrite(tree, conformer_search=search))


# --------------------------------------------------------------------------- ORCA task
#: what a two-step compound job prints, as far as the timing parser is concerned
FAKE_COMPOUND_OUT = """\
                                COMPOUND JOB MainJOB
|  3>      nprocs 4
                                COMPOUND JOB  1
|  1>       %base "optimization"
Timings for individual modules:

Sum of individual times          ...       10.000 sec (=   0.167 min)
SCF iterations                   ...        8.000 sec (=   0.133 min)  80.0 %

                                COMPOUND JOB  2
|  1>       %base "singlepoint"
Timings for individual modules:

Sum of individual times          ...        5.000 sec (=   0.083 min)
SCF iterations                   ...        4.000 sec (=   0.067 min)  80.0 %
"""


def _fake_orca(path: Path, exit_code: int = 0, banner: bool = True) -> Path:
    body = "#!/bin/bash\necho 'fake orca'\n" + "".join(f"echo '{line}'\n" for line in FAKE_COMPOUND_OUT.splitlines())
    if banner:
        body += "echo '                             ****ORCA TERMINATED NORMALLY****'\necho 'TOTAL RUN TIME: 0 days 0 hours 0 minutes 1 seconds 234 msec'\n"
    body += f"exit {exit_code}\n"
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_orca_outcome_requires_banner(tmp_path):
    out = tmp_path / "calcParam.out"
    out.write_text("...\n****ORCA TERMINATED NORMALLY****\nTOTAL RUN TIME: 0 days 1 hours 2 minutes 3 seconds 0 msec\n")
    assert orca_outcome(out, 0) == {"status": "completed", "exit_code": 0, "orca_terminated_normally": True,
                                    "orca_run_time": "0 days 1 hours 2 minutes 3 seconds 0 msec",
                                    "orca_run_time_s": 3723.0, "orca_steps_s": {}}
    out.write_text("died quietly\n")
    assert orca_outcome(out, 0)["status"] == "failed"
    assert orca_outcome(tmp_path / "missing.out", 1) == {"status": "failed", "exit_code": 1,
                                                          "orca_terminated_normally": None, "orca_run_time": None,
                                                          "orca_run_time_s": None, "orca_steps_s": None}
    out.write_text(FAKE_COMPOUND_OUT + "****ORCA TERMINATED NORMALLY****\nTOTAL RUN TIME: 0 days 0 hours 0 minutes 20 seconds 0 msec\n")
    oc = orca_outcome(out, 0)
    assert oc["status"] == "completed" and oc["orca_run_time_s"] == 20.0
    assert oc["orca_steps_s"] == {"optimization": 10.0, "singlepoint": 5.0}


def test_allocated_cpus_rules(monkeypatch):
    for var in ("SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(var, raising=False)
    assert allocated_cpus(16) == 16                      # no SLURM: the molecule's nprocs
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert allocated_cpus(16) == 8                       # ntasks x cpus_per_task (unset = 1)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert allocated_cpus(16) == 16
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "32")
    assert allocated_cpus(16) == 32                      # what the allocation holds on this node wins


def test_orca_task_records_and_status(tree, fake_sbatch, tmp_path, monkeypatch, capsys):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1002")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
    good = _fake_orca(tmp_path / "orca_ok")
    bad = _fake_orca(tmp_path / "orca_bad", exit_code=0, banner=False)
    assert run_orca_conformer(cfg, key, "conf_1", orca_cmd=str(good)) == "completed"
    assert run_orca_conformer(cfg, key, "conf_2", orca_cmd=str(bad)) == "failed"
    meta = records.load_metadata(tree["lib"], key)
    c1, c2 = meta["conformers"]["conf_1"], meta["conformers"]["conf_2"]
    assert c1["status"] == "completed" and c1["exit_code"] == 0 and c1["orca_terminated_normally"] is True
    assert c1["orca_run_time"].startswith("0 days") and c1["slurm_array_job_id"] == "1002" and "condensed" not in c1
    # what it cost: the allocation's CPUs, core-hours from the wall time, ORCA's own run time, per compound step
    assert c1["cpus"] == 4 and c1["cpu_hours"] == round(c1["wall_time_s"] / 3600 * 4, 5)
    assert c1["orca_run_time_s"] == 1.234 and c1["orca_steps_s"] == {"optimization": 10.0, "singlepoint": 5.0}
    assert c2["status"] == "failed" and c2["orca_terminated_normally"] is False
    assert c2["cpu_hours"] > 0 and c2["orca_steps_s"] == {"optimization": 10.0, "singlepoint": 5.0}  # a failure costs too
    assert "fake orca" in (tree["lib"] / key / "conf_1" / "calcParam.out").read_text()
    assert records.molecule_state(meta) == ("incomplete", "conformer(s) not completed: conf_2=failed")

    df = status_table(cfg)
    row = df.set_index("inchikey").loc[key]
    assert row["state"] == "incomplete"
    assert [row[c] for c in ("transformation", "pre_refinement", "search", "post_opt", "pruning", "dft")] == \
           ["done", "n/a", "done", "done", "done", "failed"]
    assert row["n_conf"] == 3 and row["n_dft"] == "2/3" and row["dft_done"] == "1/2"
    benzene = df.set_index("inchikey").loc[tree["keys"]["c1ccccc1"]]
    assert benzene["state"] == "in_progress"
    assert [benzene[c] for c in ("pre_refinement", "search", "post_opt", "pruning", "dft")] == ["n/a", "-", "-", "queued", "queued"]
    assert benzene["reason"] == "ensemble: preopt n/a, search -, post_opt -"
    assert main(["status", str(cfg.path)]) == 0
    out = capsys.readouterr().out
    assert "transf preopt search post_opt pruning" in " ".join(out.split()) and "needs a look" in out

    # resubmission: the failed conformer is prepared again, the completed one is left alone
    assert prepare_molecule(cfg, key, "CCO") == "prepared"
    meta = records.load_metadata(tree["lib"], key)
    assert meta["conformers"]["conf_1"]["status"] == "completed"
    assert meta["conformers"]["conf_2"]["status"] == "submitted" and meta["conformers"]["conf_2"]["resubmitted_from"] == "failed"
    assert run_orca_conformer(cfg, key, "conf_2", orca_cmd=str(good)) == "completed"
    assert records.molecule_state(records.load_metadata(tree["lib"], key))[0] == "complete"
    assert status_table(cfg).set_index("inchikey").loc[key]["dft"] == "done"


def _usage(wall: float, cpus: int, gpus: int = 0) -> dict:
    u = {"wall_time_s": wall, "cpus": cpus, "cpu_hours": round(wall / 3600 * cpus, 5)}
    if gpus:
        u["gpu_hours"] = round(wall / 3600 * gpus, 5)
    return u


def test_cost_per_molecule_and_stage(tree, fake_sbatch, tmp_path, monkeypatch, capsys):
    """``cost`` sums bceg's phases, the post-optimisation and the ORCA conformers (per compound step) per molecule."""
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    md = method_dir(tree["results"], "none", key, "etkdg")
    search = read_record(md.run_record(1)) | _usage(360.0, 2) | {
        "phases": {"pre_refinement": _usage(60.0, 2), "search": _usage(300.0, 2)}}
    write_record(md.run_record(1), search)
    write_record(md.run_record(1, LOT), read_record(md.run_record(1, LOT)) | _usage(36.0, 1, gpus=1))
    # the other molecule's records predate the phases and the usage of the post-optimisation: total -> search
    key2 = tree["keys"]["CC(C)O"]
    md2 = method_dir(tree["results"], "none", key2, "etkdg")
    write_record(md2.run_record(1), read_record(md2.run_record(1)) | _usage(720.0, 2))

    prepare_molecule(cfg, key, "CCO")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")
    good = _fake_orca(tmp_path / "orca_ok")
    assert run_orca_conformer(cfg, key, "conf_1", orca_cmd=str(good)) == "completed"
    assert run_orca_conformer(cfg, key, "conf_2", orca_cmd=str(good)) == "completed"
    meta = records.load_metadata(tree["lib"], key)
    dft_h = sum(meta["conformers"][c]["cpu_hours"] for c in ("conf_1", "conf_2"))

    df = cost_table(cfg).set_index("inchikey")
    row = df.loc[key]
    assert row["pre_refinement_cpu_h"] == round(60 / 3600 * 2, 4) and row["search_cpu_h"] == round(300 / 3600 * 2, 4)
    assert row["post_opt_cpu_h"] == 0.01 and row["post_opt_gpu_h"] == 0.01 and row["total_gpu_h"] == 0.01
    assert row["dft_cpu_h"] == round(dft_h, 4) and row["n_dft_runs"] == 2 and row["n_ensemble_runs"] == 2
    assert row["dft_optimization_cpu_h"] == round(2 * 10 * 4 / 3600, 4)      # 2 conformers x 10 s x 4 CPUs
    assert row["dft_singlepoint_cpu_h"] == round(2 * 5 * 4 / 3600, 4)
    assert row["total_cpu_h"] == round(0.2 + 0.01 + dft_h, 4)
    row2 = df.loc[key2]
    assert row2["pre_refinement_cpu_h"] == 0 and row2["search_cpu_h"] == 0.4 and row2["post_opt_cpu_h"] == 0
    assert row2["n_ensemble_runs"] == 1 and row2["dft_cpu_h"] == 0 and row2["n_dft_runs"] == 0
    benzene = df.loc[tree["keys"]["c1ccccc1"]]
    assert benzene["total_cpu_h"] == 0 and benzene["n_ensemble_runs"] == 0
    assert list(df.index[:2]) == [key2, key] or list(df.index[:2]) == [key, key2]   # sorted by total, benzene last
    assert df.index[-1] == tree["keys"]["c1ccccc1"]

    csv = tree["root"] / "out" / "cost.csv"
    assert main(["cost", str(cfg.path), "--csv", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "total   :" in out and "per mol :" in out and "gpu     : post_opt 0.01" in out
    assert "search_gpu_h" not in out                      # all-zero GPU columns are not printed...
    written = pd.read_csv(csv)
    assert "search_gpu_h" in written.columns and len(written) == 3   # ...but are in the CSV


def test_registry_keeps_first_smiles_and_aliases(tmp_path, capsys):
    records.register_molecule(tmp_path, "KEY", {"smiles": "CCO", "canonical_smiles": "CCO"})
    records.register_molecule(tmp_path, "KEY", {"smiles": "OCC", "canonical_smiles": "CCO", "name": "ethanol"})
    entry = records.load_registry(tmp_path)["KEY"]
    assert entry["smiles"] == "CCO" and entry["also_submitted_as"] == ["OCC"] and entry["name"] == "ethanol"
    assert "registered_at" in entry and capsys.readouterr().err == ""
    records.register_molecule(tmp_path, "KEY", {"smiles": "C=CO", "canonical_smiles": "C=CO"})
    assert "WARNING" in capsys.readouterr().err


def test_molecule_state_rules():
    assert records.molecule_state({})[0] == "absent"
    assert records.molecule_state({"orca_stage": {"status": "queued"}})[0] == "in_progress"
    assert records.molecule_state({"orca_stage": {"status": "ensemble_failed", "reason": "x"}}) == ("incomplete", "ensemble_failed: x")
    base = {"orca_stage": {"status": "prepared"}}
    assert records.molecule_state(base | {"conformers": {"conf_1": {"status": "running"}}})[0] == "in_progress"
    assert records.molecule_state(base | {"conformers": {"conf_1": {"status": "timeout"}}})[0] == "incomplete"
    assert records.molecule_state(base | {"conformers": {"conf_1": {"status": "completed"},
                                                         "conf_2": {"status": "skipped_energy_filter"}}})[0] == "complete"


# --------------------------------------------------------------------------- one status per stage
def _progress(search=None, refine=None, transformation="done", has_preopt=True, has_refiner=True):
    return {"transformation": transformation, "error": None, "search_record": search, "refine_record": refine,
            "has_preopt": has_preopt, "has_refiner": has_refiner, "n_conformers": None}


STAGES = ("transformation", "pre_refinement", "search", "post_opt", "pruning", "dft")


def _stages(progress, meta=None, orca_on=True):
    return tuple(stage_statuses(progress, meta or {}, orca_on)[k] for k in STAGES)


def test_stage_statuses_rules():
    queued = {"ensemble_stage": {"status": "queued"}}
    assert _stages(_progress()) == ("done", "-", "-", "-", "-", "-")
    assert _stages(_progress(), queued) == ("done", "queued", "queued", "queued", "-", "-")
    assert _stages(_progress(), queued | {"orca_stage": {"status": "queued"}}) == ("done", "queued", "queued", "queued", "queued", "queued")
    assert _stages(_progress({"status": "running", "phase": "pre_refinement"})) == ("done", "running", "queued", "queued", "-", "-")
    assert _stages(_progress({"status": "running", "phase": "search"})) == ("done", "done", "running", "queued", "-", "-")
    assert _stages(_progress({"status": "running"})) == ("done", "done", "running", "queued", "-", "-")        # legacy record
    assert _stages(_progress({"status": "done"})) == ("done", "done", "done", "-", "-", "-")
    assert _stages(_progress({"status": "done"}, {"status": "running"})) == ("done", "done", "done", "running", "-", "-")
    assert _stages(_progress({"status": "done"}, {"status": "done"})) == ("done", "done", "done", "done", "-", "-")
    assert _stages(_progress({"status": "failed", "phase": "pre_refinement"})) == ("done", "failed", "-", "-", "-", "-")
    assert _stages(_progress({"status": "failed", "phase": "search"})) == ("done", "done", "failed", "-", "-", "-")
    assert _stages(_progress({"status": "failed"})) == ("done", "failed", "failed", "-", "-", "-")                # legacy record
    assert _stages(_progress({"status": "done"}, {"status": "failed"})) == ("done", "done", "done", "failed", "-", "-")
    assert _stages(_progress({"status": "running", "phase": "search"}, has_preopt=False)) == ("done", "n/a", "running", "queued", "-", "-")
    assert _stages(_progress({"status": "done"}, has_refiner=False)) == ("done", "done", "done", "n/a", "-", "-")
    assert _stages(_progress(transformation="not_applicable"), queued) == ("n/a",) + ("n/a",) * 5
    assert _stages(_progress(transformation="failed"))[0] == "failed"
    done = _progress({"status": "done"}, {"status": "done"})
    assert _stages(done, {"orca_stage": {"status": "prepared"}}) == ("done", "done", "done", "done", "done", "queued")
    assert _stages(done, {"orca_stage": {"status": "prepared"}, "conformers": {"a": {"status": "submitted"}}})[5] == "queued"
    assert _stages(done, {"orca_stage": {"status": "prepared"}, "conformers": {"a": {"status": "running"}, "b": {"status": "submitted"}}})[5] == "running"
    assert _stages(done, {"orca_stage": {"status": "prepared"}, "conformers": {"a": {"status": "completed"}, "b": {"status": "skipped_energy_filter"}}})[5] == "done"
    assert _stages(done, {"orca_stage": {"status": "prepared"}, "conformers": {"a": {"status": "completed"}, "b": {"status": "timeout"}}})[5] == "failed"
    assert _stages(done, {"orca_stage": {"status": "ensemble_changed"}})[4:] == ("failed", "-")
    assert _stages(done, {"orca_stage": {"status": "prepared"}}, orca_on=False)[4:] == ("off", "off")


# --------------------------------------------------------------------------- guards before anything is written
def test_placeholder_is_refused_at_load(tree, capsys):
    raw = _lib_cfg()
    raw["slurm"]["ensembles"]["cpu"]["partition"] = "CHANGE_ME"
    tree["cfg"].write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="slurm.ensembles.cpu.partition"):
        Config.load(tree["cfg"])
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().err and not (tree["lib"] / "params.json").exists()


def test_unknown_partition_is_refused_before_writing(tree, monkeypatch, capsys):
    raw = _lib_cfg()
    raw["slurm"]["ensembles"]["cpu"]["partition"] = "expansion"
    raw["slurm"]["orca"]["partition"] = "nowhere"
    raw["conformer_search"]["slurm"] = {"partition": "gone"}
    tree["cfg"].write_text(yaml.safe_dump(raw, sort_keys=False))
    monkeypatch.setattr(slurm, "known_partitions", lambda: {"expansion", "gpu"})
    assert main(["submit", str(tree["cfg"])]) == 2
    err = capsys.readouterr().err
    assert "slurm.orca.partition: partition 'nowhere' does not exist" in err
    assert "conformer_search.slurm.partition: partition 'gone'" in err
    assert "slurm.ensembles.cpu.partition" not in err          # 'expansion' exists, so it is not reported
    assert not (tree["lib"] / "params.json").exists() and not (tree["lib"] / "_jobs").exists()
    assert slurm.known_partitions() == {"expansion", "gpu"}


def test_bceg_sbatch_failure_is_reported_cleanly(tree, monkeypatch, capsys):
    path = _rewrite(tree, run_bceg=True)

    def boom(self, jobdir, no_sbatch=False, force=False):
        raise RuntimeError("bceg submit.sh failed (1): sbatch: error: invalid partition specified: x")

    monkeypatch.setattr(Ensembles, "submit", boom)
    monkeypatch.setattr(Ensembles, "pending_tasks", lambda self, force=False: (["one task"], []))  # the fake tree's params.json is not bceg's
    assert main(["submit", str(path)]) == 1
    err = capsys.readouterr().err
    assert "bceg submission failed" in err and "Nothing was queued" in err


def test_queued_molecule_is_requeued_when_bceg_runs_again_for_it(tree, fake_sbatch, monkeypatch, capsys):
    path = _rewrite(tree, run_bceg=True)
    key = tree["keys"]["CCO"]
    # first submission: everything queued behind bceg
    monkeypatch.setattr(Ensembles, "pending_tasks", lambda self, force=False: ([type("T", (), {"inchikey": key})()], []))
    monkeypatch.setattr(Ensembles, "submit", lambda self, jobdir, no_sbatch=False, force=False: ["42"])
    assert main(["submit", str(path)]) == 0
    meta = records.load_metadata(tree["lib"], key)
    assert meta["orca_stage"]["status"] == "queued"
    assert meta["ensemble_stage"]["status"] == "queued" and meta["ensemble_stage"]["bceg_job_ids"] == ["42"]
    assert "ensemble_stage" not in records.load_metadata(tree["lib"], tree["keys"]["CC(C)O"])   # bceg had nothing for it
    # bceg has work for CCO again (its refinement failed and is resubmitted): CCO is re-queued, the others stay skipped
    assert main(["submit", str(path)]) == 0
    out = capsys.readouterr().out
    assert "prepare re-queued behind the new bceg job(s)" in out and f"skip  {tree['keys']['CC(C)O']}" in out
    assert fake_sbatch[-1]["dependency"] == ["42"]
    tasks = [json.loads(l) for l in (fake_sbatch[-1]["script"].parent / "prepare.jsonl").read_text().splitlines()]
    assert [t["inchikey"] for t in tasks] == [key]
    # nothing pending in bceg: queued molecules are skipped as before
    monkeypatch.setattr(Ensembles, "pending_tasks", lambda self, force=False: ([], []))
    assert main(["submit", str(path), "--dry-run"]) == 0
    assert "0 molecule(s) to prepare" in capsys.readouterr().out


def test_orca_off_marks_the_ensemble_queued(tree, fake_sbatch, monkeypatch):
    path = _rewrite(tree, run_bceg=True, orca="off")
    key = tree["keys"]["c1ccccc1"]
    monkeypatch.setattr(Ensembles, "pending_tasks", lambda self, force=False: ([type("T", (), {"inchikey": key})()], []))
    monkeypatch.setattr(Ensembles, "submit", lambda self, jobdir, no_sbatch=False, force=False: ["77"])
    assert main(["submit", str(path)]) == 0
    assert fake_sbatch == []                                               # no prepare array with orca off
    meta = records.load_metadata(tree["lib"], key)
    assert meta["ensemble_stage"]["bceg_job_ids"] == ["77"] and "orca_stage" not in meta
    assert records.load_registry(tree["lib"])[key]["name"] == "benzene"
    row = status_table(Config.load(path)).set_index("inchikey").loc[key]
    assert [row[c] for c in ("pre_refinement", "search", "post_opt", "pruning", "dft")] == ["n/a", "queued", "queued", "off", "off"]


def test_running_bceg_task_is_not_resubmitted(tree):
    """A replicate whose bceg record says running is left alone by pending_tasks (unless force)."""
    cfg = Config.load(_rewrite(tree, run_bceg=True))
    source = Ensembles(cfg)
    key = tree["keys"]["c1ccccc1"]   # benzene: no ensemble yet -> a pending chain task
    md = method_dir(tree["results"], "none", key, "etkdg")
    md.path.mkdir(parents=True, exist_ok=True)
    before, _ = source.pending_tasks()
    assert key in {t.inchikey for t in before}
    write_record(md.run_record(1), {"status": "running", "phase": "search"})
    after, _ = source.pending_tasks()
    assert key not in {t.inchikey for t in after} and len(after) == len(before) - 1
    forced, _ = source.pending_tasks(force=True)
    assert key in {t.inchikey for t in forced}
    # and status shows it running
    row = status_table(cfg).set_index("inchikey").loc[key]
    assert row["search"] == "running" and row["post_opt"] == "queued"


def test_library_config_with_the_gxtb_and_aimnet2_refiners(tree):
    """`post_optimization: {refiner: gxtb}` gives lot `gxtb`; aimnet2 on cuda is routed to the gpu profile."""
    cfg = Config.load(_rewrite(tree, post_optimization={"refiner": "gxtb"}))
    assert cfg.lot == "gxtb"
    post = cfg.compared()["post_optimization"]
    assert post["refiner"] == "gxtb" and post["lot"] == "gxtb"
    assert post["params"] == {"level": "normal", "cycles": None, "keep_unconverged": True, "lot": "gxtb"}  # no executable / threads
    assert cfg.bceg_raw()["post_optimization"] == {"refiner": "gxtb", "params": {}, "slurm": {}}
    ens = Ensembles(cfg).ensemble(InputMolecule.from_smiles("CCO"))
    assert not ens.done and "run_1_gxtb.json" in ens.reason             # the tree only has uma ensembles

    cfg = Config.load(_rewrite(tree, post_optimization={"refiner": "aimnet2", "params": {"model": "aimnet2-2025", "device": "cuda"}}))
    assert cfg.lot == "aimnet2-2025" and cfg.bceg_config().refiner.resource_class() == "gpu"
    assert "device" not in cfg.compared()["post_optimization"]["params"]
    with pytest.raises(ValueError, match="post_optimization.refiner"):
        Config.load(_rewrite(tree, post_optimization={"refiner": "g-xtb"}))
