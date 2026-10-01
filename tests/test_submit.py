"""desc_utils.submit against a fake bceg results tree: config, freeze and refuse rule, prepare (split +
energy window + rendered input + records), ORCA task with a fake orca, status, registry."""
from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest
import yaml

from bceg.inputs import InputMolecule
from bceg.layout import method_dir
from bceg.provenance import make_params, write_record
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_KCAL, Conformer, write_xyz

from desc_utils.submit import records, slurm, template
from desc_utils.submit.cli import main
from desc_utils.submit.config import Config
from desc_utils.submit.source import make_source
from desc_utils.submit.status import status_table
from desc_utils.submit.tasks import orca_outcome, prepare_molecule, run_orca_conformer

LOT = "uma-s-1p2"
LIGANDS = "CCO ethanol\nCC(C)O\nc1ccccc1 benzene\n"  # benzene gets no ensemble
LEVEL_OF_THEORY = {
    "conf_search": False, "opt_geom": True, "freq_calc": True, "nmr_calc": True, "vertical_redox": True,
    "goat_method": "XTB", "opt_method": "wb97m-d4 def2-tzvp(-f)", "opt_solvent": "", "svp_method": "m06 d4 def2-tzvp",
    "property_solvent": "", "nmr_method": "m06 d4 def2-tzvp", "nbo_run": "NBO NPA", "extra_basis": {"Ni": "def2-tzvp"},
}


def _ensemble(results_dir: Path, smiles: str, rel_kcal=(0.0, 2.0, 10.0)) -> None:
    mol = InputMolecule.from_smiles(smiles)
    structure = Structure(mol.canonical_smiles)
    md = method_dir(results_dir, "none", mol.inchikey, "etkdg")
    md.path.mkdir(parents=True, exist_ok=True)
    n = len(structure.elements)
    confs = [Conformer(coords=[[i * 0.1 + k, 0.0, 0.0] for i in range(n)], energy=-100.0 + r / HARTREE_TO_KCAL)
             for k, r in enumerate(rel_kcal)]
    write_xyz(md.refined(1, LOT), structure.elements, confs)
    sp = make_params({"searcher": "rdkit_etkdg", "params": {"n_confs": 3}, "seed_rule": "x",
                      "transformation": {"name": "none"}, "structure": structure.identity()})
    md.params().write_text(json.dumps(sp))
    md.params(LOT).write_text(json.dumps(make_params({"refiner": "uma", "lot": LOT, "search_params_sha256": sp["sha256"]})))
    write_record(md.run_record(1), {"status": "done", "params_sha256": sp["sha256"]})
    write_record(md.run_record(1, LOT), {"status": "done", "versions": {"fairchem": "x"}})


@pytest.fixture
def tree(tmp_path: Path) -> dict:
    (tmp_path / "ligands.txt").write_text(LIGANDS)
    results = tmp_path / "DATA_bceg"
    (tmp_path / "bceg.yaml").write_text(textwrap.dedent(f"""
        inputs: ligands.txt
        results_dir: DATA_bceg
        replicates: 1
        transformations: {{none: {{multiplicity: 1}}}}
        methods: {{etkdg: {{searcher: rdkit_etkdg, params: {{n_confs: 3}}}}}}
        post_optimization: {{refiner: uma, params: {{model: {LOT}}}}}
        slurm: {{setup: [], cpu: {{time: "01:00:00"}}}}
        """))
    for smi in ("CCO", "CC(C)O"):
        _ensemble(results, smi)
    lib_cfg = {
        "library_dir": "LIB",
        "source": {"kind": "bceg", "config": "bceg.yaml", "transformation": "none", "method": "etkdg",
                   "smiles": "ligands.txt", "run_bceg": False},
        "dft": {"ewin_kcal_mol": 5.0, "level_of_theory": LEVEL_OF_THEORY},
        "slurm": {"setup": ["echo setup"], "orca": {"time": "01:00:00", "mem_per_cpu": "4G", "ntasks": 2}},
        "condense": False,
    }
    (tmp_path / "lib.yaml").write_text(yaml.safe_dump(lib_cfg, sort_keys=False))
    keys = {s: InputMolecule.from_smiles(s).inchikey for s in ("CCO", "CC(C)O", "c1ccccc1")}
    return {"root": tmp_path, "cfg": tmp_path / "lib.yaml", "lib": tmp_path / "LIB", "keys": keys, "results": results}


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
    assert cfg.library_dir == tree["lib"]
    assert cfg.source["results_dir"] == str(tree["results"])          # taken from the bceg config
    assert cfg.template_path.name == "calcParam.inp.tmpl" and cfg.template_path.exists()
    assert cfg.slurm["orca"]["ntasks"] == 2 and cfg.slurm["orca"]["nodes"] == 1  # override + default
    assert cfg.orca_on and not cfg.condense
    assert "run_bceg" not in cfg.compared()["source"] and cfg.recorded()["run_bceg"] is False
    assert cfg.compared()["dft"]["calc_param_template"]["sha256"] == cfg.template_sha256()


def test_render_and_compound_steps():
    text = template.render(Config.load.__globals__["DEFAULT_TEMPLATE"], LEVEL_OF_THEORY, charge=0, multiplicity=1,
                           nprocs=8, maxcore_mb=3072)
    assert "$" not in text
    assert "%maxcore 3072" in text and "nprocs 8" in text
    assert text.count('NewGTO Ni "def2-tzvp" end') == 2
    opt = text.split('%base "optimization"')[1].split("StepEnd")[0]
    single = text.split('%base "singlepoint"')[1].split("StepEnd")[0]
    assert "%basis" in opt and "%basis" not in single
    steps = template.compound_steps(LEVEL_OF_THEORY, 0, 1)
    assert steps["oxidation"] == {"runs": True, "method": "m06 d4 def2-tzvp", "solvent": "", "population_analysis": "NBO NPA",
                                  "charge": 1, "multiplicity": 2}
    assert steps["frequency"]["extra_basis"] == "Ni def2-tzvp"
    no_eb = dict(LEVEL_OF_THEORY, extra_basis={})
    assert "%basis" not in template.render(Config.load.__globals__["DEFAULT_TEMPLATE"], no_eb, 0, 1, 1, 1000)


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


def test_submit_freezes_and_queues_prepare(tree, fake_sbatch):
    assert main(["submit", str(tree["cfg"])]) == 0
    lib = tree["lib"]
    params = json.loads((lib / "params.json").read_text())
    assert params["compared"]["dft"]["ewin_kcal_mol"] == 5.0 and "run_bceg" not in params["compared"]["source"]
    jobdirs = list((lib / "_jobs").iterdir())
    assert len(jobdirs) == 1
    jobdir = jobdirs[0]
    frozen = yaml.safe_load((jobdir / "config.yaml").read_text())
    assert frozen["_pipeline"]["repo"] and "git_commit" in frozen["_pipeline"]
    assert frozen["dft"]["calc_param_template"] == str(jobdir / "calcParam.inp.tmpl")
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


def test_changed_dft_is_refused(tree, fake_sbatch, capsys):
    assert main(["submit", str(tree["cfg"])]) == 0
    raw = yaml.safe_load(tree["cfg"].read_text())
    raw["dft"]["ewin_kcal_mol"] = 8.0
    tree["cfg"].write_text(yaml.safe_dump(raw))
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "ewin_kcal_mol: 5.0 -> 8.0" in err
    # recorded settings may change freely
    raw["dft"]["ewin_kcal_mol"] = 5.0
    raw["orca"] = "off"
    raw["slurm"]["orca"]["time"] = "99:00:00"
    tree["cfg"].write_text(yaml.safe_dump(raw))
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 0


# --------------------------------------------------------------------------- prepare
def _frozen(tree, fake_sbatch) -> Config:
    assert main(["submit", str(tree["cfg"])]) == 0
    jobdir = next((tree["lib"] / "_jobs").iterdir())
    return Config.load(jobdir / "config.yaml")


def test_prepare_splits_filters_and_submits(tree, fake_sbatch):
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
    assert meta["dft"]["charge"] == 0 and meta["dft"]["multiplicity"] == 1
    assert meta["dft"]["compound_steps"]["optimization"]["extra_basis"] == "Ni def2-tzvp"
    assert meta["dft"]["level_of_theory"]["extra_basis"] == {"Ni": "def2-tzvp"}
    assert meta["ensemble"]["params_sha256"] and meta["ensemble"]["lot"] == LOT and meta["ensemble"]["replicate"] == 1
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


def test_prepare_without_ensemble(tree, fake_sbatch):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["c1ccccc1"]
    assert prepare_molecule(cfg, key, "c1ccccc1") == "ensemble_missing"
    meta = records.load_metadata(tree["lib"], key)
    assert meta["orca_stage"]["status"] == "ensemble_missing" and "run_1_uma-s-1p2.json" in meta["orca_stage"]["reason"]
    assert records.molecule_state(meta)[0] == "incomplete"
    assert not (tree["lib"] / key / "conf_1").exists()


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


# --------------------------------------------------------------------------- ORCA task
def _fake_orca(path: Path, exit_code: int = 0, banner: bool = True) -> Path:
    body = "#!/bin/bash\necho 'fake orca'\n"
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
                                    "orca_run_time": "0 days 1 hours 2 minutes 3 seconds 0 msec"}
    out.write_text("died quietly\n")
    assert orca_outcome(out, 0)["status"] == "failed"
    assert orca_outcome(tmp_path / "missing.out", 1) == {"status": "failed", "exit_code": 1,
                                                          "orca_terminated_normally": None, "orca_run_time": None}


def test_orca_task_records_and_status(tree, fake_sbatch, tmp_path, monkeypatch):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1002")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    good = _fake_orca(tmp_path / "orca_ok")
    bad = _fake_orca(tmp_path / "orca_bad", exit_code=0, banner=False)
    assert run_orca_conformer(cfg, key, "conf_1", orca_cmd=str(good)) == "completed"
    assert run_orca_conformer(cfg, key, "conf_2", orca_cmd=str(bad)) == "failed"
    meta = records.load_metadata(tree["lib"], key)
    c1, c2 = meta["conformers"]["conf_1"], meta["conformers"]["conf_2"]
    assert c1["status"] == "completed" and c1["exit_code"] == 0 and c1["orca_terminated_normally"] is True
    assert c1["orca_run_time"].startswith("0 days") and c1["slurm_array_job_id"] == "1002" and "condensed" not in c1
    assert c2["status"] == "failed" and c2["orca_terminated_normally"] is False
    assert "fake orca" in (tree["lib"] / key / "conf_1" / "calcParam.out").read_text()
    assert records.molecule_state(meta) == ("incomplete", "conformer(s) not completed: conf_2=failed")

    df = status_table(cfg)
    row = df.set_index("inchikey").loc[key]
    assert row["state"] == "incomplete" and row["completed"] == 1 and row["failed"] == 1 and row["skipped_energy_filter"] == 1
    assert df.set_index("inchikey").loc[tree["keys"]["c1ccccc1"], "state"] == "in_progress"  # prepare queued

    # resubmission: the failed conformer is prepared again, the completed one is left alone
    assert prepare_molecule(cfg, key, "CCO") == "prepared"
    meta = records.load_metadata(tree["lib"], key)
    assert meta["conformers"]["conf_1"]["status"] == "completed"
    assert meta["conformers"]["conf_2"]["status"] == "submitted" and meta["conformers"]["conf_2"]["resubmitted_from"] == "failed"
    assert run_orca_conformer(cfg, key, "conf_2", orca_cmd=str(good)) == "completed"
    assert records.molecule_state(records.load_metadata(tree["lib"], key))[0] == "complete"


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


# --------------------------------------------------------------------------- one-file config (inline bceg)
def _inline_cfg(tree, **overrides) -> Path:
    bceg_block = {
        "transformations": {"none": {"multiplicity": 1}},
        "methods": {"etkdg": {"searcher": "rdkit_etkdg", "params": {"n_confs": 3}},
                    "other": {"searcher": "rdkit_etkdg", "params": {"n_confs": 50}}},
        "post_optimization": {"refiner": "uma", "params": {"model": LOT}},
        "slurm": {"cpu": {"time": "01:00:00"}},
    }
    raw = {
        "library_dir": "LIB_inline",
        "source": {"kind": "bceg", "config": bceg_block, "transformation": "none", "method": "etkdg",
                   "smiles": "ligands.txt", "run_bceg": False, "results_dir": "DATA_bceg"},
        "dft": {"ewin_kcal_mol": 5.0, "level_of_theory": LEVEL_OF_THEORY},
        "slurm": {"setup": ["echo setup"], "orca": {"time": "01:00:00", "mem_per_cpu": "4G", "ntasks": 2}},
        "condense": False,
    }
    for k, v in overrides.items():
        raw[k] = v
    path = tree["root"] / "lib_inline.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_inline_bceg_config(tree, fake_sbatch):
    cfg = Config.load(_inline_cfg(tree))
    b = cfg.source["config"]
    assert isinstance(b, dict) and b["replicates"] == 1 and b["inputs"] == str(tree["root"] / "ligands.txt")
    assert b["slurm"]["setup"] == ["echo setup"] and b["slurm"]["cpu"] == {"time": "01:00:00"}   # setup inherited
    assert b["transformations"]["none"]["script"] is None and cfg.source["results_dir"] == str(tree["results"])
    compared = cfg.compared()["source"]
    assert compared["bceg"]["method"] == {"searcher": "rdkit_etkdg", "params": {"n_confs": 3}}
    assert compared["bceg"]["replicates"] == 1 and "other" not in str(compared) and "results_dir" not in compared
    assert cfg.recorded()["source_paths"]["results_dir"] == str(tree["results"])
    # the inline config drives the source like a file would
    source = make_source(cfg.source)
    ens = source.ensemble(InputMolecule.from_smiles("CCO"))
    assert ens.done and len(ens.conformers) == 3 and ens.charge == 0
    # full submit on it, then prepare straight from the frozen copy
    assert main(["submit", str(cfg.path)]) == 0
    frozen = Config.load(next((tree["root"] / "LIB_inline" / "_jobs").iterdir()) / "config.yaml")
    assert isinstance(frozen.source["config"], dict)
    assert prepare_molecule(frozen, tree["keys"]["CCO"], "CCO") == "prepared"


def test_inline_defaults_results_dir_into_library(tree):
    path = _inline_cfg(tree)
    raw = yaml.safe_load(path.read_text())
    del raw["source"]["results_dir"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = Config.load(path)
    assert cfg.source["results_dir"] == str(tree["root"] / "LIB_inline" / "ensembles")
    assert cfg.source["config"]["results_dir"] == cfg.source["results_dir"]


def test_unrelated_bceg_method_may_change_but_selected_one_may_not(tree, fake_sbatch, capsys):
    path = _inline_cfg(tree)
    assert main(["submit", str(path)]) == 0
    raw = yaml.safe_load(path.read_text())
    raw["source"]["config"]["methods"]["other"]["params"]["n_confs"] = 7      # not the method this library refines
    raw["source"]["config"]["slurm"]["cpu"]["time"] = "48:00:00"               # bceg resources: recorded only
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert main(["submit", str(path), "--dry-run"]) == 0
    raw["source"]["config"]["methods"]["etkdg"]["params"]["n_confs"] = 4       # the refined ensemble's parameters
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert main(["submit", str(path), "--dry-run"]) == 2
    assert "source.bceg.method.params.n_confs: 3 -> 4" in capsys.readouterr().err


# --------------------------------------------------------------------------- guards before anything is written
def test_placeholder_is_refused_at_load(tree, capsys):
    path = _inline_cfg(tree)
    raw = yaml.safe_load(path.read_text())
    raw["source"]["config"]["slurm"]["cpu"]["partition"] = "CHANGE_ME"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="source.config.slurm.cpu.partition"):
        Config.load(path)
    assert main(["submit", str(path), "--dry-run"]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().err and not (tree["root"] / "LIB_inline" / "params.json").exists()


def test_unknown_partition_is_refused_before_writing(tree, monkeypatch, capsys):
    path = _inline_cfg(tree)
    raw = yaml.safe_load(path.read_text())
    raw["source"]["config"]["slurm"]["cpu"]["partition"] = "expansion"
    raw["slurm"]["orca"]["partition"] = "nowhere"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    monkeypatch.setattr(slurm, "known_partitions", lambda: {"expansion", "gpu"})
    assert main(["submit", str(path)]) == 2
    err = capsys.readouterr().err
    assert "slurm.orca.partition: partition 'nowhere' does not exist" in err
    assert "source.config.slurm.cpu.partition" not in err          # 'expansion' exists, so it is not reported
    assert not (tree["root"] / "LIB_inline" / "params.json").exists() and not (tree["root"] / "LIB_inline" / "_jobs").exists()
    assert slurm.known_partitions() == {"expansion", "gpu"}


def test_bceg_sbatch_failure_is_reported_cleanly(tree, monkeypatch, capsys):
    path = _inline_cfg(tree)
    raw = yaml.safe_load(path.read_text())
    raw["source"]["run_bceg"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    from desc_utils.submit.source import BcegSource

    def boom(self, jobdir, no_sbatch=False):
        raise RuntimeError("bceg submit.sh failed (1): sbatch: error: invalid partition specified: x")

    monkeypatch.setattr(BcegSource, "submit", boom)
    assert main(["submit", str(path)]) == 1
    err = capsys.readouterr().err
    assert "bceg submission failed" in err and "Nothing was queued" in err
