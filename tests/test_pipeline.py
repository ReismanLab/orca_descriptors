from __future__ import annotations

import json

import pytest

from bceg.cli import main
from bceg.config import Config
from bceg.inputs import InputMolecule
from bceg.slurm import write_jobs
from bceg.tasks import expand, run_local

BUTANE = InputMolecule.from_smiles("CCCC").inchikey
ETHANOL = InputMolecule.from_smiles("CCO").inchikey


def test_expansion_counts(project):
    root, write = project
    cfg = Config.load(write())
    tasks, problems = expand(cfg)
    assert not problems
    # none: 2 molecules, H: only ethanol -> 3 structures x 1 method x 3 replicates, x2 (search + refine)
    assert sum(t.kind == "search" for t in tasks) == 9
    assert sum(t.kind == "refine" for t in tasks) == 9


def test_local_run_layout_and_skip(project):
    root, write = project
    cfg = Config.load(write())
    counts = run_local(cfg)
    assert counts == {"done": 18}

    md = root / "results" / "none" / BUTANE / "etkdg"
    names = sorted(p.name for p in md.iterdir())
    assert names == sorted(
        ["params.json", "params_fake-lot.json"]
        + [f"raw_{i}.xyz" for i in (1, 2, 3)] + [f"raw_{i}_fake-lot.xyz" for i in (1, 2, 3)]
        + [f"run_{i}.json" for i in (1, 2, 3)] + [f"run_{i}_fake-lot.json" for i in (1, 2, 3)]
    )
    assert (root / "results" / "H" / ETHANOL / "etkdg" / "raw_1.xyz").exists()
    assert not (root / "results" / "H" / BUTANE).exists()
    assert (root / "results" / "molecules.csv").exists()

    # replicates use different seeds -> different geometries
    run1, run2 = (json.loads((md / f"run_{i}.json").read_text()) for i in (1, 2))
    assert run1["seed"] == 1 and run2["seed"] == 2
    assert (md / "raw_1.xyz").read_text() != (md / "raw_2.xyz").read_text()

    params = json.loads((md / "params.json").read_text())
    assert params["compared"]["params"]["n_confs"] == 4
    assert params["compared"]["transformation"]["name"] == "none"

    # second run: everything skipped, nothing pending
    assert expand(cfg)[0] == []
    assert run_local(cfg) == {}


def test_changed_hyperparameter_is_refused(project):
    root, write = project
    run_local(Config.load(write()))

    # resource-only change: fine
    tasks, problems = expand(Config.load(write("{n_confs: 4, threads: 2}")))
    assert tasks == [] and problems == []

    # real hyper-parameter change under the same method name: refused
    tasks, problems = expand(Config.load(write("{n_confs: 10}")))
    assert tasks == [] and len(problems) == 3
    assert "~ params.n_confs: 4 -> 10" in problems[0] and "new name" in problems[0]
    assert main(["plan", str(root / "config.yaml")]) == 2

    # post-optimisation change: refused as well
    _, problems = expand(Config.load(write(refiner_params="{scale: 2.0}")))
    assert len(problems) == 3 and "params_fake-lot.json" in problems[0]


def test_edited_transformation_script_is_refused(project):
    root, write = project
    cfg = Config.load(write())
    run_local(cfg)
    (root / "protonate.py").write_text((root / "protonate.py").read_text() + "\n# edited\n")
    _, problems = expand(Config.load(write()))
    assert len(problems) == 1 and "script_sha256" in problems[0]


def test_failed_replicate_is_rerun(project):
    root, write = project
    cfg = Config.load(write())
    run_local(cfg)
    md = root / "results" / "none" / BUTANE / "etkdg"
    (md / "raw_2.xyz").unlink()
    pending = expand(cfg)[0]
    assert [(t.kind, t.inchikey, t.replicate) for t in pending] == [("search", BUTANE, 2)]


def test_slurm_scripts(project):
    root, write = project
    cfg = Config.load(write())
    jobdir = write_jobs(cfg, expand(cfg)[0], jobs_root=root / "jobs")
    sbatch = (jobdir / "search_etkdg.sbatch").read_text()
    assert "#SBATCH --array=0-8" in sbatch and "#SBATCH --partition=short" in sbatch
    assert "--chunk $SLURM_ARRAY_TASK_ID" in sbatch
    submit = (jobdir / "submit.sh").read_text()
    assert "--dependency=afterany$deps refine_fake-lot.sbatch" in submit
    assert len((jobdir / "refine_fake-lot.jsonl").read_text().splitlines()) == 9


def test_cli_run_task_and_status(project, capsys):
    root, write = project
    cfg_path = write()
    assert main(["plan", str(cfg_path), "--dry-run"]) == 0
    cfg = Config.load(cfg_path)
    jobdir = write_jobs(cfg, expand(cfg)[0], jobs_root=root / "jobs")
    assert main(["run-task", str(cfg_path), "--tasks", str(jobdir / "search_etkdg.jsonl"), "--chunk", "0", "--chunk-size", "9"]) == 0
    assert main(["run-task", str(cfg_path), "--tasks", str(jobdir / "refine_fake-lot.jsonl"), "--chunk", "0", "--chunk-size", "9"]) == 0
    capsys.readouterr()
    assert main(["status", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "done" in out and "missing" not in out


def test_dry_run_writes_nothing(project):
    root, write = project
    assert main(["plan", str(write()), "--dry-run"]) == 0
    assert not (root / "results").exists()
