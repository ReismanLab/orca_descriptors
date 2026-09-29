from __future__ import annotations

import json

import yaml

from bceg.cli import main
from bceg.config import Config
from bceg.inputs import InputMolecule
from bceg.slurm import write_jobs
from bceg.tasks import expand, run_local

BUTANE = InputMolecule.from_smiles("CCCC").inchikey
ETHANOL = InputMolecule.from_smiles("CCO").inchikey


def test_expansion_counts(project):
    root, write = project
    # none: 2 molecules, H: only ethanol -> 3 structures x 1 method x 3 replicates
    tasks, problems = expand(Config.load(write()))
    assert not problems
    assert [t.kind for t in tasks] == ["chain"] * 9
    tasks, _ = expand(Config.load(write(mode="split")))
    assert sum(t.kind == "search" for t in tasks) == 9 and sum(t.kind == "refine" for t in tasks) == 9


def test_local_run_layout_and_skip(project):
    root, write = project
    cfg = Config.load(write())
    assert run_local(cfg) == {"done": 9}

    md = root / "results" / "none" / BUTANE / "etkdg"
    names = sorted(p.name for p in md.iterdir())
    assert names == sorted(
        ["params.json", "params_fake-lot.json"]
        + [f"raw_{i}.xyz" for i in (1, 2, 3)] + [f"raw_{i}_fake-lot.xyz" for i in (1, 2, 3)]
        + [f"run_{i}.json" for i in (1, 2, 3)] + [f"run_{i}_fake-lot.json" for i in (1, 2, 3)]
    )
    assert (root / "results" / "H" / ETHANOL / "etkdg" / "raw_1_fake-lot.xyz").exists()
    assert not (root / "results" / "H" / BUTANE).exists()
    assert (root / "results" / "molecules.csv").exists()

    # replicates use different seeds -> different geometries
    run1, run2 = (json.loads((md / f"run_{i}.json").read_text()) for i in (1, 2))
    assert run1["seed"] == 1 and run2["seed"] == 2
    assert (md / "raw_1.xyz").read_text() != (md / "raw_2.xyz").read_text()

    params = json.loads((md / "params.json").read_text())
    assert params["compared"]["params"]["n_confs"] == 4
    assert params["compared"]["transformation"]["name"] == "none"
    assert params["compared"]["structure"]["multiplicity"] == 1

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
    assert main(["submit", str(root / "config.yaml"), "--no-sbatch"]) == 2

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
    assert [(t.kind, t.inchikey, t.replicate) for t in pending] == [("chain", BUTANE, 2)]
    (md / "raw_3_fake-lot.xyz").unlink()
    assert len(expand(cfg)[0]) == 2


def test_slurm_scripts_split(project):
    root, write = project
    cfg = Config.load(write(mode="split"))
    jobdir = write_jobs(cfg, expand(cfg)[0], root / "jobs")
    sbatch = (jobdir / "search_etkdg.sbatch").read_text()
    assert "#SBATCH --array=0-8" in sbatch and "#SBATCH --partition=short" in sbatch
    assert "--chunk $SLURM_ARRAY_TASK_ID" in sbatch
    submit = (jobdir / "submit.sh").read_text()
    assert "--dependency=afterany$deps refine_fake-lot.sbatch" in submit
    assert len((jobdir / "refine_fake-lot.jsonl").read_text().splitlines()) == 9


def test_submit_chained_snapshot_and_run_task(project, capsys):
    root, write = project
    cfg_path = write()
    (root / "ligands.txt").write_text("# comment\nCCCC butane\n")
    assert main(["submit", str(cfg_path), "--smiles", str(root / "ligands.txt"), "-t", "none", "--dry-run"]) == 0
    assert not (root / "results").exists()

    assert main(["submit", str(cfg_path), "--smiles", str(root / "ligands.txt"), "-t", "none", "--no-sbatch"]) == 0
    jobdir = next((root / "results" / "_jobs").iterdir())
    snapshot = yaml.safe_load((jobdir / "config.yaml").read_text())
    assert list(snapshot["transformations"]) == ["none"] and snapshot["inputs"].endswith("ligands.txt")
    sbatch = (jobdir / "chain_etkdg.sbatch").read_text()
    assert str(jobdir / "config.yaml") in sbatch and "#SBATCH --array=0-2" in sbatch
    assert "dependency" not in (jobdir / "submit.sh").read_text()

    # what a SLURM array element does
    for i in range(3):
        assert main(["run-task", str(jobdir / "config.yaml"), "--tasks", str(jobdir / "chain_etkdg.jsonl"),
                     "--chunk", str(i)]) == 0
    md = root / "results" / "none" / BUTANE / "etkdg"
    assert all((md / f"raw_{i}_fake-lot.xyz").exists() for i in (1, 2, 3))
    capsys.readouterr()
    assert main(["status", str(cfg_path), "--smiles", str(root / "ligands.txt"), "-t", "none"]) == 0
    out = capsys.readouterr().out
    assert "done" in out and "missing" not in out


def test_multiplicity_override_renames_folder_and_is_carried(project):
    root, write = project
    (root / "ligands.txt").write_text("CCCC\n")
    assert main(["run", str(write()), "--smiles", str(root / "ligands.txt"), "-t", "none", "--multiplicity", "3"]) == 0
    md = root / "results" / "none_m3" / BUTANE / "etkdg"
    assert json.loads((md / "params.json").read_text())["compared"]["structure"]["multiplicity"] == 3
    assert json.loads((md / "run_1_fake-lot.json").read_text())["structure"]["multiplicity"] == 3
    assert not (root / "results" / "none").exists()
