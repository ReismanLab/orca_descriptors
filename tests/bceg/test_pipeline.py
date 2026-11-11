from __future__ import annotations

import hashlib
import json
import os
import signal

import pytest
import yaml

from bceg.cli import main
from bceg.config import Config
from bceg.inputs import InputMolecule
from bceg.slurm import write_jobs
from bceg.refine import REFINERS
from bceg.refine.ase_opt import _CheckpointWriter, load_checkpoint
from bceg.provenance import ParamsMismatch
from bceg.tasks import Interrupted, expand, handle_sigterm, run_local, run_task
from bceg.xyz import Conformer, read_xyz
from bceg.transforms import BUILTIN_TRANSFORMS

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


def test_compute_usage_recorded(project, monkeypatch):
    root, write = project
    monkeypatch.setattr("bceg.tasks.allocated_cpus", lambda: 8)
    monkeypatch.setattr("bceg.tasks.gpu_names", lambda: ("NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"))
    run_local(Config.load(write()))

    md = root / "results" / "none" / BUTANE / "etkdg"
    for name in ("run_1.json", "run_1_fake-lot.json"):  # search and post-optimisation both account
        rec = json.loads((md / name).read_text())
        assert rec["cpus"] == 8 and rec["gpus"] == ["NVIDIA A100-SXM4-80GB"] * 2
        assert isinstance(rec["cpu_model"], str)
        assert rec["cpu_hours"] == round(rec["wall_time_s"] / 3600 * 8, 5)
        assert rec["gpu_hours"] == round(rec["wall_time_s"] / 3600 * 2, 5)


def test_compute_usage_recorded_on_failure(project, monkeypatch):
    """A run that dies still costs core-hours; the failed record must carry them."""
    root, write = project
    monkeypatch.setattr("bceg.tasks.allocated_cpus", lambda: 4)
    monkeypatch.setattr("bceg.tasks.gpu_names", lambda: ())
    run_local(Config.load(write(params="{n_confs: 0}")))
    rec = json.loads((root / "results" / "none" / BUTANE / "etkdg" / "run_1.json").read_text())
    assert rec["status"] == "failed"
    assert rec["cpus"] == 4 and rec["gpus"] == [] and "gpu_hours" not in rec
    assert rec["cpu_hours"] == round(rec["wall_time_s"] / 3600 * 4, 5)


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
    assert "#SBATCH --signal=B:TERM@600" in sbatch and "\nexec bceg run-task " in sbatch   # the warning reaches Python
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


def test_failed_start_geometry_keeps_workdir_files(project, fake_xtb, monkeypatch):
    from bceg import geometry

    root, write = project
    monkeypatch.setattr(geometry, "coordination_problems", lambda *a: ["forced"])
    cfg = Config.load(write("{start_max_attempts: 2}", searcher="single_start", transformations="{none: null}"))
    counts = run_local(cfg)
    assert counts == {"failed": 6}
    md = root / "results" / "none" / BUTANE / "etkdg"
    rec = json.loads((md / "run_1.json").read_text())
    assert rec["status"] == "failed" and "after 2 attempt" in rec["error"]
    failed = md / "logs" / "raw_1_failed"
    assert (failed / "preopt_0" / "xtb.out").exists() and (failed / "preopt_1" / "xtbopt.xyz").exists()


def test_start_geometry_rmsd_to_other_replicates(project, fake_xtb):
    root, write = project
    cfg = Config.load(write("{}", searcher="single_start", transformations="{none: null}"))
    run_local(cfg)
    md = root / "results" / "none" / BUTANE / "etkdg"
    assert (md / "logs" / "raw_3_start_geometry.xyz").exists()
    rec3 = json.loads((md / "run_3.json").read_text())
    assert set(rec3["details"]["start_geometry"]["rmsd_to_replicates"]) == {"1", "2"}


def test_not_applicable_methods_create_no_folder(project, tmp_path):
    from pathlib import Path

    root, write = project
    script = BUILTIN_TRANSFORMS / "ni_x2.py"
    (root / "inputs.csv").write_text("smiles\nc1ccc(-c2ccccn2)nc1\n")
    cfg = Config.load(write("{}", searcher="loqi", transformations=f"{{NiF2: {{script: {script}, params: {{x: F}}}}}}"))
    assert expand(cfg)[0] == []
    assert not (root / "results").exists() or not any((root / "results").rglob("loqi"))


def test_slurm_config_checks_and_chain_overrides(project):
    import pytest

    root, write = project
    path = write()
    raw = yaml.safe_load(path.read_text())
    raw["slurm"]["task_per_job"] = 4  # typo of tasks_per_job
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown keys"):
        Config.load(path)
    cfg = Config.load(write(method_slurm="{time: '48:00:00'}", refiner_slurm="{time: '1:00:00', mem: 64G}"))
    jobdir = write_jobs(cfg, expand(cfg)[0], root / "jobs")
    sbatch = (jobdir / "chain_etkdg.sbatch").read_text()
    assert "--time=48:00:00" in sbatch and "--mem=64G" in sbatch
    assert "--nodes" not in sbatch                                          # one task: nothing to keep together

    # several tasks (GOAT's MPI ranks) stay on one node unless the profile says otherwise
    cfg = Config.load(write(method_slurm="{ntasks: 4, cpus_per_task: 1}"))
    sbatch = (write_jobs(cfg, expand(cfg)[0], root / "jobs4") / "chain_etkdg.sbatch").read_text()
    assert "--ntasks=4" in sbatch and "--nodes=1" in sbatch
    cfg = Config.load(write(method_slurm="{ntasks: 4, nodes: 2}"))
    sbatch = (write_jobs(cfg, expand(cfg)[0], root / "jobs2") / "chain_etkdg.sbatch").read_text()
    assert "--nodes=2" in sbatch and "--nodes=1" not in sbatch


def test_search_record_phases(project, fake_xtb, monkeypatch):
    """run_<i>.json says pre_refinement while the start geometry is made, search afterwards; a failure keeps its phase."""
    from bceg import geometry
    from bceg.search import SEARCHERS

    root, write = project
    cfg = Config.load(write("{}", searcher="single_start", transformations="{none: null}"))
    phases = []
    md = root / "results" / "none" / BUTANE / "etkdg"
    real_run = SEARCHERS["single_start"].run

    def spying_run(self, structure, replicate, workdir):
        phases.append(json.loads((md / f"run_{replicate}.json").read_text())["phase"])   # before the start geometry
        out = real_run(self, structure, replicate, workdir)
        phases.append(json.loads((md / f"run_{replicate}.json").read_text())["phase"])   # after it
        return out

    monkeypatch.setattr(SEARCHERS["single_start"], "run", spying_run)
    run_local(cfg)
    assert phases[:2] == ["pre_refinement", "search"]
    rec = json.loads((md / "run_1.json").read_text())
    assert rec["status"] == "done" and rec["phase"] == "search" and rec["details"]["start_geometry"]["attempts"] == 1
    # a failure in the start geometry is recorded in that phase
    monkeypatch.setattr(geometry, "coordination_problems", lambda *a: ["forced"])
    cfg2 = Config.load(write("{start_max_attempts: 1}", searcher="single_start", transformations="{none: null}"))
    (root / "results").rename(root / "results_done")
    run_local(cfg2)
    rec = json.loads((md / "run_1.json").read_text())
    assert rec["status"] == "failed" and rec["phase"] == "pre_refinement"
    # a searcher without a start step is in phase search from the start
    (root / "results").rename(root / "results_failed")
    cfg3 = Config.load(write("{n_confs: 2}", transformations="{none: null}"))
    run_local(cfg3)
    assert json.loads((md / "run_1.json").read_text())["phase"] == "search"


def test_search_usage_is_split_by_phase(project, fake_xtb, monkeypatch):
    """run_<i>.json carries the usage per phase under ``phases``, adding up to the total."""
    from bceg import geometry

    root, write = project
    monkeypatch.setattr("bceg.tasks.allocated_cpus", lambda: 4)
    monkeypatch.setattr("bceg.tasks.gpu_names", lambda: ("GPU-X",))
    md = root / "results" / "none" / BUTANE / "etkdg"

    run_local(Config.load(write("{}", searcher="single_start", transformations="{none: null}")))
    rec = json.loads((md / "run_1.json").read_text())
    ph = rec["phases"]
    assert set(ph) == {"pre_refinement", "search"}
    for part in ph.values():
        assert part["cpu_hours"] == round(part["wall_time_s"] / 3600 * 4, 5)
        assert part["gpu_hours"] == round(part["wall_time_s"] / 3600, 5)
    assert abs(ph["pre_refinement"]["wall_time_s"] + ph["search"]["wall_time_s"] - rec["wall_time_s"]) < 0.05
    assert "phases" not in json.loads((md / "run_1_fake-lot.json").read_text())   # one step, no split

    # died while making the start geometry: everything was pre-refinement
    monkeypatch.setattr(geometry, "coordination_problems", lambda *a: ["forced"])
    (root / "results").rename(root / "results_done")
    run_local(Config.load(write("{start_max_attempts: 1}", searcher="single_start", transformations="{none: null}")))
    rec = json.loads((md / "run_1.json").read_text())
    assert rec["status"] == "failed" and list(rec["phases"]) == ["pre_refinement"]
    assert rec["phases"]["pre_refinement"]["wall_time_s"] == rec["wall_time_s"]

    # no start step: everything is search
    (root / "results").rename(root / "results_failed")
    run_local(Config.load(write("{n_confs: 2}", transformations="{none: null}")))
    rec = json.loads((md / "run_1.json").read_text())
    assert list(rec["phases"]) == ["search"] and rec["phases"]["search"]["cpu_hours"] == rec["cpu_hours"]


def test_pool_mem_is_sized_from_the_cpus():
    import pytest

    from bceg.refine import REFINERS
    from bceg.slurm import _mem_gb, with_pool_mem

    uma = REFINERS["uma"]()
    assert with_pool_mem({"cpus_per_task": 8}, uma, "chain")["mem"] == "35G"       # (8 + 1) x 3.6 + 2
    assert with_pool_mem({"ntasks": 4, "cpus_per_task": 1}, uma, "chain")["mem"] == "20G"
    assert with_pool_mem({"cpus_per_task": 8}, REFINERS["uma"](workers=2), "c")["mem"] == "13G"
    assert "mem" not in with_pool_mem({"cpus_per_task": 8}, REFINERS["uma"](device="cuda"), "c")
    assert "mem" not in with_pool_mem({"cpus_per_task": 8}, REFINERS["aimnet2"](), "c")   # no estimate
    assert "mem" not in with_pool_mem({"cpus_per_task": 8}, None, "c")
    assert with_pool_mem({"cpus_per_task": 8, "mem": "48G"}, uma, "c")["mem"] == "48G"
    with pytest.warns(UserWarning, match="mem 16G is below the ~35G"):
        assert with_pool_mem({"cpus_per_task": 8, "mem": "16G"}, uma, "c")["mem"] == "16G"   # kept, but said
    assert _mem_gb("16G") == 16 and _mem_gb("16384") == 16 and _mem_gb("16384M") == 16 and _mem_gb("1T") == 1024


FakeRefiner = REFINERS["fake"]   # registered by conftest


class ResumableFake(FakeRefiner):
    """The fake refiner with the checkpoint of the ASE refiners; ``stop_after`` conformers, then SIGTERM."""

    resumable = True
    stop_after: int | None = None
    computed: dict = {}

    def run(self, structure, conformers, workdir, checkpoint=None):
        path, key = checkpoint
        done = load_checkpoint(path, key)
        write = _CheckpointWriter(path, key, resumed=bool(done))
        mine = self.computed.setdefault(path, [])
        try:
            for i, c in enumerate(conformers):
                if i in done:
                    continue
                if self.stop_after is not None and len(mine) >= self.stop_after:
                    raise Interrupted("SIGTERM (time limit or scancel)")
                done[i] = {"index": i, "coords": c.coords, "energy_ev": -float(i)}
                write(done[i])
                mine.append(i)
        finally:
            write.close()
        return FakeRefiner.run(self, structure, conformers, workdir)


def test_an_interrupted_post_optimisation_resumes_from_its_checkpoint(project, monkeypatch):
    """Killed after 2 conformers: the record says failed with what was kept, the checkpoint stays; the next
    run optimises only the others, writes the ensemble and removes the checkpoint."""
    root, write = project
    monkeypatch.setitem(REFINERS, "fake", ResumableFake)
    monkeypatch.setattr(ResumableFake, "computed", {})
    cfg_path = write()
    monkeypatch.setattr(ResumableFake, "stop_after", 2)
    run_local(Config.load(cfg_path))
    md = root / "results" / "none" / BUTANE / "etkdg"
    rec = json.loads((md / "run_1_fake-lot.json").read_text())
    partial = md / "raw_1_fake-lot.partial.jsonl"
    assert rec["status"] == "failed" and "Interrupted" in rec["error"]
    assert rec["checkpoint"] == partial.name and rec["n_checkpointed"] == 2 and partial.exists()
    assert not (md / "raw_1_fake-lot.xyz").exists()

    monkeypatch.setattr(ResumableFake, "stop_after", None)
    run_local(Config.load(cfg_path))
    rec = json.loads((md / "run_1_fake-lot.json").read_text())
    n = len(read_xyz(md / "raw_1.xyz")[1])
    assert rec["status"] == "done" and rec["n_conformers"] == n and not partial.exists()
    assert ResumableFake.computed[partial] == list(range(n))           # 0, 1 before the kill, the rest after


def test_sigterm_becomes_interrupted_once():
    handle_sigterm()
    try:
        with pytest.raises(Interrupted):
            os.kill(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN    # a second TERM must not interrupt the record
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


# --------------------------------------------------------------------------- continue_from
def _refine_again(write, **kw):
    """A second config refining the `first` lot into `second`, in the same results_dir."""
    return Config.load(write(name="refine.yaml", **kw))


def test_continue_from_refines_the_previous_lot_and_leaves_it_alone(project):
    """The second lot reads raw_<i>_<first>.xyz, not the search output, and writes a file of its own."""
    root, write = project
    first = Config.load(write(refiner_params="{label: coarse, keep: 2}"))     # 4 conformers in, 2 out
    assert run_local(first) == {"done": 9}
    md = root / "results" / "none" / BUTANE / "etkdg"
    before = {p.name: p.read_bytes() for p in md.iterdir() if p.is_file()}

    second = _refine_again(write, refiner_params="{label: tight}", continue_from="coarse")
    assert second.refine_from == "coarse" and not second.chained      # never chained behind a search
    tasks, problems = expand(second)
    assert not problems and [t.kind for t in tasks] == ["refine"] * 9  # no search task: it is already done
    assert run_local(second) == {"done": 9}

    # it read the 2-conformer coarse lot, not the 4-conformer search output
    rec = json.loads((md / "run_1_tight.json").read_text())
    assert rec["continued_from"] == {"lot": "coarse", "file": "raw_1_coarse.xyz", "n_input": 2}
    assert rec["details"]["n_input"] == 2 and rec["n_conformers"] == 2
    assert len(read_xyz(md / "raw_1_tight.xyz")[1]) == 2

    # the first lot and the search output are untouched
    assert {p.name: p.read_bytes() for p in md.iterdir() if p.is_file() and p.name in before} == before
    assert (md / "params_tight.json").exists()


def test_continue_from_chains_the_provenance_and_keeps_old_hashes(project):
    """params_<lot>.json records the source lot and its hash; without continue_from the key is absent,
    so a params file written before the feature keeps the hash it has."""
    root, write = project
    first = Config.load(write(refiner_params="{label: coarse}"))
    run_local(first)
    md = root / "results" / "none" / BUTANE / "etkdg"
    coarse = json.loads((md / "params_coarse.json").read_text())
    assert "continue_from" not in coarse["compared"]

    run_local(_refine_again(write, refiner_params="{label: tight}", continue_from="coarse"))
    tight = json.loads((md / "params_tight.json").read_text())
    assert tight["compared"]["continue_from"] == {"lot": "coarse", "params_sha256": coarse["sha256"]}
    assert tight["compared"]["search_params_sha256"] == json.loads((md / "params.json").read_text())["sha256"]
    # the source lot's own log is not rewritten
    assert json.loads((md / "params_coarse.json").read_text()) == coarse


def test_continue_from_an_unfinished_lot_warns_and_skips_that_replicate(project):
    """A replicate whose source lot is not finished has nothing pending: it is warned about and skipped,
    not queued (it could only wait) and not reported as a params mismatch (that would abort the submit,
    and an unfinished source lot is a normal transient state)."""
    root, write = project
    run_local(Config.load(write(refiner_params="{label: coarse}")))
    md = root / "results" / "none" / BUTANE / "etkdg"
    (md / "raw_2_coarse.xyz").unlink()

    second = _refine_again(write, refiner_params="{label: tight}", continue_from="coarse")
    with pytest.warns(UserWarning, match="raw_2_coarse.xyz"):
        tasks, problems = expand(second)
    assert problems == []                                             # not a params mismatch
    assert len(tasks) == 8                                            # the other 8 replicates still run
    assert not any(t.inchikey == BUTANE and t.replicate == 2 for t in tasks)


def test_continue_from_a_lot_recomputed_with_other_parameters_is_refused(project):
    """The source lot's run record must match its params_<lot>.json, or the geometries are not the ones
    this refinement's hash says it started from."""
    root, write = project
    run_local(Config.load(write(refiner_params="{label: coarse}")))
    md = root / "results" / "none" / BUTANE / "etkdg"
    rec = json.loads((md / "run_1_coarse.json").read_text())
    (md / "run_1_coarse.json").write_text(json.dumps(rec | {"params_sha256": "0" * 64}))

    second = _refine_again(write, refiner_params="{label: tight}", continue_from="coarse")
    tasks, _ = expand(second)
    with pytest.raises(ParamsMismatch, match="raw_1_coarse.xyz"):
        run_task(second, next(t for t in tasks if t.inchikey == BUTANE and t.replicate == 1))


def test_a_checkpoint_of_the_source_lot_is_not_reused_by_the_continued_one(project, monkeypatch):
    """The checkpoint key hashes the file the step actually reads, so the coarse lot's checkpoint is
    ignored by the lot continued from it even though both are for the same replicate."""
    root, write = project
    monkeypatch.setitem(REFINERS, "fake", ResumableFake)
    run_local(Config.load(write(refiner_params="{label: coarse}")))
    md = root / "results" / "none" / BUTANE / "etkdg"

    second = _refine_again(write, refiner_params="{label: tight}", continue_from="coarse")
    tasks, _ = expand(second)                                    # writes params_tight.json
    rp_sha = json.loads((md / "params_tight.json").read_text())["sha256"]

    # a checkpoint that would be valid if this step read the search output: 1 conformer already "done"
    partial = md / "raw_1_tight.partial.jsonl"
    from_search = hashlib.sha256((md / "raw_1.xyz").read_bytes() + rp_sha.encode()).hexdigest()
    partial.write_text(json.dumps({"key": from_search}) + "\n"
                       + json.dumps({"index": 0, "coords": [[0.0, 0.0, 0.0]], "energy_ev": 0.0}) + "\n")
    ResumableFake.computed.pop(partial, None)

    run_task(second, next(t for t in tasks if t.inchikey == BUTANE and t.replicate == 1))
    assert json.loads((md / "run_1_tight.json").read_text())["status"] == "done"
    # every conformer was recomputed: conformer 0 was NOT taken from the stale checkpoint
    assert ResumableFake.computed[partial] == [0, 1, 2, 3]
    assert not partial.exists()                                  # removed once raw_1_tight.xyz is written


def test_a_gfn2_lot_then_a_better_one_continues_from_it(project, fake_xtb):
    """The cascade the xtb refiner exists for: an embedding searcher's geometries are cleaned up with
    GFN2-xTB, and the expensive refiner continues from that lot instead of from the raw search output."""
    root, write = project
    cheap = Config.load(write(refiner="xtb", refiner_params="{gfn: gfn2}"))
    assert cheap.refiner.lot == "gfn2"                       # the lot follows the method, unasked
    assert run_local(cheap) == {"done": 9}
    md = root / "results" / "none" / BUTANE / "etkdg"
    assert (md / "raw_1_gfn2.xyz").exists() and (md / "params_gfn2.json").exists()

    better = Config.load(write(name="cascade.yaml", refiner_params="{label: tight}", continue_from="gfn2"))
    assert run_local(better) == {"done": 9}

    rec = json.loads((md / "run_1_tight.json").read_text())
    assert rec["continued_from"]["lot"] == "gfn2" and rec["continued_from"]["file"] == "raw_1_gfn2.xyz"
    chain = json.loads((md / "params_tight.json").read_text())["compared"]
    assert chain["continue_from"]["params_sha256"] == json.loads((md / "params_gfn2.json").read_text())["sha256"]
    # three files, three stages: search output, the cheap lot, the better one
    assert {p.name for p in md.glob("raw_1*.xyz")} == {"raw_1.xyz", "raw_1_gfn2.xyz", "raw_1_tight.xyz"}


def test_an_interrupted_xtb_refinement_keeps_the_conformers_it_finished(project, fake_xtb, monkeypatch):
    """The real xtb refiner through bceg.tasks: a walltime kill must not throw away finished conformers.

    Before the checkpoint these refiners restarted from conformer 0, so a job killed near the end of a
    long library lost everything it had computed.
    """
    from bceg.refine.xtb_opt import XtbRefiner

    root, write = project
    cfg_path = write(refiner="xtb", refiner_params="{gfn: gfn2}")
    md = root / "results" / "none" / BUTANE / "etkdg"
    partial = md / "raw_1_gfn2.partial.jsonl"

    real = XtbRefiner.optimize_one
    state = {"n": 0}

    def stop_after_two(self, structure, index, conf, workdir, exe, threads, fixed):
        if state["n"] >= 2:
            raise Interrupted("SIGTERM (time limit or scancel)")
        state["n"] += 1
        return real(self, structure, index, conf, workdir, exe, threads, fixed)

    monkeypatch.setattr(XtbRefiner, "optimize_one", stop_after_two)
    run_local(Config.load(cfg_path))

    rec = json.loads((md / "run_1_gfn2.json").read_text())
    assert rec["status"] == "failed" and "Interrupted" in rec["error"]
    assert rec["checkpoint"] == partial.name and rec["n_checkpointed"] == 2
    assert partial.exists() and not (md / "raw_1_gfn2.xyz").exists()

    monkeypatch.setattr(XtbRefiner, "optimize_one", real)   # the resubmitted job
    assert run_local(Config.load(cfg_path)) == {"done": 9}

    rec = json.loads((md / "run_1_gfn2.json").read_text())
    n = len(read_xyz(md / "raw_1.xyz")[1])
    assert rec["status"] == "done" and rec["n_conformers"] == n
    assert rec["details"]["n_resumed"] == 2          # the two from before the kill were not recomputed
    assert not partial.exists()                      # removed once raw_1_gfn2.xyz is written
    assert len(read_xyz(md / "raw_1_gfn2.xyz")[1]) == n
