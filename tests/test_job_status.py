"""ORCA job status: ``orca_parse.job_status`` on synthetic folders, its recording in the condensed file,
the rebuild rule of the condenser, and the completeness columns of the extract and aggregate stages."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desc_utils import condense as c
from desc_utils import orca_parse as op
from desc_utils.aggregate import aggregate_conformers, apply_filter, completeness, cost_summary
from desc_utils.extract import cost_columns_of, job_status_columns
from desc_utils.spec import validate_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "DATA" / "LNiH2_library_001_to_download_2026_03_24" / "AATATIAROAAKIN-SNVBAGLBSA-N" / "conf_1"
needs_fixture = pytest.mark.skipif(not (FIXTURE / "singlepoint.json").is_file(), reason="needs the tracked fixture conformer")

NORMAL = "                             ****ORCA TERMINATED NORMALLY****\n"
ERROR = "ORCA finished by error termination in SCF\n"


def _folder(tmp_path, name, out_text=None, error_text=None):
    d = tmp_path / name
    d.mkdir()
    if out_text is not None:
        (d / "calcParam.out").write_text(out_text)
    if error_text is not None:
        (d / "error_42.txt").write_text(error_text)
    return d


# --------------------------------------------------------------------------- orca_parse.job_status

def test_job_status_from_files(tmp_path):
    assert op.job_status(_folder(tmp_path, "a"))["status"] == "not_started"
    assert op.job_status(_folder(tmp_path, "b", "SCF ITERATIONS\n"))["status"] == "running"
    ok = op.job_status(_folder(tmp_path, "c", "stuff\n" + NORMAL + "TOTAL RUN TIME\n"))
    assert ok["status"] == "completed" and ok["detail"] == "" and ok["n_normal_terminations"] == 1
    bad = op.job_status(_folder(tmp_path, "d", "stuff\n" + ERROR))
    assert bad["status"] == "failed" and bad["detail"].startswith("ORCA finished by error termination")
    # calcParam.out is appended to: the last run decides, the history is reported
    twice = op.job_status(_folder(tmp_path, "e", NORMAL + "again\n" + NORMAL))
    assert twice["status"] == "completed" and twice["n_normal_terminations"] == 2 and "2 normal" in twice["detail"]
    later = op.job_status(_folder(tmp_path, "f", NORMAL + "again\n" + ERROR))
    assert later["status"] == "failed" and later["failed_after_success"]
    # SLURM killed the job before ORCA could say anything
    killed = op.job_status(_folder(tmp_path, "g", "SCF ITERATIONS\n", "slurmstepd: *** JOB 42 CANCELLED DUE TO TIME LIMIT ***"))
    assert killed == {**killed, "status": "failed", "detail": "killed by SLURM: time limit"}
    # the text can be handed over when the caller already read the file
    assert op.job_status(tmp_path / "c", text="x\n" + ERROR)["status"] == "failed"


def test_latest_error_file_wins(tmp_path):
    d = _folder(tmp_path, "h", "SCF\n")
    (d / "error_7.txt").write_text("DUE TO TIME LIMIT")
    (d / "error_12.txt").write_text("")            # the resubmission is still running
    assert op.latest_slurm_error_file(d).name == "error_12.txt"
    assert op.job_status(d)["status"] == "running"


# --------------------------------------------------------------------------- condensed file

def test_conformer_job_status_sources(tmp_path):
    cond = _folder(tmp_path, "with_file")
    (cond / c.CONDENSED_FILENAME).write_text(json.dumps({"job": {"status": "running"}}))
    assert c.conformer_job_status(cond) == "running"
    old = _folder(tmp_path, "old_file")
    (old / c.CONDENSED_FILENAME).write_text(json.dumps({"schema_version": 1}))
    assert c.conformer_job_status(old) == "unknown"
    raw = _folder(tmp_path, "raw_only", NORMAL)
    assert c.conformer_job_status(raw) == "completed"
    assert c.conformer_job_status(_folder(tmp_path, "empty")) == "missing"
    assert c.conformer_job_status(cond, condensed_file=None) == "missing"       # raw mode ignores the file
    assert c.stored_job_status({}) == "unknown" and c.stored_job_status({"job": {"status": "failed"}}) == "failed"


@needs_fixture
def test_condense_records_status_and_rebuilds_unfinished(tmp_path):
    conf = tmp_path / "MOL" / "conf_1"
    shutil.copytree(FIXTURE, conf)
    out_path = conf / "calcParam.out"
    full_text = out_path.read_text(errors="replace")
    assert full_text.count(op.ORCA_NORMAL) == 1

    # 1. the job is still writing calcParam.out: a file is written, and says so
    cut = full_text.rindex(op.ORCA_NORMAL)
    out_path.write_text(full_text[:cut])
    s1 = c.condense_tree([conf], log=None)
    assert s1["ok"] == 1 and s1["job_status"] == {"running": 1}
    data = c.load_condensed(conf / c.CONDENSED_FILENAME)
    assert data["job"]["status"] == "running" and "no termination line" in data["job"]["detail"]
    assert data["states"]["singlepoint"] is not None       # everything up to the cut is there
    assert not c.check_condensed(conf / c.CONDENSED_FILENAME)  # structurally consistent all the same

    # 2. a second pass does not keep an unfinished file
    s2 = c.condense_tree([conf], log=None)
    assert s2["ok"] == 1 and s2["skipped"] == 0 and "rebuilt, was running" in s2["results"][0][2]

    # 3. the job finished since: the rebuild picks the new status up, then the file is left alone
    out_path.write_text(full_text)
    s3 = c.condense_tree([conf], log=None)
    assert s3["ok"] == 1 and s3["job_status"] == {"completed": 1}
    assert c.load_condensed(conf / c.CONDENSED_FILENAME)["job"]["status"] == "completed"
    s4 = c.condense_tree([conf], log=None)
    assert s4["skipped"] == 1 and s4["results"][0][2] == "exists (completed)" and s4["job_status"] == {"completed": 1}

    # 4. an unfinished file without raw files next to it (laptop after rsync) cannot be rebuilt: kept as it is
    (conf / c.CONDENSED_FILENAME).write_text(json.dumps({**data, "job": {"status": "failed", "detail": "x"}}))
    out_path.unlink()
    s5 = c.condense_tree([conf], log=None)
    assert s5["skipped"] == 1 and "no raw files" in s5["results"][0][2] and s5["job_status"] == {"failed": 1}

    # 5. nothing to condense at all: the failure names the job state
    shutil.rmtree(conf)
    conf.mkdir()
    (conf / "calcParam.out").write_text(ERROR)
    s6 = c.condense_tree([conf], log=None)
    assert s6["failed"] == 1 and "[job failed]" in s6["results"][0][2]
    assert c.job_status_line(s6["job_status"]) == "job status: failed 1"


# --------------------------------------------------------------------------- run times

def _compound(steps, nprocs=8, total="TOTAL RUN TIME: 0 days 0 hours 1 minutes 5 seconds 500 msec\n"):
    """A compound output: ``steps`` is [(base or None, wall or None, {module: s})]."""
    text = "                                COMPOUND JOB MainJOB \nBasename: calcParam_Compound_0\n"
    text += f"|  2>  %pal\n|  3>      nprocs {nprocs}\n|  4>  end\n"
    for i, (base, wall, modules) in enumerate(steps, start=1):
        text += f"\n                                COMPOUND JOB  {i} \nBasename: calcParam_Compound_{i}\n"
        if base:
            text += f'|  1>       %base "{base}"\n'
        text += "| 3> ! pbe0 Opt\nTotal time                  ....     167.461 sec\nSum of individual times     ....     161.649 sec  ( 96.5%)\n"
        if wall is not None:
            text += "\nTimings for individual modules:\n\n"
            text += f"Sum of individual times          ...      {wall:.3f} sec (=   {wall / 60:.3f} min)\n"
            for name, sec in modules.items():
                text += f"{name:<33}...      {sec:.3f} sec (=   {sec / 60:.3f} min)  50.0 %\n"
            text += "\n\n"
    return text + NORMAL + total


def test_compound_step_timings():
    text = _compound([("optimization", 299.507, {"Startup calculation": 13.959, "SCF iterations": 248.393}),
                      ("singlepoint", 616.38, {"SCF iterations": 177.796}),
                      ("singlepoint", 10.0, {}),                 # a repeated %base name
                      (None, 4.0, {}),                           # a step without %base
                      ("nmr", None, {})])                        # did not reach its timings: unfinished
    t = op.compound_step_timings(text)
    assert t["total_run_time_s"] == 65.5 and t["nprocs"] == 8
    assert list(t["steps"]) == ["optimization", "singlepoint", "singlepoint_2", "step_4", "nmr"]
    assert t["steps"]["optimization"] == {"index": 1, "wall_s": 299.507,
                                          "modules": {"Startup calculation": 13.959, "SCF iterations": 248.393}}
    assert t["steps"]["singlepoint"]["wall_s"] == 616.38 and t["steps"]["singlepoint_2"]["wall_s"] == 10.0
    assert t["steps"]["nmr"] == {"index": 5, "wall_s": None, "modules": {}}
    # calcParam.out is appended to on resubmission: only the last run counts
    first = _compound([("optimization", 1.0, {})], nprocs=2, total="TOTAL RUN TIME: 0 days 0 hours 0 minutes 1 seconds 0 msec\n")
    t2 = op.compound_step_timings(first + text)
    assert t2["nprocs"] == 8 and t2["total_run_time_s"] == 65.5 and t2["steps"]["optimization"]["wall_s"] == 299.507
    # a plain (non-compound) output: one step called main; nothing at all: empty
    plain = "stuff\nTimings for individual modules:\n\nSum of individual times          ...        9.000 sec (=   0.150 min)\nSCF iterations                   ...        8.000 sec (=   0.133 min)  88.9 %\n\n" + NORMAL
    assert op.compound_step_timings(plain)["steps"] == {"main": {"index": 0, "wall_s": 9.0, "modules": {"SCF iterations": 8.0}}}
    assert op.compound_step_timings("SCF ITERATIONS\n") == {"total_run_time_s": None, "nprocs": None, "steps": {}}
    assert op.run_time_seconds("TOTAL RUN TIME: 1 days 2 hours 3 minutes 4 seconds 5 msec") == 93784.005
    assert op.run_time_seconds("no") is None


@needs_fixture
def test_condensed_file_records_step_timings():
    cond = c.build_condensed(FIXTURE)
    t = cond["job"]["timings"]
    assert isinstance(t["nprocs"], int) and t["total_run_time_s"] > 0
    assert {"optimization", "singlepoint"} <= set(t["steps"])
    assert all(st["wall_s"] > 0 for st in t["steps"].values())
    assert abs(sum(st["wall_s"] for st in t["steps"].values()) - t["total_run_time_s"]) < 0.05 * t["total_run_time_s"]
    assert cost_columns_of(cond["job"])["orca_cpu_hours"] == round(t["total_run_time_s"] * t["nprocs"] / 3600, 5)


# --------------------------------------------------------------------------- extract / aggregate

SPEC = validate_spec({
    "name": "t", "data": {"root": "DATA/x"}, "atoms": {"A": None, "B": None},
    "numbering": {"backend": "first_n"},
    "descriptors": {"descripytor": {"charges": ["A"]}},
    "aggregate": {"carry_first": ["smiles"]},
})


def test_job_status_columns_count_folders_on_disk(tmp_path):
    root = tmp_path / "LIB"
    for mol, confs in {"M1": {"conf_1": "completed", "conf_2": "running", "conf_3": None}, "M2": {"conf_1": "raw"}}.items():
        for conf, status in confs.items():
            d = root / mol / conf
            d.mkdir(parents=True)
            if status == "raw":
                (d / "calcParam.out").write_text(NORMAL)
            elif status:
                (d / c.CONDENSED_FILENAME).write_text(json.dumps({"job": {"status": status}}))
    (root / "M1" / "notes").mkdir()                                     # not a conformer folder
    paths = [str(root / "M1" / "conf_1"), str(root / "M1" / "conf_2"), str(root / "M2" / "conf_1"),
             str(root / "M2" / "conf_9")]                                  # conf_9 does not exist
    df = job_status_columns(paths, SPEC, REPO_ROOT).set_index("path")
    assert list(df.loc[paths, "orca_job_status"]) == ["completed", "running", "completed", "missing"]
    assert list(df.loc[paths[:3], "n_conformer_folders"]) == [3, 3, 1]    # M1: conf_3 has no file but is a folder
    assert np.isnan(df.loc[paths[3], "n_conformer_folders"])


def test_job_status_columns_carry_cost(tmp_path):
    root = tmp_path / "LIB"
    timings = {"total_run_time_s": 3600.0, "nprocs": 8,
               "steps": {"optimization": {"index": 1, "wall_s": 1000.0, "modules": {}},
                         "singlepoint": {"index": 2, "wall_s": 2500.0, "modules": {}}}}
    for conf, job in {"conf_1": {"status": "completed", "timings": timings},
                      "conf_2": {"status": "completed"},                                   # condensed before 2026-10-02
                      "conf_3": {"status": "failed", "timings": {"total_run_time_s": None, "nprocs": 8, "steps": {
                          "optimization": {"index": 1, "wall_s": 500.0, "modules": {}}}}}}.items():
        d = root / "M1" / conf
        d.mkdir(parents=True)
        (d / c.CONDENSED_FILENAME).write_text(json.dumps({"job": job}))
    raw = root / "M1" / "conf_4"
    raw.mkdir()
    (raw / "calcParam.out").write_text(_compound([("optimization", 100.0, {})], nprocs=4))
    paths = [str(root / "M1" / f"conf_{i}") for i in (1, 2, 3, 4)]
    df = job_status_columns(paths, SPEC, REPO_ROOT).set_index("path")
    assert list(df.columns[:2]) == ["orca_job_status", "n_conformer_folders"]
    r1 = df.loc[paths[0]]
    assert r1["orca_run_time_s"] == 3600.0 and r1["orca_nprocs"] == 8 and r1["orca_cpu_hours"] == 8.0
    assert r1["orca_step_optimization_s"] == 1000.0 and r1["orca_step_singlepoint_s"] == 2500.0
    assert df.loc[paths[1], ["orca_run_time_s", "orca_nprocs", "orca_cpu_hours", "orca_step_optimization_s"]].isna().all()
    r3 = df.loc[paths[2]]
    assert np.isnan(r3["orca_cpu_hours"]) and r3["orca_step_optimization_s"] == 500.0 and np.isnan(r3["orca_step_singlepoint_s"])
    r4 = df.loc[paths[3]]                                        # raw folder: read from calcParam.out
    assert r4["orca_job_status"] == "completed" and r4["orca_nprocs"] == 4 and r4["orca_step_optimization_s"] == 100.0
    assert r4["orca_cpu_hours"] == round(65.5 * 4 / 3600, 5)
    # a table without any timing has the three fixed columns anyway, NaN
    empty = job_status_columns([str(root / "M1" / "conf_2")], SPEC, REPO_ROOT)
    assert {"orca_run_time_s", "orca_nprocs", "orca_cpu_hours"} <= set(empty.columns) and empty["orca_cpu_hours"].isna().all()


def test_cost_summary_and_aggregation():
    df = pd.DataFrame({
        "path": ["DATA/x/M1/conf_1", "DATA/x/M1/conf_2", "DATA/x/M2/conf_1"], "inchikey": ["M1", "M1", "M2"],
        "SCF_energy": [-1.0, -1.0, -2.0], "A": [1, 1, 1], "B": [2, 2, 2], "charge_NBO_A": [0.1, 0.3, 0.5],
        "smiles": ["s", "s", "t"], "orca_job_status": ["completed", "completed", "completed"],
        "n_conformer_folders": [2, 2, 1],
        "orca_run_time_s": [3600.0, 1800.0, np.nan], "orca_nprocs": [8, 8, np.nan], "orca_cpu_hours": [8.0, 4.0, np.nan],
        "orca_step_optimization_s": [1000.0, 500.0, np.nan], "orca_step_singlepoint_s": [2000.0, np.nan, np.nan],
    })
    assert cost_summary(df[df["inchikey"] == "M1"]) == {
        "dft_cpu_hours": 12.0, "dft_optimization_cpu_hours": round(1500 * 8 / 3600, 5),
        "dft_singlepoint_cpu_hours": round(2000 * 8 / 3600, 5)}
    m2 = cost_summary(df[df["inchikey"] == "M2"])
    assert all(np.isnan(v) for v in m2.values()) and set(m2) == {"dft_cpu_hours", "dft_optimization_cpu_hours",
                                                                   "dft_singlepoint_cpu_hours"}
    agg = aggregate_conformers(df, SPEC).set_index("inchikey")
    assert agg.loc["M1", "dft_cpu_hours"] == 12.0 and np.isnan(agg.loc["M2", "dft_cpu_hours"])
    assert "charge_NBO_A_boltz" in agg.columns
    # the per-conformer cost columns are bookkeeping, not descriptors
    assert not [col for col in agg.columns if col.startswith("orca_")]
    # and the molecule's cost survives the descriptor filter
    filtered, _ = apply_filter(agg.reset_index(), SPEC)
    assert {"dft_cpu_hours", "dft_optimization_cpu_hours", "dft_singlepoint_cpu_hours"} <= set(filtered.columns)
    assert list(filtered["dft_cpu_hours"])[:1] == [12.0]


def test_completeness_flag():
    def g(statuses, folders=None):
        return pd.DataFrame({"orca_job_status": statuses,
                             "n_conformer_folders": folders if folders is not None else [len(statuses)] * len(statuses)})
    assert completeness(g(["completed", "completed"])) == \
        {"n_conformers_completed": 2, "n_conformer_folders": 2, "all_conformers_completed": True}
    assert completeness(g(["completed", "running"]))["all_conformers_completed"] is False
    assert completeness(g(["completed", "failed"]))["n_conformers_completed"] == 1
    # a folder on disk that never produced a geometry does not appear as a row: the folder count catches it
    r = completeness(g(["completed", "completed"], folders=[3, 3]))
    assert r["n_conformer_folders"] == 3 and r["all_conformers_completed"] is False
    # files condensed before the status existed: no verdict rather than a wrong one
    assert completeness(g(["completed", "unknown"]))["all_conformers_completed"] is pd.NA
    # flat layout (NaN folder count): the rows are the folders
    r = completeness(g(["completed", "completed"], folders=[np.nan, np.nan]))
    assert r["n_conformer_folders"] == 2 and r["all_conformers_completed"] is True


def test_aggregate_carries_completeness_and_does_not_aggregate_counts():
    df = pd.DataFrame({
        "path": ["DATA/x/M1/conf_1", "DATA/x/M1/conf_2", "DATA/x/M2/conf_1"], "inchikey": ["M1", "M1", "M2"],
        "SCF_energy": [-1.0, -1.0, -2.0], "A": [1, 1, 1], "B": [2, 2, 2], "charge_NBO_A": [0.1, 0.3, 0.5],
        "smiles": ["s", "s", "t"], "orca_job_status": ["completed", "failed", "completed"],
        "n_conformer_folders": [2, 2, 3],
    })
    agg = aggregate_conformers(df, SPEC).set_index("inchikey")
    assert list(agg.loc["M1", ["n_conformers", "n_conformers_completed", "n_conformer_folders"]]) == [2, 1, 2]
    assert agg.loc["M1", "all_conformers_completed"] is False or agg.loc["M1", "all_conformers_completed"] == False  # noqa: E712
    assert list(agg.loc["M2", ["n_conformers", "n_conformers_completed", "n_conformer_folders"]]) == [1, 1, 3]
    assert not [col for col in agg.columns if col.startswith("n_conformer_folders_")]
    assert "charge_NBO_A_boltz" in agg.columns
    # without the status column the table is exactly what it used to be
    plain = aggregate_conformers(df.drop(columns=["orca_job_status", "n_conformer_folders"]), SPEC)
    assert "all_conformers_completed" not in plain.columns and "n_conformers" in plain.columns


def test_conformers_to_check_lists_reasons():
    from desc_utils.extract import conformers_to_check
    df = pd.DataFrame({
        "path": ["L/M1/conf_1", "L/M1/conf_2", "L/M2/conf_1", "L/M2/conf_2", "L/M3/conf_1"],
        "numbering_status": ["ok", "failed", "ok", "incomplete", "ok"],
        "orca_job_status": ["completed", "completed", "running", "failed", "unknown"],
    })
    num = pd.DataFrame({"path": ["L/M1/conf_2/", "L/M2/conf_2/"], "numbering_notes": ["no match", "unassigned labels: ['B']"]})
    out = conformers_to_check(df, num)
    assert list(out.columns) == ["path", "reason", "numbering_status", "numbering_notes", "orca_job_status"]
    assert list(out["path"]) == ["L/M1/conf_2", "L/M2/conf_1", "L/M2/conf_2"]      # unknown status is not a reason
    assert list(out["reason"]) == ["numbering failed", "ORCA job running", "numbering incomplete; ORCA job failed"]
    assert out.loc[0, "numbering_notes"] == "no match" and pd.isna(out.loc[1, "numbering_notes"])
    assert conformers_to_check(df.iloc[[0]]).empty


def test_read_numbering_table_unfiltered_keeps_bad_rows_for_the_check(tmp_path):
    """The CLI reads every row (usable=False) so that a conformer whose numbering failed still reaches the check."""
    from desc_utils.extract import conformers_to_check
    from desc_utils.numbering import read_numbering_table, usable_rows
    csv = tmp_path / "num.csv"
    pd.DataFrame({"path": ["L/M/conf_1/", "L/M/conf_2/"], "inchikey": ["M", "M"], "SCF_energy": [-1.0, -1.0],
                  "A": [1, np.nan], "B": [2, np.nan], "numbering_status": ["ok", "failed"],
                  "numbering_notes": ["", "stage 'core': no SMARTS match"]}).to_csv(csv, index=False)
    every = read_numbering_table(csv, SPEC, usable=False)
    assert len(every) == 2 and len(read_numbering_table(csv, SPEC)) == 1
    assert len(usable_rows(every, SPEC, quiet=True)) == 1 and usable_rows(every, SPEC, quiet=True)["A"].dtype == int
    check = conformers_to_check(every[["path", "numbering_status"]], every)
    assert list(check["path"]) == ["L/M/conf_2/"] and check.loc[0, "reason"] == "numbering failed"
    assert check.loc[0, "numbering_notes"] == "stage 'core': no SMARTS match"
