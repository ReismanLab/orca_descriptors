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
from desc_utils.aggregate import aggregate_conformers, completeness
from desc_utils.extract import job_status_columns
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
        "path": ["DATA/x/M1/conf_1", "DATA/x/M1/conf_2", "DATA/x/M2/conf_1"], "inchi": ["M1", "M1", "M2"],
        "SCF_energy": [-1.0, -1.0, -2.0], "A": [1, 1, 1], "B": [2, 2, 2], "charge_NBO_A": [0.1, 0.3, 0.5],
        "smiles": ["s", "s", "t"], "orca_job_status": ["completed", "failed", "completed"],
        "n_conformer_folders": [2, 2, 3],
    })
    agg = aggregate_conformers(df, SPEC).set_index("inchi")
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
