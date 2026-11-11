"""GOAT must get all its MPI workers: a split or short allocation silently under-samples the ensemble."""

from __future__ import annotations

import stat

import numpy as np
import pytest

from bceg.external import allocated_cpus, resolve_threads
from bceg.search.goat import Goat, check_allocation, worker_counts
from bceg.structure import Structure


def _log(launched: list[int], defined: int = 8, filled: int = 0) -> str:
    """A GOAT log's worker set-up and its first two iterations' worker tables."""
    lines = [f"  Final number of workers                    ... {defined}",
             "  Number of available CPUs                   ... 4"]
    for it in range(2):
        lines += ["All nodes which are allocated for this job are already filled."] * filled
        lines += [f"   {w}     2903.97    204.00    1.00 : 0.50     127        1    goat.goat.{it}.{w}.out" for w in launched]
    return "\n".join(lines) + "\n"


def _clear_slurm(monkeypatch):
    for var in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_CPUS_ON_NODE", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(var, raising=False)


def test_worker_counts_from_the_log():
    assert worker_counts(_log(list(range(8)))) == {"defined": 8, "launched": 8}
    assert worker_counts(_log([0, 1], filled=6)) == {"defined": 8, "launched": 2}
    assert worker_counts("no worker table here") is None


def test_ntasks_allocation_counts_every_task(monkeypatch):
    """ntasks: 8, cpus_per_task: 1 is 8 CPUs, not SLURM_CPUS_PER_TASK's 1 (thread defaults, core-hours)."""
    _clear_slurm(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert allocated_cpus() == 8 and resolve_threads(None) == 8
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4")             # 8 tasks over two nodes: 4 here
    assert resolve_threads(None) == 4 and resolve_threads(3) == 3


def test_allocation_check(monkeypatch):
    _clear_slurm(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    check_allocation(8)
    with pytest.raises(RuntimeError, match="exceeds the 8 CPU"):
        check_allocation(16)
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    with pytest.raises(RuntimeError, match="spans 2"):
        check_allocation(4)


@pytest.fixture
def fake_goat(tmp_path, monkeypatch):
    """An 'orca' that writes a one-conformer final ensemble and the log in $GOAT_LOG."""
    _clear_slurm(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    exe = tmp_path / "bin" / "orca"
    exe.parent.mkdir()
    exe.write_text('#!/bin/sh\ncp input.xyz goat.finalensemble.xyz\ncat "$GOAT_LOG"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    structure = Structure("CC")
    coords = np.zeros((len(structure.elements), 3)) + np.arange(len(structure.elements))[:, None]
    monkeypatch.setattr(Goat, "start_geometry", lambda self, s, seed, wd: (coords, "etkdg"))

    def run(launched, workdir_name):
        log = tmp_path / f"{workdir_name}.log"
        log.write_text(_log(launched))
        monkeypatch.setenv("GOAT_LOG", str(log))
        wd = tmp_path / workdir_name
        wd.mkdir()
        return Goat(orca_cmd=str(exe), nprocs=8).run(structure, 1, wd)
    return run


def test_goat_with_every_worker_records_them(fake_goat):
    result = fake_goat(list(range(8)), "full")
    assert result.details["workers"] == {"defined": 8, "launched": 8} and len(result.conformers) == 1


def test_goat_missing_workers_fails(fake_goat):
    with pytest.raises(RuntimeError, match="launched 2 of its 8 workers"):
        fake_goat([0, 1], "partial")


def test_goat_refuses_a_split_allocation_before_starting(fake_goat, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    with pytest.raises(RuntimeError, match="one node"):
        fake_goat(list(range(8)), "split")
