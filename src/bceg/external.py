"""Helpers for calling external programs (CREST, ORCA, LoQI, ...)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


class ProgramNotFound(RuntimeError):
    pass


def find_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ProgramNotFound(f"'{name}' not found on PATH (load the module / activate the env, or set its path in the config)")
    return path


def run(cmd: list[str], cwd: Path, threads: int = 1, log_name: str = "stdout.log") -> subprocess.CompletedProcess:
    """Run ``cmd`` in ``cwd`` with stdout+stderr captured to ``cwd/log_name``."""
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=f"{threads},1", MKL_NUM_THREADS=str(threads), OMP_STACKSIZE="4G")
    log = cwd / log_name
    with open(log, "w") as out:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
    proc.stdout = log.read_text(errors="replace")
    return proc


def program_version(cmd: list[str], match: str = "version") -> str | None:
    """First output line of ``cmd`` containing ``match`` (case-insensitive), or None."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (out.stdout + out.stderr).splitlines():
        if match.lower() in line.lower():
            return line.strip()
    return None


def scratch_root() -> Path | None:
    """Node-local scratch on SLURM nodes if available."""
    for var in ("BCEG_SCRATCH", "SLURM_TMPDIR", "TMPDIR"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return None


@contextmanager
def workdir(keep_dir: str | Path | None = None, prefix: str = "bceg_"):
    """Temporary working directory; if ``keep_dir`` is given, create it there and keep it."""
    if keep_dir is None:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_root()) as tmp:
            yield Path(tmp)
    else:
        Path(keep_dir).mkdir(parents=True, exist_ok=True)
        yield Path(tempfile.mkdtemp(prefix=prefix, dir=keep_dir))
