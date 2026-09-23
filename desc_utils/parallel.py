"""Process pool shared by the numbering and extraction stages (``--jobs N`` on the command line).

Both stages are loops over independent conformers, so they are cut into one task per conformer and
mapped over a :class:`multiprocessing.Pool`. The heavy, read-only state (the spec, the numbering
backend, the repository root) travels once per worker through the pool initializer, the task itself
is only a row of the numbering table. Results come back in the input order.

Workers are *spawned* (the default on macOS and the only safe choice with RDKit and Open Babel), so
they re-import the calling script as their main module. That works for ``python -m desc_utils.run``,
for scripts run from a file (guard the call with ``if __name__ == "__main__":``) and for notebooks;
a script piped through stdin (``python - < script.py``) cannot be re-imported and the pool hangs.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

#: Thread-count knobs of the BLAS / OpenMP runtimes numpy and morfeus may link against. Every worker
#: is one core's worth of work; letting each one spin up ``n_cores`` threads only causes contention.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def resolve_jobs(jobs: int | None) -> int:
    """Worker count for a ``--jobs`` value: ``None`` or ``1`` = serial, ``0`` or negative = every core."""
    if jobs is None:
        return 1
    jobs = int(jobs)
    if jobs <= 0:
        return max(1, os.cpu_count() or 1)
    return jobs


def limit_blas_threads() -> None:
    """Pin the numerical libraries of the *workers* to one thread each (inherited through the environment).

    Only sets the variables that are not already set, so an explicit choice by the user is respected.
    Must run before the pool is created: spawned workers copy the environment at start-up.
    """
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")


def parallel_map(fn: Callable[[T], R], items: Sequence[T], jobs: int | None = 1, *,
                 initializer: Callable | None = None, initargs: Iterable = (),
                 progress: bool = True, desc: str = "") -> list[R]:
    """``[fn(x) for x in items]`` over ``jobs`` processes, in input order.

    ``fn`` and ``initializer`` must be importable module-level functions (the workers are spawned, not
    forked, so nothing is inherited from the calling process except what ``initargs`` carries). With
    ``jobs <= 1`` or a single item the loop runs in this process and ``initializer`` is called here.
    """
    items = list(items)
    n_workers = min(resolve_jobs(jobs), max(1, len(items)))
    if n_workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        it = items
        if progress and items:
            from tqdm import tqdm

            it = tqdm(items, desc=desc)
        return [fn(x) for x in it]

    import multiprocessing

    limit_blas_threads()
    with multiprocessing.Pool(n_workers, initializer=initializer, initargs=tuple(initargs)) as pool:
        it = pool.imap(fn, items, chunksize=1)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(items), desc=f"{desc}[{n_workers} workers]" if desc else f"{n_workers} workers")
        return list(it)
