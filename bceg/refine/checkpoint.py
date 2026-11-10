"""The per-conformer checkpoint shared by the refiner drivers.

A post-optimisation can be killed long before it finishes -- the job hits its time limit, the node
dies, someone scancels it. Without a checkpoint every conformer already optimised is lost and the
resubmitted job starts again from the first one. So each driver appends a conformer's result to a
JSON-lines file the moment it comes back, flushed and fsynced, and skips on resume whatever is
already in there (``details["n_resumed"]``).

The file is ``raw_<i>_<lot>.partial.jsonl`` (``MethodDir.refine_checkpoint``). Its first line is a
key, the hash of the file this step reads and the refinement parameters (``bceg.tasks``): a
checkpoint written for another search output, another ``continue_from`` lot or other parameters is
ignored rather than silently mixed into the new run. ``bceg.tasks`` removes the file once the final
``raw_<i>_<lot>.xyz`` is written, and reports ``n_checkpointed`` in the failed run record otherwise.

Used by :mod:`bceg.refine.ase_opt` (UMA, AIMNet2) and :mod:`bceg.refine.xtb_opt` (xtb, g-xTB); a
driver opts in by setting ``Refiner.resumable = True`` and taking a ``checkpoint=(path, key)``
argument, which is what makes ``bceg.tasks`` pass one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_checkpoint(path: Path, key: str) -> dict[int, dict[str, Any]]:
    """The results already in ``path`` (by conformer index); empty when it is absent or has another key.
    A last line cut off by the kill is skipped."""
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    try:
        if json.loads(lines[0]).get("key") != key:
            return {}
    except (IndexError, ValueError, AttributeError):
        return {}
    done = {}
    for line in lines[1:]:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done[r["index"]] = r | {"coords": np.asarray(r["coords"])}
    return done


class CheckpointWriter:
    """Appends each result to the checkpoint file (flushed and synced, so a kill keeps it)."""

    def __init__(self, path: Path, key: str, resumed: bool) -> None:
        self.fh = open(path, "a" if resumed else "w")
        if not resumed:
            self._write({"key": key})

    def _write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def __call__(self, r: dict) -> None:
        self._write(r | {"coords": np.asarray(r["coords"]).tolist()})

    def close(self) -> None:
        self.fh.close()
