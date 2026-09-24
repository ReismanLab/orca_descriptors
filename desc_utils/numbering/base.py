"""Core types for atom numbering: conformer context, backend protocol, validation, registry.

A numbering backend answers one question for one conformer: *which atom index plays which
labelled role?* It returns ``{label: 0-based RDKit index}``. Everything else (index base,
element validation, consistency across conformers, SMILES, energy) is handled by the
pipeline in :mod:`desc_utils.numbering`, so backends stay small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, runtime_checkable

STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete"          # backend returned a partial label set
STATUS_FAILED = "failed"                  # backend raised (no match, missing file, ...)
STATUS_ELEMENT_MISMATCH = "element_mismatch"
STATUS_INCONSISTENT = "inconsistent"      # conformers of one molecule disagree


class NoMatch(Exception):
    """A required SMARTS stage did not match."""


@dataclass(frozen=True)
class ConformerContext:
    molecule_id: str
    conf_dir: Path                 # absolute conformer folder
    rel_path: str                  # repo-relative POSIX path with trailing "/", the legacy `path` column
    xyz_path: Path
    sdf_path: Path

    @property
    def xyz_name(self) -> str:     # legacy column: geometry file stem ("optimization")
        return self.xyz_path.stem


@runtime_checkable
class NumberingBackend(Protocol):
    name: str

    def number(self, mol, ctx: ConformerContext) -> dict[str, int]:
        """Return ``label -> 0-based atom index``. May be partial; raise :class:`NoMatch` on failure."""


@dataclass
class NumberingRow:
    ctx: ConformerContext
    indices: dict[str, int] = field(default_factory=dict)   # 0-based
    status: str = STATUS_OK
    notes: list[str] = field(default_factory=list)


def validate_elements(mol, indices: Mapping[str, int], expected: Mapping[str, str | None]) -> list[str]:
    """Compare the element at every assigned index with ``atoms`` in the spec."""
    problems = []
    n = mol.GetNumAtoms()
    for label, idx in indices.items():
        if idx < 0 or idx >= n:
            problems.append(f"{label}: index {idx} outside 0..{n - 1}")
            continue
        want = expected.get(label)
        if want is None:
            continue
        got = mol.GetAtomWithIdx(int(idx)).GetSymbol()
        if got != want:
            problems.append(f"{label}: expected {want}, found {got} at index {idx}")
    return problems


BackendFactory = Callable[[dict, list], NumberingBackend]
#: backend name -> factory ``(numbering_cfg, labels) -> backend``; filled by ``desc_utils.numbering.backends``
BACKENDS: dict[str, BackendFactory] = {}


def make_backend(numbering_cfg: dict, labels: list[str]) -> NumberingBackend:
    """Instantiate the backend named by ``numbering_cfg['backend']``."""
    import desc_utils.numbering.backends  # noqa: F401  (registers the built-in backends)

    name = numbering_cfg.get("backend")
    if name not in BACKENDS:
        raise KeyError(f"unknown numbering backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name](numbering_cfg, labels)
