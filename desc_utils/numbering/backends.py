"""Built-in numbering backends.

``smarts``   ordered stages of SMARTS patterns with a ``roles`` map (label -> match position)
``manual``   read the indices from an existing CSV (one row per conformer)
``first_n``  label the first N atoms of the file in ``atoms`` order
``molalign`` placeholder for DescriPyTor's MCS alignment (needs the ``align`` extra)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from desc_utils.numbering.base import ConformerContext, NoMatch


class SmartsBackend:
    name = "smarts"

    def __init__(self, stages: list[dict], match_selection: str = "first"):
        self.stages = []
        for st in stages:
            patterns = [st["smarts"]] if isinstance(st["smarts"], str) else list(st["smarts"])
            compiled = []
            for p in patterns:
                patt = Chem.MolFromSmarts(p)
                if patt is None:
                    raise ValueError(f"stage {st.get('name')}: SMARTS does not parse: {p!r}")
                compiled.append((p, patt))
            self.stages.append({"name": st.get("name", "stage"), "patterns": compiled,
                                "roles": dict(st["roles"]), "required": bool(st.get("required", True))})
        if match_selection not in ("first", "lowest_indices"):
            raise ValueError("match_selection must be 'first' or 'lowest_indices'")
        self.match_selection = match_selection

    @classmethod
    def from_config(cls, numbering_cfg: dict, labels: list[str]) -> "SmartsBackend":
        cfg = numbering_cfg["smarts"]
        return cls(cfg["stages"], cfg.get("match_selection", "first"))

    def number(self, mol, ctx: ConformerContext) -> dict[str, int]:
        indices: dict[str, int] = {}
        for stage in self.stages:
            match = None
            for _, patt in stage["patterns"]:
                matches = mol.GetSubstructMatches(patt)
                if matches:
                    match = matches[0] if self.match_selection == "first" else min(matches)
                    break
            if match is None:
                if stage["required"]:
                    raise NoMatch(f"stage {stage['name']!r}: no SMARTS match")
                continue
            for label, pos in stage["roles"].items():
                indices[label] = int(match[pos])
        return indices


class ManualCsvBackend:
    """Indices from a CSV with one row per conformer; ``key_column`` identifies the folder."""

    name = "manual"

    def __init__(self, csv_path: str | Path, labels: list[str], key_column: str = "path", index_base: int = 1):
        self.csv_path = Path(csv_path)
        self.labels = list(labels)
        self.key_column = key_column
        self.index_base = int(index_base)
        df = pd.read_csv(self.csv_path)
        missing = [l for l in self.labels if l not in df.columns]
        if missing:
            raise KeyError(f"{self.csv_path}: missing label columns {missing}")
        if key_column not in df.columns:
            raise KeyError(f"{self.csv_path}: missing key column {key_column!r}")
        self.table = {self._norm(k): row for k, row in zip(df[key_column], df.to_dict("records"))}
        # Fallback for absolute paths written by another checkout: the last two components
        # (<molecule>/<conformer>) identify a conformer in the molecule/conformer layout.
        tails: dict[str, list] = {}
        for k, row in self.table.items():
            tails.setdefault(self._tail(k), []).append(row)
        self.by_tail = {t: rows[0] for t, rows in tails.items() if len(rows) == 1}

    @staticmethod
    def _norm(key) -> str:
        return str(key).replace("\\", "/").rstrip("/")

    @staticmethod
    def _tail(key: str) -> str:
        return "/".join(key.split("/")[-2:])

    @classmethod
    def from_config(cls, numbering_cfg: dict, labels: list[str]) -> "ManualCsvBackend":
        cfg = numbering_cfg["manual"]
        return cls(cfg["csv"], labels, cfg.get("key_column", "path"), cfg.get("index_base", 1))

    def number(self, mol, ctx: ConformerContext) -> dict[str, int]:
        key = self._norm(ctx.rel_path)
        row = self.table.get(key)
        if row is None:  # the CSV may hold absolute paths, or paths relative to another checkout
            row = self.by_tail.get(self._tail(key))
        if row is None:
            raise NoMatch(f"{key} not found in {self.csv_path.name} ({self.key_column})")
        out = {}
        for label in self.labels:
            v = row[label]
            if pd.isna(v):
                continue
            out[label] = int(v) - self.index_base
        return out


class FirstNBackend:
    """``atoms`` order -> atom 0, 1, 2, ... (the at1/at2 debugging convention)."""

    name = "first_n"

    def __init__(self, labels: list[str]):
        self.labels = list(labels)

    @classmethod
    def from_config(cls, numbering_cfg: dict, labels: list[str]) -> "FirstNBackend":
        return cls(labels)

    def number(self, mol, ctx: ConformerContext) -> dict[str, int]:
        n = mol.GetNumAtoms()
        if n < len(self.labels):
            raise NoMatch(f"molecule has {n} atoms, fewer than the {len(self.labels)} labels")
        return {label: i for i, label in enumerate(self.labels)}


class MolAlignBackend:
    """Map labels from a reference structure onto each conformer with DescriPyTor's MolAlign.

    Not wired yet: DescriPyTor's ``Molecules.get_renumbering_dict`` needs torch
    (``pip install "DescriPyTor[align]"``), changes the working directory, and works on a
    whole folder rather than one molecule. The backend exists so the spec can name it and
    the pipeline reports a clear error instead of a silent fallback.
    """

    name = "molalign"

    def __init__(self, reference: str, labels_in_reference: dict[str, int] | None = None):
        self.reference = reference
        self.labels_in_reference = labels_in_reference or {}

    @classmethod
    def from_config(cls, numbering_cfg: dict, labels: list[str]) -> "MolAlignBackend":
        cfg = numbering_cfg["molalign"]
        return cls(cfg["reference"], cfg.get("labels_in_reference"))

    def number(self, mol, ctx: ConformerContext) -> dict[str, int]:
        raise NotImplementedError(
            "molalign backend is not implemented; number the reference with `smarts` or `manual`, "
            "or number the conformers with DescriPyTor MolAlign yourself and use `manual`."
        )


# register factories that accept (numbering_cfg, labels)
from desc_utils.numbering import base as _base  # noqa: E402

_base.BACKENDS.update({
    "smarts": SmartsBackend.from_config,
    "manual": ManualCsvBackend.from_config,
    "first_n": FirstNBackend.from_config,
    "molalign": MolAlignBackend.from_config,
})
