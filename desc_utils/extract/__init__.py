"""Per-conformer descriptor extraction driven by a spec and a numbering table.

Two engines:

* ``orca``        ORCA property JSON / NBO output / morfeus on the optimised geometry
                  (``desc_utils.extract.orca``, wrapping the functions in ``data_extraction``)
* ``descripytor`` DescriPyTor ``Molecules`` families on the feather/JSON molecule file
                  (``desc_utils.extract.descripytor``)

:func:`run_extraction` joins their outputs on the conformer key ``path``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from desc_utils.spec import Spec

ALL_FAMILIES = ("orca.xyz", "orca.charges", "orca.nbo", "orca.redox", "descripytor")


def _meta_columns(spec: Spec, df_num: pd.DataFrame) -> list[str]:
    keep = ["path", "xyz_name", spec.data["molecule_id_column"], spec.numbering["energy"]["column"],
            "smiles", "ligand_type", "ligand_class", "numbering_status"] + spec.labels
    return [c for c in dict.fromkeys(keep) if c in df_num.columns]


def run_extraction(spec: Spec, df_num: pd.DataFrame, repo_root: Path, families: Iterable[str] | None = None,
                   overwrite_xyz_json: bool = False, progress: bool = True) -> pd.DataFrame:
    """Compute every requested family and return one wide row per conformer.

    ``families`` defaults to every family present in the spec. The numbering table's
    metadata columns (path, molecule id, energy, smiles, atom indices, ...) are carried
    into the output so the aggregation stage needs nothing else.
    """
    wanted = list(families) if families is not None else list(ALL_FAMILIES)
    unknown = [f for f in wanted if f not in ALL_FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {ALL_FAMILIES}")

    from desc_utils.data_extraction import conf_key

    df_num = df_num.reset_index(drop=True).copy()
    df_num["path"] = df_num["path"].map(lambda p: conf_key(p, repo_root))  # one join key for every engine
    out = df_num[_meta_columns(spec, df_num)].copy()

    orca_wanted = [f.split(".", 1)[1] for f in wanted if f.startswith("orca.")]
    if spec.orca and orca_wanted:
        from desc_utils.extract.orca import run_orca

        orca_df = run_orca(df_num, spec, repo_root, families=[f for f in orca_wanted if f in spec.orca],
                           overwrite_xyz_json=overwrite_xyz_json, progress=progress)
        out = out.merge(orca_df, on="path", how="left", suffixes=("", "_orca"))

    if spec.descripytor and "descripytor" in wanted:
        from desc_utils.extract.descripytor import extract_descripytor

        dp_df = extract_descripytor(df_num, spec.descripytor, repo_root, progress=progress)
        out = out.merge(dp_df, on="path", how="left", suffixes=("", "_descripytor"))

    # a column produced by two engines under the same name is kept once (first wins)
    return out.loc[:, ~out.columns.duplicated()]
