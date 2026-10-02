"""``cost``: what every molecule of the input list has cost so far, per stage, from the records
(not from SLURM's accounting database, which forgets).

One row per molecule. Core-hours (``_cpu_h``) and GPU-hours (``_gpu_h``) per stage:
``pre_refinement`` and ``search`` from the ``phases`` of bceg's ``run_<i>.json`` (a record written before
the phases existed puts its whole time under ``search``), ``post_opt`` from ``run_<i>_<lot>.json``,
``dft`` from the conformer records of ``metadata.json`` (``cpu_hours``; a record written before the
field existed uses ``wall_time_s`` x the molecule's ``nprocs``), and ``dft_<step>_cpu_h`` per compound
step from ``orca_steps_s``. Failed runs count: they were paid for. Replicates and conformers are
summed, so the row is the price of that molecule in this library.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bceg.provenance import read_record

from desc_utils.submit import records
from desc_utils.submit.config import Config
from desc_utils.submit.source import Ensembles

ENSEMBLE_STAGES = ("pre_refinement", "search", "post_opt")
#: compound steps in the order of the template; a step the template does not know is appended
STEP_ORDER = ("optimization", "frequency", "singlepoint", "nmr", "oxidation", "reduction")


def _hours(rec: dict | None, phase: str | None = None) -> tuple[float, float]:
    """(core-hours, GPU-hours) of one run record, or of one of its ``phases``."""
    rec = rec or {}
    if phase is not None:
        phases = rec.get("phases")
        if phases is None:  # recorded before the phases existed: the total is all we have, it is the search's
            rec = rec if phase == "search" else {}
        else:
            rec = phases.get(phase) or {}
    return float(rec.get("cpu_hours") or 0.0), float(rec.get("gpu_hours") or 0.0)


def molecule_cost(cfg: Config, source: Ensembles, mol) -> dict:
    """The cost row of one molecule (see the module docstring for the columns)."""
    md = source.method_dir(mol.inchikey)
    row: dict = {"inchikey": mol.inchikey, "name": mol.name or ""}
    cpu = {s: 0.0 for s in ENSEMBLE_STAGES}
    gpu = {s: 0.0 for s in ENSEMBLE_STAGES}
    n_runs = 0
    for i in source.replicates:
        search = read_record(md.run_record(i))
        if search and "wall_time_s" in search:
            n_runs += 1
            for phase, stage in (("pre_refinement", "pre_refinement"), ("search", "search")):
                c, g = _hours(search, phase)
                cpu[stage] += c
                gpu[stage] += g
        if source.lot:
            refine = read_record(md.run_record(i, source.lot))
            if refine and "wall_time_s" in refine:
                n_runs += 1
                c, g = _hours(refine)
                cpu["post_opt"] += c
                gpu["post_opt"] += g

    meta = records.load_metadata(cfg.library_dir, mol.inchikey)
    nprocs = (meta.get("dft") or {}).get("nprocs")
    dft_cpu, steps, n_dft = 0.0, {}, 0
    for rec in (meta.get("conformers") or {}).values():
        rec = rec or {}
        if rec.get("wall_time_s") is None:
            continue
        n_dft += 1
        cpus = rec.get("cpus") or nprocs or 1
        dft_cpu += float(rec["cpu_hours"]) if rec.get("cpu_hours") is not None else rec["wall_time_s"] * cpus / 3600
        for step, sec in (rec.get("orca_steps_s") or {}).items():
            if sec is not None:
                steps[step] = steps.get(step, 0.0) + sec * cpus / 3600

    for s in ENSEMBLE_STAGES:
        row[f"{s}_cpu_h"] = round(cpu[s], 4)
    row["dft_cpu_h"] = round(dft_cpu, 4)
    for step in [*STEP_ORDER, *sorted(set(steps) - set(STEP_ORDER))]:
        if step in steps:
            row[f"dft_{step}_cpu_h"] = round(steps[step], 4)
    row["total_cpu_h"] = round(sum(cpu.values()) + dft_cpu, 4)
    for s in ENSEMBLE_STAGES:
        row[f"{s}_gpu_h"] = round(gpu[s], 4)
    row["total_gpu_h"] = round(sum(gpu.values()), 4)
    row["n_ensemble_runs"], row["n_dft_runs"] = n_runs, n_dft
    row["smiles"] = mol.input_smiles
    return row


def cost_table(cfg: Config) -> pd.DataFrame:
    source = Ensembles(cfg)
    rows = [molecule_cost(cfg, source, mol) for mol in source.molecules()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    step_cols = [c for c in df.columns if c.startswith("dft_") and c != "dft_cpu_h"]
    front = ["inchikey", "name", *[f"{s}_cpu_h" for s in ENSEMBLE_STAGES], "dft_cpu_h", *step_cols, "total_cpu_h",
             *[f"{s}_gpu_h" for s in ENSEMBLE_STAGES], "total_gpu_h", "n_ensemble_runs", "n_dft_runs", "smiles"]
    df = df[front].fillna({c: 0.0 for c in step_cols})
    return df.sort_values("total_cpu_h", ascending=False).reset_index(drop=True)


def print_cost(cfg: Config, csv: str | Path | None = None) -> pd.DataFrame:
    df = cost_table(cfg)
    if df.empty:
        print("No molecules in the input list.")
        return df
    hours = [c for c in df.columns if c.endswith("_cpu_h") or c.endswith("_gpu_h")]
    shown = df.drop(columns=["smiles"] + [c for c in hours if c.endswith("_gpu_h") and not (df[c] > 0).any()])
    with pd.option_context("display.width", 250, "display.max_rows", 1000, "display.float_format", "{:.3f}".format):
        print(shown.to_string(index=False))
        print()
        total = df[hours].sum()
        print("total   : " + ", ".join(f"{c[:-6]} {total[c]:.2f}" for c in hours if c.endswith("_cpu_h")) + " core-h")
        gpu_total = {c: total[c] for c in hours if c.endswith("_gpu_h") and total[c] > 0}
        if gpu_total:
            print("gpu     : " + ", ".join(f"{c[:-6]} {v:.2f}" for c, v in gpu_total.items()) + " GPU-h")
        priced = df[(df["n_ensemble_runs"] + df["n_dft_runs"]) > 0]
        if not priced.empty:
            print(f"per mol : {priced['total_cpu_h'].mean():.2f} core-h over {len(priced)} molecule(s) with a record"
                  + (f", {len(df) - len(priced)} without" if len(priced) < len(df) else ""))
    if csv:
        Path(csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        print(f"written {csv}")
    return df
