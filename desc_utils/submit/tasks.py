"""The two cluster tasks of the ORCA stage, and what ``submit`` does per molecule.

``prepare_molecule``  (prepare array, one element per molecule, runs after bceg)
    read the ensemble (the search's replicates pooled in order), check its hyper-parameter hash against
    the library, prune it (stage 5: the energy window, or nothing), split it into conf_N/ folders, render
    calcParam.inp, record everything, submit the ORCA array.
``run_orca_conformer``  (ORCA array, one element per conformer)
    run ORCA in conf_N/, record the outcome (exit code AND the termination banner), condense.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

from bceg.inputs import InputMolecule
from bceg.provenance import ParamsMismatch
from bceg.xyz import HARTREE_TO_KCAL, Conformer, write_xyz

from desc_utils import orca_parse as op
from desc_utils.submit import records, slurm, template
from desc_utils.submit.config import Config
from desc_utils.submit.source import Ensemble, Ensembles

TERMINATION_BANNER = "ORCA TERMINATED NORMALLY"
RUN_TIME_RE = re.compile(r"^TOTAL RUN TIME:\s*(.+?)\s*$", re.MULTILINE)


def conformer_name(i: int) -> str:
    return f"conf_{i}"


# --------------------------------------------------------------------------- prepare
def energy_window(conformers: list[Conformer], ewin_kcal_mol: float | None) -> list[tuple[str, Conformer, float | None, str]]:
    """Stage 5, pruning: (name, conformer, E - Emin in kcal/mol or None, status) for every conformer, in
    the order of the pooled ensemble (N = 1-based position in it). ``ewin_kcal_mol`` None (the default) keeps every conformer; with a window, a conformer
    above it is ``skipped_energy_filter`` and one without an energy ``skipped_no_energy``."""
    energies = [c.energy for c in conformers if c.energy is not None]
    e_min = min(energies) if energies else None
    out = []
    for i, c in enumerate(conformers, start=1):
        rel = None if (c.energy is None or e_min is None) else (c.energy - e_min) * HARTREE_TO_KCAL
        if ewin_kcal_mol is None:
            status = "prepared"
        elif rel is None:
            status = "skipped_no_energy"
        else:
            status = "skipped_energy_filter" if rel > ewin_kcal_mol else "prepared"
        out.append((conformer_name(i), c, rel, status))
    return out


def _registry_entry(ens: Ensemble, smiles_line: int | None = None) -> dict:
    entry = {"smiles": ens.input_smiles, "canonical_smiles": ens.canonical_smiles, "name": ens.name,
             "source": ens.record}
    if smiles_line is not None:
        entry["smiles_line"] = smiles_line
    return entry


def prepare_molecule(cfg: Config, inchikey: str, input_smiles: str, force: bool = False,
                     submit_jobs: bool = True) -> str:
    """Prepare one molecule's ORCA jobs. Returns the molecule's ``orca_stage`` status."""
    mol = InputMolecule.from_smiles(input_smiles)
    if mol.inchikey != inchikey:
        raise RuntimeError(f"{input_smiles}: InChIKey {mol.inchikey} != {inchikey} (RDKit version drift?)")
    lib = cfg.library_dir
    mol_dir = lib / inchikey
    ens = Ensembles(cfg).ensemble(mol)
    base = {"inchikey": inchikey, "smiles": ens.input_smiles, "canonical_smiles": ens.canonical_smiles,
            "name": ens.name, "ensemble": ens.record, "pipeline": cfg.pipeline}
    records.register_molecule(lib, inchikey, _registry_entry(ens))

    if not ens.done:
        records.update_metadata(lib, inchikey, base | {
            "orca_stage": {"status": ens.status, "reason": ens.reason, "checked_at": records.now()}})
        print(f"{inchikey}: {ens.status} ({ens.reason})")
        return ens.status

    meta = records.load_metadata(lib, inchikey)
    recorded_sha = ((meta.get("ensemble") or {}).get("params_sha256"))
    if recorded_sha and ens.record.get("params_sha256") and recorded_sha != ens.record["params_sha256"]:
        reason = (f"ensemble params.json sha256 changed ({recorded_sha} -> {ens.record['params_sha256']}); "
                  f"this library was built on the earlier ensemble")
        records.update_metadata(lib, inchikey, {"orca_stage": {"status": "ensemble_changed", "reason": reason,
                                                                "checked_at": records.now()}})
        raise ParamsMismatch(f"{inchikey}: {reason}")

    # --- DFT set-up, rendered once per molecule (every conformer gets an identical copy)
    orca_prof = cfg.slurm["orca"]
    nprocs = int(orca_prof.get("ntasks") or orca_prof.get("cpus_per_task") or 1)
    maxcore = template.maxcore_mb(orca_prof.get("mem_per_cpu"), orca_prof.get("mem"), nprocs)
    lot = cfg.dft["level_of_theory"]
    rendered = template.render(cfg.template_path, lot, ens.charge, ens.multiplicity, nprocs, maxcore)
    mol_dir.mkdir(parents=True, exist_ok=True)
    (mol_dir / "calcParam.inp").write_text(rendered)
    dft_record = {
        "engine": "ORCA", "orca_module": _orca_module(), "charge": ens.charge, "multiplicity": ens.multiplicity,
        "pruning": dict(cfg.pruning), "calc_param_template": str(cfg.template_path),
        "calc_param_md5": template.md5(rendered), "nprocs": nprocs, "maxcore_mb": maxcore,
        "level_of_theory": template.validate_level_of_theory(lot),
        "compound_steps": template.compound_steps(lot, ens.charge, ens.multiplicity),
    }
    previous_md5 = (meta.get("dft") or {}).get("calc_param_md5")
    if previous_md5 and previous_md5 != dft_record["calc_param_md5"] and meta.get("conformers"):
        # params.json at the library level should have refused this; record it anyway
        dft_record["previous_calc_param_md5"] = previous_md5
        dft_record["settings_changed_at"] = records.now()

    # --- prune (stage 5) and split the ensemble into conf_N/
    existing = meta.get("conformers") or {}
    conformer_records: dict = {}
    to_run: list[str] = []
    for name, conf, rel, status in energy_window(ens.conformers, cfg.ewin):
        rec = {"energy_hartree": conf.energy, "energy_above_min_kcal_mol": rel}
        if "replicate" in conf.info:
            rec |= {"replicate": conf.info["replicate"], "index_in_replicate": conf.info["index"]}
        old = (existing.get(name) or {}).get("status")
        if status != "prepared":
            conformer_records[name] = rec | {"status": status}
            continue
        if old == "completed" or (old in records.PENDING_CONFORMER and not force):
            continue  # leave the record as it is
        conf_dir = mol_dir / name
        conf_dir.mkdir(exist_ok=True)
        # comment line: "<E> energy=<E> units=hartree": ORCA ignores it, the UMA-only descriptor
        # route reads energy= (spec: numbering.energy source xyz_comment, key energy, units hartree)
        tags = {"source": ens.record.get("lot") or "ensemble"}
        if "replicate" in conf.info:
            tags |= {"replicate": conf.info["replicate"], "replicate_index": conf.info["index"]}
        if conf.energy is not None:
            tags = {"energy": f"{conf.energy:.10f}", "units": "hartree", **tags}
        tagged = Conformer(conf.coords, conf.energy, tags)
        write_xyz(conf_dir / "input.xyz", ens.elements, [tagged])
        shutil.copyfile(mol_dir / "calcParam.inp", conf_dir / "calcParam.inp")
        conformer_records[name] = rec | {"status": "prepared", "prepared_at": records.now()}
        if old in records.BAD_CONFORMER or old in records.PENDING_CONFORMER:
            conformer_records[name]["resubmitted_from"] = old
        to_run.append(name)

    records.update_metadata(lib, inchikey, base | {
        "dft": dft_record, "conformers": conformer_records,
        "orca_stage": {"status": "prepared", "reason": "", "prepared_at": records.now(),
                       "n_conformers_in_ensemble": len(ens.conformers), "n_replicates": cfg.replicates,
                       "n_to_run": len(to_run)},
    })
    if not to_run:
        print(f"{inchikey}: nothing to run ({len(conformer_records)} conformer(s) recorded)")
        return "prepared"
    if submit_jobs:
        script, _ = slurm.write_orca_array(cfg, cfg.jobdir, inchikey, to_run)
        job_id = slurm.sbatch(script)
        records.update_metadata(lib, inchikey, {"conformers": {
            c: {"status": "submitted", "slurm_array_job_id": job_id, "slurm_array_index": i, "submitted_at": records.now()}
            for i, c in enumerate(to_run)}})
        print(f"{inchikey}: {len(to_run)} ORCA job(s) submitted as array {job_id}")
    return "prepared"


def _orca_module() -> str | None:
    for mod in os.environ.get("LOADEDMODULES", "").split(":"):
        if mod.lower().startswith("orca"):
            return mod
    return None


# --------------------------------------------------------------------------- ORCA
def _read_tail(path: Path, nbytes: int = 8192) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", errors="replace")


def allocated_cpus(default: int | None = None) -> int:
    """CPUs the SLURM allocation gives this ORCA job on its node, for the core-hours it costs.

    ``SLURM_CPUS_ON_NODE`` (what the allocation holds on this node, whether requested as ``ntasks`` or
    ``cpus_per_task``), else ``SLURM_NTASKS x SLURM_CPUS_PER_TASK``, else ``default`` (the ``nprocs``
    recorded for the molecule), else the machine's CPU count.
    """
    env = os.environ
    if env.get("SLURM_CPUS_ON_NODE"):
        return int(env["SLURM_CPUS_ON_NODE"])
    if env.get("SLURM_NTASKS") or env.get("SLURM_CPUS_PER_TASK"):
        return int(env.get("SLURM_NTASKS") or 1) * int(env.get("SLURM_CPUS_PER_TASK") or 1)
    return int(default or os.cpu_count() or 1)


def orca_outcome(out_path: Path, exit_code: int | None) -> dict:
    """``completed`` only if ORCA exited 0 AND printed its termination banner (ORCA can exit 0 after
    a Compound sub-job quietly died). ``orca_run_time`` is ORCA's ``TOTAL RUN TIME`` line as printed,
    ``orca_run_time_s`` the same in seconds, ``orca_steps_s`` the wall time of every compound step
    (``%base`` name -> seconds, None for a step that did not finish; see
    :func:`desc_utils.orca_parse.compound_step_timings`)."""
    terminated, run_time, run_time_s, steps = None, None, None, None
    if out_path.exists():
        tail = _read_tail(out_path)
        terminated = TERMINATION_BANNER in tail
        m = RUN_TIME_RE.search(tail)
        run_time = m.group(1) if m else None
        timings = op.compound_step_timings(out_path.read_text(errors="replace"))
        run_time_s = timings["total_run_time_s"]
        steps = {name: st["wall_s"] for name, st in timings["steps"].items()}
    status = "completed" if exit_code == 0 and terminated else "failed"
    return {"status": status, "exit_code": exit_code, "orca_terminated_normally": terminated, "orca_run_time": run_time,
            "orca_run_time_s": run_time_s, "orca_steps_s": steps}


def run_orca_conformer(cfg: Config, inchikey: str, conformer: str, orca_cmd: str | None = None) -> str:
    """Run ORCA for one conformer folder and record the outcome. Returns the conformer status."""
    lib = cfg.library_dir
    conf_dir = lib / inchikey / conformer
    if not (conf_dir / "calcParam.inp").exists():
        raise FileNotFoundError(f"{conf_dir}: no calcParam.inp (prepare step missing)")
    orca = orca_cmd or cfg.slurm.get("orca_cmd") or shutil.which("orca")
    if not orca:
        raise FileNotFoundError("orca not found on PATH; load the module in slurm.setup or set slurm.orca_cmd")

    nprocs = (records.load_metadata(lib, inchikey).get("dft") or {}).get("nprocs")
    cpus = allocated_cpus(nprocs)
    job_env = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
               "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "host": os.uname().nodename, "cpus": cpus}
    records.update_metadata(lib, inchikey, {"conformers": {conformer: {
        "status": "running", "started_at": records.now(), "orca_cmd": orca, **job_env}}})

    out_path = conf_dir / "calcParam.out"
    t0 = time.time()
    timed_out = {"flag": False}
    with open(out_path, "a") as out:
        proc = subprocess.Popen([orca, "calcParam.inp"], cwd=str(conf_dir), stdout=out, stderr=subprocess.STDOUT)

        def on_term(signum, frame):
            timed_out["flag"] = True
            proc.terminate()

        previous = signal.signal(signal.SIGTERM, on_term)
        try:
            exit_code = proc.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)

    outcome = orca_outcome(out_path, exit_code)
    if timed_out["flag"]:
        outcome["status"] = "timeout"
    outcome["wall_time_s"] = round(time.time() - t0, 1)
    outcome["cpu_hours"] = round(outcome["wall_time_s"] / 3600 * cpus, 5)  # the allocation's cost, as in bceg's records
    outcome["finished_at"] = records.now()

    if cfg.condense:
        outcome["condensed"] = _condense(conf_dir)
    records.update_metadata(lib, inchikey, {"conformers": {conformer: outcome}})
    print(f"{inchikey}/{conformer}: {outcome['status']} (exit {exit_code}, banner {outcome['orca_terminated_normally']})")
    return outcome["status"]


def _condense(conf_dir: Path) -> dict:
    """Write conf_dir/orca_condensed.json; never raises (the job's outcome is already recorded)."""
    try:
        from desc_utils.condense import condense_folder
        condense_folder(conf_dir, overwrite=True)
        return {"ok": True, "at": records.now()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-2000:], "at": records.now()}
