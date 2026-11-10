"""The two cluster tasks of the ORCA stage, and what ``submit`` does per molecule.

``prepare_molecule``  (prepare array, one element per molecule, runs after bceg)
    read the ensemble (the search's replicates pooled in order), check its hyper-parameter hash against
    the library, prune it (stage 5: the connectivity check and the energy window, or nothing), split it into conf_N/ folders, render
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
import threading
import time
import traceback
from pathlib import Path

from rdkit import Chem

from bceg import topology
from bceg.constraints import held_atoms
from bceg.inputs import InputMolecule
from bceg.provenance import ParamsMismatch
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_KCAL, Conformer, write_xyz

from desc_utils import orca_parse as op
from desc_utils.submit import records, slurm, template
from desc_utils.submit.cancel import REASON, was_cancelled
from desc_utils.submit.config import Config
from desc_utils.submit.source import Ensemble, Ensembles

TERMINATION_BANNER = "ORCA TERMINATED NORMALLY"
RUN_TIME_RE = re.compile(r"^TOTAL RUN TIME:\s*(.+?)\s*$", re.MULTILINE)


def conformer_name(i: int) -> str:
    return f"conf_{i}"


STRUCTURE_MOL = "structure.mol"


def write_structure(mol_dir: Path, structure: Structure) -> dict:
    """Write ``mol_dir/structure.mol``: the transformation's product as RDKit built it (hydrogens explicit,
    atoms in the order of every xyz of the molecule, dative bonds kept). ``Chem.MolFromMolFile(path,
    removeHs=False)`` gives the object back (``sanitize=False`` for a metal complex RDKit cannot
    sanitise). A start geometry returned by the transformation is written as the 3D block. Returns the
    ``structure`` record of metadata.json."""
    mol = Chem.Mol(structure.mol)
    if structure.coords is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(structure.coords):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
    mol_dir.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(mol_dir / STRUCTURE_MOL))
    return {"smiles": structure.smiles, "charge": structure.charge, "multiplicity": structure.multiplicity,
            "n_atoms": mol.GetNumAtoms(), "has_input_coords": structure.coords is not None,
            "mol_file": STRUCTURE_MOL, "metadata": structure.metadata}


# --------------------------------------------------------------------------- prepare
def energy_window(conformers: list[Conformer], ewin_kcal_mol: float | None, mol: Chem.Mol | None = None,
                  topology_params: dict | None = None) -> list[tuple[str, Conformer, float | None, str, list[str]]]:
    """Stage 5, pruning: (name, conformer, E - Emin in kcal/mol or None, status, topology changes) for every
    conformer, in the order of the pooled ensemble (N = 1-based position in it).

    With ``topology_params`` (``pruning.topology``, bceg.topology.check_params) each geometry is first
    compared with the graph ``mol``: one with a change the block does not allow is ``skipped_topology`` and
    takes no part in Emin, so a rearranged structure lower in energy cannot set the window. The changes of
    every conformer are returned as messages (the allowed ones, e.g. agostic contacts, included).
    ``ewin_kcal_mol`` None (the default) keeps every remaining conformer; with a window, a conformer above it
    is ``skipped_energy_filter`` and one without an energy ``skipped_no_energy``."""
    changes: list[list[str]] = [[] for _ in conformers]
    rejected = [False] * len(conformers)
    if topology_params is not None:
        if mol is None:
            raise ValueError("pruning.topology needs the molecular graph")
        for k, c in enumerate(conformers):
            found, bad = topology.check(mol, c.coords, topology_params)
            changes[k], rejected[k] = [str(x) for x in found], bool(bad)
    energies = [c.energy for c, r in zip(conformers, rejected) if c.energy is not None and not r]
    e_min = min(energies) if energies else None
    out = []
    for i, (c, r) in enumerate(zip(conformers, rejected), start=1):
        rel = None if (c.energy is None or e_min is None) else (c.energy - e_min) * HARTREE_TO_KCAL
        if r:
            status = "skipped_topology"
        elif ewin_kcal_mol is None:
            status = "prepared"
        elif rel is None:
            status = "skipped_no_energy"
        else:
            status = "skipped_energy_filter" if rel > ewin_kcal_mol else "prepared"
        out.append((conformer_name(i), c, rel, status, changes[i - 1]))
    return out


def _registry_entry(ens: Ensemble, smiles_line: int | None = None) -> dict:
    entry = {"smiles": ens.input_smiles, "canonical_smiles": ens.canonical_smiles, "name": ens.name,
             "transformed_smiles": None if ens.structure is None else ens.structure.smiles, "source": ens.record}
    if smiles_line is not None:
        entry["smiles_line"] = smiles_line
    return entry


def prepare_molecule(cfg: Config, inchikey: str, input_smiles: str, force: bool = False,
                     submit_jobs: bool = True) -> str:
    """Prepare one molecule's stage-6 jobs. Returns the molecule's ``orca_stage`` status."""
    from desc_utils.submit import gxtb as gxtb_run  # imports this module back for the shared run helpers
    mol = InputMolecule.from_smiles(input_smiles)
    if mol.inchikey != inchikey:
        raise RuntimeError(f"{input_smiles}: InChIKey {mol.inchikey} != {inchikey} (RDKit version drift?)")
    lib = cfg.library_dir
    mol_dir = lib / inchikey
    ens = Ensembles(cfg).ensemble(mol)
    base = {"inchikey": inchikey, "smiles": ens.input_smiles, "canonical_smiles": ens.canonical_smiles,
            "name": ens.name, "ensemble": ens.record, "pipeline": cfg.pipeline}
    if ens.structure is not None:  # the product of stage 1, as an RDKit mol file next to metadata.json
        base["structure"] = write_structure(mol_dir, ens.structure)
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
    gxtb_engine = cfg.dft_engine == "gxtb"
    # dft.constrained: the atoms the constraints name stay where stages 2 to 4 put them (same indices in
    # every conformer, so one rendering still serves them all)
    fixed = []
    if cfg.dft.get("constrained"):
        if ens.structure is None:
            raise RuntimeError(f"{inchikey}: dft.constrained needs the ensemble's structure (the molecular graph)")
        fixed = held_atoms(cfg.search["constraints"], ens.structure.mol)
    mol_dir.mkdir(parents=True, exist_ok=True)
    if gxtb_engine:
        prof = cfg.slurm["gxtb"]
        nprocs = int(prof.get("ntasks") or prof.get("cpus_per_task") or 1)
        dft_record = {
            "engine": gxtb_run.ENGINE_LABEL, "charge": ens.charge, "multiplicity": ens.multiplicity,
            "pruning": dict(cfg.pruning), "nprocs": nprocs, "gxtb": dict(cfg.gxtb),
            "compound_steps": gxtb_run.steps_to_run(cfg.gxtb),
        }
    else:
        orca_prof = cfg.slurm["orca"]
        nprocs = int(orca_prof.get("ntasks") or orca_prof.get("cpus_per_task") or 1)
        maxcore = template.maxcore_mb(orca_prof.get("mem_per_cpu"), orca_prof.get("mem"), nprocs)
        lot = cfg.dft["level_of_theory"]
        rendered = template.render(cfg.template_path, lot, ens.charge, ens.multiplicity, nprocs, maxcore, fixed)
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
    if fixed:
        dft_record["fixed_atoms"] = {str(i): f"{ens.elements[i]}{i}" for i in fixed}

    # --- prune (stage 5) and split the ensemble into conf_N/
    existing = meta.get("conformers") or {}
    conformer_records: dict = {}
    to_run: list[str] = []
    topo = cfg.topology
    if topo is not None and ens.structure is None:
        raise RuntimeError(f"{inchikey}: pruning.topology needs the ensemble's structure (the molecular graph)")
    pruned = energy_window(ens.conformers, cfg.ewin, None if ens.structure is None else ens.structure.mol, topo)
    n_topo = sum(1 for p in pruned if p[3] == "skipped_topology")
    if topo is not None:
        print(f"{inchikey}: {n_topo}/{len(pruned)} conformer(s) skipped_topology")
    for name, conf, rel, status, changes in pruned:
        rec = {"energy_hartree": conf.energy, "energy_above_min_kcal_mol": rel}
        if changes:
            rec["topology_changes"] = changes
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
        if not gxtb_engine:  # g-xTB takes its options from the config, there is no input file to copy
            shutil.copyfile(mol_dir / "calcParam.inp", conf_dir / "calcParam.inp")
        conformer_records[name] = rec | {"status": "prepared", "prepared_at": records.now()}
        if old in records.BAD_CONFORMER or old in records.PENDING_CONFORMER:
            conformer_records[name] |= {"resubmitted_from": old, "reason": None}
        to_run.append(name)

    records.update_metadata(lib, inchikey, base | {
        "dft": dft_record, "conformers": conformer_records,
        "orca_stage": {"status": "prepared", "reason": "", "prepared_at": records.now(),
                       "n_conformers_in_ensemble": len(ens.conformers), "n_replicates": cfg.replicates,
                       "n_skipped_topology": n_topo,
                       "n_to_run": len(to_run)},
    })
    if not to_run:
        print(f"{inchikey}: nothing to run ({len(conformer_records)} conformer(s) recorded)")
        return "prepared"
    if submit_jobs:
        writer = slurm.write_gxtb_array if gxtb_engine else slurm.write_orca_array
        script, _ = writer(cfg, cfg.jobdir, inchikey, to_run)
        job_id = slurm.sbatch(script)
        # with g-xTB batching, element i runs the i-th chunk of dft.gxtb.batch_size conformers
        per_element = (cfg.gxtb.get("batch_size") or 1) if gxtb_engine else 1
        records.update_metadata(lib, inchikey, {"conformers": {
            c: {"status": "submitted", "slurm_array_job_id": job_id, "slurm_array_index": i // per_element,
                "submitted_at": records.now()}
            for i, c in enumerate(to_run)}})
        n_elements = -(-len(to_run) // per_element)
        print(f"{inchikey}: {len(to_run)} {'g-xTB' if gxtb_engine else 'ORCA'} job(s) submitted as array {job_id}"
              + (f" ({n_elements} element(s) of up to {per_element})" if per_element > 1 else ""))
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


class TermWatch:
    """One SIGTERM handler for several child programs run at once from worker threads.

    ``signal.signal`` only works in the main thread, so a pool of threads that each run a program
    (the batched g-xTB engine) cannot install a handler per call the way :func:`run_watching_for_term`
    does on its own. Install one of these in the main thread (``with TermWatch() as watch``) and pass it
    to every call: a SIGTERM then sets :attr:`terminated` and terminates every child still running, and
    a child started afterwards is terminated as soon as it is registered.
    """

    def __init__(self) -> None:
        self.terminated = False
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._previous = None

    def __enter__(self) -> "TermWatch":
        self._previous = signal.signal(signal.SIGTERM, self._on_term)
        return self

    def __exit__(self, *exc) -> None:
        signal.signal(signal.SIGTERM, self._previous)

    def _on_term(self, signum, frame) -> None:
        self.terminated = True
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            proc.terminate()

    def register(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.add(proc)
        if self.terminated:
            proc.terminate()

    def unregister(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.discard(proc)


def run_watching_for_term(cmd: list[str], cwd: Path, log_path: Path,
                          env: dict[str, str] | None = None,
                          watch: TermWatch | None = None) -> tuple[int, bool]:
    """Run ``cmd`` in ``cwd``, appending its output to ``log_path``; returns ``(exit_code, timed_out)``.

    SLURM sends the job SIGTERM 10 min before its time limit (``--signal=B:TERM@600``). Forwarding it
    to the child lets the program stop on its own and the record be written, instead of the whole job
    being killed with nothing to show for it. Shared by the ORCA and g-xTB engines so a time limit, a
    ``scancel`` and a crash are told apart the same way whichever one ran.

    ``env`` overrides entries of the environment the child gets (ORCA needs none; xtb needs its thread
    settings, see :func:`desc_utils.submit.gxtb.thread_env`). ``watch`` is the handler a pool of worker
    threads shares (:class:`TermWatch`); without it this call installs its own, which needs the main thread.
    """
    child_env = {**os.environ, **env} if env else None
    with open(log_path, "a") as out:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT, env=child_env)
        if watch is not None:
            watch.register(proc)
            try:
                exit_code = proc.wait()
            finally:
                watch.unregister(proc)
            return exit_code, watch.terminated

        timed_out = {"flag": False}

        def on_term(signum, frame):
            timed_out["flag"] = True
            proc.terminate()

        previous = signal.signal(signal.SIGTERM, on_term)
        try:
            exit_code = proc.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
    return exit_code, timed_out["flag"]


def terminated_outcome(lib: Path, job_env: dict) -> dict:
    """What a SIGTERM meant: a ``orcadesc cancel`` (it lists the job ids it cancelled) is a failure,
    anything else is the walltime running out."""
    if was_cancelled(lib, [job_env.get("slurm_array_job_id"), job_env.get("slurm_job_id")]):
        return {"status": "failed", "reason": REASON}
    return {"status": "timeout"}


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
    exit_code, timed_out = run_watching_for_term([orca, "calcParam.inp"], conf_dir, out_path)

    outcome = orca_outcome(out_path, exit_code)
    if timed_out:
        outcome |= terminated_outcome(lib, job_env)
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
