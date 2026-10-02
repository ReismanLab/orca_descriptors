"""Where the conformer ensembles come from: the bceg results tree of the library.

For every molecule of the input list the ensemble is pooled, in replicate order, from the
``conformer_search.replicates`` files ``<ensembles_dir>/<transformation>/<InChIKey>/<method>/raw_<i>_<lot>.xyz``
(post-optimised at ``lot``; without a post-optimisation the raw search outputs ``raw_<i>.xyz``). The
hyper-parameters are ``params.json`` / ``params_<lot>.json``, the outcome of replicate ``i`` is
``run_<i>[_<lot>].json``. Every replicate must be finished: one without a record is ``ensemble_missing``,
one that failed is ``ensemble_failed``, so what a library refines never depends on which jobs happened
to be done. Each pooled conformer carries ``replicate`` and ``index`` (1-based, within its file) in its
``info``. Charge and multiplicity are read from ``params.json`` (bceg's ``structure``), never from the
library config.

``Ensembles.progress`` returns what the status command needs to say where stages 1 to 4 stand:
the transformation's outcome, the two run records (with their ``phase``), and whether the pipeline has
a pre-refinement step and a post-optimisation at all.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bceg.config import Config as BcegConfig
from bceg.config import write_raw as bceg_write_raw
from bceg.inputs import InputMolecule, load_inputs, update_index
from bceg.layout import method_dir
from bceg.provenance import ParamsMismatch, read_record
from bceg.structure import Structure
from bceg.xyz import Conformer, read_xyz

from desc_utils.submit.config import Config

DONE = "done"


@dataclass
class Ensemble:
    inchikey: str
    input_smiles: str
    canonical_smiles: str
    name: str | None
    status: str  # done | ensemble_missing | ensemble_failed | not_applicable
    reason: str = ""
    xyz_paths: list[Path] = field(default_factory=list)  # one per replicate, in order
    charge: int | None = None
    multiplicity: int | None = None
    elements: list[str] = field(default_factory=list)
    conformers: list[Conformer] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)  # provenance written to metadata.json / registry
    structure: Structure | None = None  # the transformation's product (RDKit mol in the xyz atom order); None when n/a

    @property
    def done(self) -> bool:
        return self.status == DONE


def aggregate_records(recs: list[dict | None]) -> dict | None:
    """One run record standing for every replicate's, for the status command.

    None when no replicate has a record. A ``not_applicable``, ``running`` or ``failed`` record wins over
    the others, in that order (the first replicate in that state). None while a replicate has no record
    yet, the stage is not done. Otherwise the first replicate's ``done`` record with ``n_conformers``
    summed over the replicates."""
    present = [r for r in recs if r]
    if not present:
        return None
    for status in ("not_applicable", "running", "failed"):
        for r in present:
            if r.get("status") == status:
                return r
    if len(present) < len(recs):
        return None
    out = dict(present[0])
    if any(r.get("n_conformers") is not None for r in present):
        out["n_conformers"] = sum(r.get("n_conformers") or 0 for r in present)
    return out


class Ensembles:
    """The ensembles of one library: one transformation, one method, one level, ``replicates`` replicates."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bcfg: BcegConfig = cfg.bceg_config()
        self.transformation = self.bcfg.transformations[cfg.transformation_name]
        self.method = cfg.method_name
        self.searcher = self.bcfg.methods[self.method].searcher
        self.lot = cfg.lot
        self.results_dir = cfg.ensembles_dir
        self.replicates = list(range(1, cfg.replicates + 1))

    def molecules(self) -> list[InputMolecule]:
        return load_inputs(self.cfg.smiles_path)

    def method_dir(self, inchikey: str):
        return method_dir(self.results_dir, self.transformation.name, inchikey, self.method)

    def ensemble(self, mol: InputMolecule) -> Ensemble:
        base = dict(inchikey=mol.inchikey, input_smiles=mol.input_smiles, canonical_smiles=mol.canonical_smiles,
                    name=mol.name)
        md = self.method_dir(mol.inchikey)
        record: dict[str, Any] = {
            "kind": "bceg", "results_dir": str(self.results_dir), "transformation": self.transformation.name,
            "method": self.method, "lot": self.lot, "replicates": len(self.replicates), "method_dir": str(md.path),
        }
        structure = self.transformation.apply(mol.canonical_smiles)
        if structure is None:
            return Ensemble(**base, status="not_applicable", record=record,
                            reason=f"transformation {self.transformation.name} does not apply to this molecule")
        base["structure"] = structure

        params = read_record(md.params())
        refine_params = read_record(md.params(self.lot)) if self.lot else None
        record["params_sha256"] = (params or {}).get("sha256")
        record["refine_params_sha256"] = (refine_params or {}).get("sha256") if self.lot else None

        runs = []  # (replicate, run record or None, xyz path)
        for i in self.replicates:
            rec_path = md.run_record(i, self.lot)
            rec = read_record(rec_path)
            xyz = md.refined(i, self.lot) if self.lot else md.raw(i)
            runs.append((i, rec, xyz))
        record["runs"] = [{"replicate": i, "run_record": str(md.run_record(i, self.lot)), "xyz": str(xyz),
                           "run_status": (rec or {}).get("status")} for i, rec, xyz in runs]
        missing = [md.run_record(i, self.lot).name for i, rec, _ in runs if rec is None]
        if missing:
            return Ensemble(**base, status="ensemble_missing", record=record,
                            reason=f"no {', '.join(missing)} in {md.path} (bceg not run or still running)")
        for i, rec, _ in runs:
            if rec.get("status") == "not_applicable":
                return Ensemble(**base, status="not_applicable", record=record, reason=rec.get("reason", ""))
        failed = []
        for i, rec, xyz in runs:
            if rec.get("status") != DONE or not xyz.exists():
                failed.append(f"{md.run_record(i, self.lot).name}: status {rec.get('status')!r}"
                              + (f": {rec.get('error')}" if rec.get("error") else "")
                              + ("" if xyz.exists() else f"; {xyz.name} missing"))
        if failed:
            return Ensemble(**base, status="ensemble_failed", record=record, reason="; ".join(failed))

        elements: list[str] = []
        conformers: list[Conformer] = []
        per_replicate = []
        for i, rec, xyz in runs:
            els, confs = read_xyz(xyz)
            if elements and els != elements:
                return Ensemble(**base, status="ensemble_failed", record=record,
                                reason=f"{xyz.name}: atom order differs from replicate {runs[0][0]}")
            elements = els
            for j, c in enumerate(confs, start=1):
                c.info.update(replicate=i, index=j)
            conformers.extend(confs)
            per_replicate.append(len(confs))
        compared = (params or {}).get("compared") or {}
        struct = compared.get("structure") or {}
        charge = struct.get("charge", structure.charge)
        multiplicity = struct.get("multiplicity", structure.multiplicity)
        record["n_conformers"] = len(conformers)
        record["n_conformers_per_replicate"] = per_replicate
        record["versions"] = runs[0][1].get("versions")
        return Ensemble(**base, status=DONE, xyz_paths=[xyz for _, _, xyz in runs], charge=int(charge),
                        multiplicity=int(multiplicity), elements=elements, conformers=conformers, record=record)

    def progress(self, mol: InputMolecule) -> dict:
        """What bceg's records say about stages 1 to 4 for this molecule (read-only, never raises).
        With several replicates the records are folded into one by ``aggregate_records``."""
        out: dict[str, Any] = {"transformation": DONE, "error": None, "search_record": None, "refine_record": None,
                               "has_preopt": self.cfg.has_preopt and self.searcher.has_start_step(),
                               "has_refiner": self.lot is not None, "n_conformers": None}
        try:
            structure = self.transformation.apply(mol.canonical_smiles)
        except Exception as exc:  # noqa: BLE001  -- a broken script must not take the status command down
            out["transformation"] = "failed"
            out["error"] = f"transformation {self.transformation.name}: {exc!r}"
            return out
        if structure is None:
            out["transformation"] = "not_applicable"
            return out
        md = self.method_dir(mol.inchikey)
        search = aggregate_records([read_record(md.run_record(i)) for i in self.replicates])
        refine = aggregate_records([read_record(md.run_record(i, self.lot)) for i in self.replicates]) if self.lot else None
        out["search_record"], out["refine_record"] = search, refine
        for rec in (refine, search):
            if rec and rec.get("n_conformers") is not None:
                out["n_conformers"] = rec["n_conformers"]
                break
        for rec in (refine, search):
            if rec and rec.get("status") == "failed" and rec.get("error"):
                out["error"] = rec["error"]
                break
        return out

    def _running(self, task) -> bool:
        """True when bceg's own record says this replicate's search or refinement is running right now.

        bceg's skip rule only recognises a finished step, so a resubmission while a job is still running
        would start the same search again (two CREST runs of one molecule, last writer wins). A record
        left at "running" by a job killed before it wrote its end is released by --force.
        """
        md = method_dir(self.results_dir, task.transformation, task.inchikey, task.method)
        for rec_path in (md.run_record(task.replicate), md.run_record(task.replicate, self.lot) if self.lot else None):
            if rec_path is not None and (read_record(rec_path) or {}).get("status") == "running":
                return True
        return False

    def pending_tasks(self, force: bool = False) -> tuple[list, list[str]]:
        """bceg tasks still to run for this library, and params-mismatch messages (read-only).
        Tasks whose record says "running" are left out unless ``force``."""
        from bceg.tasks import expand
        tasks, problems = expand(self.bcfg, self.molecules(), write_params=False)
        if not force:
            tasks = [t for t in tasks if not self._running(t)]
        return tasks, problems

    def submit(self, jobdir: Path, no_sbatch: bool = False, force: bool = False) -> list[str]:
        """Run bceg's submit for this library; return the SLURM job ids it started.

        Freezes bceg's resolved config under ``jobdir/bceg/`` exactly like ``bceg submit`` does, so
        the two stages of one submission share one ``_jobs/<timestamp>`` folder. Tasks whose bceg
        record says "running" are not resubmitted unless ``force``.
        """
        from bceg.slurm import write_jobs
        from bceg.tasks import expand

        molecules = self.molecules()
        tasks, problems = expand(self.bcfg, molecules, write_params=False)
        if problems:
            raise ParamsMismatch("bceg hyper-parameters differ from the existing logs:\n" + "\n\n".join(problems))
        if not force:
            running = [t for t in tasks if self._running(t)]
            if running:
                print(f"bceg: {len(running)} task(s) recorded running are left alone (use --force to resubmit them): "
                      + ", ".join(f"{t.inchikey}" for t in running))
            tasks = [t for t in tasks if not self._running(t)]
        if not tasks:
            print("bceg: nothing pending for this library.")
            return []
        bceg_dir = jobdir / "bceg"
        bceg_dir.mkdir(parents=True, exist_ok=True)
        bceg_write_raw(self.cfg.bceg_raw(), bceg_dir / "config.yaml")
        cfg = BcegConfig.load(bceg_dir / "config.yaml")
        update_index(cfg.results_dir, molecules)
        keep = {t.id for t in tasks}
        tasks = [t for t in expand(cfg, molecules, write_params=True)[0] if t.id in keep]
        write_jobs(cfg, tasks, bceg_dir)
        print(f"bceg: {len(tasks)} task(s); job scripts in {bceg_dir}")
        if no_sbatch:
            return []
        result = subprocess.run(["bash", str(bceg_dir / "submit.sh")], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            raise RuntimeError(f"bceg submit.sh failed ({result.returncode}): {result.stderr.strip()}")
        return re.findall(r":\s*(\d+)\s*$", result.stdout, re.MULTILINE)
