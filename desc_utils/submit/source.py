"""Where the conformer ensembles come from: the bceg results tree of the library.

For every molecule of the input list the ensemble is
``<ensembles_dir>/<transformation>/<InChIKey>/<method>/raw_1_<lot>.xyz`` (replicate 1, post-optimised at
``lot``; without a post-optimisation the raw search output ``raw_1.xyz``), its hyper-parameters are
``params.json`` / ``params_<lot>.json`` and its outcome ``run_1[_<lot>].json``. Charge and multiplicity
are read from ``params.json`` (bceg's ``structure``), never from the library config.

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
    xyz_path: Path | None = None
    charge: int | None = None
    multiplicity: int | None = None
    elements: list[str] = field(default_factory=list)
    conformers: list[Conformer] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)  # provenance written to metadata.json / registry

    @property
    def done(self) -> bool:
        return self.status == DONE


class Ensembles:
    """The ensembles of one library: one transformation, one method, one level, replicate 1."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bcfg: BcegConfig = cfg.bceg_config()
        self.transformation = self.bcfg.transformations[cfg.transformation_name]
        self.method = cfg.method_name
        self.searcher = self.bcfg.methods[self.method].searcher
        self.lot = cfg.lot
        self.results_dir = cfg.ensembles_dir

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
            "method": self.method, "lot": self.lot, "replicate": 1, "method_dir": str(md.path),
        }
        structure = self.transformation.apply(mol.canonical_smiles)
        if structure is None:
            return Ensemble(**base, status="not_applicable", record=record,
                            reason=f"transformation {self.transformation.name} does not apply to this molecule")

        params = read_record(md.params())
        refine_params = read_record(md.params(self.lot)) if self.lot else None
        record["params_sha256"] = (params or {}).get("sha256")
        record["refine_params_sha256"] = (refine_params or {}).get("sha256") if self.lot else None

        run_rec = read_record(md.run_record(1, self.lot))
        xyz = md.refined(1, self.lot) if self.lot else md.raw(1)
        record["run_record"] = str(md.run_record(1, self.lot))
        record["xyz"] = str(xyz)
        record["run_status"] = (run_rec or {}).get("status")
        if run_rec is None:
            return Ensemble(**base, status="ensemble_missing", record=record,
                            reason=f"no {md.run_record(1, self.lot).name} in {md.path} (bceg not run or still running)")
        if run_rec.get("status") == "not_applicable":
            return Ensemble(**base, status="not_applicable", record=record, reason=run_rec.get("reason", ""))
        if run_rec.get("status") != DONE or not xyz.exists():
            return Ensemble(**base, status="ensemble_failed", record=record,
                            reason=f"{md.run_record(1, self.lot).name}: status {run_rec.get('status')!r}"
                                   + (f": {run_rec.get('error')}" if run_rec.get("error") else "")
                                   + ("" if xyz.exists() else f"; {xyz.name} missing"))

        elements, conformers = read_xyz(xyz)
        compared = (params or {}).get("compared") or {}
        struct = compared.get("structure") or {}
        charge = struct.get("charge", structure.charge)
        multiplicity = struct.get("multiplicity", structure.multiplicity)
        record["n_conformers"] = len(conformers)
        record["versions"] = run_rec.get("versions")
        return Ensemble(**base, status=DONE, xyz_path=xyz, charge=int(charge), multiplicity=int(multiplicity),
                        elements=elements, conformers=conformers, record=record)

    def progress(self, mol: InputMolecule) -> dict:
        """What bceg's records say about stages 1 to 4 for this molecule (read-only, never raises)."""
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
        search = read_record(md.run_record(1))
        refine = read_record(md.run_record(1, self.lot)) if self.lot else None
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
