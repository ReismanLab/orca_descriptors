"""Where the conformer ensembles come from.

``kind: bceg`` (the normal case): a bceg results tree. For every molecule of the input list the
ensemble is ``<results_dir>/<transformation>/<InChIKey>/<method>/raw_1_<lot>.xyz`` (replicate 1,
post-optimised at ``lot``; ``lot: null`` takes the raw search output), its hyper-parameters are
``params.json`` / ``params_<lot>.json`` and its outcome ``run_1[_<lot>].json``. Charge and multiplicity
are read from ``params.json`` (bceg's ``structure``), never from the library config.

``kind: xyz``: one multi-xyz file for one molecule, with charge and multiplicity given explicitly
(legacy or hand-made ensembles). Energies, if any, are the first float of each comment line (Hartree).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bceg.config import Config as BcegConfig
from bceg.config import read_raw as bceg_read_raw
from bceg.config import select as bceg_select
from bceg.config import write_raw as bceg_write_raw
from bceg.inputs import InputMolecule, load_inputs, update_index
from bceg.layout import method_dir
from bceg.provenance import ParamsMismatch, read_record
from bceg.xyz import Conformer, read_xyz

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


class BcegSource:
    """Ensembles of one (transformation, method, lot) in a bceg results tree."""

    def __init__(self, source: dict) -> None:
        self.source = source
        raw = bceg_read_raw(source["config"])
        raw = bceg_select(raw, smiles=source["smiles"], transformations=[source["transformation"]],
                          methods=[source["method"]])
        raw["results_dir"] = source["results_dir"]
        self.raw = raw
        self.bcfg = BcegConfig.from_raw(raw, Path(source["config"]).resolve())
        self.transformation = self.bcfg.transformations[source["transformation"]]
        self.method = source["method"]
        self.lot = source.get("lot")
        if self.lot is None and self.bcfg.refiner is not None:
            self.lot = self.bcfg.refiner.lot
        self.results_dir = Path(source["results_dir"])

    def molecules(self) -> list[InputMolecule]:
        return load_inputs(self.source["smiles"])

    def ensemble(self, mol: InputMolecule) -> Ensemble:
        base = dict(inchikey=mol.inchikey, input_smiles=mol.input_smiles, canonical_smiles=mol.canonical_smiles,
                    name=mol.name)
        md = method_dir(self.results_dir, self.transformation.name, mol.inchikey, self.method)
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
                                   + (f", {run_rec.get('error')}" if run_rec.get("error") else ""))

        struct = ((params or {}).get("compared") or {}).get("structure") or {}
        charge = struct.get("charge", structure.charge)
        multiplicity = struct.get("multiplicity", structure.multiplicity)
        elements, conformers = read_xyz(xyz)
        record["n_conformers"] = len(conformers)
        record["versions"] = run_rec.get("versions")
        return Ensemble(**base, status=DONE, xyz_path=xyz, charge=int(charge), multiplicity=int(multiplicity),
                        elements=elements, conformers=conformers, record=record)

    # ------------------------------------------------------------------ running bceg
    def pending_tasks(self) -> tuple[list, list[str]]:
        """bceg tasks still to run for this selection, and params-mismatch messages (read-only)."""
        from bceg.tasks import expand
        return expand(self.bcfg, self.molecules(), write_params=False)

    def submit(self, jobdir: Path, no_sbatch: bool = False) -> list[str]:
        """Run bceg's submit for this selection; return the SLURM job ids it started.

        Freezes bceg's resolved config under ``jobdir/bceg/`` exactly like ``bceg submit`` does, so
        the two stages of one submission share one ``_jobs/<timestamp>`` folder.
        """
        from bceg.slurm import write_jobs
        from bceg.tasks import expand

        molecules = self.molecules()
        tasks, problems = expand(self.bcfg, molecules, write_params=False)
        if problems:
            raise ParamsMismatch("bceg hyper-parameters differ from the existing logs:\n" + "\n\n".join(problems))
        if not tasks:
            print("bceg: nothing pending for this selection.")
            return []
        bceg_dir = jobdir / "bceg"
        bceg_dir.mkdir(parents=True, exist_ok=True)
        bceg_write_raw(self.raw, bceg_dir / "config.yaml")
        cfg = BcegConfig.load(bceg_dir / "config.yaml")
        update_index(cfg.results_dir, molecules)
        tasks, _ = expand(cfg, molecules, write_params=True)
        write_jobs(cfg, tasks, bceg_dir)
        print(f"bceg: {len(tasks)} task(s); job scripts in {bceg_dir}")
        if no_sbatch:
            return []
        result = subprocess.run(["bash", str(bceg_dir / "submit.sh")], capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            raise RuntimeError(f"bceg submit.sh failed ({result.returncode}): {result.stderr.strip()}")
        return re.findall(r":\s*(\d+)\s*$", result.stdout, re.MULTILINE)


class XyzSource:
    """One hand-made ensemble file for one molecule."""

    def __init__(self, source: dict) -> None:
        self.source = source
        self.path = Path(source["file"])

    def molecules(self) -> list[InputMolecule]:
        return [InputMolecule.from_smiles(self.source["smiles"], self.source.get("name"))]

    def ensemble(self, mol: InputMolecule) -> Ensemble:
        base = dict(inchikey=mol.inchikey, input_smiles=mol.input_smiles, canonical_smiles=mol.canonical_smiles,
                    name=mol.name)
        record = {"kind": "xyz", "xyz": str(self.path)}
        if not self.path.exists():
            return Ensemble(**base, status="ensemble_missing", record=record, reason=f"{self.path} not found")
        elements, conformers = read_xyz(self.path)
        record["n_conformers"] = len(conformers)
        return Ensemble(**base, status=DONE, xyz_path=self.path, charge=int(self.source["charge"]),
                        multiplicity=int(self.source["multiplicity"]), elements=elements, conformers=conformers,
                        record=record)

    def pending_tasks(self):
        return [], []

    def submit(self, jobdir: Path, no_sbatch: bool = False) -> list[str]:
        return []


def make_source(source: dict):
    return BcegSource(source) if source["kind"] == "bceg" else XyzSource(source)
