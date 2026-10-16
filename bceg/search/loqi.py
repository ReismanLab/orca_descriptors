"""LoQI conformer generator (organics only): https://github.com/isayevlab/LoQI

Called through the ``loqi`` console script of the ``loqi`` pip package (``loqi sample``). The
checkpoints are registered by name (``loqi``, ``loqi_flow``, ``loqi_flow_v1.1.0``) and cached in
``$LOQI_CACHE_DIR`` (else ``~/.cache/loqi``); a compute node cannot download them, so they are fetched
once on a login node with ``loqi download --model <name>`` into the cache that ``slurm.setup`` exports.
Flags checked against upstream ``src/loqi/cli.py`` (loqi 1.1.0, 2026-10-02): ``--smiles``, ``--n-confs``,
``--output``, ``--model``, ``--config``, ``--device``, ``--seed`` (default 42, so it must be passed for the
replicates to differ), ``--steps``, ``--batch-atoms``, ``--no-add-hs``. There is no ``--version``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from rdkit import Chem

from bceg.chem import reorder_to_template
from bceg.external import ProgramNotFound, find_executable, program_version, resolve_threads, run
from bceg.search.base import SearchResult, Searcher
from bceg.structure import Structure
from bceg.xyz import Conformer

INSTALL_HINT = 'pip install "loqi @ git+https://github.com/isayevlab/LoQI"'


def cache_dir() -> Path:
    """Where LoQI looks for its checkpoints: ``$LOQI_CACHE_DIR`` or ``~/.cache/loqi`` (upstream rule)."""
    env = os.environ.get("LOQI_CACHE_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "loqi"


def _interpreter_of(script: str) -> str | None:
    """Python named on the shebang line of a console script (the env that installed ``loqi``)."""
    try:
        with open(script, "rb") as fh:
            first = fh.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    words = first[2:].decode(errors="replace").split()
    if not words:
        return None
    return words[1] if words[0].endswith("/env") and len(words) > 1 else words[0]


def loqi_version(exe: str) -> str:
    """Version of the installed ``loqi`` package (the CLI has no ``--version``)."""
    python = _interpreter_of(exe) or str(Path(exe).parent / "python")
    code = ("import importlib.metadata as m\n"
            "try: print(m.version('loqi'))\n"
            "except m.PackageNotFoundError: print('unknown')")
    return program_version([python, "-c", code], match="") or "unknown"


class LoQI(Searcher):
    name = "loqi"
    defaults = {
        "n_confs": 100,
        "model": "loqi_flow_v1.1.0",  # registered checkpoint (loqi | loqi_flow | loqi_flow_v1.1.0) or a .ckpt path
        "config": None,  # bundled config name or YAML path; only for a checkpoint that is not in the registry
        "steps": None,  # sampling steps; None: the checkpoint's training value
        "add_hs": True,  # let LoQI add the hydrogens of the SMILES (False: --no-add-hs)
        "extra_args": [],  # additional CLI arguments, appended verbatim
        "seed_offset": 0,
        "device": None,  # resource: torch device; None: cuda if available, else cpu
        "batch_atoms": None,  # resource: atom budget per batch (memory), None: the model's default
        "threads": None,  # resource: CPU threads of the torch side (None/0: SLURM_CPUS_PER_TASK)
        "executable": "loqi",
    }
    resource_keys = {"device", "batch_atoms", "threads", "executable"}

    def resource_class(self) -> str:
        return "cpu" if self.params["device"] == "cpu" else "gpu"

    def applicable(self, structure: Structure) -> str | None:
        if not structure.is_organic:
            return "LoQI is trained on organic (ChEMBL) molecules only"
        return None

    def executable(self) -> str:
        try:
            return find_executable(self.params["executable"])
        except ProgramNotFound as exc:
            raise ProgramNotFound(
                f"{exc}. The `loqi` command comes with the loqi pip package of the cluster environment "
                f"(gp_env_cluster.yml); if it is missing there, install it into that env: {INSTALL_HINT}") from None

    def checkpoint(self) -> Path | None:
        """Cached checkpoint file of a registered model; None for a checkpoint given by path."""
        model = self.params["model"]
        return None if os.sep in model or model.endswith(".ckpt") else cache_dir() / f"{model}.ckpt"

    def _weights_hint(self) -> str:
        ckpt = self.checkpoint()
        if ckpt is None or ckpt.is_file():
            return ""
        return (f"\nThe checkpoint {ckpt} is not in the cache and a compute node cannot download it. Fetch it once "
                f"on a login node into the shared cache, with the same LOQI_CACHE_DIR as slurm.setup:\n"
                f"    export LOQI_CACHE_DIR={cache_dir()}\n"
                f"    loqi download --model {self.params['model']}")

    def command(self, structure: Structure, replicate: int) -> list[str]:
        p = self.params
        cmd = [self.executable(), "sample", "--smiles", structure.smiles, "--n-confs", str(self.n_confs(structure)[0]),
               "--output", "confs.sdf", "--model", p["model"], "--seed", str(self.seed(replicate))]
        if p["config"]:
            cmd += ["--config", p["config"]]
        if p["device"]:
            cmd += ["--device", p["device"]]
        if p["steps"] is not None:
            cmd += ["--steps", str(p["steps"])]
        if p["batch_atoms"] is not None:
            cmd += ["--batch-atoms", str(p["batch_atoms"])]
        if not p["add_hs"]:
            cmd.append("--no-add-hs")
        return cmd + list(p["extra_args"])

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        cmd = self.command(structure, replicate)
        proc = run(cmd, workdir, threads=resolve_threads(self.params["threads"]), log_name="loqi.out")
        sdf = workdir / "confs.sdf"
        if proc.returncode != 0 or not sdf.exists():
            raise RuntimeError(f"LoQI failed (exit {proc.returncode}):\n{proc.stdout[-3000:]}{self._weights_hint()}")

        confs, n_failed = [], 0
        records = [] if sdf.stat().st_size == 0 else Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
        for mol in records:  # RDKit refuses to open an empty file; LoQI writes one when it skipped the SMILES
            if mol is None:
                continue
            if mol.HasProp("loqi_failed"):  # upstream: samples of the molecule that produced no geometry
                n_failed = max(n_failed, int(mol.GetProp("loqi_failed")))
            order = reorder_to_template(mol, structure.mol)
            confs.append(Conformer(np.array(mol.GetConformer().GetPositions())[order]))
        if not confs:
            raise RuntimeError(f"LoQI returned no readable conformers:\n{proc.stdout[-3000:]}")
        ckpt = self.checkpoint()
        return SearchResult(
            confs,
            versions={"loqi": loqi_version(cmd[0])},
            details={"seed": self.seed(replicate), "model": self.params["model"],
                     "checkpoint": str(ckpt) if ckpt else self.params["model"],
                     "n_sampled": len(confs), "n_failed_samples": n_failed},
            log_files=[workdir / "loqi.out"],
        )
