from __future__ import annotations

import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from bceg.refine import REFINERS, RefineResult, Refiner
from bceg.search import SEARCHERS, SearchResult, Searcher
from bceg.search.base import START_DEFAULTS, START_RESOURCE_KEYS
from bceg.xyz import Conformer


class FakeRefiner(Refiner):
    """Stand-in for UMA: shifts nothing, assigns energy = -index."""

    name = "fake"
    defaults = {"label": "fake-lot", "scale": 1.0, "device": "cpu"}
    resource_keys = {"device"}

    @property
    def lot(self) -> str:
        return self.params["label"]

    def run(self, structure, conformers, workdir):
        out = [Conformer(c.coords, -float(i) * self.params["scale"], {"converged": True}) for i, c in enumerate(conformers)]
        return RefineResult(out, versions={"fake": "0"})


REFINERS["fake"] = FakeRefiner


class SingleStart(Searcher):
    """Test searcher going through the real start-geometry path (xtb pre-opt + checks)."""

    name = "single_start"
    defaults = {**START_DEFAULTS, "seed_offset": 0}
    resource_keys = START_RESOURCE_KEYS

    def run(self, structure, replicate, workdir):
        coords, start = self.start_geometry(structure, self.seed(replicate), workdir)
        return SearchResult([Conformer(coords, start["preopt_energy"])], details={"start_geometry": start},
                            log_files=[workdir / "start_geometry.xyz"])


SEARCHERS["single_start"] = SingleStart


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    """An `xtb` on PATH that returns the input geometry and logs its arguments."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "xtb"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import shutil, sys
        open("args.txt", "w").write(" ".join(sys.argv[1:]))
        shutil.copy(sys.argv[1], "xtbopt.xyz")
        print("          | TOTAL ENERGY               -42.000000000000 Eh   |")
    """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return exe


@pytest.fixture
def project(tmp_path: Path):
    """A minimal project: 2 organics, ETKDG method, fake post-optimisation."""
    (tmp_path / "inputs.csv").write_text("name,smiles\nbutane,CCCC\nethanol,OCC\n")
    (tmp_path / "protonate.py").write_text(textwrap.dedent('''
        def transform(smiles):
            if smiles == "CCO":
                return {"smiles": "CC[OH2+]", "metadata": {"note": "protonated"}}
            return None
    '''))

    def write(params: str = "{n_confs: 4}", refiner_params: str = "{}", mode: str = "chained",
              searcher: str = "rdkit_etkdg", transformations: str = "{none: null, H: protonate.py}",
              method_slurm: str = "{}", refiner_slurm: str = "{}") -> Path:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(f"""
            inputs: inputs.csv
            results_dir: results
            replicates: 3
            transformations: {transformations}
            methods:
              etkdg:
                searcher: {searcher}
                params: {params}
                slurm: {method_slurm}
            post_optimization:
              refiner: fake
              params: {refiner_params}
              slurm: {refiner_slurm}
            slurm:
              mode: {mode}
              cpu: {{partition: short, cpus_per_task: 2}}
              gpu: {{partition: gpu, gres: "gpu:1"}}
        """))
        return cfg

    return tmp_path, write
