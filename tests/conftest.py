from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bceg.refine import REFINERS, RefineResult, Refiner
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

    def write(params: str = "{n_confs: 4}", refiner_params: str = "{}", extra: str = "") -> Path:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(f"""
            inputs: inputs.csv
            results_dir: results
            replicates: 3
            transformations:
              none: null
              H: protonate.py
            methods:
              etkdg:
                searcher: rdkit_etkdg
                params: {params}
            post_optimization:
              refiner: fake
              params: {refiner_params}
            slurm:
              cpu: {{partition: short, cpus_per_task: 2}}
            {extra}
        """))
        return cfg

    return tmp_path, write
