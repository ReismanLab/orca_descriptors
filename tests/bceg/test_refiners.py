"""The g-xTB and AIMNet2 refiners, with the xtb binary and the torch / aimnet / ase packages faked.

A fake ``xtb`` on PATH (modelled on ``fake_xtb`` in conftest) stands in for the g-xTB build; ``torch``,
``aimnet.calculators`` and ``ase`` are stand-in modules in ``sys.modules`` so the ASE loop runs for real.
"""

from __future__ import annotations

import os
import stat
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from bceg.config import Config
from bceg.external import ProgramNotFound
from bceg.refine import REFINERS
from bceg.refine.gxtb import GXTB
from bceg.structure import Structure
from bceg.xyz import HARTREE_TO_EV, Conformer


def _conformers(structure: Structure, n: int = 2) -> list[Conformer]:
    n_atoms = len(structure.elements)
    return [Conformer(np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3) * 0.1 + k) for k in range(n)]


# --------------------------------------------------------------------------- g-xTB
@pytest.fixture
def fake_gxtb(tmp_path, monkeypatch):
    """A g-xTB build of ``xtb`` on PATH: shifts x by 0.1, energy = -42 - x of the first atom.

    ``GXTB_FAKE_MODE``: ``ok`` (default), ``plain`` (a standard xtb: warns about --gxtb and runs GFN2)
    or ``unconverged`` (writes xtblast.xyz, no TOTAL ENERGY line, exit 1).
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "xtb"
    exe.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import os, sys
        if "--version" in sys.argv:
            print("* xtb version 6.7.1 (g-xTB fake) compiled for tests")
            raise SystemExit(0)
        mode = os.environ.get("GXTB_FAKE_MODE", "ok")
        open("args.txt", "w").write(" ".join(sys.argv[1:]))
        if mode == "plain":
            print("Unknown option '--gxtb' provided")
        lines = open(sys.argv[1]).read().splitlines()
        n = int(lines[0]); atoms = [l.split() for l in lines[2:2 + n]]
        energy = -42.0 - float(atoms[0][1])
        out = [str(n), f" energy: {{energy:.10f}} gnorm: 0.0004 xtb: 6.7.1 (fake)"]
        out += [f"{{a[0]}} {{float(a[1]) + 0.1:.8f}} {{a[2]}} {{a[3]}}" for a in atoms]
        if mode == "unconverged":
            open("xtblast.xyz", "w").write("\\n".join(out) + "\\n")
            print("   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION ***")
            raise SystemExit(1)
        open("xtbopt.xyz", "w").write("\\n".join(out) + "\\n")
        print("   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 7 ITERATIONS ***")
        print(f"          | TOTAL ENERGY             {{energy:.12f}} Eh   |")
    """))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GXTB_FAKE_MODE", raising=False)
    return exe


def test_gxtb_parameters_and_resources():
    r = REFINERS["gxtb"]()
    assert r.lot == "gxtb" and r.resource_class() == "cpu"
    assert set(r.compared_params()) == {"level", "cycles", "keep_unconverged", "lot"}
    assert set(r.recorded_params()) == {"executable", "threads"}
    assert GXTB(lot="gxtb-tight", level="tight").lot == "gxtb-tight"
    with pytest.raises(ValueError, match="unknown parameter"):   # g-xTB runs on CPUs: no device to route
        GXTB(device="cuda")
    with pytest.raises(ValueError, match="--opt level"):
        GXTB(level="superb").run(Structure("CCO"), [], Path("."))


def test_gxtb_runs_one_xtb_per_conformer_and_parses_the_output(fake_gxtb, tmp_path):
    structure = Structure("CCO")
    confs = _conformers(structure)
    result = GXTB(threads=2).run(structure, confs, tmp_path)
    assert [c.energy for c in result.conformers] == [-42.0 - confs[0].coords[0, 0], -42.0 - confs[1].coords[0, 0]]
    for i, (c, start) in enumerate(zip(result.conformers, confs)):
        np.testing.assert_allclose(c.coords[:, 0], start.coords[:, 0] + 0.1)   # the optimised geometry, not the input
        np.testing.assert_allclose(c.coords[:, 1:], start.coords[:, 1:])
        assert c.info == {"converged": True, "source_index": i, "cycles": 7}
        sub = tmp_path / f"conf_{i}"
        assert (sub / "xtb.out").exists() and (sub / "start.xyz").exists()
        assert (sub / "args.txt").read_text() == "start.xyz --gxtb --opt normal --chrg 0"   # singlet: no --uhf
    assert result.versions == {"xtb": "* xtb version 6.7.1 (g-xTB fake) compiled for tests"}
    assert result.details == {"executable": str(fake_gxtb), "n_input": 2, "n_unconverged": 0}


def test_gxtb_open_shell_charge_and_cycles_flags(fake_gxtb, tmp_path):
    structure = Structure("C[NH3+]", multiplicity=3)
    GXTB(level="tight", cycles=300).run(structure, _conformers(structure, 1), tmp_path)
    assert (tmp_path / "conf_0" / "args.txt").read_text() == "start.xyz --gxtb --opt tight --chrg 1 --uhf 2 --cycles 300"


def test_gxtb_refuses_a_standard_xtb(fake_gxtb, tmp_path, monkeypatch):
    monkeypatch.setenv("GXTB_FAKE_MODE", "plain")
    structure = Structure("CCO")
    with pytest.raises(RuntimeError, match="not the g-xTB build"):
        GXTB().run(structure, _conformers(structure, 1), tmp_path)


def test_gxtb_unconverged_conformers_are_flagged_or_dropped(fake_gxtb, tmp_path, monkeypatch):
    monkeypatch.setenv("GXTB_FAKE_MODE", "unconverged")
    structure = Structure("CCO")
    confs = _conformers(structure)
    result = GXTB().run(structure, confs, tmp_path / "keep")
    assert [c.info["converged"] for c in result.conformers] == [False, False]
    assert result.conformers[0].energy == pytest.approx(-42.0 - confs[0].coords[0, 0])   # from the xtblast.xyz comment
    np.testing.assert_allclose(result.conformers[1].coords[:, 0], confs[1].coords[:, 0] + 0.1)
    assert result.details["n_unconverged"] == 2
    dropped = GXTB(keep_unconverged=False).run(structure, confs, tmp_path / "drop")
    assert dropped.conformers == [] and dropped.details["n_unconverged"] == 2


def test_gxtb_needs_the_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProgramNotFound, match="no-such-xtb"):
        GXTB(executable="no-such-xtb").run(Structure("CCO"), [], tmp_path)


# --------------------------------------------------------------------------- AIMNet2
@pytest.fixture
def fake_aimnet(monkeypatch):
    """torch, aimnet.calculators and ase stand-ins; ``seen`` records what the refiner did.

    The fake optimiser shifts every coordinate by 0.1 in 3 steps; the fake model's energy is
    (-100 + x of the first atom) Hartree expressed in eV, its forces ``seen["force"]`` everywhere.
    """
    seen = {"threads": [], "models": [], "ase_calcs": [], "force": 0.0, "load_error": None}

    torch = types.ModuleType("torch")
    torch.set_num_threads = seen["threads"].append

    class AIMNet2Calculator:
        def __init__(self, model, device=None):
            if seen["load_error"] is not None:
                raise seen["load_error"]
            self.model, self.device = model, device
            seen["models"].append(self)

        @property
        def is_nse(self):
            return "nse" in self.model

    class AIMNet2ASE:
        def __init__(self, base_calc, charge=0, mult=1):
            self.base_calc, self.charge, self.mult = base_calc, charge, mult
            seen["ase_calcs"].append(self)

        def energy(self, atoms):
            return (-100.0 + atoms.positions[0, 0]) * HARTREE_TO_EV

        def forces(self, atoms):
            return np.full(atoms.positions.shape, seen["force"])

    calculators = types.ModuleType("aimnet.calculators")
    calculators.AIMNet2Calculator, calculators.AIMNet2ASE = AIMNet2Calculator, AIMNet2ASE
    aimnet = types.ModuleType("aimnet")
    aimnet.calculators = calculators

    class Atoms:
        def __init__(self, symbols, positions):
            self.symbols, self.positions, self.info, self.calc = list(symbols), np.array(positions, dtype=float), {}, None

        def get_positions(self):
            return self.positions.copy()

        def get_forces(self):
            return self.calc.forces(self)

        def get_potential_energy(self):
            return self.calc.energy(self)

    class LBFGS:
        def __init__(self, atoms, logfile=None):
            self.atoms, self.nsteps = atoms, 0
            Path(logfile).write_text("fake optimiser log\n")

        def run(self, fmax, steps):
            self.atoms.positions += 0.1
            self.nsteps = 3

    optimize = types.ModuleType("ase.optimize")
    optimize.LBFGS = LBFGS
    ase = types.ModuleType("ase")
    ase.Atoms, ase.optimize = Atoms, optimize
    for name, module in {"torch": torch, "aimnet": aimnet, "aimnet.calculators": calculators, "ase": ase, "ase.optimize": optimize}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return seen


def test_aimnet2_parameters_and_gpu_routing():
    r = REFINERS["aimnet2"]()
    assert r.lot == "aimnet2" and r.params["device"] == "cpu" and r.resource_class() == "cpu"
    assert REFINERS["aimnet2"](device="cuda").resource_class() == "gpu"
    assert REFINERS["aimnet2"](model="aimnet2-2025").lot == "aimnet2-2025"
    assert REFINERS["aimnet2"](lot="aimnet2-wb97m").lot == "aimnet2-wb97m"
    assert set(r.recorded_params()) == {"device", "threads"}
    assert set(r.compared_params()) == {"model", "optimizer", "fmax", "max_steps", "keep_unconverged", "lot"}
    with pytest.raises(ValueError, match="unknown parameter"):
        REFINERS["aimnet2"](task="omol")


def test_aimnet2_loads_the_model_once_on_the_requested_device_with_pinned_threads(fake_aimnet, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    r = REFINERS["aimnet2"](device="cuda")
    calc = r.calculator(Structure("C[NH3+]"))
    r.calculator(Structure("CCO"))
    assert fake_aimnet["threads"] == [8]
    assert [(m.model, m.device) for m in fake_aimnet["models"]] == [("aimnet2", "cuda")]   # loaded once
    assert (calc.charge, calc.mult) == (1, 1) and [c.charge for c in fake_aimnet["ase_calcs"]] == [1, 0]
    REFINERS["aimnet2"](threads=2).calculator(Structure("CCO"))
    assert fake_aimnet["threads"] == [8, 2]


def test_aimnet2_open_shell_needs_an_nse_model(fake_aimnet):
    radical = Structure("C[NH2+]", multiplicity=2)
    with pytest.raises(ValueError, match="aimnet2-nse"):
        REFINERS["aimnet2"]().calculator(radical)
    calc = REFINERS["aimnet2"](model="aimnet2-nse").calculator(radical)
    assert (calc.charge, calc.mult) == (1, 2)


def test_aimnet2_optimises_every_conformer_and_reports_hartree(fake_aimnet, tmp_path):
    structure = Structure("CCO")
    confs = _conformers(structure)
    result = REFINERS["aimnet2"](max_steps=50).run(structure, confs, tmp_path)
    assert len(result.conformers) == 2
    for i, (c, start) in enumerate(zip(result.conformers, confs)):
        np.testing.assert_allclose(c.coords, start.coords + 0.1)
        assert c.energy == pytest.approx(-100.0 + start.coords[0, 0] + 0.1)   # eV from the model, Hartree in the result
        assert c.info == {"converged": True, "steps": 3, "fmax": 0.0, "source_index": i}
        assert (tmp_path / f"opt_{i}.log").exists()
    assert set(result.versions) == {"aimnet", "ase"}
    assert result.details == {"n_input": 2, "n_unconverged": 0}


def test_aimnet2_unconverged_conformers_are_flagged_or_dropped(fake_aimnet, tmp_path):
    fake_aimnet["force"] = 1.0
    structure = Structure("CCO")
    confs = _conformers(structure)
    kept = REFINERS["aimnet2"]().run(structure, confs, tmp_path)
    assert [c.info["converged"] for c in kept.conformers] == [False, False] and kept.details["n_unconverged"] == 2
    assert kept.conformers[0].info["fmax"] == pytest.approx(np.sqrt(3.0), abs=1e-5)
    dropped = REFINERS["aimnet2"](keep_unconverged=False).run(structure, confs, tmp_path)
    assert dropped.conformers == [] and dropped.details["n_unconverged"] == 2


def test_aimnet2_download_failure_points_to_the_cache(fake_aimnet):
    class ConnectionError(Exception):
        __module__ = "requests.exceptions"

    fake_aimnet["load_error"] = ConnectionError("Max retries exceeded")
    with pytest.raises(RuntimeError, match="AIMNET_CACHE_DIR") as info:
        REFINERS["aimnet2"]().base_calculator()
    assert "aimnet download aimnet2" in str(info.value)
    fake_aimnet["load_error"] = KeyError("no such model")   # anything else surfaces as it is
    with pytest.raises(KeyError):
        REFINERS["aimnet2"]().base_calculator()


def test_uma_runs_through_the_shared_ase_loop(fake_aimnet, tmp_path):
    """UMA.run moved to bceg.refine.ase_opt; it must still convert eV to Hartree and flag convergence."""
    uma = REFINERS["uma"]()
    uma._calc = sys.modules["aimnet.calculators"].AIMNet2ASE(None)   # any ASE-like calculator will do
    structure = Structure("CCO")
    result = uma.run(structure, _conformers(structure, 1), tmp_path)
    assert result.conformers[0].energy == pytest.approx(-100.0 + 0.1) and result.conformers[0].info["converged"]
    assert set(result.versions) == {"fairchem-core", "ase"}


# --------------------------------------------------------------------------- configuration
def test_bceg_config_accepts_the_new_refiners(tmp_path):
    raw = {"results_dir": str(tmp_path), "inputs": None, "transformations": {"none": {"script": None}},
           "methods": {"etkdg": {"searcher": "rdkit_etkdg", "params": {"n_confs": 2}}}}
    cfg = Config.from_raw({**raw, "post_optimization": {"refiner": "gxtb", "params": {"level": "tight"}}}, tmp_path / "c.yaml")
    assert cfg.refiner.name == "gxtb" and cfg.refiner.lot == "gxtb" and cfg.refiner.resource_class() == "cpu"
    cfg = Config.from_raw({**raw, "post_optimization": {"refiner": "aimnet2", "params": {"device": "cuda"}}}, tmp_path / "c.yaml")
    assert cfg.refiner.lot == "aimnet2" and cfg.refiner.resource_class() == "gpu"
    with pytest.raises(ValueError, match="unknown parameter"):
        Config.from_raw({**raw, "post_optimization": {"refiner": "gxtb", "params": {"device": "cuda"}}}, tmp_path / "c.yaml")
