"""The g-xTB and AIMNet2 refiners, with the xtb binary and the torch / aimnet / ase packages faked.

A fake ``xtb`` on PATH (modelled on ``fake_xtb`` in conftest) stands in for the g-xTB build; ``torch``,
``aimnet.calculators`` and ``ase`` are stand-in modules in ``sys.modules`` so the ASE loop runs for real.
"""

from __future__ import annotations

import multiprocessing
import os
import stat
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pytest

from bceg.config import Config
from bceg.external import ProgramNotFound
from bceg.refine import REFINERS, ase_opt
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


def test_gxtb_holds_the_constrained_atoms_fixed(fake_gxtb, tmp_path):
    """`post_optimization: constrained`: the atoms the constraints name go to xtb as `$fix` (with L-BFGS); the constraints
    enter the refiner's compared parameters only when there are some."""
    structure = Structure("CCO")
    spec = {"freeze": ["[#8]"], "bonds": [{"smarts": "[#6:1]-[#6:2]", "value": 1.5}]}
    refiner = GXTB(constraints=spec)
    assert refiner.compared_params()["constraints"] == spec and "constraints" not in GXTB().compared_params()
    result = refiner.run(structure, _conformers(structure, 1), tmp_path)
    assert result.details["fixed_atoms"] == [0, 1, 2]
    assert (tmp_path / "conf_0" / "xcontrol").read_text() == "$fix\n   atoms: 1,2,3\n$opt\n   engine=lbfgs\n$end\n"
    assert (tmp_path / "conf_0" / "args.txt").read_text().endswith("--input xcontrol")
    with pytest.raises(ValueError, match="unknown keys"):
        GXTB(constraints={"fix": ["[#8]"]})


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
    # the worker processes must see these stand-ins: fork them instead of spawning fresh interpreters
    monkeypatch.setattr(ase_opt, "START_METHOD", "fork")
    return seen


def test_aimnet2_parameters_and_gpu_routing():
    r = REFINERS["aimnet2"]()
    assert r.lot == "aimnet2" and r.params["device"] == "cpu" and r.resource_class() == "cpu"
    assert REFINERS["aimnet2"](device="cuda").resource_class() == "gpu"
    assert REFINERS["aimnet2"](model="aimnet2-2025").lot == "aimnet2-2025"
    assert REFINERS["aimnet2"](lot="aimnet2-wb97m").lot == "aimnet2-wb97m"
    assert set(r.recorded_params()) == {"device", "workers", "threads"}
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


@pytest.mark.parametrize("workers", [1, 2], ids=["in-process", "pool"])
def test_aimnet2_optimises_every_conformer_and_reports_hartree(fake_aimnet, tmp_path, workers):
    structure = Structure("CCO")
    confs = _conformers(structure)
    result = REFINERS["aimnet2"](max_steps=50, workers=workers, threads=1).run(structure, confs, tmp_path)
    assert len(result.conformers) == 2
    for i, (c, start) in enumerate(zip(result.conformers, confs)):
        np.testing.assert_allclose(c.coords, start.coords + 0.1)
        assert c.energy == pytest.approx(-100.0 + start.coords[0, 0] + 0.1)   # eV from the model, Hartree in the result
        assert c.info == {"converged": True, "steps": 3, "fmax": 0.0, "source_index": i}
        assert (tmp_path / f"opt_{i}.log").exists()
    assert set(result.versions) == {"aimnet", "ase"}
    assert result.details == {"n_input": 2, "n_unconverged": 0, "workers": workers, "threads": 1}


def test_aimnet2_unconverged_conformers_are_flagged_or_dropped(fake_aimnet, tmp_path):
    fake_aimnet["force"] = 1.0
    structure = Structure("CCO")
    confs = _conformers(structure)
    kept = REFINERS["aimnet2"](workers=1).run(structure, confs, tmp_path)
    assert [c.info["converged"] for c in kept.conformers] == [False, False] and kept.details["n_unconverged"] == 2
    assert kept.conformers[0].info["fmax"] == pytest.approx(np.sqrt(3.0), abs=1e-5)
    dropped = REFINERS["aimnet2"](keep_unconverged=False, workers=1).run(structure, confs, tmp_path)
    assert dropped.conformers == [] and dropped.details["n_unconverged"] == 2


# --------------------------------------------------------------------------- the worker pool
def test_workers_default_to_the_allocation_on_cpu_and_to_one_on_a_gpu(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert ase_opt.resolve_workers(None, "cpu", 40) == 16
    assert ase_opt.resolve_workers(None, "cpu", 5) == 5         # never more workers than conformers
    assert ase_opt.resolve_workers(None, "cuda", 40) == 1       # processes do not share a GPU
    assert ase_opt.resolve_workers(4, "cuda", 40) == 4          # unless asked for
    assert ase_opt.resolve_workers(0, "cpu", 0) == 1
    assert ase_opt.threads_per_worker(None, 16) == 1 and ase_opt.threads_per_worker(None, 5) == 3
    assert ase_opt.threads_per_worker(None, 1) == 16            # one worker: all the threads, as before
    assert ase_opt.threads_per_worker(2, 16) == 2
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")                   # GOAT-style allocation: ntasks x 1 CPU
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.delenv("SLURM_CPUS_ON_NODE")
    assert ase_opt.resolve_workers(None, "cpu", 40) == 4
    monkeypatch.delenv("SLURM_JOB_ID")                          # outside SLURM: capped, a model copy per worker
    monkeypatch.setattr(ase_opt, "allocated_cpus", lambda: 64)
    assert ase_opt.resolve_workers(None, "cpu", 40) == 8



def test_pool_workers_load_the_model_with_their_share_of_the_threads(fake_aimnet, tmp_path, monkeypatch):
    """Each worker rebuilds the refiner with threads = its share and pins the numerical libraries to it;
    the parent's environment is left as it was."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    structure = Structure("CCO")

    def probe(atoms):   # runs inside a worker: report the pinned thread count through the energy
        return float(os.environ["OMP_NUM_THREADS"]) * HARTREE_TO_EV
    monkeypatch.setattr(sys.modules["aimnet.calculators"].AIMNet2ASE, "energy", staticmethod(probe))
    result = REFINERS["aimnet2"](workers=2).run(structure, _conformers(structure, 3), tmp_path)
    assert result.details == {"n_input": 3, "n_unconverged": 0, "workers": 2, "threads": 2}
    assert [c.energy for c in result.conformers] == pytest.approx([2.0, 2.0, 2.0])
    assert [c.info["source_index"] for c in result.conformers] == [0, 1, 2]   # input order, whatever finished first
    assert sorted(p.name for p in tmp_path.glob("opt_*.log")) == ["opt_0.log", "opt_1.log", "opt_2.log"]
    assert "OMP_NUM_THREADS" not in os.environ
    assert fake_aimnet["models"] == []                                         # the parent never loaded the model


def test_a_worker_exception_fails_the_step_and_a_dead_worker_is_explained(fake_aimnet, tmp_path, monkeypatch):
    structure = Structure("CCO")

    def boom(atoms):
        raise ValueError("Some species are not implemented")
    monkeypatch.setattr(sys.modules["aimnet.calculators"].AIMNet2ASE, "energy", staticmethod(boom))
    with pytest.raises(ValueError, match="not implemented"):
        REFINERS["aimnet2"](workers=2).run(structure, _conformers(structure), tmp_path)

    def die(atoms):     # one worker killed (what the OOM killer leaves behind), the others still busy
        if atoms.positions[0, 0] < 0.5:   # conformer 0
            os._exit(137)
        time.sleep(0.3)
        return 0.0
    monkeypatch.setattr(sys.modules["aimnet.calculators"].AIMNet2ASE, "energy", staticmethod(die))
    with pytest.raises(RuntimeError, match="worker process died.*4 workers x 1 threads.*`mem`"):
        REFINERS["aimnet2"](workers=4, threads=1).run(structure, _conformers(structure, 12), tmp_path)
    assert multiprocessing.active_children() == []   # the survivors are killed: none left to hang the job at exit


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
    result = uma.run(structure, _conformers(structure, 1), tmp_path / "preopt_0")   # created by the loop
    assert (tmp_path / "preopt_0" / "opt_0.log").is_file()
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


def test_the_production_start_method_spawns_workers_that_rebuild_the_refiner(tmp_path, monkeypatch):
    """Spawned workers start from a fresh interpreter: the fakes must come from disk (sys.path travels
    with the spawn), the refiner is rebuilt from its parameters and the results come back in order."""
    fakes = tmp_path / "fakes"
    (fakes / "ase").mkdir(parents=True)
    (fakes / "aimnet").mkdir()
    (fakes / "torch.py").write_text("def set_num_threads(n):\n    import os\n    os.environ['FAKE_TORCH_THREADS'] = str(n)\n")
    (fakes / "ase" / "__init__.py").write_text(textwrap.dedent("""
        import numpy as np
        from . import optimize
        class Atoms:
            def __init__(self, symbols, positions):
                self.symbols, self.positions, self.info, self.calc = list(symbols), np.array(positions, dtype=float), {}, None
            def get_positions(self): return self.positions.copy()
            def get_forces(self): return self.calc.forces(self)
            def get_potential_energy(self): return self.calc.energy(self)
    """))
    (fakes / "ase" / "optimize.py").write_text(textwrap.dedent("""
        from pathlib import Path
        class LBFGS:
            def __init__(self, atoms, logfile=None):
                self.atoms, self.nsteps = atoms, 0
                Path(logfile).write_text("spawned optimiser log\\n")
            def run(self, fmax, steps):
                self.atoms.positions += 0.1
                self.nsteps = 2
    """))
    (fakes / "aimnet" / "__init__.py").write_text("")
    (fakes / "aimnet" / "calculators.py").write_text(textwrap.dedent(f"""
        import os
        import numpy as np
        class AIMNet2Calculator:
            def __init__(self, model, device=None):
                self.model, self.device = model, device
        class AIMNet2ASE:
            def __init__(self, base_calc, charge=0, mult=1):
                self.base_calc, self.charge, self.mult = base_calc, charge, mult
            def energy(self, atoms):   # pid x 1000 + torch threads + x of the first atom, in eV
                return (os.getpid() % 1000 * 1000 + int(os.environ["FAKE_TORCH_THREADS"]) + atoms.positions[0, 0]) * {HARTREE_TO_EV!r}
            def forces(self, atoms):
                return np.zeros(atoms.positions.shape)
    """))
    monkeypatch.syspath_prepend(str(fakes))
    for name in ("torch", "ase", "ase.optimize", "aimnet", "aimnet.calculators"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    structure = Structure("CCO")
    confs = _conformers(structure, 4)
    assert ase_opt.START_METHOD == "spawn"
    (tmp_path / "work").mkdir()
    result = REFINERS["aimnet2"](workers=2).run(structure, confs, tmp_path / "work")
    assert result.details == {"n_input": 4, "n_unconverged": 0, "workers": 2, "threads": 2}
    energies = np.array([c.energy for c in result.conformers])
    pids = set(energies // 1000)
    assert len(pids) == 2 and os.getpid() % 1000 not in pids          # two workers, neither is this process
    np.testing.assert_allclose(energies % 1000, [2 + c.coords[0, 0] + 0.1 for c in confs], atol=1e-6)   # 2 torch threads each
    assert [c.info["source_index"] for c in result.conformers] == [0, 1, 2, 3]
    assert sorted(p.name for p in (tmp_path / "work").glob("opt_*.log")) == [f"opt_{i}.log" for i in range(4)]
