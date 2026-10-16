from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bceg import constraints as cons
from bceg.chem import rotatable_bonds
from bceg.geometry import make_start_geometry
from bceg.params import Configurable
from bceg.structure import Structure
from bceg.xyz import Conformer, write_xyz

SEED_RULE = "seed = seed_offset + replicate"

# Keys shared by searchers that start from a single 3D geometry (CREST, GOAT, molassembler).
# See bceg.geometry for the procedure.
START_DEFAULTS = {
    "start_geometry": "etkdg",  # "etkdg": new ETKDG embedding per replicate seed; "input": coords from the transformation
    "start_ff": "auto",  # relax the ETKDG start: auto (MMFF94s if parametrised, else none) | MMFF94 | MMFF94s | UFF | none
    "start_preopt": "gfn2",  # pre-optimisation of the start: gfn2 | gfn1 | gfn0 | gfnff (xtb) | uma | none
    "start_preopt_level": "normal",  # xtb --opt level: crude | sloppy | loose | lax | normal | tight | vtight | extreme
    "start_preopt_params": {},  # for start_preopt: uma, the UMA refiner's parameters (model, task, fmax, max_steps, ...)
    "max_metal_ligand_distance": 2.5,  # Angstrom; longer bonds to a metal reject the start geometry
    "start_max_attempts": 10,  # re-embeddings (new seeds) before giving up
    "start_allow_decoordination": False,  # True: a metal-ligand bond longer than max_metal_ligand_distance is recorded, not rejected
    "xtb_executable": "xtb",
}
START_RESOURCE_KEYS = {"xtb_executable"}

# ``n_confs`` may be a number or a formula in the molecule's rotatable-bond count ``n_rot``
# (bceg.chem.rotatable_bonds), e.g. "(n_rot + 2) * 30"; see n_confs_formula().
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}
_FUNCS = {"min": min, "max": max}


def n_confs_formula(formula: str, n_rot: int) -> int:
    """Evaluate an ``n_confs`` formula (numbers, ``n_rot``, + - * / // **, min, max), rounded to an int >= 1."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id == "n_rot":
            return n_rot
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and node.args and not node.keywords):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"n_confs formula {formula!r}: unsupported expression {ast.unparse(node)!r} "
                         f"(allowed: numbers, n_rot, + - * / // **, min, max)")

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"n_confs formula {formula!r}: {e.msg}") from None
    return max(1, round(ev(tree)))


# Searchers that honour geometric constraints during the search (CREST, GOAT); see bceg.constraints.
CONSTRAINT_DEFAULTS = {
    "constraints": {},  # freeze / bonds / angles / dihedrals / planes selected by SMARTS, force_constant
    "start_constraints": False,  # impose the constraints on the start geometry, held fixed in its pre-optimisation
    "start_impose_steps": 10,  # steps of the ramp that brings the start geometry onto the imposed values
}
#: compared only when start_constraints is on, so that the hash of every other search is unchanged
START_CONSTRAINT_KEYS = ("start_constraints", "start_impose_steps")


@dataclass
class SearchResult:
    conformers: list[Conformer]
    versions: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    log_files: list[Path] = field(default_factory=list)  # copied to <method>/logs/


class Searcher(Configurable):
    #: Optional callback ``f(info)`` run once the start geometry is accepted (bceg.tasks uses it to move
    #: the run record from phase "pre_refinement" to "search").
    on_start_geometry = None
    #: True when the program's constraints are exact and take no force constant (GOAT).
    exact_constraints_only = False

    def __init__(self, **params: Any) -> None:
        formula = params.get("n_confs") if isinstance(params.get("n_confs"), str) and "n_confs" in self.defaults else None
        if formula is not None:
            n_confs_formula(formula, 0)  # syntax check
            params = {**params, "n_confs": self.defaults["n_confs"]}
        super().__init__(**params)
        if formula is not None:
            self.params["n_confs"] = formula
        if "constraints" in self.params:
            spec = cons.validate(self.params["constraints"], f"{self.name}.constraints")
            if self.exact_constraints_only and cons.has_force_constant(spec):
                raise ValueError(f"{self.name}.constraints: the constraints of {self.name} are exact, "
                                 f"remove `force_constant`")
            if self.params["start_constraints"] and not spec:
                raise ValueError(f"{self.name}.start_constraints: there are no constraints to impose")
            if self.params["start_impose_steps"] < 1:
                raise ValueError(f"{self.name}.start_impose_steps must be at least 1")

    @classmethod
    def supports_constraints(cls) -> bool:
        return "constraints" in cls.defaults

    def compared_params(self) -> dict[str, Any]:
        """No ``constraints`` key when there are none, no ``start_constraints`` / ``start_impose_steps`` unless
        the constraints are imposed, and no ``start_allow_decoordination`` unless it is on, so the hash of
        every search without them is unchanged."""
        out = super().compared_params()
        if "constraints" in out and not out["constraints"]:
            del out["constraints"]
        if not out.get("start_constraints", True):
            for key in START_CONSTRAINT_KEYS:
                out.pop(key, None)
        if not out.get("start_allow_decoordination", True):  # the default: absent from the hash, as before
            out.pop("start_allow_decoordination", None)
        return out

    def resolve_constraints(self, structure: Structure, coords: np.ndarray) -> dict | None:
        """The constraints for this molecule and start geometry (0-based indices), None without any."""
        if not self.params.get("constraints"):
            return None
        return cons.resolve(self.params["constraints"], structure.mol, coords)

    def n_confs(self, structure: Structure) -> tuple[int, int]:
        """(conformers to request, rotatable bonds of ``structure``); ``n_confs`` may be a formula in ``n_rot``."""
        n_rot = rotatable_bonds(structure.mol)
        n = self.params["n_confs"]
        return (n_confs_formula(n, n_rot) if isinstance(n, str) else n), n_rot

    def has_start_step(self) -> bool:
        """True when this searcher starts from one pre-optimised geometry (CREST, GOAT, molassembler)."""
        return "start_preopt" in self.params and self.params["start_preopt"] != "none"

    def seed(self, replicate: int) -> int:
        return self.params["seed_offset"] + replicate

    def applicable(self, structure: Structure) -> str | None:
        """Return a reason string if this searcher cannot handle ``structure``."""
        return None

    def run(self, structure: Structure, replicate: int, workdir: Path) -> SearchResult:
        raise NotImplementedError

    def start_geometry(self, structure: Structure, seed: int, workdir: Path) -> tuple[np.ndarray, dict]:
        """Checked (and optionally xtb pre-optimised) start geometry; also written to start_geometry.xyz."""
        coords, info = make_start_geometry(structure, seed, workdir, self.params)
        write_xyz(workdir / "start_geometry.xyz", structure.elements, [Conformer(coords, info["preopt_energy"])])
        if self.on_start_geometry is not None:
            self.on_start_geometry(info)
        return coords, info
