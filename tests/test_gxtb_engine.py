"""Stage 6 with ``dft.engine: gxtb``: the parsers, the condensed file, the runner and the config guards.

The fixtures under ``tests/reference/gxtb/`` are real output of the g-xTB build of xtb 6.7.1
(``gxtb-2.0.1``) on methanol: a single point with ``--vfukui``, an ``--ohess`` run and a cation single
point. They are what the parsers are pinned against, and what the fake binary replays.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest
import yaml

from desc_utils import gxtb_condense as gc
from desc_utils import gxtb_parse as gp
from desc_utils.condense import CondenseError, check_condensed, load_condensed
from desc_utils.submit import slurm
from desc_utils.submit.config import Config
from desc_utils.submit.gxtb import run_gxtb_conformer, step_command, steps_to_run, uhf
from desc_utils.submit.tasks import prepare_molecule

from tests.test_submit import _rewrite, fake_sbatch, tree  # noqa: F401  (fixtures)

REF = Path(__file__).parent / "reference" / "gxtb"
N_ATOMS = 6  # methanol


def _text(step: str) -> str:
    return (REF / step / "xtb.out").read_text()


# --------------------------------------------------------------------------- parsers

def test_singlepoint_parsers_read_the_real_output():
    text = _text("singlepoint")
    assert gp.version(text) == "6.7.1"
    assert gp.terminated_normally(text)
    assert gp.total_energy(text) == pytest.approx(-115.720110985867)
    assert gp.electron_counts(text) == {"n_electrons": 14, "n_unpaired": 0, "charge": 0,
                                        "n_alpha": 7, "n_beta": 7, "multiplicity": 1}
    assert gp.elements(text) == ["C", "O", "H", "H", "H", "H"]
    assert gp.atomic_numbers(text) == [6, 8, 1, 1, 1, 1]
    # the charges file carries more digits than the log's table, and both agree
    assert gp.charges_file(REF / "singlepoint" / "charges") == pytest.approx(gp.atomic_charges(text), abs=1e-6)
    assert gp.shell_populations(text)[0] == {"atom": 1, "element": "C", "atomic_number": 6,
                                             "populations": [1.143528, 2.921823]}
    dip = gp.dipole(text)
    assert (dip["x"], dip["y"], dip["total"]) == pytest.approx((-0.443169, 0.75716, 0.877319))
    assert len(gp.fukui(text)) == N_ATOMS
    assert gp.fukui(text)[0] == {"atom": 1, "element": "C", "f_plus": 0.358, "f_minus": 0.188, "f_zero": 0.273}
    assert gp.wiberg(REF / "singlepoint" / "wbo")[0] == {"i": 1, "j": 2, "order": pytest.approx(0.9694709, abs=1e-6)}


def test_restricted_and_unrestricted_orbital_tables():
    """A closed shell prints one table; an open shell prints an alpha one and a beta one."""
    closed = gp.orbitals(_text("singlepoint"))
    assert closed["spin"] == "restricted" and "beta" not in closed
    assert closed["occupations"][:3] == [2.0, 2.0, 2.0]
    assert closed["occupations"][-1] == 0.0          # a virtual has a blank occupation column
    assert len(closed["energies_eh"]) == len(closed["occupations"])

    opened = gp.orbitals(_text("oxidation"))
    assert opened["spin"] == "alpha" and "beta" in opened
    assert opened["occupations"][:3] == [1.0, 1.0, 1.0]
    assert len(opened["beta"]["energies_eh"]) == len(opened["beta"]["occupations"])
    assert gp.electron_counts(_text("oxidation")) == {"n_electrons": 13, "n_unpaired": 1, "charge": 1,
                                                      "n_alpha": 7, "n_beta": 6, "multiplicity": 2}


def test_frequencies_and_thermochemistry():
    freq = gp.frequencies(REF / "frequency" / "g98.out")
    assert len(freq["modes"]) == 3 * N_ATOMS - 6     # xtb drops the translations and rotations itself
    assert freq["n_imaginary"] == 1
    first, last = freq["modes"][0], freq["modes"][-1]
    assert first["frequency_cm-1"] == pytest.approx(-358.148)
    assert first["ir_intensity_km_per_mol"] == pytest.approx(144.8841)
    assert last["frequency_cm-1"] == pytest.approx(3699.76)
    assert all(len(m["displacements"]) == N_ATOMS for m in freq["modes"])
    assert first["displacements"][0] == {"dx": -0.0, "dy": -0.0, "dz": -0.06}

    thermo = gp.thermochemistry(_text("frequency"))
    assert thermo["enthalpy_eh"] == pytest.approx(-115.667035692346)
    assert thermo["gibbs_eh"] == pytest.approx(-115.693426269961)
    assert thermo["temperature_K"] == pytest.approx(298.15)
    assert gp.thermochemistry(_text("singlepoint")) is None   # no --hess, no thermochemistry


def test_a_long_orbital_table_keeps_its_frontier():
    """Above ~40 orbitals xtb prints only the ends of the table and elides the middle with a '...' row.

    The parser must skip that row instead of stopping at it, or everything past it is lost and the
    LUMO disappears. Pinned against the values xtb itself printed for this 35-atom Ni complex.
    """
    text = (REF / "metal_singlepoint" / "xtb.out").read_text()
    orb = gp.orbitals(text)
    assert orb["truncated"] is True
    assert orb["indices"][0] == 1 and orb["indices"][1] == 38, "the gap is real and is recorded"
    assert len(orb["energies_eh"]) == len(orb["occupations"]) == len(orb["indices"])
    occupied = [e for e, o in zip(orb["energies_eh"], orb["occupations"]) if o in (1.0, 2.0)]
    virtual = [e for e, o in zip(orb["energies_eh"], orb["occupations"]) if o == 0.0]
    assert max(occupied) * gp.HARTREE_TO_EV == pytest.approx(-7.2708, abs=1e-3)   # xtb's own (HOMO)
    assert min(virtual) * gp.HARTREE_TO_EV == pytest.approx(-2.1534, abs=1e-3)    # xtb's own (LUMO)
    assert len(gp.charges_file(REF / "metal_singlepoint" / "charges")) == 35


def test_homo_lumo_survive_the_round_trip_through_the_condensed_file(tmp_path: Path):
    """The frontier orbitals of a metal complex must reach the descriptor table, gap and all."""
    import pandas as pd
    from desc_utils.extract.orca import extract_charges

    d = tmp_path / "conf_1"
    (d / gc.RUN_DIRNAME).mkdir(parents=True)
    shutil.copytree(REF / "metal_singlepoint", d / gc.RUN_DIRNAME / "singlepoint")
    shutil.copyfile(REF / "metal.xyz", d / "optimization.xyz")
    assert gc.condense_folder(d)["ok"]
    assert check_condensed(d / "orca_condensed.json") == []

    row = extract_charges(pd.DataFrame([{"path": str(d), "Ni": 1}]),
                          {"atoms": ["Ni"], "basename": "singlepoint"}, repo_root=tmp_path).iloc[0]
    assert row["HOMO_energy"] == pytest.approx(-7.2708, abs=1e-3)
    assert row["LUMO_energy"] == pytest.approx(-2.1534, abs=1e-3)
    assert row["SCF_energy"] == pytest.approx(-2253.330385978738)


def test_a_standard_xtb_that_ignored_the_flag_is_refused():
    """A plain xtb warns about --gxtb and runs GFN2; that must never pass as a g-xTB result."""
    with pytest.raises(gp.GxtbParseError, match="does not know --gxtb"):
        gp.check_is_gxtb("unknown option '--gxtb' ignored\n")
    gp.check_is_gxtb(_text("singlepoint"))                    # the real thing passes


# --------------------------------------------------------------------------- the condensed file

@pytest.fixture
def conf_dir(tmp_path: Path) -> Path:
    """A conformer folder with the three g-xTB steps and a geometry, as the runner leaves it."""
    d = tmp_path / "conf_1"
    (d / gc.RUN_DIRNAME).mkdir(parents=True)
    for step in ("singlepoint", "frequency", "oxidation"):
        shutil.copytree(REF / step, d / gc.RUN_DIRNAME / step)
    shutil.copyfile(REF / "input.xyz", d / "optimization.xyz")
    return d


def test_condensed_file_is_a_valid_schema_1_file(conf_dir: Path):
    res = gc.condense_folder(conf_dir)
    assert res["ok"], res
    path = conf_dir / "orca_condensed.json"
    assert check_condensed(path) == []          # including the g-xTB exemption from the NBO-block rule
    data = load_condensed(path)                 # raises on any schema_version but 1
    assert data["producer"]["engine"] == "g-xTB"
    assert data["producer"]["xtb_version"] == "6.7.1"
    # DescriPyTor only sniffs the first 8192 bytes for these two literals
    head = path.read_bytes()[:8192]
    assert b'"molecule"' in head and b'"atoms"' in head


def test_condensed_sections_hold_what_gxtb_can_and_cannot_produce(conf_dir: Path):
    data = gc.build_condensed(conf_dir)
    assert data["molecule"]["n_atoms"] == N_ATOMS
    assert data["molecule"]["charge"] == 0 and data["molecule"]["multiplicity"] == 1

    # the g-xTB charges fill the nbo and cm5 slots so the column names match a DFT library's
    assert len(data["charges"]["nbo"]) == N_ATOMS
    assert data["charges"]["cm5"] == data["charges"]["nbo"]
    assert data["charges"]["hirshfeld"] is None
    assert "no NPA, Hirshfeld or CM5" in data["charges"]["note"]
    assert data["states"]["singlepoint"]["charges"]["mulliken"] == data["charges"]["nbo"]
    for scheme in ("loewdin", "chelpg", "hirshfeld"):
        assert data["states"]["singlepoint"]["charges"][scheme] is None

    # what g-xTB does not compute
    assert data["properties"]["polarizability"] is None
    assert data["states"]["nmr"] is None
    assert data["states"]["singlepoint"]["nmr"] is None
    assert data["nbo"]["blocks"] == []

    # what it does
    assert data["properties"]["energies"]["final_single_point_energy_hartree_orca"] == pytest.approx(-115.72011098)
    assert data["properties"]["dipole"]["units"] == "debye"
    assert data["properties"]["dipole"]["total"] == pytest.approx(2.229924, abs=1e-4)  # xtb's own Debye value
    assert data["states"]["oxidation"]["multiplicity"] == 2
    assert data["states"]["reduction"] is None                 # the step was not run
    assert len(data["vibrations"]["modes"]) == 3 * N_ATOMS - 6
    assert data["states"]["singlepoint"]["thermochemistry"]["gibbs_eh"] == pytest.approx(-115.6934262)
    assert data["states"]["singlepoint"]["mo"]["energies_eh"]           # HOMO/LUMO come from here
    assert len(data["gxtb"]["fukui"]) == N_ATOMS and data["gxtb"]["wiberg"]


def test_condensing_needs_a_single_point(tmp_path: Path):
    d = tmp_path / "conf_1"
    (d / gc.RUN_DIRNAME).mkdir(parents=True)
    shutil.copyfile(REF / "input.xyz", d / "optimization.xyz")
    with pytest.raises(CondenseError, match="singlepoint/xtb.out missing"):
        gc.build_condensed(d)
    res = gc.condense_folder(d)                 # the folder-level call reports instead of raising
    assert res["ok"] is False and "singlepoint" in res["error"]


def test_a_sparse_gxtb_file_flows_through_the_descriptor_side(conf_dir: Path):
    """The point of the schema: the laptop side reads a g-xTB file with no changes.

    The sections g-xTB cannot fill must come out as NaN columns and notes, never as an exception.
    """
    import pandas as pd
    from desc_utils.extract.orca import extract_charges, extract_nbo, extract_redox

    gc.condense_folder(conf_dir)
    df_num = pd.DataFrame([{"path": str(conf_dir), "C": 1, "O": 2}])
    cfg = {"atoms": ["C", "O"], "basename": "singlepoint"}

    row = extract_charges(df_num, cfg, repo_root=conf_dir.parent).iloc[0]
    assert row["SCF_energy"] == pytest.approx(-115.72011098)
    assert pd.notna(row["HOMO_energy"]) and pd.notna(row["LUMO_energy"])
    assert pd.notna(row["mulliken_C"]), "the g-xTB charges land in the mulliken slot"
    # the schemes g-xTB has no analysis for are absent, not wrong
    assert not [c for c in row.index if c.startswith(("chelpG_", "hirshfeld_", "siso_"))]

    # the NBO family has nothing to read and says so, instead of blowing up the run
    nbo = extract_nbo(df_num, {**cfg, "homo_lumo_rank_range": None}, repo_root=conf_dir.parent)
    assert "no NBO block" in nbo.iloc[0]["nbo_extraction_error"]

    # the reduction state is missing, so the redox descriptors come out empty rather than raising
    redox = extract_redox(df_num, {**cfg, "prefixes": ["cm5"]}, repo_root=conf_dir.parent)
    assert len(redox) == 1


# --------------------------------------------------------------------------- commands and steps

def test_steps_follow_the_flags():
    assert steps_to_run({}) == ["singlepoint"]
    assert steps_to_run({"opt_geom": True, "freq_calc": True, "vertical_redox": True}) == [
        "optimization", "singlepoint", "frequency", "oxidation", "reduction"]
    assert steps_to_run({"vertical_redox": True}) == ["singlepoint", "oxidation", "reduction"]


def test_step_commands():
    opts = {"opt_level": "tight", "fukui": True, "accuracy": 0.5, "etemp": 500}
    base = ["xtb", "start.xyz", "--gxtb"]
    assert step_command("singlepoint", "xtb", opts, 0, 1) == base + [
        "--vfukui", "--chrg", "0", "--acc", "0.5", "--etemp", "500"]
    assert step_command("optimization", "xtb", opts, 0, 1)[:5] == base + ["--opt", "tight"]
    assert step_command("frequency", "xtb", opts, 0, 1)[3] == "--hess"
    # a closed shell oxidises to a doublet and reduces to a doublet; --uhf only appears when it is open
    assert "--uhf" not in step_command("singlepoint", "xtb", opts, 0, 1)
    ox = step_command("oxidation", "xtb", opts, 0, 1)
    assert ox[ox.index("--chrg") + 1] == "1" and ox[ox.index("--uhf") + 1] == "1"
    red = step_command("reduction", "xtb", opts, 0, 1)
    assert red[red.index("--chrg") + 1] == "-1" and red[red.index("--uhf") + 1] == "1"
    # a doublet oxidises back to a closed shell
    assert "--uhf" not in step_command("oxidation", "xtb", opts, 0, 2)
    assert uhf(None) == 0 and uhf(1) == 0 and uhf(3) == 2


# --------------------------------------------------------------------------- config

def _gxtb_cfg(tree, gxtb=None, **overrides):
    dft = {"engine": "gxtb"}
    if gxtb is not None:
        dft["gxtb"] = gxtb
    return _rewrite(tree, dft=dft, **overrides)


def test_gxtb_engine_config_loads_with_defaults(tree):
    cfg = Config.load(_gxtb_cfg(tree))
    assert cfg.dft_engine == "gxtb"
    assert cfg.gxtb["opt_geom"] is False and cfg.gxtb["vertical_redox"] is True
    assert cfg.gxtb["executable"] == "xtb"
    assert cfg.slurm["gxtb"]["ntasks"] == 8


def test_an_orca_library_keeps_its_hash(tree):
    """``engine`` is only written when it is not the default, so existing params.json files still match."""
    cfg = Config.load(tree["cfg"])
    assert cfg.dft_engine == "orca"
    assert "engine" not in cfg.dft and "gxtb" not in cfg.dft
    assert "engine" not in cfg.compared()["dft"]


def test_the_executable_is_not_part_of_the_hash(tree):
    a = Config.load(_gxtb_cfg(tree, gxtb={"executable": "xtb"})).params()["sha256"]
    b = Config.load(_gxtb_cfg(tree, gxtb={"executable": "/opt/gxtb/bin/xtb"})).params()["sha256"]
    c = Config.load(_gxtb_cfg(tree, gxtb={"executable": "xtb", "fukui": False})).params()["sha256"]
    assert a == b, "where the binary lives is a property of the machine, not of the results"
    assert a != c, "a step that changes the results must change the hash"


@pytest.mark.parametrize("gxtb, message", [
    ({"freq_calc": True}, "Hessian aborts on a geometry optimised at another level"),
    ({"opt_level": "nonsense"}, "opt_level must be one of"),
    ({"accuracy": -1}, "accuracy must be a positive number"),
    ({"opt_geom": "yes"}, "opt_geom must be true or false"),
    ({"solvent": "water"}, "gas phase only"),
    ({"nonsense": 1}, "unknown key"),
])
def test_gxtb_config_guards(tree, gxtb, message):
    with pytest.raises(ValueError, match=message):
        Config.load(_gxtb_cfg(tree, gxtb=gxtb))


def test_freq_calc_is_allowed_once_the_geometry_is_gxtb(tree):
    """Either the stage-6 optimisation or a stage-4 g-xTB refinement makes the Hessian meaningful."""
    cfg = Config.load(_gxtb_cfg(tree, gxtb={"freq_calc": True, "opt_geom": True}))
    assert cfg.gxtb["freq_calc"] is True
    cfg = Config.load(_gxtb_cfg(tree, gxtb={"freq_calc": True},
                                post_optimization={"refiner": "gxtb", "params": {"lot": "gxtb"}}))
    assert cfg.gxtb["freq_calc"] is True


def test_level_of_theory_belongs_to_the_other_engine(tree):
    with pytest.raises(ValueError, match="level_of_theory belongs to the ORCA engine"):
        _rewrite(tree, dft={"engine": "gxtb", "level_of_theory": {}})
        Config.load(tree["cfg"])


def test_unknown_engine_is_refused(tree):
    with pytest.raises(ValueError, match="dft.engine must be one of"):
        Config.load(_rewrite(tree, dft={"engine": "gaussian", "level_of_theory": {}}))


# --------------------------------------------------------------------------- the runner

FAKE_XTB = """#!/bin/sh
# replays a recorded g-xTB run: the step is whichever flag it was called with
echo "$@" > args.txt
step=singlepoint
for a in "$@"; do
  case "$a" in
    --hess) step=frequency ;;
    --opt) step=optimization ;;
  esac
done
case " $@ " in *" --chrg 1 "*) step=oxidation ;; esac
cat "{ref}/$step/xtb.out" 2>/dev/null || cat "{ref}/singlepoint/xtb.out"
cp "{ref}/singlepoint/charges" charges 2>/dev/null
cp "{ref}/singlepoint/wbo" wbo 2>/dev/null
if [ "$step" = frequency ]; then cp "{ref}/frequency/g98.out" g98.out; fi
if [ "$step" = optimization ]; then cp "{ref}/frequency/xtbopt.xyz" xtbopt.xyz; fi
exit 0
"""


@pytest.fixture
def fake_xtb(tmp_path: Path) -> str:
    """A stand-in for the g-xTB binary that replays the recorded output of the matching step."""
    exe = tmp_path / "fake_xtb"
    exe.write_text(FAKE_XTB.format(ref=REF))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(exe)


def _prepared(tree, fake_sbatch, gxtb=None):
    """Run submit + prepare with the g-xTB engine; returns (cfg, inchikey)."""
    from desc_utils.submit.cli import main
    _gxtb_cfg(tree, gxtb=gxtb, condense=True)
    main(["submit", str(tree["cfg"])])
    frozen = sorted((tree["lib"] / "_jobs").iterdir())[-1] / "config.yaml"
    cfg = Config.load(frozen)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO", submit_jobs=False)
    # the fake binary replays recorded methanol output, so the geometry has to be methanol too
    shutil.copyfile(REF / "input.xyz", tree["lib"] / key / "conf_1" / "input.xyz")
    return cfg, key


def test_prepare_writes_no_calcparam_and_submits_a_gxtb_array(tree, fake_sbatch):
    from desc_utils.submit.cli import main
    from desc_utils.submit import records
    _gxtb_cfg(tree, condense=True)
    main(["submit", str(tree["cfg"])])
    frozen = sorted((tree["lib"] / "_jobs").iterdir())[-1] / "config.yaml"
    cfg = Config.load(frozen)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO")

    conf = tree["lib"] / key / "conf_1"
    assert (conf / "input.xyz").is_file()
    assert not (conf / "calcParam.inp").exists(), "g-xTB takes its options from the config"
    meta = records.load_metadata(tree["lib"], key)
    assert meta["dft"]["engine"] == "g-xTB"
    assert meta["dft"]["compound_steps"] == ["singlepoint", "oxidation", "reduction"]
    assert meta["dft"]["gxtb"]["vertical_redox"] is True
    script = fake_sbatch[-1]["script"]
    assert script.name.startswith("gxtb_")
    text = script.read_text()
    assert "gxtb-task" in text and "--signal=B:TERM@600" in text and "--ntasks=8" in text


def test_dry_run_validates_a_gxtb_library_and_writes_nothing(tree, fake_sbatch, capsys):
    from desc_utils.submit.cli import main
    _gxtb_cfg(tree)
    assert main(["submit", str(tree["cfg"]), "--dry-run"]) == 0
    assert not tree["lib"].exists() or not (tree["lib"] / "params.json").exists()
    assert fake_sbatch == []


def test_gxtb_task_is_on_the_command_line():
    """The sbatch body calls this subcommand; it has to exist."""
    from desc_utils.submit.cli import main
    with pytest.raises(SystemExit):
        main(["gxtb-task", "--help"])


def test_run_records_the_same_keys_as_the_orca_engine(tree, fake_sbatch, fake_xtb):
    """status.py and cost.py read these keys and must not need to know which engine ran."""
    from desc_utils.submit import records
    from desc_utils.submit.cost import cost_table
    from desc_utils.submit.status import status_table

    cfg, key = _prepared(tree, fake_sbatch)
    for conf in sorted(p.name for p in (tree["lib"] / key).glob("conf_*")):
        shutil.copyfile(REF / "input.xyz", tree["lib"] / key / conf / "input.xyz")
        assert run_gxtb_conformer(cfg, key, conf, gxtb_cmd=fake_xtb) == "completed"

    rec = records.load_metadata(tree["lib"], key)["conformers"]["conf_1"]
    assert rec["status"] == "completed"
    assert rec["orca_terminated_normally"] is True
    assert rec["orca_run_time_s"] is not None and rec["cpu_hours"] >= 0
    assert set(rec["orca_steps_s"]) == {"singlepoint", "oxidation", "reduction"}
    assert rec["gxtb_failed_steps"] is None
    assert rec["condensed"]["ok"] is True

    conf = tree["lib"] / key / "conf_1"
    assert (conf / "optimization.xyz").is_file(), "the geometry descriptors read this file"
    assert check_condensed(conf / "orca_condensed.json") == []
    # each step ran in its own folder so their fixed output names cannot collide
    assert sorted(p.name for p in (conf / gc.RUN_DIRNAME).iterdir() if p.is_dir()) == [
        "oxidation", "reduction", "singlepoint"]

    assert status_table(cfg).set_index("inchikey").loc[key]["dft"] == "done"
    assert cost_table(cfg).set_index("inchikey").loc[key]["dft_cpu_h"] >= 0


def test_a_failed_optional_step_keeps_the_conformer(tree, fake_sbatch, tmp_path):
    """The Hessian is fragile on metal complexes; losing it must not throw away the single point."""
    from desc_utils.submit import records
    exe = tmp_path / "half_broken_xtb"
    exe.write_text(FAKE_XTB.format(ref=REF).replace(
        'if [ "$step" = frequency ]; then cp "{ref}/frequency/g98.out" g98.out; fi'.format(ref=REF),
        'if [ "$step" = frequency ]; then echo "Hessian element is not symmetric"; exit 1; fi'))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

    cfg, key = _prepared(tree, fake_sbatch, gxtb={"freq_calc": True, "opt_geom": True})
    assert run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=str(exe)) == "completed"
    rec = records.load_metadata(tree["lib"], key)["conformers"]["conf_1"]
    assert rec["gxtb_failed_steps"] == ["frequency"]
    data = load_condensed(tree["lib"] / key / "conf_1" / "orca_condensed.json")
    assert data["vibrations"]["modes"] == []          # the descriptors it would have fed are simply absent
    assert data["charges"]["nbo"], "everything else still came out"


def test_a_failed_single_point_fails_the_conformer(tree, fake_sbatch, tmp_path):
    from desc_utils.submit import records
    exe = tmp_path / "broken_xtb"
    exe.write_text("#!/bin/sh\necho 'SCF did not converge'\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    cfg, key = _prepared(tree, fake_sbatch)
    assert run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=str(exe)) == "failed"
    rec = records.load_metadata(tree["lib"], key)["conformers"]["conf_1"]
    assert rec["gxtb_failed_steps"] == ["singlepoint"] and "condensed" not in rec


def test_a_standard_xtb_is_caught_by_the_runner(tree, fake_sbatch, tmp_path):
    exe = tmp_path / "plain_xtb"
    exe.write_text("#!/bin/sh\necho \"unknown option '--gxtb' ignored\"\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    cfg, key = _prepared(tree, fake_sbatch)
    with pytest.raises(gp.GxtbParseError, match="does not know --gxtb"):
        run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=str(exe))


def test_a_missing_binary_says_what_to_set(tree, fake_sbatch):
    cfg, key = _prepared(tree, fake_sbatch)
    with pytest.raises(FileNotFoundError, match="dft.gxtb.executable"):
        run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd="definitely_not_a_binary_xyz")


def test_constrained_optimisation_fixes_the_atoms(tree, fake_sbatch, fake_xtb):
    """``dft.constrained`` must actually reach xtb, not just be recorded in the metadata."""
    from desc_utils.submit import records
    cfg, key = _prepared(tree, fake_sbatch, gxtb={"opt_geom": True, "vertical_redox": False})
    # the prepare step records which atoms stages 2 to 4 placed; stand in for a constrained library
    records.update_metadata(tree["lib"], key, {"dft": {"fixed_atoms": {"0": "C0", "2": "O2"}}})
    assert run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=fake_xtb) == "completed"

    opt = tree["lib"] / key / "conf_1" / gc.RUN_DIRNAME / "optimization"
    assert "--input xcontrol" in (opt / "args.txt").read_text()
    xcontrol = (opt / "xcontrol").read_text()
    assert "atoms: 1,3" in xcontrol, "xtb counts from 1, the record is 0-based"
    assert "engine=lbfgs" in xcontrol, "xtb's default ANCopt ignores $fix without a word"
    # a single point has no geometry to hold, so it gets no xcontrol
    assert not (tree["lib"] / key / "conf_1" / gc.RUN_DIRNAME / "singlepoint" / "xcontrol").exists()


def test_xtb_gets_its_thread_environment(tree, fake_sbatch, tmp_path):
    """xtb needs OPENBLAS_NUM_THREADS=1 or the conda-forge build floods the log with OpenBLAS warnings."""
    from desc_utils.submit.gxtb import thread_env
    assert thread_env(8) == {"OMP_NUM_THREADS": "8,1", "MKL_NUM_THREADS": "8",
                             "OMP_STACKSIZE": "4G", "OPENBLAS_NUM_THREADS": "1"}

    exe = tmp_path / "env_dump_xtb"
    exe.write_text("#!/bin/sh\nenv | grep -E 'OMP_|MKL_|OPENBLAS_' | sort > env.txt\n"
                   f"cat '{REF}/singlepoint/xtb.out'\ncp '{REF}/singlepoint/charges' charges\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    cfg, key = _prepared(tree, fake_sbatch, gxtb={"vertical_redox": False})
    run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=str(exe))

    dumped = (tree["lib"] / key / "conf_1" / gc.RUN_DIRNAME / "singlepoint" / "env.txt").read_text()
    assert "OPENBLAS_NUM_THREADS=1" in dumped
    assert "OMP_STACKSIZE=4G" in dumped


GXTB_BINARY = os.environ.get("GXTB_BINARY") or shutil.which("gxtb") or shutil.which("xtb")


@pytest.mark.slow
@pytest.mark.skipif(not GXTB_BINARY, reason="no g-xTB build of xtb (set GXTB_BINARY)")
def test_the_real_binary_produces_usable_descriptors(tree, fake_sbatch):
    """End to end with the actual g-xTB build: run the steps, condense, read the descriptors back.

    Skipped unless a binary is available. It must be the g-xTB build; a standard xtb raises instead of
    quietly handing back GFN2 numbers.
    """
    import pandas as pd
    from desc_utils.extract.orca import extract_charges

    cfg, key = _prepared(tree, fake_sbatch, gxtb={"vertical_redox": True})
    conf = tree["lib"] / key / "conf_1"
    shutil.copyfile(REF / "input.xyz", conf / "input.xyz")        # methanol, neutral closed shell
    assert run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=GXTB_BINARY) == "completed"
    assert check_condensed(conf / "orca_condensed.json") == []

    data = load_condensed(conf / "orca_condensed.json")
    assert data["producer"]["engine"] == "g-xTB"
    assert len(data["charges"]["nbo"]) == N_ATOMS
    assert data["states"]["oxidation"]["scf_energy_eh"] > data["states"]["singlepoint"]["scf_energy_eh"]
    assert data["states"]["reduction"] is not None

    row = extract_charges(pd.DataFrame([{"path": str(conf), "C": 1, "O": 2}]),
                          {"atoms": ["C", "O"], "basename": "singlepoint"}, repo_root=conf.parent).iloc[0]
    assert row["SCF_energy"] == pytest.approx(-115.72, abs=0.05)
    assert pd.notna(row["HOMO_energy"]) and pd.notna(row["LUMO_energy"])


def test_a_failed_optimisation_skips_the_hessian(tree, fake_sbatch, tmp_path):
    """The Hessian costs minutes and cannot succeed on a geometry the optimiser could not settle."""
    from desc_utils.submit import records
    exe = tmp_path / "no_opt_xtb"
    exe.write_text("#!/bin/sh\n"
                   'case " $@ " in *" --opt "*) echo "optimisation blew up"; exit 1 ;; esac\n'
                   f"cat '{REF}/singlepoint/xtb.out'\ncp '{REF}/singlepoint/charges' charges\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)

    cfg, key = _prepared(tree, fake_sbatch, gxtb={"opt_geom": True, "freq_calc": True, "vertical_redox": False})
    assert run_gxtb_conformer(cfg, key, "conf_1", gxtb_cmd=str(exe)) == "completed"
    rec = records.load_metadata(tree["lib"], key)["conformers"]["conf_1"]
    assert rec["gxtb_failed_steps"] == ["optimization", "frequency"]
    assert "frequency" not in rec["orca_steps_s"], "it was skipped, not run and billed"
    assert rec["orca_steps_s"]["singlepoint"] >= 0, "the properties were still computed"
