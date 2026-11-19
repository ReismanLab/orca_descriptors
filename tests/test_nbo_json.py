"""``dft.nbo_json`` and ``orcadesc nbo-json``: singlepoint.nbo -> singlepoint_nbo.json with a fake orca_2json."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from desc_utils.submit import nbo_json, records
from desc_utils.submit.cli import main
from desc_utils.submit.config import Config
from desc_utils.submit.tasks import prepare_molecule, run_orca_conformer
from tests.test_submit import _fake_orca, _frozen, _lib_cfg, _rewrite, fake_sbatch, tree  # noqa: F401 -- fixtures

LOT = _lib_cfg()["dft"]["level_of_theory"]


def _exe(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _fake_orca_2json(path: Path, fail: bool = False) -> Path:
    """Like orca_2json: refuses anything but a .gbw name, writes <stem>.json next to it (here: the bytes it read)."""
    if fail:
        return _exe(path, "echo 'boom' >&2\nexit 3\n")
    return _exe(path, '[[ "$1" == *.gbw ]] || { echo "not a gbw" >&2; exit 1; }\n'
                      'printf \'{"orbitals_of": "%s", "content": "%s"}\' "$1" "$(cat "$1")" > "${1%.gbw}.json"\n')


def _orca_writing_nbo(path: Path) -> Path:
    """The fake orca of test_submit, plus a singlepoint.nbo in the conformer folder."""
    _fake_orca(path)
    text = path.read_text().replace("#!/bin/bash\n", "#!/bin/bash\necho NBOS > singlepoint.nbo\n", 1)
    path.write_text(text)
    return path


def _conf(root: Path, mol: str, conf: str, energy: float | None, nbo: bool = True) -> Path:
    d = root / mol / conf
    d.mkdir(parents=True)
    if nbo:
        (d / "singlepoint.nbo").write_text(f"NBO {mol} {conf}")
    if energy is not None:
        (d / "orca_condensed.json").write_text(json.dumps(
            {"properties": {"energies": {"final_single_point_energy_hartree_orca": energy}}}))
    return d


# --------------------------------------------------------------------------- convert
def test_convert_copies_converts_and_removes_the_gbw(tmp_path):
    exe = _fake_orca_2json(tmp_path / "orca_2json")
    d = _conf(tmp_path, "M", "conf_1", -1.0)
    rec = nbo_json.convert(d, orca_2json=str(exe))
    assert rec["ok"] and rec["size_bytes"] > 0
    out = json.loads((d / "singlepoint_nbo.json").read_text())
    assert out == {"orbitals_of": "singlepoint_nbo.gbw", "content": "NBO M conf_1"}
    assert not (d / "singlepoint_nbo.gbw").exists() and (d / "singlepoint.nbo").exists()
    # there already: nothing run, unless overwrite
    assert nbo_json.convert(d, orca_2json="/nonexistent")["skipped"] == "exists"
    (d / "singlepoint.nbo").write_text("NEW")
    assert nbo_json.convert(d, orca_2json=str(exe), overwrite=True)["ok"]
    assert json.loads((d / "singlepoint_nbo.json").read_text())["content"] == "NEW"


def test_convert_failures_are_recorded_not_raised(tmp_path):
    d = _conf(tmp_path, "M", "conf_1", -1.0)
    rec = nbo_json.convert(d, orca_2json=str(_fake_orca_2json(tmp_path / "bad", fail=True)))
    assert not rec["ok"] and "exit 3" in rec["error"] and "boom" in rec["error"]
    assert not (d / "singlepoint_nbo.gbw").exists() and not (d / "singlepoint_nbo.json").exists()
    no_nbo = _conf(tmp_path, "M", "conf_2", -1.0, nbo=False)
    assert "no singlepoint.nbo" in nbo_json.convert(no_nbo, orca_2json="x")["error"]


def test_find_orca_2json_prefers_the_one_next_to_orca(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert nbo_json.find_orca_2json(None) is None
    orca = _exe(tmp_path / "orca", "exit 0\n")
    assert nbo_json.find_orca_2json(str(orca)) is None
    _fake_orca_2json(tmp_path / "orca_2json")
    assert nbo_json.find_orca_2json(str(orca)) == str((tmp_path / "orca_2json").resolve())


# --------------------------------------------------------------------------- the pipeline step
def test_nbo_json_config_is_off_by_default_orca_only_and_not_hashed(tree):
    plain = Config.load(_rewrite(tree))
    assert plain.nbo_json is False and "nbo_json" not in plain.dft
    on = Config.load(_rewrite(tree, dft={"level_of_theory": LOT, "nbo_json": True}))
    assert on.nbo_json is True and on.compared() == plain.compared()
    _rewrite(tree, dft={"level_of_theory": LOT, "nbo_json": "yes"})
    with pytest.raises(ValueError, match="nbo_json must be true or false"):
        Config.load(tree["cfg"])
    _rewrite(tree, dft={"engine": "gxtb", "nbo_json": True})
    with pytest.raises(ValueError, match="for dft.engine: orca"):
        Config.load(tree["cfg"])


def test_orca_task_converts_with_the_orca_2json_next_to_orca(tree, fake_sbatch, tmp_path):
    _rewrite(tree, dft={"level_of_theory": LOT, "nbo_json": True})
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO")
    bindir = tmp_path / "orca_bin"
    bindir.mkdir()
    orca = _orca_writing_nbo(bindir / "orca")
    _fake_orca_2json(bindir / "orca_2json")
    assert run_orca_conformer(cfg, key, "conf_1", orca_cmd=str(orca)) == "completed"
    conf = tree["lib"] / key / "conf_1"
    assert json.loads((conf / "singlepoint_nbo.json").read_text())["content"] == "NBOS"
    assert not (conf / "singlepoint_nbo.gbw").exists()
    rec = records.load_metadata(tree["lib"], key)["conformers"]["conf_1"]
    assert rec["status"] == "completed" and rec["nbo_json"]["ok"] is True


def test_orca_task_without_the_option_writes_no_json(tree, fake_sbatch, tmp_path):
    cfg = _frozen(tree, fake_sbatch)
    key = tree["keys"]["CCO"]
    prepare_molecule(cfg, key, "CCO")
    bindir = tmp_path / "orca_bin"
    bindir.mkdir()
    orca = _orca_writing_nbo(bindir / "orca")
    _fake_orca_2json(bindir / "orca_2json")
    assert run_orca_conformer(cfg, key, "conf_1", orca_cmd=str(orca)) == "completed"
    assert not (tree["lib"] / key / "conf_1" / "singlepoint_nbo.json").exists()
    assert "nbo_json" not in records.load_metadata(tree["lib"], key)["conformers"]["conf_1"]


# --------------------------------------------------------------------------- backfill
@pytest.fixture
def library(tmp_path) -> Path:
    lib = tmp_path / "LIB"
    _conf(lib, "AAA", "conf_1", -10.0)
    _conf(lib, "AAA", "conf_2", -10.5)
    _conf(lib, "AAA", "conf_10", -11.0, nbo=False)    # lowest, but no NBO file: not a candidate
    _conf(lib, "AAA", "conf_3", None)                 # NBO file, no energy
    _conf(lib, "BBB", "conf_1", -3.0)
    (lib / "BBB" / "metadata.json").write_text(json.dumps({"conformers": {"conf_1": {"status": "completed"}}}))
    (lib / "_jobs").mkdir()
    return lib


def test_lowest_energy_targets(library):
    assert [c.name for c in nbo_json.conformer_dirs(library / "AAA")] == ["conf_1", "conf_2", "conf_3", "conf_10"]
    assert [m.name for m in nbo_json.molecule_dirs(library)] == ["AAA", "BBB"]
    rel = lambda paths: [str(p.relative_to(library)) for p in paths]  # noqa: E731
    assert rel(nbo_json.targets(library)) == ["AAA/conf_1", "AAA/conf_2", "AAA/conf_3", "BBB/conf_1"]
    assert rel(nbo_json.targets(library, lowest_only=True)) == ["AAA/conf_2", "BBB/conf_1"]
    assert rel(nbo_json.targets(library, inchikeys=["BBB"])) == ["BBB/conf_1"]


def test_backfill_cli(library, tmp_path, monkeypatch, capsys):
    exe = _fake_orca_2json(tmp_path / "orca_2json")
    assert main(["nbo-json", str(library), "--lowest", "--dry-run", "--orca-2json", str(exe)]) == 0
    assert "AAA/conf_2: would convert" in capsys.readouterr().out
    assert not list(library.rglob("singlepoint_nbo.json"))

    assert main(["nbo-json", str(library), "--lowest", "--orca-2json", str(exe)]) == 0
    assert sorted(str(p.relative_to(library)) for p in library.rglob("singlepoint_nbo.json")) == \
           ["AAA/conf_2/singlepoint_nbo.json", "BBB/conf_1/singlepoint_nbo.json"]
    meta = records.load_metadata(library, "BBB")["conformers"]["conf_1"]
    assert meta["status"] == "completed" and meta["nbo_json"]["ok"] is True
    assert not (library / "AAA" / "metadata.json").exists()           # no record made where there was none

    capsys.readouterr()
    assert main(["nbo-json", str(library), "--orca-2json", str(exe)]) == 0     # the rest; the two done are kept
    out = capsys.readouterr().out
    assert "4 conformer(s): converted 2, exists 2" in out

    monkeypatch.setenv("PATH", "")
    assert main(["nbo-json", str(library), "--overwrite"]) == 2               # no orca_2json anywhere
    assert "module load orca/6.1.0" in capsys.readouterr().err
    assert main(["nbo-json", str(library / "nope")]) == 2


def test_backfill_from_a_config_and_failures_exit_1(tree, library, tmp_path):
    _rewrite(tree, library_dir=str(library))
    bad = _fake_orca_2json(tmp_path / "bad", fail=True)
    assert main(["nbo-json", str(tree["cfg"]), "--inchikey", "BBB", "--orca-2json", str(bad)]) == 1
    assert records.load_metadata(library, "BBB")["conformers"]["conf_1"]["nbo_json"]["ok"] is False
