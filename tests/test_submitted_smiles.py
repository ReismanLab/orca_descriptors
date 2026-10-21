"""numbering.smiles.source: the SMILES columns come from the submitted SMILES, not from the geometry."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from desc_utils.numbering import apply_submitted_smiles
from desc_utils.spec import load_spec, validate_spec

REPO = Path(__file__).resolve().parents[1]
PYOX = "CC(C)(C)[C@@H](CO1)N=C1C2=NC=C(C(F)(F)F)C=C2"   # its InChIKey names the folder
PYOX_IK = "AATATIAROAAKIN-SNVBAGLBSA-N"
# what Open Babel bond perception of the L-NiH2 xyz gave once the Ni was stripped: a dihydro tautomer
GEOMETRY_SMILES = "CC(C)(C)[C@H]1COC(=C2C=CC(C(F)(F)F)=CN2)N1"


def _spec(tmp_path, **smiles):
    raw = yaml.safe_load((REPO / "specs" / "lnih2_pyox.yaml").read_text())
    raw["data"]["root"] = str(tmp_path)
    raw["numbering"]["smiles"] = {"mode": "strip_atoms", "strip": ["Ni"], **smiles}
    return validate_spec(raw)


def _table(*iks):
    return pd.DataFrame({"inchikey": list(iks), "smiles": GEOMETRY_SMILES, "smiles_noH": GEOMETRY_SMILES,
                         "numbering_notes": ""})


def test_registry_replaces_the_geometry_smiles(tmp_path):
    (tmp_path / "molecule_registry.json").write_text(json.dumps({PYOX_IK: {"smiles": PYOX}}))
    out = apply_submitted_smiles(_table(PYOX_IK, PYOX_IK), _spec(tmp_path, source="registry"), REPO)
    assert (out["smiles"] == PYOX).all()
    assert (out["smiles_noH"] == "CC(C)(C)[C@H]1COC(c2ccc(C(F)(F)F)cn2)=N1").all()   # canonical, the pyridine kept


def test_file_source_keys_smiles_by_inchikey(tmp_path):
    (tmp_path / "ligands.txt").write_text(f"# comment\n{PYOX} tBu-CF3-pyox\nc1ccncc1\n")
    out = apply_submitted_smiles(_table(PYOX_IK), _spec(tmp_path, source="file", file=str(tmp_path / "ligands.txt")), REPO)
    assert out.loc[0, "smiles"] == PYOX


def test_missing_molecule_gets_no_smiles_and_a_note(tmp_path):
    (tmp_path / "molecule_registry.json").write_text(json.dumps({PYOX_IK: {"smiles": PYOX}}))
    out = apply_submitted_smiles(_table(PYOX_IK, "XXXXXXXXXXXXXX-UHFFFAOYSA-N"), _spec(tmp_path, source="registry"), REPO)
    assert pd.isna(out.loc[1, "smiles"]) and pd.isna(out.loc[1, "smiles_noH"])
    assert "not in numbering.smiles.source registry" in out.loc[1, "numbering_notes"]
    assert out.loc[0, "numbering_notes"] == ""


def test_geometry_source_is_the_default_and_leaves_the_table_alone(tmp_path):
    spec = _spec(tmp_path)
    assert spec.numbering["smiles"]["source"] == "geometry"
    df = _table(PYOX_IK)
    assert apply_submitted_smiles(df, spec, REPO) is df


def test_missing_registry_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="molecule_registry.json"):
        apply_submitted_smiles(_table(PYOX_IK), _spec(tmp_path, source="registry"), REPO)


@pytest.mark.parametrize("smiles, data, message", [
    ({"source": "rdkit"}, {}, "must be 'geometry', 'registry' or 'file'"),
    ({"source": "file"}, {}, "numbering.smiles.file"),
    ({"source": "registry"}, {"molecule_id_column": "name"}, "needs data.molecule_id_column: inchikey"),
])
def test_spec_validation(tmp_path, smiles, data, message):
    raw = yaml.safe_load((REPO / "specs" / "lnih2_pyox.yaml").read_text())
    raw["numbering"]["smiles"] = {"mode": "strip_atoms", "strip": ["Ni"], **smiles}
    raw["data"].update(data)
    with pytest.raises(ValueError, match=message):
        validate_spec(raw)


def test_library_002_spec_reads_the_registry():
    assert load_spec(REPO / "specs" / "lnih2_library_002.yaml").numbering["smiles"]["source"] == "registry"
