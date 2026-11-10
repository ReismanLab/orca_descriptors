"""Numbering tables with failed rows must not crash extraction; unsanitizable SDF files are still numbered."""
from pathlib import Path

import pandas as pd
import pytest

from desc_utils.numbering import usable_rows
from desc_utils.numbering.io import load_mol
from desc_utils.spec import load_spec

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec():
    return load_spec(REPO / "specs" / "aryl_alkynes.yaml")


def _table():
    return pd.DataFrame({
        "path": ["DATA/A/M1/conf_1/", "DATA/A/M1/conf_2/", "DATA/A/M2/conf_1/", "DATA/A/M3/conf_1/"],
        "inchikey": ["M1", "M1", "M2", "M3"],
        "H1": pd.array([18, 18, pd.NA, 7], dtype="Int64"),
        "C1": pd.array([1, 1, pd.NA, 1], dtype="Int64"),
        "C2": pd.array([2, 2, pd.NA, 2], dtype="Int64"),
        "Cipso": pd.array([3, 3, pd.NA, 3], dtype="Int64"),
        "numbering_status": ["ok", "ok", "failed", "inconsistent"],
        "numbering_notes": ["", "", "stage 'aryl_alkyne': no SMARTS match", "conformers disagree on ['H1']"],
    })


def test_usable_rows_drops_failed_and_casts_to_int(spec):
    out = usable_rows(_table(), spec, quiet=True)
    assert list(out["path"]) == ["DATA/A/M1/conf_1/", "DATA/A/M1/conf_2/"]
    assert all(out[l].dtype.kind == "i" for l in spec.labels)
    assert int(out.loc[0, "H1"]) == 18
    # without the status requirement the inconsistent row (which has indices) is kept, the <NA> one is not
    loose = usable_rows(_table(), spec, require_ok=False, quiet=True)
    assert list(loose["inchikey"]) == ["M1", "M1", "M3"]


def test_usable_rows_needs_label_columns(spec):
    with pytest.raises(KeyError):
        usable_rows(_table().drop(columns=["Cipso"]), spec)


def test_run_extraction_skips_failed_rows_without_engines(spec, monkeypatch):
    """The engines only see clean rows; every conformer keeps its metadata row."""
    from desc_utils import extract as ex

    seen = {}

    def fake_run_orca(df_num, spec_, repo_root, families, overwrite_xyz_json, progress):
        seen["rows"] = list(df_num["path"])
        return pd.DataFrame({"path": df_num["path"], "Bond_C1_H1": 1.06})

    monkeypatch.setattr("desc_utils.extract.orca.run_orca", fake_run_orca)
    monkeypatch.setattr(ex, "job_status_columns",
                        lambda paths, s, r: pd.DataFrame({"path": list(paths), "orca_job_status": "completed",
                                                          "n_conformer_folders": 1}))
    out = ex.run_extraction(spec, _table(), REPO, families=["orca.xyz"], progress=False)
    assert len(seen["rows"]) == 2
    assert len(out) == 4
    assert out.loc[out["inchikey"] == "M2", "Bond_C1_H1"].isna().all()
    assert out.loc[out["inchikey"] == "M1", "Bond_C1_H1"].notna().all()
    assert list(out["numbering_status"]) == ["ok", "ok", "failed", "inconsistent"]


# An sdf Open Babel could write from a DFT geometry: the internal alkyne carbon drawn with a triple
# bond and two hydrogens (five-valent), which RDKit's sanitization rejects for the whole molecule.
_OVERVALENT_SDF = """overvalent
 OpenBabel

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6000    0.9000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6000   -0.9000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.2600    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$
"""


def test_load_mol_falls_back_to_connectivity(tmp_path):
    from rdkit import Chem

    sdf = tmp_path / "optimization.sdf"
    sdf.write_text(_OVERVALENT_SDF)
    notes = []
    mol = load_mol(sdf, notes=notes)
    assert mol.GetNumAtoms() == 5
    assert notes and "sanitization failed" in notes[0]
    # the connectivity SMARTS of the specs still work: the terminal alkyne carbon is two-coordinate
    assert mol.GetSubstructMatches(Chem.MolFromSmarts("[H][#6X2]~[#6X3]")) == ((4, 1, 0),)
    assert Chem.MolToSmiles(mol)  # the smiles column can still be filled


def test_load_mol_clean_sdf_has_no_note(tmp_path):
    from rdkit import Chem

    m = Chem.AddHs(Chem.MolFromSmiles("C#Cc1ccccc1"))
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(m, randomSeed=1)
    sdf = tmp_path / "optimization.sdf"
    Chem.MolToMolFile(m, str(sdf))
    notes = []
    mol = load_mol(sdf, notes=notes)
    assert notes == []
    assert mol.GetSubstructMatches(Chem.MolFromSmarts("[H][#6]#[#6]-[#6;a]"))


def test_numbering_topology_flags_a_rearranged_geometry(spec, tmp_path):
    """numbering.topology: conf_2 has lost its terminal alkyne H (C-H 3 A): topology_changed, so usable_rows
    drops it; conf_1 stays ok and both carry the topology_changes column."""
    import copy

    import numpy as np
    from rdkit import Chem

    from bceg.chem import embed, mol_from_smiles
    from bceg.topology import check_params
    from bceg.xyz import Conformer, write_xyz
    from desc_utils.numbering import build_numbering_table

    mol = mol_from_smiles("C#Cc1ccccc1")
    coords = embed(mol, 1)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    mol_dir = tmp_path / "DATA" / "PHENYLACETYLENE"
    mol_dir.mkdir(parents=True)
    Chem.MolToMolFile(mol, str(mol_dir / "structure.mol"))
    h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetDegree() == 2)
    c = mol.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
    broken = coords.copy()
    broken[h] = coords[c] + 3.0 * (coords[h] - coords[c]) / np.linalg.norm(coords[h] - coords[c])
    for name, xyz in (("conf_1", coords), ("conf_2", broken)):
        (mol_dir / name).mkdir()
        write_xyz(mol_dir / name / "optimization.xyz", elements, [Conformer(xyz, None)])

    topo_spec = copy.deepcopy(spec)
    topo_spec.numbering["topology"] = check_params(True, "numbering.topology")
    topo_spec.numbering["consistency"] = "none"
    df = build_numbering_table(topo_spec, repo_root=tmp_path, data_root=tmp_path / "DATA", progress=False)
    status = dict(zip(df["path"].str.extract(r"(conf_\d)")[0], df["numbering_status"]))
    assert status == {"conf_1": "ok", "conf_2": "topology_changed"}
    assert "bond broken" in df.loc[df["numbering_status"] == "topology_changed", "numbering_notes"].iloc[0]
    assert "topology_changes" in df.columns
    assert len(usable_rows(df, topo_spec, quiet=True)) == 1


# --------------------------------------------------------------------------- symmetric matches
class _SymmetricMol:
    """A molecule whose SMARTS match is symmetric, standing in for one conformer of a C2 ligand.

    `uniquify=True` is RDKit's behaviour of keeping ONE representative per matched atom SET: for a
    symmetric core both orientations have the same set, so one of them is discarded, and which one
    survives depends on RDKit's internal ordering - which differs between conformers of the same
    molecule. `representative` is that arbitrary choice.
    """

    ORIENTATIONS = ((3, 2, 1, 0), (0, 1, 2, 3))

    def __init__(self, representative):
        self.representative = representative

    def GetSubstructMatches(self, patt, uniquify=True, **kw):       # noqa: N802 (RDKit's name)
        return (self.representative,) if uniquify else self.ORIENTATIONS


def _backend(match_selection):
    from desc_utils.numbering.backends import SmartsBackend
    return SmartsBackend(
        [{"name": "core", "smarts": ["[#7]~[#6]~[#6]~[#7]"],
          "roles": {"N1": 0, "C1": 1, "C2": 2, "N2": 3}, "required": True}],
        match_selection=match_selection)


def test_lowest_indices_is_the_same_labelling_whichever_orientation_rdkit_returns():
    """Otherwise a C2-symmetric ligand gets different N1/N2 labels in different conformers, the
    consistency check calls the molecule `inconsistent`, and every one of its conformers is dropped.
    On the nih2 set that was 22 of 114 ligands and 833 of 4,539 conformers of one library - lost to
    a labelling ambiguity rather than to anything chemical.
    """
    backend = _backend("lowest_indices")
    labellings = {tuple(backend.number(_SymmetricMol(rep), None)[l]
                        for l in ("N1", "C1", "C2", "N2"))
                  for rep in _SymmetricMol.ORIENTATIONS}
    assert labellings == {(0, 1, 2, 3)}, "lowest_indices must not depend on RDKit's representative"


def test_first_match_selection_does_depend_on_it_which_is_why_it_is_not_the_default_here():
    """`first` is unchanged and still means "RDKit's first match"; the nih2 specs ask for
    `lowest_indices` precisely because this is what `first` does on a symmetric core."""
    backend = _backend("first")
    labellings = {tuple(backend.number(_SymmetricMol(rep), None)[l]
                        for l in ("N1", "C1", "C2", "N2"))
                  for rep in _SymmetricMol.ORIENTATIONS}
    assert len(labellings) == 2
