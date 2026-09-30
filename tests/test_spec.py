import copy

import pytest

from desc_utils.spec import SpecError, load_spec, resolve_terms, atom_map, validate_spec

MINIMAL = {
    "name": "mini",
    "data": {"root": "DATA/x"},
    "atoms": {"A": "C", "B": "N", "M": None},
    "numbering": {"backend": "first_n"},
    "descriptors": {"descripytor": {"charges": ["A", "B"], "bond_length": [["A", "B"]]}},
}


def test_shipped_specs_validate(repo_root):
    for name in ("lnih2_pyox.yaml", "test_molecules_at1_at2.yaml"):
        spec = load_spec(repo_root / "specs" / name)
        assert spec.labels
        assert spec.descripytor is not None and spec.orca is not None


def test_minimal_spec_fills_defaults():
    spec = validate_spec(MINIMAL)
    assert spec.labels == ["A", "B", "M"]
    assert spec.numbering["index_base"] == 1
    assert spec.aggregate["temperature_K"] == 298.15
    assert spec.descripytor["stretch_threshold"] == 1400
    assert spec.descripytor["sterimol"] == []          # unset families default to empty
    assert spec.output["numbering_csv"] == "{date}_{name}_atom_numbering_results.csv"


def test_all_problems_reported_at_once():
    raw = copy.deepcopy(MINIMAL)
    raw["descriptors"]["descripytor"]["sterimol"] = [["A", "B", "M"], ["A", "Zz"]]
    raw["descriptors"]["descripytor"]["charges"] = ["A", "A"]
    raw["descriptors"]["descripytor"]["bogus"] = 1
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    text = str(exc.value)
    assert "expected 2 labels, got 3" in text
    assert "unknown atom label(s) ['Zz']" in text
    assert "duplicate term ['A']" in text
    assert "unknown key 'bogus'" in text


def test_smarts_stage_validation():
    raw = copy.deepcopy(MINIMAL)
    raw["numbering"] = {
        "backend": "smarts",
        "smarts": {"stages": [{"name": "s", "smarts": "[#6]~[#7]", "roles": {"A": 0, "B": 5, "Q": 1}}]},
    }
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    text = str(exc.value)
    assert "unknown label 'Q'" in text
    assert "exceed the 2 atoms" in text
    assert "labels ['M'] are never assigned" in text


def test_unquoted_smarts_gives_hint(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\ndata: {root: d}\natoms: {A: C}\nnumbering:\n  backend: smarts\n  smarts:\n    stages:\n      - smarts: [#6]~[#7]\n        roles: {A: 0}\n")
    with pytest.raises(SpecError) as exc:
        load_spec(p)
    assert "quoted" in str(exc.value)


def test_resolve_terms_and_atom_map():
    row = {"A": 3, "B": 7, "M": 1}
    assert resolve_terms(["A", ["A", "B"], ["B", "A", "M"]], row) == [(3,), (3, 7), (7, 3, 1)]
    assert atom_map(row, ["M", "A"]) == {"M": 1, "A": 3}


def test_output_section_is_validated():
    import pytest
    from desc_utils.spec import SpecError, validate_spec
    base = {"name": "t", "data": {"root": "DATA/x"}, "atoms": {"A": None}, "numbering": {"backend": "first_n"},
            "descriptors": {"descripytor": {"charges": ["A"]}}}
    with pytest.raises(SpecError, match="unknown key 'numbering_cvs'"):
        validate_spec({**base, "output": {"numbering_cvs": "x.csv"}})
    with pytest.raises(SpecError, match="unknown placeholder"):
        validate_spec({**base, "output": {"aggregated_csv": "{data}_x.csv"}})
    spec = validate_spec({**base, "output": {"aggregated_csv": "{date}_{name}_final.csv"}})
    assert spec.output["aggregated_csv"] == "{date}_{name}_final.csv" and spec.output["check_csv"].startswith("{date}")
    with pytest.raises(SpecError, match="nbo_block_map"):
        validate_spec({**base, "descriptors": {"orca": {"redox": {"atoms": ["A"], "nbo_block_map": {"singlepoint": 0}}}}})


def test_duplicate_top_level_key_is_refused(tmp_path):
    """A second `numbering:` block must not silently replace the first (YAML keeps the last one)."""
    text = (
        "name: dup\ndata: {root: DATA/x}\natoms: {A: C}\n"
        "numbering: {backend: first_n, energy: {source: xyz_comment, column: uma_energy_hartree}}\n"
        "descriptors: {orca: {xyz: {atoms: [A]}}}\n"
        "numbering: {backend: first_n}\n"
    )
    (tmp_path / "dup.yaml").write_text(text)
    with pytest.raises(SpecError) as exc:
        load_spec(tmp_path / "dup.yaml")
    assert "key 'numbering' appears twice" in str(exc.value) and "line 6" in str(exc.value) and "line 4" in str(exc.value)
    # nested duplicates are caught too
    (tmp_path / "dup2.yaml").write_text(text.replace("numbering: {backend: first_n}\n", "")
                                        .replace("descriptors:", "aggregate: {stats: [min], stats: [max]}\ndescriptors:"))
    with pytest.raises(SpecError, match="key 'stats' appears twice"):
        load_spec(tmp_path / "dup2.yaml")
