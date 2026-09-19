import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LNIH2_ROOT = REPO_ROOT / "DATA" / "LNiH2_library_001_to_download_2026_03_24"
TEST_MOL_ROOT = REPO_ROOT / "DATA" / "test_molecules"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def scratch(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("orca_desc")


def pytest_collection_modifyitems(config, items):
    """Skip slow regression tests when their inputs are not on disk."""
    have_data = LNIH2_ROOT.is_dir() and (REPO_ROOT / "DescriPyTor" / "M2_data_extractor").is_dir()
    skip = pytest.mark.skip(reason="needs DATA/ and the DescriPyTor submodule (git submodule update --init)")
    for item in items:
        if "slow" in item.keywords and not have_data:
            item.add_marker(skip)
