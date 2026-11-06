from importlib.metadata import version

import langchain_boxd
from langchain_boxd import BoxdSandbox

EXPECTED_ALL = ["BoxdSandbox", "__version__"]


def test_all_imports() -> None:
    assert sorted(langchain_boxd.__all__) == sorted(EXPECTED_ALL)
    assert BoxdSandbox is not None


def test_version_matches_package_metadata() -> None:
    assert langchain_boxd.__version__ == version("langchain-boxd")
