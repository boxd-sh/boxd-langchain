from importlib.metadata import version

import boxd_langchain
from boxd_langchain import BoxdSandbox

EXPECTED_ALL = ["BoxdSandbox", "__version__"]


def test_all_imports() -> None:
    assert sorted(boxd_langchain.__all__) == sorted(EXPECTED_ALL)
    assert BoxdSandbox is not None


def test_version_matches_package_metadata() -> None:
    assert boxd_langchain.__version__ == version("boxd-langchain")
