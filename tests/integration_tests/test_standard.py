"""The langchain-tests standard sandbox suite against a live boxd machine."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from packaging.version import Version

from boxd_langchain import BoxdSandbox
from tests.integration_tests._live import (
    SAFETY_DELETE_AFTER,
    machine_labels,
    machine_name,
    requires_boxd,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol

pytestmark = requires_boxd

# langchain-tests 1.1.9 still expects `write` to refuse an existing file, and
# glob to return paths relative to the search root. deepagents 0.7 overwrites,
# and since 0.7.7 glob returns absolute paths. A minimal local `BaseSandbox`
# fails the same tests on the same versions, so this is skew between the two
# packages, not boxd behaviour. Strict, so the marks fail loudly once the
# suite catches up.
_ABSOLUTE_GLOB = Version(version("deepagents")) >= Version("0.7.7")
_WRITE_SKEW = pytest.mark.xfail(
    strict=True,
    reason="langchain-tests 1.1.9 expects write to refuse; deepagents 0.7 overwrites",
)
_GLOB_SKEW = pytest.mark.xfail(
    _ABSOLUTE_GLOB,
    strict=True,
    reason="langchain-tests 1.1.9 expects relative glob paths; deepagents>=0.7.7 "
    "returns absolute ones",
)


class TestBoxdSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        with BoxdSandbox.create(
            name=machine_name("std"),
            labels=machine_labels(),
            delete_after=SAFETY_DELETE_AFTER,
        ) as backend:
            yield backend

    @_GLOB_SKEW
    def test_glob(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob(sandbox_backend, sandbox_test_root)

    @_GLOB_SKEW
    def test_glob_basic_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_basic_pattern(sandbox_backend, sandbox_test_root)

    @_GLOB_SKEW
    def test_glob_with_directories(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_directories(sandbox_backend, sandbox_test_root)

    @_GLOB_SKEW
    def test_glob_hidden_files_explicitly(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_hidden_files_explicitly(sandbox_backend, sandbox_test_root)

    @_GLOB_SKEW
    def test_glob_with_character_class(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_character_class(sandbox_backend, sandbox_test_root)

    @_GLOB_SKEW
    def test_glob_with_question_mark(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_question_mark(sandbox_backend, sandbox_test_root)

    @_WRITE_SKEW
    def test_write_existing_file_fails(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_write_existing_file_fails(sandbox_backend, sandbox_test_root)
