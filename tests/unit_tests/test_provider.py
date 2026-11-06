from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

import pytest
from boxd.errors import AuthenticationError, NotFoundError
from deepagents_code.integrations.sandbox_provider import SandboxNotFoundError

from langchain_boxd import BoxdSandbox
from langchain_boxd.provider import DEFAULT_COMMAND_TIMEOUT, BoxdProvider
from tests.unit_tests.fakes import FakeBoxd, FakeMachine

if TYPE_CHECKING:
    from collections.abc import Callable


def _resolver(values: dict[str, str]) -> Callable[[str], str | None]:
    return values.get


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBoxd:
    """Patch the provider's `Boxd` constructor and record its arguments."""
    client = FakeBoxd()
    client.constructed_with = []  # type: ignore[attr-defined]

    def make(**kwargs: object) -> FakeBoxd:
        client.constructed_with.append(kwargs)  # type: ignore[attr-defined]
        return client

    monkeypatch.setattr("langchain_boxd.provider.Boxd", make)
    return client


def _provider(env: dict[str, str] | None = None) -> BoxdProvider:
    return BoxdProvider(resolve_env_var=_resolver({"BOXD_API_KEY": "k", **(env or {})}))


def test_metadata_does_not_need_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOXD_API_KEY", raising=False)
    metadata = BoxdProvider().metadata
    assert metadata.name == "boxd"
    assert metadata.working_dir == "/home/boxd"
    assert metadata.supports_sandbox_id is True
    assert metadata.supports_snapshot_name is True
    assert metadata.backend_module == "langchain_boxd"
    assert metadata.install is not None
    assert metadata.install.name == "langchain-boxd"
    assert metadata.install.kind == "package"


def test_entry_point_loads_provider() -> None:
    entries = entry_points(group="deepagents_code.sandbox_providers")
    entry = next(item for item in entries if item.name == "boxd")
    assert entry.value == "langchain_boxd.provider:BoxdProvider"
    assert entry.load() is BoxdProvider


def test_creates_machine_with_defaults(fake: FakeBoxd) -> None:
    backend = _provider().get_or_create()
    assert isinstance(backend, BoxdSandbox)
    assert backend.id == "vm-1"
    assert fake.constructed_with == [{"api_key": "k"}]  # type: ignore[attr-defined]
    [call] = fake.machines.create_calls
    assert call["image"] is None
    assert call["from_snapshot"] is None
    assert call["delete_after"] is None
    assert backend._default_timeout == DEFAULT_COMMAND_TIMEOUT
    # The provider's client belongs to the backend, which closes it.
    backend.close()
    assert fake.closed is True


def test_creates_machine_with_options(fake: FakeBoxd) -> None:
    backend = _provider().get_or_create(
        name="t-x",
        labels={"team": "ai"},
        image="ubuntu:24.04",
        delete_after="600",
        workdir="/srv",
        command_timeout=42,
    )
    [call] = fake.machines.create_calls
    assert call["name"] == "t-x"
    assert call["labels"] == {"team": "ai"}
    assert call["image"] == "ubuntu:24.04"
    assert call["delete_after"] == 600
    assert backend._workdir == "/srv"
    assert backend._default_timeout == 42


def test_reads_image_and_delete_after_from_environment(fake: FakeBoxd) -> None:
    _provider({"BOXD_IMAGE": "img:1", "BOXD_DELETE_AFTER": "3600"}).get_or_create()
    [call] = fake.machines.create_calls
    assert call["image"] == "img:1"
    assert call["delete_after"] == 3600


def test_snapshot_restores_instead_of_image(fake: FakeBoxd) -> None:
    _provider({"BOXD_IMAGE": "img:1"}).get_or_create(snapshot="golden")
    [call] = fake.machines.create_calls
    assert call["from_snapshot"] == "golden"
    assert call["image"] is None


def test_snapshot_and_sandbox_id_conflict() -> None:
    with pytest.raises(ValueError, match="snapshot cannot be combined"):
        _provider().get_or_create(sandbox_id="vm", snapshot="golden")


@pytest.mark.parametrize(
    ("status", "call"),
    [
        ("running", None),
        ("stopped", "start"),
        ("suspended", "resume"),
        ("hibernated", "wake"),
    ],
)
def test_attaches_to_existing_machine_and_wakes_it(
    fake: FakeBoxd, status: str, call: str | None
) -> None:
    fake.machines.machines["vm-9"] = FakeMachine("vm-9", "my-box", status)
    backend = _provider().get_or_create(sandbox_id="my-box")
    assert backend.id == "vm-9"
    expected = [] if call is None else [(call, "vm-9")]
    assert fake.machines.lifecycle == [*expected, ("wait_until_ready", "vm-9")]
    assert fake.machines.create_calls == []


def test_missing_machine_maps_to_sandbox_not_found(fake: FakeBoxd) -> None:
    fake.machines.get_exc = NotFoundError("no machine")
    with pytest.raises(SandboxNotFoundError) as exc_info:
        _provider().get_or_create(sandbox_id="vm-missing")
    assert exc_info.value.args == ("vm-missing",)
    assert fake.closed is True


def test_rejected_key_gives_clear_error(fake: FakeBoxd) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        msg = "refused"
        raise AuthenticationError(msg, reason="invalid_api_key")

    fake.machines.create = refuse  # type: ignore[method-assign]
    with pytest.raises(ValueError, match=r"invalid_api_key.*BOXD_API_KEY"):
        _provider().get_or_create()
    assert fake.closed is True


def test_without_key_falls_back_to_sdk_discovery(fake: FakeBoxd) -> None:
    BoxdProvider(resolve_env_var=_resolver({})).get_or_create()
    assert fake.constructed_with == [{}]  # type: ignore[attr-defined]


def test_prefixed_env_var_wins(fake: FakeBoxd, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOXD_API_KEY", "plain")
    monkeypatch.setenv("DEEPAGENTS_CODE_BOXD_API_KEY", "prefixed")
    BoxdProvider().get_or_create()
    assert fake.constructed_with == [{"api_key": "prefixed"}]  # type: ignore[attr-defined]


def test_delete_is_idempotent(fake: FakeBoxd) -> None:
    provider = _provider()
    provider.delete(sandbox_id="vm-1")
    assert fake.machines.deleted == ["vm-1"]
    fake.machines.delete_exc = NotFoundError("gone")
    provider.delete(sandbox_id="vm-1")
    assert fake.closed is True


def test_rejects_unsupported_options() -> None:
    with pytest.raises(TypeError, match="Received unsupported arguments: metadata"):
        _provider().get_or_create(metadata={"purpose": "test"})
    with pytest.raises(TypeError, match="Received unsupported arguments: force"):
        _provider().delete(sandbox_id="x", force=True)


@pytest.mark.parametrize("value", ["abc", "-1"])
def test_rejects_bad_command_timeout(value: str) -> None:
    with pytest.raises(ValueError, match="command_timeout"):
        _provider().get_or_create(command_timeout=value)
