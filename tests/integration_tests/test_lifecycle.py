"""Live tests of everything around the protocol: create, attach, wake, delete."""

from __future__ import annotations

import contextlib
import os
import time

import pytest
from boxd import Boxd
from boxd.errors import AuthenticationError, NotFoundError

from langchain_boxd import BoxdSandbox
from langchain_boxd.provider import BoxdProvider
from langchain_boxd.sandbox import TIMEOUT_EXIT_CODE
from tests.integration_tests._live import (
    SAFETY_DELETE_AFTER,
    machine_labels,
    machine_name,
    requires_boxd,
)

pytestmark = requires_boxd


def _gone(client: Boxd, machine_id: str, *, wait: float = 30.0) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            machine = client.machines.get(machine_id)
        except NotFoundError:
            return True
        if machine.status == "destroyed":
            return True
        time.sleep(1)
    return False


def test_create_execute_files_and_delete_on_exit() -> None:
    with Boxd() as client:
        with BoxdSandbox.create(
            client=client,
            name=machine_name("life"),
            labels=machine_labels(),
            delete_after=SAFETY_DELETE_AFTER,
        ) as backend:
            machine_id = backend.id
            machine = client.machines.get(machine_id)
            assert machine.status == "running"
            for key, value in machine_labels().items():
                assert machine.labels[key] == value

            result = backend.execute("echo $0; pwd; whoami; [[ 1 == 1 ]] && echo bash")
            assert result.exit_code == 0
            assert result.output.splitlines() == ["bash", "/home/boxd", "boxd", "bash"]

            failed = backend.execute("echo out; echo err >&2; exit 3")
            assert failed.exit_code == 3
            assert failed.output == "out\n\nerr\n"

            payload = bytes(range(256)) * 64
            [upload] = backend.upload_files([("/home/boxd/deep/dir/blob.bin", payload)])
            assert upload.error is None
            [download] = backend.download_files(["/home/boxd/deep/dir/blob.bin"])
            assert download.content == payload
            assert backend.execute("wc -c < deep/dir/blob.bin").output.strip() == str(
                len(payload)
            )
        assert _gone(client, machine_id)


def test_timeout_kills_command_and_returns_124() -> None:
    with BoxdSandbox.create(
        name=machine_name("timeout"),
        labels=machine_labels(),
        delete_after=SAFETY_DELETE_AFTER,
        timeout=3,
    ) as backend:
        started = time.monotonic()
        result = backend.execute("sleep 30; echo late")
        assert time.monotonic() - started < 20
        assert result.exit_code == TIMEOUT_EXIT_CODE
        assert "timed out after 3 seconds" in result.output

        explicit = backend.execute("sleep 30", timeout=2)
        assert explicit.exit_code == TIMEOUT_EXIT_CODE

        # The machine is still usable, and the killed command left nothing behind.
        assert backend.execute("echo still-here", timeout=30).output == "still-here\n"
        assert backend.execute("pgrep -x sleep || echo none").output.strip() == "none"


async def test_async_methods() -> None:
    with BoxdSandbox.create(
        name=machine_name("async"),
        labels=machine_labels(),
        delete_after=SAFETY_DELETE_AFTER,
    ) as backend:
        result = await backend.aexecute("echo async")
        assert result.output == "async\n"
        write = await backend.awrite("/home/boxd/a/notes.md", "# notes\n")
        assert write.error is None
        read = await backend.aread("/home/boxd/a/notes.md")
        assert read.file_data is not None
        assert "# notes" in read.file_data["content"]
        [download] = await backend.adownload_files(["/home/boxd/a/notes.md"])
        assert download.content == b"# notes\n"


def test_wrap_existing_machine_by_id_and_by_name() -> None:
    with Boxd() as client:
        machine = client.machines.create(
            machine_name("wrap"),
            labels=machine_labels(),
            delete_after=SAFETY_DELETE_AFTER,
        )
        try:
            client.machines.wait_until_ready(machine.id)
            by_id = BoxdSandbox(client=client, machine_id=machine.id)
            by_id.write("/home/boxd/hello.txt", "hi\n")
            by_name = BoxdSandbox(client=client, machine_id=machine.name)
            assert by_name.execute("cat hello.txt").output == "hi\n"
            # Leaving a wrapped machine's `with` block keeps the machine.
            with by_id:
                pass
            assert client.machines.get(machine.id).status == "running"
        finally:
            client.machines.delete(machine.id)
        assert _gone(client, machine.id)


def test_provider_create_hibernate_reattach_and_delete() -> None:
    provider = BoxdProvider()
    backend = provider.get_or_create(
        name=machine_name("dcode"), delete_after=SAFETY_DELETE_AFTER
    )
    machine_id = backend.id
    try:
        backend.client.machines.update_labels(machine_id, set=machine_labels())
        assert backend.execute("echo provider-ready > state.txt").exit_code == 0
        backend.client.machines.hibernate(machine_id)
        deadline = time.monotonic() + 120
        while backend.client.machines.get(machine_id).status != "hibernated":
            assert time.monotonic() < deadline, "machine did not hibernate"
            time.sleep(2)
        backend.close()

        again = provider.get_or_create(sandbox_id=machine_id)
        assert again.id == machine_id
        assert again.execute("cat state.txt").output == "provider-ready\n"
        again.close()
    finally:
        provider.delete(sandbox_id=machine_id)
    provider.delete(sandbox_id=machine_id)  # idempotent
    with Boxd() as client:
        assert _gone(client, machine_id)


def test_provider_reattaches_stopped_machine() -> None:
    provider = BoxdProvider()
    backend = provider.get_or_create(
        name=machine_name("stopped"),
        labels=machine_labels(),
        delete_after=SAFETY_DELETE_AFTER,
    )
    machine_id = backend.id
    try:
        backend.execute("echo kept > disk.txt")
        backend.client.machines.stop(machine_id)
        deadline = time.monotonic() + 120
        while backend.client.machines.get(machine_id).status != "stopped":
            assert time.monotonic() < deadline, "machine did not stop"
            time.sleep(2)
        backend.close()

        again = provider.get_or_create(sandbox_id=machine_id)
        assert again.execute("cat disk.txt").output == "kept\n"
        again.close()
    finally:
        provider.delete(sandbox_id=machine_id)


def test_provider_unknown_machine_is_not_found() -> None:
    from deepagents_code.integrations.sandbox_provider import (  # noqa: PLC0415
        SandboxNotFoundError,
    )

    with pytest.raises(SandboxNotFoundError):
        BoxdProvider().get_or_create(sandbox_id=machine_name("missing"))


def test_wrong_api_key_is_a_clear_error() -> None:
    with pytest.raises(AuthenticationError) as exc_info:  # noqa: PT012
        backend = BoxdSandbox(client=Boxd(api_key="not-a-real-key"), machine_id="x")
        backend.execute("true")
    assert exc_info.value.reason in {"invalid_api_key", "invalid_token"}

    provider = BoxdProvider(resolve_env_var={"BOXD_API_KEY": "not-a-real-key"}.get)
    with pytest.raises(ValueError, match="BOXD_API_KEY"):
        provider.get_or_create()


def test_missing_api_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("BOXD_TOKEN"):
        pytest.skip("BOXD_TOKEN would be used instead")
    monkeypatch.delenv("BOXD_API_KEY")
    # Inside a boxd machine the SDK would fall back to the machine's own
    # credential. Hide the hints it looks for, as on a laptop or in CI.
    for hint in ("BOXD_VM_ID", "BOXD_VM_NAME", "BOXD_PROXY_ENDPOINT"):
        monkeypatch.delenv(hint, raising=False)
    provider = BoxdProvider(resolve_env_var={}.get)
    with pytest.raises(ValueError, match="BOXD_API_KEY"):
        provider.get_or_create()


def test_provider_restores_snapshot() -> None:
    snapshot = machine_name("snap")
    with Boxd() as client:
        try:
            with BoxdSandbox.create(
                client=client,
                name=machine_name("snapsrc"),
                labels=machine_labels(),
                delete_after=SAFETY_DELETE_AFTER,
            ) as source:
                source.write("/home/boxd/from-snapshot.txt", "baked in\n")
                client.snapshots.create(source.id, snapshot)
                deadline = time.monotonic() + 300
                while client.snapshots.get(snapshot).status != "ready":
                    assert time.monotonic() < deadline, "snapshot never got ready"
                    time.sleep(3)
            provider = BoxdProvider()
            backend = provider.get_or_create(
                snapshot=snapshot,
                name=machine_name("restored"),
                labels=machine_labels(),
                delete_after=SAFETY_DELETE_AFTER,
            )
            try:
                assert backend.execute("cat from-snapshot.txt").output == "baked in\n"
            finally:
                provider.delete(sandbox_id=backend.id)
        finally:
            with contextlib.suppress(NotFoundError):
                client.snapshots.delete(snapshot)
