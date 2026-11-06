from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, cast

import pytest
from boxd.errors import (
    APIConnectionError,
    AuthenticationError,
    BoxdError,
    MachineAgentOutdatedError,
    MachineFileNotFoundError,
    MachineIsADirectoryError,
    MachineNotADirectoryError,
    MachinePermissionError,
    NotFoundError,
    PermissionDeniedError,
)
from deepagents.backends.sandbox import BaseSandbox

from langchain_boxd import BoxdSandbox
from langchain_boxd.sandbox import DEFAULT_WORKDIR, TIMEOUT_EXIT_CODE
from tests.unit_tests.fakes import FakeBoxd, FakeExecResult, FakeFiles, FakeFs

if TYPE_CHECKING:
    from boxd import Boxd

FILE_PATH = "/home/boxd/file.txt"


def _backend(
    fake: FakeBoxd | None = None,
    **kwargs: object,
) -> tuple[BoxdSandbox, FakeBoxd]:
    fake = fake or FakeBoxd()
    backend = BoxdSandbox(
        machine_id="vm-test",
        client=cast("Boxd", fake),
        **kwargs,  # type: ignore[arg-type]
    )
    return backend, fake


def _fs_error(cls: type[BoxdError], reason: str) -> BoxdError:
    return cls(f"{FILE_PATH}: {reason}", reason=reason)


# -- shape -------------------------------------------------------------------


def test_is_a_base_sandbox() -> None:
    backend, _ = _backend()
    assert isinstance(backend, BaseSandbox)


def test_id_is_the_machine_id() -> None:
    backend, _ = _backend()
    assert backend.id == "vm-test"


def test_client_property_exposes_the_sdk_client() -> None:
    backend, fake = _backend()
    assert backend.client is fake


def test_default_client_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class RecordingBoxd(FakeBoxd):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr("langchain_boxd.sandbox.Boxd", RecordingBoxd)
    backend = BoxdSandbox(machine_id="vm-env")
    assert created == [{}]
    backend.close()
    assert cast("FakeBoxd", backend.client).closed is True


# -- execute -----------------------------------------------------------------


def test_execute_runs_command_under_bash_in_workdir() -> None:
    backend, fake = _backend(workdir="/srv/app", timeout=42)
    fake.machines.result = FakeExecResult(stdout="hello\n")

    result = backend.execute("echo hello | cat")

    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert result.truncated is False
    call = fake.machines.exec_calls[0]
    assert call["machine_id"] == "vm-test"
    assert call["command"] == "bash -c 'echo hello | cat'"
    assert shlex.split(call["command"]) == ["bash", "-c", "echo hello | cat"]
    assert call["cwd"] == "/srv/app"
    assert call["timeout"] == 42
    assert call["region"] is None


def test_execute_default_workdir_is_boxd_home() -> None:
    backend, fake = _backend()
    backend.execute("pwd")
    assert fake.machines.exec_calls[0]["cwd"] == DEFAULT_WORKDIR == "/home/boxd"


def test_execute_without_shell_passes_command_through() -> None:
    backend, fake = _backend(shell=None)
    backend.execute("echo $0")
    assert fake.machines.exec_calls[0]["command"] == "echo $0"


def test_execute_custom_shell() -> None:
    backend, fake = _backend(shell="/bin/zsh")  # noqa: S604
    backend.execute("echo hi")
    assert fake.machines.exec_calls[0]["command"] == "/bin/zsh -c 'echo hi'"


def test_execute_quotes_heredocs_and_quotes() -> None:
    command = "cat <<'EOF'\nit's \"quoted\" $HOME\nEOF"
    backend, fake = _backend()
    backend.execute(command)
    assert shlex.split(fake.machines.exec_calls[0]["command"]) == [
        "bash",
        "-c",
        command,
    ]


def test_execute_passes_region() -> None:
    backend, fake = _backend(region="nyc")
    backend.execute("true")
    assert fake.machines.exec_calls[0]["region"] == "nyc"


def test_execute_explicit_timeout_overrides_default() -> None:
    backend, fake = _backend(timeout=100)
    backend.execute("true", timeout=5)
    assert fake.machines.exec_calls[0]["timeout"] == 5


def test_execute_zero_timeout_waits_indefinitely() -> None:
    backend, fake = _backend()
    backend.execute("true", timeout=0)
    assert fake.machines.exec_calls[0]["timeout"] is None


def test_execute_rejects_negative_timeout() -> None:
    backend, _ = _backend()
    with pytest.raises(ValueError, match="timeout must be non-negative"):
        backend.execute("true", timeout=-1)


def test_execute_combines_stdout_and_stderr() -> None:
    backend, fake = _backend()
    fake.machines.result = FakeExecResult(stdout="out", stderr="err", exit_code=0)
    assert backend.execute("x").output == "out\nerr"


def test_execute_stderr_only() -> None:
    backend, fake = _backend()
    fake.machines.result = FakeExecResult(stderr="boom", exit_code=2)
    result = backend.execute("false")
    assert result.output == "boom"
    assert result.exit_code == 2


def test_execute_timeout_returns_124() -> None:
    backend, fake = _backend()
    fake.machines.exec_exc = BoxdError("exec timed out after 5s")
    result = backend.execute("sleep 10", timeout=5)
    assert result.exit_code == TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 5 seconds"


def test_execute_deadline_exceeded_code_returns_124() -> None:
    backend, fake = _backend()
    fake.machines.exec_exc = BoxdError("deadline", code="deadline_exceeded")
    assert backend.execute("sleep 10", timeout=1).exit_code == TIMEOUT_EXIT_CODE


def test_execute_transport_error_is_returned_not_raised() -> None:
    backend, fake = _backend()
    fake.machines.exec_exc = APIConnectionError("could not reach boxd")
    result = backend.execute("ls")
    assert result.exit_code == 1
    assert "APIConnectionError" in result.output
    assert "could not reach boxd" in result.output


def test_execute_missing_machine_is_returned() -> None:
    backend, fake = _backend()
    fake.machines.exec_exc = NotFoundError("machine vm-test not found")
    result = backend.execute("ls")
    assert result.exit_code == 1
    assert "not found" in result.output


@pytest.mark.parametrize(
    "exc",
    [
        AuthenticationError("bad key", reason="invalid_api_key"),
        PermissionDeniedError("not yours"),
    ],
)
def test_execute_raises_credential_errors(exc: BoxdError) -> None:
    backend, fake = _backend()
    fake.machines.exec_exc = exc
    with pytest.raises(type(exc)):
        backend.execute("ls")


async def test_aexecute_runs_in_thread() -> None:
    backend, fake = _backend()
    fake.machines.result = FakeExecResult(stdout="async")
    result = await backend.aexecute("echo async")
    assert result.output == "async"


# -- download ----------------------------------------------------------------


def test_download_relative_path_is_invalid() -> None:
    backend, fake = _backend()
    [response] = backend.download_files(["relative.txt"])
    assert response.error == "invalid_path"
    assert response.content is None
    assert fake.machines.fs.calls == []


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_fs_error(MachineFileNotFoundError, "not_found"), "file_not_found"),
        (_fs_error(MachineIsADirectoryError, "is_a_directory"), "is_directory"),
        (
            _fs_error(MachinePermissionError, "permission_denied"),
            "permission_denied",
        ),
        (_fs_error(MachineNotADirectoryError, "not_a_directory"), "file_not_found"),
    ],
)
def test_download_maps_filesystem_errors(exc: BoxdError, expected: str) -> None:
    backend, fake = _backend()
    fake.machines.fs.read_exc = exc
    [response] = backend.download_files([FILE_PATH])
    assert response.error == expected
    assert response.content is None
    assert response.path == FILE_PATH


def test_download_other_error_is_described() -> None:
    backend, fake = _backend()
    fake.machines.fs.read_exc = APIConnectionError("network down")
    [response] = backend.download_files([FILE_PATH])
    assert response.error == "APIConnectionError: network down"


def test_download_passes_machine_and_region() -> None:
    fs = FakeFs(entries={FILE_PATH: b"x"})
    fake = FakeBoxd()
    fake.machines.fs = fs
    backend, _ = _backend(fake, region="fra")
    backend.download_files([FILE_PATH])
    assert fs.calls == [("read", "vm-test", "fra")]


# -- upload ------------------------------------------------------------------


def test_upload_relative_path_is_invalid() -> None:
    backend, fake = _backend()
    [response] = backend.upload_files([("relative.txt", b"x")])
    assert response.error == "invalid_path"
    assert fake.machines.fs.calls == []


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            _fs_error(MachinePermissionError, "permission_denied"),
            "permission_denied",
        ),
        (_fs_error(MachineIsADirectoryError, "is_a_directory"), "is_directory"),
        (_fs_error(MachineNotADirectoryError, "not_a_directory"), "invalid_path"),
        (_fs_error(MachineFileNotFoundError, "not_found"), "file_not_found"),
    ],
)
def test_upload_maps_filesystem_errors(exc: BoxdError, expected: str) -> None:
    backend, fake = _backend()
    fake.machines.fs.write_exc = exc
    [response] = backend.upload_files([(FILE_PATH, b"x")])
    assert response.error == expected


def test_upload_and_download_round_trip() -> None:
    backend, _ = _backend()
    [upload] = backend.upload_files([(FILE_PATH, b"\x00\xffbinary")])
    [download] = backend.download_files([FILE_PATH])
    assert upload.error is None
    assert download.error is None
    assert download.content == b"\x00\xffbinary"


def test_partial_success_keeps_order() -> None:
    backend, _ = _backend()
    paths = [f"/tmp/f{i}.txt" for i in range(4)]
    responses = backend.upload_files(
        [(paths[0], b"0"), ("rel", b"r"), (paths[1], b"1")]
    )
    assert [r.path for r in responses] == [paths[0], "rel", paths[1]]
    assert [r.error for r in responses] == [None, "invalid_path", None]

    downloads = backend.download_files([paths[1], paths[3], paths[0]])
    assert [d.path for d in downloads] == [paths[1], paths[3], paths[0]]
    assert [d.content for d in downloads] == [b"1", None, b"0"]


def test_base_sandbox_write_uses_upload() -> None:
    backend, fake = _backend()
    # The write preflight runs a check script and expects an empty result.
    fake.machines.result = FakeExecResult(stdout="", exit_code=0)
    result = backend.write("/tmp/new.txt", "content")
    assert result.error is None
    assert fake.machines.fs.entries["/tmp/new.txt"] == b"content"


# -- machines started before machines.fs existed ------------------------------


def _outdated() -> MachineAgentOutdatedError:
    return MachineAgentOutdatedError("reboot to update", reason="agent_outdated")


def test_legacy_upload_creates_parent_then_uses_files_stream() -> None:
    fake = FakeBoxd()
    fake.machines.fs.write_exc = _outdated()
    fake.machines.files = FakeFiles()
    backend, _ = _backend(fake)

    [response] = backend.upload_files([("/tmp/a/b/c.txt", b"x")])

    assert response.error is None
    assert fake.machines.files.entries["/tmp/a/b/c.txt"] == b"x"
    assert fake.machines.exec_calls[0]["command"] == "bash -c 'mkdir -p /tmp/a/b'"


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            "mkdir: cannot create directory '/root/x': Permission denied",
            "permission_denied",
        ),
        ("mkdir: cannot create directory '/f/x': Not a directory", "invalid_path"),
        ("mkdir: cannot create directory '/f': File exists", "invalid_path"),
        ("disk full", "disk full"),
    ],
)
def test_legacy_upload_mkdir_failure(stderr: str, expected: str) -> None:
    fake = FakeBoxd()
    fake.machines.fs.write_exc = _outdated()
    fake.machines.result = FakeExecResult(stderr=stderr, exit_code=1)
    backend, _ = _backend(fake)
    [response] = backend.upload_files([("/root/x/y.txt", b"x")])
    assert response.error == expected


def test_legacy_download_reads_files_stream() -> None:
    fake = FakeBoxd()
    fake.machines.fs.read_exc = _outdated()
    fake.machines.files = FakeFiles(entries={FILE_PATH: b"old"})
    backend, _ = _backend(fake)
    [response] = backend.download_files([FILE_PATH])
    assert response.content == b"old"


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        ("dir\n", "is_directory"),
        ("missing\n", "file_not_found"),
        ("denied\n", "permission_denied"),
        ("", "BoxdError: stream failed"),
    ],
)
def test_legacy_download_classifies_failures(probe: str, expected: str) -> None:
    fake = FakeBoxd()
    fake.machines.fs.read_exc = _outdated()
    fake.machines.files = FakeFiles(download_exc=BoxdError("stream failed"))
    fake.machines.result = FakeExecResult(stdout=probe)
    backend, _ = _backend(fake)
    [response] = backend.download_files([FILE_PATH])
    assert response.error == expected


# -- lifecycle ---------------------------------------------------------------


def test_create_makes_machine_waits_and_owns_it() -> None:
    fake = FakeBoxd()
    backend = BoxdSandbox.create(
        client=cast("Boxd", fake),
        name="t-agent",
        image="ubuntu:24.04",
        labels={"team": "ai"},
        delete_after=600,
        region="fra",
        vcpu=2,
    )
    assert backend.id == "vm-1"
    assert fake.machines.create_calls == [
        {
            "name": "t-agent",
            "image": "ubuntu:24.04",
            "from_snapshot": None,
            "labels": {"team": "ai"},
            "delete_after": 600,
            "region": "fra",
            "vcpu": 2,
        }
    ]
    assert fake.machines.lifecycle == [("wait_until_ready", "vm-1")]

    with backend:
        pass
    assert fake.machines.deleted == ["vm-1"]
    assert fake.closed is False  # a client passed in stays open


def test_create_without_client_closes_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeBoxd()
    monkeypatch.setattr("langchain_boxd.sandbox.Boxd", lambda: fake)
    with BoxdSandbox.create() as backend:
        assert backend.client is fake
    assert fake.machines.deleted == ["vm-1"]
    assert fake.closed is True


def test_create_deletes_machine_that_never_gets_ready() -> None:
    fake = FakeBoxd()
    fake.machines.ready_exc = TimeoutError("not ready")
    with pytest.raises(TimeoutError):
        BoxdSandbox.create(client=cast("Boxd", fake), ready_timeout=1)
    assert fake.machines.deleted == ["vm-1"]


def test_context_manager_keeps_machine_it_did_not_create() -> None:
    backend, fake = _backend()
    with backend:
        pass
    assert fake.machines.deleted == []


def test_context_manager_deletes_even_when_body_raises() -> None:
    fake = FakeBoxd()
    backend = BoxdSandbox.create(client=cast("Boxd", fake))
    with pytest.raises(RuntimeError), backend:
        raise RuntimeError
    assert fake.machines.deleted == ["vm-1"]


def test_delete_machine_ignores_missing() -> None:
    backend, fake = _backend()
    fake.machines.delete_exc = NotFoundError("gone")
    backend.delete_machine()


def test_delete_machine_propagates_other_errors() -> None:
    backend, fake = _backend()
    fake.machines.delete_exc = PermissionDeniedError("not yours")
    with pytest.raises(PermissionDeniedError):
        backend.delete_machine()


def test_close_client_override() -> None:
    backend, fake = _backend(close_client=True)
    backend.close()
    assert fake.closed is True


def test_create_failure_closes_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBoxd()

    def refuse(*_a: object, **_k: object) -> None:
        msg = "quota"
        raise PermissionDeniedError(msg)

    fake.machines.create = refuse  # type: ignore[method-assign]
    monkeypatch.setattr("langchain_boxd.sandbox.Boxd", lambda: fake)
    with pytest.raises(PermissionDeniedError):
        BoxdSandbox.create()
    assert fake.closed is True


def test_create_keeps_original_error_when_cleanup_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeBoxd()
    fake.machines.ready_exc = TimeoutError("not ready")
    fake.machines.delete_exc = APIConnectionError("network down")
    with pytest.raises(TimeoutError, match="not ready"):
        BoxdSandbox.create(client=cast("Boxd", fake))
    assert "Could not delete machine vm-1" in caplog.text


@pytest.mark.parametrize(
    "exc",
    [
        AuthenticationError("revoked", reason="invalid_api_key"),
        PermissionDeniedError("account disabled"),
    ],
)
def test_file_transfers_raise_credential_errors(exc: BoxdError) -> None:
    backend, fake = _backend()
    fake.machines.fs.read_exc = exc
    fake.machines.fs.write_exc = exc
    with pytest.raises(type(exc)):
        backend.download_files([FILE_PATH])
    with pytest.raises(type(exc)):
        backend.upload_files([(FILE_PATH, b"x")])


def test_path_permission_error_is_not_a_credential_error() -> None:
    # MachinePermissionError subclasses PermissionDeniedError, but it is
    # about the file, so it stays a per-file result.
    backend, fake = _backend()
    fake.machines.fs.read_exc = _fs_error(MachinePermissionError, "permission_denied")
    [response] = backend.download_files([FILE_PATH])
    assert response.error == "permission_denied"


def test_legacy_paths_raise_credential_errors() -> None:
    fake = FakeBoxd()
    fake.machines.fs.read_exc = _outdated()
    fake.machines.files = FakeFiles(
        download_exc=AuthenticationError("revoked", reason="invalid_api_key")
    )
    backend, _ = _backend(fake)
    with pytest.raises(AuthenticationError):
        backend.download_files([FILE_PATH])


def test_fake_calls_match_the_real_sdk_signatures() -> None:
    """Every call the backend makes must bind to the real boxd SDK methods."""
    import inspect  # noqa: PLC0415

    from boxd.resources.fs import Fs  # noqa: PLC0415
    from boxd.resources.machines import Files, Machines  # noqa: PLC0415

    real = {
        "exec": Machines.exec,
        "create": Machines.create,
        "wait_until_ready": Machines.wait_until_ready,
        "delete": Machines.delete,
        "read": Fs.read,
        "write": Fs.write,
        "upload": Files.upload,
        "download": Files.download,
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def recorder(name: str, result: object = None) -> object:
        def record(*args: object, **kwargs: object) -> object:
            calls.append((name, args, kwargs))
            return result

        return record

    fake = FakeBoxd()
    m = fake.machines
    m.exec = recorder("exec", FakeExecResult())  # type: ignore[method-assign]
    m.wait_until_ready = recorder("wait_until_ready")  # type: ignore[method-assign]
    m.delete = recorder("delete")  # type: ignore[method-assign]
    original_create = m.create

    def create(*args: object, **kwargs: object) -> object:
        calls.append(("create", args, kwargs))
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    m.create = create  # type: ignore[method-assign]
    m.fs.read = recorder("read", b"x")  # type: ignore[method-assign]
    m.fs.write = recorder("write")  # type: ignore[method-assign]
    m.files.upload = recorder("upload", 1)  # type: ignore[method-assign]
    m.files.download = recorder("download", b"x")  # type: ignore[method-assign]

    with BoxdSandbox.create(
        client=cast("Boxd", fake), name="n", labels={"a": "b"}, delete_after=60, vcpu=2
    ) as backend:
        backend.execute("true", timeout=5)
        backend.upload_files([(FILE_PATH, b"x")])
        backend.download_files([FILE_PATH])
        m.fs.read = recorder("read_outdated")  # type: ignore[method-assign]
        backend._read_file_legacy(FILE_PATH)
        backend._write_file_legacy(FILE_PATH, b"x")

    seen = set()
    for name, args, kwargs in calls:
        if name not in real:
            continue
        seen.add(name)
        inspect.signature(real[name]).bind(object(), *args, **kwargs)
    assert seen == set(real)
