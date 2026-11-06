"""boxd sandbox backend implementation.

[`BoxdSandbox`][langchain_boxd.sandbox.BoxdSandbox] runs a Deep Agents agent's
shell commands and file operations inside a [boxd](https://boxd.sh?ref=langchain-boxd)
machine: a full Ubuntu VM with a persistent disk.

It subclasses [`BaseSandbox`][deepagents.backends.sandbox.BaseSandbox], so the
filesystem tools (``ls``, ``read``, ``write``, ``edit``, ``glob``, ``grep``,
``delete``) come from the base class on top of three primitives implemented
here with the boxd Python SDK:

- ``execute`` runs a shell command with ``client.machines.exec``.
- ``upload_files`` / ``download_files`` move raw bytes with
  ``client.machines.fs.write`` / ``client.machines.fs.read``.
"""

from __future__ import annotations

import contextlib
import posixpath
import shlex
from typing import TYPE_CHECKING, Any, Self

from boxd import Boxd
from boxd.errors import (
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
from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType


DEFAULT_WORKDIR = "/home/boxd"
"""Home directory of the ``boxd`` user on the default boxd image."""

DEFAULT_TIMEOUT = 30 * 60
"""Default command timeout in seconds."""

DEFAULT_SHELL = "bash"
"""Shell that runs each command. The machine's own exec shell is ``/bin/sh``."""

DEFAULT_READY_TIMEOUT = 180.0
"""Seconds ``BoxdSandbox.create`` waits for a new machine to accept commands."""

TIMEOUT_EXIT_CODE = 124
"""Exit code returned when a command runs past its timeout (as ``timeout(1)``)."""

_TIMEOUT_CODE = "deadline_exceeded"
_TIMEOUT_MESSAGE_PREFIX = "exec timed out"


def _text(data: str | bytes | None) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data or ""


def _combine_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    output = _text(stdout)
    stderr = _text(stderr)
    if stderr:
        output += "\n" + stderr if output else stderr
    return output


def _is_timeout(exc: BoxdError) -> bool:
    return exc.code == _TIMEOUT_CODE or str(exc).startswith(_TIMEOUT_MESSAGE_PREFIX)


def _download_error(exc: BoxdError) -> FileOperationError | str:
    if isinstance(exc, MachineFileNotFoundError):
        return FILE_NOT_FOUND
    if isinstance(exc, MachineIsADirectoryError):
        return IS_DIRECTORY
    if isinstance(exc, MachinePermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, MachineNotADirectoryError):
        # A path component is a file, so the requested file cannot exist.
        return FILE_NOT_FOUND
    return f"{type(exc).__name__}: {exc}"


def _upload_error(exc: BoxdError) -> FileOperationError | str:
    if isinstance(exc, MachinePermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, MachineIsADirectoryError):
        return IS_DIRECTORY
    if isinstance(exc, MachineNotADirectoryError):
        # A path component is an existing file, so no directory can be made.
        return INVALID_PATH
    if isinstance(exc, MachineFileNotFoundError):
        return FILE_NOT_FOUND
    return f"{type(exc).__name__}: {exc}"


class BoxdSandbox(BaseSandbox):
    """Deep Agents sandbox backend that runs inside a boxd machine.

    Wrap a machine you already have:

    ```python
    from boxd import Boxd
    from langchain_boxd import BoxdSandbox

    client = Boxd()  # reads BOXD_API_KEY
    machine = client.machines.create(labels={"app": "agent"})
    client.machines.wait_until_ready(machine.id)

    backend = BoxdSandbox(client=client, machine_id=machine.id)
    print(backend.execute("uname -a").output)
    ```

    Or let the backend create the machine and delete it when you are done:

    ```python
    from langchain_boxd import BoxdSandbox

    with BoxdSandbox.create(delete_after=3600) as backend:
        print(backend.execute("python3 --version").output)
    ```
    """

    def __init__(
        self,
        *,
        machine_id: str,
        client: Boxd | None = None,
        workdir: str = DEFAULT_WORKDIR,
        timeout: int = DEFAULT_TIMEOUT,
        shell: str | None = DEFAULT_SHELL,
        region: str | None = None,
        delete_on_exit: bool = False,
        close_client: bool | None = None,
    ) -> None:
        """Create a backend for an existing boxd machine.

        Args:
            machine_id: Id (or name) of the machine to run in. The machine
                should be running; see ``Boxd.machines.wait_until_ready``.
            client: Authenticated ``boxd.Boxd`` client. If omitted, one is
                created with ``Boxd()``, which reads ``BOXD_API_KEY`` (and
                ``BOXD_REGION``) from the environment. A client created here
                is closed by ``close()``.
            workdir: Directory commands run in.
            timeout: Default command timeout in seconds, used when
                ``execute()`` is called without one. ``0`` waits indefinitely.
            shell: Shell each command runs under (``shell -c '<command>'``).
                ``None`` hands the command line to the machine's exec shell
                (``/bin/sh``) unchanged, for images without bash.
            region: boxd region the machine lives in, when it is not the
                client's region.
            delete_on_exit: Delete the machine when the backend is used as a
                context manager and the ``with`` block ends. ``create()`` sets
                this.
            close_client: Close ``client`` in ``close()``. Defaults to closing
                only a client this backend created itself.
        """
        self._owns_client = client is None if close_client is None else close_client
        self._client = client if client is not None else Boxd()
        self._machine_id = machine_id
        self._workdir = workdir
        self._default_timeout = timeout
        self._shell = shell
        self._region = region
        self._delete_on_exit = delete_on_exit

    @classmethod
    def create(
        cls,
        *,
        client: Boxd | None = None,
        name: str | None = None,
        image: str | None = None,
        from_snapshot: str | None = None,
        labels: Mapping[str, str] | None = None,
        delete_after: int | None = None,
        region: str | None = None,
        ready_timeout: float = DEFAULT_READY_TIMEOUT,
        workdir: str = DEFAULT_WORKDIR,
        timeout: int = DEFAULT_TIMEOUT,
        shell: str | None = DEFAULT_SHELL,
        close_client: bool | None = None,
        **create_kwargs: Any,
    ) -> BoxdSandbox:
        """Create a new boxd machine and return a backend that owns it.

        Waits until the machine accepts commands. Used as a context manager,
        the backend deletes the machine when the ``with`` block ends;
        otherwise call ``delete_machine()`` yourself.

        Args:
            client: Authenticated ``boxd.Boxd`` client. If omitted, one is
                created with ``Boxd()`` (reads ``BOXD_API_KEY``).
            name: Machine name. Names are unique across boxd; leave it out and
                boxd picks a free one.
            image: Image to boot. Defaults to the boxd image (Ubuntu 24.04
                with Python 3, git, Node, Docker and more).
            from_snapshot: Restore a saved snapshot instead of booting an image.
            labels: ``key=value`` labels to put on the machine.
            delete_after: Seconds after which boxd deletes the machine even if
                this process never does. A safety net for crashed runs.
            region: boxd region to create the machine in.
            ready_timeout: Seconds to wait for the machine to accept commands.
            workdir: Directory commands run in.
            timeout: Default command timeout in seconds.
            shell: Shell each command runs under, or ``None`` for the
                machine's exec shell.
            close_client: Close ``client`` in ``close()``. Defaults to closing
                only a client created here.
            **create_kwargs: Any other ``client.machines.create`` option, for
                example ``vcpu``, ``memory``, ``env``, ``org``, ``isolated``,
                ``networks`` or ``spec``.

        Returns:
            A ready ``BoxdSandbox`` with ``delete_on_exit=True``.

        Raises:
            TimeoutError: The machine did not become ready in time. The
                machine is deleted before this is raised.
        """
        owns_client = client is None if close_client is None else close_client
        sdk = client if client is not None else Boxd()
        try:
            machine = sdk.machines.create(
                name,
                image=image,
                from_snapshot=from_snapshot,
                labels=labels,
                delete_after=delete_after,
                region=region,
                **create_kwargs,
            )
        except BaseException:
            if owns_client:
                sdk.close()
            raise
        try:
            sdk.machines.wait_until_ready(
                machine.id, timeout=ready_timeout, region=region
            )
        except BaseException:
            try:
                sdk.machines.delete(machine.id, region=region)
            finally:
                if owns_client:
                    sdk.close()
            raise
        return cls(
            machine_id=machine.id,
            client=sdk,
            workdir=workdir,
            timeout=timeout,
            shell=shell,
            region=region,
            delete_on_exit=True,
            close_client=owns_client,
        )

    @property
    def id(self) -> str:
        """Return the boxd machine id (or the name it was opened with)."""
        return self._machine_id

    @property
    def client(self) -> Boxd:
        """The ``boxd.Boxd`` client, for anything beyond the sandbox protocol.

        For example ``backend.client.machines.hibernate(backend.id)``.
        """
        return self._client

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the machine.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for this command. If
                ``None``, uses the backend's default timeout. ``0`` waits
                indefinitely. When it runs out, boxd kills the command's
                process group and exit code ``124`` is returned.

        Returns:
            ExecuteResponse with stdout and stderr combined, and the exit code.

        Raises:
            ValueError: If ``timeout`` is negative.
            AuthenticationError: If boxd rejects the credentials.
            PermissionDeniedError: If the credentials may not use the machine.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout < 0:
            msg = f"timeout must be non-negative, got {effective_timeout}"
            raise ValueError(msg)

        line = (
            command
            if self._shell is None
            else f"{self._shell} -c {shlex.quote(command)}"
        )
        try:
            result = self._client.machines.exec(
                self._machine_id,
                line,
                cwd=self._workdir,
                timeout=effective_timeout or None,
                region=self._region,
            )
        except (AuthenticationError, PermissionDeniedError):
            raise
        except BoxdError as exc:
            if effective_timeout and _is_timeout(exc):
                return ExecuteResponse(
                    output=f"Command timed out after {effective_timeout} seconds",
                    exit_code=TIMEOUT_EXIT_CODE,
                    truncated=False,
                )
            return ExecuteResponse(
                output=f"Error executing command ({type(exc).__name__}): {exc}",
                exit_code=1,
                truncated=False,
            )

        return ExecuteResponse(
            output=_combine_output(result.stdout, result.stderr),
            exit_code=result.exit_code,
            truncated=False,
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the machine, in the same order as ``files``.

        Missing parent directories are created. Each file is replaced as a
        whole. Failures are reported per file, not raised.
        """
        return [self._write_file(path, content) for path, content in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the machine, in the same order as ``paths``.

        Failures are reported per file, not raised.
        """
        return [self._read_file(path) for path in paths]

    def delete_machine(self) -> None:
        """Delete the boxd machine. Does nothing if it is already gone."""
        with contextlib.suppress(NotFoundError):
            self._client.machines.delete(self._machine_id, region=self._region)

    def close(self) -> None:
        """Close the boxd client if this backend created it.

        Does not touch the machine; see ``delete_machine()``.
        """
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        """Use the backend as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Delete the machine if ``delete_on_exit`` is set, then close."""
        try:
            if self._delete_on_exit:
                self.delete_machine()
        finally:
            self.close()

    def _read_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, content=None, error=INVALID_PATH)
        try:
            content = self._client.machines.fs.read(
                self._machine_id, path, region=self._region
            )
        except MachineAgentOutdatedError:
            return self._read_file_legacy(path)
        except BoxdError as exc:
            return FileDownloadResponse(
                path=path, content=None, error=_download_error(exc)
            )
        return FileDownloadResponse(path=path, content=content, error=None)

    def _write_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error=INVALID_PATH)
        try:
            self._client.machines.fs.write(
                self._machine_id, path, content, region=self._region
            )
        except MachineAgentOutdatedError:
            return self._write_file_legacy(path, content)
        except BoxdError as exc:
            return FileUploadResponse(path=path, error=_upload_error(exc))
        return FileUploadResponse(path=path, error=None)

    # Machines started before boxd's filesystem API existed answer
    # `machines.fs` with MachineAgentOutdatedError until they are rebooted.
    # They still take the older `machines.files` stream, which neither
    # creates parent directories nor names the failure, so those two are
    # handled with a shell command.

    def _read_file_legacy(self, path: str) -> FileDownloadResponse:
        try:
            content = self._client.machines.files.download(
                self._machine_id, path, region=self._region
            )
        except BoxdError as exc:
            quoted = shlex.quote(path)
            probe = self.execute(
                f"if [ -d {quoted} ]; then echo dir; "
                f"elif [ ! -e {quoted} ]; then echo missing; "
                f"elif [ ! -r {quoted} ]; then echo denied; fi"
            )
            verdict = probe.output.strip()
            error: FileOperationError | str = {
                "dir": IS_DIRECTORY,
                "missing": FILE_NOT_FOUND,
                "denied": PERMISSION_DENIED,
            }.get(verdict, f"{type(exc).__name__}: {exc}")
            return FileDownloadResponse(path=path, content=None, error=error)
        return FileDownloadResponse(path=path, content=content, error=None)

    def _write_file_legacy(self, path: str, content: bytes) -> FileUploadResponse:
        parent = posixpath.dirname(path)
        if parent and parent != "/":
            mkdir = self.execute(f"mkdir -p {shlex.quote(parent)}")
            if mkdir.exit_code != 0:
                return FileUploadResponse(path=path, error=PERMISSION_DENIED)
        try:
            self._client.machines.files.upload(
                self._machine_id, path, content, region=self._region
            )
        except BoxdError as exc:
            return FileUploadResponse(path=path, error=f"{type(exc).__name__}: {exc}")
        return FileUploadResponse(path=path, error=None)
