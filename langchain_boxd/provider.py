"""boxd sandbox lifecycle provider for Deep Agents Code.

Registered under the ``deepagents_code.sandbox_providers`` entry point, so
``dcode --sandbox boxd`` creates (or reattaches to) a boxd machine. Importing
this module needs ``deepagents-code``; ``langchain_boxd`` itself does not.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from typing import NoReturn

from boxd import Boxd
from boxd.errors import AuthenticationError, NotFoundError
from deepagents_code.integrations.sandbox_provider import (
    SandboxInstallHint,
    SandboxNotFoundError,
    SandboxProvider,
    SandboxProviderMetadata,
)

from langchain_boxd.sandbox import (
    DEFAULT_READY_TIMEOUT,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKDIR,
    BoxdSandbox,
)

DEFAULT_COMMAND_TIMEOUT = DEFAULT_TIMEOUT

EnvResolver = Callable[[str], str | None]

_WAKE_POLL_INTERVAL = 0.5

# Machines that are asleep come back with these calls before they take commands.
_WAKE_CALLS = {
    "stopped": "start",
    "suspended": "resume",
    "hibernated": "wake",
}


def _environ_resolve_env_var(name: str) -> str | None:
    if not name.startswith("DEEPAGENTS_CODE_"):
        prefixed = f"DEEPAGENTS_CODE_{name}"
        if prefixed in os.environ:
            return os.environ[prefixed] or None
    return os.environ.get(name) or None


def _default_resolve_env_var(name: str) -> str | None:
    # dcode's own resolver also sees the variables of the active workspace,
    # which `os.environ` does not. Older deepagents-code releases lack it.
    try:
        from deepagents_code.model_config import (  # noqa: PLC0415
            resolve_env_var,
        )
    except ImportError:
        return _environ_resolve_env_var(name)
    return resolve_env_var(name)


def _resolve_int(
    value: int | str | None,
    *,
    name: str,
    default: int | None,
    allow_zero: bool = False,
) -> int | None:
    if value is None:
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be an integer number of seconds"
        raise ValueError(msg) from exc
    lower_bound = 0 if allow_zero else 1
    if resolved < lower_bound:
        requirement = "non-negative" if allow_zero else "positive"
        msg = f"{name} must be {requirement}"
        raise ValueError(msg)
    return resolved


def _raise_unsupported_kwargs(kwargs: dict[str, object]) -> NoReturn:
    names = ", ".join(sorted(kwargs))
    msg = f"Received unsupported arguments: {names}"
    raise TypeError(msg)


class BoxdProvider(SandboxProvider):
    """Create, reattach to and delete boxd machines for Deep Agents Code."""

    _metadata = SandboxProviderMetadata(
        name="boxd",
        working_dir=DEFAULT_WORKDIR,
        install=SandboxInstallHint(kind="package", name="langchain-boxd"),
        supports_sandbox_id=True,
        supports_snapshot_name=True,
        backend_module="langchain_boxd",
    )

    def __init__(self, *, resolve_env_var: EnvResolver | None = None) -> None:
        """Initialize the provider without touching credentials.

        Args:
            resolve_env_var: Environment resolver used for provider settings.
        """
        self._resolve_env_var = resolve_env_var or _default_resolve_env_var

    @property
    def metadata(self) -> SandboxProviderMetadata:
        """Return metadata used by Deep Agents Code provider discovery."""
        return self._metadata

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        snapshot: str | None = None,
        image: str | None = None,
        name: str | None = None,
        labels: Mapping[str, str] | None = None,
        delete_after: int | str | None = None,
        workdir: str = DEFAULT_WORKDIR,
        command_timeout: int | str | None = None,
        **kwargs: object,
    ) -> BoxdSandbox:
        """Reattach to a boxd machine, or create one.

        Args:
            sandbox_id: Existing machine id or name, or ``None`` to create one.
                A stopped, suspended or hibernated machine is brought back.
            snapshot: Snapshot to restore the new machine from.
            image: Image for a new machine. Defaults to ``BOXD_IMAGE``, then
                to the boxd image (Ubuntu 24.04 with Python 3).
            name: Name for a new machine. boxd picks one if omitted.
            labels: ``key=value`` labels for a new machine.
            delete_after: Seconds after which boxd deletes a new machine.
                Defaults to ``BOXD_DELETE_AFTER``; unset means never.
            workdir: Directory commands run in.
            command_timeout: Default command timeout in seconds.
            **kwargs: Unsupported provider options.

        Returns:
            boxd sandbox backend for a running machine.

        Raises:
            SandboxNotFoundError: If ``sandbox_id`` does not exist.
            TypeError: If unsupported provider options are passed.
            ValueError: If credentials or timeout settings are invalid.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)
        if sandbox_id is not None and snapshot is not None:
            msg = "snapshot cannot be combined with sandbox_id"
            raise ValueError(msg)

        default_command_timeout = _resolve_int(
            command_timeout,
            name="command_timeout",
            default=DEFAULT_COMMAND_TIMEOUT,
            allow_zero=True,
        )
        assert default_command_timeout is not None  # noqa: S101
        client = self._client()

        try:
            if sandbox_id is not None:
                return self._attach(
                    client,
                    sandbox_id,
                    workdir=workdir,
                    timeout=default_command_timeout,
                )
            backend = BoxdSandbox.create(
                client=client,
                name=name,
                labels=labels,
                image=None
                if snapshot is not None
                else image or self._resolve_env_var("BOXD_IMAGE"),
                from_snapshot=snapshot,
                delete_after=_resolve_int(
                    delete_after
                    if delete_after is not None
                    else self._resolve_env_var("BOXD_DELETE_AFTER"),
                    name="delete_after",
                    default=None,
                ),
                workdir=workdir,
                timeout=default_command_timeout,
                close_client=True,
            )
        except AuthenticationError as exc:
            client.close()
            raise ValueError(self._auth_message(exc)) from exc
        except BaseException:
            client.close()
            raise
        return backend

    def delete(self, *, sandbox_id: str, **kwargs: object) -> None:
        """Delete a boxd machine. Does nothing if it is already gone.

        Args:
            sandbox_id: Machine id or name.
            **kwargs: Unsupported provider options.

        Raises:
            TypeError: If unsupported provider options are passed.
            ValueError: If credentials are missing or rejected.
        """
        if kwargs:
            _raise_unsupported_kwargs(kwargs)
        client = self._client()
        try:
            client.machines.delete(sandbox_id)
        except NotFoundError:
            pass
        except AuthenticationError as exc:
            raise ValueError(self._auth_message(exc)) from exc
        finally:
            client.close()

    def _attach(
        self,
        client: Boxd,
        sandbox_id: str,
        *,
        workdir: str,
        timeout: int,
    ) -> BoxdSandbox:
        try:
            machine = client.machines.get(sandbox_id)
        except NotFoundError as exc:
            raise SandboxNotFoundError(sandbox_id) from exc
        wake = _WAKE_CALLS.get(machine.status)
        if wake is not None:
            getattr(client.machines, wake)(machine.id)
            # wait_until_ready treats "stopped" as final, so let the status
            # move on first.
            deadline = time.monotonic() + DEFAULT_READY_TIMEOUT
            while client.machines.get(machine.id).status in _WAKE_CALLS:
                if time.monotonic() >= deadline:
                    msg = f"machine {machine.id} did not wake up"
                    raise TimeoutError(msg)
                time.sleep(_WAKE_POLL_INTERVAL)
        client.machines.wait_until_ready(machine.id, timeout=DEFAULT_READY_TIMEOUT)
        return BoxdSandbox(
            machine_id=machine.id,
            client=client,
            workdir=workdir,
            timeout=timeout,
            close_client=True,
        )

    def _client(self) -> Boxd:
        # Pass what the resolver finds explicitly: under dcode it can come from
        # the workspace environment, which the SDK would not read itself.
        api_key = self._resolve_env_var("BOXD_API_KEY")
        region = self._resolve_env_var("BOXD_REGION")
        try:
            # Without a key, Boxd() still works inside a boxd machine, where it
            # picks up that machine's own credential.
            if api_key and region:
                return Boxd(api_key=api_key, region=region)
            if api_key:
                return Boxd(api_key=api_key)
            if region:
                return Boxd(region=region)
            return Boxd()
        except AuthenticationError as exc:
            raise ValueError(self._auth_message(exc)) from exc

    @staticmethod
    def _auth_message(exc: AuthenticationError) -> str:
        return (
            "boxd rejected the credentials or found none "
            f"({exc.reason or 'unknown'}). Set BOXD_API_KEY or "
            "DEEPAGENTS_CODE_BOXD_API_KEY to a key from https://app.boxd.sh/keys."
        )
