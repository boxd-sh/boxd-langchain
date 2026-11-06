"""In-memory stand-ins for the parts of the boxd SDK the backend uses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from boxd.errors import MachineFileNotFoundError


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class FakeMachine:
    id: str
    name: str
    status: str = "running"


@dataclass
class FakeFs:
    entries: dict[str, bytes] = field(default_factory=dict)
    read_exc: BaseException | None = None
    write_exc: BaseException | None = None
    calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def read(self, machine_id: str, path: str, *, region: str | None = None) -> bytes:
        self.calls.append(("read", machine_id, region))
        if self.read_exc is not None:
            raise self.read_exc
        if path not in self.entries:
            msg = f"{path}: No such file or directory"
            raise MachineFileNotFoundError(msg, reason="not_found", filename=path)
        return self.entries[path]

    def write(
        self,
        machine_id: str,
        path: str,
        data: bytes,
        *,
        region: str | None = None,
    ) -> None:
        self.calls.append(("write", machine_id, region))
        if self.write_exc is not None:
            raise self.write_exc
        self.entries[path] = data


@dataclass
class FakeFiles:
    entries: dict[str, bytes] = field(default_factory=dict)
    download_exc: BaseException | None = None
    upload_exc: BaseException | None = None

    def download(
        self, machine_id: str, path: str, *, region: str | None = None
    ) -> bytes:
        del machine_id, region
        if self.download_exc is not None:
            raise self.download_exc
        return self.entries[path]

    def upload(
        self,
        machine_id: str,
        path: str,
        data: bytes,
        *,
        region: str | None = None,
    ) -> int:
        del machine_id, region
        if self.upload_exc is not None:
            raise self.upload_exc
        self.entries[path] = data
        return len(data)


@dataclass
class FakeMachines:
    result: FakeExecResult | None = field(default_factory=FakeExecResult)
    results: list[FakeExecResult] = field(default_factory=list)
    exec_exc: BaseException | None = None
    fs: FakeFs = field(default_factory=FakeFs)
    files: FakeFiles = field(default_factory=FakeFiles)
    machines: dict[str, FakeMachine] = field(default_factory=dict)
    exec_calls: list[dict[str, Any]] = field(default_factory=list)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    lifecycle: list[tuple[str, str]] = field(default_factory=list)
    ready_exc: BaseException | None = None
    delete_exc: BaseException | None = None
    get_exc: BaseException | None = None

    def exec(self, machine_id: str, command: str, **kwargs: Any) -> FakeExecResult:
        self.exec_calls.append({"machine_id": machine_id, "command": command, **kwargs})
        if self.exec_exc is not None:
            raise self.exec_exc
        if self.results:
            return self.results.pop(0)
        assert self.result is not None
        return self.result

    def create(self, name: str | None = None, **kwargs: Any) -> FakeMachine:
        self.create_calls.append({"name": name, **kwargs})
        machine = FakeMachine(id=f"vm-{len(self.create_calls)}", name=name or "auto")
        self.machines[machine.id] = machine
        return machine

    def get(self, machine_id: str, *, region: str | None = None) -> FakeMachine:
        del region
        if self.get_exc is not None:
            raise self.get_exc
        for machine in self.machines.values():
            if machine_id in {machine.id, machine.name}:
                return machine
        msg = f"no machine {machine_id}"
        raise KeyError(msg)

    def wait_until_ready(
        self,
        machine_id: str,
        *,
        timeout: float = 90.0,
        region: str | None = None,
    ) -> FakeMachine:
        del timeout, region
        self.lifecycle.append(("wait_until_ready", machine_id))
        if self.ready_exc is not None:
            raise self.ready_exc
        return self.machines.get(machine_id) or FakeMachine(machine_id, machine_id)

    def delete(self, machine_id: str, *, region: str | None = None) -> None:
        del region
        if self.delete_exc is not None:
            raise self.delete_exc
        self.deleted.append(machine_id)

    def start(self, machine_id: str) -> None:
        self.lifecycle.append(("start", machine_id))

    def resume(self, machine_id: str) -> None:
        self.lifecycle.append(("resume", machine_id))

    def wake(self, machine_id: str) -> None:
        self.lifecycle.append(("wake", machine_id))


@dataclass
class FakeBoxd:
    machines: FakeMachines = field(default_factory=FakeMachines)
    closed: bool = False

    def close(self) -> None:
        self.closed = True
