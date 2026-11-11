# Changelog

## 0.0.1 (unreleased)

Initial release.

- `BoxdSandbox`: a Deep Agents `BaseSandbox` backend over the boxd Python SDK
  (`execute`, `upload_files`, `download_files`, `id`; `ls`/`read`/`write`/
  `edit`/`grep`/`glob`/`delete` come from the base class). Wraps an existing
  machine, or creates one with `BoxdSandbox.create()` and deletes it when used
  as a context manager.
- `BoxdProvider`: a `deepagents-code` sandbox provider, registered via the
  `deepagents_code.sandbox_providers` entry point and selectable as
  `dcode --sandbox boxd`. Reattaches by id or name and wakes sleeping machines;
  restores snapshots with `--sandbox-snapshot-name`.
- Passes the `langchain-tests` `SandboxIntegrationTests` standard suite, apart
  from 7 tests that fail on any `BaseSandbox` with deepagents 0.7 (marked
  `xfail(strict=True)`).
- Three examples (`examples/`): a quick start, a persistent workspace that
  hibernates between runs, and two agents working on two forks in parallel.
- Ships `py.typed`.
