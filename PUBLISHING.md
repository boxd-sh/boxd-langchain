# Publishing

Nothing here is published yet.

1. Check everything, against real boxd too:

   ```sh
   uv sync --group test
   make lint
   make test
   BOXD_API_KEY=... ANTHROPIC_API_KEY=... make integration_tests
   ```

   The integration tests run against production unless `BOXD_BASE_URL` is
   set. They include the `langchain-tests` standard sandbox suite; 7 tests are
   `xfail(strict=True)` because of version skew between `langchain-tests` and
   `deepagents`. If one of them starts passing, the strict marker fails the
   run: remove that marker.

2. Run the three examples with a real model (`python examples/<name>.py`), and
   `examples/persistent_workspace.py --delete` at the end.

3. Build and check the package:

   ```sh
   rm -rf dist && uv build
   uvx twine check dist/*
   unzip -l dist/*.whl
   ```

   The wheel holds `boxd_langchain/` with `py.typed`, and the entry point
   `deepagents_code.sandbox_providers: boxd`.

4. Install the wheel in a clean virtualenv and run the README quick start
   from it, so you test what users get. Check `dcode install boxd-langchain
   --package` once the package is on PyPI.

5. Set the date in `CHANGELOG.md`. The first release ships `0.1.0` as it
   stands in `pyproject.toml`; bump `version` there for every later release.

6. The banner at the top of the README to
   its full `raw.githubusercontent.com` URL: PyPI doesn't show relative images.

7. Publish, then tag:

   ```sh
   uv publish
   git tag v0.1.0 && git push --tags
   ```

8. Submit the package through LangChain's integration listing form.
