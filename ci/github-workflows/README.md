# GitHub Actions workflows (to move)

`ci.yml` (unit tests on Python 3.11 to 3.14, lint and type checks, package
build) and `integration.yml` (live tests, manual trigger, needs the
`BOXD_API_KEY` and `ANTHROPIC_API_KEY` secrets) belong in `.github/workflows/`.

They are here because the token that pushed this branch could not write
workflow files. Move them with:

```bash
git mv ci/github-workflows/ci.yml ci/github-workflows/integration.yml .github/workflows/
git rm ci/github-workflows/README.md
```
