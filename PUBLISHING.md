# Publishing opsaudit

The release workflow uses PyPI Trusted Publishing. It avoids storing a long-lived PyPI token in the repository.

## One-time PyPI setup

Create a pending publisher for both [TestPyPI](https://test.pypi.org) and [PyPI](https://pypi.org) with these values:

| setting | value |
| --- | --- |
| PyPI owner | `ram-polisetti` |
| Repository name | `opsaudit` |
| Workflow filename | `publishing.yml` |
| Environment | `testpypi` for TestPyPI; `pypi` for PyPI |

Use GitHub's environment protection rules for the `pypi` environment if production publication should require approval.

## Release procedure

1. Confirm `pytest --cov=src/opsaudit --cov-fail-under=80`, `python -m build`, and `twine check dist/*` are green.
2. Update `CHANGELOG.md` and `RELEASE_NOTES_v0.1.0.md`; commit the release-ready state.
3. Tag the exact commit as `v0.1.0` and push the tag.
4. Run the `publish` GitHub Actions workflow manually against `v0.1.0`, first with the `testpypi` target.
5. Create a fresh environment, install the TestPyPI package, and run the README workflow.
6. Run the same workflow with the `pypi` target after the TestPyPI check passes.
7. Create the corresponding GitHub Release and use `RELEASE_NOTES_v0.1.0.md` as its body.

PyPI releases are immutable. Do not reuse a version number after an upload; make a new version for a corrected production release.
