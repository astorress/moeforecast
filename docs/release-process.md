# Release process

The `release` branch is the testing lane. `master` (or `main`) is the stable
release lane. Ordinary commits do not publish to either package index.

## Development and TestPyPI

1. Create or update the `release` branch from `master`/`main`.
2. Push development commits to `release`.
3. GitHub Actions runs Ruff and the full test suite. Only if they pass, the
   workflow builds a unique development version (for example,
   `0.1.0.dev42`) and publishes it to TestPyPI.
4. Install the exact version shown in the workflow log to validate the
   distribution:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ --pre "moeforecast==0.1.0.dev42"
   ```

The workflow generates the `.devN` suffix from the GitHub Actions run number;
it changes the version only in the build checkout, not in the branch. This
avoids TestPyPI's rule that an uploaded version cannot be uploaded again.

## Stable PyPI release

1. After validating the release-branch build, set `__version__` in
   `src/moeforecast/__init__.py` to the next stable version, such as `0.1.0`.
2. Merge the tested code and version change from `release` into `master` or
   `main`.
3. The PyPI workflow runs Ruff and tests again. It publishes only if the
   package version is stable; development and pre-release versions are skipped.

Every stable release needs a new version. PyPI does not allow replacing an
already-published version, so a package-code change after a release must bump
the version before merging.

## One-time account setup

Configure a separate Trusted Publisher on each index:

| Index | Workflow | GitHub environment |
|---|---|---|
| TestPyPI | `publish-testpypi.yml` | `testpypi` |
| PyPI | `publish-pypi.yml` | `pypi` |

Use owner `astorress` and repository `moeforecast` for both. In GitHub, restrict
the `testpypi` environment to the `release` branch and the `pypi` environment
to `master`/`main`. The PyPI package and all uploaded distributions are public,
even while the GitHub repository is private.
