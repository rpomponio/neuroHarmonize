# PyPI Release Instructions

> **Note**: These instructions are **OUTDATED** (last updated 2020). The package now uses GitHub Actions for automated releases.

## Current Release Process (Automated)

Releases are now automated via GitHub Actions. See [.github/RELEASE.md](.github/RELEASE.md) for complete instructions.

### Quick Start

1. Update version in `setup.py`, `pyproject.toml`, and `README.md`
2. Update `CHANGELOG.md`
3. Commit and push changes
4. Create and push git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Create GitHub release at: https://github.com/rpomponio/neuroHarmonize/releases/new
6. GitHub Actions automatically builds and publishes to PyPI

### Prerequisites (One-Time)

- PyPI API token stored in GitHub Secrets as `PYPI_API_TOKEN`
- See [.github/RELEASE.md](.github/RELEASE.md) for setup instructions

---

## Old Manual Process (Pre-2024)

<details>
<summary>Click to expand legacy instructions (for reference only)</summary>

These instructions are preserved for historical reference but are no longer used.

See the following link for detailed instructions, though note that several steps are different:
<https://packaging.python.org/tutorials/packaging-projects/#uploading-your-project-to-pypi>

Follow the instructions below:

1. Delete old distributions in `dist/` and `src/`.
2. Upgrade packages `pip`, `build`, and `twine`.
3. Copy contents of `neuroHarmonize/` to `src/neuroHarmonize/`.

      ```cp neuroHarmonize/* src/neuroHarmonize/```

4. Move .setup.cfg to setup.cfg and move setup.py to .setup.py (this overrides setup.py).

      ```mv .setup.cfg setup.cfg```
      ```mv setup.py .setup.py```

5. Update version number in `setup.cfg`.

6. Build the package.

      ```python3 -m build ```

7. Upload the distribution. Note the username will be `__token__` and the password will be the API token from <https://pypi.org>.

      ```python3 -m twine upload dist/*```

8. Finally (optional) discard all local changes in the repository to revert to developer state.

      ```mv setup.cfg .setup.cfg```
      ```mv .setup.py setup.py```

**Note**: This process is outdated. The package no longer uses `.setup.cfg` or `src/` layout. Use the GitHub Actions workflow instead.

</details>

---

## Manual Release (Emergency Fallback)

If GitHub Actions is unavailable, you can release manually using the modern build process:

```bash
# Clean build artifacts
rm -rf dist/ build/ neuroHarmonize.egg-info/

# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token

For detailed instructions, see [.github/RELEASE.md](.github/RELEASE.md)
