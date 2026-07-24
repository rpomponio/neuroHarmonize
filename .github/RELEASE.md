# Release Process for neuroHarmonize

This document describes how to release a new version of neuroHarmonize to PyPI using GitHub Actions.

## Prerequisites (One-Time Setup)

### 1. PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `neuroHarmonize-github-actions`
4. Scope: "Project: neuroHarmonize"
5. Copy the token immediately

### 2. Add Token to GitHub Secrets

1. Go to https://github.com/rpomponio/neuroHarmonize/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Paste the PyPI token
5. Click "Add secret"

### 3. (Optional) TestPyPI Setup

For testing releases before production:

1. Create account: https://test.pypi.org/account/register/
2. Create API token: https://test.pypi.org/manage/account/token/
3. Add to GitHub secrets as `TEST_PYPI_API_TOKEN`

## Release Process

### Step 1: Prepare the Release

1. **Update version number** in:
   - `setup.py`
   - `pyproject.toml`
   - `README.md`

2. **Update CHANGELOG.md**:
   - Document all changes
   - Include breaking changes
   - Add migration guide if needed

3. **Run tests**:
   ```bash
   pytest tests/ -m "not slow"
   ```

4. **Verify package builds locally**:
   ```bash
   rm -rf dist/ build/
   python -m build
   twine check dist/*
   ```

5. **Commit and push changes**:
   ```bash
   git add .
   git commit -m "Prepare for vX.Y.Z release"
   git push origin master
   ```

### Step 2: Create and Push Git Tag

```bash
# Create annotated tag
git tag -a vX.Y.Z -m "Release vX.Y.Z - Brief description"

# Push tag to GitHub
git push origin vX.Y.Z
```

### Step 3: Create GitHub Release

1. Go to: https://github.com/rpomponio/neuroHarmonize/releases/new
2. Choose tag: `vX.Y.Z`
3. Release title: `vX.Y.Z - Brief Title`
4. Description: Copy from CHANGELOG.md or use release template
5. For testing: Check "This is a pre-release" (triggers TestPyPI workflow)
6. For production: Click "Publish release" (triggers PyPI workflow)

### Step 4: Monitor Workflow

1. Go to: https://github.com/rpomponio/neuroHarmonize/actions
2. Watch the "Publish to PyPI" workflow
3. Check for any errors in the logs

### Step 5: Verify Release

```bash
# Wait a few minutes for PyPI to process

# Install from PyPI
pip install --upgrade neuroHarmonize==X.Y.Z

# Verify import
python -c "from neuroHarmonize import harmonizationLearn; print('Success!')"

# Run quick test
python -c "
from neuroHarmonize import harmonizationLearn
import numpy as np
import pandas as pd
np.random.seed(42)
data = np.random.randn(10, 5)
covars = pd.DataFrame({'SITE': ['A']*5 + ['B']*5, 'age': np.random.uniform(20, 80, 10)})
model, _ = harmonizationLearn(data, covars, seed=42)
print('Quick test passed!')
"
```

### Step 6: Announce Release

- Update documentation sites
- Notify users via mailing lists/forums
- Post on social media if applicable
- For breaking changes: Send direct notifications to known users

## Testing a Release (TestPyPI)

Before creating a production release, test with a pre-release:

1. **Create pre-release version**:
   ```bash
   # Use a pre-release version like v2.5.0-rc1
   git tag -a v2.5.0-rc1 -m "Release candidate 1"
   git push origin v2.5.0-rc1
   ```

2. **Create GitHub pre-release**:
   - Mark as "This is a pre-release"
   - This triggers the TestPyPI workflow

3. **Test installation from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               neuroHarmonize==2.5.0rc1
   ```

4. **If successful**, create the production release

## Workflow Files

- **`.github/workflows/publish.yml`**: Publishes to PyPI on release
- **`.github/workflows/publish-test.yml`**: Publishes to TestPyPI on pre-release
- **`.github/workflows/test.yml`**: Runs tests on push/PR

## Troubleshooting

### Workflow doesn't trigger
- Ensure release was "published" (not just saved as draft)
- Check workflow file is on the default branch
- Verify the `on.release.types` matches your action

### Authentication fails
- Check `PYPI_API_TOKEN` secret is set correctly
- Verify token hasn't expired
- Ensure token has upload permissions for neuroHarmonize

### Build fails
- Run `python -m build` locally to reproduce
- Check all required files are committed
- Verify version numbers are consistent

### "File already exists" on PyPI
- Cannot overwrite existing versions
- Must bump version number
- Delete the GitHub release/tag and try again with new version

### Package doesn't appear on PyPI
- Wait 5-10 minutes for PyPI to process
- Check https://pypi.org/project/neuroHarmonize/
- Verify workflow completed successfully

## Manual Release (Fallback)

If GitHub Actions fails, you can release manually:

```bash
# Build
rm -rf dist/ build/
python -m build

# Check
twine check dist/*

# Upload
twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking changes, incompatible API changes
- **Minor (x.Y.0)**: New features, backward-compatible
- **Patch (x.y.Z)**: Bug fixes, backward-compatible

## Checklist Template

Use this checklist for each release:

- [ ] Version updated in setup.py, pyproject.toml, README.md
- [ ] CHANGELOG.md updated with all changes
- [ ] Tests pass locally
- [ ] Package builds without errors
- [ ] Changes committed and pushed
- [ ] Git tag created and pushed
- [ ] GitHub release created
- [ ] Workflow completed successfully
- [ ] Package available on PyPI
- [ ] Installation verified
- [ ] Release announced
