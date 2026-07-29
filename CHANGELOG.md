# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.1] - 2026-07-29

### Fixed

- **NIFTI output now preserves floating-point precision** ([#51](https://github.com/rpomponio/neuroHarmonize/issues/51)): Removed the `np.round().astype(np.int16)` cast in `applyModelNIFTIs` that was destroying fractional precision. Output is now float32 by default, matching the tabular path exactly. An `output_dtype` parameter is available for legacy int16 behavior.

- **Reference batch samples no longer incorrectly adjusted in NIFTI path** ([#51](https://github.com/rpomponio/neuroHarmonize/issues/51)): `applyModelOne` now honors `ref_level` by returning original data for samples belonging to the reference batch, consistent with `adjust_data_final` in the tabular path.

- **Mask saved as int8 instead of int16**: `createMaskNIFTI` now writes with a clean header (int8 dtype, no slope/intercept) to avoid scaling artifacts.

### Added

- **Affine and shape validation in NIFTI functions**: `applyModelNIFTIs` and `flattenNIFTIs` now raise `ValueError` on dimension mismatch and emit a `UserWarning` when an image's affine does not match the mask.

- **Smoothing guard in `applyModelOne`**: Raises `NotImplementedError` if the model was trained with `smooth_terms`, directing users to the batch path (`flattenNIFTIs` + `harmonizationApply`).

- **`output_dtype` parameter for `applyModelNIFTIs`**: Defaults to `np.float32`; pass `np.int16` to restore the legacy rounding behavior.

- **NIFTI consistency test suite** (`tests/test_nifti_consistency.py`): 8 tests covering tabular/NIFTI equivalence, ref_level handling, affine validation, and the smoothing guard.

---

## [2.5.0] - 2026-07-24

### ⚠️ BREAKING CHANGES

**CRITICAL: All NIFTI images harmonized with previous versions must be re-harmonized.**

The bug fix in `applyModelOne` means that all NIFTI images harmonized using `applyModelNIFTIs` in versions prior to 2.5.0 are **incorrect**. They are missing covariate effects (age, sex, etc.) that should have been preserved during harmonization. Please re-harmonize all NIFTI data after updating to this version.

- **Python 3.6-3.8 no longer supported**. Minimum Python version is now 3.9.
- **NIFTI harmonization results will differ** from previous versions (this is the fix).
- **NIFTI reference output file renamed** from `_stand_mean.nii.gz` to `_reference_mean.nii.gz` and now includes complete reference (stand_mean + mod_mean).

### Fixed

- **Critical bug in `applyModelOne`**: Fixed missing covariate effects (`mod_mean`) during de-standardization ([#issue-number])
  - NIFTI harmonization via `applyModelNIFTIs` was incorrectly stripping biological covariate adjustments (age, sex, etc.)
  - `applyModelOne` now produces identical results to `harmonizationApply` for single samples
  - Root cause: During 2024 refactoring (Issues #47-48), the manual de-standardization in `applyModelOne` was only partially updated when `grand_mean` was split into `stand_mean` and `mod_mean`
  - Files affected: `neuroHarmonize/harmonizationApply.py`, `neuroHarmonize/harmonizationNIFTI.py`

- **Random seed logic was inverted in `harmonizationLearn`**: Fixed backwards if/else logic ([#issue-number])
  - When seed was provided, it was ignored; when not provided, attempted to set `seed=None`
  - Now correctly sets the random seed when provided
  - File affected: `neuroHarmonize/harmonizationLearn.py`

### Added

- **Comprehensive test suite** with 11 passing tests
  - Regression tests verifying both critical bug fixes
  - Tests for consistency between `applyModelOne` and `harmonizationApply`
  - Tests for random seed determinism
  - Tests for edge cases (new sites, different covariates)
  - Test files: `tests/test_harmonization_consistency.py`, `tests/test_determinism.py`, `tests/conftest.py`

- **Modern build configuration** (PEP 621)
  - Full project metadata in `pyproject.toml`
  - Dependency version constraints to prevent breaking changes
  - PyPI classifiers and project URLs
  - Development dependencies for testing

- **GitHub Actions CI workflow**
  - Test matrix: Python 3.9, 3.10, 3.11, 3.12 on Ubuntu and macOS
  - Automated testing on pull requests and pushes
  - Coverage reporting

### Changed

- **Python requirement updated from 3.6+ to 3.9+**
  - Python 3.6-3.8 are end-of-life
  - Python 3.9 is supported through October 2025
  - Updated in `setup.py`, `pyproject.toml`, and `README.md`

- **Updated deprecated pandas API**
  - Replaced `.values` with `.to_numpy()` to prevent FutureWarning
  - Files: `neuroHarmonize/harmonizationApply.py`, `neuroHarmonize/harmonizationNIFTI.py`

- **Dependency version constraints added**
  - `numpy>=1.19.0,<2.0.0`
  - `pandas>=1.1.0,<3.0.0`
  - `nibabel>=3.0.0,<6.0.0`
  - `statsmodels>=0.12.0,<1.0.0`
  - `neuroCombat==0.2.12` (unchanged)

- **Build configuration consolidated in `pyproject.toml`**
  - Moved pytest configuration from `pytest.ini` to `pyproject.toml`
  - Added full PEP 621 metadata
  - `setup.py` kept for backward compatibility with enhanced metadata

- **Updated `.gitignore`** with modern Python patterns
  - Added cache directories (`.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`)
  - Added virtual environment and coverage directories
  - Added IDE-specific files

### Removed

- **Removed conflicting configuration files**
  - Deleted `.setup.cfg` (conflicted with flat package structure)
  - Deleted `pytest.ini` (configuration moved to `pyproject.toml`)

### Migration Guide

#### For Users with Previously Harmonized NIFTI Data

**Action Required**: Re-harmonize all NIFTI images that were processed with `applyModelNIFTIs` in versions < 2.5.0.

1. Update to version 2.5.0:
   ```bash
   pip install --upgrade neuroHarmonize
   ```

2. Re-run your NIFTI harmonization pipeline:
   ```python
   from neuroHarmonize import harmonizationLearn, applyModelNIFTIs
   
   # Re-train your model (if needed)
   model, data_adj = harmonizationLearn(data, covars)
   
   # Re-harmonize all NIFTI images
   applyModelNIFTIs(covars, model, paths, mask_path)
   ```

3. Note: The reference NIFTI file output name has changed from `_stand_mean.nii.gz` to `_reference_mean.nii.gz`

#### For Users on Python 3.6-3.8

Version 2.5.0 requires Python 3.9 or higher. To continue using neuroHarmonize:

1. Upgrade to Python 3.9 or higher, OR
2. Pin to the last compatible version: `pip install neuroHarmonize==2.4.5`

Note: Python 3.6-3.8 reached end-of-life and are no longer receiving security updates.

### Technical Details

#### Why NIFTI Results Were Incorrect

The ComBat harmonization algorithm is designed to:
1. Remove site effects (scanner/center differences)
2. **Preserve covariate effects** (age, sex, diagnosis, etc.)

The bug in `applyModelOne` caused step 2 to fail for NIFTI images. The function was missing `+ mod_mean` in the final de-standardization step, where `mod_mean` represents the covariate effects. This meant that harmonized NIFTI images were missing important biological adjustments that the tabular harmonization pathway correctly preserved.

#### Verification

To verify the fix works correctly:
```python
# Compare single-sample and batch harmonization
from neuroHarmonize import harmonizationApply
from neuroHarmonize.harmonizationApply import applyModelOne

# Single sample
result_single = applyModelOne(data[[0], :], covars.iloc[[0], :], model)

# Batch (extract first sample)
result_batch = harmonizationApply(data, covars, model)

# Should now be identical
assert np.allclose(result_single, result_batch[0, :])
```

---

## [2.4.5] and earlier

For changes prior to 2.5.0, please see the [commit history](https://github.com/rpomponio/neuroHarmonize/commits/master).
