# neuroHarmonize Test Suite

This directory contains the test suite for the neuroHarmonize package.

## Running Tests

### Install test dependencies

```bash
pip install -e ".[dev]"
```

### Run all tests

```bash
pytest tests/
```

### Run tests with coverage

```bash
pip install pytest-cov
pytest tests/ --cov=neuroHarmonize --cov-report=html
```

### Run only fast tests (skip slow tests marked with @pytest.mark.slow)

```bash
pytest tests/ -m "not slow"
```

## Test Organization

### `test_harmonization_consistency.py`
Tests for the critical bug fix where `applyModelOne` was missing `mod_mean` (covariate effects) during de-standardization. These tests verify:

- `applyModelOne` produces identical results to `harmonizationApply` for single samples
- Covariate effects are correctly included in harmonized values
- NIFTI workflow produces consistent results with tabular workflow
- Handling of samples from new (unseen) sites

### `test_determinism.py`
Tests for the random seed bug fix. These tests verify:

- Same seed produces identical results across multiple runs
- Random seed is actually set when provided (fixes inverted logic bug)
- Function works correctly with and without empirical Bayes
- Reproducibility across different configurations

### `conftest.py`
Shared pytest fixtures for generating synthetic test data:

- `synthetic_data`: Multi-site neuroimaging-style data
- `synthetic_covars`: Site labels and covariates (age, sex)
- `trained_model`: Pre-trained harmonization model
- `small_nifti_data`: Simulated flattened NIFTI data

## Key Tests for Bug Fixes

### Bug Fix #1: Missing `mod_mean` in `applyModelOne`
**Test**: `test_applyModelOne_matches_harmonizationApply_single_sample`

This test directly verifies that the bug fix works by comparing `applyModelOne` (used for NIFTI harmonization) with `harmonizationApply` (tabular harmonization) on the same data. Before the fix, these would produce different results because `mod_mean` was missing.

### Bug Fix #2: Inverted random seed logic
**Test**: `test_same_seed_produces_identical_results`

This test verifies that providing a seed makes the function deterministic. Before the fix, the logic was inverted and the seed was never actually set.

## Test Coverage

The test suite focuses on:

1. **Regression tests** for the two critical bugs fixed in Phase 1
2. **Consistency tests** between different harmonization pathways
3. **Determinism tests** for reproducibility
4. **Edge case handling** (new sites, different covariate values)

## Adding New Tests

When adding new tests:

1. Use existing fixtures from `conftest.py` when possible
2. Mark slow tests with `@pytest.mark.slow`
3. Use descriptive test names that explain what is being tested
4. Include docstrings explaining the purpose and what the test verifies
