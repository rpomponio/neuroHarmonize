import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data():
    """
    Create synthetic neuroimaging data for testing.
    Returns data array with shape (n_samples, n_features).
    """
    np.random.seed(42)
    n_samples = 50
    n_features = 100

    # Simulate multi-site data with site effects
    data = np.random.randn(n_samples, n_features) * 10 + 100

    return data


@pytest.fixture
def synthetic_covars():
    """
    Create synthetic covariate data for testing.
    Includes SITE labels and continuous covariates (age, sex).
    """
    np.random.seed(42)
    n_samples = 50

    # Create 3 sites with different sample sizes
    sites = ['site1'] * 20 + ['site2'] * 20 + ['site3'] * 10

    # Create age and sex covariates
    ages = np.random.uniform(20, 80, n_samples)
    sex = np.random.randint(0, 2, n_samples)

    covars = pd.DataFrame({
        'SITE': sites,
        'age': ages,
        'sex': sex
    })

    return covars


@pytest.fixture
def trained_model(synthetic_data, synthetic_covars):
    """
    Create a trained harmonization model for testing.
    """
    from neuroHarmonize import harmonizationLearn

    model, _ = harmonizationLearn(synthetic_data, synthetic_covars, seed=42)

    return model


@pytest.fixture
def small_nifti_data():
    """
    Create small synthetic data that simulates flattened NIFTI data.
    Returns single sample with same number of features as synthetic_data.
    """
    np.random.seed(42)
    n_voxels = 100  # Same as synthetic_data n_features

    data = np.random.randn(1, n_voxels) * 10 + 100

    return data
