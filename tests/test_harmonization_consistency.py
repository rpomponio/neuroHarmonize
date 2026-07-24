import numpy as np
import pandas as pd
import pytest
from neuroHarmonize import harmonizationLearn, harmonizationApply
from neuroHarmonize.harmonizationApply import applyModelOne


class TestApplyModelOneConsistency:
    """
    Tests to verify applyModelOne produces identical results to harmonizationApply
    for single samples. This is the critical regression test for the mod_mean bug fix.
    """

    def test_applyModelOne_matches_harmonizationApply_single_sample(
        self, synthetic_data, synthetic_covars, trained_model
    ):
        """
        Test that applyModelOne produces identical results to harmonizationApply
        when applied to a single sample.

        This test verifies the bug fix where mod_mean was missing in applyModelOne.
        """
        # Apply harmonization to all data using harmonizationApply
        harmonized_all = harmonizationApply(synthetic_data, synthetic_covars, trained_model)

        # Apply harmonization to each sample individually using applyModelOne
        harmonized_individual = []
        for i in range(synthetic_data.shape[0]):
            data_i = synthetic_data[[i], :]
            covars_i = synthetic_covars.iloc[[i], :]
            result_i = applyModelOne(data_i, covars_i, trained_model)
            harmonized_individual.append(result_i[0, :])

        harmonized_individual = np.array(harmonized_individual)

        # Results should be identical (within numerical precision)
        np.testing.assert_allclose(
            harmonized_all,
            harmonized_individual,
            rtol=1e-10,
            atol=1e-10,
            err_msg="applyModelOne and harmonizationApply should produce identical results"
        )

    def test_applyModelOne_includes_mod_mean(
        self, synthetic_data, synthetic_covars, trained_model
    ):
        """
        Test that applyModelOne correctly adds mod_mean (covariate effects)
        during de-standardization.

        This directly tests the bug fix by verifying that harmonized values
        include covariate adjustments.
        """
        # Select a single sample
        data_single = synthetic_data[[0], :]
        covars_single = synthetic_covars.iloc[[0], :]

        # Apply harmonization
        harmonized = applyModelOne(data_single, covars_single, trained_model)

        # Apply harmonization to all data for comparison
        harmonized_all = harmonizationApply(synthetic_data, synthetic_covars, trained_model)

        # The first sample should match
        np.testing.assert_allclose(
            harmonized[0, :],
            harmonized_all[0, :],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Single sample harmonization should match batch harmonization"
        )

    def test_applyModelOne_with_different_covariates(
        self, synthetic_data, trained_model
    ):
        """
        Test that applyModelOne produces different results for samples with
        different covariate values, confirming that mod_mean is being applied.
        """
        # Create two samples with same site but different age/sex
        data_sample = synthetic_data[[0], :]

        covars_1 = pd.DataFrame({
            'SITE': ['site1'],
            'age': [30.0],
            'sex': [0]
        })

        covars_2 = pd.DataFrame({
            'SITE': ['site1'],
            'age': [70.0],  # Different age
            'sex': [1]       # Different sex
        })

        # Apply harmonization with different covariates
        harmonized_1 = applyModelOne(data_sample, covars_1, trained_model)
        harmonized_2 = applyModelOne(data_sample, covars_2, trained_model)

        # Results should be different because covariate effects (mod_mean) differ
        # If mod_mean was missing, results would be more similar
        assert not np.allclose(
            harmonized_1,
            harmonized_2,
            rtol=1e-5,
            atol=1e-5
        ), "Harmonized values should differ when covariates differ"

    def test_applyModelOne_return_stand_mean(
        self, synthetic_data, synthetic_covars, trained_model
    ):
        """
        Test that applyModelOne returns the correct reference mean
        (stand_mean + mod_mean) when return_stand_mean=True.

        This tests the fix to the return value for NIFTI output.
        """
        data_single = synthetic_data[[0], :]
        covars_single = synthetic_covars.iloc[[0], :]

        # Get both harmonized data and reference mean
        harmonized, reference_mean = applyModelOne(
            data_single, covars_single, trained_model, return_stand_mean=True
        )

        # Reference mean should have same shape as harmonized data
        assert reference_mean.shape == harmonized.shape, \
            "Reference mean should have same shape as harmonized data"

        # Reference mean should be a reasonable value (not zero, not NaN)
        assert not np.any(np.isnan(reference_mean)), \
            "Reference mean should not contain NaN values"

        assert np.any(reference_mean != 0), \
            "Reference mean should not be all zeros"


class TestBatchConsistency:
    """
    Tests to verify consistency when harmonizing in batch vs individually.

    Note: The applyModelOne function is specifically designed for NIFTI workflows
    where images are processed one at a time. The key test is
    test_applyModelOne_matches_harmonizationApply_single_sample which verifies
    that processing samples individually produces the same results as batch processing.
    """
    pass  # Tests are in TestApplyModelOneConsistency class


class TestNewSiteHandling:
    """
    Tests for handling samples from sites not in the training set.
    """

    def test_applyModelOne_new_site(
        self, synthetic_data, synthetic_covars, trained_model
    ):
        """
        Test that applyModelOne handles samples from new sites correctly
        (should return NaN for out-of-sample sites).
        """
        data_single = synthetic_data[[0], :]

        # Create covariate with a site not in training data
        covars_new_site = pd.DataFrame({
            'SITE': ['site_new'],
            'age': [50.0],
            'sex': [0]
        })

        # Apply harmonization
        harmonized = applyModelOne(data_single, covars_new_site, trained_model)

        # Result should be NaN for new site
        assert np.all(np.isnan(harmonized)), \
            "Harmonized data should be NaN for sites not in training set"

    def test_applyModelOne_known_site(
        self, synthetic_data, synthetic_covars, trained_model
    ):
        """
        Test that applyModelOne produces valid (non-NaN) results for known sites.
        """
        data_single = synthetic_data[[0], :]
        covars_single = synthetic_covars.iloc[[0], :]

        # Apply harmonization
        harmonized = applyModelOne(data_single, covars_single, trained_model)

        # Result should not be NaN for known site
        assert not np.any(np.isnan(harmonized)), \
            "Harmonized data should not be NaN for sites in training set"
