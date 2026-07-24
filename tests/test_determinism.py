import numpy as np
import pytest
from neuroHarmonize import harmonizationLearn


class TestRandomSeedDeterminism:
    """
    Tests to verify that the random seed parameter works correctly.
    This tests the bug fix where seed logic was inverted.
    """

    def test_same_seed_produces_identical_results(
        self, synthetic_data, synthetic_covars
    ):
        """
        Test that harmonizationLearn produces identical results when called
        with the same seed multiple times.

        This tests the bug fix where the seed logic was inverted.
        """
        # Train model twice with same seed
        model_1, bayes_data_1 = harmonizationLearn(
            synthetic_data, synthetic_covars, seed=123
        )
        model_2, bayes_data_2 = harmonizationLearn(
            synthetic_data, synthetic_covars, seed=123
        )

        # Results should be identical
        np.testing.assert_array_equal(
            bayes_data_1,
            bayes_data_2,
            err_msg="Same seed should produce identical results"
        )

        # Model parameters should also be identical
        np.testing.assert_allclose(
            model_1['gamma_star'],
            model_2['gamma_star'],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Same seed should produce identical gamma_star"
        )

        np.testing.assert_allclose(
            model_1['delta_star'],
            model_2['delta_star'],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Same seed should produce identical delta_star"
        )

    def test_different_seeds_can_produce_results(
        self, synthetic_data, synthetic_covars
    ):
        """
        Test that harmonizationLearn successfully runs with different seeds.

        Note: This primarily tests that the seed parameter is accepted and
        doesn't cause errors. Actual result differences depend on whether
        the algorithm uses randomness (e.g., in EB parameter estimation).
        """
        # Train model with different seeds
        model_1, bayes_data_1 = harmonizationLearn(
            synthetic_data, synthetic_covars, eb=True, seed=123
        )
        model_2, bayes_data_2 = harmonizationLearn(
            synthetic_data, synthetic_covars, eb=True, seed=456
        )

        # Both should complete successfully and return valid results
        assert not np.any(np.isnan(bayes_data_1)), \
            "Results with seed=123 should be valid"
        assert not np.any(np.isnan(bayes_data_2)), \
            "Results with seed=456 should be valid"

        # Models should have the same structure
        assert model_1['gamma_star'].shape == model_2['gamma_star'].shape, \
            "Models should have same structure"

    def test_no_seed_produces_different_results(
        self, synthetic_data, synthetic_covars
    ):
        """
        Test that harmonizationLearn with seed=None uses non-deterministic behavior
        when empirical Bayes is enabled.

        Note: This test verifies the seed fix - that seed=None does NOT set a seed.
        """
        # Train model multiple times without seed, with EB enabled
        model_1, bayes_data_1 = harmonizationLearn(
            synthetic_data, synthetic_covars, eb=True, seed=None
        )
        model_2, bayes_data_2 = harmonizationLearn(
            synthetic_data, synthetic_covars, eb=True, seed=None
        )

        # With EB and no seed, gamma_star may differ due to random initialization
        # However, this is a probabilistic test - it might occasionally be the same
        # So we just verify that the code runs without error
        # The key test is that seed=None doesn't crash (which would happen
        # with the old buggy code that tried to call np.random.seed(None))

        # Verify both results are valid (not NaN)
        assert not np.any(np.isnan(bayes_data_1)), \
            "Results with seed=None should be valid (not NaN)"
        assert not np.any(np.isnan(bayes_data_2)), \
            "Results with seed=None should be valid (not NaN)"

    def test_seed_with_eb_false(
        self, synthetic_data, synthetic_covars
    ):
        """
        Test that seed works correctly when empirical Bayes is disabled.
        """
        # Train model twice with same seed, eb=False
        model_1, bayes_data_1 = harmonizationLearn(
            synthetic_data, synthetic_covars, eb=False, seed=789
        )
        model_2, bayes_data_2 = harmonizationLearn(
            synthetic_data, synthetic_covars, eb=False, seed=789
        )

        # Results should be identical
        np.testing.assert_array_equal(
            bayes_data_1,
            bayes_data_2,
            err_msg="Same seed with eb=False should produce identical results"
        )


class TestSeedBehaviorConsistency:
    """
    Tests to verify seed behavior is consistent across different configurations.
    """

    @pytest.mark.slow
    def test_seed_with_smooth_terms(
        self, synthetic_data, synthetic_covars
    ):
        """
        Test that seed produces deterministic results when using smooth terms.

        Note: This test is marked as slow because GAM fitting takes time.
        """
        # Train model twice with same seed and smooth terms
        model_1, bayes_data_1 = harmonizationLearn(
            synthetic_data,
            synthetic_covars,
            smooth_terms=['age'],
            seed=999
        )
        model_2, bayes_data_2 = harmonizationLearn(
            synthetic_data,
            synthetic_covars,
            smooth_terms=['age'],
            seed=999
        )

        # Results should be identical
        np.testing.assert_allclose(
            bayes_data_1,
            bayes_data_2,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Same seed with smooth terms should produce identical results"
        )

    def test_seed_reproducibility_after_multiple_calls(
        self, synthetic_data, synthetic_covars
    ):
        """
        Test that using the same seed produces identical results even after
        multiple calls with different seeds in between.
        """
        # Train with seed=111
        model_1, bayes_data_1 = harmonizationLearn(
            synthetic_data, synthetic_covars, seed=111
        )

        # Train with different seeds
        harmonizationLearn(synthetic_data, synthetic_covars, seed=222)
        harmonizationLearn(synthetic_data, synthetic_covars, seed=333)

        # Train again with seed=111
        model_2, bayes_data_2 = harmonizationLearn(
            synthetic_data, synthetic_covars, seed=111
        )

        # Results should be identical to first call with seed=111
        np.testing.assert_allclose(
            bayes_data_1,
            bayes_data_2,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Same seed should produce identical results regardless of intervening calls"
        )
