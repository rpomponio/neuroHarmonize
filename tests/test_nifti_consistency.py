import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import pytest
from neuroHarmonize import harmonizationLearn, harmonizationApply
from neuroHarmonize.harmonizationApply import applyModelOne
from neuroHarmonize.harmonizationNIFTI import applyModelNIFTIs, flattenNIFTIs, createMaskNIFTI


@pytest.fixture
def nifti_dataset(tmp_path):
    """Create a synthetic NIFTI dataset with known float values."""
    np.random.seed(99)
    n_images = 6
    shape = (10, 10, 10)
    affine = np.eye(4)

    # create images with site effects baked in
    paths_list = []
    paths_new_list = []
    for i in range(n_images):
        vol = np.random.randn(*shape).astype(np.float64) * 10 + 100
        img = nib.Nifti1Image(vol, affine)
        p = str(tmp_path / f'sub{i}.nii.gz')
        p_new = str(tmp_path / f'sub{i}_harmonized.nii.gz')
        img.to_filename(p)
        paths_list.append(p)
        paths_new_list.append(p_new)

    # create a mask (central 8x8x8 cube)
    mask_vol = np.zeros(shape, dtype=np.int8)
    mask_vol[1:9, 1:9, 1:9] = 1
    mask_path = str(tmp_path / 'mask.nii.gz')
    mask_img = nib.Nifti1Image(mask_vol, affine)
    mask_img.to_filename(mask_path)

    paths = pd.DataFrame({'PATH': paths_list, 'PATH_NEW': paths_new_list})
    covars = pd.DataFrame({
        'SITE': ['siteA'] * 3 + ['siteB'] * 3,
        'age': [30, 50, 70, 35, 55, 75],
        'sex': [0, 1, 0, 1, 0, 1],
    })

    return paths, covars, mask_path, tmp_path


class TestNIFTITabularConsistency:
    """Verify that applyModelNIFTIs matches the tabular path numerically."""

    def test_nifti_output_matches_tabular(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset

        # flatten and train
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42)

        # tabular apply
        tabular_result = harmonizationApply(nifti_array, covars, model)

        # NIFTI apply
        applyModelNIFTIs(covars, model, paths, mask_path)

        # load mask for extraction
        mask_data = (nib.load(mask_path).get_fdata().round().astype(int) == 1)

        for i in range(paths.shape[0]):
            nifti_out = nib.load(paths.PATH_NEW.iloc[i]).get_fdata()
            voxels = nifti_out[mask_data]
            np.testing.assert_allclose(
                voxels, tabular_result[i, :], rtol=1e-5, atol=1e-5,
                err_msg=f'NIFTI output for image {i} differs from tabular')

    def test_output_is_float32_by_default(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42)
        applyModelNIFTIs(covars, model, paths, mask_path)

        img = nib.load(paths.PATH_NEW.iloc[0])
        assert img.get_data_dtype() == np.float32

    def test_output_dtype_int16_rounds(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42)
        applyModelNIFTIs(covars, model, paths, mask_path, output_dtype=np.int16)

        img = nib.load(paths.PATH_NEW.iloc[0])
        assert img.get_data_dtype() == np.int16
        data = img.get_fdata()
        # all values should be integers (within float tolerance)
        np.testing.assert_allclose(data, np.round(data), atol=1e-7)


class TestRefLevel:
    """Verify ref_level is honored in applyModelOne."""

    def test_ref_batch_sample_unchanged(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42,
                                      ref_batch='siteA')

        # single sample from the reference batch
        data_i = nifti_array[[0], :]
        covars_i = covars.iloc[[0], :]
        result = applyModelOne(data_i, covars_i, model)
        np.testing.assert_allclose(
            result, data_i, rtol=1e-10, atol=1e-10,
            err_msg='Reference batch sample should be returned unchanged')

    def test_ref_batch_nifti_matches_tabular(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42,
                                      ref_batch='siteA')

        tabular_result = harmonizationApply(nifti_array, covars, model)
        applyModelNIFTIs(covars, model, paths, mask_path)

        mask_data = (nib.load(mask_path).get_fdata().round().astype(int) == 1)
        for i in range(paths.shape[0]):
            nifti_out = nib.load(paths.PATH_NEW.iloc[i]).get_fdata()
            voxels = nifti_out[mask_data]
            np.testing.assert_allclose(
                voxels, tabular_result[i, :], rtol=1e-5, atol=1e-5,
                err_msg=f'ref_level: NIFTI differs from tabular for image {i}')


class TestAffineValidation:
    """Verify that affine/shape mismatches are caught."""

    def test_shape_mismatch_raises(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42)

        # create an image with wrong shape
        bad_vol = np.zeros((5, 5, 5))
        bad_path = str(tmp_path / 'bad_shape.nii.gz')
        nib.Nifti1Image(bad_vol, np.eye(4)).to_filename(bad_path)

        bad_paths = pd.DataFrame({
            'PATH': [bad_path],
            'PATH_NEW': [str(tmp_path / 'bad_out.nii.gz')]
        })
        bad_covars = covars.iloc[[0], :]

        with pytest.raises(ValueError, match='identical dimensions'):
            applyModelNIFTIs(bad_covars, model, bad_paths, mask_path)

    def test_affine_mismatch_warns(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42)

        # create an image with different affine but same shape
        vol = np.random.randn(10, 10, 10).astype(np.float64) * 10 + 100
        bad_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        bad_path = str(tmp_path / 'bad_affine.nii.gz')
        nib.Nifti1Image(vol, bad_affine).to_filename(bad_path)

        bad_paths = pd.DataFrame({
            'PATH': [bad_path],
            'PATH_NEW': [str(tmp_path / 'bad_affine_out.nii.gz')]
        })
        bad_covars = covars.iloc[[0], :]

        with pytest.warns(UserWarning, match='different affine'):
            applyModelNIFTIs(bad_covars, model, bad_paths, mask_path)


class TestSmoothingGuard:
    """Verify that applyModelOne raises for smooth models."""

    def test_smooth_model_raises(self, nifti_dataset):
        paths, covars, mask_path, tmp_path = nifti_dataset
        nifti_array = flattenNIFTIs(paths, mask_path,
                                    output_path=str(tmp_path / 'flat.npy'))
        model, _ = harmonizationLearn(nifti_array, covars, seed=42)

        # patch model to simulate smoothing
        model['smooth_model']['perform_smoothing'] = True

        data_i = nifti_array[[0], :]
        covars_i = covars.iloc[[0], :]

        with pytest.raises(NotImplementedError, match='smooth_terms'):
            applyModelOne(data_i, covars_i, model)
