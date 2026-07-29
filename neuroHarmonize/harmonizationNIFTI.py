import os
import warnings
import nibabel as nib
import numpy as np
import pandas as pd
from .harmonizationApply import applyModelOne

def createMaskNIFTI(paths, threshold=0.0, output_path='thresholded_mask.nii.gz'):
    """
    Creates a binary mask from a list of NIFTI images. Image intensities will be
    averaged, then thresholded across the entire dataset. Result will have the
    same affine matrix as the first image in the dataset.
    
    Arguments
    ---------
    paths : a pandas DataFrame
        must contain a single column "PATH" with file paths to NIFTIs
        dimensions must be identical for all images
    
    threshold : a float, default 0.0
        the threshold at which to binarize the mask
        average intensity must be greater than threshold to be included in mask
        
    output_path : str, default "thresholded_mask.nii.gz"
        the output file path, must include extension (.nii.gz)
        
    Returns
    -------
    nifti_avg : a numpy array
        array of average image intensities
        dimensions are identical to images in `paths`
    
    nifti_mask : a numpy array
        array of binarized mask (1=include, 0=exclude)
        dimensions are identical to images in `paths`

    affine : a numpy array
        affine matrix used to save mask
    
    """
    # count number of images
    n_images = paths.shape[0]
    # begin summing image intensities
    i = 0
    nifti_i = nib.load(paths.PATH[i])
    affine_0 = nifti_i.affine
    hdr_0 = nifti_i.header
    nifti_sum = nifti_i.get_fdata()
    # iterate over all images
    for i in range(0, n_images):
        nifti_i = nib.load(paths.PATH[i])
        nifti_sum += nifti_i.get_fdata()
        if (i==500):
            print('\n[neuroHarmonize]: loaded %d of %d images...' % (i, n_images))
    # compute average intensities
    nifti_avg = nifti_sum / n_images    
    # apply threshold
    nifti_avg[nifti_avg<threshold] = 0.0
    # create mask and save as NIFTI image
    nifti_mask = nifti_avg.copy()
    nifti_mask[nifti_mask>0.0] = 1.0
    mask_header = nib.Nifti1Header()
    mask_header.set_data_dtype(np.int8)
    img = nib.Nifti1Image(nifti_mask.astype(np.int8), affine_0, mask_header)
    img.to_filename(output_path)
    return nifti_avg, nifti_mask, affine_0, hdr_0

def flattenNIFTIs(paths, mask_path, output_path='flattened_NIFTI_array.npy'):
    """
    Flattens a dataset of NIFTI images to a 2D array.
        
    Arguments
    ---------
    paths : a pandas DataFrame
        must contain a single column "PATH" with file paths to NIFTIs
        dimensions must be identical for all images

    mask_path : a str
        file path to the mask, must be created with `createMaskNIFTI`

    output_path : a str, default "flattened_NIFTI_array.npy"

    Returns
    -------
    nifti_array : a numpy array
        array of flattened image intensities
        dimensions are N_Images x N_Masked_Voxels

    """
    print('\n[neuroHarmonize]: Flattening NIFTIs will consume large amounts of memory. Down-sampling may help.')
    # load mask (1=GM tissue, 0=Non-GM)
    mask_nifti = nib.load(mask_path)
    nifti_mask = (mask_nifti.get_fdata().round().astype(int)==1)
    mask_affine = mask_nifti.affine
    n_voxels_flattened = np.sum(nifti_mask)
    n_images = paths.shape[0]
    # initialize empty container
    nifti_array = np.zeros((n_images, n_voxels_flattened))
    # iterate over images and fill container
    print('\n[neuroHarmonize]: Flattening %d NIFTI images with %d voxels...' % (n_images, n_voxels_flattened))
    for i in range(0, n_images):
        nifti_i = nib.load(paths.PATH[i])
        if nifti_i.shape[:3] != nifti_mask.shape:
            raise ValueError(
                '[neuroHarmonize] Image %s has shape %s but mask has shape %s. '
                'Images must have identical dimensions to the mask.'
                % (paths.PATH[i], nifti_i.shape[:3], nifti_mask.shape))
        if not np.allclose(nifti_i.affine, mask_affine, atol=1e-5):
            warnings.warn(
                '[neuroHarmonize] Image %s has a different affine from the mask. '
                'Results may be incorrect. Consider resampling images to the same '
                'space as the mask.' % paths.PATH[i], UserWarning, stacklevel=2)
        nifti_array[i, :] = nifti_i.get_fdata()[nifti_mask]
        if (i==500):
            print('\n[neuroHarmonize]: loaded %d of %d images...' % (i, n_images))
    # save array of flattened images
    print('\n[neuroHarmonize]: Size of array in MB: %2.3f' % (nifti_array.nbytes / 1e6))
    np.save(output_path, nifti_array)
    return nifti_array   

def applyModelNIFTIs(covars, model, paths, mask_path, output_dtype=np.float32):
    """
    Applies harmonization model sequentially to NIFTI images. This function
    will reduce burden on memory resources for large datasets.

    Arguments
    ---------
    covars : a pandas DataFrame
        contains covariates to control for during harmonization
        all covariates must be encoded numerically (no categorical variables)
        must contain a single column "SITE" with site labels for ComBat
        dimensions are N_samples x (N_covariates + 1)

    model : a dictionary of model parameters
        the output of a call to `harmonizationLearn`

    paths : a pandas DataFrame
        must contain a column "PATH" with file paths to NIFTIs and must also
        contain a column "PATH_NEW" with file paths to the new NIFTIS that
        will be created with this function
        dimensions must be identical for all images

    mask_path : a str
        file path to the mask, must be created with `createMaskNIFTI`

    output_dtype : numpy dtype, default np.float32
        data type for output NIFTI images; use np.float32 (default) for full
        precision matching the tabular output, or np.int16 for legacy behavior

    Returns
    -------
    None
    """
    # load mask (1=include, 0=exclude)
    mask_nifti = nib.load(mask_path)
    nifti_mask = (mask_nifti.get_fdata().round().astype(int)==1)
    mask_affine = mask_nifti.affine
    n_voxels_flattened = np.sum(nifti_mask)
    # count number of images
    n_images = paths.shape[0]
    # apply harmonization model
    for i in range(0, n_images):
        path_new = paths.PATH_NEW.to_numpy()[i]
        covarsSel = covars.iloc[[i], :]
        nifti = nib.load(paths.PATH[i])
        if nifti.shape[:3] != nifti_mask.shape:
            raise ValueError(
                '[neuroHarmonize] Image %s has shape %s but mask has shape %s. '
                'Images must have identical dimensions to the mask.'
                % (paths.PATH[i], nifti.shape[:3], nifti_mask.shape))
        if not np.allclose(nifti.affine, mask_affine, atol=1e-5):
            warnings.warn(
                '[neuroHarmonize] Image %s has a different affine from the mask. '
                'Results may be incorrect. Consider resampling images to the same '
                'space as the mask.' % paths.PATH[i], UserWarning, stacklevel=2)
        nifti_array = nifti.get_fdata()[nifti_mask].reshape((1, n_voxels_flattened))
        affine = nifti.affine
        nifti_array_adj, nifti_array_reference = applyModelOne(nifti_array, covarsSel, model, True)

        # write harmonized image
        nifti_out = np.zeros(nifti_mask.shape, dtype=output_dtype)
        nifti_out[nifti_mask] = nifti_array_adj[0, :].astype(output_dtype)
        if np.issubdtype(output_dtype, np.integer):
            nifti_out = np.round(nifti_out).astype(output_dtype)
        out_header = nib.Nifti1Header()
        out_header.set_data_dtype(output_dtype)
        nifti_out = nib.Nifti1Image(nifti_out, affine, out_header)
        nifti_out.to_filename(path_new)

        # save reference mean (stand_mean + mod_mean) in nifti
        nifti_out_reference = np.zeros(nifti_mask.shape, dtype=output_dtype)
        nifti_out_reference[nifti_mask] = nifti_array_reference[0, :].astype(output_dtype)
        if np.issubdtype(output_dtype, np.integer):
            nifti_out_reference = np.round(nifti_out_reference).astype(output_dtype)
        out_header_ref = nib.Nifti1Header()
        out_header_ref.set_data_dtype(output_dtype)
        nifti_out_reference = nib.Nifti1Image(nifti_out_reference, affine, out_header_ref)
        path_new_reference = path_new.replace('_harmonized.nii.gz', '_reference_mean.nii.gz')
        nifti_out_reference.to_filename(path_new_reference)

        if (i==500):
            print('\n[neuroHarmonize]: saved %d of %d images...' % (i, n_images))
    return None
