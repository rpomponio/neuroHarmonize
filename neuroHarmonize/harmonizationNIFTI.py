import os
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
    nifti_sum = nifti_i.get_fdata()
    # iterate over all images
    for i in range(0, n_images):
        nifti_i = nib.load(paths.PATH[i])
        nifti_sum += nifti_i.get_fdata()
        if (i==500):
            print('PROGRESS: loaded %d of %d images...' % (i, n_images))
    # compute average intensities
    nifti_avg = nifti_sum / n_images    
    # apply threshold
    nifti_avg[nifti_avg<threshold] = 0.0
    # create mask and save as NIFTI image
    nifti_mask = nifti_avg.copy()
    nifti_mask[nifti_mask>0.0] = 1.0
    img = nib.Nifti1Image(nifti_mask, affine_0)
    img.to_filename(output_path)
    return nifti_avg, nifti_mask, affine_0

def flattenNIFTIs(paths, mask_path, output_path='flattened_NIFTI_array.npy'):
    """
    Flattens a dataset of NIFTI images to a 2D array.
        
    Arguments
    ---------
    paths : a pandas DataFrame
        must contain a single column "PATH" with file paths to NIFTIs
        dimensions must be identical for all images

    mask_path : a str
        file path to the mask, must be created with `create_NIFTI_mask`

    output_path : a str, default "flattened_NIFTI_array.npy"

    Returns
    -------
    nifti_array : a numpy array
        array of flattened image intensities
        dimensions are N_Images x N_Masked_Voxels

    """
    print('\nWARNING: this procedure will consume large amounts of memory.')
    print(' Consider down-sampling, masking aggressively, and/or sub-sampling NIFTIs...')
    # load mask (1=GM tissue, 0=Non-GM)
    nifti_mask = (nib.load(mask_path).get_fdata().astype(int)==1)
    n_voxels_flattened = np.sum(nifti_mask)
    # count images
    n_images = paths.shape[0]
    # initialize empty container
    nifti_array = np.zeros((n_images, n_voxels_flattened))
    # iterate over images and fill container
    print('Flattening %d NIFTI images with %d voxels...' % (n_images, n_voxels_flattened))
    for i in range(0, n_images):
        nifti_i = nib.load(paths.PATH[i]).get_fdata()
        nifti_array[i, :] = nifti_i[nifti_mask]
        if (i==500):
            print('PROGRESS: loaded %d of %d images...' % (i, n_images))
    # save array of flattened images
    print('Size of array in MB: %2.3f' % (nifti_array.nbytes / 1e6))
    np.save(output_path, nifti_array)
    return nifti_array   

