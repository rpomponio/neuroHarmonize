import os
import nibabel as nib
import numpy as np
import pandas as pd

def create_NIFTI_mask(file_paths, threshold=0.0, output_path='thresholded_mask.nii.gz'):
    """
    Utility function for loading multiple NIFTI images and vectorizing for
    harmonization. All images must have the same dimensions and orientation.
    
    Arguments
    ---------
    file_paths : a pandas DataFrame
        list of file paths (absolute or relative) for each NIFTI image
        must contain a single column "PATH" with file paths
    
    threshold : float, default 1.0
        determines the threshold value below which voxel intensities will be
            masked, or excluded
        images in file_paths are averaged and a mask is created for all voxels
            with average intensities > threshold

    output_path : str, default "thresholded_mask.nii.gz"
        desired output path for the image mask
        
    Returns
    -------
    nifti_avg : a numpy array
        average image intensities, dimensions are same as input images
    
    nifti_mask : a numpy array
        mask of voxels included in data_array
        dimensions are same as input images
        1=included, 0=excluded/masked
    
    """
    # count number of images
    n_images = file_paths.shape[0]
    # begin summing image intensities
    i = 0
    nifti_i = nib.load(file_paths.PATH[i])
    nifti_sum = nifti_i.get_fdata()
    # iterate over all images
    for i in range(0, n_images):
        nifti_i = nib.load(file_paths.PATH[i])
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
    img = nib.Nifti1Image(nifti_mask, np.eye(4))
    img.to_filename(output_path)
    return nifti_avg, nifti_mask

def flatten_NIFTIs(file_paths, mask_path, output_path='flattened_nifti_array.npy'):
    """
    mask must be created with create_NIFTI_mask()
        
    Arguments
    ---------
    file_paths : a pandas DataFrame
        list of file paths (absolute or relative) for each NIFTI image
        must contain a single column "PATH" with file paths

    """
    # load mask (1=GM tissue, 0=Non-GM)
    nifti_mask = (nib.load(mask_path).get_fdata().astype(int)==1)
    n_voxels_flattened = np.sum(nifti_mask)
    # count images
    n_images = file_paths.shape[0]
    # initialize empty container
    nifti_array = np.zeros((n_images, n_voxels_flattened))
    # iterate over images and fill container
    print('Flattening %d NIFTI images with %d voxels...' % (n_images, n_voxels_flattened))
    for i in range(0, n_images):
        nifti_i = nib.load(file_paths.PATH[i]).get_fdata()
        nifti_array[i, :] = nifti_i[nifti_mask]
        if (i==500):
            print('PROGRESS: loaded %d of %d images...' % (i, n_images))
    # save array of flattened images
    print('Size of array in MB: %2.3f' % (nifti_array.nbytes / 1e6))
    np.save(output_path, nifti_array)
    return nifti_array   

