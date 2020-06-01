import os
import nibabel as nib
import numpy as np
import pandas as pd

def loadNIFTI(file_paths, threshold=1.0):
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
        
    Returns
    -------
    data_array : a numpy array
        vectorized NIFTI data
        dimensions are N_samples x N_voxels (after masking)
    
    mask : a numpy array of int
        mask of voxels included in data_array
        dimensions are identical to images in file_paths
        1=included, 0=excluded/masked
    
    """
    

