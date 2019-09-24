import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from statsmodels.gam.api import BSplines
from .neuroCombat import make_design_matrix, adjust_data_final

def harmonizationApply(data, covars, model):
    """
    Applies harmonization model with neuroCombat functions to new data.
    
    Arguments
    ---------
    data : a numpy array
        data to harmonize with ComBat, dimensions are N_samples x N_features
    
    covars : a pandas DataFrame 
        contains covariates to control for during harmonization
        all covariates must be encoded numerically (no categorical variables)
        must contain a single column "SITE" with site labels for ComBat
        dimensions are N_samples x (N_covariates + 1)
        
    model : a dictionary of model parameters
        the output of a call to harmonizationLearn()
    
    Returns
    -------
    
    bayes_data : a numpy array
        harmonized data, dimensions are N_samples x N_features
        
    """
    # transpose data as per ComBat convention
    data = data.T
    # prep covariate data
    batch_col = covars.columns.get_loc('SITE')
    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']
    covars = np.array(covars, dtype='object')
    # load the smoothing model
    smooth_model = model['smooth_model']
    smooth_cols = smooth_model['smooth_cols']
    ### additional setup code from neuroCombat implementation:
    # convert batch col to integer
    covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]
    # create dictionary that stores batch info
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    }
    ###
    # check sites are identical in training dataset
    check_sites = info_dict['n_batch']==model['info_dict']['n_batch']
    if not check_sites:
        raise ValueError('Number of sites in holdout data not identical to training data.')
    # apply ComBat without re-learning model parameters
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols)
    ### additional setup if smoothing is performed
    if smooth_model['perform_smoothing']:
        # create cubic spline basis for smooth terms
        X_spline = covars[:, smooth_cols].astype(float)
        bs_basis = smooth_model['bsplines_constructor'].transform(X_spline)
        # construct formula and dataframe required for gam
        formula = 'y ~ '
        df_gam = {}
        for b in batch_levels:
            formula = formula + 'x' + str(b) + ' + '
            df_gam['x' + str(b)] = design[:, b]
        for c in num_cols:
            if c not in smooth_cols:
                formula = formula + 'c' + str(c) + ' + '
                df_gam['c' + str(c)] = covars[:, c].astype(float)
        formula = formula[:-2] + '- 1'
        df_gam = pd.DataFrame(df_gam)
        # check formulas are identical in training dataset
        check_formula = formula==smooth_model['formula']
        if not check_formula:
            raise ValueError('GAM formula for holdout data not identical to training data.')
        # for matrix operations, a modified design matrix is required
        design = np.concatenate((df_gam, bs_basis), axis=1)
    ###
    s_data, stand_mean, var_pooled = ApplyStandardizationAcrossFeatures(data, design, info_dict, model)
    bayes_data = adjust_data_final(s_data, design, model['gamma_star'], model['delta_star'],
                                   stand_mean, var_pooled, info_dict)
    # transpose data to return to original shape
    bayes_data = bayes_data.T
    
    return bayes_data

def ApplyStandardizationAcrossFeatures(X, design, info_dict, model):   
    """
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    
    This function will apply a pre-trained harmonization model to new data.
    """
    
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']

    B_hat = model['B_hat']
    grand_mean = model['grand_mean']
    var_pooled = model['var_pooled']

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T
    
    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled

def loadHarmonizationModel(file_name):
    """
    For loading model contents, this function will load a model specified
    by file_name using the pickle package.
    """
    if not os.path.exists(file_name):
        raise ValueError('Model file does not exist: %s' % file_name)
    in_file = open(file_name,'rb')
    model = pickle.load(in_file)
    in_file.close()
    
    return model

def harmonizationApplyNIFTI(data, covars, model, img_mask, path_output='./'):
    """
    Applies harmonization model with neuroCombat functions to NIFTI-formatted
    brain images.
    
    Arguments
    ---------
    data : a pandas DataFrame
        must contain a column "PATH" which contains the file paths to the images
        that will be harmonized
        dimensions are N_samples x 1
    
    covars : a pandas DataFrame 
        contains covariates to control for during harmonization
        all covariates must be encoded numerically (no categorical variables)
        must contain a single column "SITE" with site labels for ComBat
        dimensions are N_samples x (N_covariates + 1)
        
    model : a dictionary of model parameters
        the output of a call to harmonizationLearn
        
    img_mask : a numpy array
        an array of 1's and 0's, defining a mask for the images to be harmonized
        any array element that is set to 1 will be masked-out of the images
        must be same dimensions as target images
    
    path_output : str
        the file path where the harmonized images will be written
    
    """
    # prep covariate data
    batch_col = covars.columns.get_loc('SITE')
    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']
    covars = np.array(covars, dtype='object')
    ### additional setup code from neuroCombat implementation:
    # convert batch col to integer
    covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]
    # create dictionary that stores batch info
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    }
    ###
    # check sites are identical in training dataset
    check_sites = info_dict['n_batch']==model['n_batch']
    if not check_sites:
        raise ValueError('Number of sites in holdout data not identical to training data.')
    # apply ComBat without re-learning model parameters
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols)
    # apply standardization across features and adjust data to each image iteratively
    #i = 0
    for i in range(0, info_dict['n_sample']):
        img_fname = data.PATH[i].split('/')[-1]
        print('Applying harmonization to NIFTI image: %s' % img_fname)
        j = covars[i, batch_col] # batch number, using new encoding
        img_subject = nib.load(data.PATH[i])
        img_subject_mat = img_subject.get_fdata()
        X = img_subject_mat[~img_mask].flatten()
        X = X.T.reshape((len(X), 1))
        n_sample = 1
        sample_per_batch = 1
        D = design[[i], :]
        ### from neuroCombat implementation:
        n_batch = info_dict['n_batch']

        B_hat = model['B_hat']
        grand_mean = model['grand_mean']
        var_pooled = model['var_pooled']
        
        stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
        tmp = np.array(D.copy())
        tmp[:,:n_batch] = 0
        stand_mean  += np.dot(tmp, B_hat).T
        
        s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))
        
        # adjust final data
        batch_design = D[:,:n_batch]

        bayesdata = s_data
        gamma_star = np.array(model['gamma_star'])
        delta_star = np.array(model['delta_star'])

        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, sample_per_batch)))
        numer = np.array(bayesdata - np.dot(batch_design, gamma_star).T)

        bayesdata = numer / denom

        vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
        bayesdata = bayesdata * np.dot(vpsq, np.ones((1, n_sample))) + stand_mean
        ###
        # transform data to image space and save NIFTI
        img_subject_mat[~img_mask] = bayesdata[:, 0]
        img_subject = nib.Nifti1Image(img_subject_mat, img_subject.affine, img_subject.header)
        nib.save(img_subject, path_output + img_fname)
    
    return None