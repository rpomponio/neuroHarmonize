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
    batch_labels = np.unique(covars.SITE)
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
        raise ValueError('Number of sites in holdout data not identical to training data. Check `covars` argument.')
    check_sites = np.mean(batch_labels==model['SITE_labels'])
    if check_sites!=1:
        raise ValueError('Labels of sites in holdout data not identical to training data. Check values in "SITE" column.')
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
            raise ValueError('GAM formula for holdout data not identical to training data. Check arguments.')
        # for matrix operations, a modified design matrix is required
        design = np.concatenate((df_gam, bs_basis), axis=1)
    ###
    s_data, stand_mean, var_pooled = applyStandardizationAcrossFeatures(data, design, info_dict, model)
    bayes_data = adjust_data_final(s_data, design, model['gamma_star'], model['delta_star'],
                                   stand_mean, var_pooled, info_dict)
    # transpose data to return to original shape
    bayes_data = bayes_data.T
    
    return bayes_data

def applyStandardizationAcrossFeatures(X, design, info_dict, model):   
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

def applyModelOne(data, covars, model):
    """
    Utility function to apply model to one data point.
    """
    if data.shape[0]>1:
        raise ValueError('Argument `data` contains more than one sample!')
    if covars.shape[0]>1:
        raise ValueError('Argument `covars` contains more than one sample!')
    # transpose data as per ComBat convention
    X = data.T
    # prep covariate data
    batch_labels = model['SITE_labels']
    batch_i = covars.SITE.values[0]
    if batch_i not in batch_labels:
        raise ValueError('Site Label "%s" not in the training set. Check `covars` argument.' % batch_i)
    batch_level_i = np.argwhere(batch_i==batch_labels)[0]
    batch_col = covars.columns.get_loc('SITE')
    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']
    covars = np.array(covars, dtype='object')
    # apply design matrix construction (needs to be modified)
    design_i = make_design_matrix(covars, batch_col, cat_cols, num_cols)
    # encode batches as in larger dataset
    design_i_batch = np.zeros((1, len(batch_labels)))
    design_i_batch[:, batch_level_i] = 1
    design_i = np.concatenate((design_i_batch, design_i[:, 1:]), axis=1)
    # additional setup with batch info
    n_sample = 1
    sample_per_batch = 1
    D = design_i
    n_batch = len(batch_labels)
    j = batch_level_i[0]
    ### from neuroCombat implementation:
    B_hat = model['B_hat']
    grand_mean = model['grand_mean']
    var_pooled = model['var_pooled']

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    tmp = np.array(D.copy())
    tmp[:,:n_batch] = 0
    stand_mean += np.dot(tmp, B_hat).T

    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    # adjust_data_final
    batch_design = D[:,:n_batch]

    bayesdata = s_data
    gamma_star = np.array(model['gamma_star'])
    delta_star = np.array(model['delta_star'])

    dsq = np.sqrt(delta_star[j, :])
    dsq = dsq.reshape((len(dsq), 1))
    denom = np.dot(dsq, np.ones((1, sample_per_batch)))
    numer = np.array(bayesdata - np.dot(batch_design, gamma_star).T)

    bayesdata = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, n_sample))) + stand_mean
    ###
    return bayesdata.T

def loadHarmonizationModel(file_name):
    """
    For loading model contents, this function will load a model specified
    by file_name using the pickle package.
    """
    if not os.path.exists(file_name):
        raise ValueError('Model file does not exist: %s. Did you run `saveHarmonizationModel`?' % file_name)
    in_file = open(file_name,'rb')
    model = pickle.load(in_file)
    in_file.close()
    
    return model