import os
import numpy as np
import pandas as pd
from .neuroCombat import make_design_matrix, fit_LS_model_and_find_priors, find_parametric_adjustments, adjust_data_final

def harmonizationLearn(data, covars):
    """
    Wrapper for neuroCombat function that returns the harmonization model.
    
    Arguments
    ---------
    data : a numpy array
        data to harmonize with ComBat, dimensions are N_samples x N_features
    
    covars : a pandas DataFrame 
        contains covariates to control for during harmonization
        all covariates must be encoded numerically (no categorical variables)
        must contain a single column "SITE" with site labels for ComBat
        dimensions are N_samples x (N_covariates + 1)
    
    Returns
    -------
    model : a dictionary of estimated model parameters
        design, s_data, stand_mean, var_pooled, B_hat, grand_mean,
        gamma_star, delta_star, info_dict (a neuroCombat invention)
    
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
    # run steps to perform ComBat
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols)
    s_data, stand_mean, var_pooled, B_hat, grand_mean = StandardizeAcrossFeatures(data, design, info_dict)
    LS_dict = fit_LS_model_and_find_priors(s_data, design, info_dict)
    gamma_star, delta_star = find_parametric_adjustments(s_data, LS_dict, info_dict)
    bayes_data = adjust_data_final(s_data, design, gamma_star, delta_star, stand_mean, var_pooled, info_dict)
    # save model parameters in single object
    model = {'design': design, 's_data': s_data, 'stand_mean': stand_mean, 'var_pooled':var_pooled,
             'B_hat':B_hat, 'grand_mean': grand_mean, 'gamma_star': gamma_star,
             'delta_star': delta_star, 'n_batch': info_dict['n_batch']}
    # transpose data to return to original shape
    bayes_data = bayes_data.T
    return model, bayes_data

def StandardizeAcrossFeatures(X, design, info_dict):
    """Modified from neuroCombat to return coefficients and mean estimates in
    addition to default standardization parameters."""
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']

    B_hat = np.dot(np.dot(np.linalg.inv(np.dot(design.T, design)), design.T), X.T)
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled, B_hat, grand_mean

def saveHarmonizationModel(model, fldr_name):
    """Helper function to save a model to a new folder. Will save numpy arrays."""
    fldr_name = fldr_name.replace('/', '')
    if os.path.exists(fldr_name):
        raise ValueError('Model folder already exists: %s Change name or delete to save.' % fldr_name)
    else:
        os.makedirs(fldr_name)
    # cleanup model object for saving to file
    del model['design']
    del model['s_data']
    del model['stand_mean']
    del model['n_batch']
    for key in list(model.keys()):
        obj_size = model[key].nbytes / 1e6
        print('Saving model object: %s, size in MB: %4.2f' % (key, obj_size))
        np.save(fldr_name + '/' + key + '.npy', model[key])
    return None