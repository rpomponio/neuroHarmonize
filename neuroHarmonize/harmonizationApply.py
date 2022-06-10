import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from statsmodels.gam.api import BSplines
from .neuroCombat import make_design_matrix, adjust_data_final

def harmonizationApply(data, covars, model,return_stand_mean=False):
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
    isTrainSite = covars['SITE'].isin(model['SITE_labels'])
    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']
    covars = np.array(covars, dtype='object')
    # load the smoothing model
    smooth_model = model['smooth_model']
    smooth_cols = smooth_model['smooth_cols']
    
    ### additional setup code from neuroCombat implementation:
    # convert training SITEs in batch col to integers
    site_dict = dict(zip(model['SITE_labels'], np.arange(len(model['SITE_labels']))))
    covars[:,batch_col] = np.vectorize(site_dict.get)(covars[:,batch_col],-1)

    # compute samples_per_batch for training data
    sample_per_batch = [np.sum(covars[:,batch_col]==i) for i in list(site_dict.values())]
    sample_per_batch = np.asarray(sample_per_batch)
    
    # create dictionary that stores batch info
    batch_levels = np.unique(list(site_dict.values()),return_counts=False)
    info_dict = {
        'batch_levels': batch_levels.astype('int'),
        'n_batch': len(batch_levels),
        'n_sample': int(covars.shape[0]),
        'sample_per_batch': sample_per_batch.astype('int'),
        'batch_info': [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    }
    covars[~isTrainSite, batch_col] = 0
    covars[:,batch_col] = covars[:,batch_col].astype(int)
    ###
    # isolate array of data in training site
    # apply ComBat without re-learning model parameters
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols,nb_class = len(model['SITE_labels']))
    design[~isTrainSite,0:len(model['SITE_labels'])] = np.nan
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
        # for matrix operations, a modified design matrix is required
        design = np.concatenate((df_gam, bs_basis), axis=1)

    ###
    s_data, stand_mean, var_pooled = applyStandardizationAcrossFeatures(data, design, info_dict, model)
    if sum(isTrainSite)==0:
        bayes_data = np.full(s_data.shape,np.nan)
    else:
        bayes_data = adjust_data_final(s_data, design, model['gamma_star'], model['delta_star'],
                                    stand_mean, var_pooled, info_dict)
        bayes_data[:,~isTrainSite] = np.nan
                                   
    # transpose data to return to original shape
    stand_mean = stand_mean.T
    bayes_data = bayes_data.T

    #return either bayes_data or both
    if return_stand_mean:
        return bayes_data, stand_mean
    else:
        return bayes_data


def applyStandardizationAcrossFeatures(X, design, info_dict, model):   
    """
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    
    This function will apply a pre-trained harmonization model to new data.
    """
    
    n_batch = info_dict['n_batch']
    n_sample = design.shape[0]
    sample_per_batch = info_dict['sample_per_batch']

    B_hat = model['B_hat']
    grand_mean = model['grand_mean']
    var_pooled = model['var_pooled']

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    tmp = np.array(design.copy())
    tmp = np.concatenate((np.zeros(shape=(n_sample,len(model['SITE_labels']))), tmp[:,n_batch:]),axis=1)
    stand_mean  += np.dot(tmp, B_hat).T
    
    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled

def applyModelOne(data, covars, model,return_stand_mean=False):
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
    isTrainSite = covars['SITE'].isin(model['SITE_labels'])

    if batch_i not in batch_labels:
#        raise ValueError('Site Label "%s" not in the training set. Check `covars` argument.' % batch_i)
        batch_level_i = np.array([0])
    else:
        batch_level_i = np.argwhere(batch_i==batch_labels)[0]

    batch_col = covars.columns.get_loc('SITE')
    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']
    covars = np.array(covars, dtype='object')
    
    # convert batch col to integer
    covars[:,batch_col] = np.unique(covars[:,batch_col],return_inverse=True)[-1]

    # apply design matrix construction (needs to be modified)
#    design_i = make_design_matrix(covars, batch_col, cat_cols, num_cols)
    design_i = make_design_matrix(covars, batch_col, cat_cols, num_cols,nb_class = len(model['SITE_labels']))

    # encode batches as in larger dataset
    design_i_batch = np.zeros((1, len(batch_labels)))
    design_i_batch[:, batch_level_i] = 1
#    design_i = np.concatenate((design_i_batch, design_i[:, 1:]), axis=1)
    design_i = np.concatenate((design_i_batch, design_i[:, len(batch_labels):]), axis=1)

    design_i[~isTrainSite,0:len(model['SITE_labels'])] = np.nan

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


    if sum(isTrainSite)==0:
        bayesdata = np.full(s_data.shape,np.nan)
    else:
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
    

    # return either bayesdata or both
    if return_stand_mean:
        return bayesdata.T, stand_mean.T
    else:
        return bayesdata.T
#    return bayesdata.T

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
