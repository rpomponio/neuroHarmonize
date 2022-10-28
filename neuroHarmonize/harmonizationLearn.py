import os
import pickle
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
from .harmonizationApply import applyStandardizationAcrossFeatures
from .neuroCombat import make_design_matrix, find_parametric_adjustments, adjust_data_final, aprior, bprior
import copy

def harmonizationLearn(data, covars, eb=True, smooth_terms=[],
                       smooth_term_bounds=(None, None), return_s_data=False,
                       orig_model=None, seed=None):
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
        
    eb : bool, default True
        whether to use empirical Bayes estimates of site effects
        
    smooth_terms (Optional) :  a list, default []
        names of columns in covars to include as smooth, nonlinear terms
        can be any or all columns in covars, except "SITE"
        if empty, ComBat is applied with a linear model of covariates
        if not empty, Generalized Additive Models (GAMs) are used
        will increase computation time due to search for optimal smoothing
        
    smooth_term_bounds (Optional) : tuple of float, default (None, None)
        feature to support custom boundaries of the smoothing terms
        useful when holdout data covers different range than 
        specify the bounds as (minimum, maximum)
        currently not supported for models with mutliple smooth terms
        
    return_s_data (Optional) : bool, default False
        whether to return s_data, the standardized data array
        can be useful for diagnostics but will be costly to save/load if large

    seed (Optional) : int, default None
        By default, this function is non-deterministic. Setting the optional
        argument `seed` will make the function deterministic.


    Returns
    -------
    model : a dictionary of estimated model parameters
        design, var_pooled, B_hat, grand_mean,
        gamma_star, delta_star, info_dict (a neuroCombat invention),
        gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior, smooth_model
    
    bayes_data : a numpy array
        harmonized data, corrected for effects of SITE
        dimensions are N_samples x N_features

    s_data (Optional) : a numpy array
        standardized residuals after accounting for `covars` other than `SITE`
        set return_s_data=True to output the variable
    
    """
    # set optional random seed
    if seed is not None:
        pass
    else:
        np.random.seed(seed)

    if orig_model is None:
        pass
    else:
        model = copy.deepcopy(orig_model)
    
    # transpose data as per ComBat convention
    data = data.T
    # prep covariate data
    covar_levels = list(covars.columns)
    batch_labels = np.unique(covars.SITE)
    batch_col = covars.columns.get_loc('SITE')

    if orig_model is None:
        pass
    else:
        isTrainSite = covars['SITE'].isin(model['SITE_labels'])
        isTrainSiteLabel = set(model['SITE_labels'])
        isTrainSiteColumns = np.where((pd.DataFrame(np.unique(covars['SITE'])).isin(model['SITE_labels']).values).flat)
        isTrainSiteColumnsOrig = np.where((pd.DataFrame(model['SITE_labels']).isin(np.unique(covars['SITE'])).values).flat)
        isTestSiteColumns = np.where((~pd.DataFrame(np.unique(covars['SITE'])).isin(model['SITE_labels']).values).flat)

    cat_cols = []
    num_cols = [covars.columns.get_loc(c) for c in covars.columns if c!='SITE']
    smooth_cols = [covars.columns.get_loc(c) for c in covars.columns if c in smooth_terms]
    # maintain a dictionary of smoothing information
    smooth_model = {
        'perform_smoothing': len(smooth_terms)>0,
        'smooth_terms': smooth_terms,
        'smooth_cols': smooth_cols,
        'bsplines_constructor': None,
        'formula': None,
        'df_gam': None
    }
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
    design = make_design_matrix(covars, batch_col, cat_cols, num_cols)

    
    
    ### additional setup if smoothing is performed
    if smooth_model['perform_smoothing']:
        # create cubic spline basis for smooth terms
        X_spline = covars[:, smooth_cols].astype(float)
        if orig_model is None:
            bs = BSplines(X_spline, df=[10] * len(smooth_cols), degree=[3] * len(smooth_cols),
                        knot_kwds=[{'lower_bound':smooth_term_bounds[0], 'upper_bound':smooth_term_bounds[1]}])
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
            design = np.concatenate((df_gam, bs.basis), axis=1)
            # store objects in dictionary
            smooth_model['bsplines_constructor'] = bs
            smooth_model['formula'] = formula
            smooth_model['df_gam'] = df_gam
        else:
            bs_basis = model['smooth_model']['bsplines_constructor'].transform(X_spline)
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
    # run steps to perform ComBat
    if orig_model is None:
        s_data, stand_mean, var_pooled, B_hat, grand_mean = standardizeAcrossFeatures(
            data, design, info_dict, smooth_model)
        LS_dict = fitLSModelAndFindPriors(s_data, design, info_dict, eb=eb)
        # optional: avoid EB estimates
        if eb:
            gamma_star, delta_star = find_parametric_adjustments(s_data, LS_dict, info_dict)
        else:
            gamma_star = LS_dict['gamma_hat']
            delta_star = np.array(LS_dict['delta_hat'])
        bayes_data = adjust_data_final(s_data, design, gamma_star, delta_star, stand_mean, var_pooled, info_dict)
        # save model parameters in single object
        model = {'design': design, 'SITE_labels': batch_labels,
                'var_pooled':var_pooled, 'B_hat':B_hat, 'grand_mean': grand_mean,
                'gamma_star': gamma_star, 'delta_star': delta_star, 'info_dict': info_dict,
                'gamma_hat': LS_dict['gamma_hat'], 'delta_hat': np.array(LS_dict['delta_hat']),
                'gamma_bar': LS_dict['gamma_bar'], 't2': LS_dict['t2'],
                'a_prior': LS_dict['a_prior'], 'b_prior': LS_dict['b_prior'],
                'smooth_model': smooth_model, 'eb': eb,'SITE_labels_train':batch_labels,'Covariates':covar_levels}
        # transpose data to return to original shape
        bayes_data = bayes_data.T
    else:
        # Create train data 
        (batch_levels, sample_per_batch) = np.unique(covars[isTrainSite,batch_col],return_counts=True)
        if batch_levels.size == 0:
            bayes_data_train = np.zeros(shape=(0,data.shape[0]))
            s_data_train = np.zeros(shape=(0,data.shape[0])).T
        else:
            info_dict_train = model['info_dict'].copy()
            info_dict_train['sample_per_batch'] = sample_per_batch.astype('int')
            info_dict_train['batch_info'] = [list(np.where(covars[isTrainSite,batch_col]==idx)[0]) for idx in batch_levels]
            tmp = np.concatenate((np.zeros(shape=(info_dict['n_sample'],len(model['SITE_labels']))), design[:,len(batch_labels):]),axis=1)
            s_data_train, stand_mean_train, _ = applyStandardizationAcrossFeatures(data[:,isTrainSite], tmp[isTrainSite,:], info_dict_train, model)
            design2=tmp.copy()
            design2[:,isTrainSiteColumnsOrig[0]] = design[:,isTrainSiteColumns[0]]
            bayes_data_train = adjust_data_final(s_data_train, design2[isTrainSite,:], model['gamma_star'], model['delta_star'], stand_mean_train, model['var_pooled'], info_dict_train)
            # transpose data to return to original shape
            bayes_data_train = bayes_data_train.T

        # Create test data (new SITE)
        (batch_levels, sample_per_batch) = np.unique(covars[~isTrainSite,batch_col],return_counts=True)
        if batch_levels.size == 0:
            bayes_data_test = np.zeros(shape=(0,data.shape[0]))
            s_data_test = np.zeros(shape=(0,data.shape[0])).T
        else:
            info_dict_test = {
                'batch_levels': batch_levels.astype('int'),
                'n_batch': len(batch_levels),
                'n_sample': int(covars[~isTrainSite,:].shape[0]),
                'sample_per_batch': sample_per_batch.astype('int'),
                'batch_info': [list(np.where(covars[~isTrainSite,batch_col]==idx)[0]) for idx in batch_levels]
            }
            design_tmp = np.concatenate((design[:,isTestSiteColumns[0]], design[:,len(batch_labels):]),axis=1)
            s_data_test, stand_mean_test, _ = applyStandardizationAcrossFeatures(data[:,~isTrainSite], design_tmp[~isTrainSite,:], info_dict_test, model)
            LS_dict = fitLSModelAndFindPriors(s_data_test, design_tmp[~isTrainSite,:], info_dict_test, eb=eb)
            if eb:
                gamma_star, delta_star = find_parametric_adjustments(s_data_test, LS_dict, info_dict_test)
            else:
                gamma_star = LS_dict['gamma_hat']
                delta_star = np.array(LS_dict['delta_hat'])
            betas = []
            for i in range(info_dict_test['n_batch']):
                diff_mean = np.mean(data[:,info_dict_test['batch_info'][i]]-np.dot(design[info_dict_test['batch_info'][i],info_dict['n_batch']:],model['B_hat'][len(model['SITE_labels']):,:]).T,axis=1)
                betas.append(diff_mean)
            new_betas = np.array(betas)
            model['B_hat'] = np.concatenate((model['B_hat'][:len(model['SITE_labels']),:],new_betas,model['B_hat'][len(model['SITE_labels']):,:]))
            model['SITE_labels'] = np.append(model['SITE_labels'],list(set(batch_labels)-isTrainSiteLabel))
            model['gamma_star'] = np.append(model['gamma_star'],gamma_star,axis=0)
            model['delta_star'] = np.append(model['delta_star'],delta_star,axis=0)
            model['info_dict']['n_batch'] = len(model['SITE_labels'])
            bayes_data_test = adjust_data_final(s_data_test, design_tmp[~isTrainSite,:], gamma_star, delta_star, stand_mean_test, model['var_pooled'], info_dict_test)
            # transpose data to return to original shape
            bayes_data_test = bayes_data_test.T
        bayes_data = np.zeros(shape=data.T.shape)
        bayes_data[isTrainSite,:] = bayes_data_train
        bayes_data[~isTrainSite,:] = bayes_data_test
        s_data = np.zeros(shape=data.T.shape)
        s_data[isTrainSite,:] = s_data_train.T
        s_data[~isTrainSite,:] = s_data_test.T

    if return_s_data:
        return model, bayes_data, s_data.T
    else:
        return model, bayes_data

def standardizeAcrossFeatures(X, design, info_dict, smooth_model):
    """
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    
    This function will return all estimated parameters in addition to the
    standardized data.
    """
    n_batch = info_dict['n_batch']
    n_sample = info_dict['n_sample']
    sample_per_batch = info_dict['sample_per_batch']

    ### perform smoothing with GAMs if specified
    if smooth_model['perform_smoothing']:
        smooth_cols = smooth_model['smooth_cols']
        bs = smooth_model['bsplines_constructor']
        formula = smooth_model['formula']
        df_gam = smooth_model['df_gam']
        
        if X.shape[0] > 10:
            print('\n[neuroHarmonize]: smoothing more than 10 variables may take several minutes of computation.')
        # initialize penalization weight (not the final weight)
        alpha = np.array([1.0] * len(smooth_cols))
        # initialize an empty matrix for beta
        B_hat = np.zeros((design.shape[1], X.shape[0]))
        # estimate beta for each variable to be harmonized
        for i in range(0, X.shape[0]):
            df_gam.loc[:, 'y'] = X[i, :]
            gam_bs = GLMGam.from_formula(formula, data=df_gam, smoother=bs, alpha=alpha)
            res_bs = gam_bs.fit()
            # Optimal penalization weights alpha can be obtained through gcv/kfold
            # Note: kfold is faster, gcv is more robust
            gam_bs.alpha = gam_bs.select_penweight_kfold()[0]
            res_bs_optim = gam_bs.fit()
            B_hat[:, i] = res_bs_optim.params
    ###
    else:
        B_hat = np.dot(np.dot(np.linalg.inv(np.dot(design.T, design)), design.T), X.T)
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((n_sample, 1)) / float(n_sample))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample))) # nothing but grand mean
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T  
    s_data = ((X- stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, n_sample))))

    return s_data, stand_mean, var_pooled, B_hat, grand_mean

def fitLSModelAndFindPriors(s_data, design, info_dict, eb=True):
    """
    The original neuroCombat function fit_LS_model_and_find_priors plus
    necessary modifications.
    
    This function will return no EB information if eb=False
    """
    n_batch = info_dict['n_batch']
    batch_info = info_dict['batch_info'] 
    
    batch_design = design[:,:n_batch]
    gamma_hat = np.array(np.dot(np.dot(np.linalg.inv(np.matrix(np.dot(batch_design.T, batch_design))), batch_design.T), s_data.T))

    delta_hat = []
    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(np.var(s_data[:,batch_idxs],axis=1,ddof=1))
    
    if eb:
        gamma_bar = np.mean(gamma_hat, axis=1) 
        t2 = np.var(gamma_hat,axis=1, ddof=1)

        a_prior = list(map(aprior, delta_hat))
        b_prior = list(map(bprior, delta_hat))

        LS_dict = {}
        LS_dict['gamma_hat'] = gamma_hat
        LS_dict['delta_hat'] = delta_hat
        LS_dict['gamma_bar'] = gamma_bar
        LS_dict['t2'] = t2
        LS_dict['a_prior'] = a_prior
        LS_dict['b_prior'] = b_prior
        return LS_dict
    else:
        LS_dict = {}
        LS_dict['gamma_hat'] = gamma_hat
        LS_dict['delta_hat'] = delta_hat
        LS_dict['gamma_bar'] = None
        LS_dict['t2'] = None
        LS_dict['a_prior'] = None
        LS_dict['b_prior'] = None
        return LS_dict


def saveHarmonizationModel(model, file_name):
    """
    Save a harmonization model from harmonizationLearn().
    
    For saving model contents, this function will create a new file specified
    by file_name, and store the model using the pickle package.
    
    """
    if os.path.exists(file_name):
        raise ValueError('Model file already exists: %s. Change name or delete to save.' % file_name)
    # estimate size of out_file
    est_size = 0
    for key in ['design', 'B_hat', 'grand_mean', 'var_pooled',
                'gamma_star', 'delta_star', 'gamma_hat', 'delta_hat']:
        est_size += model[key].nbytes / 1e6
    print('\n[neuroHarmonize]: Saving model object, estimated size in MB: %4.2f' % est_size)
    out_file = open(file_name, 'wb')
    pickle.dump(model, out_file)
    out_file.close()
    
    return None
