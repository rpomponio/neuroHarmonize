==============
neuroHarmonize
==============

Harmonization tools for multi-site neuroimaging studies.

Overview
---------

This package extends the functionality of the package developed by Nick Cullen,
``neuroCombat``, which is hosted on GitHub: https://github.com/ncullen93/neuroCombat
(To make installation easier, ``neuroCombat`` is not a dependency for this package,
but the source code is included to call ``neuroCombat`` functions).

``neuroCombat`` allows the user to perform a harmonization procedure using
the ComBat [1]_ algorithm for correcting multi-site data.

``neuroHarmonize`` is a package with similar functionality, but also allows the
user to perform the following additional procedures:

1. Train a harmonization model on a subset of data, then apply the model to the
   whole set. For example, in longitudinal analyses, one may wish to train a
   harmonization model on baseline cases and apply the model to follow-up cases.
2. Specify covariates with nonlinear effects. Age tends to exhibit nonlinear
   relationships with brain volumes. Nonlinear effects are implemented using
   Generalized Additive Models (GAMs) via the ``statsmodels`` package.
3. Apply a pre-trained harmonization model to NIFTI images. When performing
   image-level harmonization, loading the entire set of images may exceed
   memory capacity. In such cases, it is still possible to harmonize images by
   sequentially adjusting images one-by-one. This functionality is made
   available via the ``nibabel`` package.

Installation
------------

Option 1: Install from PyPI (recommended)

*instructions will be written once package is published*

Option 2: Install from GitHub

Download the zipped repository on GitHub: https://github.com/rpomponio/neuroHarmonize

Unzip the download. Navigate to the directory neuroHarmonize-master/ which is
probably in your downloads folder.

Open a terminal and run:

    >>> pip install .

Quick Start
-----------

You must provide a **data matrix** which is a ``numpy`` array containing the
features to be harmonized. For example, an array of brain volumes:

::
  
  array([[3138.0, 3164.2,  ..., 206.4],
         [1708.4, 2351.2,  ..., 364.0],
         ...,
         [1119.6, 1071.6,  ..., 326.6]])
         
The dimensionality of this matrix must be: N_samples x N_features.

You must also provide a **covariate matrix** which is a ``pandas`` DataFrame
containing covariates to control for during harmonization. All covariates must
be encoded numerically (no categorical covariates allowed). The DataFrame must
also contain a single column "SITE" with the site labels for ComBat.

::

       SITE   AGE  SEX_M
  0  SITE_A  76.5      1
  1  SITE_B  80.1      1
  2  SITE_A  82.9      0
  ...   ...   ...    ...
  9  SITE_B  82.1      0
  

After preparing both inputs, you can call ``harmonizationLearn`` to harmonize
the provided dataset.

Example usage:

    >>> from neuroHarmonize import harmonizationLearn
    >>> import pandas as pd
    >>> import numpy as np
    >>> # load your data and all numeric covariates
    >>> my_data = pd.read_csv('brain_volumes.csv')
    >>> my_data = np.array(my_data)
    >>> covars = pd.read_csv('subject_info.csv')
    >>> # run harmonization and store the adjusted data
    >>> my_model, my_data_adj = harmonizationLearn(my_data, covars)

Applying Pre-Trained Models to New Data
---------------------------------------

``harmonizationApply``

Specifying Nonlinear Covariate Effects
--------------------------------------

Optional argument: ``smooth_terms``

Working with NIFTI Images
-------------------------

*In development*


.. [1] W. Evan Johnson and Cheng Li, Adjusting batch effects in microarray expression data
   using empirical Bayes methods. Biostatistics, 8(1):118-127, 2007.

    
