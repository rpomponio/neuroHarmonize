neuroHarmonize
--------------

Harmonization tools for multi-center neuroimaging studies.

Overview
---------

This package extends the functionality of the package developed by Nick Cullen,
neuroCombat, which is hosted on GitHub: https://github.com/ncullen93/neuroCombat
(To make installation easier, neuroCombat is not a dependency for this package,
but the source code is included to call neuroCombat functions)

Specifically, the user is able to perform the following procedures using this
package:

1. Train a harmonization model on a subset of data, then apply the model to the
whole set (e.g. longitudinal analysis may require harmonization, but one should
train on baseline images and avoid training on the entire set of images).

2. Specify covariates with nonlinear effects. For example, age tends to exhibit
a nonlinear relationship with brain volumes, particularly in developmental and
aging cohorts. Nonlinear effects are implemented using generalized additive
models (GAMs).

3. Apply a harmonization model to NIFTI images. In cases where loading the
entire set of images would exceed memory capacity, it is still possible to
harmonize the images by sequentially loading and adjusting images one-by-one.
This functionality is available in this package.

Installation
------------

Option 1: Install from GitHub

Download the zipped repository on GitHub: https://github.com/rpomponio/neuroHarmonize

Unzip the download. Navigate to the directory neuroHarmonize-master/

Open a terminal and run:

    >>> pip install .

Option 2: Install from PyPI

(instructions will be written once package is published)

Quick Start
-----------

If you want to harmonize a dataset of brain volumes from multiple sites, you
begin by loading the brain data as a numpy array. Next, you load covariates
as a pandas DataFrame. The covariates DataFrame must include one column called
"SITE", which indicates the site from each each sample came from.

Example use:

    >>> from neuroHarmonize import harmonizationLearn
    >>> import pandas as pd
    >>> import numpy as np
    >>> # load your data and all numeric covariates
    >>> my_data = pd.read_csv('brain_volumes.csv')
    >>> my_data = np.array(my_data)
    >>> covars = pd.read_csv('subject_info.csv')
    >>> # run harmonization and store the adjusted data
    >>> my_model, my_data_adj = harmonizationLearn(my_data, covars)
    
