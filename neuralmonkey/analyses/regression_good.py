"""
Stuff to 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.pandastools import append_col_with_grp_index

def fit_and_score_regression(X_train, y_train, X_test=None, y_test=None, 
                             do_upsample=False, version="ridge", PRINT=False,
                             ridge_alpha=1):
    """
    [GOOD] Generic train/test for OLS regression
    PARAMS:
    - X_train, (ndat, nfeat)
    - y_train, (ndat,)
    ...

    """
    ### Fit regression here.
    from sklearn.linear_model import LinearRegression, Ridge

    if do_upsample:
        from pythonlib.tools.statstools import decode_resample_balance_dataset
        # balance the dataset
        X_train, y_train = decode_resample_balance_dataset(X_train, y_train, plot_resampled_data_path_nosuff=None)
        # print(x.shape, y.shape)
        # print(x_resamp.shape, y_resamp.shape)
        # x = x_resamp
        # y = y_resamp

    if version=="ols":
        reg = LinearRegression()
    elif version=="ridge":
        reg = Ridge(alpha=ridge_alpha)
    else:
        print(version)
        assert False

    reg.fit(X_train, y_train)
    r2_train = reg.score(X_train, y_train)
    # print(reg.coef_)
    # print(reg.intercept_)

    ### Test
    if X_test is not None:
        r2_test = reg.score(X_test, y_test)
    else:
        r2_test = None

    if PRINT:
        print("r2_train: ", r2_train)
        print("r2_test: ", r2_test)

    return reg, r2_train, r2_test


def ordinal_fit_and_score_train_test_splits(X, y_ordinal, max_nsplits=None, expected_n_min_across_classes=2):
    """
    To train and test on a dataset, oridnal y.
    
    THe reason this is specific to ordinal is that it does stratified train/test splits.

    PARAMS:
    - X, (ndat, nfeat)
    - y_ordina, (ndat,) ordinal (e.g., 1,2,3, ..)
    """
    from sklearn.model_selection import StratifiedKFold
    from pythonlib.tools.listtools import tabulate_list

    ## PREP PARAMS
    # Count the lowest n data across classes.
    n_min_across_labs = min(tabulate_list(y_ordinal).values())
    n_max_across_labs = max(tabulate_list(y_ordinal).values())

    if max_nsplits is None:
        max_nsplits = 30
    n_splits = min([max_nsplits, n_min_across_labs]) # num splits. max it at 12...

    # Check that enough data
    if n_min_across_labs<expected_n_min_across_classes:
        print(n_min_across_labs)
        print(expected_n_min_across_classes)
        assert False

    ######################## RUN for each split
    RES = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(y_ordinal)), y_ordinal)):
        # Each fold is a unique set of test idnices, with this set as small as posibiel whiel still having at laset
        # 1 datapt fore ach class.

        # print(" fold ", i)
        X_train = X[train_index, :]
        y_train = [y_ordinal[i] for i in train_index]
        X_test = X[test_index, :]
        y_test = [y_ordinal[i] for i in test_index]

        reg, r2_train, r2_test = fit_and_score_regression(X_train, y_train, X_test, y_test)


        # Save
        RES.append({
            "reg":reg,
            "iter_kfold":i,
            "r2_test":r2_test,
            "n_dat":len(y_ordinal),
            "n_splits":n_splits,
            "n_min_across_labs":n_min_across_labs,
            "n_max_across_labs":n_max_across_labs
        })

    dfres = pd.DataFrame(RES)
    r2_test_mean = np.mean(dfres["r2_test"])

    return dfres, r2_test_mean
