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


def fit_and_score_regression_with_categorical_predictor(data_train, y_var, x_vars, x_vars_is_cat, data_test, 
                                                        PRINT=False, demean_y = True):
    """
    More flexible version compared to fit_and_score_regression -- here can use combination of continuos and
    categorical variable predictors.
    PARAMS:
    - data_train, each row is observation, columns are variables.
    - y_var, string, column
    - x_vars, list of strings
    - x_vars_is_cat, list of bools, whether to treat each x_var as categorical (True).
    - demean_y, bool, this only affects the returned values, by offsetting within each of train and test, the
    mean value for y. This is to allow demean before collecting across dimensions, to be fair, since 
    predicted values are allowed to have different intercepts per dimension. In other words, this is like applying 
    the model that takes the mean y.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from pythonlib.tools.statstools import coeff_determination_R2

    # print("---")
    # display(data_train[y_var].value_counts())
    # display(data_test[y_var].value_counts())

    assert len(x_vars)==len(x_vars_is_cat)

    ### Construct function string
    # list_feature_names = []
    func = f"{y_var} ~"
    for var, var_is_cat in zip(x_vars, x_vars_is_cat):
        if var_is_cat == False:
            func += f" {var} + "
            # list_feature_names.append(var)
    for var, var_is_cat in zip(x_vars, x_vars_is_cat):
        if var_is_cat == True:
            func += f" C({var}) + "
            # list_feature_names.append(var)
    # remove the + at the end
    func = func[:-3]

    ### Run regression
    model = smf.ols(func, data=data_train).fit()

    # Extract the coefficients
    feature_names = model.params.index.tolist()
    coef_array = model.params.values  # shape (1, nfeat)

    dict_coeff = {f:c for f, c in zip(feature_names, coef_array)}

    # Map from dummy variables back to original variables
    original_feature_mapping = {}
    for feat in feature_names:
        if 'C(' in feat:
            # e.g., feat = 'C(gender)[T.male]'
            base = feat.split('[')[0]  # 'C(gender)'
            base = base.replace('C(', '').replace(')', '')  # 'gender'
            original_feature_mapping[feat] = base
        else:
            original_feature_mapping[feat] = feat

    if PRINT:
        print(model.summary())
        print(feature_names)
        print(coef_array)   

    # display(data_train)
    # display(data_test)

    ### Get details of predictions
    # - Train
    y_train_pred = model.predict(data_train).values
    y_train = data_train[y_var].values
    # - Test
    y_test_pred = model.predict(data_test).values
    y_test = data_test[y_var].values

    if demean_y:
        # Demean before collecting across dimensions, to be fair, since predicted values are allowed to have
        # different intercepts per dimension.
        y_train_mean = np.mean(y_train)
        y_train  = y_train - y_train_mean
        y_train_pred = y_train_pred - y_train_mean

        y_test_mean = np.mean(y_test)
        y_test = y_test - y_test_mean
        y_test_pred = y_test_pred - y_test_mean

    r2_train, _, _, ss_resid_train, ss_tot_train = coeff_determination_R2(y_train, y_train_pred, doplot=False, return_ss=True)
    r2_test, _, _, ss_resid_test, ss_tot_test = coeff_determination_R2(y_test, y_test_pred, doplot=False, return_ss=True)

    results = {
        "r2_train":r2_train,
        "r2_test":r2_test,
        "y_train":y_train,
        "y_test":y_test,
        "y_train_pred":y_train_pred,
        "y_test_pred":y_test_pred,
        "ss_resid_train":ss_resid_train,
        "ss_tot_train":ss_tot_train,
        "ss_resid_test":ss_resid_test,
        "ss_tot_test":ss_tot_test,
    }

    return dict_coeff, model, original_feature_mapping, results

def fit_and_score_regression(X_train, y_train, X_test=None, y_test=None, 
                             do_upsample=False, version="ridge", PRINT=False,
                             ridge_alpha=1, demean=True, also_return_predictions=False):
    """
    [GOOD] Generic train/test for OLS regression
    PARAMS:
    - X_train, (ndat, nfeat)
    - y_train, (ndat,)
    - demean, bool, demean X data using the mean from training. Doesnt affect r2.
    """
    ### Fit regression here.
    from sklearn.linear_model import LinearRegression, Ridge

    if demean:
        if X_test is not None:
            # Then demean both training and testing together
            xmean = np.mean(np.concatenate([X_train, X_test], axis=0), axis=0, keepdims=True)
            X_train = X_train-xmean
            X_test = X_test-xmean

            ymean = np.mean(np.concatenate([y_train, y_test], axis=0), axis=0, keepdims=True)
            y_train = y_train-ymean
            y_test = y_test-ymean
        else:
            # Then demean both training and testing together
            xmean = np.mean(X_train, axis=0, keepdims=True)
            X_train = X_train-xmean

            ymean = np.mean(y_train, axis=0, keepdims=True)
            y_train = y_train-ymean

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

    # Also return the predictions and residuals.
    ### Test
    if X_test is not None:
        r2_test = reg.score(X_test, y_test)
    else:
        r2_test = None
    
    if PRINT:
        print("r2_train: ", r2_train)
        print("r2_test: ", r2_test)

    if also_return_predictions:
        y_train_pred = reg.predict(X_train)
        if X_test is not None:
            y_test_pred = reg.predict(X_test)
        else:
            y_test_pred = None

        # and also return values, which may have been demeaned
        return reg, r2_train, r2_test, y_train, y_test, y_train_pred, y_test_pred

    else:
        # Just return the stats
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
