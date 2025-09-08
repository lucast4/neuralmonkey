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

def plot_ols_results(model, ci=True, alpha=0.05, figsize=(8, 5)):
    """
    Plot OLS regression coefficient estimates with confidence intervals or standard errors.

    Plots results from: fit_and_score_regression_with_categorical_predictor()
    
    Parameters:
    - model: a fitted statsmodels OLS model object.
    - ci (bool): If True, plot confidence intervals. If False, plot ±1 standard error.
    - alpha (float): significance level, for what to color pvals in plots
    - figsize (tuple): size of the plot.
    """
    # Extract values
    summary_df = model.summary2().tables[1]
    summary_df = summary_df.rename(columns={
        'Coef.': 'coef',
        'Std.Err.': 'se',
        '[0.025': 'ci_lower',
        '0.975]': 'ci_upper',
        'P>|t|': 'pval'
    })

    # Add column for error bars
    if ci:
        lower = summary_df['ci_lower']
        upper = summary_df['ci_upper']
        error_lower = summary_df['coef'] - lower
        error_upper = upper - summary_df['coef']
    else:
        error_lower = summary_df['se']
        error_upper = summary_df['se']

    # Prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(summary_df))
    ax.errorbar(summary_df['coef'], y_pos,
                xerr=[error_lower, error_upper],
                fmt='o', capsize=5, color='black')

    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df.index)
    ax.set_xlabel('Coefficient Estimate')
    ax.invert_yaxis()  # Highest term on top

    # Annotate with p-values
    for i, pval in enumerate(summary_df['pval']):
        if pval < alpha:
            color = "r"
        else:
            color = "b"
        ax.text(summary_df['coef'].iloc[i], i,
                f"p={pval:.3g}",
                va='center', ha='left' if summary_df['coef'].iloc[i] >= 0 else 'right',
                fontsize=9, color=color, alpha=0.5)

    return fig

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


def kernel_ordinal_logistic_regression(X, y, rescale_std=True, PLOT=False, do_grid_search=True,
                                       grid_n_splits=3, apply_kernel=True):
    """
    Ordinal logistic regressino, with option (defualt) to use kernel transformation, which is
    useful if you have non-linear relationship between X and y. 

    y is ordinal (0, 1, 2, 3), and X is continuously varying data
    
    PARAMS:
    - X, (ntrials, ndims)
    - y, (ntrials), ordered labels, must be integers. They must be 0, 1, 2..., (ie no negative, no gaps)
    - rescale_std, if True, then z-scores. If False, then just demeans.
    - apply_kernel, bool, whether to apply kernel to allow nonlinear mapping

    NOTE:
    - Given returned model, res["model"], can score any new data in same space as input X by running
    y_pred = model.predict(X)
    """
    import numpy as np
    # import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    # from sklearn.metrics import accuracy_score
    # from scipy.stats import spearmanr
    from sklearn.metrics import pairwise_distances, balanced_accuracy_score
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.kernel_approximation import Nystroem  # or RBFSampler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    import mord  # ordered logit

    assert all([isinstance(yy, (int, np.integer)) for yy in y])
    if True:
        # Yes, this must pass or else will fail
        assert np.all(np.diff(sorted(set(y)))==1)
    # assert min(y)==0
    assert len(set(y))>1

    ### Determine CV params
    # First, determine length scale (gamma, inverse radius), based on heuristic using the 
    # median inter-point distance
    D2 = pairwise_distances(X, metric="sqeuclidean")
    median_d2 = np.median(D2)
    gamma0 = 1.0 / median_d2
    gammas = gamma0 * np.logspace(-1.5, 1.5, 5)

    # n_components (cannot be more than the n samples, or else warning)
    n_samples = X.shape[0]
    ker__n_components = [x for x in [2, 4, 8, 16, 32, 64, 128] if x < 0.8*n_samples]
    ker__n_components = ker__n_components[-4:]
    n_components0 = max([x for x in [2, 4, 8, 16, 32, 64] if x<0.8*n_samples])

    ### Pipeline
    if apply_kernel:
        steps = [
            ('sc', StandardScaler(with_std=rescale_std)),
            ('ker', Nystroem(kernel='rbf', gamma=gamma0, n_components=n_components0, random_state=None)),
            ('ord', mord.LogisticIT(alpha=1.0))  # proportional odds (ordered logit)
        ]
    else:
        steps = [
            ('sc', StandardScaler(with_std=rescale_std)),
            ('ord', mord.LogisticIT(alpha=1.0))  # proportional odds (ordered logit)
        ]
    pipe = Pipeline(steps)

    if do_grid_search:

        if False:
            print("Median heuristic gamma:", gamma0)
            print("Gamma grid:", gammas)
        
        if apply_kernel:
            param_grid = {
                # 'ker__gamma': [0.1, 0.3, 1.0, 3.0],
                'ker__gamma': gammas,
                'ker__n_components': ker__n_components,
                'ord__alpha': [0.05, 0.2, 1.0, 5.0],
            }
        else:
            param_grid = {
                'ord__alpha': [0.05, 0.2, 1.0, 5.0],
            }
        # print(param_grid)

        ### Do grid-search
        cv = StratifiedKFold(n_splits=grid_n_splits, shuffle=True, random_state=None)
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        gs.fit(X, y)

        ### Return results
        best_params = gs.best_params_
        model = gs.best_estimator_
    else:
        model = pipe
        best_params = None
        model.fit(X, y)

    y_pred = model.predict(X)

    # Compute the latent score manually
    X_trans = model[:-1].transform(X) # Get transformed features (after scaler/kernel)
    ord_model = model.named_steps['ord']
    s = X_trans @ ord_model.coef_
    # theta = model.named_steps['ord'].theta_        # thresholds separating classes

    # Get score
    score = balanced_accuracy_score(y, y_pred)

    res = {
        "cv_best_params":best_params,
        "model":model,
        "y_pred":y_pred,
        "s":s, # latent state
        "score":score,
        "coeff":model[-1].coef_, # The coef_ attribute contains the weight vector for the features. It has a shape of (n_features,).
        "theta":model[-1].theta_, # The theta_ attribute contains the thresholds for the class boundaries. For K classes, there will be K-1 thresholds.
    }

    if PLOT:
        fig = kernel_ordinal_logistic_regression_plot(X, y, res)
        return res, fig
    else:
        return res

def kernel_ordinal_logistic_regression_plot(X, y, res):
    """
    Plot results of Ordinal logistic regressino, gotten from 
    kernel_ordinal_logistic_regression()
    """
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.metrics import balanced_accuracy_score

    # s = res["s"]
    # y_pred = res["y_pred"]
    # score = res["score"]
    model = res["model"]
    X_trans = model[:-1].transform(X) # Get transformed features (after scaler/kernel)
    ord_model = model.named_steps['ord']
    s = X_trans @ ord_model.coef_ # Latent 1D variable.
    y_pred = model.predict(X)
    score = balanced_accuracy_score(y, y_pred)

    # Plot just the first 2 dimensions
    S = 4
    fig, axes = plt.subplots(1, 7, figsize=(7*S, S))

    ax = axes.flatten()[0]
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue=y, ax=ax, alpha=0.65)
    ax.set_title("original labels")

    ax = axes.flatten()[1]
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue=s, ax=ax, alpha=0.65)
    ax.set_title("latent state (ord regress)")

    # ---- 2. Simple geometry test: first PC projection vs. ordinal labels ----
    pca = PCA(n_components=1)
    s_pca = pca.fit_transform(X).ravel()
    # rho, _ = spearmanr(s, y)
    ax = axes.flatten()[2]
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue=s_pca, ax=ax, alpha=0.65)
    ax.set_title("latent state (1D PCA)")

    ax = axes.flatten()[3]
    sns.scatterplot(x = X[:, 0], y = X[:, 1], hue=y_pred, ax=ax, alpha=0.65)
    ax.set_title("predicted labels")

    ax = axes.flatten()[4]
    sns.histplot(x=y, y=y_pred, ax=ax)
    # ax.scatter(y, y_pred, c=s)
    ax.set_xlabel("y actual")
    ax.set_ylabel("y pred")
    ax.set_title(f"score: {score:.2f}")

    # Also plot on a line, relative to the latent variable.
    ax = axes.flatten()[5]
    sns.histplot(x=s, hue=y, element="poly", ax=ax)
    ax.set_xlabel("s (latent variable)")
    ax.set_title("actual labels")
    
    ax = axes.flatten()[6]
    sns.histplot(x=s, hue=y_pred, element="poly", ax=ax)
    ax.set_xlabel("s (latent variable)")
    ax.set_title("predicted labels")
    
    return fig
    
def _kernel_ordinal_logistic_regression_example(rescale_std=True):
    """
    Simulate data and run example
    """

    # ---- 1. Simulate neural-like data on a curved 2D manifold ----
    # We'll embed ordinal categories along a nonlinear curve (arc of a circle)
    rng = np.random.default_rng(0)
    n_per_class = 80

    # Arc angles for 3 ordinal categories: bad, ok, good
    angles = {
        0: rng.normal(loc=0.2*np.pi, scale=0.05, size=n_per_class),
        1: rng.normal(loc=0.5*np.pi, scale=0.05, size=n_per_class),
        2: rng.normal(loc=0.8*np.pi, scale=0.05, size=n_per_class),
        3: rng.normal(loc=1.5*np.pi, scale=0.05, size=n_per_class),
        4: rng.normal(loc=1.2*np.pi, scale=0.05, size=n_per_class),
        5: rng.normal(loc=0.65*np.pi, scale=0.05, size=n_per_class),
    }

    X = []
    y = []
    for label, angs in angles.items():
        for a in angs:
            x = np.array([np.cos(a), np.sin(a)])  # points on circle
            x += 0.05 * rng.standard_normal(2)    # small noise
            X.append(x)
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    return kernel_ordinal_logistic_regression(X, y, rescale_std, PLOT=True)

def formula_string_construct(var_response, variables, variables_is_cat, exclude_var_response=False):
    """
    For statsmodels
    Create formula string for regression.
    PARAMS:
    - var_response, string
    - variables, list of variable strings.
    - variables_is_cat, list of bool, if each variable is categorical(True) or continuous.
    - exclude_var_response, if True, then returns string like: 'motor_onsetx +  motor_onsety +  gap_from_prev_x +  gap_from_prev_y +  velmean_x +  velmean_y +  C(gridloc) +  C(DIFF_gridloc) +  C(chunk_rank) +  C(shape) +  C(chunk_within_rank_fromlast)'
    Is like:
    'frate ~ motor_onsetx +  motor_onsety +  gap_from_prev_x +  gap_from_prev_y +  velmean_x +  velmean_y +  C(gridloc) +  C(DIFF_gridloc) +  C(chunk_rank) +  C(shape) +  C(chunk_within_rank_fromlast)'
    """
    ### Construct formula string
    # list_feature_names = []
    if exclude_var_response:
        func = ""
    else:
        func = f"{var_response} ~"
        
    for var, var_is_cat in zip(variables, variables_is_cat):
        if var_is_cat == False:
            func += f" {var} + "
            # list_feature_names.append(var)
    for var, var_is_cat in zip(variables, variables_is_cat):
        if var_is_cat == True:
            func += f" C({var}) + "
            # list_feature_names.append(var)
    
    # remove the + at the end
    func = func[:-3]
    
    # Remove empty space
    if exclude_var_response:
        func = func[1:]
        
    return func
