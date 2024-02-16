"""
For decoding variables from population activity (and regression to predict continuosu variables),

Devo in notebook: 240128_snippets_demixed_PCA

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig

def decode_categorical_wrapper(X, labels, expected_n_min_across_classes,
                               plot_resampled_data_path_nosuff=None, max_nsplits=12,
                               do_center=True, do_std=False):
    """
    Helper to run multiple train-test splits (Kfold), including rebalancing data if labels are unbalanced, and
    then scoring using a balanced accuracty score.
    PARAMS;
    - X, predictor data, in shape (nsamps, nfeats)
    - labels, usually ints or str, len nsamps.
    - expected_n_min_across_classes, int, used as a sanity check, fails if any class has less than this.
    - plot_resampled_data_path_nosuff, str, path to save scatter plots of data plottied in first 2 dims,
    without extension(suffix).
    RETURNS:
    - res, list of dicts, each for a single split, holding params and results for that split,
    e.g., {'iter_kfold': 0,
         'score_xval': 0.5069444444444444,
         'score_xval_adjusted': 0.34259259259259256,
         'n_dat': 422,
         'n_splits': 12,
         'n_min_across_labs': 91,
         'n_max_across_labs': 113}
    NOTE: Splits must be done BEFORE upsampling. otherwise, test data will contribute to the upsampling and
    therefore leak into training data.
    """
    from pythonlib.tools.nptools import bin_values_categorical_factorize
    from pythonlib.tools.listtools import tabulate_list
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    ## PREP PARAMS
    # Count the lowest n data across classes.
    n_min_across_labs = min(tabulate_list(labels).values())
    n_max_across_labs = max(tabulate_list(labels).values())

    list_dims_plot = [(0,1), (2,3), (4,5), (6,7)]
    list_dims_plot = [dims for dims in list_dims_plot if X.shape[1]>max(dims)]

    n_splits = min([max_nsplits, n_min_across_labs]) # num splits. max it at 12...
    # nclasses = len(set(labels))

    # Check that enough data
    if n_min_across_labs<expected_n_min_across_classes:
        print(n_min_across_labs)
        print(expected_n_min_across_classes)
        assert False

    # Check that enough data
    if False:
        # Rebalance the data by throwing out levels without neough data.
        # # Not needed here, since its done above in extract...
        from pythonlib.tools.pandastools import extract_with_levels_of_var_good
        dflab = panorm_scal.Xlabels["trials"]
        # n_min_per_lev = 8
        _, inds_keep = extract_with_levels_of_var_good(dflab, var_decode, None, n_min_per_lev)
        # Prune data using these indices
        X = X[inds_keep, :]
        labels = dflab[var_decode].values
        labels = labels[inds_keep]

    ######################## RUN for each split
    RES = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # skf.get_n_splits(X, labels)
    for i, (train_index, test_index) in enumerate(skf.split(X, labels)):
        # Each fold is a unique set of test idnices, with this set as small as posibiel whiel still having at laset
        # 1 datapt fore ach class.

        X_train = X[train_index]
        labels_train = labels[train_index]
        X_test = X[test_index]
        labels_test = labels[test_index]

        if False:
            print("---------")
            print("ntot:", len(labels))
            print("n train", len(train_index))
            print("n test", len(test_index))
            print("n splits", n_splits)
            print("n classes", len(set(labels)))

        # Do upsampling here, since it is hleped by having more data
        # Balance the train data
        if n_min_across_labs/n_max_across_labs<0.8:
            from imblearn.over_sampling import SMOTE
            n_min_across_labs = min(tabulate_list(labels_train).values())
            smote = SMOTE(sampling_strategy="not majority", k_neighbors=n_min_across_labs-1)
            _x_resamp, _lab_resamp = smote.fit_resample(X_train, labels_train)

            # Plot
            if plot_resampled_data_path_nosuff is not None and i==0:
                # Only do for i==0 since each time is very similar (trains are overlapoing).
                from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotbydims_scalar

                fig, axes = trajgood_plot_colorby_splotbydims_scalar(X_train, labels_train, list_dims_plot);
                savefig(fig, f"{plot_resampled_data_path_nosuff}-raw-xvalfold_{i}.png")

                fig, axes = trajgood_plot_colorby_splotbydims_scalar(_x_resamp, _lab_resamp, list_dims_plot);
                savefig(fig, f"{plot_resampled_data_path_nosuff}-upsampled-xvalfold_{i}.png")
            X_train = _x_resamp
            labels_train = _lab_resamp

        # Fit model (Classifier)
        try:
            from neuralmonkey.population.classify import _model_fit
            model_params_optimal = {"C":0.01} # optimized regularization params
            mod, _ = _model_fit(X_train, labels_train, model_params=model_params_optimal,
                                do_center=do_center, do_std=do_std, do_train_test_split=False)
        except Exception as err:
            # Plot the data then raise error
            fig, axes = trajgood_plot_colorby_splotbydims_scalar(X_train, labels_train, list_dims_plot);
            path = f"{plot_resampled_data_path_nosuff}-raw-xvalfold_{i}.png"
            savefig(fig, path)
            print("PLOTTED DATA THAT LED TO FAILURE AT:", f"{plot_resampled_data_path_nosuff}-raw-xvalfold_{i}.png")
            raise err

        ################# Test on held out
        labels_predicted = mod.predict(X_test)
        score = balanced_accuracy_score(labels_test, labels_predicted, adjusted=False)
        score_adjusted = balanced_accuracy_score(labels_test, labels_predicted, adjusted=True)

        # Save
        RES.append({
            "iter_kfold":i,
            "score_xval":score,
            "score_xval_adjusted":score_adjusted,
            "n_dat":len(labels),
            "n_splits":n_splits,
            "n_min_across_labs":n_min_across_labs,
            "n_max_across_labs":n_max_across_labs
        })

    return RES