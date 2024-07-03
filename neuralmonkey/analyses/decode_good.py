"""
For decoding variables from population activity (and regression to predict continuosu variables),
Only for categorical varialbes.

Devo in notebook: 240128_snippets_demixed_PCA

Scripts:
- analy_decode_script.py (good)
- (There is also a substrokes decode script which DOES NOT use stuff here, it was older. It should be updated to use this)

"""
from neuralmonkey.utils.frmat import bin_frmat_in_time
from neuralmonkey.classes.population_mult import extract_single_pa
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.pandastools import append_col_with_grp_index

LABELS_IGNORE = ["IGN", ("IGN",), "IGNORE", ("IGNORE",)] # values to ignore during dcode.
N_MIN_TRIALS = 5 # min trials per level, otherwise throws level out.

def decode_apply_model_to_test_data(mod, x_test, labels_test,
                                    return_predictions_all_trials=False):
    """ Apply a trained linear SVC model to score decoding
    accuracy given new test data. Must prune input to only have
    classes present in both test abd predicted data.
    PARAMS:
    - x_test, (ntrials, ndims)
    - labels_test (ntrials)
    """
    from sklearn.metrics import balanced_accuracy_score

    x_test, labels_test, inds_keep = cleanup_remove_labels_ignore(x_test, labels_test)
    assert x_test.shape[0]==len(labels_test)

    labels_predicted = mod.predict(x_test)

    classes_in_ypred_but_not_in_ytrue = [x for x in set(labels_predicted) if x not in set(labels_test)]
    if len(classes_in_ypred_but_not_in_ytrue)>0:
        print(classes_in_ypred_but_not_in_ytrue)
        print(set(labels_predicted))
        print(set(labels_test))
        assert False, "should first prune trainig data so that it onlyincludes classes present in testing data"
    score = balanced_accuracy_score(labels_test, labels_predicted, adjusted=False)
    score_adjusted = balanced_accuracy_score(labels_test, labels_predicted, adjusted=True)

    if return_predictions_all_trials:
        # Also get conficence scores.
        conf_scores = mod.decision_function(x_test) # (ntrials, nclasses), distance of samp from
        # decision boundary

        return score, score_adjusted, labels_predicted, labels_test, conf_scores
    else:
        return score, score_adjusted

def decode_upsample_dataset(X_train, labels_train, plot_resampled_data_path_nosuff=False):
    """
    Upsample data to balance n datapts across classes of labels.
    PARAMS:
    - X_train, scalar acgivity to decode, a single time bin, (ntrials, nchans)
    - labels_train, list of cat valuyes. len ntrials
    """
    from pythonlib.tools.listtools import tabulate_list
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotbydims_scalar

    list_dims_plot = [(0,1), (2,3), (4,5), (6,7)]
    list_dims_plot = [dims for dims in list_dims_plot if X_train.shape[1]>max(dims)]

    n_min_across_labs = min(tabulate_list(labels_train).values())
    n_max_across_labs = max(tabulate_list(labels_train).values())

    # Skip, if only 1 class exists...
    if len(set(labels_train))==1:
        print("SKipping, ony one class lewvel found... [decode_train_model]")
        return None

    if False:
        print("---------")
        print("ntot:", len(labels))
        print("n train", len(train_index))
        print("n test", len(test_index))
        print("n splits", n_splits)
        print("n classes", len(set(labels)))

    # Do upsampling here, since it is hleped by having more data
    # Balance the train data
    if n_min_across_labs/n_max_across_labs<0.85:
        print("[Upsampling] Across levels, nmin_dat / nmax_dat:", n_min_across_labs, n_max_across_labs)
        from imblearn.over_sampling import SMOTE
        n_min_across_labs = min(tabulate_list(labels_train).values())
        smote = SMOTE(sampling_strategy="not majority", k_neighbors=n_min_across_labs-1)
        _x_resamp, _lab_resamp = smote.fit_resample(X_train, labels_train)

        # Plot
        if plot_resampled_data_path_nosuff is not None:
            # Only do for i==0 since each time is very similar (trains are overlapoing).

            fig, axes = trajgood_plot_colorby_splotbydims_scalar(X_train, labels_train, list_dims_plot)
            savefig(fig, f"{plot_resampled_data_path_nosuff}-raw.png")

            fig, axes = trajgood_plot_colorby_splotbydims_scalar(_x_resamp, _lab_resamp, list_dims_plot)
            savefig(fig, f"{plot_resampled_data_path_nosuff}-upsampled.png")
        X_train = _x_resamp
        labels_train = _lab_resamp

    return X_train, labels_train


def decode_train_model(X_train, labels_train, plot_resampled_data_path_nosuff=None,
                       do_center=True, do_std=False):
    """
    Good helper to train linearSVC decoder (without scoring), with resampling in ordert to balance across
    labels, if needed, for a single time step.
    PARAMS:
    - X_train, scalar acgivity to decode, a single time bin, (ntrials, nchans)
    - labels_train, list of cat valuyes. len ntrials
    RETURNS:
        - mod
    """
    from pythonlib.tools.listtools import tabulate_list
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotbydims_scalar

    assert len(X_train.shape)==2
    assert X_train.shape[0]==len(labels_train)

    X_train, labels_train, inds_keep = cleanup_remove_labels_ignore(X_train, labels_train)

    if X_train is None or len(labels_train)==0:
        return None

    # Skip, if only 1 class exists...
    if len(set(labels_train))==1:
        print("SKipping, ony one class lewvel found... [decode_train_model]")
        return None

    if False:
        print("---------")
        print("ntot:", len(labels))
        print("n train", len(train_index))
        print("n test", len(test_index))
        print("n splits", n_splits)
        print("n classes", len(set(labels)))

    # Do upsampling here, since it is hleped by having more data
    # Balance the train data
    X_train, labels_train = decode_upsample_dataset(X_train, labels_train, "/tmp/test")
    # list_dims_plot = [(0,1), (2,3), (4,5), (6,7)]
    # list_dims_plot = [dims for dims in list_dims_plot if X_train.shape[1]>max(dims)]

    # n_min_across_labs = min(tabulate_list(labels_train).values())
    # n_max_across_labs = max(tabulate_list(labels_train).values())
    # if n_min_across_labs/n_max_across_labs<0.85:
    #     from imblearn.over_sampling import SMOTE
    #     n_min_across_labs = min(tabulate_list(labels_train).values())
    #     smote = SMOTE(sampling_strategy="not majority", k_neighbors=n_min_across_labs-1)
    #     _x_resamp, _lab_resamp = smote.fit_resample(X_train, labels_train)

    #     # Plot
    #     if plot_resampled_data_path_nosuff is not None:
    #         # Only do for i==0 since each time is very similar (trains are overlapoing).

    #         fig, axes = trajgood_plot_colorby_splotbydims_scalar(X_train, labels_train, list_dims_plot)
    #         savefig(fig, f"{plot_resampled_data_path_nosuff}-raw.png")

    #         fig, axes = trajgood_plot_colorby_splotbydims_scalar(_x_resamp, _lab_resamp, list_dims_plot)
    #         savefig(fig, f"{plot_resampled_data_path_nosuff}-upsampled.png")
    #     X_train = _x_resamp
    #     labels_train = _lab_resamp


    # Fit model (Classifier)
    try:
        from neuralmonkey.population.classify import _model_fit
        model_params_optimal = {"C":0.01} # optimized regularization params
        mod, _ = _model_fit(X_train, labels_train, model_params=model_params_optimal,
                            do_center=do_center, do_std=do_std, do_train_test_split=False)
    except Exception as err:
        # Plot the data then raise error
        fig, axes = trajgood_plot_colorby_splotbydims_scalar(X_train, labels_train, list_dims_plot)
        if plot_resampled_data_path_nosuff is None:
            plot_resampled_data_path_nosuff = "/tmp/test"
        path = f"{plot_resampled_data_path_nosuff}-raw.png"
        savefig(fig, path)
        print("PLOTTED DATA THAT LED TO FAILURE AT:", f"{plot_resampled_data_path_nosuff}-raw.png")
        raise err

    return mod


def decode_categorical_within_condition(X, dflab, var_decode, vars_conj_condition,
                                       do_center=True, do_std=True,
                                        max_nsplits=None):
    """
    Decode <var_decode> separately for each level of <vars_conj_condition>, and
    return the average decoding across levels.
    PARAMS:
    - X, predictor data, in shape (nsamps, nfeats)
    - dflab, dataframe holding labels, each a column
    - var_decode, string, variable to decode.
    - vars_conj_condition, list of str, conjuctive variable that will condition decoder on
    vars_conj_condition = ["seqc_0_loc"]
    """
    from pythonlib.tools.listtools import tabulate_list
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

    assert X.shape[0]==len(dflab)
    assert len(X.shape)==2

    # First facotrize the l;abels, BNEFORE splitting be,low or else bug.
    labels_all = dflab[var_decode].tolist()

    X, labels_all, inds_keep = cleanup_remove_labels_ignore(X, labels_all)
    dflab = dflab.iloc[inds_keep].copy().reset_index(drop=True)

    # Get all levels of conditioned var
    groupdict = grouping_append_and_return_inner_items(dflab, vars_conj_condition)

    # Iterate over all levels of conditioned grp vars
    RES = []
    for grp_condition, inds_train in groupdict.items():

        # This group provides training data
        X_train = X[inds_train, :] # (ntrials, nchans)
        labels_train = [labels_all[i] for i in inds_train]

        train_labels_counts = tabulate_list(labels_train)

        if len(set(labels_train))>1:
            # Train/test decoder
            score, score_adjusted = decode_categorical(X_train, labels_train, min(train_labels_counts.values()),
                               max_nsplits=max_nsplits, do_center=do_center, do_std=do_std,
                               return_mean_score_over_splits=True)

            # Save results
            RES.append({
                "var_decode":var_decode,
                "vars_conj_condition":tuple(vars_conj_condition),
                "grp_condition":grp_condition,
                "score":score,
                "score_adjusted":score_adjusted,
                "train_labels_counts":train_labels_counts,
                "n_classes_test":len(train_labels_counts)
            })

    dfres = pd.DataFrame(RES)

    # aggregate, to get one score for each var_decode
    from pythonlib.tools.pandastools import aggregGeneral
    dfres_agg = aggregGeneral(dfres, ["var_decode", "vars_conj_condition"], values=["score", "score_adjusted", "n_classes_test"])
    # dfres_agg = dfres.groupby(["var_decode"]).mean()
    if False: # Skip this for now... later on will have to deal with this issue of diff n classes across groups.
        tmp = dfres.groupby(["var_decode"]).std()
        if np.any(tmp["n_classes_test"]>0.):
            print(dfres)
            print(dfres["n_classes_test"].value_counts())
            assert False, "diff n classes across tests (groups of othervar). Solve by restrticting to same n classes?"

    return dfres, dfres_agg


def decode_categorical_cross_condition(X, dflab, var_decode, vars_conj_condition,
                                       do_center=True, do_std=False):
    """
    Decode <var_decode> in a cross-condition manner, i.e,, train on on level of
    <vars_conj_condition> and test tjhis decoder on data from every other level of
    <vars_conj_condition> separately. Applies same normalization to data across levels,
    for a single time step.
    PARAMS:
    - X, predictor data, in shape (nsamps, nfeats)
    - dflab, dataframe holding labels, each a column
    - var_decode, string, variable to decode.
    - vars_conj_condition, list of str, conjuctive variable that will condition decoder on
    vars_conj_condition = ["seqc_0_loc"]
    """

    # labels_decode = pd.factorize(dflab[var_decode])[0]
    # labels_condition = pd.factorize(dflab[var_condition])[0]
    from sklearn.metrics import balanced_accuracy_score
    from pythonlib.tools.listtools import tabulate_list
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

    assert X.shape[0]==len(dflab)
    assert len(X.shape)==2

    # First facotrize the l;abels, BNEFORE splitting be,low or else bug.
    labels_all = dflab[var_decode].tolist()
    # labels_all, tmp = pd.factorize(dflab[var_decode])
    # map_int_to_lab = {i:lab for i, lab in enumerate(tmp)}

    X, labels_all, inds_keep = cleanup_remove_labels_ignore(X, labels_all)
    dflab = dflab.iloc[inds_keep].copy().reset_index(drop=True)

    # Get all levels of conditioned var
    groupdict = grouping_append_and_return_inner_items(dflab, vars_conj_condition)

    # Iterate over all levels of conditioned grp vars
    RES = []
    for grp_train, inds_train in groupdict.items():
        # print(grp_train, len(inds_train))

        # This group provides training data
        X_train = X[inds_train, :] # (ntrials, nchans)
        labels_train = [labels_all[i] for i in inds_train]

        train_labels_counts = tabulate_list(labels_train)

        if len(set(labels_train))>1:
            # Train decoder
            mod = decode_train_model(X_train, labels_train, None,
                                     do_center, do_std)

            # Test decoder on all other levels of grouping var
            for grp_test, inds_test in groupdict.items():
                if not grp_train == grp_test:

                    # Only keep test inds that are labels which exist in training
                    # - find inds in labels_all that are labels that exist in train.
                    indstmp = np.argwhere(np.isin(labels_all, labels_train)).squeeze().tolist() # (n,) array of int indices
                    inds_test = [i for i in inds_test if i in indstmp]

                    # Gather test data
                    X_test = X[inds_test, :] # (ntrials, nchans)
                    # labels_test = labels_all[inds_test]
                    labels_test = [labels_all[i] for i in inds_test]

                    # check the distribution of var_decode labels.
                    test_labels_counts = tabulate_list(labels_test)

                    # Each test label  must exist in training data
                    for lab in test_labels_counts.keys():
                        if lab not in train_labels_counts.keys():
                            print(train_labels_counts)
                            print(test_labels_counts)
                            assert False, "add clause to skip test cases that have labels that dont exist in training data?"

                    if len(X_test)>0:
                        # score it
                        score, score_adjusted = decode_apply_model_to_test_data(mod, X_test, labels_test)

                        # Save results
                        RES.append({
                            "var_decode":var_decode,
                            "vars_conj_condition":tuple(vars_conj_condition),
                            "grp_train":grp_train,
                            "grp_test":grp_test,
                            "score":score,
                            "score_adjusted":score_adjusted,
                            "train_labels_counts":train_labels_counts,
                            "test_labels_counts":test_labels_counts,
                            "n_classes_test":len(test_labels_counts)
                        })
    dfres = pd.DataFrame(RES)

    if len(dfres)>0:

        # aggregate, to get one score for each var_decode
        from pythonlib.tools.pandastools import aggregGeneral
        dfres_agg = aggregGeneral(dfres, ["var_decode", "vars_conj_condition"], values=["score", "score_adjusted", "n_classes_test"])
        # dfres_agg = dfres.groupby(["var_decode"]).mean()

        if False: # skip for now
            tmp = dfres.groupby(["var_decode"]).std()
            if np.any(tmp["n_classes_test"]>0.):
                print(dfres)
                print(dfres["n_classes_test"].value_counts())
                assert False, "diff n classes across tests (groups of othervar). Solve by restrticting to same n classes?"
    else:
        dfres_agg = None
    return dfres, dfres_agg



def decode_categorical(X, labels, expected_n_min_across_classes,
                       plot_resampled_data_path_nosuff=None, max_nsplits=None,
                       do_center=True, do_std=False,
                       return_mean_score_over_splits=False):
    """
    Helper to run multiple train-test splits (Kfold), including rebalancing data if labels are unbalanced, and
    then scoring using a balanced accuracty score, for a single time step.
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

    labels = list(labels)
    X, labels, inds_keep = cleanup_remove_labels_ignore(X, labels)

    ## PREP PARAMS
    # Count the lowest n data across classes.
    n_min_across_labs = min(tabulate_list(labels).values())
    n_max_across_labs = max(tabulate_list(labels).values())

    if max_nsplits is None:
        max_nsplits = 10
    n_splits = min([max_nsplits, n_min_across_labs]) # num splits. max it at 12...

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
    for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Each fold is a unique set of test idnices, with this set as small as posibiel whiel still having at laset
        # 1 datapt fore ach class.

        X_train = X[train_index, :]
        labels_train = [labels[i] for i in train_index]
        X_test = X[test_index, :]
        labels_test = [labels[i] for i in test_index]

        if i==0:
            mod = decode_train_model(X_train, labels_train, plot_resampled_data_path_nosuff,
                                     do_center=do_center, do_std=do_std)
        else:
            mod = decode_train_model(X_train, labels_train, None,
                                     do_center=do_center, do_std=do_std)

        if mod is not None: # Happens, if only one class in dataset
            ################# Test on held out
            score, score_adjusted = decode_apply_model_to_test_data(mod, X_test, labels_test)
            # labels_predicted = mod.predict(X_test)
            # score = balanced_accuracy_score(labels_test, labels_predicted, adjusted=False)
            # score_adjusted = balanced_accuracy_score(labels_test, labels_predicted, adjusted=True)

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

    if return_mean_score_over_splits:
        score = np.mean([r["score_xval"] for r in RES])
        score_adjusted = np.mean([r["score_xval_adjusted"] for r in RES])
        return score, score_adjusted
    else:
        return RES

def decodewrap_categorical_timeresolved_within_condition(pa, var_decode,
                                                        vars_conj_condition,
                                                        time_bin_size=None, slide=None,
                                                        do_center=True, do_std=False,
                                                         max_nsplits=None,
                                                         prune_do = True,
                                                         prune_min_n_trials = N_MIN_TRIALS,
                                                         prune_min_n_levs = 2,
                                                         plot_counts_heatmap_savepath=None):
    """
    Wrapper to compute time-resolved decoding, where at each time step compute decode (of var_decode)
    within each level of <vars_conj_conjunction> (doing held out kfold splits), training separate
    decoder each time, and then averaging over those.
    Useful if want to control for <vars_conj_conjunction> while asking about decoding.
    And then takes average over all levels of <vars_conj_conjunction>
    PARAMS:
    - var_decode, str,
    - vars_conj_conjunction, list of str, conditioned vairalb.e
    - prune_do, bool, then autoatmically prunes data so that each level of conj var has at least
    <prune_min_n_levs> levels with at least <prune_min_n_trials> trials.
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

    pa = pa.copy()

    # Prune data to just cases with at least 2 levels of decode var
    if prune_do:
        balance_no_missed_conjunctions = False # Since this is within condition.
        dflab = pa.Xlabels["trials"]
        dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var_decode, vars_conj_condition,
                                                                 n_min_across_all_levs_var=prune_min_n_trials,
                                                                 lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                                                                 prune_levels_with_low_n=True,
                                                                 ignore_values_called_ignore=True,
                                                                 plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                                                 balance_no_missed_conjunctions=balance_no_missed_conjunctions)

        if len(dfout)==0:
            print("all data pruned!!")
            return None

        # Only keep the indices in dfout
        pa = pa.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)
        assert len(pa.X)>0, "all data pruned!!"

    # Optionally, bin X in time, to have fewer time bins to decode
    if time_bin_size is not None:
        pa = pa.agg_by_time_windows_binned(time_bin_size, slide)

    # 2. Extract X from pa
    X = pa.X # (nchans, ntrials, ntimes)
    times = pa.Times
    dflab = pa.Xlabels["trials"]

    # Decode, for each time bin.
    list_tbin = range(X.shape[2])
    res = []
    for tbin in list_tbin:
        Xscal = X[:, :, tbin].T # (ntrials, nchans)
        dfresthis, dfres_agg = decode_categorical_within_condition(Xscal, dflab,
                                                                  var_decode,
                                                                  vars_conj_condition,
                                                                  do_center=do_center,
                                                                    do_std=do_std,
                                                                   max_nsplits=max_nsplits)
        if len(dfresthis)>0:
            assert len(dfres_agg)==1

            # 3. Collect data
            res.append({
                "tbin":tbin,
                "time":times[tbin],
                "var_decode":var_decode,
                "vars_conj_condition":tuple(vars_conj_condition),
                "var_decode_and_conj":tuple([var_decode] + vars_conj_condition),
                "score":dfres_agg["score"].values[0],
                "score_adjusted":dfres_agg["score_adjusted"].values[0],
            })

    return res

def decodewrap_categorical_timeresolved_cross_condition(pa, var_decode,
                                                        vars_conj_condition,
                                                        subtract_mean_vars_conj=False,
                                                        time_bin_size=None, slide=None,
                                                        do_center=True, do_std=False,
                                                        prune_do=True,
                                                        plot_counts_heatmap_savepath=None):
    """
    Wrapper to compute time-resolved decoding, where at each time step compute cross-generalized
    decoding, by fitting decoder on each level of <vars_conj_conjunction> and testing decoder on
    each other level of <vars_conj_conjunction>, and then taking average over all other levels.
    PARAMS:
    - var_decode, str,
    - vars_conj_conjunction, list of str, conditioned vairalb.e
    - subtract_mean_vars_conj, bool, if True, then subtracts mean with ineach level of vars_conj_conjunction
    RETURNS:
        - (None, if prunes all data).
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

    pa = pa.copy()

    # Prune data to just cases with at least 2 levels of decode var
    if prune_do:
        balance_no_missed_conjunctions = True
        dflab = pa.Xlabels["trials"]
        prune_min_n_trials = N_MIN_TRIALS
        prune_min_n_levs = 2
        dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var_decode, vars_conj_condition,
                                                                 n_min_across_all_levs_var=prune_min_n_trials,
                                                                 lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                                                                 prune_levels_with_low_n=True,
                                                                 ignore_values_called_ignore=True,
                                                                 plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                                                 balance_no_missed_conjunctions=balance_no_missed_conjunctions)

        if len(dfout)==0:
            print("all data pruned!!")
            return None

        # Only keep the indices in dfout
        pa = pa.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)

        # Skip, if not enough data
        if len(pa.X)==0:
            print("all data pruned!!")
            return None

        if len(dict_dfthis)==1:
            # Then you can't test generlaization.. skip!
            print("Not enough levels of conj var to test generlaization! Skipping")
            return None

    # Normalize by substrcintg mean within other vars?
    if subtract_mean_vars_conj:
        pa = pa.norm_by_label_subtract_mean("trials", vars_conj_condition)

    # Optionally, bin X in time, to have fewer time bins to decode
    if time_bin_size is not None:
        pa = pa.agg_by_time_windows_binned(time_bin_size, slide)

    # 2. Extract X from pa
    X = pa.X # (nchans, ntrials, ntimes)
    times = pa.Times
    dflab = pa.Xlabels["trials"]

    # Decode, for each time bin.
    list_tbin = range(X.shape[2])
    res = []
    for tbin in list_tbin:
        Xscal = X[:, :, tbin].T # (ntrials, nchans)
        dfresthis, dfres_agg = decode_categorical_cross_condition(Xscal, dflab,
                                                                  var_decode,
                                                                  vars_conj_condition,
                                                                  do_center=do_center, do_std=do_std)
        if len(dfresthis)>0:
            # print("---------------")
            # display(dfresthis)
            # display(dfres_agg)
            assert len(dfres_agg)==1

            # 3. Collect data
            res.append({
                "tbin":tbin,
                "time":times[tbin],
                "var_decode":var_decode,
                "vars_conj_condition":tuple(vars_conj_condition),
                "var_decode_and_conj":tuple([var_decode] + vars_conj_condition),
                "score":dfres_agg["score"].values[0],
                "score_adjusted":dfres_agg["score_adjusted"].values[0],
            })

    return res


def decodewrap_categorical_timeresolved_singlevar(X, times, dflab, vars_decode,
                                      time_bin_size=None, slide=None,
                                      max_nsplits=None):
    """
    Time-resolved decoding of a single categorical variable, which
    labels static across time.
    PARAMS:
    - X, (dims, trials, times).
    - times, timestap for each time. just for labeling, so pass in anything
    - dflab, dframe holding labels, len trials
    - vars_decode, list of str, will do decode separately for each variable
    RETURNS:
    - res, list of dicts, each hollding score for a single combo of (time_bin x decode-var)
    """

    assert len(times)==X.shape[2]
    assert len(dflab)==X.shape[1]

    # Optionally, bin X in time, to have fewer time bins to decode
    if time_bin_size is not None:
        X, times = bin_frmat_in_time(X, times, time_bin_size=time_bin_size, slide=slide)

    # 2. Apply this method to pa
    list_tbin = range(X.shape[2])
    res = []
    for tbin in list_tbin:
        # 3a. Extract data for this time bin
        Xthis = X[:, :, tbin].T # (ntrials, nchans)

        # 3. Run decoder
        for var_decode in vars_decode:

            # labels = pd.factorize(dflab[var_decode])[0]
            labels = dflab[var_decode].tolist()

            # expected_n_min_across_classes = len(set(labels))
            score, score_adjusted = decode_categorical(Xthis, labels, 1,
                                                       return_mean_score_over_splits=True,
                                                       max_nsplits=max_nsplits)

            # 3. Collect data
            res.append({
                "tbin":tbin,
                "time":times[tbin],
                "var_decode":var_decode,
                "score":score,
                "score_adjusted":score_adjusted
            })
    return res

def decodewrap_categorical_single_decoder_across_time(x_train, labels_train, X_test, labels_test,
                                                      times_test, do_std=False,
                                                      do_train_test_kfold_splits=True,
                                                      n_splits=4, savepath_ndata=None):

    """
    Time-resolved decoding using a single trained decoder. Does both steps (training and testing).
    E.g., use motor-related data and apply it to decode shapes from visual data.
    PARAMS:
    - x_train, (ntrials_1, ndims), scalar data to train a isngle decoder
    - labels_train (ntrials_1,), class labels for training data.
    - X_test, (ndims, ntrials_2, ntimes), time-varying data to apply decoder to across time.
    - labels_test (ntrials_2,), class labels for test data, static across time.
    - do_train_test_kfold_splits, bool, if True, then does kfold train-test splits, which is important
    if the train and test data are from same trials. Will fail if it detects that they are not same trails (by
    comparing labels).
    NOTE: Returns None if not enough data to do this.
    """

    # Check data
    assert x_train.shape[0]==len(labels_train)
    assert X_test.shape[1]==len(labels_test)
    assert X_test.shape[0]==x_train.shape[1]
    assert len(x_train.shape)==2
    assert len(X_test.shape)==3
    assert X_test.shape[2]==len(times_test)

    # Train model
    # pathis = pa.slice_by_dim_values_wrapper("times", [0.4, 0.6])
    # pathis = pathis.agg_wrapper("times")
    # x_train = pathis.X.squeeze(axis=2).T # (ntrials, nchans)
    # labels_train = pathis.Xlabels["trials"][var_decode].tolist()

    # Get test data
    # X_test = pa.X # (chans, trials, times)
    # labels_test = pa.Xlabels["trials"][var_decode].tolist()

    # if LABELS_IGNORE is not None:
    #     # print(type(labels_train[0]))
    #     # print(labels_ignore)
    #     inds_keep_train = [i for i, l in enumerate(labels_train) if l not in LABELS_IGNORE]
    #     # print(inds_keep_train)
    #     # assert False
    #     x_train = x_train[inds_keep_train, :]
    #     labels_train = [labels_train[i] for i in inds_keep_train]
    #
    #     inds_keep_test = [i for i, l in enumerate(labels_test) if l not in LABELS_IGNORE]
    #     X_test = X_test[:, inds_keep_test, :]
    #     labels_test = [labels_test[i] for i in inds_keep_test]

    # Make sure training and testing data have the same set of classes
    # (prune if not).
    inds_keep_train, inds_keep_test = preprocess_match_training_and_test_labels(labels_train, labels_test)
    x_train = x_train[inds_keep_train,:]
    labels_train = [labels_train[i] for i in inds_keep_train]
    X_test = X_test[:, inds_keep_test, :]
    labels_test = [labels_test[i] for i in inds_keep_test]

    if not all([l in labels_train for l in labels_test]):
        print(labels_train)
        print(labels_test)
        assert False

    # Convert labels to ints, making sure that ints:lab matches across
    # train and test data
    labs_unique = sorted(set(labels_train + labels_test))
    map_lab_to_int = {lab:i for i, lab in enumerate(labs_unique)}
    map_int_to_lab = {i:lab for i, lab in enumerate(labs_unique)}

    # convert original values
    labels_train_int = [map_lab_to_int[lab] for lab in labels_train]
    labels_test_int = [map_lab_to_int[lab] for lab in labels_test]

    if len(labels_train_int)==0:
        return None
    if len(set(labels_train_int))==1:
        return None

    if savepath_ndata is not None:
        # Save sample size info
        from pythonlib.tools.pandastools import grouping_print_n_samples
        dftmp = pd.DataFrame({"var_decode":labels_train})
        grouping_print_n_samples(dftmp, ["var_decode"], savepath=f"{savepath_ndata}-TRAIN.yaml", save_as="txt")
        dftmp = pd.DataFrame({"var_decode":labels_test})
        grouping_print_n_samples(dftmp, ["var_decode"], savepath=f"{savepath_ndata}-TEST.yaml", save_as="txt")

    ###### Two methods.
    res = []
    if do_train_test_kfold_splits:
        from sklearn.model_selection import StratifiedKFold

        # Make split, then run ebntire thing (one decoder, multiple times for testing),
        # Then make another split, etc. Collect results from each split.
        assert labels_train_int == labels_test_int, "only do kfold splits if these are same trials..."
        labels_int = labels_train_int # pick either one...
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        for i, (train_index, test_index) in enumerate(skf.split(x_train, labels_int)):
            x_train_this = x_train[train_index,:]
            labels_int_train_this = [labels_int[_i] for _i in train_index]

            X_test_this = X_test[:, test_index, :]
            labels_int_test_this = [labels_int[_i] for _i in test_index]

            # Train a single decoder
            mod = decode_train_model(x_train_this, labels_int_train_this, do_std=do_std)

            # Test decoder at all time bins
            list_tbin = range(X_test_this.shape[2])
            for tbin in list_tbin:

                # 3a. Extract data for this time bin
                x_this = X_test_this[:, :, tbin].T # (ntrials, nchans)
                score, score_adjusted, labels_predicted, labels_test, conf_scores = decode_apply_model_to_test_data(mod, x_this,
                                                                                           labels_int_test_this,
                                                                                           return_predictions_all_trials=True)

                res.append({
                    "iter_kfold":i,
                    "tbin":tbin,
                    "time":times_test[tbin],
                    "score":score,
                    "score_adjusted":score_adjusted,
                    "labels_predicted":labels_predicted,
                    "labels_test":labels_test,
                    "conf_scores":conf_scores,
                    "map_int_to_lab":map_int_to_lab,
                    "inds_keep_test":inds_keep_test
                })
    else:
        # Train a single decoder
        mod = decode_train_model(x_train, labels_train_int, do_std=do_std)

        if mod is None:
            return None
            # print(x_train.shape)
            # print(set(labels_train_int))
            # assert False
        # Test decoder at all time bins
        list_tbin = range(X_test.shape[2])
        for tbin in list_tbin:

            # 3a. Extract data for this time bin
            x_this = X_test[:, :, tbin].T # (ntrials, nchans)
            score, score_adjusted, labels_predicted, labels_test, conf_scores = decode_apply_model_to_test_data(mod, x_this,
                                                                                       labels_test_int,
                                                                                       return_predictions_all_trials=True)

            # 3. Collect data
            res.append({
                "tbin":tbin,
                "time":times_test[tbin],
                "score":score,
                "score_adjusted":score_adjusted,
                "labels_predicted":labels_predicted,
                "labels_test":labels_test,
                "conf_scores":conf_scores,
                "map_int_to_lab":map_int_to_lab,
                "inds_keep_test":inds_keep_test
            })

    return res

def decodewrapouterloop_preprocess_extract_params(DFallpa):
    """
    Quikc extraction of params as helper.
    :param DFallpa:
    :return:
    """
    list_br = DFallpa["bregion"].unique().tolist()
    list_tw = DFallpa["twind"].unique().tolist()
    list_ev = DFallpa["event"].unique().tolist()

    # Figure out how long is seuqence
    pa = DFallpa["pa"].values[0]
    n_strokes_max = -1
    for i in range(8):
        n_ignore = sum(pa.Xlabels["trials"][f"seqc_{i}_shape"].isin(["IGN", "IGNORE"]))
        n_total = len(pa.Xlabels["trials"][f"seqc_{i}_shape"])
        print(n_ignore, n_total)
        if n_ignore<n_total:
            n_strokes_max=i+1
    assert n_strokes_max>0
    print("THIS MANY STROKES MAX:", n_strokes_max)

    return list_br, list_tw, list_ev, n_strokes_max


def decodewrapouterloop_categorical_timeresolved_within_condition(DFallpa, list_var_decode,
                                                                 list_vars_conj,
                                                                SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                                  max_nsplits = None,
                                                                  filtdict = None,
                                                                  separate_by_task_kind = True):
    """
    Wrapper to within-condition decode, for each PA, each time point, separately for each var in list_vars_decode,
    decode decode separately for each level of othervar, and then averaging decode. Think of this as "controlling"
    for levels of othervar.
    :param DFallpa:
    :param list_vars_decode:
    :param time_bin_size:
    :param slide:
    :param n_min_trials:
    :return:
    """

    RES = []
    already_done = []
    for i, row in DFallpa.iterrows():
        br = row["bregion"]
        tw = row["twind"]
        ev = row["event"]
        PA = row["pa"]

        if separate_by_task_kind:
            PA.Xlabels["trials"]["_task_kind"] = PA.Xlabels["trials"]["task_kind"]
        else:
            PA.Xlabels["trials"]["_task_kind"] = "dummy"
        list_task_kind = PA.Xlabels["trials"]["_task_kind"].unique()

        for task_kind in list_task_kind:
            pa = PA.slice_by_labels("trials", "_task_kind", [task_kind])

            for var_decode, vars_conj_condition in zip(list_var_decode, list_vars_conj):
                print(ev, br, tw, task_kind, var_decode)

                if (ev, var_decode, vars_conj_condition, task_kind) not in already_done:
                    plot_counts_heatmap_savepath = f"{SAVEDIR}/counts_{ev}_{task_kind}-var={var_decode}-varconj={'|'.join(vars_conj_condition)}"
                    already_done.append((ev, var_decode, vars_conj_condition, task_kind))
                else:
                    plot_counts_heatmap_savepath = None

                if filtdict is not None:
                    pa = pa.copy()
                    for _var, _levs in filtdict.items():
                        pa.Xlabels["trials"] = pa.Xlabels["trials"][pa.Xlabels["trials"][_var].isin(_levs)].reset_index(drop=True)
                        if len(pa.Xlabels["trials"])==0:
                            print("var:", _var, ", levels keep:", _levs)
                            print("Filter completely removed all trials... SKIPPING")
                if len(pa.Xlabels["trials"])==0:
                    continue

                # 1. Extract the specific pa for this (br, tw)
                res = decodewrap_categorical_timeresolved_within_condition(pa, var_decode,
                                                            vars_conj_condition,
                                                            time_bin_size=time_bin_size, slide=slide,
                                                            do_center=True, do_std=False,
                                                            max_nsplits = max_nsplits, plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
                if res is not None:
                    for r in res:
                        r["bregion"]=br
                        r["twind"]=tw
                        r["event"]=ev
                        r["task_kind"]=task_kind
                    RES.extend(res)
    DFRES = pd.DataFrame(RES)

    ### PLOT
    import seaborn as sns
    import matplotlib.pyplot as plt

    if len(DFRES)>0:
        DFRES = append_col_with_grp_index(DFRES, ["task_kind", "event"], "tk_ev", False)
        LIST_TASK_KIND = DFRES["task_kind"].unique().tolist()
        for task_kind in LIST_TASK_KIND:
            dfthis = DFRES[DFRES["task_kind"] == task_kind]
            for y in ["score", "score_adjusted"]:
                # PLOTS
                fig = sns.relplot(data=dfthis, x="time", y=y, hue="var_decode_and_conj", row="event", col="bregion",
                                  kind="line", height=4)
                savefig(fig, f"{SAVEDIR}/splot_bregion-task_kind_{task_kind}-{y}.pdf")

                fig = sns.relplot(data=dfthis, x="time", y=y, hue="bregion", row="event", col="var_decode_and_conj",
                                  kind="line", height=4)
                savefig(fig, f"{SAVEDIR}/splot_var_decode-task_kind_{task_kind}-{y}.pdf")

                plt.close("all")

    return DFRES


def decodewrapouterloop_categorical_timeresolved_cross_condition(DFallpa, list_var_decode,
                                                                 list_vars_conj,
                                                 SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                                 subtract_mean_vars_conj=False,
                                                                  filtdict = None,
                                                                 separate_by_task_kind = True):
    """
    Wrapper to cross-condition decode, for each PA, each time point, separately for each var in list_vars_decode,
    scoring decoder generaliation across levels of grouping variables vars_conj_condition
    :param DFallpa:
    :param list_vars_decode:
    :param time_bin_size:
    :param slide:
    :param n_min_trials:
    :return:
    """

    RES = []
    already_done = []
    for i, row in DFallpa.iterrows():
        br = row["bregion"]
        tw = row["twind"]
        ev = row["event"]
        PA = row["pa"]

        if separate_by_task_kind:
            PA.Xlabels["trials"]["_task_kind"] = PA.Xlabels["trials"]["task_kind"]
        else:
            PA.Xlabels["trials"]["_task_kind"] = "dummy"
        list_task_kind = PA.Xlabels["trials"]["_task_kind"].unique()

        for task_kind in list_task_kind:
            pa = PA.slice_by_labels("trials", "_task_kind", [task_kind])
            # print(pa.Xlabels["trials"]["task_kind"].unique())
            # assert False

            for var_decode, vars_conj_condition in zip(list_var_decode, list_vars_conj):
                print(ev, br, tw, var_decode)

                if (ev, var_decode, vars_conj_condition, task_kind) not in already_done:
                    plot_counts_heatmap_savepath = f"{SAVEDIR}/counts_{ev}_{task_kind}-var={var_decode}-varconj={'|'.join(vars_conj_condition)}"
                    already_done.append((ev, var_decode, vars_conj_condition, task_kind))
                else:
                    plot_counts_heatmap_savepath = None

                if filtdict is not None:
                    pa = pa.copy()
                    for _var, _levs in filtdict.items():
                        pa = pa.slice_by_labels("trials", _var, _levs)
                        # pa.Xlabels["trials"] = pa.Xlabels["trials"][pa.Xlabels["trials"][_var].isin(_levs)].reset_index(drop=True)
                        if len(pa.Xlabels["trials"])==0:
                            print("var:", _var, ", levels keep:", _levs)
                            print("Filter completely removed all trials... SKIPPING")
                if len(pa.Xlabels["trials"])==0:
                    continue

                # 1. Extract the specific pa for this (br, tw)
                res = decodewrap_categorical_timeresolved_cross_condition(pa, var_decode,
                                                            vars_conj_condition,
                                                            subtract_mean_vars_conj=subtract_mean_vars_conj,
                                                            time_bin_size=time_bin_size, slide=slide,
                                                            do_center=True, do_std=False,
                                                            plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
                if res is not None:
                    for r in res:
                        r["bregion"]=br
                        r["twind"]=tw
                        r["event"]=ev
                        r["task_kind"]=task_kind
                    RES.extend(res)
    DFRES = pd.DataFrame(RES)

    ### PLOT
    from pythonlib.tools.pandastools import convert_to_2d_dataframe, grouping_plot_n_samples_conjunction_heatmap, plot_subplots_heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    if len(DFRES)>0:
        DFRES = append_col_with_grp_index(DFRES, ["task_kind", "event"], "tk_ev", False)
        LIST_TASK_KIND = DFRES["task_kind"].unique().tolist()
        for task_kind in LIST_TASK_KIND:
            dfthis = DFRES[DFRES["task_kind"] == task_kind]
            for y in ["score", "score_adjusted"]:
                # PLOTS
                fig = sns.relplot(data=dfthis, x="time", y=y, hue="var_decode_and_conj", row="event", col="bregion",
                                  kind="line", height=4)
                savefig(fig, f"{SAVEDIR}/splot_bregion-task_kind_{task_kind}-{y}.pdf")

                fig = sns.relplot(data=dfthis, x="time", y=y, hue="bregion", row="event", col="var_decode_and_conj",
                                  kind="line", height=4)
                savefig(fig, f"{SAVEDIR}/splot_var_decode-task_kind_{task_kind}-{y}.pdf")

                plt.close("all")

    return DFRES


def decodewrapouterloop_categorical_timeresolved(DFallpa, list_vars_decode,
                                                 SAVEDIR, time_bin_size=0.1, slide=0.05,
                                                 n_min_trials=N_MIN_TRIALS,
                                                 max_nsplits=None):
    """
    Wrapper to time-resolved decode, for each PA, each time point,
    separately for each var in list_vars_decode.
    :param DFallpa:
    :param list_vars_decode:
    :param time_bin_size:
    :param slide:
    :param n_min_trials:
    :return:
    """

    RES = []
    for i, row in DFallpa.iterrows():

        br = row["bregion"]
        tw = row["twind"]
        ev = row["event"]
        PA = row["pa"]

        list_task_kind = PA.Xlabels["trials"]["task_kind"].unique()

        for task_kind in list_task_kind:
            pa = PA.slice_by_labels("trials", "task_kind", [task_kind])

            # 2. Extract X from pa
            X = pa.X # (nchans, ntrials, ntimes)
            times = pa.Times
            dflab = pa.Xlabels["trials"]

            for var_decode in list_vars_decode:

                # Prune dflab
                from pythonlib.tools.pandastools import filter_by_min_n
                dftmp = filter_by_min_n(dflab, var_decode, n_min_trials)

                if len(dftmp)>0:

                    indskeep = dftmp["_index"].tolist()
                    Xthis = X[:, indskeep, :]
                    dflab_this = dflab.iloc[indskeep]

                    if len(dflab_this[var_decode].unique())==1:
                        print("SKIPPING, becuase only one label:")
                        print(dflab_this[var_decode].unique())
                        continue

                    if len(Xthis)>0:
                        res = decodewrap_categorical_timeresolved_singlevar(Xthis, times, dflab_this,
                                                                            [var_decode],
                                                                            time_bin_size=time_bin_size,
                                                                            slide=slide, max_nsplits=max_nsplits)
                        for r in res:
                            r["event"]=ev
                            r["bregion"]=br
                            r["twind"]=tw
                            r["var_decode"]=var_decode
                            r["task_kind"] = task_kind

                        RES.extend(res)
    DFRES = pd.DataFrame(RES)

    ### PLOT
    from pythonlib.tools.pandastools import convert_to_2d_dataframe, grouping_plot_n_samples_conjunction_heatmap, plot_subplots_heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    DFRES = append_col_with_grp_index(DFRES, ["task_kind", "event"], "tk_ev", False)
    LIST_TASK_KIND = DFRES["task_kind"].unique().tolist()

    for task_kind in LIST_TASK_KIND:
        dfthis = DFRES[DFRES["task_kind"] == task_kind]
        for y in ["score", "score_adjusted"]:
            # PLOTS
            fig = sns.relplot(data=dfthis, x="time", y=y, hue="var_decode", row="event", col="bregion", kind="line", height=4)
            savefig(fig, f"{SAVEDIR}/splot_bregion-task_kind_{task_kind}-{y}.pdf")

            fig = sns.relplot(data=dfthis, x="time", y=y, hue="bregion", row="event", col="var_decode", kind="line", height=4)
            savefig(fig, f"{SAVEDIR}/splot_var_decode-task_kind_{task_kind}-{y}.pdf")

            plt.close("all")

    return DFRES

def decodewrapouterloop_categorical_cross_time_plot_compare_contexts(DFRES_INPUT, var_decode,
                                                                     SAVEDIR):
    """
    Plotting -- Split and group data based on wherther the train-test datasets were "same context" or "diff context", where
    the datasets are defined by (event train, event test, task kind train, taks kind test) and
    same means all 4 are identical, and there are 3 kinds of diffs (diff fore ach one, diff for both).
    Computes and plots scores both (i) a single average for each context dataset and (ii) difference betwee each
    pair of datasets.
    :param DFRES_INPUT: Usually the output of decodewrapouterloop_categorical_cross_time.
    :param var_decode: str variable that plot.
    :return:
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items, grouping_print_n_samples
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import summarize_featurediff

    # Keep only a single varialbe, or else plots get too big..
    # "sahpe_this_event" variable
    # vars_keep = ["shape_this_event"]
    vars_keep = [var_decode]
    DFRES = DFRES_INPUT[DFRES_INPUT["var_decode"].isin(vars_keep)].reset_index(drop=True)

    # (1) Find each of the contexts.
    grpdict = grouping_append_and_return_inner_items(DFRES, ["task_kind_train", "task_kind_test", "event_train", "event_test"])

    if len(grpdict)==1:
        return

    list_same_context = []
    list_diff_tk_same_ev = []
    list_same_tk_diff_ev = []
    list_diff_tk_diff_ev = []
    for grp, inds in grpdict.items():
        tktrain, tktest, evtrain, evtest = grp
        if "06_on_" in evtrain and "06_on_" in evtest: # Then is during strokes, good.
            # Decide if this is same or different context
            if tktrain==tktest and evtrain==evtest:
                list_same_context.append([tktrain, tktest, evtrain, evtest])
            elif (tktrain==tktest) and (not evtrain==evtest):
                list_same_tk_diff_ev.append([tktrain, tktest, evtrain, evtest])
            elif (not tktrain==tktest) and (evtrain==evtest):
                list_diff_tk_same_ev.append([tktrain, tktest, evtrain, evtest])
            else:
                list_diff_tk_diff_ev.append([tktrain, tktest, evtrain, evtest])
    # Finally, combine all the diff contexts
    list_diff_context = list_diff_tk_same_ev + list_same_tk_diff_ev + list_diff_tk_diff_ev
    # For each kind of context, compute average scores for each bregion
    print("Got these conditions for SAME CONTEXT:")
    for c in list_same_context:
        print(c)
    print("Got these conditions for DIFF TASKKIND, SAME EVENT:")
    for c in list_diff_tk_same_ev:
        print(c)
    print("Got these conditions for SAME TASKKIND, DIFF EVENT:")
    for c in list_same_tk_diff_ev:
        print(c)
    print("Got these conditions for DIFF TASKKIND, DIFF EVENT:")
    for c in list_diff_tk_diff_ev:
        print(c)

    if len(list_diff_tk_same_ev)==0 and len(list_same_tk_diff_ev)==0 and len(list_diff_tk_diff_ev)==0:
        # NOT enough data
        return

    # # Collect all
    # inds = []
    # for grp in list_diff_tk_same_ev:
    #     inds.extend(grpdict[tuple(grp)])
    # dfresthis = DFRES.iloc[inds].reset_index(drop=True)
    #
    # # take average
    # from pythonlib.tools.pandastools import aggregGeneral
    # dfresthis_agg = aggregGeneral(dfresthis, ["bregion", "var_decode", "tbin_train", "time_train", "tbin_test", "time_test"], values=["score", "score_adjusted"])
    # dfresthis[:2]
    #
    # from neuralmonkey.analyses.decode_good import decodewrapouterloop_categorical_cross_time_plot
    # dfresthis["task_kind_train"] = "dummy"
    # dfresthis["task_kind_test"] = "dummy"
    # dfresthis["event_train"] = "dummy"
    # dfresthis["event_test"] = "dummy"
    # decodewrapouterloop_categorical_cross_time_plot(dfresthis, "/tmp")

    ##### Aggregate across contexts
    map_contextname_to_listconditions = {
        "same": list_same_context,
        # "diff":list_diff_context,
        "diff_tk_same_ev":list_diff_tk_same_ev,
        "same_tk_diff_ev":list_same_tk_diff_ev,
        "diff_tk_diff_ev":list_diff_tk_diff_ev
    }
    # Run aggregation...
    res = []
    for contextname, conditions in map_contextname_to_listconditions.items():
        # Given two sets of rows in DFRES, get their heatmaps, and also compute difference.

        inds = []
        for grp in conditions:
            inds.extend(grpdict[tuple(grp)])
        dfresthis = DFRES.iloc[inds].reset_index(drop=True)
        dfresthis_agg = aggregGeneral(dfresthis, ["bregion", "var_decode", "tbin_train", "time_train", "tbin_test", "time_test"], values=["score", "score_adjusted"])
        dfresthis_agg["context_relation"] = contextname
        dfresthis_agg["task_kind_train"] = contextname
        dfresthis_agg["task_kind_test"] = contextname
        dfresthis_agg["event_train"] = contextname
        dfresthis_agg["event_test"] = contextname
        res.append(dfresthis_agg)
    DFRES_AGG = pd.concat(res).reset_index(drop=True)

    # 1) Plot each context conditions
    savedir = f"{SAVEDIR}/each_context"
    os.makedirs(savedir, exist_ok=True)
    decodewrapouterloop_categorical_cross_time_plot(DFRES_AGG, savedir)

    ################ Plot differences between contexts
    res = []
    for contextname1 in map_contextname_to_listconditions:
        for contextname2 in map_contextname_to_listconditions:
            if not contextname1==contextname2:

                dfresthis_agg_both_diff, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = summarize_featurediff(
                    DFRES_AGG, "context_relation", [contextname1, contextname2],
                    ["score", "score_adjusted"], ["bregion", "var_decode", "tbin_train", "tbin_test", "time_train", "time_test"])

                dfresthis_agg_both_diff["task_kind_train"] = f"{contextname2}-min-{contextname1}"
                dfresthis_agg_both_diff["task_kind_test"] = f"{contextname2}-min-{contextname1}"
                dfresthis_agg_both_diff["event_train"] = f"{contextname2}-min-{contextname1}"
                dfresthis_agg_both_diff["event_test"] = f"{contextname2}-min-{contextname1}"

                # convert name to support plotting
                dfresthis_agg_both_diff["score"] = dfresthis_agg_both_diff[f"score-{contextname2}min{contextname1}"]
                dfresthis_agg_both_diff["score_adjusted"] = dfresthis_agg_both_diff[f"score_adjusted-{contextname2}min{contextname1}"]

                res.append(dfresthis_agg_both_diff)
    DFRES_AGG_DIFFS = pd.concat(res).reset_index(drop=True)

    # Plot differences
    savedir = f"{SAVEDIR}/differences_between_context_effects"
    os.makedirs(savedir, exist_ok=True)
    decodewrapouterloop_categorical_cross_time_plot(DFRES_AGG_DIFFS, savedir)

def decodewrapouterloop_categorical_cross_time_plot(DFRES, SAVEDIR):
    """
    Plot across time (and event) decoding, results from
    decodewrapouterloop_categorical_cross_time_cross_var() [diferent variables]
    decodewrapouterloop_categorical_cross_time() [same variable]

    :param DFRES:
    :param SAVEDIR:
    :return:
    """
    from pythonlib.tools.pandastools import convert_to_2d_dataframe, grouping_plot_n_samples_conjunction_heatmap, plot_subplots_heatmap
    from pythonlib.tools.pandastools import convert_to_2d_dataframe, grouping_plot_n_samples_conjunction_heatmap, plot_subplots_heatmap


    if "var_decode_test" in DFRES.columns:
        SEPARATE_VARS_TRAIN_TEST = True
        DFRES = append_col_with_grp_index(DFRES, ["event_train", "event_test", "var_decode_train", "var_decode_test"], "e1_e2_v1_v2", True, strings_compact=True)
    else:
        assert "var_decode" in DFRES.columns
        SEPARATE_VARS_TRAIN_TEST = False
        DFRES = append_col_with_grp_index(DFRES, ["event_train", "event_test", "var_decode"], "e1_e2_v1_v2", True, strings_compact=True)


    ########## (1) Plots by bregion, each subplot a different var.
    list_br = DFRES["bregion"].unique()
    if len(DFRES["e1_e2_v1_v2"].unique())>18:
        # THen split into multiple smaller plots...
        for task_kind_train in DFRES["task_kind_train"].unique().tolist():
            for task_kind_test in DFRES["task_kind_test"].unique().tolist():
                print("=== PLOTTING (all the following plots):", task_kind_train, task_kind_test)

                dfthis = DFRES[
                    (DFRES["task_kind_train"]==task_kind_train) &
                    (DFRES["task_kind_test"]==task_kind_test)
                    ]
                if len(dfthis)==0:
                    print("! no data, skipping")
                    continue
                for br in list_br:
                    print("Plotting bregion: ", br)
                    dfres_this = dfthis[
                            (dfthis["bregion"] == br)].reset_index(drop=True)

                    fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "e1_e2_v1_v2", share_zlim=True, title_size=7, diverge=True)
                    savefig(fig, f"{SAVEDIR}/BYBREGION-tktrain={task_kind_train}-tktest={task_kind_test}-{br}-score_adjusted.pdf")
                    plt.close("all")
    else:
        dfthis = DFRES
        for br in list_br:
            print("Plotting bregion: ", br)
            dfres_this = dfthis[
                    (dfthis["bregion"] == br)].reset_index(drop=True)

            fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "e1_e2_v1_v2", share_zlim=True, title_size=7, diverge=True)
            savefig(fig, f"{SAVEDIR}/BYBREGION-{br}-score_adjusted.pdf")
            plt.close("all")


    list_ev_train = DFRES["event_train"].unique()
    list_ev_test = DFRES["event_test"].unique()
    for task_kind_train in DFRES["task_kind_train"].unique().tolist():
        for task_kind_test in DFRES["task_kind_test"].unique().tolist():
            for ev_train in list_ev_train:
                for ev_test in list_ev_test:
                    print("=== PLOTTING (all the following plots):", task_kind_train, task_kind_test, ev_train, ev_test)

                    dfthis = DFRES[
                        (DFRES["task_kind_train"]==task_kind_train) &
                        (DFRES["task_kind_test"]==task_kind_test) &
                        (DFRES["event_train"]==ev_train) &
                        (DFRES["event_test"]==ev_test)
                        ]
                    if len(dfthis)==0:
                        print("! no data, skipping")
                        continue

                    if False: # Plotting above, more subplots per figure
                        ########## (1) Plots by bregion, each subplot a different var.
                        for br in list_br:
                            print("Plotting bregion: ", br)
                            dfres_this = dfthis[
                                    (dfthis["bregion"] == br)].reset_index(drop=True)

                            fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "e1_e2_v1_v2", share_zlim=True, title_size=7, diverge=True)
                            savefig(fig, f"{SAVEDIR}/BYBREGION-tktrain={task_kind_train}-tktest={task_kind_test}-evtrain={ev_train}-evtest={ev_test}-{br}-score_adjusted.pdf")

                            if False: # Takes too long
                                fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score", "e1_e2_v1_v2", share_zlim=True, title_size=7)
                                savefig(fig, f"{SAVEDIR}/tktrain_{task_kind_train}-tktest_{task_kind_test}-fig_by_bregion-{br}.pdf")
                                fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "e1_e2_v1_v2", share_zlim=True, title_size=7, diverge=True, ZLIMS=(-0.5, 0.5))
                                savefig(fig, f"{SAVEDIR}/tktrain_{task_kind_train}-tktest_{task_kind_test}-fig_by_bregion-{br}-score_adjusted-zlim.pdf")

                            plt.close("all")

                    ######### (2) Plots by each variable, across all regions.
                    # For each set of ev_train and ev_test, plot across all bregions.
                    if SEPARATE_VARS_TRAIN_TEST:
                        A = sort_mixed_type(set(DFRES["var_decode_train"].tolist()))
                        B = sort_mixed_type(set(DFRES["var_decode_test"].tolist()))
                        list_var_decode_train_test = [(a, b) for a in A for b in B]

                        for var_decode_train, var_decode_test in list_var_decode_train_test:

                            print("Plotting var_decode_train, var_decode_test: ", var_decode_train, var_decode_test)
                            dfres_this = dfthis[
                                    (dfthis["var_decode_train"] == var_decode_train) &
                                    (dfthis["var_decode_test"] == var_decode_test)].reset_index(drop=True)

                            if len(dfres_this)==0:
                                print("! no data...")
                                continue

                            fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "bregion", share_zlim=True, title_size=7, diverge=True)
                            savefig(fig, f"{SAVEDIR}/BYVAR-tktrain={task_kind_train}-tktest={task_kind_test}-evtrain={ev_train}-evtest={ev_test}-vartrain={var_decode_train}-vartest={var_decode_test}-score_adjusted.pdf")
                            if False: # Takes too long
                                fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score", "bregion", share_zlim=True, title_size=7)
                                savefig(fig, f"{SAVEDIR}/tktrain_{task_kind_train}-tktest_{task_kind_test}-fig_by_condition-ev_train={ev_train}-ev_test={ev_test}-var_train={var_decode_train}-var_test={var_decode_test}.pdf")
                                fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "bregion", share_zlim=True, title_size=7, diverge=True, ZLIMS=(-0.5, 0.5))
                                savefig(fig, f"{SAVEDIR}/tktrain_{task_kind_train}-tktest_{task_kind_test}-fig_by_condition-ev_train={ev_train}-ev_test={ev_test}-var_train={var_decode_train}-var_test={var_decode_test}-score_adjusted-zlims.pdf")
                            plt.close("all")
                    else:
                        list_var_decode = sort_mixed_type(set(DFRES["var_decode"].tolist()))
                        for var_decode in list_var_decode:
                            print("Plotting var_decode: ", var_decode)

                            dfres_this = dfthis[
                                    (dfthis["var_decode"] == var_decode)].reset_index(drop=True)

                            if len(dfres_this)==0:
                                print("! no data...")
                                continue

                            fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "bregion", share_zlim=True, title_size=7, diverge=True)
                            savefig(fig, f"{SAVEDIR}/BYVAR-tktrain={task_kind_train}-tktest={task_kind_test}-evtrain={ev_train}-evtest={ev_test}-var={var_decode}-score_adjusted.pdf")

                            if False: # Takes too long
                                fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score", "bregion", share_zlim=True, title_size=7)
                                savefig(fig, f"{SAVEDIR}/tktrain_{task_kind_train}-tktest_{task_kind_test}-fig_by_condition-ev_train={ev_train}-ev_test={ev_test}-var_decode={var_decode}.pdf")
                                fig, axes = plot_subplots_heatmap(dfres_this, "time_train", "time_test", "score_adjusted", "bregion", share_zlim=True, title_size=7, diverge=True, ZLIMS=(-0.5, 0.5))
                                savefig(fig, f"{SAVEDIR}/tktrain_{task_kind_train}-tktest_{task_kind_test}-fig_by_condition-ev_train={ev_train}-ev_test={ev_test}-var_decode={var_decode}-score_adjusted-zlims.pdf")

                            plt.close("all")


def decodewrapouterloop_categorical_cross_time(DFallpa, list_var_decode, time_bin_size,
                                               slide, savedir=None, extract_params=True,
                                               ignore_same_events=True, which_level="trial"):
    """ Decode across events_taskkinds and time, asking how decoder generalizes.
    Should run this outer loop. goes over all
    events and task_kinds.
    """

    if extract_params:
        list_br, list_tw, list_ev, n_strokes_max = decodewrapouterloop_preprocess_extract_params(DFallpa)
    else:
        list_br = DFallpa['bregion'].unique().tolist()
        list_tw = DFallpa['twind'].unique().tolist()
        list_ev = DFallpa['event'].unique().tolist()
        n_strokes_max = None

    assert len(DFallpa["twind"].unique())==1, "not big deal. just change code below to iter over all (ev, tw)."

    RES = []
    for br in list_br:
        for tw in list_tw:
            for i, ev_train in enumerate(list_ev):
                for j, ev_test in enumerate(list_ev):

                    if ignore_same_events and j<i: # Dont so same events.
                        print("skipping same event")
                    else:
                        print(br, tw, ev_train, ev_test)

                        # TRAIN
                        PA_train_orig = extract_single_pa(DFallpa, br, tw, event=ev_train, which_level=which_level)
                        PA_test_orig = extract_single_pa(DFallpa, br, tw, event=ev_test, which_level=which_level)

                        list_task_kind_train = PA_train_orig.Xlabels["trials"]["task_kind"].unique()
                        list_task_kind_test = PA_test_orig.Xlabels["trials"]["task_kind"].unique()

                        for task_kind_train in list_task_kind_train:
                            for task_kind_test in list_task_kind_test:

                                PA_train = PA_train_orig.slice_by_labels("trials", "task_kind", [task_kind_train])

                                # print("JHERER", PA_test_orig.X.shape)
                                PA_test = PA_test_orig.slice_by_labels("trials", "task_kind", [task_kind_test])

                                if PA_train.X.shape[1]==0 or PA_test.X.shape[1]==0:
                                    continue
                                
                                
                                # if time_bin_size is not None:
                                #     print(PA_train.X.shape)
                                #     PA_train = PA_train.agg_by_time_windows_binned(time_bin_size, slide)
                                #
                                # # TEST
                                # if time_bin_size is not None:
                                #     PA_test = PA_test.agg_by_time_windows_binned(time_bin_size, slide)

                                for var_decode in list_var_decode:

                                    X_train, labels_train, times_train = preprocess_extract_X_and_labels(PA_train,
                                                                                         var_decode, time_bin_size, slide)
                                    # print(PA_test.Times)
                                    # assert False
                                    X_test, labels_test, times_test = preprocess_extract_X_and_labels(PA_test,
                                                                                          var_decode, time_bin_size, slide)


                                    if X_train is None or X_test is None:
                                        print("SKIPPING, becuase only not enough data:")
                                        continue
                                    if len(set(labels_train))<2 or len(set(labels_test))<2:
                                        print("SKIPPING, becuase only one label:")
                                        print("Train:", set(labels_train))
                                        print("Test:", set(labels_test))
                                        continue

                                    # Only do splits if these are same trials
                                    do_train_test_kfold_splits = labels_train==labels_test

                                    # print("HERERE", X_train.shape)
                                    # print("HERERE", len(labels_train))
                                    # print("HERERE", X_test.shape)
                                    # print("HERERE", len(labels_test))

                                    if savedir is not None:
                                        savepath_ndata = f"{savedir}/{br}-{tw}-evtrain={ev_train}-evtest={ev_test}-tktrain={task_kind_train}-tktest={task_kind_test}-var={var_decode}"
                                    else:
                                        savepath_ndata = None

                                    res = decodewrap_categorical_cross_time(X_train, labels_train, times_train,
                                                                      X_test, labels_test, times_test,
                                                                      do_std=False, do_train_test_kfold_splits=do_train_test_kfold_splits,
                                                                            savepath_ndata=savepath_ndata)

                                    for r in res:
                                        r["var_decode"]=var_decode
                                        r["bregion"]=br
                                        r["event_train"]=ev_train
                                        r["event_test"]=ev_test
                                        r["task_kind_train"] = task_kind_train
                                        r["task_kind_test"] = task_kind_test

                                    RES.extend(res)

    DFRES = pd.DataFrame(RES)

    # save results (too large, like 1GB. This does to like 10MB).
    if savedir is not None:
        dfres = DFRES.copy()
        dfres = dfres.drop(["conf_scores", "labels_test", "labels_predicted"], axis=1)
        path = f"{savedir}/DFRES.pkl"
        pd.to_pickle(dfres, path)

    return DFRES


def decodewrapouterloop_categorical_cross_time_cross_var(DFallpa, list_var_decode_train_test,
                                                         time_bin_size, slide, savedir=None, extract_params=True,
                                                         ignore_same_events=True, which_level="trial"):
    """ Decode across events_taskkinds and time, asking how decoder generalizes, where
    now trains to decode one variable, and asks how well that decoder generalizes to
    decode another variable. THis only works for things liek seqc_0_shape --> seqc_1_shape,
    i.e,, variables, with semanticalyl overlapping labels.

    Should run this outer loop. goes over all
    events and task_kinds.

    :param: list_var_decode_train_test, list of list, where each inner 2-list is [var_train, var_test]
    """
    if extract_params:
        list_br, list_tw, list_ev, n_strokes_max = decodewrapouterloop_preprocess_extract_params(DFallpa)
    else:
        list_br = DFallpa['bregion'].unique().tolist()
        list_tw = DFallpa['twind'].unique().tolist()
        list_ev = DFallpa['event'].unique().tolist()
        n_strokes_max = None

    assert len(DFallpa["twind"].unique())==1, "not big deal. just change code below to iter over all (ev, tw)."

    RES = []
    for br in list_br:
        for tw in list_tw:
            for i, ev_train in enumerate(list_ev):
                for j, ev_test in enumerate(list_ev):
                    if ignore_same_events and j<=i: # Dont so same events.
                        print("skipping same event")
                    else:
                        print(br, tw, ev_train, ev_test)

                        # TRAIN
                        PA_train_orig = extract_single_pa(DFallpa, br, tw, event=ev_train, which_level=which_level)
                        PA_test_orig = extract_single_pa(DFallpa, br, tw, event=ev_test, which_level=which_level)

                        list_task_kind_train = PA_train_orig.Xlabels["trials"]["task_kind"].unique()
                        list_task_kind_test = PA_test_orig.Xlabels["trials"]["task_kind"].unique()

                        for task_kind_train in list_task_kind_train:
                            for task_kind_test in list_task_kind_test:

                                PA_train = PA_train_orig.slice_by_labels("trials", "task_kind", [task_kind_train])

                                # print("JHERER", PA_test_orig.X.shape)
                                PA_test = PA_test_orig.slice_by_labels("trials", "task_kind", [task_kind_test])

                                if PA_train.X.shape[1]==0 or PA_test.X.shape[1]==0:
                                    continue

                                for var_decode_train, var_decode_test in list_var_decode_train_test:

                                    X_train, labels_train, times_train = preprocess_extract_X_and_labels(PA_train,
                                                                                         var_decode_train, time_bin_size, slide)
                                    X_test, labels_test, times_test = preprocess_extract_X_and_labels(PA_test,
                                                                                          var_decode_test, time_bin_size, slide)

                                    if X_train is None or X_test is None:
                                        print("SKIPPING, becuase only not enough data:")
                                        continue

                                    if len(set(labels_train))<2 or len(set(labels_test))<2:
                                        print("SKIPPING, becuase only one label:")
                                        print("Train:", set(labels_train))
                                        print("Test:", set(labels_test))
                                        continue

                                    # Only do splits if these are same trials
                                    do_train_test_kfold_splits = labels_train==labels_test

                                    # print("HERERE", X_train.shape)
                                    # print("HERERE", len(labels_train))
                                    # print("HERERE", X_test.shape)
                                    # print("HERERE", len(labels_test))

                                    if savedir is not None:
                                        savepath_ndata = f"{savedir}/{br}-{tw}-evtrain={ev_train}-evtest={ev_test}-tktrain={task_kind_train}-tktest={task_kind_test}-vartrain={var_decode_train}-vartest={var_decode_test}"
                                    else:
                                        savepath_ndata = None

                                    res = decodewrap_categorical_cross_time(X_train, labels_train, times_train,
                                                                      X_test, labels_test, times_test,
                                                                      do_std=False,
                                                                        do_train_test_kfold_splits=do_train_test_kfold_splits,
                                                                        savepath_ndata=savepath_ndata)

                                    for r in res:
                                        r["var_decode_train"]=var_decode_train
                                        r["var_decode_test"]=var_decode_test
                                        r["bregion"]=br
                                        r["event_train"]=ev_train
                                        r["event_test"]=ev_test
                                        r["task_kind_train"] = task_kind_train
                                        r["task_kind_test"] = task_kind_test
                                    RES.extend(res)
    DFRES = pd.DataFrame(RES)

    # save results (too large, like 1GB. This does to like 10MB).
    if savedir is not None:
        dfres = DFRES.copy()
        print(len(dfres))
        dfres = dfres.drop(["conf_scores", "labels_test", "labels_predicted"], axis=1)
        path = f"{savedir}/DFRES.pkl"
        pd.to_pickle(dfres, path)

    return DFRES

def decodewrap_categorical_cross_time(X_train, labels_train, times_train,
                                      X_test, labels_test, times_test,
                                      do_std=False, do_train_test_kfold_splits=True,
                                      n_splits=4, savepath_ndata=None):
    """
    Time-resolved decoding where each time bin for X_train is used to train decoder, which
    is tested against each time bin in X_test.
    INputs should NOT have bneen kfold splitted already
    PARAMS:
    - X_train, (ndims, ntrials_1, ntimes_1),
    - labels_train (ntrials_1,), class labels for training data.
    - X_test, (ndims, ntrials_2, ntimes_2), time-varying data to apply decoder to across time.
    - labels_test (ntrials_2,), class labels for test data, static across time.
    - do_train_test_kfold_splits, bool, if True, then does kfold train-test splits, which is important
    if the train and test data are from same trials. Will fail if it detects that they are not same trails (by
    comparing labels).
    - savepath_ndata, str, path to save data on n labels per category. This is path minus extension.
    """

    assert X_train.shape[1]==len(labels_train)
    assert X_test.shape[1]==len(labels_test)

    assert X_train.shape[0]==X_test.shape[0]

    assert len(X_train.shape)==3
    assert len(X_test.shape)==3

    assert X_train.shape[2]==len(times_train)
    assert X_test.shape[2]==len(times_test)


    # Make sure training and testing data have the same set of classes
    # (prune if not).
    # DO this here, before saving labels ndat
    inds_keep_train, inds_keep_test = preprocess_match_training_and_test_labels(labels_train, labels_test)
    X_train = X_train[:, inds_keep_train, :]
    labels_train = [labels_train[i] for i in inds_keep_train]
    X_test = X_test[:, inds_keep_test, :]
    labels_test = [labels_test[i] for i in inds_keep_test]

    if savepath_ndata is not None:
        # Save sample size info
        from pythonlib.tools.pandastools import grouping_print_n_samples
        dftmp = pd.DataFrame({"var_decode":labels_train})
        grouping_print_n_samples(dftmp, ["var_decode"], savepath=f"{savepath_ndata}-TRAIN.yaml", save_as="txt")
        dftmp = pd.DataFrame({"var_decode":labels_test})
        grouping_print_n_samples(dftmp, ["var_decode"], savepath=f"{savepath_ndata}-TEST.yaml", save_as="txt")

    # Go thru each time bin for the training data:
    RES = []
    for tbin_train, time_train in enumerate(times_train):
        if do_train_test_kfold_splits:
            # Multiple kfolds.
            from sklearn.model_selection import StratifiedKFold

            # Make split
            assert labels_train == labels_test, "only do kfold splits if these are same trials..."
            labels = labels_train # pick either one...
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

            for i, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

                x_train_this = X_train[:, train_index, tbin_train].T # (trials, chans)
                labels_train_this = [labels[_i] for _i in train_index]

                X_test_this = X_test[:, test_index, :]
                labels_test_this = [labels[_i] for _i in test_index]

                res = decodewrap_categorical_single_decoder_across_time(x_train_this, labels_train_this,
                                                                        X_test_this, labels_test_this,
                                                                        times_test, do_std=do_std,
                                                                        do_train_test_kfold_splits=False)
                if res is not None:
                    for r in res:
                        r["iter_kfold"] = i
                        r["tbin_train"] = tbin_train
                        r["time_train"] = time_train
                        r["tbin_test"] = r["tbin"]
                        r["time_test"] = r["time"]
                        del r["time"]
                        del r["tbin"]
                    RES.extend(res)
        else:
            # one train-test score.
            x_train = X_train[:,:,tbin_train].T # (trials, chans)
            res = decodewrap_categorical_single_decoder_across_time(x_train, labels_train, X_test, labels_test,
                                                                  times_test, do_std=do_std,
                                                                    do_train_test_kfold_splits=False)
            if res is not None:
                for r in res:
                    r["tbin_train"] = tbin_train
                    r["time_train"] = time_train
                    r["tbin_test"] = r["tbin"]
                    r["time_test"] = r["time"]
                    del r["time"]
                    del r["tbin"]
                RES.extend(res)
    return RES

#################################### HELPER FUNCTIONS
def preprocess_match_training_and_test_labels(labels_train, labels_test):
    """
    Return indices such athat the resulting labels have identical classes,
    (i.e., to prune each list)
    RETURNS:
        - inds_keep_train, list of ints, idnices into labels_train,
        - inds_keep_test,...
    """
    classes_shared = [x for x in set(labels_train) if x in set(labels_test)]
    inds_keep_train = [i for i, x in enumerate(labels_train) if x in classes_shared]
    inds_keep_test = [i for i, x in enumerate(labels_test) if x in classes_shared]

    return inds_keep_train, inds_keep_test

def preprocess_extract_X_and_labels(PA, var_decode, time_bin_size=None,
                                    slide=None, n_min_trials=N_MIN_TRIALS):
    """ extract X and labels, converting labels to ints, and
    pruning classes which dont have neough data.
    RETURNS:
        - X, (chans, trials, times)
        - labels, list of values of <var_decode>
        - times, time bins.
    """
    from pythonlib.tools.pandastools import filter_by_min_n

    if PA.X.shape[1]==0:
        return None, None, None

    if time_bin_size is not None:
        PA = PA.agg_by_time_windows_binned(time_bin_size, slide)

    X = PA.X
    times = PA.Times
    dflab = PA.Xlabels["trials"]
    # print("HERE", dflab.columns, len(dflab), PA.X.shape)
    labels = dflab[var_decode].tolist()
    # labels_int, tmp = pd.factorize(dflab[var_decode])
    # map_int_to_lab = {i:lab for i, lab in enumerate(tmp)}

    # Prune to cases with enoughd ata
    dftmp = filter_by_min_n(dflab, var_decode, n_min_trials)
    indskeep = dftmp["_index"].tolist()
    X = X[:, indskeep, :] # (chans, trials, times)
    labels = [labels[i] for i in indskeep]

    # Remove ignores
    X, labels, inds_keep = cleanup_remove_labels_ignore(X, labels)

    if len(labels)==0:
        return None, None, None

    return X, labels, times


def preprocess_factorize_class_labels_ints(DFallpa, savepath=None):
    """
    Convert all shape and location class labels to integers, which are identical across
    all pa in DFallpa, and which covers all variables iwth strings ("shape") and ("loc"),
    except those with classes in labels_to_ignore
    PARAMS:
    - savepath, full path to save dict holding mapping between ints and labels.
    RETURNS:
        - (Modifies without returning, DFallpa, with pa updated so its labeles are ints)
        - (REturns) MAP_LABELS_TO_INT, holding params, including maps from int to class labels.
    """


    ################# GET LIST OF VARIABLES
    pa = DFallpa["pa"].values[0]
    dflab = pa.Xlabels["trials"]

    # CODE IN PORGRESS - to extract all vars that should be factorized -- decided to hand code instead, as this wasnt accurate.
    # pa = DFallpa["pa"].values[0]
    # dflab = pa.Xlabels["trials"]
    #
    # cols_keep = []
    # for col in dflab.columns:
    #     # a = ("loc" in col) or ("shape" in col) or ("taskconfig" in col) or ("_binned" in col)
    #     a = True
    #     b = ("list" not in col)
    #     c = ("shape_is_novel_" not in col)
    #     d = "velocity" not in col
    #     e = (col!="shape_semantic_labels")
    #     # f = (isinstance(dflab[col].values[0], str)) or (isinstance(dflab[col].values[0], tuple) and isinstance(dflab[col].values[0][0], (str, int)))
    #     f = (isinstance(dflab[col].values[0], tuple) and isinstance(dflab[col].values[0][0], (str, int)))
    #     g = ("locx" not in col)
    #     h = ("locy" not in col)
    #     # k = ("locon" not in col)
    #
    #     if a & b & c & d & e & f & g & h:
    #         cols_keep.append(col)
    #
    # print(cols_keep)



    # def _extract_location_variables(dfthis):
    #     list_variables = [col for col in dfthis.columns if "loc" in col]
    #     list_variables = [col for col in list_variables if not col=="velocity" and ("list" not in col)]
    #     list_variables = [var for var in list_variables if not isinstance(dfthis[var].values[0], str)]
    #     return list_variables
    #
    # def _extract_shape_variables(dfthis):
    #     shape_variables = [col for col in dfthis.columns if ("shape" in col) and ("shape_is_novel_" not in col) and (col!="shape_semantic_labels") and ("list" not in col)]
    #     shape_variables = [var for var in shape_variables if isinstance(dfthis[var].values[0], str)]
    #     return shape_variables
    #
    # def _extract_taskconfig_variables(dfthis):
    #     list_variables = [col for col in dfthis.columns if ("taskconfig_" in col) and ("list" not in col)]
    #     list_variables = [var for var in list_variables if isinstance(dfthis[var].values[0], tuple)]
    #     return list_variables

    loc_variables = ["gridloc", "loc_this_event", "CTXT_loc_prev", "CTXT_loc_next"]
    for i in range(10):
        if f"seqc_{i}_loc" in dflab.columns:
            loc_variables.append(f"seqc_{i}_loc")
        if f"seqc_{i}_locon_binned" in dflab.columns:
            loc_variables.append(f"seqc_{i}_locon_binned")
        if f"seqc_{i}_locon_bin_in_loc" in dflab.columns:
            loc_variables.append(f"seqc_{i}_locon_bin_in_loc")
        if f"seqc_{i}_center_binned" in dflab.columns:
            loc_variables.append(f"seqc_{i}_center_binned")

    shape_variables = ["shape", "shape_this_event", "shape_oriented", "CTXT_shape_next", "CTXT_shape_prev"]
    for i in range(10):
        if f"seqc_{i}_shape" in dflab.columns:
            shape_variables.append(f"seqc_{i}_shape")
        if f"seqc_{i}_shapesem" in dflab.columns:
            shape_variables.append(f"seqc_{i}_shapesem")
        if f"seqc_{i}_shapesemcat" in dflab.columns:
            shape_variables.append(f"seqc_{i}_shapesemcat")

    # Is tuples, so must convert.
    taskconfig_variables = [var for var in dflab.columns if "taskconfig" in var]

    ########################################
    # def _extract_list_variables_all(df)
    def _isin(val, list_vals):
        """ Return True iff val is in list_vals"""
        from pythonlib.tools.checktools import check_objects_identical
        for v in list_vals:
            if check_objects_identical(v, val):
                return True
        return False

    MAP_LABELS_TO_INT = {}
    def _replace_labels(name, list_variables):
        # ------- RUN
        map_int_to_class = {}
        ct=0
        for i, row in DFallpa.iterrows():
            dflab = row["pa"].Xlabels["trials"]
            for var in list_variables:
                if var in dflab.columns:
                    classes = dflab[var].unique()
                    for cl in classes:
                        try:
                            if cl in LABELS_IGNORE:
                                if cl not in map_int_to_class:
                                    map_int_to_class[cl] = cl
                            else:
                                if not _isin(cl, list(map_int_to_class.values())):
                                # if cl not in list(map_int_to_class.values()):
                                    map_int_to_class[ct] = cl
                                    ct +=1
                        except Exception as err:
                            print(var)
                            print(list_variables)
                            print(classes)
                            print(map_int_to_class.values())
                            for x in map_int_to_class.values():
                                print(type(x))
                            print(cl)
                            print(type(cl))
                            print("Issue with  mixed types...?")
                            raise err
        map_class_to_int = {cl:i for i, cl in map_int_to_class.items()}

        for k, v in map_class_to_int.items():
            print(k, " --- ", v)
        for i, row in DFallpa.iterrows():
            pathis = row["pa"]
            dflab = pathis.Xlabels["trials"]
            for var in list_variables:
                if var in dflab.columns:
                    dflab[var] = [map_class_to_int[cl] for cl in dflab[var]]
        MAP_LABELS_TO_INT[name] = {
            "labels_to_ignore":LABELS_IGNORE,
            "map_int_to_class":map_int_to_class,
            "map_class_to_int":map_class_to_int
        }

    ### TASKCONFIG (tuples of shape or loc)
    # collect set of all classes across all data
    _replace_labels("taskconfig", taskconfig_variables)

    ### SHAPE
    _replace_labels("shape", shape_variables)

    ### LOCATION
    _replace_labels("loc", loc_variables)

    # SAVE
    if savepath is not None:
        from pythonlib.tools.expttools import writeDictToTxt
        writeDictToTxt(MAP_LABELS_TO_INT, savepath)

    return MAP_LABELS_TO_INT

def cleanup_remove_labels_ignore(X, labels):
    """
    Remove trials that have classes matching those in LABELS_IGNORE
    PARAMS:
    - X, eitehr (chans, trials, times) or (trials, chans)
    - labels, list, len trials.
    RETURNS:
    - X and labels, pruned in the trials dimension. Copies.
    """

    inds_keep = [i for i, l in enumerate(labels) if l not in LABELS_IGNORE]

    # Prune X, finding the correct dimension
    if len(X.shape)==3 and X.shape[1]==len(labels):
        X = X[:, inds_keep, :]
    elif len(X.shape)==2 and X.shape[0]==len(labels):
        X = X[inds_keep, :]
    else:
        print(X.shape)
        print(len(labels))
        assert False, "inputed wrong shaep"
    labels = [labels[i] for i in inds_keep]

    return X, labels, inds_keep