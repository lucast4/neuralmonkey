"""
Methods for For computing and plots of eucldian dsitances, all should collect here.
Usually using approach of pairwise distances between trials, then splitting/grouping by different label relations.

Previously:
- state_space_good.py --> For scoring euclidian distance
- analy_euclidian_dist_pop_script.py --> Scripts for running entire pipeline (generic).
--- This is too unweildy, trying to be too flexible across expts.
Expt-specific plots
- psychometric
- char_sp
- shape_invariance.

HERE: collects all the functions that generate from PA.
"""
from pythonlib.tools.pandastools import append_col_with_grp_index
import numpy as np
from pythonlib.tools.plottools import savefig
import matplotlib.pyplot as plt
import pandas as pd
import os

def timevarying_compute(PA, vars_group):
    """
    Generate dataset of parwise distances between each level of each variable in vars_group.
    Gets value for each time bin independelty, which makes this slow.
    This is the old method.
    PARAMS:
    - vars_group, list of str, variables that define groups.
    """

    version = "traj"
    DFDIST = PA.dataextractwrap_distance_between_groups(vars_group, version)

    # Add column names reflecting the "sameness" state of variables.
    for var in vars_group:
        DFDIST[f"{var}_same"] = DFDIST[f"{var}_1"] == DFDIST[f"{var}_2"]
        DFDIST = append_col_with_grp_index(DFDIST, [f"{var}_1", f"{var}_2"], f"{var}_12")
    
    for i in range(len(vars_group)):
        for j in range(len(vars_group)):
            if j>i:
                var1 = vars_group[0]
                var2 = vars_group[1]
                DFDIST = append_col_with_grp_index(DFDIST, [f"{var1}_same", f"{var2}_same"], f"same-{var1}|{var2}")

    return DFDIST

def timevarying_convert_to_scalar(DFDIST, twind_scalar):
    """
    Genreate a dataframe of scalar vallues suing this timw window,
    given the time-resolved distances in DFDIST.

    PARAMS:
    - DFDIST, holds distances for each time bin, is returned from timevarying_compute()
    """
    from pythonlib.tools.pandastools import aggregGeneral

    dfthis_sub = DFDIST[(DFDIST["time_bin"]>=twind_scalar[0]-0.001) & (DFDIST["time_bin"]<=twind_scalar[1]+0.001)].reset_index(drop=True)

    # Agg, averaging over time
    # dfscal = aggregGeneral(dfthis_sub, ["animal", "date", "combine_areas", "event", "bregion", "metaparams", "same-task|shape", "prune_version", "subspace_projection", "remove_drift", "raw_subtract_mean_each_timepoint", 
    #                                 "remove_singleprims_unstable"], values=["dist_mean", "dist_norm", "dist_yue_diff"])
    dfscal = aggregGeneral(dfthis_sub, ["labels_1", "labels_2"], values=["dist_mean", "dist_norm", "dist_yue_diff"], nonnumercols="all")

    return dfscal


def timevarying_compute_fast_to_scalar(PA, label_vars=("seqc_0_shape", "seqc_0_loc"),
                                       rsa_heatmap_savedir=None, var_context_same=None,
                                       plot_conjunctions_savedir=None, prune_levs_min_n_trials=2,
                                       get_group_distances=True, context_dict=None,
                                       get_only_one_direction=True):
    """
    Compute pariwise euclidean distance, using trajectories.

    [Fast code] -- do all steps to extract dfdist, starting from PA.

    POTENTIAL PROBLOEM (is ok, emeprically): it doesnt get time-varying, it goes straight from (ndims, ntimes) --> scalar,
    for each trial. This is so that it can work with the distmat_construct_wrapper helper
    function, but it is really not necesary, and would be a quick way to get time-varying also,
    if needed --> This also means that the final dist_yue_diff uses the scalars after agging
    across time, instead of computing in each bin, then averaging them. May have slight difference,
    but not too much.
    
    PARAMS:
    - var_context_same, if is not None, then only takes pairs of datapts that have the same level for this group. This is like
    "controlling" for this variable. E.g., hold size constant (var_context_same="gridsize"), while testing for
    effect of shape and location.
    - prune_levs_min_n_trials, then throws out any levels of grouping vars, label_vars + [var_context_same], which lack at least 2
    trials. Need at laest 2, otherwise error in dist computation.
    - context_dict, dict with {"same":[], "diff":[]}, where each holds list of strings (variables).

    MS: checked
    """
    from pythonlib.tools.distfunctools import distmat_construct_wrapper
    from pythonlib.cluster.clustclass import Clusters

    # Deprecated, beucase:
    # "Confirmed that if you set context_dict[same]=[var_context_same], this will work the same (better)"
    assert var_context_same is None, "deprecated. Instead, use context_dict[same]=..., as this is hacky and not general."

    # (ndims, ntrials, ntimes)
    # --> (ndims, ntimes) X ntrials.
    # trial1 = 0
    # trial2 = 1
    # x1 = pa.X[:, trial1, :] 
    # x2 = pa.X[:, trial2, :] 

    # x1.shape
    # import numpy as np

    # res = []
    # for trial1 in range(ntrials):
    #     print(trial1)
    #     for trial2 in range(ntrials):
    #         x1 = pa.X[:, trial1, :] 
    #         x2 = pa.X[:, trial2, :] 

    #         x1.shape

    #         (np.sum((x1 - x2)**2, axis=0))**0.5

    ###### ALTERANTIVE METHODS:
    if False:
        # (1) SLOWEST: Usual way of computing
        DFDIST = timevarying_compute(PA, vars_group)
        
        # (2) FAST! but not as fast.
        # It works, output is idneticla, but is about 20-40% slower (1.2 sec vs. 0.9 sec).
        DIAGONAL_VALUE = 0
        distmat = np.zeros((len(indsall), len(indsall))) + DIAGONAL_VALUE
        # distmat = distmat - np.inf

        indsall = list(range(pa.X.shape[1]))

        for i in range(1, len(indsall)):
            print(i)
            
            inds1 = indsall[:len(indsall)-i]
            inds2 = indsall[i:]

            xmat1 = pa.X[:, inds1, :]
            xmat2 = pa.X[:, inds2, :]
            xmat2.shape

            xdiff = np.sum((xmat1 - xmat2)**2, axis=0)**0.5 # (ntrials, ntimes)
            # mean over time
            xdiff_scal = np.mean(xdiff, axis=1) # (ntrials, )

            # # store into distance matrix
            # for i1, i2, d in zip(inds1, inds2, xdiff_scal):
            #     distmat[i1, i2] = d
            #     distmat[i2, i1] = d

            # TO compare this with the output of outer function
            fig, ax = plt.subplots()
            ax.imshow(distmat)
            fig, ax = plt.subplots()
            ax.imshow(dmat)

            fig, ax = plt.subplots()
            ax.imshow(dmat - distmat)

            np.min(dmat - distmat)


    # Context dict preprocessing
    if var_context_same is not None:
        label_vars_orig = [l for l in label_vars]
        label_vars_for_cldist = tuple([l for l in label_vars] + [var_context_same])

    label_vars_for_cldist = [l for l in label_vars]
    if context_dict is not None:
        if context_dict["same"] is not None:
            label_vars_for_cldist = tuple([l for l in label_vars_for_cldist] + context_dict["same"])
        if context_dict["diff"] is not None:
            label_vars_for_cldist = tuple([l for l in label_vars_for_cldist] + context_dict["diff"])

    ### Prune levels
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good, grouping_plot_n_samples_conjunction_heatmap_helper
    dflab = PA.Xlabels["trials"]
    _, _indskeep = extract_with_levels_of_var_good(dflab, label_vars_for_cldist, prune_levs_min_n_trials)

    # - Save counts before prune
    if plot_conjunctions_savedir is not None and len(_indskeep)<len(dflab):
        fig = grouping_plot_n_samples_conjunction_heatmap_helper(dflab, label_vars_for_cldist)
        savefig(fig, f"{plot_conjunctions_savedir}/timevarying_compute_fast_to_scalar-counts_heatmap-before_prune.pdf")

    # - Do slice
    PA = PA.slice_by_dim_indices_wrapper("trials", _indskeep, reset_trial_indices=True) 

    # - Save counts after prune
    if plot_conjunctions_savedir is not None:
        fig = grouping_plot_n_samples_conjunction_heatmap_helper(PA.Xlabels["trials"], label_vars_for_cldist)
        savefig(fig, f"{plot_conjunctions_savedir}/timevarying_compute_fast_to_scalar-counts_heatmap-final.pdf")
    plt.close("all")

    # Collect each trial.
    ntrials = PA.X.shape[1]
    list_x = []
    for trial in range(ntrials):
        list_x.append(PA.X[:, trial, :])
    if len(list_x) == 0:
        return None, None

    # Get distnace matrix.
    def dist_func(x1, x2):
        """
        Euclidean ditance
        x1, x2, (ndims, ntimes), a trial-slice.
        Return scalar distance, averaged over all time.
        """
        return np.mean((np.sum((x1 - x2)**2, axis=0))**0.5)
    dmat = distmat_construct_wrapper(list_x, list_x, dist_func)

    # Convert to Cl
    dflab = PA.Xlabels["trials"]
    list_lab = [tuple(x) for x in dflab.loc[:, label_vars_for_cldist].values.tolist()]
    params = {
        "label_vars":label_vars_for_cldist,
        "version_distance":"euclidian",
        "Clraw":None,
    }
    trialcodes = dflab["trialcode"].tolist()
    assert dmat.shape[0]==dmat.shape[1]==len(trialcodes)
    Cldist = Clusters(dmat, list_lab, list_lab, ver="dist", params=params, trialcodes=trialcodes)

    if rsa_heatmap_savedir is not None:
        from itertools import permutations
        # This fn also returns dfdist. The reason I dont keep ti is that here I get both directions. I am not sure
        # if this would do wierd things downstream
        zlims = None
        # zlims = [0, 0.3]
        _, CldistAgg = Cldist.rsa_distmat_score_all_pairs_of_label_groups(label_vars=label_vars, get_only_one_direction=False, 
                                                                          return_as_clustclass=True,
                                                                          context_dict=context_dict)
        n = min([len(label_vars), 3])
        list_sort_order = sorted(permutations(range(n)))
        for sort_order in list_sort_order:

            fig, ax = CldistAgg.rsa_plot_heatmap(sort_order, zlims=zlims)
            savefig(fig, f"{rsa_heatmap_savedir}/rsa_heatmap-sort_order={sort_order}.pdf")

            varthis = label_vars[sort_order[0]]
            _, fig = CldistAgg.rsa_distmat_construct_theoretical(varthis, PLOT=True, sort_order=list_sort_order[0]) # use the same sort order for each var so can compare them
            savefig(fig, f"{rsa_heatmap_savedir}/rsa_heatmap-var={varthis}-sort_order={list_sort_order[0]}-THEOR.pdf")
            plt.close("all")
        
        # There might be nan, so save that
        ma_not_nan = ~np.isnan(CldistAgg.Xinput)
        CldistAgg.rsa_matindex_print_mask_labels(ma_not_nan, f"{rsa_heatmap_savedir}/rsa_heatmap-not_nan.txt")
        ma_nan = np.isnan(CldistAgg.Xinput)
        CldistAgg.rsa_matindex_print_mask_labels(ma_nan, f"{rsa_heatmap_savedir}/rsa_heatmap-is_nan.txt")

    if get_group_distances:
        # convert to cldist.
        dfdist = Cldist.rsa_distmat_score_all_pairs_of_label_groups(label_vars=label_vars, get_only_one_direction=get_only_one_direction, 
                                                                    context_dict=context_dict)
        
        #### If this has context input, then additional steps
        if var_context_same is not None:
            if False:
                from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
                grouping_plot_n_samples_conjunction_heatmap(dfdist, "seqc_0_shape_12", "seqc_0_loc_12", ["gridsize_12"])

            from pythonlib.tools.pandastools import append_col_with_grp_index

            # This is usualyl the case, so just do it.
            if len(label_vars_orig)>1:
                var_effect = label_vars_orig[0]
                var_other = label_vars_orig[1]
                var_same_same = f"same-{var_effect}|{var_other}"
                dfdist = append_col_with_grp_index(dfdist, [f"{var_effect}_same", f"{var_other}_same"], var_same_same)

            # Keep only pairs that have the same context
            dfdist = dfdist[dfdist[f"{var_context_same}_same"]==True].reset_index(drop=True)

            # Agg across levels of context
            # -- 
            from pythonlib.tools.pandastools import aggregGeneral
            group = [f"{v}_12" for v in label_vars_orig] # each unique kind of pair
            dfdist = aggregGeneral(dfdist, group, ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff"], nonnumercols="all")

            # Reassign labels using just (var_eff, var_other)
            for i in [1,2]:
                grp = [f"{v}_{i}" for v in label_vars_orig]
                dfdist = append_col_with_grp_index(dfdist, grp, f"labels_{i}", False)
    else:
        dfdist = None

    return dfdist, Cldist


def dfdist_postprocess_condition_prune_to_var_pairs_exist(dfdist, var_effect, var_context,
                                                          plot_counts_savedir=None):
    """
    Keep only var_effect levels that exist across all levels of context.
    This is used for postprocessing, generally, for cleaning dfdist.
    
    Note this is assymetric, var_effect vs. var_context

    Assumes that the following columns exist:
    <var_effect>_1
    <var_effect>_2
    <var_context>_1
    <var_context>_2
    """
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

    # (2) Keep only those shapes that have data across the pair of (task_kind, si_is_first)

    # First, collect all (var1, var2) levels
    tmp = dfdist.loc[:, [f"{var_effect}_1", f"{var_context}_1"]].values.tolist() + dfdist.loc[:, [f"{var_effect}_2", f"{var_context}_2"]].values.tolist()
    dftmp = pd.DataFrame(tmp, columns=[var_effect, var_context])

    if plot_counts_savedir:
        fig = grouping_plot_n_samples_conjunction_heatmap(dftmp, var_effect, var_context)
        savefig(fig, f"{plot_counts_savedir}/counts_before_prune_to_shapes_enough_data.pdf")

    # if plot_counts_savedir:
    #     plot_counts_heatmap_savepath = f"{plot_counts_savedir}/counts.pdf"
    # else:
    #     plot_counts_heatmap_savepath = None
    plot_counts_heatmap_savepath = None # redundant with the final plot below
    n_levs_context = len(dftmp[var_context].unique())
    dfout, _ = extract_with_levels_of_conjunction_vars_helper(dftmp, var_context, 
                                                [var_effect], 1, plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                                lenient_allow_data_if_has_n_levels=n_levs_context)
    
    if len(dfout)>0:
        shapes_keep = dfout[var_effect].unique().tolist()
    else:
        shapes_keep = []

    dfdist = dfdist[(dfdist[f"{var_effect}_1"].isin(shapes_keep)) & (dfdist[f"{var_effect}_2"].isin(shapes_keep))].reset_index(drop=True)

    if plot_counts_savedir:
        if len(dfout)>0: # somtimetimes shap emight not exist across these cases
            fig = grouping_plot_n_samples_conjunction_heatmap(dfout, var_effect, var_context)
            savefig(fig, f"{plot_counts_savedir}/counts_after_prune_to_shapes_enough_data.pdf")

    return dfdist

def dfdist_extract_label_vars_specific(dfdists, label_vars):
    """
    Automatically populates new columns reflecting the relations between the columns in 
    label_vars (which can be any length), such as same_shape

    Uses whatever is in labels_1 and labels_2
    
    PARAMS:
    - dfdists, output from things like rsa_distmat_score_all_pairs_of_label_groups
    - label_vars, list of n strings
    
    e.g., label_vars = [shape, loc], means that labels_1 is a column with items like (circle, (0,1)), and
    will populate new columns called shape_1, shape_2, loc_1, loc_2, etc....

    RETURNS:
    - copy of dfdists
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index

    if "labels_1_datapt" in dfdists.columns:
        var1 = "labels_1_datapt"
        var2 = "labels_2_grp"
    else:
        var1 = "labels_1"
        var2 = "labels_2"

    dfdists = dfdists.copy()

    assert isinstance(dfdists[var1].values[0], tuple)

    # Replace columns which are now incorrect
    # label_vars = ["seqc_0_shape", "seqc_0_loc"]
    assert isinstance(label_vars[0], str)

    # e..g, seqc_0_shape_1
    for i, var in enumerate(label_vars):
        dfdists[f"{var}_1"] = [x[i] for x in dfdists[var1]]
        dfdists[f"{var}_2"] = [x[i] for x in dfdists[var2]]
        dfdists = append_col_with_grp_index(dfdists, [f"{var}_1", f"{var}_2"], f"{var}_12")
        dfdists[f"{var}_same"] = dfdists[f"{var}_1"] == dfdists[f"{var}_2"]

    if len(label_vars)==2:
        for i, var in enumerate(label_vars):
            dfdists = append_col_with_grp_index(dfdists, [f"{label_vars[0]}_same", f"{label_vars[1]}_same"], f"same-{label_vars[0]}|{label_vars[1]}")

    return dfdists

def dfdist_expand_convert_from_triangular_to_full(self, dfdists, label_vars=None, PLOT=False,
                                                repopulate_relations=True):
    """
    Given a dfdists that is triangular (inclues diagonmal usually), convert to 
    full matrix by copying and swapping labels 1 and 2, assuming that
    distances are symmetric.
    
    RETURNS:
    - copy of dfdists, but more rows.
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

    if PLOT:
        grouping_plot_n_samples_conjunction_heatmap(dfdists, "labels_1", "labels_2");

    dftmp = dfdists.copy()

    # Flip labels
    dftmp["labels_1"] = dfdists["labels_2"]
    dftmp["labels_2"] = dfdists["labels_1"]
    
    # Remove diagonal
    dftmp = dftmp[dftmp["labels_1"]!=dftmp["labels_2"]]
    
    # concat
    dfdists = pd.concat([dfdists, dftmp]).reset_index(drop=True)

    if repopulate_relations:
        # Repopulation all label columns
        assert label_vars is not None        
        # label_vars = ["seqc_0_shape", var_other]
        # from pythonlib.cluster.clustclass import Clusters
        # cl = Clusters(None)
        dfdists = self.rsa_distmat_population_columns_label_relations(dfdists, label_vars)

    if PLOT:
        grouping_plot_n_samples_conjunction_heatmap(dfdists, "labels_1", "labels_2");

    # Sanity check that populated all cells in distance matrix
    if False: # I know this code works, so no need for this.
        assert dfdists.groupby(["labels_2"]).size().min() == dfdists.groupby(["labels_2"]).size().max()

    return dfdists

def dfdist_postprocess_wrapper(DFDISTS, var_effect, var_other, savedir, 
                               do_pruning_cleaning=True, prune_min_n_trials=None):
    """
    Wrapper for all usual postprocessing steps.
    DFDISTS can be across animals and dates and metaparams (those need to be columns, regardless)
    
    There is assymetry in var_effect vs var_other, but this works just fine for expts where you care about loking at both
    directions distances. But usually var_efect is like shape and var_other is like context (e.g., tasK_kind) the one with
    fewer levels.

    PARAMS:
    - prune_min_n_trials, None or int (4 is good) which means throws out data pair if it has less than 4 datapts in original data.
    """ 
    from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good, plot_subplots_heatmap, append_col_with_grp_index, sort_by_two_columns_separate_keys
    from neuralmonkey.analyses.euclidian_distance import dfdist_postprocess_condition_prune_to_var_pairs_exist
    from pythonlib.tools.pandastools import append_col_with_grp_index

    DFDISTS = DFDISTS.reset_index(drop=True)

    assert "animal" in DFDISTS.columns
    assert "date" in DFDISTS.columns

    if "metaparams" not in DFDISTS:
        tmp = ["prune_version", "subspace_projection", "subspace_twind", "remove_drift", "raw_subtract_mean_each_timepoint", "remove_singleprims_unstable"]
        possible_keys = []
        for k in tmp:
            if (k in DFDISTS) and (len(DFDISTS[k].unique())>1):
                possible_keys.append(k)
        DFDISTS = append_col_with_grp_index(DFDISTS, possible_keys, "metaparams")

    if "n_1_2" in DFDISTS:
        DFDISTS["n1"] = [int(x[0]) for x in DFDISTS["n_1_2"]]
        DFDISTS["n2"] = [int(x[1]) for x in DFDISTS["n_1_2"]]

    ### Make sure have all required columns
    if f"{var_effect}_1" not in DFDISTS:
        DFDISTS = dfdist_extract_label_vars_specific(DFDISTS, [var_effect, var_other])

    ### Only keep var_effect labels which are present in both contexts.
    if do_pruning_cleaning:
        grpdict = grouping_append_and_return_inner_items_good(DFDISTS, ["metaparams", "animal", "date"])
        list_dfdist =[]
        for grp, inds in grpdict.items():
            dfdists = DFDISTS.iloc[inds].reset_index(drop=True)
            plot_counts_savedir = f"{savedir}/{grp}"
            os.makedirs(plot_counts_savedir, exist_ok=True)
            dfdists_new = dfdist_postprocess_condition_prune_to_var_pairs_exist(dfdists, var_effect, var_other, plot_counts_savedir)
            list_dfdist.append(dfdists_new)

            plt.close("all")

            print(grp, "       |       ", len(dfdists), " ---> ", len(dfdists_new), f"[{len(dfdists_new)/len(dfdists):.2f}]")

        DFDISTS = pd.concat(list_dfdist).reset_index(drop=True)

    ### Keep only cases with at least a minimum number of trials.
    if prune_min_n_trials is not None:
        assert isinstance(prune_min_n_trials, int)
        # nmin = 4 # This many or more.

        n1 = len(DFDISTS)
        grpdict = grouping_append_and_return_inner_items_good(DFDISTS, ["metaparams", "animal", "date"])
        list_dfdist =[]
        for grp, inds in grpdict.items():
            dfdists = DFDISTS.iloc[inds].reset_index(drop=True)

            # Tally n per shape
            a = dfdists.loc[:, [f"{var_effect}_1", "n1", f"{var_other}_1"]].values.tolist()
            b = dfdists.loc[:, [f"{var_effect}_2", "n2", f"{var_other}_2"]].values.tolist()
            dftmp = pd.DataFrame(a+b, columns=[var_effect, "n", var_other])

            dftmp2 = dftmp.groupby([var_effect, "n", var_other]).mean()
            dftmp3 = dftmp.groupby([var_effect, "n", var_other]).mean()
            if len(dftmp2)!=len(dftmp3):
                print(dftmp2)
                print(dftmp3)
                assert False, "this means that there exists a (var_effect, task_kind) that has more than one unique n. this sghould not be possible. probaly multiole higher metarparams of some sort?"
            
            if False:
                from pythonlib.tools.pandastools import grouping_print_n_samples
                grouping_print_n_samples(dftmp, ["shape_semantic_grp", "n", "task_kind"])

            # The pool of shapes to ignore with too few trials.
            # - ignore a shape if it has less than n for _ANY_ task_kind or 
            shapes_ignore = dftmp[dftmp["n"] < prune_min_n_trials][var_effect].unique().tolist()

            # Keep just shapes that are not ignored
            a = dfdists[f"{var_effect}_1"].isin(shapes_ignore)
            b = dfdists[f"{var_effect}_2"].isin(shapes_ignore)

            dfdists_new = dfdists[~(a | b)].reset_index(drop=True)

            # Store
            list_dfdist.append(dfdists_new)

        DFDISTS = pd.concat(list_dfdist).reset_index(drop=True)
        n2 = len(DFDISTS)

        print(f"After pruning due to min num trials ({prune_min_n_trials}): ", n1, " --> ", n2)

    from pythonlib.tools.pandastools import aggregGeneral
    var_same_same = f"same-{var_effect}|{var_other}"
    DFDISTS_AGG = aggregGeneral(DFDISTS, ["bregion", "which_level", "event", var_same_same, "metaparams"],
                                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")

    return DFDISTS, DFDISTS_AGG


def dfdist_summary_plots_wrapper(DFDISTS, DFDISTS_AGG, var_effect, var_other, SAVEDIR,
                                 PLOT_EACH_PAIR=False, list_metaparams_plot_each_pair=None,
                                 do_catplots=True, do_quick=False):
    """
    Wrapper for all summary plots related to pairwise euclidean distances stored in DFDISTS.
    Including catplots, scatterplots, and plots of all pairwise comparisons
    
    PARAMS:
    - PLOT_EACH_PAIR, bool, if True, then plots heatmap of distances between each condition. Takes time.
    - list_metaparams_plot_each_pair, list of str, the metaparams levsl to plots for each pair, If None, then plots
    all. This is useful to reduce amount of time, focusing on just what matters.

    MS: checked
    """
    import seaborn as sns
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, plot_45scatter_means_flexible_grouping, grouping_append_and_return_inner_items_good
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import euclidian_time_resolved_fast_shuffled_mult_reload
    import os
    from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_append_and_return_inner_items_good
    from pythonlib.tools.pandastools import aggregGeneral, stringify_values
    import pandas as pd
    import matplotlib.pyplot as plt
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.pandastools import plot_subplots_heatmap, sort_by_two_columns_separate_keys

    var_same_same = f"same-{var_effect}|{var_other}"
    var_value = "dist_yue_diff"
    yvar = "dist_yue_diff"

    if do_quick:
        do_catplots = False

    if do_catplots:
        ### CATPLOTS
        # fig = grouping_plot_n_samples_conjunction_heatmap(DFDISTS, f"{var_effect}_1", f"{var_effect}_2", ["date", "metaparams"]);
        # savefig(fig, f"{savedir}/counts.pdf")
        # plt.close("all")

        savedir = f"{SAVEDIR}/catplots"
        os.makedirs(savedir, exist_ok=True)

        fig = sns.catplot(data=DFDISTS, x="bregion", y=yvar, hue=var_same_same, kind="bar", col="date", row="metaparams", aspect=1)
        savefig(fig, f"{savedir}/catplot-1.pdf")

        fig = sns.catplot(data=DFDISTS, x="bregion", y=yvar, hue=var_same_same, alpha=0.5, jitter=True, col="date", row="metaparams", aspect=1)
        savefig(fig, f"{savedir}/catplot-2.pdf")

        # Summary plot
        fig = sns.catplot(data=DFDISTS_AGG, x="bregion", y=yvar, hue=var_same_same, kind="bar", col="date", row="metaparams", aspect=1)
        savefig(fig, f"{savedir}/catplot-agg-1.pdf")

        fig = sns.catplot(data=DFDISTS_AGG, x="bregion", y=yvar, hue=var_same_same, alpha=0.5, jitter=True, col="date", row="metaparams", aspect=1)
        savefig(fig, f"{savedir}/catplot-agg-2.pdf")

    ### Scatter
    savedir = f"{SAVEDIR}/scatterplots"
    os.makedirs(savedir, exist_ok=True)
    
    if "prune_version" not in DFDISTS:
        DFDISTS["prune_version"] = "dummy"
    if "prune_version" not in DFDISTS_AGG:
        DFDISTS_AGG["prune_version"] = "dummy"

    # Each event
    if not do_quick:
        grp_vars = ["which_level", "prune_version", "event"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "subspace|twind", 
                                                var_value, "bregion", True, shareaxes=True)
            if fig is not None:
                savefig(fig, f"{savedir}/EVENTS-scatter45-grp={grp}.pdf")
                plt.close("all")
            
        # Each event
        # grp_vars = ["which_level", "metaparams"]
        grp_vars = ["which_level", "prune_version", "subspace|twind"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "event", 
                                                var_value, "bregion", True, shareaxes=True)
            
            if fig is not None:
                savefig(fig, f"{savedir}/SUBSPACE-scatter45-grp={grp}.pdf")
                plt.close("all")

        # Show each date
        grp_vars = ["which_level", "event", "metaparams"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "bregion", 
                                                var_value, "date", True, shareaxes=True)

            if fig is not None:        
                savefig(fig, f"{savedir}/DATES-scatter45-grp={grp}.pdf")
                plt.close("all")

        ## Each region
        grp_vars = ["bregion"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)
        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "metaparams", 
                                                var_value, "date", True, shareaxes=True, SIZE=3.5)
            
            if fig is not None:
                savefig(fig, f"{savedir}/REGIONS-scatter45-grp={grp}.pdf")
                plt.close("all")


    ### Plot
    dfthis_dat = DFDISTS_AGG
    _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "metaparams", 
                                        var_value, "bregion", True, shareaxes=True, SIZE=3.5)
    if fig is not None:
        savefig(fig, f"{savedir}/ALL.pdf")
        plt.close("all")

    # All dates
    grp_vars = ["which_level", "metaparams", "event"]
    grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)
    for grp, inds_dat in grpdict_dat.items():
        # inds_pval = grpdict_pval[grp]
        dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

        ### Plot
        _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "date", 
                                            var_value, "bregion", True, shareaxes=True)

        if fig is not None:
            savefig(fig, f"{savedir}/ALL_DATES-scatter45-grp={grp}.pdf")
            plt.close("all")

    ### Also plot each pair of conditions (low-level plots) in heatmap
    if not do_quick:
        if PLOT_EACH_PAIR:
            print("Plotting each pair...")
            DFDISTS = sort_by_two_columns_separate_keys(DFDISTS, "bregion", var_same_same)
            DFDISTS = append_col_with_grp_index(DFDISTS, ["bregion", var_same_same], "br_same")
            grpdict = grouping_append_and_return_inner_items_good(DFDISTS, ["metaparams", "animal", "date"])

            if list_metaparams_plot_each_pair is not None:
                grpdict = {grp:inds for grp, inds in grpdict.items() if grp[0] in list_metaparams_plot_each_pair}
                
            if len(grpdict)==0:
                print(DFDISTS["metaparams"].unique().tolist())
                print(list_metaparams_plot_each_pair)
                assert False, "typo?"
            
            for grp, inds in grpdict.items():
                dfdists = DFDISTS.iloc[inds].reset_index(drop=True)

                dfdists = dfdists[dfdists[var_same_same] != "1|1"].reset_index(drop=True) # to remove 1/4 of subplots.
                fig, _ = plot_subplots_heatmap(dfdists, f"{var_effect}_1", f"{var_effect}_2", "dist_yue_diff", "br_same", 
                                                False, True, ncols=6)

                savefig(fig, f"{savedir}/allpairs_heatmap-{grp}.pdf")
                plt.close("all")
        else:
            # Then just plot the counts, is faster
            print("Skipping PLOT_EACH_PAIR")
            grpdict = grouping_append_and_return_inner_items_good(DFDISTS, ["animal", "date", "metaparams"])
            for grp, inds in grpdict.items():
                dfdists = DFDISTS.iloc[inds].reset_index(drop=True)
                # grouping_print_n_samples(dfdists, ["animal", "date", "metaparams", "same-shape_semantic_grp|task_kind", "shape_semantic_grp_1", "shape_semantic_grp_2"])
                # grouping_print_n_samples(dfdists, ["same-shape_semantic_grp|task_kind", "shape_semantic_grp_1", "shape_semantic_grp_2", "animal", "date"])
                # asds
                fig = grouping_plot_n_samples_conjunction_heatmap(dfdists, f"{var_effect}_1", f"{var_effect}_2", 
                                                            [var_same_same], annotate_heatmap=False,
                                                            FIGSIZE=5, n_columns=4)
                savefig(fig, f"{savedir}/allpairs_counts-{grp}.pdf")
                plt.close("all")

    plt.close("all")