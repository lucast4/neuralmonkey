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
    Cldist = Clusters(dmat, list_lab, list_lab, ver="dist", params=params)

    if rsa_heatmap_savedir is not None:
        # This fn also returns dfdist. The reason I dont keep ti is that here I get both directions. I am not sure
        # if this would do wierd things downstream
        zlims = None
        # zlims = [0, 0.3]
        _, CldistAgg = Cldist.rsa_distmat_score_all_pairs_of_label_groups(label_vars=label_vars, get_only_one_direction=False, 
                                                                          return_as_clustclass=True,
                                                                          context_dict=context_dict)
        n = min([len(label_vars), 3])
        from itertools import permutations
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
        # for i in range(len(label_vars)):
        #     for j in range(len(label_vars)):
        #         if j>i:
        #             var1 = label_vars[0]
        #             var2 = label_vars[1]
        #             dfdist = append_col_with_grp_index(dfdist, [f"{var1}_same", f"{var2}_same"], f"same-{var1}|{var2}")

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


