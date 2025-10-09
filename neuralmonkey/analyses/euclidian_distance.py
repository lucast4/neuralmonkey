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

    RETURNS:

    """
    from pythonlib.tools.distfunctools import distmat_construct_wrapper
    from pythonlib.cluster.clustclass import Clusters
    from pythonlib.tools.pandastools import grouping_print_n_samples

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
    if len(PA.X)==0:
        return None, None
    
    # - Save counts after prune
    if plot_conjunctions_savedir is not None:
        fig = grouping_plot_n_samples_conjunction_heatmap_helper(PA.Xlabels["trials"], label_vars_for_cldist)
        savefig(fig, f"{plot_conjunctions_savedir}/timevarying_compute_fast_to_scalar-counts_heatmap-final.pdf")

        grouping_print_n_samples(PA.Xlabels["trials"], label_vars_for_cldist, savepath=f"{plot_conjunctions_savedir}/counts-{label_vars_for_cldist}.txt")

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

    # Sanity check that all distances are normalized to the same DIST_98 value.
    if len(dfdist["DIST_98"].unique())>1:
    # if dfdist["DIST_98"].max() - dfdist["DIST_98"].min()>0.001:
        print(dfdist["DIST_98"].unique())
        assert False, "fix this bug.."

    dfdist["data_dim"] = PA.X.shape[0]
    
    return dfdist, Cldist

def compute_angle_between_conditions(PA, dfdist, var_effect, vars_grp):
    """
    Get angles between each pair of conditions, one for each row of dfdist. 
    
    Note that these rows should be individual levels of [var_effect] + vars_group, but that
    is assumed to be true in this code.

    PA is the dataset used to get dfdist.

    See dfdist = timevarying_compute_fast_to_scalar(PA)

    In all cases is the angle from labels_1 to labels_2

    RETURNS:
    - dfangle, should be same length as dfdist.
    """
    from pythonlib.tools.vectools import cart_to_polar
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    assert PA.X.shape[2]==1, "only coded for scalar"
    assert PA.X.shape[0]==2, "assumes 2d space to get polar coordinates"

    ### (3) Get angles between all conditions
    dflab = PA.Xlabels["trials"]
    dflab = append_col_with_grp_index(dflab, vars_grp, "_var_other", use_strings=False)

    # Collect data for each row of dfdist
    res = []
    for _, row in dfdist.iterrows():
        if row["labels_1"] == row["labels_2"]:
            # By definition. 
            theta = 0.
            norm = 0.
        else:
            try:
                inds1 = dflab[(dflab[var_effect] == row["labels_1"][0]) & (dflab["_var_other"] == row["labels_1"][1:])].index.tolist()
                inds2 = dflab[(dflab[var_effect] == row["labels_2"][0]) & (dflab["_var_other"] == row["labels_2"][1:])].index.tolist()
                assert len(inds1)>0
                assert len(inds2)>0
            except Exception as err:
                print(dflab[var_effect])
                print(dflab["_var_other"])
                print(sum((dflab[var_effect] == row["labels_1"][0])))
                print(sum((dflab["_var_other"] == row["labels_1"][1])))
                print("---")
                print(dflab["_var_other"].unique())
                print(row["labels_1"][1:])
                print("---")
                raise err
            
            # Get the neural data
            x1 = PA.X[:, inds1].squeeze()
            x2 = PA.X[:, inds2].squeeze()

            x1_mean = np.mean(x1, axis=1)            
            x2_mean = np.mean(x2, axis=1)            

            vec = x2_mean - x1_mean
            assert len(vec)==2, "you need to input a 2d subspace"

            theta, norm = cart_to_polar(vec[0], vec[1])
            if np.isnan(theta):
                print(x1_mean, x2_mean, vec, inds1, inds2)
                assert False

        # Append
        res.append({
            "labels_1":row["labels_1"],
            "labels_2":row["labels_2"],
            "theta":theta,
            "norm":norm,
            # "vector":vec,
        })
    dfangle = pd.DataFrame(res)

    return dfangle

def compute_average_angle_between_pairs_of_levels_of_vareffect(dfdist, var_effect, min_levs_per_levother = 2, PRINT=False):
    """
    Get sum of vectors connecting levels of <var_effect>, within each level of vars_others. E.g, you have two vectors 
    connectings var_effect pairs (0,1) and (1,2) within some value of vars_others. Get the sum of those vectors (does two ways) 
    and store as a single mean angle for this level of vars_others. 

    THink of this as asking how aligned the vectors are.

    DEtermines the ordering of levels of <var_effect> based on sorted().

    Older notes:
    E.g., within each level <var_other>, get average of (vector between var_effect==0 and var_effect==1) and
    (vector between var_effect==1 and var_effect==2), if there are three levesl of var_effect: 0, 1, 2.

    RETURNS:
    - dfanglemean, one row for each level of vars_others. 
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from pythonlib.tools.vectools import get_vector_from_angle, average_vectors_wrapper

    assert len(dfdist["var_effect"].unique())==1
    assert len(dfdist["vars_others"].unique())==1
    dfdist_variables_append_vars_others(dfdist)

    # First, concat the reverese version, beucase this assumes var_effect counts up from labels_1 to labels_2.
    from math import pi
    dfdist_copy = dfdist.copy()
    dfdist_copy["theta"] = (dfdist_copy["theta"] + pi) % (2*pi)
    dfdist_copy["labels_1"] = dfdist["labels_2"]
    dfdist_copy["labels_2"] = dfdist["labels_1"]
    dfdist = pd.concat([dfdist, dfdist_copy]).reset_index(drop=True)    
    
    assert "theta" in dfdist.columns, "first run compute_angle_between_conditions"
    list_vars_others = sorted(set(dfdist["vars_others_1"].unique().tolist() + dfdist["vars_others_2"].unique().tolist()))

    res = []
    for var_other_grp in list_vars_others:

        DFTMP = dfdist[(dfdist["vars_others_same"] == True) & (dfdist["vars_others_1"] == var_other_grp)]
        
        if len(DFTMP)>0:
            levs_exist = sorted(DFTMP[f"{var_effect}_1"].unique())

            if len(levs_exist)>=min_levs_per_levother:
                if PRINT:
                    print("levs_exist:", levs_exist)
                assert len(levs_exist)>1

                # Get adjacent values of var_effect
                tmp = []
                for lev1, lev2 in zip(levs_exist[:-1], levs_exist[1:]):
                    this = DFTMP[(DFTMP[f"{var_effect}_1"] == lev1) & (DFTMP[f"{var_effect}_2"] == lev2)]
                    
                    if len(this)==2:
                        # This is possible if you've added the mirror data. Check this to allow pass
                        assert len(this["norm"].unique())==1 # rows are identical, just with flipped theta.
                        this = this.iloc[:1, :] # Take the top row
                    
                    if len(this)!=1:
                        print(len(this))
                        print(this)
                        print("-----------------")
                        print(DFTMP[f"{var_effect}_1"] == lev1)
                        print(DFTMP[f"{var_effect}_2"] == lev1)
                        print(DFTMP[f"{var_effect}_1"] == lev2)
                        print(DFTMP[f"{var_effect}_2"] == lev2)
                        print(lev1, lev2)
                        assert False, "this should not be possible"

                    tmp.append(this)
                dftmp = pd.concat(tmp) # should have just this in order
                
                # Compute average vector, different possible methods
                for var_vector_length in ["dist_norm", "dist_yue_diff"]:
                    angles = dftmp["theta"].values
                    weights = dftmp[var_vector_length].values
                    vectors_arr = np.stack([w * get_vector_from_angle(a) for a, w in zip(angles, weights)])
                    
                    for length_method in ["sum", "dot"]:
                        # More general
                        angle_mean, norm_mean = average_vectors_wrapper(vectors_arr, length_method=length_method)

                        res.append({
                            "var_other":var_other_grp,
                            "levs_exist":levs_exist,
                            "angles":angles,
                            "weights":weights,
                            "angle_mean":angle_mean,
                            "norm_mean":norm_mean,
                            "var_vector_length":var_vector_length,
                            "length_method":length_method
                        })
    dfanglemean = pd.DataFrame(res)
    
    return dfanglemean

def dfdist_postprocess_condition_prune_to_var_pairs_exist(dfdist, var_effect, var_context,
                                                          plot_counts_savedir=None, lenient_allow_data_if_has_n_levels=2):
    """
    Keep only var_effect levels that exist across <lenient_allow_data_if_has_n_levels> levels of context.

    if want to force to be across ALL contexts, then use lenient_allow_data_if_has_n_levels=="all"
    NOTE: beucase this is ALL, you can run into paradoxical cases where you actually throw out data
    beucase there are additional levels of var_context.
    
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
    if True:
        plot_counts_heatmap_savepath = None # redundant with the final plot below
        if lenient_allow_data_if_has_n_levels=="all":
            n_levs_context = len(dftmp[var_context].unique())
        else:
            n_levs_context = lenient_allow_data_if_has_n_levels
        dfout, _ = extract_with_levels_of_conjunction_vars_helper(dftmp, var_context, 
                                                    [var_effect], 1, plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                                    lenient_allow_data_if_has_n_levels=n_levs_context)
    else:    
        from pythonlib.tools.pandastools import conjunction_vars_prune_to_balance
        dfout, _ = conjunction_vars_prune_to_balance(dftmp, var_context, var_effect, prefer_to_drop_which=None)

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

def dfdist_extract_label_vars_specific_single(dfdists, label_vars, var1=None):
    """
    Helper to extract new columns to put in dfdists, expanding out the individual variables in label_vars.
    
    Is similar to dfdist_extract_label_vars_specific, but here is for anlayses not comparing pairs of trials, but
    instead considering each trial-group on its own (e.g., decoding).

    """
    # from pythonlib.tools.pandastools import append_col_with_grp_index
    
    if var1 is None:
        if "labels_1_datapt" in dfdists.columns:
            var1 = "labels_1_datapt"
        else:
            var1 = "labels_1"
    
    dfdists = dfdists.copy()

    assert isinstance(dfdists[var1].values[0], tuple)
    # Replace columns which are now incorrect
    # label_vars = ["seqc_0_shape", "seqc_0_loc"]
    assert isinstance(label_vars[0], str)

    # e..g, seqc_0_shape_1
    for i, var in enumerate(label_vars):
        dfdists[f"{var}"] = [x[i] for x in dfdists[var1]]

    return dfdists

def dfdist_extract_label_vars_specific(dfdists, label_vars, return_var_same=False,
                                       var1=None, var2=None):
    """
    Automatically populates new columns reflecting the relations between the columns in 
    label_vars (which can be any length), such as same_shape

    Uses whatever is in labels_1 and labels_2
    
    PARAMS:
    - dfdists, output from things like rsa_distmat_score_all_pairs_of_label_groups
    - label_vars, list of n strings
    
    e.g., label_vars = [shape, loc], means that labels_1 is a column with items like (circle, (0,1)), and
    will populate new columns called shape_1, shape_2, loc_1, loc_2, etc....

    NOTE: any variables that are not in label_vars will be INCORRECT since they don't flip correctly.
    RETURNS:
    - copy of dfdists
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index

    assert (var1 is None) == (var2 is None)
    
    if var1 is None:
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

    # Append a conjunctive column
    colname_conj_same = dfdist_variables_generate_var_same(label_vars)
    # colname_conj_same = "same-"
    # for v in label_vars:
    #     colname_conj_same+=f"{v}|"
    # colname_conj_same = colname_conj_same[:-1] # remove the last |
    dfdists = append_col_with_grp_index(dfdists, [f"{v}_same" for v in label_vars], colname_conj_same)
    # if len(label_vars)==2:
    #     dfdists = append_col_with_grp_index(dfdists, [f"{label_vars[0]}_same", f"{label_vars[1]}_same"], f"same-{label_vars[0]}|{label_vars[1]}")
    # if len(label_vars)==3:
    #     dfdists = append_col_with_grp_index(dfdists, [f"{label_vars[0]}_same", f"{label_vars[1]}_same", f"{label_vars[2]}_same"], f"same-{label_vars[0]}|{label_vars[1]}|{label_vars[2]}")

    if return_var_same:
        return dfdists, colname_conj_same
    else:
        return dfdists

def dfdist_convert_merge_pair_to_get_all_levels(dfdist, list_vars, vars_others):
    """
    Given dfdist that has paired data, each row pairing (labels_1, labels_2),
    return a dfdist that is twice as long, holding all the labels (ignoring 
    whether they were on 1 or 2). Useful to get all the unique levels, esp when 
    labels_1 -- labels_2 are assymetric.

    Eg.
    if columns are (shape_1, gridloc_1, shape_2, gridloc_2) return a new dataframe 
    with columns (shape, gridloc).

    PARAMS:
    - list_vars, the variables to merge. list of str, where each str has 
    <str>_1 and <str>_2 columns.
    - vars_others, other variables to take. each should exist in dfdist 
    (no 1 and 2 extension)
    """

    list_df = []
    for i in [1, 2]:
        rename_mapper = {f"{v}_{i}":v for v in list_vars}
        df = dfdist.loc[:, vars_others + [f"{v}_{i}" for v in list_vars]]
        df = df.rename(rename_mapper, axis=1)
        list_df.append(df)

    dfmerged = pd.concat(list_df, axis=0).reset_index(drop=True)
    return dfmerged

def dfdist_variables_append_vars_others(dfdist):
    """
    Helper to append columns: "vars_others_1", "vars_others_2", "vars_others_same".
    These are the levels (tuple) of the cunjucntive var_other

    RETURNS:
    - (Nothing) Modifies dfdist.
    """
    assert isinstance(dfdist["labels_1"].values[0], tuple)
    assert isinstance(dfdist["labels_2"].values[0], tuple)

    dfdist["vars_others_1"] = [x[1:] for x in dfdist["labels_1"]]
    dfdist["vars_others_2"] = [x[1:] for x in dfdist["labels_2"]]
    dfdist["vars_others_same"] = dfdist["vars_others_1"] == dfdist["vars_others_2"]

def dfdist_variables_generate_var_same(label_vars):
    """
    Genreate the name of the column that holds contrast strings for these label vars.
    """
    colname_conj_same = "same-"
    for v in label_vars:
        colname_conj_same+=f"{v}|"
    colname_conj_same = colname_conj_same[:-1] # remove the last |
    return colname_conj_same

def dfdist_expand_convert_from_triangular_to_full(dfdists, label_vars=None, PLOT=False,
                                                repopulate_relations=True,
                                                var1=None, var2=None, remove_diagonal=True):
    """
    Given a dfdists that is triangular (inclues diagonmal usually), convert to 
    full matrix by copying and swapping labels 1 and 2, assuming that
    distances are symmetric.
    
    RETURNS:
    - copy of dfdists, but more rows.
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

    if var1 is None:
        if "labels_1_datapt" in dfdists.columns:
            var1 = "labels_1_datapt"
            var2 = "labels_2_grp"
        else:
            var1 = "labels_1"
            var2 = "labels_2"

    if PLOT:
        grouping_plot_n_samples_conjunction_heatmap(dfdists, var1, var2);

    dftmp = dfdists.copy()

    # Flip labels
    dftmp[var1] = dfdists[var2]
    dftmp[var2] = dfdists[var1]
    
    # Remove diagonal
    if remove_diagonal:
        dftmp = dftmp[dftmp[var1]!=dftmp[var2]]
    
    # concat
    dfdists = pd.concat([dfdists, dftmp]).reset_index(drop=True)

    if repopulate_relations:
        # Repopulation all label columns
        assert label_vars is not None        
        # label_vars = ["seqc_0_shape", var_other]
        # from pythonlib.cluster.clustclass import Clusters
        # cl = Clusters(None)
        dfdists = dfdist_extract_label_vars_specific(dfdists, label_vars, var1=var1, var2=var2)
        # dfdists = self.rsa_distmat_population_columns_label_relations(dfdists, label_vars)

    if PLOT:
        grouping_plot_n_samples_conjunction_heatmap(dfdists, var1, var2);

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



def dfdist_variables_expand_indices_called_either(this_tuple):
    """
    Given a binary tuple (e.g., (0, 1, 0, -1)) where slots represent variables, and 
    0 and 1 represent different/same, and -1 represents etiher, return list of all
    the possible tuples after "expanding" the -1 to 0 or 1.
    
    Returns list of tuples. where items with -1 are expanded to return two tuples
    with that item replaced with 0 and 1.

    For example:
    expand_indices_called_either([0, -1, -1, -1])
    --> 
        [[0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1]]
    """

    def _expand_indices_called_either(this_tuple):
        """
        Helper to expand this specific tuple.
        Returns list of lists of ints.
        """
        if -1 in this_tuple:
            ind_either = this_tuple.index(-1)    
            
            this_tuple_copy_1 = [x for x in this_tuple]
            this_tuple_copy_2 = [x for x in this_tuple]

            this_tuple_copy_1[ind_either] = 0
            this_tuple_copy_2[ind_either] = 1
            return [this_tuple_copy_1, this_tuple_copy_2], True
        else:
            return [this_tuple], False
        
    all_tuples = []

    tmp, success = _expand_indices_called_either(this_tuple)
    if not success:
        all_tuples.extend(tmp)
        return tmp
    else:
        # Try again
        for _this_tuple in tmp:
            tmp = dfdist_variables_expand_indices_called_either(_this_tuple)
            all_tuples.extend(tmp)
    return all_tuples

def dfdist_variables_generate_constrast_strings(vars_in_order, contrasts_diff, contrasts_either, contrasts_same=None):
    """
    Generate list of contrast strings that satisfie these criteria of diffs, where a contrast string
    is something like "0|1|1|0", each slot a vraible, and 0 and 1 are diff/same.

    PARAMS:
    - vars_in_order, list of variables, ie in order of contrasts.
    - contrasts_diff, list of str, variables that should be differnet.
    - contrasts_either, list of str, varialbes that can be etiher diff or same. Will return all strings after
    "expanding" these variables to be 0 and 1 (in different returned stringes)
    
    RETURNS:
    - list of contrast strings.
    """
    assert contrasts_same is None, "not coded yet"

    if contrasts_same is None:
        contrasts_same = [v for v in vars_in_order if v not in contrasts_diff + contrasts_either]

    for v in contrasts_diff:
        assert v in vars_in_order, f"sanity check no typo - {v} -- {vars_in_order}"

    # For "either" is ok if it is not in the variables.
    contrasts_either = [v for v in contrasts_either if v in vars_in_order]
    # for v in contrasts_either:
    #     assert v in vars_in_order, f"sanity check no typo - {v} -- {vars_in_order}"
    for v in contrasts_same:
        assert v in vars_in_order, f"sanity check no typo - {v} -- {vars_in_order}"
    this_tuple = []
    for v in vars_in_order:
        if v in contrasts_diff:
            this_tuple.append(0)
        elif v in contrasts_either:
            this_tuple.append(-1)
        elif v in contrasts_same:
            this_tuple.append(1)
        else:
            assert False

    # this_tuple = [-1, 1, 0, -1, -1, -1]
    # print(this_tuple)
    list_tuples = dfdist_variables_expand_indices_called_either(this_tuple)

    # Finally, convert all tuples to strings
    contrast_strings = []
    for tup in list_tuples:
        contrast_strings.append("|".join([str(x) for x in tup]))

    return contrast_strings

def dfdist_variables_effect_extract_helper(DFDIST, colname_conj_same, vars_in_order, contrasts_diff, 
                                           contrasts_either, PRINT=False):
    """
    [Useful] Get slice of dfdist holding the desired effects, defined by their contrast values (e.g, 1|0|1, ..), 
    and other parameters

    This helps you extract all contrast values satistfying a set of varialbes that must be same/diff/either.

    E.g., you want to only get rows that have same shape and different location.
    
    PARAMS:
    - colname_conj_same, the column name (e.g., "same...") that holds the contrast string (e.g., "1|0|1").
    - vars_in_order, list of variables, ie in order of contrasts.
    - contrasts_diff, list of str, variables that should be differnet.
    - contrasts_either, list of str, varialbes that can be etiher diff or same. Will return all strings after
    "expanding" these variables to be 0 and 1 (in different returned stringes)

    RETURNS:
    - Slice of DFDIST.
    """
    assert len(DFDIST)>0
    # Finally, get just the desired contrasts
    contrast_strings = dfdist_variables_generate_constrast_strings(vars_in_order, contrasts_diff, contrasts_either)
    if PRINT:
        print("Getting these contrast_strings: ", contrast_strings)
        print("Existing contrast strings: ", DFDIST[colname_conj_same].unique())
    dfdist = DFDIST[(DFDIST[colname_conj_same].isin(contrast_strings))].reset_index(drop=True)

    # also give a generic column name with contrast strings
    dfdist["contrast_vars"] = [tuple(vars_in_order) for _ in range(len(dfdist))]
    dfdist["contrast_string"] = dfdist[colname_conj_same]

    return dfdist

def dfdist_compute_regions_diff(dfdist, vars_datapt, var_value, bregion_2, do_plot=False):
    """
    Helper to compare all other areas to one area (bregion_2), getting a difference (bregion 2 minus 1)
    for each level of vars_datapt. 

    Then returns the same data in both long and wide (one col for each bregion) format.

    Get values <bregion_1> minus each of the other brain regions.

    PARAMS:
    - vars_datapt, list of str, each level will be one datapt (which will be one row, with one value(column) for eahc region)
    - var_value, str, the scalar value
    - bregion_2, the single region who will be in bregion 2 minus 1.
    RETURNS:
    - dfdifference_long, dfdifference_wide

    """
    from pythonlib.tools.pandastools import pivot_table, aggregGeneral, convert_wide_to_long

    # Agg, to get one datapt per var_datapt
    DFEFFECT_AGG = aggregGeneral(dfdist, vars_datapt + ["bregion"], [var_value])
    
    # Expand all the bregions
    DFEFFECT_AGG_WIDE = pivot_table(DFEFFECT_AGG, vars_datapt, ["bregion"], [var_value])
    
    # For each <other bregion> get its value subtracted from bregion_1
    list_bregion = DFEFFECT_AGG["bregion"].unique().tolist()
    res = {}
    bregions_other = []
    for bregion_1 in list_bregion:
        if bregion_1!=bregion_2:
            vals_diff = DFEFFECT_AGG_WIDE[var_value][bregion_2] - DFEFFECT_AGG_WIDE[var_value][bregion_1]
            res[bregion_1] = vals_diff
            bregions_other.append(bregion_1)
    for col in vars_datapt:
        res[col] = DFEFFECT_AGG_WIDE[col]
        # res["date"] = DFEFFECT_AGG_WIDE["date"]
        # res["effect"] = DFEFFECT_AGG_WIDE["effect"]
    dfdifference_wide = pd.DataFrame(res)
    dfdifference_long = convert_wide_to_long(dfdifference_wide, bregions_other, vars_datapt, "bregion_1", f"{var_value}")

    if do_plot:
        import seaborn as sns
        fig = sns.catplot(data=dfdifference_long, y="bregion_1", x=var_value, col="effect", jitter=True, alpha=0.5)
        for ax in fig.axes.flatten():
            ax.axvline(0, color="k", alpha=0.5)

        fig = sns.catplot(data=dfdifference_long, y="bregion_1", x=var_value, col="effect", kind="boxen")
        for ax in fig.axes.flatten():
            ax.axvline(0, color="k", alpha=0.5)

        # Also plot histograms
        ncols = 6
        size=4
        aspect=1.5
        list_effects = dfdifference_wide["effect"].unique().tolist()
        nrows = int(np.ceil(len(list_effects)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(size*ncols*aspect, size*nrows), sharex=True, sharey=True)
        list_bregions = dfdist["bregion"].unique().tolist()
        for ax, effect in zip(axes.flatten(), list_effects):
            df = dfdifference_wide[dfdifference_wide["effect"]==effect].reset_index(drop=True)

            for br in list_bregions:
                if br!=bregion_2:
                    # col = map_bregion_to_color[br]
                    sns.histplot(df, x=br, ax=ax, element="step", alpha=0.2)
            ax.axvline(0, color="k", alpha=0.5)
            ax.set_xlabel(var_value)
            ax.set_title(effect)
            
    return dfdifference_long, dfdifference_wide


def dfdist_compute_effects_diff_wideform(dfdist, var_effect, eff1, eff2, vars_grp, diff_func="minus"):
    """
    Return df where each row is a level of vars_grp, and there are two columns, one for each of the two effects.  
    Also return df where you subtract those two effects 

    PARAMS:
    - diff_func, "minus", "div", .. 
    """
    ###### GOOD
    from pythonlib.tools.pandastools import summarize_featurediff
    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    
    yvar = "dist_yue_diff"

    # (2) For each (date, bregion) get ratio of eff1 to eff2
    FEATURE_NAMES = [yvar]
    dfsummary, _, _, _, COLNAMES_DIFF, dfpivot = summarize_featurediff(
        dfdist, var_effect, [eff1, eff2], FEATURE_NAMES, vars_grp, 
        return_dfpivot=True, diff_func=diff_func, diff_col_name_include_feature=False)

    new_cols = []
    for col in dfpivot.columns.values:
        if col[1]=="":
            new_cols.append(col[0])
        else:
            new_cols.append("-".join(col))
    dfpivot.columns = new_cols
    # effect_div_name = f"{yvar}-{COLNAMES_DIFF[0]}"
    effect_div_name = f"{COLNAMES_DIFF[0]}"
    if effect_div_name not in dfsummary:
        print(effect_div_name)
        print(dfpivot)
        print(dfsummary)
        assert False

    # display(dfsummary)
    # display(dfpivot)
    # print(effect_div_name)

    eff1_name = f"{yvar}-{eff1}"
    eff2_name = f"{yvar}-{eff2}"

    ### Merge
    return dfsummary, dfpivot, effect_div_name, eff1_name, eff2_name
