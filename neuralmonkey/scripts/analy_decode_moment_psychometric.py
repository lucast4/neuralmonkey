

import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys
from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_good_morphsets_no_switching
from pythonlib.tools.vectools import projection_onto_axis_subspace
import seaborn as sns
import pickle

from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single


def _analy_switching_statespace_euclidian_traj_computediff(df):
    # savedir = f"{SAVEDIR}/plots"
    # os.makedirs(savedir, exist_ok=True)
    ################ Convert to a single number - separation between base1 and base2
    # reshape
    # -----
    from pythonlib.tools.pandastools import convert_to_2d_dataframe

    # reshape
    # list_cat_1 = ["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"]
    dfthis, _, _, _ = convert_to_2d_dataframe(df, "assigned_base", "time_bin", False, "mean", 
                                                "dist_index")

    dist_index_diff_base = dfthis.loc["base2", :] - dfthis.loc["base1", :]

    if ("not_ambig_base1" in dfthis.index.tolist()) and ("not_ambig_base2" in dfthis.index.tolist()):
        dist_index_diff_notambig = dfthis.loc["not_ambig_base2", :] - dfthis.loc["not_ambig_base1", :]
    else:
        dist_index_diff_notambig = None

    if ("ambig_base1" in dfthis.index.tolist()) and ("ambig_base2" in dfthis.index.tolist()):
        dist_index_diff_ambig = dfthis.loc["ambig_base2", :] - dfthis.loc["ambig_base1", :]
    else:
        dist_index_diff_ambig = None

    # dfsummary = pd.concat([dist_index_diff_base, dist_index_diff_notambig, dist_index_diff_ambig], axis=1)
    # dfsummary = dfsummary.reset_index(names=["time_bin"])

    times = dist_index_diff_base.index.values
    if not np.all(np.diff(times)>0):
        print(times)
        print(df)
        assert False, "why"
    return times, dist_index_diff_base, dist_index_diff_notambig, dist_index_diff_ambig

def _analy_switching_statespace_euclidian_traj_plots(DFPROJ_INDEX, DFDIST, savedir):
    # savedir = f"{SAVEDIR}/plots"
    # os.makedirs(savedir, exist_ok=True)
    ################ Convert to a single number - separation between base1 and base2
    # reshape
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    for norm_sub, zlims in [
        (None, None),
        # (None, [0, 1]),
        ("all_sub", None)
        ]:
        list_cat_1 = sorted(DFPROJ_INDEX["idxmorph_assigned"].unique())
        dfthis, fig, _, _ = convert_to_2d_dataframe(DFPROJ_INDEX, "idxmorph_assigned", "time_bin", True, 
                                                            "mean", "dist_index", annotate_heatmap=False, norm_method=norm_sub,
                                                            list_cat_1=list_cat_1, zlims=zlims);
        savefig(fig, f"{savedir}/2D-idxmorph_assigned-norm={norm_sub}-zlims={zlims}.pdf")

        # reshape
        list_cat_1 = ["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"]
        dfthis, fig, _, _ = convert_to_2d_dataframe(DFPROJ_INDEX, "assigned_base", "time_bin", True, 
                                                            "mean", "dist_index", annotate_heatmap=False, norm_method=norm_sub,
                                                            list_cat_1=list_cat_1, zlims=zlims);
        savefig(fig, f"{savedir}/2D-assigned_base-norm={norm_sub}-zlims={zlims}.pdf")

    # -----
    # A single score (difference of dist index)
    times, dist_index_diff_base, dist_index_diff_notambig, dist_index_diff_ambig = _analy_switching_statespace_euclidian_traj_computediff(DFPROJ_INDEX)
    # dist_index_diff_base = dfthis.loc["base2", :] - dfthis.loc["base1", :]

    # if ("not_ambig_base1" in dfthis.index.tolist()) and ("not_ambig_base2" in dfthis.index.tolist()):
    #     dist_index_diff_notambig = dfthis.loc["not_ambig_base2", :] - dfthis.loc["not_ambig_base1", :]
    # else:
    #     dist_index_diff_notambig = None

    # if ("ambig_base1" in dfthis.index.tolist()) and ("ambig_base2" in dfthis.index.tolist()):
    #     dist_index_diff_ambig = dfthis.loc["ambig_base2", :] - dfthis.loc["ambig_base1", :]
    # else:
    #     dist_index_diff_ambig = None

    # dfsummary = pd.concat([dist_index_diff_base, dist_index_diff_notambig, dist_index_diff_ambig], axis=1)
    # dfsummary = dfsummary.reset_index(names=["time_bin"])

    ###################### PLOTS
    import seaborn as sns

    # (1) Plots using euclidian dist (norm)
    # Clean plots
    df_base12_only = DFDIST[DFDIST["idxmorph_assigned_2"].isin(["0|base1", "99|base2"])].reset_index(drop=True)

    for dfthis, suff in [
        # (df_base12_only, "BASEONLY"),
        (DFDIST, "ALL"),
        ]:
        
        col_order = sorted(dfthis["idxmorph_assigned_1"].unique())
        fig = sns.relplot(data=dfthis, x="time_bin", y="dist_norm", col="idxmorph_assigned_1", col_wrap=5, hue="idxmorph_assigned_2", kind="line", 
                            errorbar=("ci", 68), col_order=col_order)
        savefig(fig, f"{savedir}/DFDIST-dist_norm-idxmorph_assigned_1-{suff}.pdf")

        hue_order = sorted(dfthis["idxmorph_assigned_1"].unique())
        fig = sns.relplot(data=dfthis, x="time_bin", y="dist_norm", col="idxmorph_assigned_2", col_wrap=5, hue="idxmorph_assigned_1", kind="line", 
                            errorbar=("ci", 68), hue_order=hue_order)
        savefig(fig, f"{savedir}/DFDIST-dist_norm-idxmorph_assigned_2-1-{suff}.pdf")

        # hue_order = sorted(dfthis["assigned_base"].unique())
        hue_order = ["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"]
        fig = sns.relplot(data=dfthis, x="time_bin", y="dist_norm", col="idxmorph_assigned_2", col_wrap=5, hue="assigned_base", 
                        kind="line", errorbar=("ci", 68), hue_order=hue_order)
        savefig(fig, f"{savedir}/DFDIST-dist_norm-idxmorph_assigned_2-2-{suff}.pdf")
    plt.close("all")
    
    # (2) Plots using dist index
    hue_order = sorted(DFPROJ_INDEX["idxmorph_assigned"].unique())
    fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", hue="idxmorph_assigned", kind="line", errorbar=("ci", 68),
                    hue_order=hue_order)
    savefig(fig, f"{savedir}/DFPROJ_INDEX-dist_index-idxmorph_assigned.pdf")

    if False:
        hue_order = ["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "not_ambig_base2", "base2"]
        fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", hue="assigned_base", kind="line", errorbar=("ci", 68),
                        hue_order=hue_order)
        savefig(fig, f"{savedir}/DFPROJ_INDEX-dist_index-assigned_base.pdf")

    hue_order = sorted(DFPROJ_INDEX["assigned_base_simple"].unique())
    col_order = sorted(DFPROJ_INDEX["idx_morph_temp"].unique())
    fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", hue="assigned_base_simple", 
                    col="idx_morph_temp", kind="line", errorbar=("ci", 68), hue_order=hue_order, col_order=col_order)
    savefig(fig, f"{savedir}/DFPROJ_INDEX-dist_index-assigned_base_simple-1.pdf")

    if False:
        hue_order = sorted(DFPROJ_INDEX["assigned_base_simple"].unique())
        fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", hue="assigned_base_simple", col="assigned_label", kind="line", errorbar=("ci", 68),
                        hue_order=hue_order)
        savefig(fig, f"{savedir}/DFPROJ_INDEX-dist_index-assigned_base_simple-2.pdf")

    plt.close("all")

    # (3) Dist index, single trial
    col = "idxmorph_assigned"
    col_order = sorted(DFPROJ_INDEX[col].unique())
    fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", hue="idx_row_datapt", col=col, col_wrap=5, kind="line",
                col_order=col_order)
    savefig(fig, f"{savedir}/DFPROJ_INDEX-TRIALS-idx_row_datapt.pdf")


    # (4) Plot dist index, difference bteween base2 - base1
    # from pythonlib.tools.plottools import makeColors
    # pcols = makeColors(3)

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    for ax, ylim in zip(axes.flatten(), [None, (-0.1, 0.6)]):
        ax.set_xlabel("time (sec)")
        ax.set_ylabel("dist index (base2-base1)")
        ax.axhline(0, color="k", alpha=0.3)
        ax.axvline(0, color="k", alpha=0.3)
        
        for i, (dat, suff) in enumerate([
            (dist_index_diff_base, "base"),
            (dist_index_diff_notambig, "not_ambig"),
            (dist_index_diff_ambig, "ambig"),
            ]):

            if dat is not None:
                times = dat.index.values
                scores = dat.values

                ax.plot(times, scores, label=suff)
        ax.legend()
        ax.set_ylim(ylim)
        ax.set_title(f"ylim={ylim}")
    savefig(fig, f"{savedir}/DFPROJ_INDEX_DIFFERENCE.pdf")

    plt.close("all")

def _analy_switching_statespace_euclidian_traj_condition(PAredu, DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG=None, DFDIST_AGG=None):
    """
    Condition the data dfs, modifiying them in place
    """
    dflab = PAredu.Xlabels["trials"]

    #  Map from idx|assign to label
    map_idxassign_to_label = {}
    map_idxassign_to_assignedbase = {}
    map_idxassign_to_assignedbase_simple = {}
    map_idxassign_to_idx_morph = {}
    for i, row in dflab.iterrows():
        if row["idxmorph_assigned"] not in map_idxassign_to_label:
            map_idxassign_to_label[row["idxmorph_assigned"]] = row["assigned_label"]
            map_idxassign_to_assignedbase[row["idxmorph_assigned"]] = row["assigned_base"]
            map_idxassign_to_assignedbase_simple[row["idxmorph_assigned"]] = row["assigned_base_simple"]
            map_idxassign_to_idx_morph[row["idxmorph_assigned"]] = row["idx_morph_temp"]
        else:
            assert map_idxassign_to_label[row["idxmorph_assigned"]] == row["assigned_label"]
            assert map_idxassign_to_assignedbase[row["idxmorph_assigned"]] == row["assigned_base"]
            assert map_idxassign_to_assignedbase_simple[row["idxmorph_assigned"]] == row["assigned_base_simple"]
            assert map_idxassign_to_idx_morph[row["idxmorph_assigned"]] == row["idx_morph_temp"]

    for df in [DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG]:
        if df is not None:
            df["assigned_base_simple"] = [map_idxassign_to_assignedbase_simple[x] for x in df["idxmorph_assigned"]]
            df["assigned_base"] = [map_idxassign_to_assignedbase[x] for x in df["idxmorph_assigned"]]
            df["assigned_label"] = [map_idxassign_to_label[x] for x in df["idxmorph_assigned"]]
            df["idx_morph_temp"] = [map_idxassign_to_idx_morph[x] for x in df["idxmorph_assigned"]]


def analy_switching_statespace_euclidian_traj(PAredu, savedir, make_plots=False, save_df=True):
    """
    Good plots related to switching (ambigious trials), using euclidian distance, and making state space plots.

    Two related updates
    (1) Trajectories (of euclidian dist, indices etc)
    (2) Single trials.
    """
    import pickle

    ########################################
    ### Get single trial pairwise distances over time.
    var_effect = "idxmorph_assigned"
    effect_lev_base1 = "0|base1"
    effect_lev_base2 = "99|base2"
    list_grps_get = [
        ("0|base1",),  
        ("99|base2",)
        ] # This is important, or else will fail if there are any (idx|assign) with only one datapt.

    # var_effect = "idx_morph_temp"
    # effect_lev_base1 = 0
    # effect_lev_base2 = 99
    # list_grps_get = [(0,), (99,)] # This is important, or else will fail if there are any (idx|assign) with only one datapt.

    from neuralmonkey.scripts.analy_decode_moment_psychometric import _compute_df_using_dist_index_traj
    DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS = _compute_df_using_dist_index_traj(PAredu, var_effect, 
                                                                                           effect_lev_base1, effect_lev_base2,
                                                                                           list_grps_get=list_grps_get)

    #################### ADD LABELS
    _analy_switching_statespace_euclidian_traj_condition(PAredu, DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG)

    ###### PLOTS
    if make_plots:
        _analy_switching_statespace_euclidian_traj_plots(DFPROJ_INDEX, DFDIST, savedir)

    if save_df:
        # Save the reuslts
        pd.to_pickle(DFPROJ_INDEX, f"{savedir}/DFPROJ_INDEX.pkl")
        pd.to_pickle(DFDIST, f"{savedir}/DFDIST.pkl")
        pd.to_pickle(DFPROJ_INDEX_AGG, f"{savedir}/DFPROJ_INDEX_AGG.pkl")
        pd.to_pickle(DFDIST_AGG, f"{savedir}/DFDIST_AGG.pkl")

    return DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG

def analy_switching_statespace_euclidian_good(PAredu, SAVEDIR, PLOT_CLEAN_VERSION=False,
                                              skip_eucl_dist=False):
    """
    Good plots related to switching (ambigious trials), using euclidian distance, and making state space plots.
    """
    import pickle

    ##############################
    ### State space plots
    savedir = f"{SAVEDIR}/state_space"
    os.makedirs(savedir, exist_ok=True)

    LIST_VAR = [
        # "idxmorph_assigned",
        "idxmorph_assigned",
        "idx_morph_temp",
        "assigned_base_simple",
        "assigned_base_simple",
    ]
    LIST_VARS_OTHERS = [
        # ("task_kind",),
        ("assigned_label",),
        ("assigned_base_simple",),
        ("idx_morph_temp",),
        ("assigned_label",),
    ]
    LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]
    nmin_trials_per_lev = 1
    list_dim_timecourse = list(range(6))
    list_dims = [(0,1), (1,2), (2,3), (3,4)]
    PAredu.plot_state_space_good_wrapper(savedir, LIST_VAR, LIST_VARS_OTHERS, 
                                        LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS, nmin_trials_per_lev=nmin_trials_per_lev,
                                        list_dim_timecourse=list_dim_timecourse, PLOT_CLEAN_VERSION=PLOT_CLEAN_VERSION,
                                        list_dims=list_dims)

    if not skip_eucl_dist:
        ##############################
        ### Eucl dist (distribution-wise distance from base1 and base2)
        savedir = f"{SAVEDIR}/dist_index"
        os.makedirs(savedir, exist_ok=True)

        # - Eucl dist (score as "index", showing separation within each trial) [one pt per condition]
        var_effect = "idxmorph_assigned"
        effect_lev_base1="0|base1"
        effect_lev_base2="99|base2"
        dfproj_index, _ = _compute_df_using_dist_index(PAredu, var_effect, effect_lev_base1, effect_lev_base2)
        fig = sns.catplot(data=dfproj_index, x=var_effect, y="dist_index", aspect=2, kind="point")
        savefig(fig, f"{savedir}/dfproj_index.pdf")

        ##############################
        ### One pt per trial (distnace, on axis between base1 and base2)
        savedir = f"{SAVEDIR}/proj_on_axis_btw_bases"
        os.makedirs(savedir, exist_ok=True)
        if PAredu.X.shape[2]==1:
            from pythonlib.tools.vectools import projection_onto_axis_subspace
            dflab = PAredu.Xlabels["trials"]
            X = PAredu.X.squeeze().T # (trials, dims)

            # Do this in cross-validated way -- split base prims into two sets, run this 2x, then 
            # average over base prims.
            from pythonlib.tools.statstools import crossval_folds_indices

            inds_pool_base1 = dflab[dflab["idx_morph_temp"] == 0].index.values
            inds_pool_base2 = dflab[dflab["idx_morph_temp"] == 99].index.values
            list_inds_1, list_inds_2 = crossval_folds_indices(len(inds_pool_base1), len(inds_pool_base2), 2)

            # shuffle inds
            np.random.shuffle(inds_pool_base1)
            np.random.shuffle(inds_pool_base2)

            list_inds_1 = [inds_pool_base1[_inds] for _inds in list_inds_1]
            list_inds_2 = [inds_pool_base2[_inds] for _inds in list_inds_2]

            list_X =[]
            for _i, (inds_base_1, inds_base_2) in enumerate(zip(list_inds_1, list_inds_2)):
                # - get mean activity for base1, base2
                xmean_base1 = np.mean(X[inds_base_1,:], axis=0) # (ndims,)
                xmean_base2 = np.mean(X[inds_base_2,:], axis=0) # (ndims,)

                # get mean projected data for each state
                # xproj, fig = projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=True)
                plot_color_labels = dflab["idxmorph_assigned"].values
                xproj, fig = projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=True, 
                                                        plot_color_labels=plot_color_labels)
                savefig(fig, f"{savedir}/projections_preprocess-iter={_i}.pdf")
                # replace the train data with nan
                xproj[inds_base_1] = np.nan
                xproj[inds_base_2] = np.nan

                list_X.append(xproj)

            # Get average over iterations
            Xproj = np.nanmean(np.stack(list_X), axis=0)

            dfproj = PAredu.Xlabels["trials"].copy()
            dfproj["x_proj"] = Xproj
            order = sorted(dfproj["idxmorph_assigned"].unique())
            fig = sns.catplot(data=dfproj, x="idxmorph_assigned", y="x_proj", aspect=2, hue="assigned_base_simple",
                        order = order)
            savefig(fig, f"{savedir}/x_proj-1.pdf")

            order = sorted(dfproj["idx_morph_temp"].unique())
            fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", aspect=2, hue="assigned_base_simple",
                        order = order)    
            savefig(fig, f"{savedir}/x_proj-1.pdf")

            plt.close("all")
        else:
            savedir = f"{SAVEDIR}/proj_on_axis_btw_bases_SKIPPED_BECUASE_THIS_IS_TRAJ"
            os.makedirs(savedir, exist_ok=True)

        ########################## Save PA
        with open(f"{SAVEDIR}/PAredu.pkl", "wb") as f:
            pickle.dump(PAredu, f)

def convert_dist_to_distdiff(dfthis, var_score, var_idx = "idx_morph_temp_rank"):
    """
    Get projection distance between adjacent indices 
    Get differenes between adjancent morph indices, projected onto the axis.

    NOTE: if dfthis has multiple trials per idx_morph_temp_rank, and 
    mult time bins (ie.., score for each tbin), then will 
    average over all trials and time bins.
    """
    
    list_idx_morph = sorted(dfthis[var_idx].unique().tolist())
    res = []
    for i in range(len(list_idx_morph)-1):
        idx1 = list_idx_morph[i]
        idx2 = list_idx_morph[i+1]
        
        score1 = dfthis[dfthis[var_idx] == idx1][var_score]
        score2 = dfthis[dfthis[var_idx] == idx2][var_score]

        res.append({
            # f"{var_score}-idx2-min-idx1":np.mean(score2) - np.mean(score1),
            "dist":np.mean(score2) - np.mean(score1),
            "idx_along_morph":i,
            "idx1":idx1,
            "idx2":idx2,
            "var_idx":var_idx,
        })
    
    return pd.DataFrame(res)

def _compute_diffs_varying_category_boundary(PAredu, n_flank_boundary=3):
    """
    Iteratively split data by idx, using different boudnaries from small to large, to ask if there is a 
    boundary that maximally splits indices into two groups.

    .e.g, these are successive category boundaries
        [0]  --  [1, 2, 3]
        ... computing distance matrices, using distnace: euclidian
        [0, 1]  --  [2, 3, 4]
        ... computing distance matrices, using distnace: euclidian
        [0, 1, 2]  --  [3, 4, 5]
        ... computing distance matrices, using distnace: euclidian
        [1, 2, 3]  --  [4, 5, 6]
        ... computing distance matrices, using distnace: euclidian
        [2, 3, 4]  --  [5, 6, 99]
        ... computing distance matrices, using distnace: euclidian
        [3, 4, 5]  --  [6, 99]
        ... computing distance matrices, using distnace: euclidian
        [4, 5, 6]  --  [99]
    """

    dflab = PAredu.Xlabels["trials"]
    idxs_in_order = sorted(dflab["idx_morph_temp"].unique())
    list_df = []
    for idx_boundary_left in idxs_in_order[:-1]:
        idxs_left = [i for i in idxs_in_order if i<=idx_boundary_left]
        idxs_right = [i for i in idxs_in_order if i>idx_boundary_left]
        # idxs_left = [0, 1]
        # idxs_right = [3,4]

        if n_flank_boundary is not None:
            idxs_left = idxs_left[-n_flank_boundary:]
            idxs_right = idxs_right[:n_flank_boundary]
        
        print(idxs_left, " -- ", idxs_right)

        # Update the category label
        dflab = PAredu.Xlabels["trials"]
        dflab["idx_morph_temp_catg"] = "ignore"
        dflab.loc[dflab["idx_morph_temp"].isin(idxs_left), "idx_morph_temp_catg"] = "left"
        dflab.loc[dflab["idx_morph_temp"].isin(idxs_right), "idx_morph_temp_catg"] = "right"
        PAredu.Xlabels["trials"] = dflab

        # Compute distance between categhories.
        cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp_catg", None, 
                                                            version_distance="euclidian", return_cldist=True, get_reverse_also=False,
                                                            compute_same_diff_scores=False)
        dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

        dfdist["idx_boundary_left"] = idx_boundary_left
        dfdist["idxs_left"] = [tuple(idxs_left) for _ in range(len(dfdist))]
        dfdist["idxs_right"] = [tuple(idxs_right) for _ in range(len(dfdist))]

        list_df.append(dfdist)
    
    DFDISTS = pd.concat(list_df).reset_index(drop=True)

    # Keep just distances between the two categories.
    DFDISTS = DFDISTS[(DFDISTS["labels_1"] == ("left", )) & (DFDISTS["labels_2"] == ("right", ))].reset_index(drop=True)

    return DFDISTS

def compute_pa_to_df_diffs_vary_categ_bound_using_splits(PA, PRINT=False, n_flank_boundary=3):
    """
    Computes differences across successive category boundaires, then re-aligns to center index, 
    finding that index in a cross-validated way.
    """
    # 1. Split into stratified dfatasets (2)
    paredu1, paredu2 = PA.split_sample_stratified_by_label(["idx_morph_temp", "seqc_0_loc"], PRINT=PRINT)

    dfdiffs_1 = _compute_diffs_varying_category_boundary(paredu1, n_flank_boundary)
    dfdiffs_2 = _compute_diffs_varying_category_boundary(paredu2, n_flank_boundary)

    exclude_n_at_edge = 1 # ensure that each category has at elast 2 indices, to reduce noise. Otherwise the edge often wins.
    dfdiffs_2, _, _ = _recenter_idx_general(dfdiffs_1, dfdiffs_2, "dist_yue_diff", "idx_boundary_left", exclude_n_at_edge=exclude_n_at_edge)
    dfdiffs_1, _, _ = _recenter_idx_general(dfdiffs_2, dfdiffs_1, "dist_yue_diff", "idx_boundary_left", exclude_n_at_edge=exclude_n_at_edge)

    # concatenate reuslts, trakcing the split index
    dfdiffs_1["split_idx"] = 1
    dfdiffs_2['split_idx'] = 2
    dfdiffs = pd.concat([dfdiffs_1, dfdiffs_2], axis=0).reset_index(drop=True)

    # RETURNS
    return dfdiffs

def compute_pa_to_df_dist_index_using_splits(PA, PRINT=False, var_idx = "idx_morph_temp_rank"):
    """
    Compute "dist index" which is d1/(d1+d2) where d1 and d2 are distances to base indices.
    """
    # 1. Split into stratified dfatasets (2)
    paredu1, paredu2 = PA.split_sample_stratified_by_label(["idx_morph_temp", "seqc_0_loc"], PRINT=PRINT)

    dfproj_index_1, dfdiffsindex_1 = _compute_df_using_dist_index(paredu1)
    dfproj_index_2, dfdiffsindex_2 = _compute_df_using_dist_index(paredu2)

    if PRINT:
        display(dfproj_index_1)
        display(dfproj_index_2)
        display(dfdiffsindex_1)
        display(dfdiffsindex_2)

    dfproj_index_2, dfdiffsindex_2 = _recenter_idx(dfdiffsindex_1, dfproj_index_2, dfdiffsindex_2)
    dfproj_index_1, dfdiffsindex_1 = _recenter_idx(dfdiffsindex_2, dfproj_index_1, dfdiffsindex_1)

    if PRINT:
        display(dfproj_index_1)
        display(dfproj_index_2)
        display(dfdiffsindex_1)
        display(dfdiffsindex_2)

    # concatenate reuslts, trakcing the split index
    dfdiffsindex_1["split_idx"] = 1
    dfdiffsindex_2['split_idx'] = 2
    dfdiffsindex = pd.concat([dfdiffsindex_1, dfdiffsindex_2], axis=0).reset_index(drop=True)

    dfproj_index_1["split_idx"] = 1
    dfproj_index_2['split_idx'] = 2
    dfproj_index = pd.concat([dfproj_index_1, dfproj_index_2], axis=0).reset_index(drop=True)

    # RETURNS
    return dfdiffsindex, dfproj_index


def _rank_idxs_append(df):
    """
    Rank idxs, so that 0-->0 and 99-->(max morph idx +1)
    e.g. [-1, 0, 1, 2, 99, 100] --> [-1, 0, 1, 2, 3, 4]

    Modifies df by appending column: idx_morph_temp_rank
    """
    # from pythonlib.tools.listtools import rank_items
    # rank_items(dfproj_index_1["idx_morph_temp"], "dense")
    idx_max = df[df["idx_morph_temp"] < 99]["idx_morph_temp"].max()
    def f(x):
        if x<=idx_max:
            return x
        elif x>=99:
            return x - 99 + idx_max + 1
        else:
            assert False    
    df["idx_morph_temp_rank"] = [f(x) for x in df["idx_morph_temp"]]
    

def _compute_df_using_dist_index_traj(pa, var_effect = "idx_morph_temp", effect_lev_base1=0, effect_lev_base2=99,
                                      list_grps_get=None, version="pts_time", var_context_diff=None, plot_conjunctions_savedir=None):
    """
    """
    DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS = pa.dataextract_as_distance_index_between_two_base_classes(
        var_effect, effect_lev_base1, effect_lev_base2, list_grps_get, version, var_context_diff=var_context_diff, 
        plot_conjunctions_savedir=plot_conjunctions_savedir)
    return DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS
    
    # from neuralmonkey.scripts.analy_decode_moment_psychometric import dfdist_to_dfproj_index
    # from neuralmonkey.scripts.analy_decode_moment_psychometric import dfdist_to_dfproj_index_datapts
    # # At each time, score distance between pairs of groupigs 
    # from pythonlib.tools.pandastools import aggregGeneral 

    # if version=="pts_time":
    #     # Use indiv datapts (distnace between pts vs. groups), and separately for each time bin
    #     pts_or_groups="pts"
    #     return_as_single_mean_over_time=False
    # elif version == "pts_scal":
    #     # Use indiv datapts, but use mean eucl distance across tiem bins. 
    #     # NOTE: this mean is taken of the eucl dist across time ibns, which themselves are computed independetmyl.
    #     # so this is nothing special. Is not that good - might as well run using pts_time, and then average the result over tmime.
    #     pts_or_groups="pts"
    #     return_as_single_mean_over_time=True
    # elif version == "grps_time":
    #     # Distance bewteen (group vs. group), separately fro each time bin.
    #     pts_or_groups="grps"
    #     return_as_single_mean_over_time=False
    # elif version == "grps_scal":
    #     # See above.
    #     pts_or_groups="grps"
    #     return_as_single_mean_over_time=True
    # else:
    #     print(version)
    #     assert False, "typo for version?"

    # ### Get distance between all trials at each time bin
    # version_distance = "euclidian"
    # if return_as_single_mean_over_time:
    #     # Each trial pair --> scalar
    #     cldist = pa.dataextract_as_distance_matrix_clusters_flex([var_effect], version_distance=version_distance,
    #                                                                             accurately_estimate_diagonal=False, 
    #                                                                             return_as_single_mean_over_time=return_as_single_mean_over_time)

    #     if pts_or_groups=="pts":
    #         # Score each datapt
    #         # For each datapt, get its distance to each of the groupings.
    #         # --> nrows = (ndatapts x n groups).
    #         # list_grps_get = [
    #         #     ("0|base1",),  
    #         #     ("99|base2",)
    #         #     ] # This is important, or else will fail if there are any (idx|assign) with only one datapt.
    #         DFDIST = cldist.rsa_distmat_score_all_pairs_of_label_groups_datapts(get_only_one_direction=False, list_grps_get=list_grps_get)
    #         DFPROJ_INDEX = dfdist_to_dfproj_index_datapts(DFDIST, var_effect=var_effect, 
    #                                                 effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
    #         # dfproj_index
    #         # order = sorted(dfproj_index["idxmorph_assigned_1"].unique())
    #         # sns.catplot(data=dfproj_index, x="idxmorph_assigned_1", y="dist_index", aspect=2, order=order)
    #     elif pts_or_groups=="grps":
    #         # Score pairs of (group, group)
    #         # Obsolete, because this is just above, followed by agging
    #         DFDIST = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

    #         # convert distnaces to distance index
    #         DFPROJ_INDEX = dfdist_to_dfproj_index(DFDIST, var_effect=var_effect, 
    #                                                 effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
    #     else:
    #         assert False

    #     # Take mean over trials
    #     if pts_or_groups=="pts":
    #         DFPROJ_INDEX_AGG = aggregGeneral(DFPROJ_INDEX, ["labels_1_datapt", var_effect], ["dist_index"])
    #         DFDIST_AGG = aggregGeneral(DFDIST, ["labels_1_datapt", "labels_2_grp", var_effect, f"{var_effect}_1", f"{var_effect}_2"], ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff"])
    #     elif pts_or_groups == "grps":
    #         DFPROJ_INDEX_AGG = None
    #         DFDIST_AGG = None
    #     else:
    #         assert False

    # else:
    #     # Each trial pair --> vector
    #     list_cldist, list_time = pa.dataextract_as_distance_matrix_clusters_flex([var_effect], version_distance=version_distance,
    #                                                                             accurately_estimate_diagonal=False, 
    #                                                                             return_as_single_mean_over_time=return_as_single_mean_over_time)
    

    #     ### For each time bin, for each trial, get its dist index relative to base1 and base2.
    #     list_dfproj_index = []
    #     list_dfdist = []
    #     for i, (cldist, time) in enumerate(zip(list_cldist, list_time)):
    #         if pts_or_groups=="pts":
    #             # Score each datapt
    #             # For each datapt, get its distance to each of the groupings.
    #             # --> nrows = (ndatapts x n groups).
    #             # list_grps_get = [
    #             #     ("0|base1",),  
    #             #     ("99|base2",)
    #             #     ] # This is important, or else will fail if there are any (idx|assign) with only one datapt.
    #             dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups_datapts(get_only_one_direction=False, list_grps_get=list_grps_get)
    #             dfproj_index = dfdist_to_dfproj_index_datapts(dfdist, var_effect=var_effect, 
    #                                                     effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
    #             # dfproj_index
    #             # order = sorted(dfproj_index["idxmorph_assigned_1"].unique())
    #             # sns.catplot(data=dfproj_index, x="idxmorph_assigned_1", y="dist_index", aspect=2, order=order)
    #         elif pts_or_groups=="grps":
    #             # Score pairs of (group, group)
    #             # Obsolete, because this is just above, followed by agging
    #             dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

    #             # convert distnaces to distance index
    #             dfproj_index = dfdist_to_dfproj_index(dfdist, var_effect=var_effect, 
    #                                                     effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
    #         else:
    #             assert False

    #         dfproj_index["time_bin"] = time
    #         dfdist["time_bin"] = time

    #         dfproj_index["time_bin_idx"] = i
    #         dfdist["time_bin_idx"] = i

    #         list_dfproj_index.append(dfproj_index)
    #         list_dfdist.append(dfdist)

    #     ### Clean up the results
    #     DFPROJ_INDEX = pd.concat(list_dfproj_index).reset_index(drop=True)
    #     DFDIST = pd.concat(list_dfdist).reset_index(drop=True)
    #     # DFDIST[var_effect] = DFDIST[f"{var_effect}_1"]

    #     # Take mean over trials
    #     if pts_or_groups=="pts":
    #         DFPROJ_INDEX_AGG = aggregGeneral(DFPROJ_INDEX, ["labels_1_datapt", var_effect, "time_bin_idx"], ["dist_index"], nonnumercols=["time_bin"])
    #         DFDIST_AGG = aggregGeneral(DFDIST, ["labels_1_datapt", "labels_2_grp", var_effect, f"{var_effect}_1", f"{var_effect}_2", "time_bin_idx"], ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff", "time_bin"])
    #     elif pts_or_groups == "grps":
    #         DFPROJ_INDEX_AGG = None
    #         DFDIST_AGG = None
    #     else:
    #         assert False

    # ######## GET DIFFERENCE ACROSS ADJACENT IDNICES
    # if var_effect == "idx_morph_temp":
    #     _rank_idxs_append(DFPROJ_INDEX)
    #     DFPROJ_INDEX_DIFFS = convert_dist_to_distdiff(DFPROJ_INDEX, "dist_index", "idx_morph_temp_rank")
    # else:
    #     DFPROJ_INDEX_DIFFS = None

    # return DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS

def _compute_df_using_dist_index(pa, var_effect = "idx_morph_temp", effect_lev_base1=0, effect_lev_base2=99):
    """
    Helper to extract dist (eucl, pairwise btw pts) between each level of idx_morph_temp
    and then convert that into an index, scaled between (0, 1), for how close each morph is to base1 and base2.
    Also get diference of this score between adjacent indices

    """

    # THis is using the good general purpose code.
    version = "grps_scal"
    dfproj_index, _, _, _, dfdiffsindex = _compute_df_using_dist_index_traj(pa, var_effect, 
                                                                                        effect_lev_base1, effect_lev_base2, 
                                                                                        version = version)
    
    # # 1. Convert to pairwise distances dist_yue_diff)
    # cldist, _ = euclidian_distance_compute_trajectories_single(pa, var_effect, ["epoch"], 
    #                                                         version_distance="euclidian", return_cldist=True, get_reverse_also=False)
    # dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)

    # dfproj_index = dfdist_to_dfproj_index(dfdist, var_effect=var_effect, 
    #                                       effect_lev_base1=effect_lev_base1, effect_lev_base2=effect_lev_base2)
    
    # if var_effect == "idx_morph_temp":
    #     _rank_idxs_append(dfproj_index)
    #     dfdiffsindex = convert_dist_to_distdiff(dfproj_index, "dist_index", "idx_morph_temp_rank")
    # else:
    #     dfdiffsindex = None

    return dfproj_index, dfdiffsindex

def dfdist_to_dfproj_index_datapts(dfdist_pts, var_score="dist_mean", var_effect = "idx_morph_temp",
                           effect_lev_base1=0, effect_lev_base2=99):
    """Like dfdist_to_dfproj_index, but here dfdist rows to represent datapts/trials (So this would be trial vs. group of tirals), 
    as oposed to there, whihc is rows are distances between groups of trials. 

    PARAMS:
    - var_score, which scoree to use. By default, use dist_mean, which is the most raw. Simply the eucl distance.
    """

    from pythonlib.cluster.clustclass import Clusters
    cl = Clusters(None)
    dfproj_index = cl.rsa_dfdist_to_dfproj_index_datapts(dfdist_pts, var_score, var_effect, effect_lev_base1, effect_lev_base2)
    return dfproj_index

    # list_idx_datapt = sorted(dfdist_pts["idx_row_datapt"].unique())

    # res_dist_index = []
    # for idx_datapt in list_idx_datapt:
    #     tmp = dfdist_pts[(dfdist_pts["idx_row_datapt"] == idx_datapt) & (dfdist_pts[f"{var_effect}_2"]==effect_lev_base1)]
    #     if not len(tmp)==1:
    #         print(idx_datapt)
    #         print(tmp)
    #         assert False, "prob need to set get_only_one_direction==False or version_datapts==True"
    #     d1 = tmp[var_score].values[0]

    #     tmp = dfdist_pts[(dfdist_pts["idx_row_datapt"] == idx_datapt) & (dfdist_pts[f"{var_effect}_2"]==effect_lev_base2)]
    #     if not len(tmp)==1:
    #         print(idx_datapt)
    #         print(tmp)
    #         assert False, "prob need to set get_only_one_direction==False or version_datapts==True"
    #     d2 = tmp[var_score].values[0]

    #     dist_index = d1/(d1+d2)

    #     res_dist_index.append({
    #         "idx_row_datapt":idx_datapt,
    #         "labels_1_datapt":tmp["labels_1_datapt"].values[0],
    #         f"{var_effect}":tmp[f"{var_effect}_1"].values[0],
    #         "dist_index":dist_index,
    #     })
    # dfproj_index = pd.DataFrame(res_dist_index)

    # return dfproj_index


def dfdist_to_dfproj_index(dfdist, var_score="dist_mean", var_effect = "idx_morph_temp",
                           effect_lev_base1=0, effect_lev_base2=99, version_datapts=False):
    """Convert pairwise distances (between idx) into dist
    index, which is:
    for a given index, d1/(d1+d2) where d1 and d2 are distances from that index to base1 and base2.
    PARAMS:
    - dist_mean, should use "dist_mean", the pairiwse distance betwene
    all pts... This means that even for a base prim, the index will not be 0 or 1. but this is ok,
    since you can normalize the final score to whatever it is for the base prims. Dont use dist_yue, becuase 
    it can be small and negetive, which can lead to problems when take division here.
    - version_datapts, if True, then allows for dfdist rows to represent datapts/trials (So this would be trial vs. group of tirals), 
    as oposed to default, whihc is rows are distances between groups of trials.
    """

    assert False, "insetad, use dfdist_to_dfproj_index_datapts, this is redundant. is fine if you need to uncomment this."
    list_idx_morph = sorted(dfdist[f"{var_effect}_1"].unique().tolist())
    res_dist_index = []
    for idx in list_idx_morph:

        tmp = dfdist[(dfdist[f'{var_effect}_1']==effect_lev_base1) & (dfdist[f'{var_effect}_2']==idx)]
        if version_datapts:
            d1 = tmp[var_score].mean()
        else:
            if not len(tmp)==1:
                print(idx)
                print(tmp)
                assert False, "prob need to set get_only_one_direction==False or version_datapts==True"
            d1 = tmp[var_score].values[0]
        
        tmp = dfdist[(dfdist[f'{var_effect}_1']==idx) & (dfdist[f'{var_effect}_2']==effect_lev_base2)]
        if version_datapts:
            d2 = tmp[var_score].mean()
        else:
            if not len(tmp)==1:
                print(idx)
                print(tmp)
                assert False, "prob need to set get_only_one_direction==False or version_datapts==True"
            d2 = tmp[var_score].values[0]

        dist_index = d1/(d1+d2)

        # print(idx, d1, d2, " -- ", dist_index)
        res_dist_index.append({
            "dist_index":dist_index,
            var_effect:idx,
        })
    dfproj_index = pd.DataFrame(res_dist_index)

    return dfproj_index

def _recenter_idx_general(df_for_index, df_to_mod, var_score="dist", var_idx="idx_morph_temp_rank",
                          exclude_n_at_edge=0):
    """
    Find index in <df_for_index> that maximizes <var_score> and then recenter that index in 
    <df_to_mod> so that the max idx is now 0. 

    PARAMS:
    - df_for_index, the df used to find index of max score.
    - exclude_n_at_edge, if >1, then exludes this many items at the edges. Useful if you want to ensure that the max index is within
    some bounds. e..g, if exclude_n_at_edge==1, then considers indices [1,2,3,4,5, n-1], insetad of [0, 1...., n].
    RETURNS:
    - copy of df_to_mod, with new coolumn <var_idx>centered
    EXAMPLE:
    - useful for aligning in heatmap plot
    """
    from pythonlib.tools.pandastools import _check_index_reseted
    _check_index_reseted(df_for_index)
    
    # Find point of max
    if exclude_n_at_edge>0:
        indrow_max = df_for_index.iloc[exclude_n_at_edge:-exclude_n_at_edge][var_score].idxmax()
        indrow_min = df_for_index.iloc[exclude_n_at_edge:-exclude_n_at_edge][var_score].idxmin()
    else:
        indrow_max = df_for_index[var_score].idxmax()
        indrow_min = df_for_index[var_score].idxmin()

    # Modify the other df
    if df_to_mod is not None:
        _check_index_reseted(df_to_mod)
        df_to_mod[f"{var_idx}_centered"] = df_to_mod[var_idx] - df_for_index.iloc[indrow_max][var_idx]

    return df_to_mod, indrow_min, indrow_max

def _recenter_idx(dfdiffsindex_in, dfproj_index_out, dfdiffsindex_out, var_diff_in="dist", 
                  var_idx="idx_morph_temp_rank", exclude_n_at_edge=0):
    """
    Append columns in dfproj_index_out, dfdiffsindex_out, recentering their indices based
    on the index in dfdiffsindex_in that has max diff. 

    PARAMS:
    - exclude_n_at_edge, if >1, then exludes this many items at the edges. Useful if you want to ensure that the max index is within
    some bounds. e..g, if exclude_n_at_edge==1, then considers indices [1,2,3,4,5, n-1], insetad of [0, 1...., n].
    E.g., useful for aligning in heatmap plot
    """
    # 3. find point of max
    if exclude_n_at_edge>0:
        indrow_max = dfdiffsindex_in.iloc[exclude_n_at_edge:-exclude_n_at_edge][var_diff_in].idxmax()
    else:
        indrow_max = dfdiffsindex_in[var_diff_in].idxmax()

    if not dfdiffsindex_in["var_idx"].unique().tolist() == [var_idx]:
        print(dfdiffsindex_in)
        print(var_idx)
        assert False

    # 4. apply that max to the other dataset
    # - apply to both the (dist from base) and (dist between adjacent) datasets
    dfproj_index_out[f"{var_idx}_centered"] = dfproj_index_out[var_idx] - dfdiffsindex_in.iloc[indrow_max]["idx1"]
    dfdiffsindex_out["idx_along_morph_centered"] = dfdiffsindex_out["idx_along_morph"] - dfdiffsindex_in.iloc[indrow_max]["idx_along_morph"]

    return dfproj_index_out, dfdiffsindex_out


def analy_morphsmooth_euclidian_score(PAredu, savedir, exclude_flankers, morphset, n_flank_boundary, DOPLOTS=False):
    """
    Compute all stats related to categorical switching between base prims, given a continuos morph.
    THese are the good/latest analyses, using euclidian distnace (e.g., distnace between adjancent indices).
    """
    
    assert isinstance(n_flank_boundary, int)

    if exclude_flankers:
        _idxs_no_flankers = [x for x in PAredu.Xlabels["trials"]["idx_morph_temp"].unique().tolist() if x>=0 and x<=99]
        PAredu = PAredu.slice_by_labels_filtdict({"idx_morph_temp":_idxs_no_flankers})

    ####################### Project trials onto the axis (base1 - base2)
    if PAredu.X.shape[2]>1:
        # This is traj. Hacky solution, to avgg over time. Is ok since I prob dont care this value (proj) anyway.
        X = np.mean(PAredu.X, axis=2).T # (trials, ndims)
    else:
        X = PAredu.X.squeeze(axis=2).T # (trials, ndims)
    dflab = PAredu.Xlabels["trials"]

    # Do this in cross-validated way -- split base prims into two sets, run this 2x, then 
    # average over base prims.
    from pythonlib.tools.statstools import crossval_folds_indices

    inds_pool_base1 = dflab[dflab["idx_morph_temp"] == 0].index.values
    inds_pool_base2 = dflab[dflab["idx_morph_temp"] == 99].index.values
    list_inds_1, list_inds_2 = crossval_folds_indices(len(inds_pool_base1), len(inds_pool_base2), 2)

    # shuffle inds
    np.random.shuffle(inds_pool_base1)
    np.random.shuffle(inds_pool_base2)

    list_inds_1 = [inds_pool_base1[_inds] for _inds in list_inds_1]
    list_inds_2 = [inds_pool_base2[_inds] for _inds in list_inds_2]

    list_X =[]
    for _i, (inds_base_1, inds_base_2) in enumerate(zip(list_inds_1, list_inds_2)):
        # inds_base_1 = inds_pool_base1[_inds1]
        # inds_base_2 = inds_pool_base2[_inds2]
        # print(inds_base_1)

        # - get mean activity for base1, base2
        xmean_base1 = np.mean(X[inds_base_1,:], axis=0) # (ndims,)
        xmean_base2 = np.mean(X[inds_base_2,:], axis=0) # (ndims,)

        # get mean projected data for each state
        # xproj, fig = projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=True)
        plot_color_labels = dflab["idx_morph_temp"].values
        xproj, fig = projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=True, 
                                                plot_color_labels=plot_color_labels)
        savefig(fig, f"{savedir}/morphset={morphset}-projections_preprocess-iter={_i}.pdf")
        # replace the train data with nan
        xproj[inds_base_1] = np.nan
        xproj[inds_base_2] = np.nan

        list_X.append(xproj)

    # Get average over iterations
    Xproj = np.nanmean(np.stack(list_X), axis=0)

    dfproj = PAredu.Xlabels["trials"].copy()
    dfproj["x_proj"] = Xproj

    ########################### Get eucl distnaces between adjacent indices
    # Get distances between adjacent indices
    if True:
        # Use dist_yue instead
        cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp", ["epoch"], 
                                                                version_distance="euclidian", return_cldist=True, get_reverse_also=False)
        # get distance between 0 and 99
        # res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98 = cldist.rsa_distmat_score_same_diff_by_context("idx_morph_temp", ["epoch"], None, "pts", )
        # pd.DataFrame(res)
        dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)
        VAR_SCORE = "dist_yue_diff"
    else:
        # use euclidian dist norm -- problem --> this is noisy.
        assert False, "dist index below assumes is doing euclidian, not euclidian_unbiased"
        cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp", ["epoch"], 
                                                                version_distance="euclidian_unbiased", return_cldist=True, get_reverse_also=False)
        # get distance between 0 and 99
        dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups()
        VAR_SCORE = "dist_mean"

    # Collect all adjacent pairs
    dflab = PAredu.Xlabels["trials"]
    list_idx_morph = sorted(dflab[(dflab["idx_morph_temp"]>=0) & (dflab["idx_morph_temp"]<=99)]["idx_morph_temp"].unique().tolist())
    res_diffs = []
    for i in range(len(list_idx_morph)-1):
        tmp = dfdist[(dfdist["idx_morph_temp_1"] == list_idx_morph[i]) & (dfdist["idx_morph_temp_2"] == list_idx_morph[i+1])]
        assert len(tmp)==1, f"{i}"

        # Get the distance
        d = tmp[VAR_SCORE].values[0]

        # save
        res_diffs.append({
            "idx_along_morph":i,
            "idx1":list_idx_morph[i],
            "idx2":list_idx_morph[i+1],
            "dist":d,
            # "animal":animal,
            # "date":date,
            # "morphset":morphset,
            # "bregion":bregion
        })
    dfdiffs = pd.DataFrame(res_diffs)            

    ####################### EXTRA STUFF
    # Also figure out whether base1 and base2 are sufficiently separated to include this data
    # - Collect distances between adjacent morph indices.
    # cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp", 
    #                                                                 ["epoch"], return_cldist=True, get_reverse_also=False)
    # get distance between 0 and 99
    # dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups()
    tmp = dfdist[(dfdist["idx_morph_temp_1"] == 0) & (dfdist["idx_morph_temp_2"] == 99)]
    assert len(tmp)==1
    dist_between_bases = tmp[VAR_SCORE].values[0]
    DIST_98 = tmp["DIST_98"].values[0]
    # dfproj["dist_between_bases"] = dist_between_bases
    # dfproj["dist_norm_between_bases"] = dist_between_bases/DIST_98

    # dfdiffs["dist_between_bases"] = dist_between_bases
    # dfdiffs["dist_norm_between_bases"] = dist_between_bases/DIST_98

    ### Base prim separation score
    # - ie., frac (between 0, 1), if 1 then base prim separation is greater than 100% of the other prims.
    _dfdist_notbase = dfdist[(~dfdist["idx_morph_temp_1"].isin([0, 99])) & (~dfdist["idx_morph_temp_2"].isin([0, 99]))]
    n = sum(_dfdist_notbase[VAR_SCORE]<dist_between_bases)
    ntot = len(_dfdist_notbase)
    base_prim_separation_score = n/ntot
    # dfproj["base_prim_separation_score"] = base_prim_separation_score
    # dfdiffs["base_prim_separation_score"] = base_prim_separation_score

    if DOPLOTS:
        fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", jitter=True, alpha=0.5)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
            ax.axhline(1, color="k", alpha=0.5)
        savefig(fig, f"{savedir}/morphset={morphset}-catplot-1.pdf")

        fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", kind="point")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
            ax.axhline(1, color="k", alpha=0.5)
        savefig(fig, f"{savedir}/morphset={morphset}-catplot-2.pdf")

        fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", kind="violin")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
            ax.axhline(1, color="k", alpha=0.5)
        savefig(fig, f"{savedir}/morphset={morphset}-catplot-3.pdf")

    plt.close("all")

    ### COLLECT ALL
    # Rescale so that 0 <--> 99 maps onto 0 <--> 1 index, so can compare across expts
    idx_max = dfproj[dfproj["idx_morph_temp"] < 99]["idx_morph_temp"].max()

    def f(x):
        if x<=idx_max:
            return x
        elif x>=99:
            return x - 99 + idx_max + 1
        else:
            print(x)
            assert False
    
    _rank_idxs_append(dfproj)
    dfproj["idx_morph_rescaled"] = [f(x) for x in dfproj["idx_morph_temp"]]
    from pythonlib.tools.listtools import rank_items
    # assert np.all(dfproj["idx_morph_rescaled"].tolist() == list(rank_items(dfproj["idx_morph_temp"]-1, method="dense"))), "sanity cehck."
    dfproj["idx_morph_rescaled"] = dfproj["idx_morph_rescaled"]/(idx_max+1) # rescale
    # dfproj.groupby(["idx_morph_temp", "idx_morph_rescaled"]).size()

    # dfproj["animal"] = animal
    # dfproj["date"] = date
    # dfproj["morphset"] = morphset
    # dfproj["bregion"] = bregion
    
    # Track how many morphs exist
    nmorphs = len(dfproj[(dfproj["idx_morph_temp"]>=0) & (dfproj["idx_morph_temp"]<=99)]["idx_morph_temp"].unique())
    dfproj["nmorphs"] = nmorphs
    dfdiffs["nmorphs"] = nmorphs
        
    ########################## GET projection distance between adjacent indices
    var_score = "x_proj"
    dfthis = dfproj
    # Get differenes between adjancent morph indices, projected onto the axis.
    dfdiffsproj = convert_dist_to_distdiff(dfthis, var_score)

    ### [Best method?] dist index
    if True:
        # Does splits, returning 2x num rows, splitting lets you recenter indinces without overfitting.
        dfdiffsindex, dfproj_index = compute_pa_to_df_dist_index_using_splits(PAredu)
    else:

        dfproj_index = dfdist_to_dfproj_index(dfdist)
        # Get diffs
        dfdiffsindex = convert_dist_to_distdiff(dfproj_index, "dist_index")

    ################## SPLIT INTO TWO CATEGORIES, EACH TIME COMPUTING DIFFERENCE BETWEEN THEM
    if False:
        # Do using all data
        dfdiffs_categ = _compute_diffs_varying_category_boundary(PAredu, n_flank_boundary)
    else:
        # Do splitting, and align to center index.
        dfdiffs_categ = compute_pa_to_df_diffs_vary_categ_bound_using_splits(PAredu, False, n_flank_boundary)

    # Append useful stuff
    for _df in [dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex, dfdiffs_categ]:
        _df["base_prim_separation_score"] = base_prim_separation_score
        _df["dist_between_bases"] = dist_between_bases
        _df["dist_norm_between_bases"] = dist_between_bases/DIST_98
        _df["nmorphs"] = nmorphs

    return dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex, dfdiffs_categ


def analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                 scalar_or_traj,
                                 EVENT, raw_subtract_mean_each_timepoint,
                                 dim_red_method, proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP,
                                 savedir, restricted_twind_for_dpca=None,
                                 exclude_flankers=False, skip_dim_redu=False):
    """
    For smooth morphs (categorical)
    General purpose prep dataset for morphsets, extracting, adding variables for this morphset, and then dim reduction.
    PARAMS:
    - proj_twind, window of final data. 
    - restricted_twind_for_dpca, window for computing PC space. if not None, then (t1, t2) defines a window for identifying dpca subspace. Data from the entire 
    window of proj_twind will be projected into this space.
    RETURNS:
    - PA (a copy)
    """ 

    # Prune neural data to keep only those trialcodes that are in DSmorphsets
    list_trialcode = sorted(set([x[0] for x in map_tcmorphset_to_idxmorph]))
    DFallpa["pa"] = [pa.slice_by_labels("trials", "trialcode", list_trialcode) for pa in DFallpa["pa"].values]

    ############### PREPARE DATASET
    PA = extract_single_pa(DFallpa, bregion, None, "trial", EVENT)

    ### Prep, with indices for this morphset
    # (Given morphset, assign new column which is the trial's role in that morphset)
    dflab = PA.Xlabels["trials"]
    dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

    # keep just trials that are in this morphset
    idx_exist = list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"]))
    filtdict = {
        "idx_morph_temp":idx_exist,
    }
    PA = PA.slice_by_labels_filtdict(filtdict)

    if exclude_flankers:
        _idxs_no_flankers = [x for x in PA.Xlabels["trials"]["idx_morph_temp"].unique().tolist() if x>=0 and x<=99]
        PA = PA.slice_by_labels_filtdict({"idx_morph_temp":_idxs_no_flankers})

    # For each trial, which base prims was it assigned to. Possibilities:
    # not_ambig_base2    69
    # not_ambig_base1    42
    # base2              33
    # base1              31
    # ambig_base2        22
    # ambig_base1         4
    dflab = PA.Xlabels["trials"]
    dflab["assigned_base"] = [map_tcmorphset_to_info[(tc, morphset)][0] for tc in dflab["trialcode"]]
    def f(x):
        if x in ["base1", "base2"]:
            return "base"
        elif x in ["not_ambig_base1", "not_ambig_base2"]:
            return "not_ambig"
        elif x in ["ambig_base1", "ambig_base2"]:
            return "ambig"
        elif x in ["not_enough_trials"]:
            return "not_enough_trials"
        else:
            print(x)
            assert False
    dflab["assigned_label"] = [f(x) for x in dflab["assigned_base"]]

    # Simplify it
    from pythonlib.tools.pandastools import append_col_with_grp_index
    def map_to_simple(x):
        a = "base1" in x
        b  = "base2" in x
        if a and not b:
            return "base1"
        elif b and not a:
            return "base2"
        elif x=="not_enough_trials":
            return "not_enough_trials"
        else:
            print(x, a, b)
            assert False
    dflab["assigned_base_simple"] = [map_to_simple(x) for x in dflab["assigned_base"]]
    dflab = append_col_with_grp_index(dflab, ["idx_morph_temp", "assigned_base_simple"], "idxmorph_assigned")

    # Also by location
    dflab = append_col_with_grp_index(dflab, ["idx_morph_temp", "seqc_0_loc"], "idx_morph_temp_loc")
    
    # Save 
    PA.Xlabels["trials"] = dflab

    ############### DIM REDUCTION
    if skip_dim_redu:
        Xredu = None
        PAredu = PA
    else:
        twind_final = proj_twind
        Xredu, PAredu = _analy_extract_PA_dim_reduction(PA, savedir, restricted_twind_for_dpca, twind_final,
                                        scalar_or_traj, dim_red_method, tbin_dur, pca_tbin_slice,
                                        NPCS_KEEP, raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)

    # # -- Fixed params
    # superv_dpca_var = "idx_morph_temp"
    # superv_dpca_vars_group = None
    # # dim_red_method = "dpca"

    # ### DIM REDUCTION
    # savedirpca = f"{savedir}/pca_construction"
    # os.makedirs(savedirpca, exist_ok=True)

    # if restricted_twind_for_dpca is None:
    #     restricted_twind_for_dpca = proj_twind
    # #     twind_pca = proj_twind
    # # else:
    # #     twind_pca = restricted_twind_for_dpca

    # Xredu, PAredu = PA.dataextract_dimred_wrapper(scalar_or_traj, dim_red_method, savedirpca, 
    #         restricted_twind_for_dpca, tbin_dur, pca_tbin_slice, NPCS_KEEP = NPCS_KEEP,
    #         dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_proj_twind = proj_twind, 
    #         raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)
    
    return Xredu, PAredu

def _analy_extract_PA_dim_reduction(PA, savedir, restricted_twind_for_dpca, twind_final,
                                    scalar_or_traj, dim_red_method, tbin_dur, pca_tbin_slice,
                                    NPCS_KEEP, raw_subtract_mean_each_timepoint=False):
    """
    Just dim reduction
    """
    ############### DIM REDUCTION
    # -- Fixed params
    superv_dpca_var = "idx_morph_temp"
    superv_dpca_vars_group = None
    # dim_red_method = "dpca"

    ### DIM REDUCTION
    savedirpca = f"{savedir}/pca_construction"
    os.makedirs(savedirpca, exist_ok=True)

    if restricted_twind_for_dpca is None:
        restricted_twind_for_dpca = twind_final
    #     twind_pca = proj_twind
    # else:
    #     twind_pca = restricted_twind_for_dpca

    Xredu, PAredu = PA.dataextract_dimred_wrapper(scalar_or_traj, dim_red_method, savedirpca, 
            restricted_twind_for_dpca, tbin_dur, pca_tbin_slice, NPCS_KEEP = NPCS_KEEP,
            dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_proj_twind = twind_final, 
            raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint, n_min_per_lev_lev_others=1)

    return Xredu, PAredu

def analy_psychoprim_statespace_euclidian(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, list_morphset):
    """
    Plot euclidian distances, looking for categorical represntation, continuous morphts.

    Does simple stuff:
    (1) State space plots.
    (2) Pairwise distances between indices within morph.
    
    NOTE: To summarize across days, run:
    analy_decode_moment_psychometric_mult.py, which loads these results and does computation.
    This is obsolete, instead, should run all things directly here and save the computation.
    """
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER

    list_bregion = DFallpa["bregion"].unique().tolist()
    
    EVENT = "06_on_strokeidx_0"

    # Extract data for this morphset, wiht labels updated
    for bregion in list_bregion:
        for morphset in list_morphset:
            ############### PREPARE DATASET
            # PA = extract_single_pa(DFallpa, bregion, None, "trial", EVENT)

            # # Given morphset, assign new column which is the trial's role in that morphset.
            # dflab = PA.Xlabels["trials"]
            # dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

            # # keep just trials that are in this morphset
            # idx_exist = list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"]))
            # filtdict = {
            #     "idx_morph_temp":idx_exist,
            # }
            # PA = PA.slice_by_labels_filtdict(filtdict)

            ############### DIM REDUCTION
            # Train -- PC space.
            # -- Params
            # NPCS_KEEP = 6
            # twind_train = (0.1, 1.0)

            # -- Fixed params
            superv_dpca_var = "idx_morph_temp"
            superv_dpca_vars_group = None

            # for raw_subtract_mean_each_timepoint in [True, False]:
            for raw_subtract_mean_each_timepoint in [False]:
                for scalar_or_traj in ["traj"]:
                # for scalar_or_traj in ["scal", "traj"]:
                    if EVENT == "03_samp" and scalar_or_traj=="traj":
                        NPCS_KEEP = 6
                        # list_twind = [(-0.5, 1.2), twind_train]
                        # list_twind = [(0.1, 1.2), (0.1, 0.6), (0.6, 1.2)]
                        # list_twind = [(0.1, 1.2), (0.6, 1.2)]
                        list_twind = [(0.1, 1.1)]
                    elif EVENT == "03_samp" and scalar_or_traj=="scal":
                        NPCS_KEEP = 8
                        # list_twind = [(0.1, 1.2), (0.1, 0.6), (0.6, 1.2)]
                        list_twind = [(0.1, 1.1), (0.6, 1.1)]
                    elif EVENT == "06_on_strokeidx_0" and scalar_or_traj=="traj":
                        NPCS_KEEP = 6
                        list_twind = [(-0.1, 0.3)]
                    elif EVENT == "06_on_strokeidx_0" and scalar_or_traj=="scal":
                        NPCS_KEEP = 8
                        list_twind = [(-0.1, 0.3)]
                    else:
                        assert False

                    if scalar_or_traj == "scal":
                        tbin_dur = 0.15
                        pca_tbin_slice = 0.1
                    elif scalar_or_traj == "traj":
                        tbin_dur = 0.15
                        pca_tbin_slice = 0.02
                    else:
                        assert False
                    
                    for proj_twind in list_twind:
                        # for dim_red_method in ["pca", "dpca"]:
                        for dim_red_method in ["dpca"]:
                            
                            Xredu, PAredu, savedir = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                                         scalar_or_traj, 
                                EVENT, raw_subtract_mean_each_timepoint, 
                                dim_red_method, proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP, 
                                SAVEDIR)

                            # assert False

                            # savedir = f"{SAVEDIR}/statespace_euclidian/{EVENT}/bregion={bregion}/morphset={morphset}-ver={scalar_or_traj}-{dim_red_method}-subtrmean={raw_subtract_mean_each_timepoint}-projtwind={proj_twind}-npcs={NPCS_KEEP}"
                            # os.makedirs(savedir, exist_ok=True)

                            # ### DIM REDUCTION
                            # savedirpca = f"{savedir}/pca_construction"
                            # os.makedirs(savedirpca, exist_ok=True)
                    
                            # Xredu, PAredu = PA.dataextract_dimred_wrapper(scalar_or_traj, dim_red_method, savedirpca, 
                            #         proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP = NPCS_KEEP,
                            #         dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_proj_twind = None, 
                            #         raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)

                            ### STATE SPACE PLOTS
                            dflab = PAredu.Xlabels["trials"]
                            times = PAredu.Times

                            if scalar_or_traj == "traj":
                                ### Trajectory plots 
                                time_bin_size = 0.05
                                list_var_color_var_subplot = [
                                    [superv_dpca_var, None],
                                ]
                                LIST_DIMS = [(0,1), (2,3)]
                                for var_color, var_subplot in list_var_color_var_subplot:
                                    trajgood_plot_colorby_splotby_WRAPPER(Xredu, times, dflab, var_color, savedir,
                                                                vars_subplot=var_subplot, list_dims=LIST_DIMS, time_bin_size=time_bin_size)
                                plt.close("all")

                            elif scalar_or_traj == "scal":
                                ### Plot scalars
                                list_var_color_var_subplot = [
                                    [superv_dpca_var, None],
                                ]
                                LIST_DIMS = [(0,1), (2,3)]
                                for var_color, var_subplot in list_var_color_var_subplot:

                                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu, dflab, var_color, savedir,
                                                                                    vars_subplot=var_subplot, list_dims=LIST_DIMS,
                                                                                    overlay_mean_orig=True
                                                                                    )
                            else:
                                assert False
                            
                            ############################## EUCLIDIAN DISTANCE
                            # Euclidian distance between adjacent steps in morphset
                            from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single
                            var_effect = "idx_morph_temp"
                            vars_others = None
                            PLOT_HEATMAPS = False
                            Cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, var_effect, vars_others, PLOT_HEATMAPS=PLOT_HEATMAPS, 
                                                                        savedir_heatmaps=savedir, get_reverse_also=False, return_cldist=True,
                                                                        compute_same_diff_scores=False)
                            dfdists = Cldist.rsa_dataextract_with_labels_as_flattened_df(plot_heat=False, exclude_diagonal=True)

                            import seaborn as sns
                            from pythonlib.tools.pandastools import plot_subplots_heatmap

                            fig, _ =plot_subplots_heatmap(dfdists, "idx_morph_temp_row", "idx_morph_temp_col", "dist", None)
                            savefig(fig, f"{savedir}/heatmap_distances.pdf")

                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", hue="idx_morph_temp_col", kind="point")
                            savefig(fig, f"{savedir}/catplot-1.pdf")

                            # Test hypothesis -- logistic curve
                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", col="idx_morph_temp_col", kind="point")
                            savefig(fig, f"{savedir}/catplot-2.pdf")

                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", col="idx_morph_temp_col", jitter=True, alpha=0.15)
                            savefig(fig, f"{savedir}/catplot-3.pdf")

                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", col="idx_morph_temp_col", kind="violin")
                            savefig(fig, f"{savedir}/catplot-4.pdf")

                            # For each index, get its distance to its adjacent indices
                            dfdists = Cldist.rsa_dataextract_with_labels_as_flattened_df(exclude_diagonal=True, plot_heat=False)
                            # Middle states -- evidence for trial by trial variation? i.e.,
                            # Scatterplot -- if similar to base1, then is diff from base 2?

                            from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
                            # subplot = trial kind 
                            # x and y are "decoders"
                            # var_datapt = same as "trialcode"
                            _, fig = plot_45scatter_means_flexible_grouping(dfdists, "idx_morph_temp_col", 0, 99, "idx_morph_temp_row", 
                                                                "dist", "idx_row", plot_text=False, plot_error_bars=True, SIZE=4.5, 
                                                                shareaxes=True)
                            savefig(fig, f"{savedir}/scatter-1.pdf")

                            plt.close("all")

                            # SAVE:
                            import pickle
                            with open(f"{savedir}/PAredu.pkl", "wb") as f:
                                pickle.dump(PAredu, f)
                            with open(f"{savedir}/Cldist.pkl", "wb") as f:
                                pickle.dump(Cldist, f)

def analy_psychoprim_statespace_euclidian_smooth_good(DFallpa, SAVEDIR_BASE, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                         animal, date, exclude_flankers=False):
    """
    [GOOD] - Final approach for smooth morphs --> category?
    Uses dist_index (i.e, relative to base1 and base2), and does score for each morphset, and then collects those scores across morphsetes.
    The score is max minus min distance between adjacnet indices.

    Uses traj, dist index is computed at each bin, then averaged over time. 
    """

    from neuralmonkey.scripts.analy_decode_moment_psychometric import _compute_df_using_dist_index_traj
    import seaborn as sns
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.plottools import savefig

    scalar_or_traj = "traj"
    tbin_dur = 0.1
    pca_tbin_slice = 0.1
    proj_twind = (-0.1, 1.1)
    restricted_twind_for_dpca = (0.1, 1.1)
    
    # Final, score, agging over time.

    raw_subtract_mean_each_timepoint = False
    EVENT = "03_samp"
    dim_red_method = "dpca"
    NPCS_KEEP = 7

    for twind_score in [
        (0.05, 1.1), # Already done.
        (0.05, 0.7),
        ]:
        # twind_score = (0.05, 1.1) 
        # twind_score = (0.05, 0.7) # Using shorter, since main effect seems to happen quickly

        # n_splits_tot = 200
        n_splits_tot = 100 # emprically, 100 looks same as 200.
        # n_splits = 4
        n_splits = 2

        SAVEDIR = f"{SAVEDIR_BASE}/smooth_euclidian_dist_index_good/ev={EVENT}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={twind_score}-npcs={NPCS_KEEP}-nsplitstot={n_splits_tot}-nsplitinn={n_splits}-exclflank={exclude_flankers}"
        os.makedirs(SAVEDIR, exist_ok=True)

        PLOT_EACH_FOLD = 6 # plot the first <PLOT_EACH_FOLD> folds
        n_flank = 1

        # Save params
        params = {
            "scalar_or_traj":scalar_or_traj,
            "tbin_dur":tbin_dur,
            "pca_tbin_slice":pca_tbin_slice,
            "proj_twind":proj_twind,
            "restricted_twind_for_dpca":restricted_twind_for_dpca,
            "twind_score":twind_score,
            "raw_subtract_mean_each_timepoint":raw_subtract_mean_each_timepoint,
            "EVENT":EVENT,
            "dim_red_method":dim_red_method,
            "NPCS_KEEP":NPCS_KEEP,
            "n_splits_tot":n_splits_tot,
            "n_splits":n_splits,
            "PLOT_EACH_FOLD":PLOT_EACH_FOLD,
            "n_flank":n_flank,
        }

        from pythonlib.tools.expttools import writeDictToTxtFlattened
        writeDictToTxtFlattened(params, f"{SAVEDIR}/params.yaml")

        res_neural = []
        res_strokesbeh = []
        res_strokestask = []
        morphsets_done = []

        list_bregion = DFallpa["bregion"].unique().tolist()
        list_morphset = params_good_morphsets_no_switching(animal, date)
        for bregion in list_bregion:
            for morphset in list_morphset:

                savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"
                os.makedirs(savedir, exist_ok=True)

                _, PAredu = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                            scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                    dim_red_method, proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP, 
                                    savedir, restricted_twind_for_dpca=restricted_twind_for_dpca, exclude_flankers=exclude_flankers)
                
                # (2) Behavior
                version = "beh"
                PAstrokebeh = PAredu.behavior_replace_neural_with_strokes(version)

                # (3) Task
                version = "task"
                PAstroketask = PAredu.behavior_replace_neural_with_strokes(version)
                
                ######## save paredu
                import pickle
                with open(f"{savedir}/PAredu.pkl", "wb") as f:
                    pickle.dump(PAredu, f)

                # Only run beh and task one time per morphset
                if morphset not in morphsets_done:
                    LIST_CONDITIONS = [
                        (PAredu, "neural", res_neural, twind_score, n_splits_tot),
                        (PAstrokebeh, "stkbeh", res_strokesbeh, None, n_splits_tot),
                        (PAstroketask, "stktask", res_strokestask, None, 2),
                        ]

                    # # Skip task for now, it is bad.
                    # LIST_CONDITIONS = [
                    #     (PAredu, "neural", res_neural, twind_score, n_splits_tot),
                    #     (PAstrokebeh, "stkbeh", res_strokesbeh, None, n_splits_tot),
                    #     ]
                else:
                    LIST_CONDITIONS = [
                        (PAredu, "neural", res_neural, twind_score, n_splits_tot),
                        ]
                morphsets_done.append(morphset)

                for PA, savename, res, twind_score_this, n_splits_tot_this in LIST_CONDITIONS:

                    savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}/{savename}"
                    os.makedirs(savedir, exist_ok=True)

                    ######################################## STATE SPACE TRAJECTORIES
                    # Plot state space trajectories.
                    savedir_statespace = f"{savedir}/STATE_SPACE"
                    os.makedirs(savedir_statespace, exist_ok=True)
                    LIST_VAR = [
                        "idx_morph_temp",
                    ]
                    LIST_VARS_OTHERS = [
                        ("task_kind",),
                    ]
                    LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]
                    nmin_trials_per_lev = 1
                    list_dim_timecourse = [0,1,2,3,4,5, 6, 7]
                    list_dims = [(0,1), (2,3), (4,5), (6,7)]
                    PA.plot_state_space_good_wrapper(savedir_statespace, LIST_VAR, LIST_VARS_OTHERS, 
                                                        LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS, nmin_trials_per_lev=nmin_trials_per_lev,
                                                        list_dim_timecourse=list_dim_timecourse, list_dims=list_dims)

                    ########################################
                    ##### (1) Plot over time
                    ### Get single trial pairwise distances over time.
                    var_effect = "idx_morph_temp"
                    effect_lev_base1 = 0
                    effect_lev_base2 = 99
                    list_grps_get = [(0,), (99,)] # This is important, or else will fail if there are any (idx|assign) with only one datapt.

                    # (A) plot over time, including negetive times.
                    version = "pts_time"    
                    DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS = PA.dataextract_as_distance_index_between_two_base_classes(var_effect, 
                                                                                                        effect_lev_base1, effect_lev_base2, list_grps_get=list_grps_get,
                                                                                                        version = version)
                    DFPROJ_INDEX["idx_morph_temp_str"] = DFPROJ_INDEX["idx_morph_temp"].astype(str) # So it plots as categorical, not numeric.
                    hue_order = [str(x) for x in sorted(DFPROJ_INDEX["idx_morph_temp"].unique())]
                    fig = sns.relplot(data=DFPROJ_INDEX, x="time_bin", y="dist_index", kind="line", hue="idx_morph_temp_str", 
                                    hue_order = hue_order, errorbar=("ci", 68))
                    savefig(fig, f"{savedir}/DFPROJ_INDEX-time.pdf")

                    # (B) Average over time, post-samp
                    if twind_score_this is not None:
                        pa = PA.slice_by_dim_values_wrapper("times", twind_score_this)
                    else:
                        pa = PA.copy()
                    version = "pts_scal"
                    DFPROJ_INDEX, DFDIST, DFPROJ_INDEX_AGG, DFDIST_AGG, DFPROJ_INDEX_DIFFS = pa.dataextract_as_distance_index_between_two_base_classes(var_effect, 
                                                                                                        effect_lev_base1, effect_lev_base2, list_grps_get=list_grps_get,
                                                                                                        version = version)
                    fig = sns.catplot(data=DFPROJ_INDEX, x="idx_morph_temp", y="dist_index", kind="point", errorbar=("ci", 68))
                    savefig(fig, f"{savedir}/DFPROJ_INDEX-scal-twind={twind_score_this}.pdf")

                    fig = sns.catplot(data=DFPROJ_INDEX_DIFFS, x="idx_along_morph", y="dist", kind="point", errorbar=("ci", 68))
                    savefig(fig, f"{savedir}/DFPROJ_INDEX_DIFFS-scal-twind={twind_score_this}.pdf")

                    plt.close("all")


                    ######################### 
                    ### SCORE, over entire data, not train-test split.

                    ######################################
                    # (2) Score, using train-test split
                    from pythonlib.tools.statstools import balanced_stratified_kfold
                    from pythonlib.tools.pandastools import append_col_with_grp_index
                    from neuralmonkey.scripts.analy_decode_moment_psychometric import _recenter_idx_general

                    # Params
                    plot_train_test_scores = False
                    var_exclude = "idx_morph_temp"
                    levels_exclude = [0, 99]
                    label_grp_vars = ["idx_morph_temp", "seqc_0_loc"]

                    ### Get splits
                    # Repeat it, to get many folds
                    folds_dflab = []
                    while len(folds_dflab)<n_splits_tot_this:
                        _folds_dflab = PA.split_balanced_stratified_kfold_subsample_level_of_var(label_grp_vars, var_exclude, levels_exclude,
                                                                n_splits=n_splits, do_balancing_of_train_inds=False, plot_train_test_counts=False,
                                                                shuffle=True)
                        folds_dflab.extend(_folds_dflab)

                    ################ SCORE
                    if twind_score_this is not None:
                        PA_twind = PA.slice_by_dim_values_wrapper("times", twind_score_this)
                    else:
                        PA_twind = PA.copy()

                    for i, (train_inds, test_inds) in enumerate(folds_dflab):
                        paredu_train = PA_twind.slice_by_dim_indices_wrapper("trials", train_inds)
                        paredu_test = PA_twind.slice_by_dim_indices_wrapper("trials", test_inds)

                        # Get scores, for train and test
                        version = "grps_scal"
                        _, _, _, _, df_train = paredu_train.dataextract_as_distance_index_between_two_base_classes(var_effect, 
                                                                                                            effect_lev_base1, effect_lev_base2, list_grps_get=list_grps_get,
                                                                                                            version = version, PLOT=plot_train_test_scores)
                        _, _, _, _, df_test = paredu_test.dataextract_as_distance_index_between_two_base_classes(var_effect, 
                                                                                                            effect_lev_base1, effect_lev_base2, list_grps_get=list_grps_get,
                                                                                                            version = version, PLOT=plot_train_test_scores)

                        if isinstance(PLOT_EACH_FOLD, int) and i<PLOT_EACH_FOLD:
                            # Then plot the first n fold
                            _doplot= True
                        elif PLOT_EACH_FOLD==True:
                            _doplot= True
                        else:
                            _doplot= False

                        if _doplot:
                            fig = sns.catplot(data=df_train, x="idx_along_morph", y="dist", kind="point")
                            savefig(fig, f"{savedir}/DFPROJ_INDEX_DIFFS_train-scal-fold-{i}.pdf")
                            fig = sns.catplot(data=df_test, x="idx_along_morph", y="dist", kind="point")
                            savefig(fig, f"{savedir}/DFPROJ_INDEX_DIFFS_test-scal-fold-{i}.pdf")

                        ##################################
                        # Get value of min and max            

                        # this makes downstream indexing easeir.
                        df_train = df_train.sort_values("idx_along_morph").reset_index(drop=True)
                        df_test = df_test.sort_values("idx_along_morph").reset_index(drop=True)
                        assert np.all(df_train["idx_along_morph"] == df_test["idx_along_morph"]), "This must be. otherwise therw was mistake in train-test split"

                        _, indrow_min, indrow_max = _recenter_idx_general(df_train, None)

                        if True:
                            # This gets max +/1 n_flank, to reduce noise. 
                            # get flanking
                            if n_flank>indrow_max:
                                i1 = 0
                            else:
                                i1 = indrow_max-n_flank
                            i2 = indrow_max+n_flank+1
                            # display(df_test.iloc[i1:i2])
                            # display(df_test.iloc[:i1])
                            # display(df_test.iloc[i2:])

                            val_max_single = df_test.iloc[indrow_max]["dist"]
                            val_max = np.mean(df_test.iloc[i1:i2]["dist"])
                            val_min = df_test.iloc[indrow_min]["dist"]
                            val_mean_notmax = np.mean(np.concatenate([df_test.iloc[:i1]["dist"], df_test.iloc[i2:]["dist"]]))
                            val_mean = np.mean(df_test["dist"])
                            # print(val_min, val_mean_notmax, val_max)
                        else:
                            # Old./ still works, but not coded to do this: take n_flank.
                            def _get_value_in_test(indrow_in_train):
                                # Get the value, in test df, for index that matches the train df, along var idx_along_morph, 
                                # taking the train df values at row index indrow_in_train.
                                idx_along_morph = df_train.iloc[indrow_in_train]["idx_along_morph"]
                                tmp = df_test[df_test["idx_along_morph"]==idx_along_morph]
                                assert len(tmp)==1
                                return tmp["dist"].values[0]

                            val_min = _get_value_in_test(indrow_min)
                            val_max = _get_value_in_test(indrow_max)

                            # The man value across all indices that are NOT indrow_max
                            val_mean_notmax = np.mean(df_test[~(df_test["idx_along_morph"]==df_train.iloc[indrow_max]["idx_along_morph"])]["dist"])
                            print(val_min, val_mean_notmax, val_max)

                        ### Another meothd -- (min - max - min)
                        # Also use method that forces to get max flanked to left and right by minima
                        _, _, indrow_max = _recenter_idx_general(df_train, None, exclude_n_at_edge=1)
                        indrow_min_left = df_train.iloc[:indrow_max]["dist"].idxmin()
                        indrow_min_right = df_train.iloc[indrow_max+1:]["dist"].idxmin()
                        
                        val_max_middle = df_test.iloc[indrow_max]["dist"]
                        val_min_left = df_test.iloc[indrow_min_left]["dist"]
                        val_min_right = df_test.iloc[indrow_min_right]["dist"]

                        res.append({
                            "bregion":bregion,
                            "morphset":morphset,
                            "i_fold":i,
                            "idx_along_morph_max":df_train.iloc[indrow_max]["idx_along_morph"],
                            "idx_along_morph_min":df_train.iloc[indrow_min]["idx_along_morph"],
                            "indrow_max":indrow_max,
                            "indrow_min":indrow_min,
                            "dist_index_diff_min":val_min, 
                            "dist_index_diff_max":val_max, 
                            "dist_index_diff_max_single":val_max_single, 
                            "dist_index_diff_mean_notmax":val_mean_notmax, 
                            "dist_index_diff_mean":val_mean, 
                            "dist_index_diff_v2_max":val_max_middle, 
                            "dist_index_diff_v2_minleft":val_min_left, 
                            "dist_index_diff_v2_minright":val_min_right, 
                            "indices_v2":(indrow_min_left, indrow_max, indrow_min_right)
                        })

                        plt.close("all")

        
        DF_DISTIDX_SCORE_SPLITS = pd.DataFrame(res_neural)
        DF_DISTIDX_SCORE_SPLITS_STKBEH = pd.DataFrame(res_strokesbeh)
        DF_DISTIDX_SCORE_SPLITS_STKTASK = pd.DataFrame(res_strokestask)

        # Final metrics
        # DF_DISTIDX_SCORE = aggregGeneral(DF_DISTIDX_SCORE_SPLITS, ["bregion", "morphset"], ["dist_index_diff_min", "dist_index_diff_max", "dist_index_diff_mean_notmax", "dist_index_diff_max_single"])
        # DF_DISTIDX_SCORE["maxsingle_minus_min"] = DF_DISTIDX_SCORE["dist_index_diff_max_single"] - DF_DISTIDX_SCORE["dist_index_diff_min"]
        # DF_DISTIDX_SCORE["maxsingle_minus_mean"] = DF_DISTIDX_SCORE["dist_index_diff_max_single"] - DF_DISTIDX_SCORE["dist_index_diff_mean_notmax"]
        # DF_DISTIDX_SCORE["max_minus_min"] = DF_DISTIDX_SCORE["dist_index_diff_max"] - DF_DISTIDX_SCORE["dist_index_diff_min"]
        # DF_DISTIDX_SCORE["max_minus_mean"] = DF_DISTIDX_SCORE["dist_index_diff_max"] - DF_DISTIDX_SCORE["dist_index_diff_mean_notmax"]

        ############## SAVE DATA
        pd.to_pickle(DF_DISTIDX_SCORE_SPLITS, f"{SAVEDIR}/DF_DISTIDX_SCORE_SPLITS.pkl")
        pd.to_pickle(DF_DISTIDX_SCORE_SPLITS_STKBEH, f"{SAVEDIR}/DF_DISTIDX_SCORE_SPLITS_STKBEH.pkl")
        pd.to_pickle(DF_DISTIDX_SCORE_SPLITS_STKTASK, f"{SAVEDIR}/DF_DISTIDX_SCORE_SPLITS_STKTASK.pkl")
        # pd.to_pickle(DF_DISTIDX_SCORE, f"{SAVEDIR}/DF_DISTIDX_SCORE.pkl")

        for dfthis_splits, savepref in [
            (DF_DISTIDX_SCORE_SPLITS, "neural"),
            (DF_DISTIDX_SCORE_SPLITS_STKBEH, "stkbeh"),
            (DF_DISTIDX_SCORE_SPLITS_STKTASK, "stktask")
            ]:

            dfthis = aggregGeneral(dfthis_splits, ["bregion", "morphset"], ["dist_index_diff_min", "dist_index_diff_max", 
                                                                            "dist_index_diff_mean_notmax", "dist_index_diff_max_single",
                                                                            "dist_index_diff_v2_max", "dist_index_diff_v2_minleft", "dist_index_diff_v2_minright"])
            dfthis["maxsingle_minus_min"] = dfthis["dist_index_diff_max_single"] - dfthis["dist_index_diff_min"]
            dfthis["maxsingle_minus_mean"] = dfthis["dist_index_diff_max_single"] - dfthis["dist_index_diff_mean_notmax"]
            dfthis["max_minus_min"] = dfthis["dist_index_diff_max"] - dfthis["dist_index_diff_min"]
            dfthis["max_minus_mean"] = dfthis["dist_index_diff_max"] - dfthis["dist_index_diff_mean_notmax"]

            dfthis["max_minus_min_v2"] = dfthis["dist_index_diff_v2_max"] - 0.5*(dfthis["dist_index_diff_v2_minleft"] + dfthis["dist_index_diff_v2_minright"])

            ############### PLOTS 
            savedir = f"{SAVEDIR}/combined_plots-{savepref}"
            os.makedirs(savedir, exist_ok=True)

            # agg over folds
            # for yvar in ["max_minus_mean", "max_minus_min", "maxsingle_minus_mean", "maxsingle_minus_min", 
            #              "dist_index_diff_max", "dist_index_diff_min", "dist_index_diff_mean_notmax"]:
            for yvar in ["max_minus_mean", "max_minus_min", "dist_index_diff_max", "dist_index_diff_min", "dist_index_diff_mean_notmax", "max_minus_min_v2"]:
                fig = sns.catplot(data=dfthis, x="bregion", y=yvar, kind="bar", errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.25)
                savefig(fig, f"{savedir}/{yvar}-1.pdf")
                
                fig = sns.catplot(data=dfthis, x="bregion", y=yvar, hue="morphset", kind="point", errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.25)
                savefig(fig, f"{savedir}/{yvar}-2.pdf")

                fig = sns.catplot(data=dfthis, x="morphset", y=yvar, hue="bregion", kind="point", errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.25)
                savefig(fig, f"{savedir}/{yvar}-3.pdf")

                plt.close("all")

def analy_psychoprim_statespace_euclidian_score_and_plot(DFallpa, SAVEDIR_BASE, map_tcmorphset_to_idxmorph, 
                                                         animal, date, list_morphset):
    """
    This gets derived metrics, such as distance index along morphset, and makes plot.
    
    This does in one step what was previously:
    1. analy_psychoprim_statespace_euclidian
    2. analy_decode_moment_psychometric_mult.py (which loads results from analy_psychoprim_statespace_euclidian).
    (Here, does those two steps just for a single day. Still need to collate across days and plot)

    """
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER

    list_bregion = DFallpa["bregion"].unique().tolist()
    
    EVENT = "03_samp"
    # EVENT = "06_on_strokeidx_0"
    exclude_flankers = True
    n_flank_boundary = 2

    # Extract data for this morphset, wiht labels updated
    for raw_subtract_mean_each_timepoint in [False]:
    # for raw_subtract_mean_each_timepoint in [False, True]:
        # for scalar_or_traj in ["traj", "scal"]:
        for scalar_or_traj in ["traj"]:
            if EVENT == "03_samp" and scalar_or_traj=="traj":
                list_npcs_keep = [10, 7, 4]
                list_twind = [(0.1, 1.)]
            elif EVENT == "03_samp" and scalar_or_traj=="scal":
                list_npcs_keep = [10, 7, 4]
                list_twind = [(0.1, 1.)]
            else:
                assert False

            if scalar_or_traj == "scal":
                tbin_dur = 0.15
                pca_tbin_slice = 0.15
            elif scalar_or_traj == "traj":
                tbin_dur = 0.1
                pca_tbin_slice = 0.1
            else:
                assert False
            
            for proj_twind in list_twind:
                for dim_red_method in ["dpca"]:
                    for NPCS_KEEP in list_npcs_keep:
                        ### Run
                        SAVEDIR = f"{SAVEDIR_BASE}/euclidian_score_and_plot/ev={EVENT}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={proj_twind}-npcs={NPCS_KEEP}-flank={exclude_flankers}-cat_n_flank={n_flank_boundary}"
                        list_dfproj = []
                        list_dfdiffs = []
                        list_dfproj_index = []
                        list_dfdiffsproj = []
                        list_dfdiffsindex = []
                        list_dfdiffscateg = []
                        for bregion in list_bregion:
                            for morphset in list_morphset:

                                savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"
                                os.makedirs(savedir, exist_ok=True)

                                _, PAredu = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                                            scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                                    dim_red_method, proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP, 
                                                    savedir)
                                
                                ### Analysis -- get all distance scores.
                                dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex, dfdiffs_categ = analy_morphsmooth_euclidian_score(PAredu, 
                                                                                                                                savedir, 
                                                                                                                                exclude_flankers,
                                                                                                                                morphset, n_flank_boundary,
                                                                                                                                DOPLOTS=True)
                                for _dftmp in [dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex, dfdiffs_categ]:
                                    _dftmp["animal"] = animal
                                    _dftmp["date"] = date
                                    _dftmp["morphset"] = morphset
                                    _dftmp["bregion"] = bregion

                                list_dfproj.append(dfproj)
                                list_dfproj_index.append(dfproj_index)

                                list_dfdiffs.append(dfdiffs)
                                list_dfdiffsproj.append(dfdiffsproj)
                                list_dfdiffsindex.append(dfdiffsindex)
                                list_dfdiffscateg.append(dfdiffs_categ)

                        ### Collect across regions and morphsets
                        DFPROJ = pd.concat(list_dfproj).reset_index(drop=True)
                        DFPROJ_INDEX = pd.concat(list_dfproj_index).reset_index(drop=True)
                        DFDIFFS = pd.concat(list_dfdiffs).reset_index(drop=True)
                        DFDIFFSPROJ = pd.concat(list_dfdiffsproj).reset_index(drop=True)
                        DFDIFFSINDEX = pd.concat(list_dfdiffsindex).reset_index(drop=True)
                        DFDIFFSCATEG = pd.concat(list_dfdiffscateg).reset_index(drop=True)
                
                        # SAVE:
                        pd.to_pickle(DFPROJ, f"{SAVEDIR}/DFPROJ.pkl")
                        pd.to_pickle(DFPROJ_INDEX, f"{SAVEDIR}/DFPROJ_INDEX.pkl")
                        pd.to_pickle(DFDIFFS, f"{SAVEDIR}/DFDIFFS.pkl")
                        pd.to_pickle(DFDIFFSPROJ, f"{SAVEDIR}/DFDIFFSPROJ.pkl")
                        pd.to_pickle(DFDIFFSINDEX, f"{SAVEDIR}/DFDIFFSINDEX.pkl")
                        pd.to_pickle(DFDIFFSCATEG, f"{SAVEDIR}/DFDIFFSCATEG.pkl")

                        ### PLOT
                        savedir = f"{SAVEDIR}/plots_combined"
                        os.makedirs(savedir, exist_ok=True)
                        from neuralmonkey.scripts.analy_decode_moment_psychometric_mult import plot_with_preprocess
                        list_thresh_separation = [0.] # Plot all data, since this is just a single (animal, date)
                        plot_with_preprocess(DFPROJ, DFPROJ_INDEX, DFDIFFS, DFDIFFSPROJ, DFDIFFSINDEX, DFDIFFSCATEG, savedir,
                                            list_thresh_separation=list_thresh_separation)


def analy_switching_statespace_euclidian_score_and_plot(DFallpa, SAVEDIR_BASE, map_tcmorphset_to_idxmorph, 
                                                        list_morphset, map_tcmorphset_to_info, HACK=False):
    """
    Srapper for all good plots for switchign, whcih use state space and eucldian ditsances. Complemente previous stuff using decode.
    """
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER

    list_bregion = DFallpa["bregion"].unique().tolist()
    
    EVENT = "03_samp"
    # EVENT = "06_on_strokeidx_0"

    ########################### SCORING DIST_INDEX (vs base1/base2), TIMECOURSE AND SINGLE-TRIAL.
    scalar_or_traj = "traj"
    tbin_dur = 0.1
    pca_tbin_slice = 0.1
    proj_twind = (-0.1, 1.1)
    restricted_twind_for_dpca = (0.1, 1.1)

    raw_subtract_mean_each_timepoint = False
    EVENT = "03_samp"
    dim_red_method = "dpca"
    NPCS_KEEP = 7
    
    if HACK==False: # Hacky, skip for a specific run
        for bregion in list_bregion:
            for morphset in list_morphset:

                # scalar_or_traj = "scal"
                # tbin_dur = 0.15
                # pca_tbin_slice = 0.15
                # proj_twind = (0.01, 1.2)


                SAVEDIR = f"{SAVEDIR_BASE}/switching_euclidian_score_and_plot_traj/ev={EVENT}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={proj_twind}-npcs={NPCS_KEEP}"
                savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"

                _, PAredu = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                            scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                    dim_red_method, proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP, 
                                    savedir, restricted_twind_for_dpca=restricted_twind_for_dpca)

                ### Analysus -- single trials, adn timecourses
                analy_switching_statespace_euclidian_traj(PAredu, savedir)

    ########################### STATE SPACE PLOTS (all, good), AND VARIOUS METRICS
    # Extract data for this morphset, wiht labels updated
    
    list_npcs_keep = [NPCS_KEEP]
    
    # list_npcs_keep = [7, 10, 4]
    # list_twind = [(0.1, 1.2)]
    
    # if EVENT == "03_samp" and scalar_or_traj=="traj":
    #     list_npcs_keep = [7, 10, 4]
    #     list_twind = [(0.1, 1.2)]
    # elif EVENT == "03_samp" and scalar_or_traj=="scal":
    #     list_npcs_keep = [7, 10, 4]
    #     list_twind = [(0.1, 1.2)]
    # else:
    #     assert False

    for raw_subtract_mean_each_timepoint in [False]:
    # for raw_subtract_mean_each_timepoint in [False, True]:
        for NPCS_KEEP in list_npcs_keep:
            for scalar_or_traj in ["traj", "scal"]:
                
                if scalar_or_traj=="traj":
                    # Then can take longer time window
                    list_twind = [proj_twind]
                    tbin_dur = 0.1
                    pca_tbin_slice = 0.05
                elif scalar_or_traj=="scal":
                    # Then must use the actual data time window.
                    list_twind = [restricted_twind_for_dpca]
                    tbin_dur = 0.15
                    pca_tbin_slice = 0.15
                else:
                    assert False

                for proj_twind in list_twind:
                    for dim_red_method in ["dpca"]:
                            ### Run
                            SAVEDIR = f"{SAVEDIR_BASE}/switching_euclidian_score_and_plot/ev={EVENT}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={proj_twind}-npcs={NPCS_KEEP}"

                            for bregion in list_bregion:
                                for morphset in list_morphset:

                                    savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"
                                    os.makedirs(savedir, exist_ok=True)

                                    _, PAredu = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                                                scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                                        dim_red_method, proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP, 
                                                        savedir, restricted_twind_for_dpca=restricted_twind_for_dpca)
                                    
                                    ### Analysis -- get all distance scores.
                                    analy_switching_statespace_euclidian_good(PAredu, savedir)


def analy_switching_GOOD_stats_linear_2br_compute(DFPROJ_INDEX_AGG_DIFF_SCAL, savedir):
    """

    Considers each row of DFPROJ_INDEX_AGG_DIFF_SCAL as a datapt. Each is expeted to be
    a single (ani, date, ms). -- ie this combines all data across experiments. 

    Then gets paired effect (conditions on ani_date_ms) for each pair of regions.

    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    import statsmodels.formula.api as smf

    label_lev = "ambig"
    yvar = "dist_index_diff"
    list_bregion = DFPROJ_INDEX_AGG_DIFF_SCAL["bregion"].unique().tolist()
    
    ### Collect data across all pairs of bregions
    res = []
    for i in range(len(list_bregion)):
        for j in range(len(list_bregion)):
            if j>i:
                bregion1 = list_bregion[i]
                bregion2 = list_bregion[j]
                
                dflm = DFPROJ_INDEX_AGG_DIFF_SCAL[
                    (DFPROJ_INDEX_AGG_DIFF_SCAL["label"]==label_lev) & (DFPROJ_INDEX_AGG_DIFF_SCAL["bregion"].isin([bregion1, bregion2]))
                    ].reset_index(drop=True)

                formula = f"{yvar} ~ C(bregion, Treatment('{bregion1}')) + C(an_da_ms)"
                md = smf.ols(formula, dflm)
                mdf = md.fit()
                mdf.summary()
                
                coefficients = mdf.params
                dfcoeff = coefficients.reset_index()
                dfpvals = mdf.pvalues.reset_index()

                coeffname = f"C(bregion, Treatment('{bregion1}'))[T.{bregion2}]"
                assert dfcoeff.iloc[1]["index"] == coeffname
                assert dfpvals.iloc[1]["index"] == coeffname

                res.append({
                    "results":mdf,
                    "coeffname":coeffname,
                    "coeff_val":dfcoeff.iloc[1][0],
                    "pval":dfpvals.iloc[1][0],
                    "label":label_lev,
                    "bregion1":bregion1,
                    "bregion2":bregion2,
                    "formula":formula,
                })
    DFSTATS_2BR = pd.DataFrame(res)
    DFSTATS_2BR["pval_log10"] = np.log10(DFSTATS_2BR["pval"])

    # Make a mirror image
    dftmp = DFSTATS_2BR.copy()
    dftmp["bregion1"] = DFSTATS_2BR["bregion2"]
    dftmp["bregion2"] = DFSTATS_2BR["bregion1"]
    dftmp["coeff_val"] = -dftmp["coeff_val"] 
    dftmp["coeffname"] = "ignore"
    dftmp["formula"] = "ignore"
    DFSTATS_2BR = pd.concat([DFSTATS_2BR, dftmp]).reset_index(drop=True)

    ### Plot
    z = np.max(np.abs(np.percentile(DFSTATS_2BR["coeff_val"], [0.5, 99.5])))
    ZLIMS = [-z, z]

    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    from math import factorial, comb

    var_same_same = None
    grp = None
    order_bregion = _REGIONS_IN_ORDER_COMBINED
    npairs = comb(len(order_bregion), 2)
    ncomp = 2
    alpha=0.05
    alpha_bonf_easy = alpha/(npairs)
    alpha_bonf_hard = alpha/(npairs * ncomp)

    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values

    dfstats = DFSTATS_2BR.copy()

    fig, axes = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "coeff_val", var_same_same, 
                                True, True, None, True, W=6, ZLIMS=ZLIMS, row_values=order_bregion, col_values=order_bregion)
    savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}.pdf")

    zlims = [-5, 0]
    fig, _ = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "pval_log10", var_same_same, 
                                False, True, None, True, W=6, ZLIMS=zlims, row_values=order_bregion, col_values=order_bregion)
    savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}-pvals.pdf")

    zlims = [np.log10(alpha_bonf_hard)-3, np.log10(alpha_bonf_easy)]
    fig, _ = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "pval_log10", var_same_same, 
                                False, True, None, True, W=6, ZLIMS=zlims, row_values=order_bregion, col_values=order_bregion)
    savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}-pvals_bonfeasy.pdf")

    zlims = [np.log10(alpha_bonf_hard)-3, np.log10(alpha_bonf_hard)]
    fig, _ = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "pval_log10", var_same_same, 
                                False, True, None, True, W=6, ZLIMS=zlims, row_values=order_bregion, col_values=order_bregion)
    savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}-pvals_bonfhard.pdf")

    fig = sns.catplot(data=dfstats, x="bregion1", y="coeff_val", hue=var_same_same, col="bregion2", 
                col_order=order_bregion, order=order_bregion, col_wrap=6, kind="bar")
    savefig(fig, f"{savedir}/COMPARE_AREAS-catplot-grp={grp}.pdf")

    fig = sns.catplot(data=dfstats, x="bregion1", y="pval_log10", hue=var_same_same, col="bregion2", 
                col_order=order_bregion, order=order_bregion, col_wrap=6, kind="bar")
    for ax in fig.axes:
        ax.axhline(np.log10(0.05))
        ax.axhline(np.log10(0.005))
        ax.axhline(np.log10(0.0005))
        ax.axhline(np.log10(alpha_bonf_easy), color="r")
        ax.axhline(np.log10(alpha_bonf_hard), color="r")
        ax.set_ylim(bottom=-8)
    savefig(fig, f"{savedir}/COMPARE_AREAS-catplot-grp={grp}-pvalues.pdf")

    plt.close("all")


def analy_switching_GOOD_euclidian_index(DFallpa, SAVEDIR_BASE, map_tcmorphset_to_idxmorph, 
                                                        list_morphset, map_tcmorphset_to_info,
                                                        make_plots=True, save_df=True,
                                                        DO_RSA_HEATMAPS=False, var_context_diff="seqc_0_loc",
                                                        do_train_test_splits=True):
    """
    Good--final plots of distance index (i.e,, relative distance from 0 and 99) but clean.
    i.e., is like analy_switching_statespace_euclidian_traj, but the follwoing:
    - train-test splits for dim reduction, otherwise there is overfitting.
    - using different twinds (smaller) for fitting subspace compared to the entire data window.
    
    PARAMS:
    - var_context_diff, if None ignore, otherwise is string variable. Theneucl distance will be taken between pairs of datapts that
    are different for this variable. (e.g., seqc_0_loc, then needs to generalize across locations)
    - do_train_test_splits, should leave True, as if not, then there is bias. Note that this is (probably) true even if you
    have something for var_context_diff.
    """
    
    twind_scal_rsa = (0.6, 1.0) # LATE

    var_effect = "idxmorph_assigned"
    effect_lev_base1 = "0|base1"
    effect_lev_base2 = "99|base2"

    # (1) Split into two
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _preprocess_pa_dim_reduction
    assert False, "previously (when Pancho PMVl was not included) I used NPCS_KEEP=8. Then I used NPCS_KEEP=6 for the example plot Pancho morphset 8. Decide which to use. Prob use 6. I checked and they are pretty much identical"
    NPCS_KEEP = 6
    scalar_or_traj = "traj"
    tbin_dur = 0.2
    tbin_slide = 0.02
    raw_subtract_mean_each_timepoint = False

    fit_twind = (0.05, 0.9)
    final_twind = (-0.3, 1.2)
    superv_dpca_var = "idx_morph_temp_loc"
    superv_dpca_vars_group = None
    dim_red_method = "dpca"
    EVENT = "03_samp"
    
    exclude_flankers = True

    if do_train_test_splits:
        # N_SPLITS_OUTER = 6 # 2 is fine, but 6 is prob better, might be high variance.
        # N_SPLITS_INNER = 2
        N_SPLITS_OUTER = 1
        N_SPLITS_INNER = 8
    else:

        N_SPLITS_OUTER = 1
        N_SPLITS_INNER = 1
    
    ########################### STATE SPACE PLOTS (all, good), AND VARIOUS METRICS
    # Extract data for this morphset, wiht labels updated

    list_bregion = DFallpa["bregion"].unique().tolist()

    ### Run
    SAVEDIR = f"{SAVEDIR_BASE}/analy_switching_GOOD_euclidian_index/ev={EVENT}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={final_twind}-npcs={NPCS_KEEP}"

    for bregion in list_bregion:
        for morphset in list_morphset:
            
            print(list_morphset, morphset)
            savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"
            os.makedirs(savedir, exist_ok=True)
            print("Saving to:", savedir)

            if DO_RSA_HEATMAPS:
                # Plot pairwise distances (rsa heatmaps).
                # This is done separatee to below becuase it doesnt use the train-test splits.
                # It shold but I would have to code way to merge multple Cl, which is doable.
                from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

                _, PAthis = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                            scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                                            dim_red_method, final_twind, tbin_dur, tbin_slide, NPCS_KEEP,
                                                            None, restricted_twind_for_dpca=fit_twind, exclude_flankers=exclude_flankers,
                                                            skip_dim_redu=False)
                savedirthis = f"{savedir}/rsa_heatmap/twindscal={twind_scal_rsa}"
                os.makedirs(savedirthis, exist_ok=True)

                # Prune to scalar window
                PAthis = PAthis.slice_by_dim_values_wrapper("times", twind_scal_rsa)

                # Make rsa heatmaps.
                vars_group = ["idxmorph_assigned"]
                timevarying_compute_fast_to_scalar(PAthis, vars_group, rsa_heatmap_savedir=savedirthis)


            ### Extract data (without dim redu)
            _, PA = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                        scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                                        None, None, None, None, NPCS_KEEP,
                                                        savedir, exclude_flankers=exclude_flankers,
                                                        skip_dim_redu=True)
            TRIALCODES_ORDERED = PA.Xlabels["trials"]["trialcode"].tolist()

            ### Iter over splits, each time collecting and scoring.
            list_dfindex =[]
            list_dfdist = []
            already_plotted_dimredu =False
            for i_outer in range(N_SPLITS_OUTER):
                
                # For train-test split, make sure all final cases are present (e.g., ambig)
                if var_context_diff is not None:
                    _vars_grp = [var_effect, var_context_diff]
                else:
                    _vars_grp = [var_effect]

                if N_SPLITS_INNER==1:
                    # Then dont split. Put all the trials into the inds.
                    _indsall = list(range(len(PA.Trials)))
                    folds_dflab = [(_indsall, _indsall)]
                else:
                    if False:
                        # Old, which can fail by not including the required data
                        # vars_group = ["idx_morph_temp"]
                        folds_dflab = PA.split_balanced_stratified_kfold_subsample_level_of_var(_vars_grp, None, None, 
                                                                                                    n_splits=N_SPLITS_INNER, 
                                                                                                    do_balancing_of_train_inds=False, 
                                                                                                    shuffle=True)
                    else:
                        # Better, more careful, ensuring enough data for euclidian distance.
                        fraction_constrained_set=0.75
                        n_constrained = 3 # Ideally have more than 1 pair

                        # - must get these
                        dflab = PA.Xlabels["trials"]
                        if var_context_diff is not None:
                            a = [(effect_lev_base1, _lev) for _lev in dflab[var_context_diff].unique().tolist()]
                            b = [(effect_lev_base2, _lev) for _lev in dflab[var_context_diff].unique().tolist()]
                            list_labels_need_n = a + b # Need the endpoints
                        else:
                            list_labels_need_n = [effect_lev_base1, effect_lev_base2]

                        min_frac_datapts_unconstrained=None
                        plot_train_test_counts=True
                        plot_indices=False
                        from pythonlib.tools.exceptions import NotEnoughDataException
                        try:
                            min_n_datapts_unconstrained=len(PA.Xlabels["trials"][superv_dpca_var].unique())
                            folds_dflab, fig_unc, fig_con = PA.split_stratified_constrained_grp_var(N_SPLITS_INNER, _vars_grp, 
                                                                            fraction_constrained_set, n_constrained, 
                                                                            list_labels_need_n, min_frac_datapts_unconstrained,  
                                                                            min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
                        except NotEnoughDataException as err:
                            # Try, more lenient
                            min_n_datapts_unconstrained=int(len(PA.Xlabels["trials"][superv_dpca_var].unique())/2)
                            folds_dflab, fig_unc, fig_con = PA.split_stratified_constrained_grp_var(N_SPLITS_INNER, _vars_grp, 
                                                                            fraction_constrained_set, n_constrained, 
                                                                            list_labels_need_n, min_frac_datapts_unconstrained,  
                                                                            min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
                        except Exception as err:
                            raise err
                            
                        savefig(fig_con, f"{savedir}/after_split_constrained_fold_0.pdf")
                        savefig(fig_unc, f"{savedir}/after_split_unconstrained_fold_0.pdf")
                        plt.close("all")

                for i_dimredu, (train_inds, test_inds) in enumerate(folds_dflab):
                    print(f"...splits, i_outer={i_outer}, i_inner={i_dimredu}")
                    # train_inds, more inds than than test_inds
                    inds_pa_fit = [int(i) for i in train_inds] # 
                    inds_pa_final = [int(i) for i in test_inds] # each ind occurs only once

                    savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}/preprocess/i_outer={i_outer}-i_dimredu={i_dimredu}"
                    os.makedirs(savedir, exist_ok=True)

                    if already_plotted_dimredu:
                        savedir_this = None
                    else:
                        savedir_this = savedir
                        already_plotted_dimredu = True
                    _, PAredu = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir_this, 
                                                    fit_twind, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
                                                    NPCS_KEEP = NPCS_KEEP,
                                                    dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, 
                                                    dpca_proj_twind = final_twind, 
                                                    raw_subtract_mean_each_timepoint=False,
                                                    inds_pa_fit=inds_pa_fit, inds_pa_final=inds_pa_final,
                                                    n_min_per_lev_lev_others=1)

                    # (2) For each split, get a dist_index score.
                    from neuralmonkey.scripts.analy_decode_moment_psychometric import _compute_df_using_dist_index_traj
                    ########################################
                    ### Get single trial pairwise distances over time.
                    list_grps_get = [(effect_lev_base1,), (effect_lev_base2,)]
                        # ("0|base1",),  
                        # ("99|base2",)
                        # ] # This is important, or else will fail if there are any (idx|assign) with only one datapt.
                    print("Getting dist_index vs. these base levels: ", list_grps_get)
                    dfproj_index, dfdist, _, _, _ = _compute_df_using_dist_index_traj(PAredu, var_effect, 
                                                                                      effect_lev_base1, effect_lev_base2,
                                                                                      list_grps_get=list_grps_get,
                                                                                      var_context_diff=var_context_diff,
                                                                                      plot_conjunctions_savedir=savedir)

                    ############## SAVE
                    # NOTE: dfproj_index already has trialcodes, so it doesnt need inner index...
                    dfproj_index["i_dimredu"] = i_dimredu
                    dfproj_index["i_outer"] = i_outer

                    dfdist["i_dimredu"] = i_dimredu
                    dfdist["i_outer"] = i_outer

                    list_dfindex.append(dfproj_index)
                    list_dfdist.append(dfdist)

            # Check that the output matches the input trialcodes
            DFDIST = pd.concat(list_dfdist).reset_index(drop=True)
            DFPROJ_INDEX = pd.concat(list_dfindex).reset_index(drop=True)

            # give new index row, based on unique trialcodes
            # trialcodes = DFDIST["trialcode"].unique().tolist()
            DFPROJ_INDEX["idx_row_datapt"] = [TRIALCODES_ORDERED.index(tc) for tc in DFPROJ_INDEX["trialcode"]]
            DFDIST["idx_row_datapt"] = [TRIALCODES_ORDERED.index(tc) for tc in DFDIST["trialcode"]]

            # Aggregate so that each trialcode gets single datapt
            from pythonlib.tools.pandastools import aggregGeneral
            DFPROJ_INDEX = aggregGeneral(DFPROJ_INDEX, ["idx_row_datapt", "trialcode", "time_bin_idx", "labels_1_datapt"], 
                                         ["dist_index", "dist_index_norm", "time_bin"], nonnumercols="all")
            DFDIST = aggregGeneral(DFDIST, ["idx_row_datapt", "trialcode", "time_bin_idx", "labels_1_datapt", "labels_2_grp"], 
                                   ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff", "time_bin"], nonnumercols="all")
            
            # Round the times to nearest ms
            DFDIST["time_bin"] = np.round(DFDIST["time_bin"]*1000)/1000
            DFPROJ_INDEX["time_bin"] = np.round(DFPROJ_INDEX["time_bin"]*1000)/1000

            # Condition
            _analy_switching_statespace_euclidian_traj_condition(PA, DFPROJ_INDEX, DFDIST, None, None)

            ### PLOTS
            savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}/plots"
            os.makedirs(savedir, exist_ok=True)

            ###### PLOTS
            if make_plots:
                _analy_switching_statespace_euclidian_traj_plots(DFPROJ_INDEX, DFDIST, savedir)

            if save_df:
                # Save the reuslts
                savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"
                os.makedirs(savedir, exist_ok=True)
                
                pd.to_pickle(DFPROJ_INDEX, f"{savedir}/DFPROJ_INDEX.pkl")
                pd.to_pickle(DFDIST, f"{savedir}/DFDIST.pkl")
                # pd.to_pickle(DFPROJ_INDEX_AGG, f"{savedir}/DFPROJ_INDEX_AGG.pkl")
                # pd.to_pickle(DFDIST_AGG, f"{savedir}/DFDIST_AGG.pkl")
            plt.close("all")


def analy_switching_GOOD_state_space(DFallpa, SAVEDIR_BASE, map_tcmorphset_to_idxmorph, 
                                                        list_morphset, map_tcmorphset_to_info):
    """
    FINAL good state space plots, for switching morph expts, just focused on trajectories.
    Is better than analy_switching_statespace_euclidian_score_and_plot, in that here are 
    plots clean version, and uses larger time window.
    NOTE: here does not do scalar state space plots.
    """
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER

    list_bregion = DFallpa["bregion"].unique().tolist()
    TWIND_ANALY = (-0.4, 1.2) # This is just for windowing final data, not for fitting pca.

    EVENT = "03_samp"

    NPCS_KEEP = 8
    scalar_or_traj = "traj"
    tbin_dur = 0.2
    tbin_slide = 0.01
    raw_subtract_mean_each_timepoint = False

    fit_twind = (0.05, 0.9)
    final_twind = TWIND_ANALY
    # superv_dpca_var = "idx_morph_temp"
    # superv_dpca_vars_group = None
    # dim_red_method = "dpca"
    EVENT = "03_samp"
    exclude_flankers = True
    ########################### STATE SPACE PLOTS (all, good), AND VARIOUS METRICS    
    list_npcs_keep = [NPCS_KEEP]
    
    for raw_subtract_mean_each_timepoint in [False]:
    # for raw_subtract_mean_each_timepoint in [False, True]:
        for NPCS_KEEP in list_npcs_keep:
            for dim_red_method in ["pca_proj", "dpca"]:
                ### Run
                SAVEDIR = f"{SAVEDIR_BASE}/analy_switching_GOOD_state_space/ev={EVENT}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={final_twind}-npcs={NPCS_KEEP}"

                for bregion in list_bregion:
                    for morphset in list_morphset:

                        savedir = f"{SAVEDIR}/bregion={bregion}/morphset={morphset}"
                        os.makedirs(savedir, exist_ok=True)
                        
                        _, PAredu = analy_extract_PA_conditioned(DFallpa, bregion, morphset, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                                    scalar_or_traj, EVENT, raw_subtract_mean_each_timepoint, 
                                            dim_red_method, final_twind, tbin_dur, tbin_slide, NPCS_KEEP, 
                                            savedir, restricted_twind_for_dpca=fit_twind, exclude_flankers=exclude_flankers)

                        ### Analysis -- get all distance scores.
                        analy_switching_statespace_euclidian_good(PAredu, savedir, PLOT_CLEAN_VERSION=True, skip_eucl_dist=True)


def prune_dfscores_good_morphset(dfscores, morphset_get, animal, date):
    """
    """

    print("morphset_get:", morphset_get)
    if morphset_get == "good_ones":
        # Hand modified
        if (animal, date) == ("Diego", 240515):
            # Angle rotation
            morphsets_ignore = [2] # Did most with two strokes...
        if (animal, date) == ("Diego", 240523):
            # THis is a garbage morphset, is not actually morphing.
            morphsets_ignore = [0]
        elif (animal, date) == ("Pancho", 240521):
            morphsets_ignore = [0] # one base prim is garbage
        elif (animal, date) == ("Pancho", 240524):
            morphsets_ignore = [4] # doesnt actually vaciallte across tirals
        else:
            morphsets_ignore = []
    elif morphset_get is None:
        # Get all morphsets
        morphsets_ignore = []
    elif isinstance(morphset_get, int):
        # get just this one morphset (exclude the others)
        morphsets_ignore = [ms for ms in DF_TCRES["run_morphset"].unique().tolist() if ms!=morphset_get]
    else:
        print(morphset_get)
        assert False

    # Prune dfscores
    morphsets_keep = [ms for ms in dfscores["run_morph_set_idx"].unique().tolist() if ms not in morphsets_ignore]
    print("morphsets_ignore: ", morphsets_ignore)
    print("morphsets_exist: ", sorted(dfscores["run_morph_set_idx"].unique()))
    print("morphsets_keep: ", morphsets_keep)
    dfscores = dfscores[dfscores["run_morph_set_idx"].isin(morphsets_keep)].reset_index(drop=True)
    print("morphsets_exist (after pruen): ", sorted(dfscores["run_morph_set_idx"].unique()))

    return dfscores


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    which_plots = sys.argv[3] 
    # WHICH_PLOTS = sys.argv[3] # 01, 11, ...

    question = "SP_psycho_trial"

    classifier_ver = "logistic"
    # classifier_ver = "ensemble"

    which_level = "trial"
    # LIST_COMBINE_AREAS = [False, True]
    
    if animal=="Diego":
        LIST_COMBINE_AREAS = [True]
    elif animal=="Pancho":
        # LIST_COMBINE_AREAS = [False]
        LIST_COMBINE_AREAS = [True]
    else:
        assert False

    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    PLOT_EACH_IDX = False

    if which_plots == "smooth":
        # PLOTS_DO = [1, 2, 0]
        # PLOTS_DO = [4.1, 4.2]
        PLOTS_DO = [4.2]
    elif which_plots == "switching":
        # PLOTS_DO = [5]
        PLOTS_DO = [5.1, 5.2]
    elif which_plots == "switching_dist":
        PLOTS_DO = [5.1]
    elif which_plots == "switching_ss":
        PLOTS_DO = [5.2]
    else:
        assert False

    for COMBINE_AREAS in LIST_COMBINE_AREAS:
        
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PSYCHO_SP/{animal}-{date}-{classifier_ver}-combine={COMBINE_AREAS}"
        os.makedirs(SAVEDIR, exist_ok=True)

        #####################################
        # Method 2 - Combine two dfallpa
        DFallpa = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS, question=question)

        #################### PREPROCESSING
        from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
        dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date, fr_mean_subtract_method=fr_normalization_method)

        from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import dfallpa_preprocess_condition
        shape_var_suff = "shape"
        loc_var_suff = "loc"
        dfallpa_preprocess_condition(DFallpa, shape_var_suff, loc_var_suff)

        from neuralmonkey.analyses.decode_moment import analy_psychoprim_prepare_beh_dataset
        savedir = f"{SAVEDIR}/morphsets_drawings"
        os.makedirs(savedir, exist_ok=True)
        DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, \
            map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = analy_psychoprim_prepare_beh_dataset(animal, date, savedir)
        
        # Hacky, remove trialcodes that are by eye messy
        from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_good_morphsets_switching_ignore_trialcodes
        trialcodes_ignore = params_good_morphsets_switching_ignore_trialcodes(animal, date)

        # Prune neural data to keep only good triacldoes.
        list_trialcode = DSmorphsets.Dat["trialcode"].unique().tolist() # trialcodes that are in DSmorphsets
        list_trialcode = [tc for tc in list_trialcode if tc not in trialcodes_ignore]

        # Do prune
        DFallpa["pa"] = [pa.slice_by_labels("trials", "trialcode", list_trialcode) for pa in DFallpa["pa"].values]

        if 0 in PLOTS_DO:
            # Summary:
            # - Timecourse of decode --> ambig images, see separation over time.
            from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp_better
            savedir_base = f"{SAVEDIR}/better_seperate_for_each_morph_idx"
            os.makedirs(savedir_base, exist_ok=True)
            analy_psychoprim_score_postsamp_better(DFallpa, DSmorphsets, 
                                                map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                                map_tcmorphset_to_info,
                                                savedir_base, animal, date,
                                                PLOT_EACH_IDX=PLOT_EACH_IDX)
        
        ###########################
        if 1 in PLOTS_DO:
            # Summary:
            # - Main (first) analysis, decode moment, scalar values for each index, etc. Useful for things like
            # whether ambig trails show switchign in neural state.
            # - To do further analyses, see 240617_momentbymoment_decode_psychoprim.ipynb
            # Cell called "[SWITCHING MORPHSETS, but also all others] MULTIPLE AREAS - combined analysis, loading all saved data"
            # - Additional analyses derived from these results, in notebook:
            # --- Bimodal states for smooth morphs? See cell "Categorization, for non-ambiguous images (bimodal across trials)"
            # --- Compare logistic vs. linear fit to decode state vs. index. see "plot_logistic_vs_linear_curve"
            # --- Compare logistic vs. linear fit, using activity projected along base1-base2 axis. See Compoare logistic vs. linear fit [good]
            from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp
            analy_psychoprim_score_postsamp(DFallpa, DSmorphsets, 
                                                map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                                map_tcmorphset_to_info,
                                                SAVEDIR,
                                                animal=animal, date=date)
                
        if 2 in PLOTS_DO:
            # [SMOOTH MORPH] 
            # Summary: For continuous morphs, looking for discrete switches in represnetation. 
            # - The first of these kinds of analyses.
            # Does simple stuff:
            # (1) State space plots.
            # (2) Pairwise distances between indices within morph.
            # NOTE: To summarize across days, run:
            # analy_decode_moment_psychometric_mult.py, which loads these results and does computation.
            # This is obsolete, instead, should run all things directly here and save the computation, as is done in 
            # analy_psychoprim_statespace_euclidian_score_and_plot

            list_morphset = DSmorphsets.Dat["morph_set_idx"].unique().tolist()
            analy_psychoprim_statespace_euclidian(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, list_morphset)

        if 3 in PLOTS_DO:
            ### sliceTCA (testing this out..)
            # Summary:
            # - Quick testing of whether sliceTCA represntations (neurons x time) show categorical effects, focusing 
            # on continuous morphs.
            
            from neuralmonkey.scripts.analy_slicetca_script import run
            if True:
                list_bregion = DFallpa["bregion"].unique().tolist()
            else:
                # list_bregion = ["PMv", "PMd", "SMA", "vlPFC", "M1"]
                list_bregion = ["PMv", "PMd", "M1"]

            list_morphset = DSmorphsets.Dat["morph_set_idx"].unique().tolist()
            # list_twind_analy = [(-0.1, 1.)]

            list_event_twindanaly = [
                ("03_samp", (-0.05, 1.1)),
                ("06_on_strokeidx_0", (-0.1, 0.4)),
            ]

            # Extract data for this morphset, wiht labels updated
            for event, twind_analy in list_event_twindanaly:
                for bregion in list_bregion:
                    for morphset in list_morphset:
                        ############### PREPARE DATASET
                        PA = extract_single_pa(DFallpa, bregion, None, "trial", event)

                        # Given morphset, assign new column which is the trial's role in that morphset.
                        dflab = PA.Xlabels["trials"]
                        dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

                        # keep just trials that are in this morphset
                        idx_exist = list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"]))
                        filtdict = {
                            "idx_morph_temp":idx_exist,
                        }
                        PA = PA.slice_by_labels_filtdict(filtdict)

                        savedir = f"{SAVEDIR}/sliceTCA/event={event}-twind={twind_analy}/bregion={bregion}/morphset={morphset}"
                        os.makedirs(savedir, exist_ok=True)

                        list_var_color = ["idx_morph_temp", "seqc_0_loc", "seqc_0_shape"]
                        run(PA, savedir, twind_analy, list_var_color)

        if 4.1 in PLOTS_DO:
            # [SMOOTH MORPH]
            # Summary: Just the smooth morphs, do all things related to categories.
            # THis is a better version of analy_psychoprim_statespace_euclidian, in that here does in one step what used to do
            # there in two steps (see doc there). Also, here is cleaner and updated code. 
            # - To summarize across (animal, dates) see notebook 240617_momentbymoment_decode_psychoprim.ipynb,
            # Cell called "# STATE SPACE VISUALIZATION PLOTS [METHOD 2]"

            from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_good_morphsets_no_switching
            list_morphset = params_good_morphsets_no_switching(animal, date)
            analy_psychoprim_statespace_euclidian_score_and_plot(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, animal, date, list_morphset)

        if 4.2 in PLOTS_DO:
            # [GOOD PLOTS]- Smooth morph, the "final" analysis?
            # Summary
            # Plot dist_index over time.
            # Score each morphset by taking max minus min (dist index), using cross-validated approach.
            
            # To make mult plots combining across (animal, dates), see notebook: 240617_momentbymoment_decode_psychoprim.ipynb
            # (1) "### (1) Set of plots #1 -- simply, specific of effect to PMv" 
            # (2) ### (2) Set of plots #2 -- comparison of neural to behavioral jumps
            for exclude_flankers in [True, False]:
                analy_psychoprim_statespace_euclidian_smooth_good(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info,
                                                            animal, date, exclude_flankers=exclude_flankers)


        if 5 in PLOTS_DO:
            # [GOOD FINAL plots for switching euclidina - atually, these are 2nd pbest plots. See below, in 5.1] 
            # Does both (1) Dfproj index (scores) and (2) state space traj plots
            # STEPS:
            # (1) Run this --> saved in "switching_euclidian_score_and_plot_traj"
            # (2) Load results across (animal, dates), see notebook, cell called# [State space and euclidian] Switching morphs -- Traj

            # Summary (Older notes):
            # - Morphsets with switching (ambiguous images), here is not using decode, like cases above,
            # just using state space and eucl.
            # NOTE: this can also work even if no swithcing on this day.
            # - (1) Older, statespace and eucl version: Load results across (animal, dates), see notebook, cell called "### Load results across (animal, dates) [switching]"
            # - (2) Newest (traj version) - Load results across (animal, dates), see notebook, cell called# [State space and euclidian] Switching morphs -- Traj
            HACK = False
            list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())
            analy_switching_statespace_euclidian_score_and_plot(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, 
                                                        list_morphset, map_tcmorphset_to_info, HACK=HACK)

        ########################### GOOD SWITCHING PLOTS (These replace (5) above)
        if 5.1 in PLOTS_DO:
            # [BEST, dist index euclidian plots] ACtually best (final) plots for euclidian distance index. See within for updates relative to analy_switching_statespace_euclidian_score_and_plot      
            # [Also does all RSA heatmaps]
            # Load and do FINAL PLOTS:
            # Use notebook: /home/lucas/code/neuralmonkey/neuralmonkey/notebooks_tutorials/241122_manuscript_psycho_categ_neural.ipynb
            DO_RSA_HEATMAPS = True # May take time..
            list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())
            analy_switching_GOOD_euclidian_index(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, 
                                                        list_morphset, map_tcmorphset_to_info,
                                                        make_plots=True, save_df=True,
                                                        DO_RSA_HEATMAPS=DO_RSA_HEATMAPS)
            
        if 5.2 in PLOTS_DO:
            # [BEST, state space trajectory plots]
            list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())
            analy_switching_GOOD_state_space(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, 
                                                        list_morphset, map_tcmorphset_to_info)

        

            
        