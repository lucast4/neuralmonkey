""" Mixture of things, related to population-level analyses.
- DECODING
- STATE SPACE PLOTS (trajectories)
- RSA (NO: this moved to /rsa.py).
"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.expttools import load_yaml_config
from pythonlib.tools.plottools import makeColors, legend_add_manual, savefig, rotate_x_labels, rotate_y_labels
from pythonlib.tools.expttools import writeDictToYaml
from pythonlib.tools.pandastools import append_col_with_grp_index, convert_to_2d_dataframe
from pythonlib.tools.snstools import rotateLabel
from pythonlib.tools.plottools import savefig

LABELS_IGNORE = ["IGN", ("IGN",), "IGNORE", ("IGNORE",)] # values to ignore during dcode.
N_MIN_TRIALS = 5 # min trials per level, otherwise throws level out.

def _popanal_preprocess_normalize_softzscore(PA, PLOT=False):
    """ Normalize firing rates so that similar acorss neruons (higha nd low fr) whiel
    still having higher for high fr.
    Similar to "soft normalization" used by Churchland group.
    """

    x = PA.X.reshape(PA.X.shape[0], -1)
    
    x, DENOM, CENTER = _popanal_preprocess_normalize_softzscore_raw(x)

    # Do normalization.
    PAnorm = PA.copy()
    PAnorm.X = PAnorm.X/DENOM[:, :, None]

    # # STD (across all trials and times)
    # normvec = np.std(x, axis=1)
    # assert len(normvec.shape)==1
    # normvec = np.reshape(normvec, [normvec.shape[0], 1,1]) # (chans, 1, 1) # std for each chan, across (times, trials).
    # normmin = np.percentile(normvec, [2.5]) # get the min (std fr across time/conditions) across channels, add this on to still have

    # # min fr, to make this a "soft" normalization
    # frmean_each_chan = np.mean(x, axis=1) # (chans, 1, 1) # std for each chan, across (times, trials).
    # frmin = np.min(frmean_each_chan) # (chans, 1, 1)) # min (mean fr across time/condition) across chans
    # # frmin = np.min(np.mean(np.mean(PA.X, axis=1, keepdims=True), axis=2, keepdims=True)) # (chans, 1, 1)) # min (mean fr across time/condition) across chans

    # # to further help making this "soft"
    # abs_fr_min = 3 # any fr around this low, want to penalize drastically, effectively making it not contyribute much to population activit.

    # # DENOM = (normvec+normmin)
    # # DENOM = (normvec + normmin + frmin + abs_fr_min) # To further lower infleunce of low FR neurons (started 2/8/24, 2:55pm)
    # DENOM = (normvec + frmin + abs_fr_min) # To make more similar across chans. (started 2/11/24)

    # # Do normalization.
    # PAnorm = PA.copy()
    # PAnorm.X = PAnorm.X/DENOM

    if PLOT:
        from pythonlib.tools.plottools import plotScatter45
        x = PA.X.reshape(PA.X.shape[0], -1)
        frmean_base = np.mean(x, axis=1)
        frstd_base = np.std(x, axis=1)

        x = PAnorm.X.reshape(PAnorm.X.shape[0], -1)
        frmean_norm = np.mean(x, axis=1)
        frstd_norm = np.std(x, axis=1)

        fig, axes = plt.subplots(2,3, figsize=(10, 8))

        ax = axes.flatten()[0]
        ax.plot(frmean_base, frmean_norm, "ok")
        ax.plot(0,0, "wx")
        ax.set_xlabel("frmean_base")
        ax.set_ylabel("frmean_normed")

        ax = axes.flatten()[1]
        ax.plot(frstd_base, frstd_norm, "ok")
        ax.plot(0,0, "wx")
        ax.set_xlabel("frstd_base")
        ax.set_ylabel("frstd_normed")

        ax = axes.flatten()[2]
        plotScatter45(frmean_base, frstd_base, ax=ax)
        ax.set_xlabel("frmean_base")
        ax.set_ylabel("frstd_base")

        ax = axes.flatten()[3]
        plotScatter45(frmean_norm, frstd_norm, ax=ax)
        ax.set_xlabel("frmean_norm")
        ax.set_ylabel("frstd_norm")

        ax = axes.flatten()[4]
        plotScatter45(frmean_base, DENOM.squeeze(), ax=ax)
        ax.set_xlabel("mean fr (baseline)")
        ax.set_ylabel("denom for norm")

        ax = axes.flatten()[5]
        plotScatter45(frstd_base, DENOM.squeeze(), ax=ax)
        ax.set_xlabel("std fr (baseline)")
        ax.set_ylabel("denom for norm")
    else:
        fig = None

    return PAnorm, fig

def _popanal_preprocess_normalize_softzscore_raw(x):
    """ 
    Normalize firing rates so that similar acorss neruons (higha nd low fr) whiel
    still having higher for high fr.
    Similar to "soft normalization" used by Churchland group.

    PARAMS:
    - x, (nchans, ndat), where ndat is usually (trials x times). Raw (or sqrt transfomred) firing rates
    RETURNS:
    - x_norm, (nchans, ndat), rescaled, then mean-subtracted (global mean)
    - DENOM, (nchans, 1)
    - CENTER, (nchans, 1)
    -- can apply DENOM and CENTER to x to return x:
        x_norm = x.copy()
        x_norm = x_norm/DENOM
        x_norm = x_norm-CENTER
    """

    assert len(x.shape)==2

    #### (1) How much to rescale FR
    
    # STD (across all trials and times)
    normvec = np.std(x, axis=1)
    assert len(normvec.shape)==1
    normvec = np.reshape(normvec, [normvec.shape[0], 1]) # (chans, 1) # std for each chan, across (times, trials).
    # normvec = np.reshape(normvec, [normvec.shape[0]]) # (chans, 1, 1) # std for each chan, across (times, trials).
    # normmin = np.percentile(normvec, [2.5]) # get the min (std fr across time/conditions) across channels, add this on to still have

    # Min fr across all chans (mean frs), to make this a "soft" normalization
    frmean_each_chan = np.mean(x, axis=1) # (chans, 1, 1) # mean for each chan, across (times, trials).
    frmin = np.min(frmean_each_chan) # scalar
    # frmin = np.min(np.mean(np.mean(PA.X, axis=1, keepdims=True), axis=2, keepdims=True)) # (chans, 1, 1)) # min (mean fr across time/condition) across chans

    # to further help making this "soft"
    abs_fr_min = 3 # any fr around this low, want to penalize more, effectively making it contyribute less to population activit.

    # DENOM = (normvec+normmin)
    # DENOM = (normvec + normmin + frmin + abs_fr_min) # To further lower infleunce of low FR neurons (started 2/8/24, 2:55pm)
    DENOM = (normvec + frmin + abs_fr_min) # To make more similar across chans. (started 2/11/24)

    ### Do normalization.
    # Rescale fr [0, around 2]
    x_norm = x.copy()
    assert len(DENOM.shape)==len(x_norm.shape)
    x_norm = x_norm/DENOM

    # After rescaling (by dividing), get mean (which will be subtracted to do soft z-score)
    CENTER = np.mean(x_norm, axis=1, keepdims=True) # (nchans, 1)
    x_norm = x_norm-CENTER

    return x_norm, DENOM, CENTER

def popanal_preprocess_scalar_normalization(PA, grouping_vars, subtract_mean_each_level_of_var="IGNORE",
                                            plot_example_chan_number=None, plot_example_split_var_string=None,
                                            DO_AGG_TRIALS=True, subtract_mean_at_each_timepoint=True,
                                            subtract_mean_across_time_and_trial=False):
    """ Preprocess PA, with different options for normalization ,etc
    PARAMS:
    - subtract_mean_each_level_of_var, eitehr None (ignore) or str, in which case will
    find mean fr for each level of this var, then subtract this mean from all datpts that
    have this level for this var. e.g., if subtract_mean_each_level_of_var=="gridloc", then
    de-means for each location.
    RETURNS:
        - new PA, without modifiying input PA
    """
    # if plot_example_chan==False:
    #     plot_example_chan = None
    # assert isinstance(plot_example_chan, bool) or isinstance(plot_example_chan, int)

    if subtract_mean_each_level_of_var is None:
        subtract_mean_each_level_of_var = "IGNORE"
    else:
        assert isinstance(subtract_mean_each_level_of_var, str)

    # 1) First, rescale all FR (as in Churchland stuff), but isntead of using
    # range, use std (like z-score)
    PAnorm_pre, _ = _popanal_preprocess_normalize_softzscore(PA)
    PAnorm = PAnorm_pre.copy()

    # if False:
    #     # FR
    #     normvec = np.mean(np.mean(PA.X, axis=1, keepdims=True), axis=2, keepdims=True) # (chans, 1, 1)
    # else:
    #     # STD (across all trials and times)
    #     normvec = np.std(np.reshape(PA.X, (PA.X.shape[0], -1)), axis=1) # (chans, 1, 1)
    #     assert len(normvec.shape)==1
    #     normvec = np.reshape(normvec, [normvec.shape[0], 1,1])
    # normmin = np.percentile(normvec, [2.5]) # get the min across channels, add this on to still have
    # # higher FR neurons more important
    # # DENOM = (normvec+normmin)
    # tmp = np.min(np.mean(np.mean(PA.X, axis=1, keepdims=True), axis=2, keepdims=True)) # (chans, 1, 1)) # min mean fr across chans
    # DENOM = (normvec + normmin + 3 + tmp) # To further lower infleunce of low FR neurons (started 2/8/24, 2:55pm)
    # PAnorm = PA.copy()
    # PAnorm.X = PAnorm.X/DENOM

    if subtract_mean_at_each_timepoint:
        # 1) subtract global mean (i.e., at each time bin)
        PAnorm = PAnorm.norm_subtract_trial_mean_each_timepoint()

    if subtract_mean_across_time_and_trial:
        PAnorm = PAnorm.norm_subtract_mean_each_chan()

    if False: # Replaced with rescale step above
        # 2) rescale (for each time bin, across all trials
        Xstd = np.std(PAnorm.X, axis=1, keepdims=True) # (chans, 1, times)
        if False:
            # Just z-score.
            PAnorm.X = PAnorm.X/Xstd
        else:
            # Greater penalty for low FR neurons. I.e., This solves problem
            # that z-score places low FR neruons at same scale as high FR.
            # instead, this penalizes low FR.
            fr_min_1 = np.percentile(np.mean(Xstd, axis=2), [2.5]).squeeze() # Lowest fr across channels.
            fr_min_2 = np.percentile(np.mean(PAnorm.X, axis=2), [2.5]).squeeze() # Lowest fr across channels.
            print(np.mean(Xstd, axis=2))
            print(np.mean(PAnorm.X, axis=2))
            assert False
            PAnorm.X = PAnorm.X/(Xstd + fr_min)

    # 3) Subtract mean based on grouping by other variable
    if not subtract_mean_each_level_of_var=="IGNORE":
        from neuralmonkey.classes.population import concatenate_popanals
        list_pa = PAnorm.split_by_label("trials", subtract_mean_each_level_of_var)[0]
        list_pa_norm = []
        for pa in list_pa:
            # subtract global mean
            list_pa_norm.append(pa.norm_subtract_trial_mean_each_timepoint())
        PAnorm = concatenate_popanals(list_pa_norm, "trials")

    # Finally, convert to scalars.
    PAscal = PAnorm.agg_wrapper("times") # mean over time --> (chans, trials)

    ### GET SCALARS (by averaging over time and grouping by variables of interest)
    if DO_AGG_TRIALS:
        # - get single "pseudotrial" for each conjunctive level
        # vars = ["shape_oriented", "gridloc", "FEAT_num_strokes_task", "stroke_index"]
        # vars = ["shape_oriented", "gridloc"]
        PAscalagg, groupdict = PAscal.slice_and_agg_wrapper("trials", grouping_vars, return_group_dict=True)
        # print("Sample sizes for each level of grouping vars")
        # for k,v in groupdict.items():
        #     print(k, " -- ", len(v))
        # assert False
    else:
        PAscalagg = PAscal
        groupdict = None

    # # Finally, convert to scalars.
    # PAagg = PAagg.agg_wrapper("times") # mean over time --> (chans, trials)

    ######## PLOTS
    if plot_example_chan_number is not None:
        add_legend = True
        fig, axes = plt.subplots(2,2, figsize=(8,8))
        titles = ["PA - raw", "PAnorm_pre - after churchland norm", "PAnorm - final norm", "PAscalagg"]
        for pathis, ax, tit in zip([PA, PAnorm_pre, PAnorm, PAscalagg], axes.flatten(), titles):
            pathis.plotwrapper_smoothed_fr_split_by_label("trials", plot_example_split_var_string,
                                                          ax=ax, add_legend=add_legend, chan=plot_example_chan_number)
            ax.set_title(tit)
            ax.axhline(0)
    else:
        fig, axes = None, None

    return PAnorm, PAscal, PAscalagg, fig, axes, groupdict


# def rsa_distmat_quantify_same_diff_variables(Clsim, ind_var, ignore_diagonal=True):
#     """
#     PARAMS:
#     - ind_var = 0 # e..g, if each row is labeled with a tuple like (shape, loc), then if
#     ind_var==0, then this means "same" is defined as having same shape
#     """
#
#     # Collect mapping
#     map_pair_labels_to_indices = {} # (lab1, lab2) --> (col, row)
#     for i, lr in enumerate(Clsim.Labels):
#         for j, lc in enumerate(Clsim.LabelsCols):
#             # only take off diagonal
#             if ignore_diagonal:
#                 if i>=j:
#                     continue
#             else:
#                 if i>j:
#                     continue
#             map_pair_labels_to_indices[(lr, lc)] = (i, j)
#
#     # Find the coordinates of "same" and "diff" pairs.
#     # given a var dimension, get all indices that are "same" along that dimension
#     list_inds_same = []
#     list_inds_diff = []
#     for lab_pair, inds in map_pair_labels_to_indices.items():
#         a = lab_pair[0][ind_var]
#         b = lab_pair[1][ind_var]
#         if a==b:
#             # then is "same"
#             list_inds_same.append(inds)
#         else:
#             list_inds_diff.append(inds)
#
#     # Collect data
#     list_i = [x[0] for x in list_inds_same]
#     list_j = [x[1] for x in list_inds_same]
#     vals_same = Clsim.Xinput[(list_i, list_j)]
#
#     list_i = [x[0] for x in list_inds_diff]
#     list_j = [x[1] for x in list_inds_diff]
#     vals_diff = Clsim.Xinput[(list_i, list_j)]
#
#     return vals_same, vals_diff



# def load_single_data(RES, bregion, twind, which_level):
#     """ Helper to load...
#     """
#
#     tmp = [res for res in RES if res["which_level"]==which_level]
#     if len(tmp)!=1:
#         print(tmp)
#         assert False
#     res = tmp[0]
#
#     # Extract specifics
#     key = (bregion, twind)
#     PA = res["DictBregionTwindPA"][key]
#     Clraw = res["DictBregionTwindClraw"][key]
#     Clsim = res["DictBregionTwindClsim"][key]
#
#     return res, PA, Clraw, Clsim


    # #################### COMPUTE DISTANCE MATRICES AND SCORE RELATIVE TO THEORETICAL MATRICES.
    # PLOT = PLOT_INDIV
    # list_dfres = []
    # list_dfres_theor = []
    # DictBregionTwindClraw = {}
    # DictBregionTwindClsim = {}
    # ct = 0
    # for (bregion, twind), pa in DictBregionTwindPA.items():
    #     print("Scoring, for: ", bregion, twind)
    #
    #     sdir = f"{savedir}/preprocess/{bregion}-{twind}"
    #     os.makedirs(sdir, exist_ok=True)
    #     print("Saving to: ", sdir)
    #
    #     PLOT_THEORETICAL_SIMMATS = ct==0 and PLOT==True # Only do once. this same across bregions and twinds.
    #     dfres_same_diff, dfres_theor, Clraw, Clsim, PAagg = _preprocess_rsa_scalar_population(pa, EFFECT_VARS, version_distance,
    #                                                                                           PLOT=PLOT,
    #                                                                                           sdir=sdir,
    #                                                                                           subtract_mean_each_level_of_var=subtract_mean_each_level_of_var,
    #                                                                                            PLOT_THEORETICAL_SIMMATS=PLOT_THEORETICAL_SIMMATS)
    #
    #     # Collect results
    #     dfres_same_diff["bregion"] = bregion
    #     dfres_same_diff["twind"] = [twind for _ in range(len(dfres_same_diff))]
    #     dfres_theor["bregion"] = bregion
    #     dfres_theor["twind"] = [twind for _ in range(len(dfres_theor))]
    #
    #     list_dfres.append(dfres_same_diff)
    #     list_dfres_theor.append(dfres_theor)
    #
    #     DictBregionTwindClraw[(bregion, twind)] = Clraw
    #     DictBregionTwindClsim[(bregion, twind)] = Clsim
    #
    #     plt.close("all")
    #
    # DFRES_SAMEDIFF = pd.concat(list_dfres).reset_index(drop=True)
    # DFRES_THEOR = pd.concat(list_dfres_theor).reset_index(drop=True)
    #
    # ###############################
    # # SUMMARY PLOTS
    # sdir = f"{savedir}/summary"
    # os.makedirs(sdir, exist_ok=True)
    #
    # ########################### 1. Same (level within var) vs. diff.
    # if version_distance=="pearson":
    #     GROUPING_LEVELS = ["diff", "same"]
    # elif version_distance=="euclidian":
    #     GROUPING_LEVELS = ["same", "diff"]
    # else:
    #     assert False
    # dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = summarize_featurediff(DFRES_SAMEDIFF,
    #                                                                                               "same_or_diff", GROUPING_LEVELS, ["mean"], ["ind_var", "ind_var_str", "bregion", "sort_order", "grouping_vars", "subtract_mean_each_level_of_var"])
    # dfsummaryflat = SP.SN.datamod_sitegetter_reorder_by_bregion(dfsummaryflat)
    #
    # # Summarize all
    # var_subplot = ["grouping_vars", "subtract_mean_each_level_of_var"]
    # _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "shape_oriented", "gridloc", var_subplot, "value", "bregion");
    # savefig(fig, f"{sdir}/plot45_same_vs_diff-1.pdf")
    # if "stroke_index" in EFFECT_VARS:
    #     _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "shape_oriented", "stroke_index", var_subplot, "value", "bregion");
    #     savefig(fig, f"{sdir}/plot45_same_vs_diff-2.pdf")
    #     _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "gridloc", "stroke_index", var_subplot, "value", "bregion");
    #     savefig(fig, f"{sdir}/plot45_same_vs_diff-3.pdf")
    # plt.close("all")
    #
    # if False:
    #     # Not working, problem is dfthis comes up empty. easy fix.
    #     # Is redundant given the following "theoretical comparison" plots.
    #     # Plot for each grouping_vars
    #
    #     grouping_vars = ("shape_oriented", "gridloc", "stroke_index")
    #     # grouping_vars = ["shape_oriented", "gridloc"]
    #     a = DFRES_SAMEDIFF["grouping_vars"].isin([grouping_vars])
    #     b = DFRES_SAMEDIFF["subtract_mean_each_level_of_var"].isin(["IGNORE"])
    #     dfthis = DFRES_SAMEDIFF[(a) & (b)]
    #
    #     a = dfsummaryflat["grouping_vars"].isin([grouping_vars])
    #     b = dfsummaryflat["subtract_mean_each_level_of_var"].isin(["IGNORE"])
    #
    #     dfthis_diff = dfsummaryflat[(a) & (b)]
    #     dfthis_diff
    #     from pythonlib.tools.pandastools import plot_pointplot_errorbars
    #     from pythonlib.tools.plottools import rotate_x_labels
    #
    #     # pull out the non-hierarchical dataframes
    #     fig, axes = plt.subplots(2,2, figsize=(10,10), sharey=True)
    #     list_ind_var = sorted(dfthis["ind_var"].unique().tolist())
    #     list_same_diff = ["same", "diff"]
    #     ct = 0
    #     for ind_var, ax in zip(list_ind_var, axes.flatten()):
    #         ind_var_str = grouping_vars[ind_var]
    #         ax.set_title(ind_var_str)
    #         ct+=1
    #         for same_diff in list_same_diff:
    #             dfthisthis = dfthis[dfthis["ind_var"]==ind_var]
    #             plot_pointplot_errorbars(dfthisthis, "bregion", "mean", ax=ax, yvar_err=f"sem", hue="same_or_diff")
    #             # assert False
    #             #
    #             # x=dfthisthis["bregion"]
    #             # y=dfthisthis[f"{same_diff}_mean"]
    #             # yerr = dfthisthis[f"{same_diff}_sem"]
    #             # lab = f"{same_diff}-{ind_var_str}"
    #             # ax.errorbar(x=x, y=y, yerr=yerr, label=lab)
    #             # # ax.bar(x, y, yerr=yerr, label=lab, alpha=0.4)
    #             # # sns.barplot(data=dfthisthis, x="bregion", y="same_mean", yerr=dfthisthis["same_sem"])
    #
    #         ax.axhline(0, color="k", alpha=0.25)
    #         rotate_x_labels(ax, 75)
    #         ax.legend()
    #
    #     ax = axes.flatten()[ct]
    #     list_ind_var = sorted(dfthis["ind_var"].unique().tolist())
    #     list_same_diff = ["same", "diff"]
    #     for ind_var in list_ind_var:
    #         ind_var_str = grouping_vars[ind_var]
    #         dfthisthis = dfthis_diff[dfthis_diff["ind_var"]==ind_var]
    #
    #         x=dfthisthis["bregion"]
    #         y=dfthisthis["value"]
    #         lab = f"{ind_var_str}"
    #         ax.plot(x, y, label=lab)
    #
    #     ax.axhline(0, color="k", alpha=0.25)
    #     rotate_x_labels(ax, 75)
    #     ax.legend()
    #
    # #################### 2. Comparing data simmat to theoretical simmats.
    # for yvar in ["cc", "mr_coeff"]:
    #     fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", kind="point", aspect=1.5,
    #                     row="twind")
    #     rotateLabel(fig)
    #     savefig(fig, f"{sdir}/vs_theor_simmat-pointplot-{yvar}.pdf")
    #
    #     fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", alpha=0.5, aspect=1.5, row="twind")
    #     rotateLabel(fig)
    #     savefig(fig, f"{sdir}/vs_theor_simmat-scatterplot-{yvar}.pdf")
    #
    #     # Summarize results in a heatmap (region x effect)
    #     for norm in ["col_sub", "row_sub", None]:
    #         for twind in list_time_windows:
    #             dfthis = DFRES_THEOR[DFRES_THEOR["twind"] == twind]
    #             _, fig, _, _ = convert_to_2d_dataframe(dfthis, "bregion", "var",
    #                                                    True, "mean",
    #                                                    yvar, annotate_heatmap=False, dosort_colnames=False,
    #                                                    norm_method=norm)
    #             savefig(fig, f"{sdir}/vs_theor_simmat-heatmap-{yvar}-norm_{norm}-twind_{twind}.pdf")
    #
    #             plt.close("all")
    #
    # return DFRES_SAMEDIFF, DFRES_THEOR, DictBregionTwindPA, \
    #     DictBregionTwindClraw, DictBregionTwindClsim, savedir


def preprocess_pca(SP, pca_trial_agg_grouping, pca_time_agg_method=None,
                   list_event_window=None,
                   pca_norm_subtract_condition_invariant = True,
                   list_bregion = None,
                   list_vars_others=None,
                   do_balance=True):
    # pca_trial_agg_grouping = "epoch"
    # pca_trial_agg_grouping = "seqc_0_shape"
    # list_bregion = ["PMv_m"]

    assert False, "Quicker: first extract PA using PA, _ = SP.dataextract_as_popanal_statespace, then split into bregions and timw windows, then do below. see rsa for how donw."
    import pandas as pd

    if list_bregion is None:
        from neuralmonkey.classes.session import _REGIONS_IN_ORDER as list_bregion

    RES = []
    for i, (event, pre_dur, post_dur) in enumerate(list_event_window):
        # Get PCA space using all data in window
        for bregion in list_bregion:

            print(f"{event} - {pre_dur} - {post_dur} - {bregion}")

            # Get sites for this regions
            sites = SP.sitegetter_map_region_to_sites(bregion)

            # Do PCA
            PApca, fig, PA, sample_meta = SP.dataextract_as_popanal_statespace_balanced_pca(sites, event, pre_dur, post_dur,
                                                                                            pca_trial_agg_grouping=pca_trial_agg_grouping,
                                                                                            pca_time_agg_method=pca_time_agg_method,
                                                                                            pca_norm_subtract_condition_invariant=pca_norm_subtract_condition_invariant,
                                                                                            pca_plot=False,
                                                                                            list_vars_others=list_vars_others,
                                                                                            do_balance=do_balance)

            RES.append({
                "event":event,
                "pre_dur":pre_dur,
                "post_dur":post_dur,
                "event_wind":f"{event}_{pre_dur}_{post_dur}",
                "bregion":bregion,
                "PApca":PApca,
                "PA":PA,
                "sample_meta":sample_meta
            })

    DF_PA_SPACES = pd.DataFrame(RES)

    return DF_PA_SPACES

# Which dataset to use to construct PCA?
# def pca_make_space(PA, DF, trial_agg_method, trial_agg_grouping,
#     time_agg_method=None,
#     norm_subtract_condition_invariant=False,
#     ploton=True):
#     """ Prperocess data (e.g,, grouping by trial and time) and then
#     Make a PopAnal object holding (i) data for PCA and (ii) the results of
#     PCA.
#     PARAMS:
#     - PA, popanal object, holds all data.
#     - DF, dataframe, with one column for each categorical variable you care about (in DATAPLOT_GROUPING_VARS).
#     The len(DF) must equal num trials in PA (asserts this)
#     - trial_agg_grouping, list of str defining how to group trials, e.g,
#     ["shape_oriented", "gridloc"]
#     - norm_subtract_condition_invariant, bool, if True, then at each timepoint subtracts
#     mean FR across trials
#     RETURNS:
#     - PApca, a popanal holding the data that went into PCA, and the results of PCA,
#     and methods to project any new data to this space.
#     """
#
#     assert DF==None, "instead, put this in PA.Xlabels"
#
#     # First, decide whether to take mean over some way of grouping trials
#     if trial_agg_method==None:
#         # Then dont aggregate by trials
#         PApca = PA.copy()
#     elif trial_agg_method=="grouptrials":
#         # Then take mean over trials, after grouping, so shape
#         # output is (nchans, ngrps, time), where ngrps < ntrials
#         if DF is None:
#             DF = PA.Xlabels["trials"]
#         if False:
#             groupdict = grouping_append_and_return_inner_items(DF, trial_agg_grouping)
#             # groupdict = DS.grouping_append_and_return_inner_items(trial_agg_grouping)
#             PApca = PA.agg_by_trialgrouping(groupdict)
#         else:
#             # Better, since it retains Xlabels
#             PApca = PA.slice_and_agg_wrapper("trials", trial_agg_grouping)
#     else:
#         print(trial_agg_method)
#         assert False
#
#     # First, whether to subtract mean FR at each timepoint
#     if norm_subtract_condition_invariant:
#         PApca = PApca.norm_subtract_mean_each_timepoint()
#
#     # second, whether to agg by time (optional). e..g, take mean over time
#     if time_agg_method=="mean":
#         PApca = PApca.agg_wrapper("times")
#         # PApca = PApca.mean_over_time(return_as_popanal=True)
#     else:
#         assert time_agg_method==None
#
#     print("Shape of data going into PCA (chans, trials, times):", PApca.X.shape)
#     fig = PApca.pca("svd", ploton=ploton)
#
#     return PApca, fig

def _preprocess_extract_PApca(DF_PA_SPACES, event_wind, bregion):
    tmp = DF_PA_SPACES[(DF_PA_SPACES["event_wind"]==event_wind) & (DF_PA_SPACES["bregion"]==bregion)]
    assert len(tmp)==1
    PApca = tmp.iloc[0]["PApca"]
    return PApca


def _preprocess_extract_plot_params(SP, PApca):
    list_event = SP.DfScalar["event"].unique().tolist()
    sites = PApca.Chans

    return list_event, sites

def plot_statespace_grpbyevent_overlay_othervars(PApca, SP, var, vars_others, PLOT_TRIALS=False,
    dims_pc = (0,1), alpha_mean=0.5, alpha_trial=0.2, n_trials_rand=10, time_windows_mean=None,
    list_event_data=None):

    if list_event_data is None:
        # get all events
        list_event_data, _ = _preprocess_extract_plot_params(SP, PApca)

    # 2) One subplot for each event.
    ncols = 3
    nrows = int(np.ceil(len(list_event_data)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))
    for event_dat, ax in zip(list_event_data, axes.flatten()):
        print(event_dat)
        ax.set_title(event_dat)

        # 1) get all data, so that can get what levels of othervar exist
        _, levels_var, levels_othervar = SP.statespate_pca_extract_data(PApca, event_dat, var, vars_others,
                                                                                  levels_var=None,
                                                                                  levels_othervar=None)
        # 2) then iterate over all levles of othervar
        for levother in levels_othervar:
            DICT_DF_DAT, levels_var, _ = SP.statespate_pca_extract_data(PApca, event_dat, var, vars_others,
                                                                                      levels_var=None,
                                                                                      levels_othervar=[levother])
            # Plot
            _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_mean,
                ax, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial, n_trials_rand)
    return fig


def plot_statespace_grpbyevent(PApca, SP, var, vars_others, PLOT_TRIALS=False,
    dims_pc = (0,1), time_windows_mean=None, alpha_mean=0.5, alpha_trial=0.2,
    n_trials_rand=10):
    """ 
    Plot results for this PApca (specifying sites, such as for a bregion), 
    trajectories for each event.
    One subplot for each event, overlaying all levels of (var, othervar)
    """
    list_event, sites = _preprocess_extract_plot_params(SP, PApca)
    ncols = 3
    nrows = int(np.ceil(len(list_event)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))

    for event, ax in zip(list_event, axes.flatten()):
        ax.set_title(event)

        DICT_DF_DAT, _, _ = SP._statespace_pca_extract_data(sites, event, var, vars_others)

        _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_mean,
            ax, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial, n_trials_rand)
    return fig


def plot_statespace_grpbyvarsothers(event, PApca, SP, var, vars_others, PLOT_TRIALS=False,
    dims_pc = (0,1), alpha_trial=0.2, alpha_mean=0.5,
    time_windows_mean=None, n_trials_rand=10):
    """
    One figure for this event
    Each subplot is a level of varsothers
    PARAMS:
    - time_windows_mean, list of 2-tuples, each a (pre_dur, post_dur), where negative
    pre_dur means before. Converts fr from (nchans, times) to (nchans, len(times_windows_mean))
    """
    ## One figure for each event (each subplot a level of othervar)

    # import numpy as np
    # from pythonlib.tools.plottools import makeColors
    # from pythonlib.tools.plottools import legend_add_manual
    # dims_pc = (0,1)
    # sites = PApca.Chans
    # PLOT_TRIALS = False


    # Extract data for this event
    list_event, sites = _preprocess_extract_plot_params(SP, PApca)
    DICT_DF_DAT, levels_var, levels_othervar = SP._statespace_pca_extract_data(sites, event, var, vars_others)

    list_cols = makeColors(len(levels_var))

    # print("levels_var:", levels_var)
    # print("levels_othervar:", levels_othervar)

    # one subplot for each level of othervar
    ncols = 3
    nrows = int(np.ceil(len(levels_othervar)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))

    for levother, ax in zip(levels_othervar, axes.flatten()):
        ax.set_title(levother)
        for lev, col in zip(levels_var, list_cols):
            key = (lev, levother)
            if key in DICT_DF_DAT.keys():
                df = DICT_DF_DAT[key]

                if len(df)==0:
                    print(f"No data, skipping: {event}, {key}")
                    continue

                _plot_statespace_df_on_ax(PApca, SP, df, time_windows_mean,
                    ax, col, PLOT_TRIALS, dims_pc, alpha_mean, n_trials_rand=n_trials_rand)

        # overlay legend
        legend_add_manual(ax, levels_var, list_cols, 0.2)

    return fig

def _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_mean,
    ax, PLOT_TRIALS=False, dims_pc=(0,1), alpha_mean=0.5,
    alpha_trial=0.2, n_trials_rand=10):

    list_cols = makeColors(len(DICT_DF_DAT))

    for (key, df), col in zip(DICT_DF_DAT.items(), list_cols):

        if len(df)==0:
            print(f"No data, skipping: {key}")
            continue

        _plot_statespace_df_on_ax(PApca, SP, df, time_windows_mean,
            ax, col, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial=alpha_trial,
            n_trials_rand=n_trials_rand)

    # overlay legend
    legend_add_manual(ax, list(DICT_DF_DAT.keys()), list_cols, 0.1)


def _plot_statespace_df_on_ax(PApca, SP, df, time_windows_mean,
    ax, col = "k", PLOT_TRIALS=False, dims_pc=(0,1), alpha_mean=0.5,
    alpha_trial=0.2, n_trials_rand=10):
    """ Very low-level, plot a single trajectory on a single axis.
    PARAMS:
    - PApca, Popanal, holding the pca results.
    - SP, Snippets, holding data.
    - df, the specific slice of SP.DfScalar which you want to plot.
    - time_windows_mean, list of tuples, each (predur, postdur) if
    you want to bin fr into this trajectory.
    """

    _, sites = _preprocess_extract_plot_params(SP, PApca)

    times_to_mark = [0]
    # times_to_mark_markers = [f"{t}" for t in times_to_mark]
    times_to_mark_markers = ["d"]

    # convert to pa
    # get frmat from data
    PAdata = SP.dataextract_as_popanal_good(df, chans_needed=sites)

    if time_windows_mean is not None:
        PAdata.X, PAdata.Times = PAdata.agg_by_time_windows(time_windows=time_windows_mean)

    # plot
    # For each level of a var, plot it in different color, and overlay many trials
    if False: # Do this outside, using SP
        list_pa, list_levels = PA.split_by_label("trials", var)

    # Plot single datapts
    if PLOT_TRIALS:
        # plot a random subset of trials
        ntrials = PAdata.X.shape[1]
        trials_plot = list(range(ntrials))
        if ntrials > n_trials_rand:
            if False:
                # determistic, but spread throughout trials
                from pythonlib.tools.listtools import random_inds_uniformly_distributed
                trials_plot = random_inds_uniformly_distributed(trials_plot, n_trials_rand)
            else:
                import random
                trials_plot = random.sample(trials_plot, n_trials_rand)

        for trial in trials_plot:
            frmat = PAdata.X[:, trial, :]
            PApca.statespace_pca_plot_projection(frmat, ax, dims_pc=dims_pc, color_for_trajectory=col,
                                        times=PAdata.Times, times_to_mark=times_to_mark,
                                         times_to_mark_markers=times_to_mark_markers,
                                         alpha = alpha_trial,
                                         markersize=3)

    # Overlay the mean trajectory for a level            
    frmean = np.mean(PAdata.X, axis=1)
    PApca.statespace_pca_plot_projection(frmean, ax, dims_pc=dims_pc, color_for_trajectory=col,
                                        times=PAdata.Times, times_to_mark=times_to_mark,
                                         times_to_mark_markers=times_to_mark_markers,
                                         alpha = alpha_mean,
                                         markersize=7, marker="P")
    # grid on, for easy comparisons
    ax.grid()

def plotwrapper_statespace_mult_events():
    # Extract PApca
    tmp = DF_PA_SPACES[(DF_PA_SPACES["event_wind"]==event_wind_pca) & (DF_PA_SPACES["bregion"]==bregion)]
    assert len(tmp)==1
    PApca = tmp["PApca"].item()

    # WHICH DATA?
    fig = plot_statespace_grpbyevent(PApca, SP, var_dat, vars_others_dat, dims_pc = dims_pc,
                                  time_windows_mean = time_windows_traj, alpha_mean=0.2,
                                  PLOT_TRIALS=PLOT_TRIALS)
    fig.savefig(f"{SAVEDIR}/eventpca_{event_wind}-{bregion}-var_{var}-OV_{[str(x) for x in vars_others]}.pdf")


def plotwrapper_statespace_single_event_bregion(DF_PA_SPACES, SP, event_wind_pca, bregion, event_dat,
                             var_dat, vars_others_dat, time_windows_traj=None,
                             savedir=None, dims_pc = (0,1),
                             PLOT_TRIALS=False, n_trials_rand=10,
                             alpha_mean=0.5, alpha_trial=0.2):
    """
    event_wind_pca = "03_samp_0.04_0.6", bregion_pca = "vlPFC_p",
    """
    from pythonlib.tools.plottools import saveMultToPDF


    # Extract PApca
    tmp = DF_PA_SPACES[(DF_PA_SPACES["event_wind"]==event_wind_pca) & (DF_PA_SPACES["bregion"]==bregion)]
    assert len(tmp)==1
    PApca = tmp["PApca"].item()

    _, sites = _preprocess_extract_plot_params(SP, PApca)

    LIST_FIG = []

    ##### 1) Plot all on a single axis
    DICT_DF_DAT, _, _ = SP._statespace_pca_extract_data(sites, event_dat, var_dat, vars_others_dat)

    if len(DICT_DF_DAT)==0:
        return

    if len(DICT_DF_DAT.keys())<8: # Otherwise it is too crowded
        fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
        _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_traj,
            ax, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial, n_trials_rand)
        LIST_FIG.append(fig)

    ##### 2) Plot
    if vars_others_dat is not None:
        fig = plot_statespace_grpbyvarsothers(event_dat, PApca, SP, var_dat, vars_others_dat,
                                           time_windows_mean = time_windows_traj, alpha_mean=alpha_mean,
                                           PLOT_TRIALS=PLOT_TRIALS, dims_pc=dims_pc, n_trials_rand=n_trials_rand,
                                           alpha_trial=alpha_trial)
        LIST_FIG.append(fig)
        # if savedir:
        #     fig.savefig(f"{savedir}/eventpca_{event_wind}-{bregion}-var_{var_dat}-OV_{[str(x) for x in vars_others_dat]}-eventdat_{event_dat}-1.pdf")

        ##### 2) Plot,  FLip the order of var and vars_others
        if "vars_others_tmp" in SP.DfScalar.columns:
            fig = plot_statespace_grpbyvarsothers(event_dat, PApca, SP, "vars_others_tmp", [var_dat],
                                               time_windows_mean = time_windows_traj, alpha_mean=alpha_mean,
                                               PLOT_TRIALS=PLOT_TRIALS, dims_pc=dims_pc, n_trials_rand=n_trials_rand,
                                               alpha_trial=alpha_trial)
            LIST_FIG.append(fig)

            # if savedir:    
            #     fig.savefig(f"{savedir}/eventpca_{event_wind}-{bregion}-var_{var_dat}-OV_{[str(x) for x in vars_others_dat]}-eventdat_{event_dat}-2.pdf")

    if savedir is not None:
        if vars_others_dat is not None:
            vars_others_dat_str = "-".join(vars_others_dat)
        else:
            vars_others_dat_str = "None"
        path = f"{savedir}/eventpca-{event_wind_pca}|{bregion}|eventdat_{event_dat}|dims_{dims_pc}"
        saveMultToPDF(path, LIST_FIG)

def _plot_pca_results(PApca):

    fig, ax = plt.subplots()
    w = PApca.Saved["pca"]["w"]

    # Cum variance explained by traning data
    ax.plot(np.cumsum(w)/np.sum(w), '-or')
    ax.set_title("Training data")
    ax.set_title('cumulative var expl.')
    ax.hlines(0.9, 0, len(w))
    ax.set_ylim(0, 1)
    return fig


def _plot_variance_explained_timecourse(PApca, SP, event_dat, var, vars_others=None,
    time_windows=None, Dimslist = (0,1, 2)):
    """ Plot(overlay) timecourse of variance explained by each dimension, by 
    reprojecting data onto the subspace.
    """
    # 1) How much variance (across levels) accounted for by the first N pcs?
    # - keep the time window constant

    # assert var is None, "not yet coded"
    assert vars_others is None, "not yet coded"
    assert False, "might not be working, is not giving reasonable results..."

    sites = PApca.Chans
    DICT_DF_DAT, _, _ = SP._statespace_pca_extract_data(sites, event_dat)
    dfall = DICT_DF_DAT["alldata"]
    w = PApca.Saved["pca"]["w"]

    # get time windows and popanal
    PAdata = SP.dataextract_as_popanal_good(dfall, chans_needed=sites)

    if time_windows is not None:
        PAdata.X, PAdata.Times = PAdata.agg_by_time_windows(time_windows=time_windows)

    # agg: take mean for each level.
    PAdata = PAdata.slice_and_agg_wrapper("trials", [var])

    # compute total variance
    Vtot = np.var(PAdata.X, axis=1, keepdims=True) # (ndims, 1, ntimes)
    Vtot = np.sum(Vtot, axis=0, keepdims=True) # (1, 1, ntimes), i.e., total variance across neurons at each timepoint

    # project to low-D
    frmat = PAdata.X
    X = PApca.reprojectInput(frmat, Ndim=None, Dimslist=Dimslist)

    # compute variance across levels
    V = np.var(X, axis=1, keepdims=True) # (ndims, 1, ntimes)

    # normalize V to total variance (of unprojeted data)
    Vfrac = V/Vtot

    # plot
    fig, axes = plt.subplots(2,2, sharex=False, figsize=(2*3.5, 2*2.5))

    # - raw variance
    ax = axes.flatten()[0]
    ax.set_title("raw var explained, each dim")
    # ax.set_ylabel("frac var explained")
    for idim in range(V.shape[0]):
        ntimes = V.shape[2]
        ax.plot(PAdata.Times, V[idim, 0, :], label=idim)
    ax.legend()
    ax.set_ylim(bottom=0)
    # ax.set_ylim(bottom=0, top=np.sum(w))

    # - normalized variance
    ax = axes.flatten()[1]
    ax.set_title("var explained, each dim")
    ax.set_ylabel("frac var explained")
    from pythonlib.tools.plottools import makeColors
    pcols = makeColors(len(Dimslist))
    for idim, col in zip(range(Vfrac.shape[0]), pcols):
        ntimes = Vfrac.shape[2]
        ax.plot(PAdata.Times, Vfrac[idim, 0, :], color=col, label=idim)

        # - overlay frac variance explained by each dim in training data
        pcdim = Dimslist[idim]
    #     ax.axhline(w[pcdim], linestyle="--", label=f"trainingdat:{pcdim}", color=col)
        ax.axhline(w[pcdim], linestyle="--", color=col)
    ax.legend()
    ax.set_ylim(0, 1)

    # sum of normalized variance
    ax = axes.flatten()[2]
    ax.set_title("sum (across dims)")
    ax.set_ylabel("sum frac var explained")
    Vfrac_sum = np.sum(Vfrac, axis=0, keepdims=True)
    ax.plot(PAdata.Times, Vfrac_sum[0, 0,:], color="k", label=idim)
    ax.legend()
    ax.set_ylim(0, 1)

#     # - overlay frac variance explained by each dim in training data
#     pcdim = Dimslist[idim]
# #     ax.axhline(w[pcdim], linestyle="--", label=f"trainingdat:{pcdim}", color=col)
#     ax.axhline(w[pcdim], linestyle="--", color=col)

    # Cum variance explained by traning data
    ax = axes.flatten()[3]
    ax.plot(np.cumsum(w)/np.sum(w), '-or')
    ax.set_title("Training data")
    ax.set_title('cumulative var expl.')
    ax.hlines(0.9, 0, len(w))
    ax.set_ylim(0, 1)
    return fig

def _load_pca_space(pca_trial_agg_grouping, animal, DATE):
    SAVEDIR = f"/gorilla1/analyses/recordings/main/pca/{animal}/{DATE}/aggby_{pca_trial_agg_grouping}"

    path = f"{SAVEDIR}/DF_PA_SPACES.pkl"
    with open(path, "rb") as f:
        DF_PA_SPACES = pickle.load(f)

    path = f"{SAVEDIR}/params_pca_space.yaml"
    params_pca_space = load_yaml_config(path)

    print("Loaded this already computed PCA space, with these params:")
    print(params_pca_space)

    return DF_PA_SPACES, params_pca_space, SAVEDIR

def plot_variance_explained_timecourse(SP, animal, DATE, pca_trial_agg_grouping, bregion, event_wind_pca, event_dat):
    """ Plot timecourse of variance explained
    """
    list_vars = [
        "seqc_0_shape",
        "seqc_0_loc",
        "gridsize"
    ]
    list_vars_others = [
        ["seqc_0_loc", "gridsize"],
        ["seqc_0_shape", "gridsize"],
        ["seqc_0_shape", "seqc_0_loc"],
    ]
    vars_others_dat = None
    Dimslist=list(range(3))

    for pca_trial_agg_grouping in list_vars:
        DF_PA_SPACES, params_pca_space, SAVEDIR = _load_pca_space(pca_trial_agg_grouping, animal, DATE)

        tmp = DF_PA_SPACES[(DF_PA_SPACES["bregion"]==bregion) & (DF_PA_SPACES["event_wind"]==event_wind_pca)]
        assert len(tmp)==1
        PApca = tmp.iloc[0]["PApca"]

        for var_dat in list_vars:

            savedir = f"{SAVEDIR}/FIGS/var_explained_timecourse"
            import os
            os.makedirs(savedir, exist_ok=True)

            fig = _plot_variance_explained_timecourse(PApca, SP, event_dat, var_dat, Dimslist=Dimslist);

            path = f"{savedir}/eventpca-{event_wind_pca}|{bregion}|eventdat_{event_dat}|var_{var_dat}|{bregion}.pdf"
            fig.savefig(path)
    #         assert False

            print("--- SAVING AT:", f"{savedir}/var_{var_dat}.pdf")
        plt.close("all")


def _trajgood_make_colors_discrete_var(labels, which_dim_of_labels_to_use=None,
                                       force_continuous=False):
    """
    Helper to make colors for plotting, mapping from unque item
    in labels to rgba color. Can be continuous or discrete (and will
    check this automatically).
    PARAMS:
    - labels, values, either cont or discrete.
    - which_dim_of_labels_to_use, either NOne (use entire label) or int, which indexes into
    each lab in labels, and just uses that to determine the color. e.g,, (shape, loc) --> are the labels,
    and you just want to color by shape, then use which_dim_of_labels_to_use = 0.
    RETURNS:
    - dict,  mapping from value to color (if discrete), otherw sie None
    - color_type, str, either "cont" or "discrete".
    - colors, list of colors, matching input labels.
    """
    from pythonlib.tools.plottools import color_make_map_discrete_labels
    return color_make_map_discrete_labels(labels, which_dim_of_labels_to_use, force_continuous=force_continuous)
        
    # if which_dim_of_labels_to_use is None:
    #     labels_for_color = labels
    # else:
    #     labels_for_color = [lab[which_dim_of_labels_to_use] for lab in labels]
    # labels_color_uniq = sort_mixed_type(list(set(labels_for_color)))

    # if len(set([type(x) for x in labels_color_uniq]))>1:
    #     # more than one type...
    #     color_type = "discr"
    #     pcols = makeColors(len(labels_color_uniq))
    #     _map_lev_to_color = {}
    #     for lev, pc in zip(labels_color_uniq, pcols):
    #         _map_lev_to_color[lev] = pc
    # # continuous?
    # elif len(labels_color_uniq)>50 and isinstance(labels_color_uniq[0], (int)):
    #     color_type = "cont"
    #     # from pythonlib.tools.plottools import map_continuous_var_to_color_range as mcv
    #     # valmin = min(df[var_color_by])
    #     # valmax = max(df[var_color_by])
    #     # def map_continuous_var_to_color_range(vals):
    #     #     return mcv(vals, valmin, valmax)
    #     # label_rgbs = map_continuous_var_to_color_range(df[var_color_by])
    #     _map_lev_to_color = None
    # elif len(labels_color_uniq)>8 and isinstance(labels_color_uniq[0], (np.ndarray, float)):
    #     color_type = "cont"
    #     _map_lev_to_color = None
    # else:
    #     color_type = "discr"
    #     # label_rgbs = None
    #     pcols = makeColors(len(labels_color_uniq))
    #     _map_lev_to_color = {}
    #     for lev, pc in zip(labels_color_uniq, pcols):
    #         _map_lev_to_color[lev] = pc

    # # Return the color for each item
    # if _map_lev_to_color is None:
    #     colors = labels_color_uniq
    # else:
    #     colors = [_map_lev_to_color[lab] for lab in labels_for_color]

    # return _map_lev_to_color, color_type, colors

def trajgood_plot_colorby_splotbydims_scalar(X, labels_color, list_dims,
                                         overlay_mean=False, plot_text_over_examples=False,
                                         text_to_plot=None,
                                         alpha=0.5, SIZE=5):
    """ [GOOD], to plot scatter of pts, colored by one variable, and split across
    different slices (pairs of dimensions).
    PARAMS:
    - X, (npts, ndims)
    - labels_color, (npts,) discrete labels for coloring.
    - list_dims, list of 2-length iterables, holding dimensions .e,g [(0,1), (2,3)]
    Other columns are flexible, defnining varialbes.
    """
    from pythonlib.tools.plottools import makeColors
    from neuralmonkey.population.dimreduction import statespace_plot_single
    from pythonlib.tools.plottools import legend_add_manual
    from pythonlib.tools.plottools import plotScatterOverlay

    assert len(labels_color)==X.shape[0]
    map_lev_to_color, color_type, _ = _trajgood_make_colors_discrete_var(labels_color)

    # One subplot per othervar
    nsplots = len(list_dims)
    ncols = 3
    nrows = int(np.ceil(nsplots/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(ncols*SIZE, nrows*SIZE))
    for ax, dims in zip(axes.flatten(), list_dims):
        xs = X[:, dims[0]]
        ys = X[:, dims[1]]
        ax.set_title(dims)

        _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels_color, ax,
                                                map_lev_to_color, color_type,
                                                overlay_mean, plot_text_over_examples,
                                                text_to_plot, alpha, SIZE)

    return fig, axes

def trajgood_plot_colorby_groupby_meanscalar_BASE(ax, xs, ys, dflab, vars_mean,
                                                  colorby_ind_in_vars_mean):
    """
    Helper to plot means by grouping with <vars_mean> but optionally coloring the means using only one of the variables in vars_mean,
    indicated by <colorby_ind_in_vars_mean>
    :param xs:
    :param ys:
    :param dflab:
    :param vars_mean: list of str, e.g, (shape, gridloc)
    :param colorby_ind_in_vars_mean: int index into vars_mean, e.g,, 0 means color by shape, even though grouping by (shape, loc)
    :return:
    """

    if isinstance(vars_mean, str):
        vars_mean = [vars_mean]

    df = dflab.copy()
    df["x"] = xs
    df["y"] = ys
    # get colors
    # if colorby_ind_in_vars_mean is not None:
    var_color = vars_mean[colorby_ind_in_vars_mean]
    # else:
    #     var_color =

    labels_color_uniq = df[var_color].unique().tolist()
    pcols = makeColors(len(labels_color_uniq))
    map_var_to_col = {}
    for i, v in enumerate(df[var_color].unique()):
        map_var_to_col[v] = pcols[i]
    df["color"] = [map_var_to_col[v] for v in df[var_color]]

    dfmean = df.groupby(vars_mean).mean().reset_index(drop=True)
    for i, row in dfmean.iterrows():
        ax.plot(row["x"], row["y"], "s", color=row["color"], markersize=12)

def trajgood_plot_colorby_scalar_splitmeanlines(X, dflab, var_color, var_lines_within_subplot, vars_subplot,
                                                plot_method="overlay_mean_lines",
                                                desired_levels_var_color_in_order = None,
                                                map_linelev_to_color = None, dims = (0,1),
                                                n_min_across_all_levs_var=4,
                                                lenient_allow_data_if_has_n_levels=1,
                                                SIZE = 5):
    """
    GOOD -- Split subplots by one set of variables, and within each subplot split into lines by antoher set of
    variables, with those lines connecting mean pts that reflect levels of yet another variable (var_color).
    E.g. useful to look at consisdent encoding of states within each syntax_concrete, split by shape_chunks.
    Also options to either aggregate data by taking mean (as described above) or, inteeadf of overlaying lines
    within single subpltos (plot_method=overlay_mean_lines) insetad mnow each line is a different subplot row,
    and now var_subplot defines the columns.

    :param X: data (ntrials, ndims)
    :param dflab: dataframe with labels (len ntrials)
    :param var_color:
    :param var_lines_within_subplot:
    :param vars_subplot:
    :param desired_levels_var_color_in_order:
    :param map_lev_to_color:
    :param map_linelev_to_color:
    :param dims:
    :param n_min_across_all_levs_var:
    :param lenient_allow_data_if_has_n_levels:
    :param SIZE:
    :return:
    """


    from pythonlib.tools.listtools import sort_mixed_type
    from neuralmonkey.analyses.state_space_good import _trajgood_make_colors_discrete_var, _trajgood_plot_colorby_scalar_splitmeanlines
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    import numpy as np
    from pythonlib.tools.plottools import legend_add_manual

    #### Define colors for everything here, so that they are consistent across subplots and lines.
    # (0) What order to make the levels of var_color?
    if desired_levels_var_color_in_order is None:
        desired_levels_var_color_in_order = sort_mixed_type(list(dflab[var_color].unique()))
    # - get colors.
    map_lev_to_color, color_type, _ = _trajgood_make_colors_discrete_var(desired_levels_var_color_in_order)

    # (2) Get levels of lines within each subplot (across subplots)
    _dfout, _dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var_color, var_lines_within_subplot,
                                                                   n_min_across_all_levs_var=n_min_across_all_levs_var,
                                                                   lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels)
    # How to color lines (within subplots)?
    if map_linelev_to_color is None:
        # color each line by getting ordering the line classes
        levs_var_lines = sort_mixed_type(list(_dict_dfthis.keys()))
    # - colors
    map_linelev_to_color, _, _ = _trajgood_make_colors_discrete_var(levs_var_lines)

    # (1) Get subplot levels
    dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var_color, vars_subplot,
                                                                 n_min_across_all_levs_var=n_min_across_all_levs_var,
                                                                 lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels)
    levs_subplot = sort_mixed_type(list(dict_dfthis.keys()))

    # How to color dots (within lines)?
    # - Prune to just those that exist in data
    desired_levels_var_color_in_order = [lev for lev in desired_levels_var_color_in_order if lev in dflab[var_color].unique().tolist()]

    if plot_method=="overlay_mean_lines":
        # Then take mean --> then multiple lines in a single plot
        # Prep plots
        ncols = 5
        nplots = len(levs_subplot) + 1 # plus 1 for last plot of legend color for lines
        nrows = int(np.ceil(nplots/ncols))
        subplot_kw = None
        sharex, sharey = True, True
        # sharex, sharey = False, False
        fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                                 figsize=(ncols*SIZE, nrows*SIZE), subplot_kw=subplot_kw)
        ct = 0
        for ax, (grp_subplot, dfthis) in zip(axes.flatten(), dict_dfthis.items()):
            xthis = X[dfthis["_index"].tolist(),:]
            dflabthis = dfthis.reset_index(drop=True)

            # A single subplot
            _trajgood_plot_colorby_scalar_splitmeanlines(xthis, var_color, var_lines_within_subplot, dflabthis, ax,
                                                             desired_levels_var_color_in_order = desired_levels_var_color_in_order,
                                                             map_lev_to_color = map_lev_to_color,
                                                            map_linelev_to_color=map_linelev_to_color,
                                                             dims = dims, n_min_across_all_levs_var=n_min_across_all_levs_var)
            ax.set_title(grp_subplot)
            ct+=1

        # Plot legend of color for lines
        ax = axes.flatten()[ct]
        legend_add_manual(ax, list(map_linelev_to_color.keys()), list(map_linelev_to_color.values()))
        ax.set_title("Legend, for lines within subplots")
    elif plot_method=="separate_scatters":
        # Then is grid of subplots (line levels, subplot levels), and within each plot scatter.

        # Prep plots
        ncols = len(levs_subplot)
        nrows = len(levs_var_lines)
        subplot_kw = None
        sharex, sharey = True, True
        # sharex, sharey = False, False
        fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                                 figsize=(ncols*SIZE, nrows*SIZE), subplot_kw=subplot_kw)

        for i, (grp_subplot, dfthis_subplot) in enumerate(dict_dfthis.items()):
            for j, grp_line in enumerate(_dict_dfthis.keys()):
                ax = axes[j][i]
                # get this specific df
                dfthis_subplot = append_col_with_grp_index(dfthis_subplot, var_lines_within_subplot, "_line_var", use_strings=False)
                dfthisthis = dfthis_subplot[dfthis_subplot["_line_var"] == grp_line]

                # print(dfthis_subplot["_line_var"])
                # print(grp_line)
                # print(len(dfthisthis))
                # assert False
                if len(dfthisthis)>0:
                    # Get this specific data
                    xthis = X[dfthisthis["_index"].tolist(),:]
                    dflabthis = dfthisthis.reset_index(drop=True)

                    # print(xthis)
                    # print(dflabthis)
                    # assert False
                    # A single subplot
                    _trajgood_plot_colorby_scalar_splitmeanlines(xthis, var_color, var_lines_within_subplot, dflabthis, ax,
                                                                     desired_levels_var_color_in_order = desired_levels_var_color_in_order,
                                                                     map_lev_to_color = map_lev_to_color,
                                                                    map_linelev_to_color=map_linelev_to_color,
                                                                     dims = dims, n_min_across_all_levs_var=n_min_across_all_levs_var,
                                                                    plot_scatter=True)
                if i==0 and j==0:
                    ax.set_title(f"{grp_subplot} -- {grp_line}")
                elif i==0 and j>0:
                    ax.set_title(f"{grp_line}")
                elif j==0 and i>0:
                    ax.set_title(f"{grp_subplot}")

        # # Plot legend of color for lines
        # ax = axes.flatten()[ct]
        # legend_add_manual(ax, list(map_linelev_to_color.keys()), list(map_linelev_to_color.values()))
        # ax.set_title("Legend, for lines within subplots")

    else:
        print(plot_method)
        assert False

    return fig

def _trajgood_plot_colorby_scalar_splitmeanlines(X, var_color, var_lines_within_subplot, dflab, ax,
                                                 desired_levels_var_color_in_order = None,
                                                 map_lev_to_color = None, map_linelev_to_color = None,
                                                 dims = (0,1), n_min_across_all_levs_var=4,
                                                 lenient_allow_data_if_has_n_levels=1,
                                                 overlay_mean=True, plot_scatter=False,
                                                 alpha=0.4):
    """
    Low-level plotter for trajgood_plot_colorby_scalar_splitmeanlines. See docs therein.
    :param X:
    :param dflab:
    :param ax:
    :return:
    """
    mean_markersize = 6
    mean_alpha = 0.7

    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    assert X.shape[0]==len(dflab)
    dflab = dflab.copy()

    dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var_color, var_lines_within_subplot,
                                                             n_min_across_all_levs_var=n_min_across_all_levs_var,
                                                             lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels)

    if map_linelev_to_color is None:
        # color each line by getting ordering the line classes
        levs_var_lines = sort_mixed_type(list(dict_dfthis.keys()))
        map_linelev_to_color, _, _ = _trajgood_make_colors_discrete_var(levs_var_lines)

    # Plot each line
    for grp, dfthis in dict_dfthis.items():
        if desired_levels_var_color_in_order is None:
            desired_levels_var_color_in_order = sorted(dfthis[var_color].unique())
        inds = dfthis["_index"].tolist()
        xs = X[inds, dims[0]]
        ys = X[inds, dims[1]]
        labels = dfthis[var_color].tolist()

        line_color = map_linelev_to_color[grp]
        _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax, map_lev_to_color, overlay_mean=overlay_mean, plot_scatter=plot_scatter,
                                                connect_means_with_line=True, connect_means_with_line_levels=desired_levels_var_color_in_order,
                                                connect_means_with_line_color=line_color, mean_markersize=mean_markersize, mean_alpha=mean_alpha,
                                                alpha=alpha)

def _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels_color, ax,
                                            map_lev_to_color=None, color_type="discr",
                                            overlay_mean=False,
                                            plot_text_over_examples=False, text_to_plot=None,
                                            alpha=0.6, SIZE=5,
                                            connect_means_with_line=False, connect_means_with_line_levels=None,
                                            connect_means_with_line_color=None,
                                            plot_3D=False, zs=None,
                                            mean_markersize=10, mean_alpha=0.9, plot_scatter=True):
    """
    [LOW-LEVEL base plot for scatterplot]
    Like trajgood_plot_colorby_splotby_scalar, but passing in the raw data directly, instead
    of dataframe. Here constructs datafrane and runs for you.
    :param xs: (n, 1)
    :param ys: (n,1)
    :param labels_color: len(n) list
    :param labels_subplot: can be None to skip splitting by subplot
    :return:
    """
    from pythonlib.tools.plottools import plotScatterOverlay

    if len(xs.shape)==1:
        xs = xs[:, None]
    if len(ys.shape)==1:
        ys = ys[:, None]
    assert xs.shape[1]==1
    assert ys.shape[1]==1
    if zs is not None and len(zs.shape)==1:
        zs = zs[:, None]


    if plot_3D:
        # print(xs.shape, ys.shape, zs)
        X = np.concatenate((xs, ys, zs), axis=1)
        dimsplot = (0,1,2)
    else:
        X = np.concatenate((xs, ys), axis=1)
        dimsplot = (0,1)

    # Overwrite with user inputed
    if map_lev_to_color is None:
        map_lev_to_color, color_type, _ = _trajgood_make_colors_discrete_var(labels_color)

    # if plot_3D:
    #     x = X.T # (dims, datapts)
    #     pcol = "r"
    #     ax.plot(x[0,:], x[1,:], x[2,:], "o", color=pcol, alpha=alpha)
    #     ax.plot(x[0,-1], x[1,-1], x[2,-1], "s", mfc="w", color=pcol, alpha=0.4)
    #     ax.plot(x[0, 0], x[1,0], x[2,0], "c", mfc="w", color=pcol, alpha=0.4)
    #     ax.view_init(45, -70)
    #     ax.set_xlabel(f"dim 0")
    #     ax.set_ylabel(f"dim 1")
    #     ax.set_zlabel(f"dim 2")
    # else:
    plotScatterOverlay(X, labels_color, dimsplot, alpha=alpha, ax=ax, overlay_mean=overlay_mean,
                       plot_scatter=plot_scatter,
                       overlay_ci=False, plot_text_over_examples=plot_text_over_examples,
                       text_to_plot=text_to_plot, map_lev_to_color=map_lev_to_color,
                       SIZE=SIZE, color_type=color_type,
                       mean_markersize=mean_markersize, mean_alpha=mean_alpha)

    # if overlay_mean_which_dim_of_labels_to_use is not None:
    #     # Then overlay means using var for color that is not identical to var for grouping.
    #     if map_lev_to_color is None:
    #         _, _, colors = _trajgood_make_colors_discrete_var(labels_color, overlay_mean_which_dim_of_labels_to_use)
    #         # Collect means, in sequence
    #         Xmeans = []
    #         for lev in connect_means_with_line_levels:
    #             if lev in labels_color:
    #                 Xmeans.append(np.mean(X[labels_color==lev, :], axis=0))
    #         if len(Xmeans)>0:
    #             Xmeans = np.stack(Xmeans)
    #             # Plot
    #             ax.plot(Xmeans[:,0], Xmeans[:,1], "-")

    if connect_means_with_line:
        if connect_means_with_line_levels is None:
            # get sorted list of levels
            connect_means_with_line_levels = sorted(list(set(labels_color)))
        else:
            assert isinstance(connect_means_with_line_levels, (list, tuple))

        # Collect means, in sequence
        Xmeans = []
        for lev in connect_means_with_line_levels:
            if lev in labels_color:
                # print(type(lev))
                # print(type(labels_color))
                # print()
                # print(labels_color==lev)
                # assert False
                # print(sum(labels_color==lev))
                # print(X[labels_color==lev, :])
                # assert False
                Xmeans.append(np.mean(X[[l==lev for l in list(labels_color)], :], axis=0))

        if len(Xmeans)>0:
            Xmeans = np.stack(Xmeans)
            # Plot
            ax.plot(Xmeans[:,0], Xmeans[:,1], "-", color=connect_means_with_line_color)

def trajgood_plot_colorby_splotby_scalar_2dgrid_bregion(dfallpa, var_effect, var_other, savedir, 
                                                        pa_var = "pa_redu", prune_min_n_trials=3,
                                                        pretty_plot=False, alpha=0.6):
    """
    Make a 2d grid, where each row is a pa from dfallpa (usually different brain regions), 
    and each column is a level of var_other, and each dot is a trial, colored by var_effect
    """
    from pythonlib.tools.plottools import savefig

    nrows = len(dfallpa)

    _pa = dfallpa[pa_var].values[0]
    dflab = _pa.Xlabels["trials"]
    levels_col = dflab[var_other].unique().tolist()
    ncols = len(levels_col)
    SIZE =5

    # First, prune n trials
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    list_pa_redu = []
    for i, row in dfallpa.iterrows():
        bregion = row["bregion"]
        event = row["event"]
        PAredu = row[pa_var]     

        # Only keep effect that is present across all var conj
        dflab = PAredu.Xlabels["trials"]
        balance_no_missed_conjunctions = True
        prune_min_n_levs = 2
        plot_counts_heatmap_savepath = f"{savedir}/counts.pdf"
        dfout, _ = extract_with_levels_of_conjunction_vars(dflab, var_effect, [var_other],
                                                                n_min_across_all_levs_var=prune_min_n_trials,
                                                                lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                                                                prune_levels_with_low_n=True,
                                                                ignore_values_called_ignore=True,
                                                                plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                                                balance_no_missed_conjunctions=balance_no_missed_conjunctions)
        plt.close("all")
        list_pa_redu.append(PAredu.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True))
    dfallpa[f"{pa_var}_here"] = list_pa_redu

    ### Plot
    # for dims in [(0,1), (1,2), (2,3), (3,4)]:
    for dims in [(0,1), (1,2)]:
        for share_axes in [False, True]:
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), 
                                     sharex=share_axes, sharey=share_axes, squeeze=False)

            for i, row in dfallpa.iterrows():

                bregion = row["bregion"]
                event = row["event"]
                PAredu = row[f"{pa_var}_here"]     

                for j, lev_col in enumerate(levels_col):
                    try:
                        ax = axes[i][j]
                    except Exception as err:
                        print(axes)
                        print(i, j)
                        raise err
                    
                    ax.set_title((bregion, lev_col))

                    pa = PAredu.slice_by_labels_filtdict({var_other:[lev_col]})

                    # # Only keep effect that is present across all var conj
                    # from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
                    # dflab = pa.Xlabels["trials"]
                    # balance_no_missed_conjunctions = True
                    # prune_min_n_levs = 2
                    # plot_counts_heatmap_savepath = f"{savedir}/counts.pdf"
                    # dfout, _ = extract_with_levels_of_conjunction_vars(dflab, var_effect, [var_other],
                    #                                                         n_min_across_all_levs_var=prune_min_n_trials,
                    #                                                         lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                    #                                                         prune_levels_with_low_n=True,
                    #                                                         ignore_values_called_ignore=True,
                    #                                                         plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                    #                                                         balance_no_missed_conjunctions=balance_no_missed_conjunctions)
                    # plt.close("all")
                    # pa = pa.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)

                    if dims[1]<=pa.X.shape[0]-1:
                        xs = pa.X[dims[0], :, 0]
                        ys = pa.X[dims[1], :, 0]
                        # zs = pa.X[2, :, 0]
                        dflab = pa.Xlabels["trials"]
                        labels = dflab[var_effect].tolist()

                        # _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax, plot_3D=False, zs = zs)
                        _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax, alpha=alpha)

                    # Make it pretty
                    if pretty_plot:

                        if False: # Beucase if this on, then for some reason share_axes_row_or_col_of_subplots() fails
                            # INcrease fontsize
                            for item in ([ax.xaxis.label, ax.yaxis.label] +
                                        ax.get_xticklabels() + ax.get_yticklabels()):
                                item.set_fontsize(15)
                            
                        # if j>0:
                        #     from pythonlib.tools.plottools import naked_erase_axes
                        #     naked_erase_axes(ax)
                        # else:
                        #     # INcrease fontsize
                        #     for item in ([ax.xaxis.label, ax.yaxis.label] +
                        #                 ax.get_xticklabels() + ax.get_yticklabels()):
                        #         item.set_fontsize(20)
                        if i>0 or j>0:
                            try:
                                ax.get_legend().remove()
                            except AttributeError as err:
                                print("Skipping: ", err)
                                pass
                            
            # Must share axes, within bregino
            if share_axes==False:
                from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
                share_axes_row_or_col_of_subplots(axes, "row", "both")

            # Save
            savefig(fig, f"{savedir}/scatter-event={event}-dims={dims}-shareax={share_axes}.pdf")
            plt.close("all")
    
def trajgood_plot_colorby_splotby_scalar_WRAPPER(X, dflab, var_color, savedir,
                                                 vars_subplot=None, list_dims=None,
                                                 STROKES_BEH=None, STROKES_TASK=None,
                                                 n_min_per_levo=None,
                                                 overlay_mean=False, overlay_mean_var_color=None,
                                                 connect_means_with_line=False, connect_means_with_line_levels=None,
                                                 SIZE=7, alpha=0.5,
                                                 skip_subplots_lack_mult_colors=True, save_suffix=None,
                                                 plot_3D = False,
                                                 overlay_mean_orig = False,
                                                 plot_kde=False, force_continuous=False):
    """
    Final wrapper to make many plots, each figure showing supblots one for each levv of otehr var, colored
    by levels of var. Across figures, show different projections to dim pairs. And plot sepraerpte figuers for
    with and without strokes overlaid.

    :param X: scalar data, (ndat, nfeat)
    :param dflab:
    :param var_color:
    :param savedir:
    :param vars_subplot:
    :param list_dims: list of 2-tuples
    :param STROKES_BEH:
    :param STROKES_TASK:
    :param overlay_mean_var_color, str, which variable (in var_color, if it is list) to use for coloring. This doesnt affect
    the grouping of trails to compute means, which always uses var_color. Controls only how means are collored.
    :return:
    """
    from neuralmonkey.analyses.state_space_good import cleanup_remove_labels_ignore

    if save_suffix is not None:
        save_suffix = f"-{save_suffix}"
    else:
        save_suffix = ""

    if overlay_mean:
        assert isinstance(var_color, list)
        assert overlay_mean_var_color in var_color

    if overlay_mean_var_color is not None and overlay_mean:
        # Currently, overlay_mean_var_color must be subset of var_color
        if isinstance(var_color, str):
            assert overlay_mean_var_color==var_color
            colorby_ind_in_vars_mean = 0
        elif isinstance(var_color, (list, tuple)):
            assert overlay_mean_var_color in var_color
            colorby_ind_in_vars_mean = var_color.index(overlay_mean_var_color)
            assert colorby_ind_in_vars_mean > -1
    else:
        # Color variables are same as grouping variables.
        colorby_ind_in_vars_mean = None

    if len(X.shape)==1:
        # Then is 1-d. hack solution by adding a fake column
        X = np.stack([X, np.ones_like(X)], axis=1)

    if not len(X.shape)==2:
        print(X)
        print(X.shape)
        assert False, "prob catch this upstream and block."

    if not len(X)==len(dflab):
        print(X.shape)
        print(len(dflab))
        assert False

    if list_dims is None:
        list_dims = [(0,1,2)]

    # Make sure is lewngth 3
    list_dims = [list(dims)+[max(dims)+1] if len(dims)==2 else dims for dims in list_dims]

    list_dims_keep = []
    for dims in list_dims:
        # Determine if keep this dims
        keep = True
        for d in dims[:2]: # Just take 2, since 3rd is optional
            if X.shape[1]<d+1:
                print("Excluding this dims .. (not enough data)", dims, "data : ", X.shape)
                # print(X.shape)
                # print(list_dims)
                # assert False, "cannot get these dims"   
                keep = False
        if keep:
            list_dims_keep.append(dims)
    list_dims = list_dims_keep
    if len(list_dims)==0:
        return
    # assert len(list_dims)>0
    # else:
    #     if plot_3D:
    #         for dims in list_dims:
    #             assert len(dims)==3

    var_color_for_name = var_color
    if isinstance(var_color, (tuple, list)):
        dflab = append_col_with_grp_index(dflab, var_color, "_tmp")
        var_color = "_tmp"

    if isinstance(vars_subplot, str):
        vars_subplot = [vars_subplot]

    labels_color = dflab[var_color].tolist()
    # text_to_plot = labels_color
    text_to_plot = None

    if vars_subplot is None:
        labels_subplot = None
        vars_subplot_string = None
    else:
        # is a conjunctive var
        dflab = append_col_with_grp_index(dflab, vars_subplot, "_tmp", strings_compact=True)
        labels_subplot = dflab["_tmp"].tolist()
        vars_subplot_string = "|".join(vars_subplot)

    # One figure for each pair of dims
    for dim1, dim2, dim3 in list_dims:
        xs = X[:, dim1]
        ys = X[:, dim2]

        if plot_3D:
            zs = X[:, dim3]
        else:
            zs = None

        # Remove ignored labels.
        if False:
            # If use this, then trajgood_plot_colorby_groupby_meanscalar_BASE will fail (not alinged indices).
            xs, ys, labels_color, labels_subplot = cleanup_remove_labels_ignore(xs, ys, labels_color, labels_subplot)

        if len(xs)==0:
            continue

        # Without overlaid drawings.
        fig, axes, map_levo_to_ax, map_levo_to_inds = trajgood_plot_colorby_splotby_scalar(xs, ys,
                                                                                           labels_color, labels_subplot, var_color,
                                                                                           vars_subplot_string, SIZE=7,
                                                                                           alpha=alpha,
                                                                                           overlay_mean=overlay_mean_orig,
                                                                                           text_to_plot=text_to_plot,
                                                                                           skip_subplots_lack_mult_colors=skip_subplots_lack_mult_colors,
                                                                                           n_min_per_levo=n_min_per_levo,
                                                                                           connect_means_with_line=connect_means_with_line,
                                                                                           connect_means_with_line_levels=connect_means_with_line_levels,
                                                                                           plot_3D=plot_3D, zs=zs,
                                                                                           plot_kde=plot_kde,
                                                                                           force_continuous=force_continuous)

        # Overlay means, including option to use one set of variables for grouping, and a subset of those variables for coloring.
        if overlay_mean and colorby_ind_in_vars_mean is not None:
            for levo, ax in map_levo_to_ax.items():
                inds = map_levo_to_inds[levo]
                xsthis = xs[inds]
                ysthis = ys[inds]
                dflabthis = dflab.loc[inds, var_color_for_name]
                # print(type(dflab))
                # print(type(dflabthis.reset_index(drop=True)))
                # assert False
                trajgood_plot_colorby_groupby_meanscalar_BASE(ax, xsthis, ysthis, dflabthis, var_color_for_name,
                                                              colorby_ind_in_vars_mean)

        # Save
        path = f"{savedir}/color={var_color_for_name}-sub={vars_subplot_string}-dims={dim1, dim2}{save_suffix}.pdf"
        print("fig:", path)
        if fig is not None:
            print("Saving ... ", path)
            savefig(fig, path)

        # With drawings
        if STROKES_BEH is not None or STROKES_TASK is not None:
            fig, axes, map_levo_to_ax, map_levo_to_inds = trajgood_plot_colorby_splotby_scalar(xs, ys,
                                                                                               labels_color, labels_subplot, var_color,
                                                                                               vars_subplot_string, SIZE=7, alpha=0.2,
                                                                                               overlay_mean=False, text_to_plot=text_to_plot,
                                                                                               STROKES_BEH=STROKES_BEH, STROKES_TASK=STROKES_TASK,
                                                                                               n_strokes_overlay_per_lev=3,
                                                                                               skip_subplots_lack_mult_colors=skip_subplots_lack_mult_colors,
                                                                                               n_min_per_levo=n_min_per_levo,
                                                                                               plot_3D=plot_3D)
            if fig is not None:
                path = f"{savedir}/color={var_color_for_name}-sub={vars_subplot_string}-dims={dim1, dim2}{save_suffix}-STROKES_OVERLAY.pdf"
                print("Saving ... ", path)
                savefig(fig, path)

        plt.close("all")


def _trajgood_construct_df_from_PAscal(PA, var_effect, vars_others):
    """
    Given array/list inputs, each same length, constructs df useful for 
    downstream plots, where each row is a single scalar value (x, y, and potentially z).
    """
    Xredu = PA.X # (chans, trials, 1)
    dflab = PA.Xlabels["trials"]
    assert Xredu.shape[2]==1
    x = Xredu.squeeze().T # (trials, chans)
    xs = x[:, 0]
    ys = x[:, 1]
    labels_color = dflab[var_effect].tolist()
    labels_subplot = [tuple(x) for x in dflab[vars_others].values.tolist()]
    dfthis = _trajgood_construct_df(xs, ys, labels_color, labels_subplot)
    return dfthis

def _trajgood_construct_df(xs, ys, labels_color, labels_subplot,
                           color_var="color", subplot_var="subplot", 
                           plot_3D=False,  zs=None):
    """
    Given array/list inputs, each same length, constructs df useful for 
    downstream plots, where each row is a single scalar value (x, y, and potentially z).

    """

    if False: 
        # I had this code previously, but doesnt seem like I need it.
        # Make this True if plotting fails.
        if len(xs.shape)==1:
            xs = xs[:, None]
        if len(ys.shape)==1:
            ys = ys[:, None]
        if zs is not None and len(zs.shape)==1:
            zs = zs[:, None]
            assert zs.shape[1]==1

        assert xs.shape[1]==1
        assert ys.shape[1]==1

    if labels_subplot is None:
        # subplot_var = None
        tmp = {
            color_var:labels_color,
            # "x":xs.tolist(),
            # "y":ys.tolist(),
            "x":xs,
            "y":ys,
        }
    else:
        tmp = {
            color_var:labels_color,
            subplot_var:labels_subplot,
            # "x":xs.tolist(),
            # "y":ys.tolist(),
            "x":xs,
            "y":ys,
        }

    if plot_3D:
        tmp["z"] = zs
        # tmp["z"] = zs.tolist()

    # print(tmp["x"][0].shape)
    # print(tmp["y"][0].shape)
    # print(tmp["z"][0].shape)
    # print(tmp["x"][:2])
    # # print(tmp["y"][0].shape)
    # print(tmp["z"][:2])
    # assert False

    dfthis = pd.DataFrame(tmp)

    return dfthis


def trajgood_plot_colorby_splotby_scalar(xs, ys, labels_color, labels_subplot,
                                         color_var, subplot_var,
                                         overlay_mean=False,
                                         plot_text_over_examples=False,
                                         text_to_plot=None,
                                         alpha=0.5, SIZE=5,
                                         STROKES_BEH = None,
                                         STROKES_TASK = None,
                                         n_strokes_overlay_per_lev = 4,
                                         skip_subplots_lack_mult_colors = False,
                                         n_min_per_levo=None,
                                       connect_means_with_line=False,
                                       connect_means_with_line_levels=None,
                                         plot_3D=False, zs=None,
                                         plot_kde=False, force_continuous=False
                                         ):
    """
    Like trajgood_plot_colorby_splotby_scalar, but passing in the raw data directly, instead
    of dataframe. Here constructs datafrane and runs for you.

    To ignore any variable, set eithe rhte labels=None or the variable=None

    :param xs: (n, 1)
    :param ys: (n,1)
    :param labels_color: len(n) list
    :param labels_subplot: can be None to skip splitting by subplot
    :param STROKES_BEH and STROKES_TASK: overlays strokes on figure.
    :params skip_subplots_lack_mult_colors, bool, if True, then only plots subplots if there are >1
    classes (for categorical variables only).
    :return:
    """
    from pythonlib.drawmodel.strokePlots import overlay_stroke_on_plot_mult_rand

    if True:
        dfthis = _trajgood_construct_df(xs, ys, labels_color, labels_subplot,
                            color_var, subplot_var, plot_3D, zs)
    else:
        if len(xs.shape)==1:
            xs = xs[:, None]
        if len(ys.shape)==1:
            ys = ys[:, None]
        if zs is not None and len(zs.shape)==1:
            zs = zs[:, None]
            assert zs.shape[1]==1

        assert xs.shape[1]==1
        assert ys.shape[1]==1

        if labels_subplot is None:
            subplot_var = None
            tmp = {
                color_var:labels_color,
                "x":xs.tolist(),
                "y":ys.tolist()
            }
        else:
            tmp = {
                color_var:labels_color,
                subplot_var:labels_subplot,
                "x":xs.tolist(),
                "y":ys.tolist()
            }
        if plot_3D:
            tmp["z"] = zs.tolist()
        # print(tmp["x"][0].shape)
        # print(tmp["y"][0].shape)
        # print(tmp["z"][0].shape)
        # print(tmp["x"][:2])
        # # print(tmp["y"][0].shape)
        # print(tmp["z"][:2])
        # assert False
        dfthis = pd.DataFrame(tmp)
    
    fig, axes, map_levo_to_ax, map_levo_to_inds = _trajgood_plot_colorby_splotby_scalar(dfthis, color_var, subplot_var,
                                                overlay_mean, plot_text_over_examples,
                                                text_to_plot, alpha, SIZE,
                                                skip_subplots_lack_mult_colors=skip_subplots_lack_mult_colors,
                                                n_min_per_levo=n_min_per_levo,
                                                connect_means_with_line=connect_means_with_line,
                                                connect_means_with_line_levels=connect_means_with_line_levels,
                                                plot_3D=plot_3D, plot_kde=plot_kde,
                                                force_continuous=force_continuous)

    if fig is None:
        return None, None, None, None

    ### OVERLAY strokes, if passed in
    LIST_STROKES_PLOT = []
    if STROKES_BEH is not None:
        LIST_STROKES_PLOT.append([STROKES_BEH, "onset", "k", n_strokes_overlay_per_lev])
    if STROKES_TASK is not None:
        LIST_STROKES_PLOT.append([STROKES_TASK, "center", "m", 1])
    if len(LIST_STROKES_PLOT)>0:
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        # Is this discrete?
        _, color_type, _ = _trajgood_make_colors_discrete_var(labels_color)
        if color_type=="discr":
            dflab = pd.DataFrame({color_var:labels_color, subplot_var:labels_subplot})
            grpdict = grouping_append_and_return_inner_items(dflab, [color_var, subplot_var])
            levels_var_color = dflab[color_var].unique().tolist()

            for strokes, align_to, pcol, n_rand in LIST_STROKES_PLOT:

                # Overlay random strokes, sampling within each level of labels_color
                for levo, ax in map_levo_to_ax.items():
                    for lev in levels_var_color:
                        if (lev, levo) in grpdict.keys():
                            inds = grpdict[(lev, levo)]
                            overlay_stroke_on_plot_mult_rand([strokes[i] for i in inds], xs[inds], ys[inds], ax,
                                                             n_rand, align_to, color=pcol)
        elif color_type=="cont":
            for strokes, align_to, pcol, n_rand in LIST_STROKES_PLOT:
                # Continuous -- change colro to not interfere
                if pcol=="k":
                    pcol = "g"
                elif pcol=="m":
                    pcol = "c"
                for levo, ax in map_levo_to_ax.items():
                    # Overlay nplot random strokes
                    inds = map_levo_to_inds[levo]
                    nplot = min([30, n_rand*5])
                    overlay_stroke_on_plot_mult_rand([strokes[i] for i in inds], xs[inds], ys[inds], ax, nplot, align_to, color=pcol)
        else:
            assert False

    return fig, axes, map_levo_to_ax, map_levo_to_inds


def _trajgood_plot_colorby_splotby_scalar(df, var_color_by, var_subplots,
                                         overlay_mean=False,
                                          plot_text_over_examples=False,
                                         text_to_plot=None,
                                         alpha=0.5, SIZE=5,
                                         skip_subplots_lack_mult_colors=False,
                                          n_min_per_levo=None,
                                          connect_means_with_line=False,
                                          connect_means_with_line_levels=None,
                                          plot_3D=False,
                                          plot_kde=False, kde_plot_scatter=True, kde_ellipses=True, kde_text_labels=False, kde_plot_contours=True, kde_levels=6,
                                          force_continuous=False):
    """ [GOOD], to plot scatter of pts, colored by one variable, and split across
    subplots by another variable.
    PARAMS:
    - df, standard form for holding trajectories, each row holds; one condition:
    --- "x", each a (1,) array, the value to plot on x coord (e.g,, dim1)
    --- "y", see "x"
    - var_subplots, None if no subplots
    Other columns are flexible, defnining varialbes.
    """
    from pythonlib.tools.plottools import makeColors
    from neuralmonkey.population.dimreduction import statespace_plot_single
    from pythonlib.tools.plottools import legend_add_manual
    from pythonlib.tools.plottools import plotScatterOverlay

    if n_min_per_levo is None:
        n_min_per_levo = 4
    # Color the labels
    # One color for each level of effect var

    labellist = df[var_color_by].unique().tolist()
    map_lev_to_color, color_type, _ = _trajgood_make_colors_discrete_var(labellist, force_continuous=force_continuous)

    # If you pass in continuous variable as othervar, then overwrite that and just plot a single plot.
    if var_subplots is not None:
        _, tmp, _ = _trajgood_make_colors_discrete_var(df[var_subplots].tolist())
        if tmp!="discr":
            # Overwrite input
            var_subplots = None

    # # continuous?
    # from pythonlib.tools.plottools import makeColors
    # if len(labellist)>50 and isinstance(labellist[0], (int, np.ndarray, float)):
    #     color_type = "cont"
    #     # from pythonlib.tools.plottools import map_continuous_var_to_color_range as mcv
    #     # valmin = min(df[var_color_by])
    #     # valmax = max(df[var_color_by])
    #     # def map_continuous_var_to_color_range(vals):
    #     #     return mcv(vals, valmin, valmax)
    #     # label_rgbs = map_continuous_var_to_color_range(df[var_color_by])
    #     map_lev_to_color = None
    # else:
    #     color_type = "discr"
    #     # label_rgbs = None
    #     pcols = makeColors(len(labellist))
    #     map_lev_to_color = {}
    #     for lev, pc in zip(labellist, pcols):
    #         map_lev_to_color[lev] = pc

    if var_subplots is None:
        # dummy
        df["_dummy"] = "dummy"
        var_subplots = "_dummy"

    # One subplot per othervar
    levs_other = sort_mixed_type(df[var_subplots].unique().tolist())

    if skip_subplots_lack_mult_colors and color_type=="discr":
        # Keep only subplots with >1 color and >n datapts total
        levs_other = [levo for levo in levs_other if len(df[df[var_subplots] == levo][var_color_by].unique())>1]
        levs_other = [levo for levo in levs_other if len(df[df[var_subplots] == levo][var_color_by])>=n_min_per_levo]

    if len(levs_other)==0:
        return None, None, None, None

    max_n_subplots = 30
    if len(levs_other)>max_n_subplots:
        # sort by n datapts, and take the top n
        if True:
            tmp = [(levo, sum(df[var_subplots]==levo)) for levo in levs_other]
            tmp = sorted(tmp, key = lambda x:-x[1])
            levs_other = [x[0] for x in tmp][:max_n_subplots]
        else:
            import random
            print("[trajgood_plot_colorby_splotby_scalar], too many subplots", len(levs_other), "...")
            levs_other = sort_mixed_type(random.sample(levs_other, max_n_subplots))
            print("... pruned to: ", len(levs_other))

    ncols = 5
    nrows = int(np.ceil(len(levs_other)/ncols))

    if plot_3D:
        subplot_kw = dict(projection='3d')
        # sharex, sharey = False, False
        share_axes = False
    else:
        subplot_kw = None
        share_axes = True
        # sharex, sharey = True, True
    sharex, sharey = False, False
    fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                             figsize=(ncols*SIZE, nrows*SIZE), subplot_kw=subplot_kw)

    map_levo_to_inds = {}
    map_levo_to_ax ={}
    for ax, levo in zip(axes.flatten(), levs_other):
        ax.set_title(levo)
        dfthis = df[df[var_subplots]==levo]
        map_levo_to_inds[levo] = dfthis.index.tolist()
        map_levo_to_ax[levo] = ax

        if text_to_plot is not None:
            # df[df[var_subplots]==levo].index.tolist()
            text_to_plot_this = np.array(text_to_plot)[df[var_subplots]==levo].tolist()
        else:
            text_to_plot_this = None
        xs = np.stack(dfthis["x"])
        ys = np.stack(dfthis["y"])
        if plot_3D:
            zs = np.stack(dfthis["z"])
        else:
            zs = None
        labels_color = dfthis[var_color_by].values

        if plot_kde:
            from pythonlib.tools.pandastools import plot_class_kde
            plot_class_kde(dfthis, "x", "y", var_color_by, levels=kde_levels, ax=ax, 
                           text_labels=kde_text_labels, scatter=kde_plot_scatter,
                           ellipses=kde_ellipses, cmap_per_class=map_lev_to_color,
                           plot_contours=kde_plot_contours)
        else:
            _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels_color, ax,
                                                    map_lev_to_color, color_type,
                                                    overlay_mean, plot_text_over_examples,
                                                    text_to_plot_this, alpha, SIZE,
                                                    connect_means_with_line=connect_means_with_line,
                                                    connect_means_with_line_levels=connect_means_with_line_levels,
                                                    plot_3D=plot_3D, zs=zs)
    if share_axes:
        from pythonlib.tools.plottools import share_axes
        share_axes(fig.axes, "both")

    if plot_3D:
        xs = np.stack(df["x"])
        ys = np.stack(df["y"])
        zs = np.stack(df["z"])

        # set some params for 3d plot.
        for ax in axes.flatten():
            ax.view_init(45, -70)
            ax.set_xlabel(f"dim 0")
            ax.set_ylabel(f"dim 1")
            ax.set_zlabel(f"dim 2")

            ax.set_xlim([min(xs), max(xs)])
            ax.set_xlim([min(ys), max(ys)])
            ax.set_xlim([min(zs), max(zs)])

    return fig, axes, map_levo_to_ax, map_levo_to_inds

def trajgood_construct_df_from_raw(X, times, labels, labelvars):
    """
    Generate df that can pass into all trajgood plotting functions.
    PARAMS:
    - X, neural data, shape (chans, trials, times).
    - times, timestaps, matches X.shape[2]
    - labels, list of tuples, one for each trial, holding the value of labels for that trial.
    - labelvars, list of str, the names of the label variables, matching the order within each
    tuple in labels.
    RETURNS:
    - df, standard form for holding trajectories, each row holds; one condition (e.g., shape,location):
    --- "z", activity (ndims, ntrials, ntimes),
    --- "z_scalar", scalarized version (ndims, ntrials, 1) in "z_scalar".
    --- "times", matching ntimes
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

    assert len(times)==X.shape[2]

    dflab = pd.DataFrame(labels, columns=labelvars)

    # Get indices for each grouping level
    groupdict = grouping_append_and_return_inner_items(dflab, labelvars, sort_keys=True)

    # Get sliced pa for each grouping level
    out = []
    for grp in groupdict:
        inds = groupdict[grp]

        z = X[:, inds, :]
        z_scalar = np.mean(z, axis=2, keepdims=True)

        tmp = {}
        for lev, var in zip(grp, labelvars):
            tmp[var] = lev
        tmp["z"] = z
        tmp["z_scalar"] = z_scalar
        tmp["times"] = times

        out.append(tmp)

    df = pd.DataFrame(out)
    return df

def trajgood_plot_colorby_splotby_WRAPPER(X, times, dflab, var_color, savedir,
                                                 vars_subplot=None, list_dims=None,
                                                time_bin_size = None, alpha=0.5,
                                          save_suffix=None, mean_over_trials=True,
                                          ntrials=5, cont_color_kind="circular",
                                          plot_dots_on_traj=True,
                                          xlim_force=None, ylim_force=None):
    """
    NOTE: Only useful if var_color is categorical!! 
    :param X: neural data, shape (chans, trials, times).
    :param times: timestaps, matches X.shape[2]
    :param time_bin_size: time in sec, to make plot easier to visaulize, you can bin in time.
    :param dflab:
    :return:
    """

    # assert False, "good, but havent tested"

    for dims in list_dims:
        
        if X.shape[0] < max(dims)+1:
            continue

        # 1) Construct dataframe
        if vars_subplot is None:
            if isinstance(var_color, (list, tuple)):
                grpvars = list(var_color)
            else:
                grpvars = [var_color]
        else:
            if isinstance(var_color, (list, tuple)):
                grpvars = list(var_color) + list(vars_subplot)
            else:
                grpvars = [var_color] + list(vars_subplot)
        labels = dflab.loc[:, grpvars]
        labelvars = grpvars
        df = trajgood_construct_df_from_raw(X, times, labels, labelvars)

        # 2) Plot
        if times[0]<0. and times[-1]>0.:
            times_to_mark = [0.] # you can mark specific times on the plot. here marks the 0. sec mark.
            times_to_mark_markers = ["x"] # mark with a diamond ("d")
        else:
            times_to_mark = None
            times_to_mark_markers = None

        fig, axes = trajgood_plot_colorby_splotby(df, var_color, vars_subplot, dims, "traj", mean_over_trials=mean_over_trials,
                                      times_to_mark = times_to_mark, times_to_mark_markers = times_to_mark_markers,
                                      time_bin_size=time_bin_size, alpha=alpha, ntrials=ntrials,
                                      plot_dots_on_traj=plot_dots_on_traj, xlim_force=xlim_force, ylim_force=ylim_force)

        path = f"{savedir}/color={var_color}-sub={vars_subplot}-dims={dims[0], dims[1]}-suff={save_suffix}.pdf"
        print("Saving ... ", path)
        savefig(fig, path)

        plt.close("all")

def trajgood_plot_colorby_splotby(df, var_color_by, var_subplots, dims=(0,1),
                                  traj_or_scalar="traj", mean_over_trials=True,
                                  times_to_mark=None,
                                   times_to_mark_markers=None,
                                   time_bin_size=None,
                                   markersize=6, marker="o",
                                   text_plot_pt1=None,
                                   alpha=0.5,
                                   ntrials=5, plot_dots_on_traj=True,
                                   overlay_trials_on_mean=False, 
                                   n_trials_overlay_on_mean=5,
                                   xlim_force=None, ylim_force=None, SIZE=3.5,
                                   ncols=4):
    """ [GOOD], to plot trajectories colored by one variable, and split across subplots by another
    variable.
    PARAMS:
    - df, standard form for holding trajectories, each row holds; one condition (e.g., shape,location):
    --- "z", activity (ndims, ntrials, ntimes),
    --- "z_scalar", scalarized version (ndims, ntrials, 1) in "z_scalar".
    --- "times", matching ntimes
    - mean_over_trials, bool, if True, then plots mean, if False, plots ntrials random trials.
    Other columns are flexible, defnining varialbes. must have var_color_by, var_subplots
    """
    from pythonlib.tools.plottools import makeColors
    from neuralmonkey.population.dimreduction import statespace_plot_single
    from pythonlib.tools.plottools import legend_add_manual

    if var_subplots is None:
        df = df.copy()
        df["_dummy"] = "dummy"
        var_subplots = ["_dummy"]

    var_color_for_name = var_color_by
    if isinstance(var_color_by, (tuple, list)):
        df = append_col_with_grp_index(df, var_color_by, "_tmp")
        var_color_by = "_tmp"

    var_color_for_name_subplots = var_subplots
    if isinstance(var_subplots, (tuple, list)):
        df = append_col_with_grp_index(df, var_subplots, "_tmp2")
        var_subplots = "_tmp2"

    # One color for each level of effect var
    levs_effect = sort_mixed_type(df[var_color_by].unique().tolist())
    pcols = makeColors(len(levs_effect))
    map_lev_to_color = {}
    for lev, pc in zip(levs_effect, pcols):
        map_lev_to_color[lev] = pc

    # One subplot per othervar
    levs_other = sort_mixed_type(df[var_subplots].unique().tolist())
    nrows = int(np.ceil(len(levs_other)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*SIZE, nrows*SIZE), squeeze=False)
    for ax, levo in zip(axes.flatten(), levs_other):
        ax.set_title(levo)
        dfthis = df[df[var_subplots]==levo]
        # Plot each row
        for _, row in dfthis.iterrows():

            if traj_or_scalar=="traj":
                X = row["z"] # (dims, trials, times)
            else:
                X = row["z_scalar"] # (dims, trials, 1)
                # times_to_mark = [0]
                # times_to_mark_markers = ["d"]
                # time_bin_size = 0.05

            times = row["times"]
            color_for_trajectory = map_lev_to_color[row[var_color_by]]

            if mean_over_trials:
                x = X[dims, :] # (dims, trials, times)
                assert len(x.shape)==3
                x = np.mean(x, axis=1)

                statespace_plot_single(x, ax, color_for_trajectory,
                                       times, times_to_mark, times_to_mark_markers,
                                       time_bin_size = time_bin_size,
                                       markersize=markersize, marker=marker,
                                       text_plot_pt1=text_plot_pt1, alpha=alpha, plot_dots_on_traj=plot_dots_on_traj)
                if overlay_trials_on_mean:
                    # Optionally overlay some single trials, so show variability
                    n = X.shape[1]
                    n_trials_overlay_on_mean = 5
                    if n>n_trials_overlay_on_mean:
                        import random
                        trials_get = random.sample(range(n), n_trials_overlay_on_mean)
                    else:
                        trials_get = range(n)
                    
                    for tr in trials_get:
                        x = X[dims, tr, :] # (dims, times)
                        statespace_plot_single(x, ax, color_for_trajectory,
                                            times, times_to_mark, times_to_mark_markers,
                                            time_bin_size = time_bin_size,
                                            markersize=markersize, marker="",
                                            text_plot_pt1=None, alpha=0.2, plot_dots_on_traj=False)
            else:
                # Loop over all trials
                # Pick subset of trials
                n = X.shape[1]
                if n>ntrials:
                    import random
                    trials_get = random.sample(range(n), ntrials)
                else:
                    trials_get = range(n)
                
                for tr in trials_get:
                    x = X[dims, tr, :] # (dims, times)
                    statespace_plot_single(x, ax, color_for_trajectory,
                                           times, times_to_mark, times_to_mark_markers,
                                           time_bin_size = time_bin_size,
                                           markersize=markersize, marker=marker,
                                           text_plot_pt1=text_plot_pt1, alpha=alpha, plot_dots_on_traj=plot_dots_on_traj)
        if xlim_force is not None:
            ax.set_xlim(xlim_force)
        if ylim_force is not None:
            ax.set_ylim(ylim_force)
        plt.grid(False)

    # Add legend to the last axis
    legend_add_manual(ax, map_lev_to_color.keys(), map_lev_to_color.values(), 0.2, "best")

    return fig, axes

def trajgood_plot_colorby_splotby_timeseries(df, var_color_by, var_subplots, 
                                             dim=0, plot_trials=True, plot_trials_n=10,
                                             plot_mean=True, 
                                             alpha=0.5, SUBPLOT_OPTION="combine_levs"):
    """ 
    NOTE: OBSOLETE -- SHOULD replace with:
    pa.plotwrappergrid_smoothed_fr_splot_neuron (if SUBPLOT_OPTION="combine_levs")
    pa.plotwrappergrid_smoothed_fr_splot_var (if SUBPLOT_OPTION="split_levs")

    [GOOD], to plot trajectories colored by one variable, and split across subplots by another
    variable, where x axis is always time.

    PARAMS:
    - df, standard form for holding trajectories, each row holds; one condition (e.g., shape,location):
    --- "z", activity (ndims, ntrials, ntimes),
    --- "z_scalar", scalarized version (ndims, ntrials, 1) in "z_scalar".
    --- "times", matching ntimes
    - mean_over_trials, bool, if True, then plots mean, if False, plots ntrials random trials.
    Other columns are flexible, defnining varialbes. must have var_color_by, var_subplots
    """
    from pythonlib.tools.plottools import makeColors
    from neuralmonkey.population.dimreduction import statespace_plot_single
    from pythonlib.tools.plottools import legend_add_manual
    from neuralmonkey.neuralplots.population import plot_smoothed_fr

    if var_subplots is None:
        df = df.copy()
        df["_dummy"] = "dummy"
        var_subplots = ["_dummy"]

    var_color_for_name = var_color_by
    if isinstance(var_color_by, (tuple, list)):
        df = append_col_with_grp_index(df, var_color_by, "_tmp")
        var_color_by = "_tmp"

    var_color_for_name_subplots = var_subplots
    if isinstance(var_subplots, (tuple, list)):
        df = append_col_with_grp_index(df, var_subplots, "_tmp2")
        var_subplots = "_tmp2"

    # One color for each level of effect var
    levs_effect = sort_mixed_type(df[var_color_by].unique().tolist())
    pcols = makeColors(len(levs_effect))
    map_lev_to_color = {}
    for lev, pc in zip(levs_effect, pcols):
        map_lev_to_color[lev] = pc

    # One subplot per othervar
    levs_other = sort_mixed_type(df[var_subplots].unique().tolist())

    if SUBPLOT_OPTION=="combine_levs":
        ncols = 3
        nrows = int(np.ceil(len(levs_other)/ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))
        for ax, levo in zip(axes.flatten(), levs_other):
            ax.set_title(levo)
            dfthis = df[df[var_subplots]==levo]

            # # Optionally sample a fixed n for each level of var_color
            # from pythonlib.tools.pandastools import extract_trials_spanning_variable
            # dfthis = dfthis.reset_index(drop=True)
            # inds = extract_trials_spanning_variable(dfthis, var_color_by, None, 1)
            # dfthis = dfthis.iloc[inds]

            # Plot each row (each level of effect)
            for _, row in dfthis.iterrows():

                X = row["z"] # (dims, trials, times)
                times = row["times"]
                color_for_trajectory = map_lev_to_color[row[var_color_by]]

                if X.shape[0]>dim:
                    if plot_trials:
                        if X.shape[1]>plot_trials_n:
                            import random
                            _inds = random.sample(range(X.shape[1]), plot_trials_n)
                            Xthis = X[:, _inds, :]
                        else:
                            Xthis = X
                        for _trial in range(Xthis.shape[1]):
                            ax.plot(times, Xthis[dim, _trial, :], "-", color=color_for_trajectory, alpha=alpha)
                    
                    if plot_mean:
                        plot_smoothed_fr(X[dim, :, :], times, ax, color=color_for_trajectory)
                        # Xmean = np.mean(X[dim, :, :], axis=0)
                        # ax.plot(times, Xmean, "-", color=color_for_trajectory, linewidth=4, alpha=0.7)

    # OPTION 2 - grid of subplots (var vs. othervar)
    elif SUBPLOT_OPTION=="split_levs":
        ncols = len(levs_effect)
        nrows = len(levs_other)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5),
                                 squeeze=False)
        for i, levo in enumerate(levs_other):
            for j, leveff in enumerate(levs_effect):

                ax = axes[i][j]
                dfthis = df[(df[var_subplots]==levo) & (df[var_color_by]==leveff)]
                if i==0 and j==0:
                    ax.set_title(f"{leveff} - {levo}", fontsize=6)
                elif i==0:
                    ax.set_title(f"{leveff}", fontsize=6)
                elif j==0:
                    ax.set_title(f"{levo}", fontsize=6)
                    

                if len(dfthis)==0:
                    continue
                elif len(dfthis)>1:
                    print(dfthis)
                    assert False
                else:                    
                    X = dfthis["z"].values[0] # (dims, trials, times)
                    times = dfthis["times"].values[0]
                    color_for_trajectory = map_lev_to_color[leveff]
                    if X.shape[0]>dim:
                        if plot_trials:
                            if X.shape[1]>plot_trials_n:
                                import random
                                _inds = random.sample(range(X.shape[1]), plot_trials_n)
                                Xthis = X[:, _inds, :]
                            else:
                                Xthis = X
                            for _trial in range(Xthis.shape[1]):
                                ax.plot(times, Xthis[dim, _trial, :], "-", alpha=alpha)
                        
                        if plot_mean:
                            plot_smoothed_fr(X[dim, :, :], times, ax, color=color_for_trajectory)
    else:
        print(SUBPLOT_OPTION)
        assert False

    # Add legend to the last axis
    legend_add_manual(ax, map_lev_to_color.keys(), map_lev_to_color.values(), 0.2)

    return fig, axes


def dimredgood_nonlinear_embed_data(X, METHOD="umap", n_components=2, tsne_perp="auto", umap_n_neighbors="auto"):
    """
    Good wrapper, holding all methods for dimensionality reduction of X, esp nonlinear methods like tsne and
    umap, with focus not on leanring parametric space, but instead on returning embedding
    PARAMS:
    - X, already-preprocessed data, (nsamps, ndims)
    RETURNS:
    - Xredu, (nsamp, n_components)
    """
    nsamp = X.shape[0]
    if METHOD == "tsne":
        from sklearn.manifold import TSNE
        if tsne_perp =="auto":
            perp = int(max([10, min([50, 0.1*nsamp])])) # heuristic
        else:
            perp = tsne_perp
        print("TSNE, Using this perp:", perp, ", nsamp =", nsamp)
        Xredu = TSNE(n_components=n_components, perplexity=perp, learning_rate="auto", init="pca").fit_transform(X)
        reducer = None
    elif METHOD == "umap":
        import umap
        if umap_n_neighbors =="auto":
            umap_n_neighbors = int(max([10, min([30, 0.05*nsamp])])) # heuristic
        print("UMAP, Using this n_neighbors:", umap_n_neighbors, ", nsamp =", nsamp, ", n_components: ", n_components)
        min_dist = 0.1
        reducer = umap.UMAP(n_components=n_components, n_neighbors=umap_n_neighbors, min_dist=min_dist)
        Xredu = reducer.fit_transform(X)
    elif METHOD == "mds":
        from sklearn.manifold import MDS
        print("MDS, embedding... n_components:", n_components)
        reducer = MDS(n_components=n_components, normalized_stress='auto')
        Xredu = reducer.fit_transform(X)
    else:
        assert False

    return Xredu, reducer

def dimredgood_subspace_variance_accounted_for(X, subspace_1, subspace_2):
    """
    Compute variance accounted for (VAF), i.e., how much variance of activity in subspace 1 still remains
    when you then projct to susbpace 2 (and vice versa).
    
    See Xie, Liping Wang, Science 2022
    
    PARAMS:
    - X, data, usually (nchans, nconditions or ntrials). This is the original data from which the original variance
    is computed.
    - subspace_1/2, arrays (nchans, ndims), the two subspaces to compare. Will autmatically make sure
    their columns are unit length.
    """

    # First, project data to subspaces
    data_1, _ = dimredgood_project_data_denoise_simple(X, subspace_1, "denoise", normalization="norm")
    data_2, _ = dimredgood_project_data_denoise_simple(X, subspace_2, "denoise", normalization="norm")

    # Second, project those projectsions to the other subspace
    data_1_2, _ = dimredgood_project_data_denoise_simple(data_1, subspace_2, "denoise", normalization="norm")
    data_2_1, _ = dimredgood_project_data_denoise_simple(data_2, subspace_1, "denoise", normalization="norm")

    
    # Compute variances
    def _compute_variance(x):
        return np.sum((x - np.mean(x, axis=0))**2)
    
    variance_total = _compute_variance(X)
    variance_1 = _compute_variance(data_1)
    variance_2 = _compute_variance(data_2)
    variance_1_2 = _compute_variance(data_1_2)
    variance_2_1 = _compute_variance(data_2_1)

    if False:
        print(X.shape)
        print(data_1.shape)
        print(data_1.shape)
        print(data_1_2.shape)
        print(data_2_1.shape)
        heatmap_mat(X, annotate_heatmap=False, diverge=True);
        heatmap_mat(data_1, annotate_heatmap=False, diverge=True);
        heatmap_mat(data_2, annotate_heatmap=False, diverge=True);
        heatmap_mat(data_1_2, annotate_heatmap=False, diverge=True);
        heatmap_mat(data_2_1, annotate_heatmap=False, diverge=True);
        print(variance_total, variance_1, variance_2, variance_1_2, variance_2_1)
        print(vaf_1_2, vaf_2_1)

    # Compute VAF
    vaf_1 = variance_1/variance_total
    vaf_2 = variance_2/variance_total

    vaf_1_2 = variance_1_2/variance_1 
    vaf_2_1 = variance_2_1/variance_2

    out = {
        "variance_total":variance_total,
        "variance_1":variance_1,
        "variance_2":variance_2,
        "variance_1_2":variance_1_2,
        "variance_2_1":variance_2_1,
        "vaf_1":vaf_1,
        "vaf_2":vaf_2,
        "vaf_1_2":vaf_1_2,
        "vaf_2_1":vaf_2_1
        }
    
    return out

def dimredgood_project_data_denoise_simple(X, basis_vectors, version="projection", normalization=None,
                                           plot_orthonormalization=False):
    """
    Project X to the subspace spanned by the basis vectors, and optionally denoise by projecting back to the original space
    PARAMS:
    - X, data, (nchans, nfeatures) or (nchans, ntrials)
    - basis_vectors, (nchans, ndims_project), where nchans matches self.CHans
    - version, whethr to project or to denoise (i.e, project the reproject out)
    - do_orthonormal, bool, if True, then orthonormlaizes the basis using QR decomspotion. The order of columns
    in basis matters. ie sequentially gets orthognalizes each column by the subspace spanned by the preceding columns.
    """

    if not basis_vectors.shape[0] == X.shape[0]:
        print(basis_vectors.shape)
        print(X.shape)
        assert False

    if normalization=="orthonormal":
        # Optionally, orthonormalize the vectors
        basis_vectors_ortho, r = np.linalg.qr(basis_vectors)
        if plot_orthonormalization:
            from pythonlib.tools.snstools import heatmap_mat
            heatmap_mat(basis_vectors, annotate_heatmap=False, diverge=True)
            heatmap_mat(basis_vectors_ortho, annotate_heatmap=False, diverge=True)
            heatmap_mat(r, annotate_heatmap=False, diverge=True)
            # print(basis_vectors.shape, X.shape)
        basis_vectors = basis_vectors_ortho
    elif normalization=="norm":
        # Each vector length to 1
        basis_vectors = basis_vectors/np.sum(basis_vectors**2, axis=0)**0.5
        # print(np.sum(basis_vectors**2, axis=0))
        # assert False
    else:
        assert normalization is None

    # Get data
    if len(X.shape)>2:
        assert X.shape[2]==1, "need to be scalar (i.e., not time-series)"
        X = X.squeeze() # (nchans, ntrials)

    # project data onto basis vectors
    if version=="projection":
        Xnew = basis_vectors.T @ X
        if False:
            # NOTE: this does exactly the same thing -- I confirmed
            from neuralmonkey.analyses.state_space_good import dimredgood_pca_project
            Xredu, stats, Xredu_in_orig_shape = dimredgood_pca_project(basis_vectors.T, X.T, plot_pca_explained_var_path="/tmp/pca.pdf")
        # print(stats)
        # assert False
    elif version=="denoise":
        # Project to subspace, and then expand back out -- this is to "denoise" the activity
        D = basis_vectors @ basis_vectors.T # (nchans, nchans)
        Xnew = D @ X # (nchans, nfeatures)
        # print(D.shape)
        # print(X.shape)
        # print(Xnew.shape)
        # assert False
    else:
        print(version)
        assert False

    return Xnew, basis_vectors

def dimredgood_pca_project(components, X, plot_pca_explained_var_path=None,
                           do_additional_reshape_from_ChTrTi=False,
                        #    reshape_method_that_was_used="trials_x_chanstimes"
                           ):
    """
    Project new data onto a subspace, e.g.,, PCA subspace, and compute variance explained.
    
    PARAMS
    :param components: pca loadings, (n_components, n_features) [i.e., (ndims, nchans)]
    :param X: data to project, already demeaned, etc. and already reshaped to (ntrials, nfeats), .. [i.e., (ntrials, nchans)]
    except if do_additional_reshape_from_ChTrTi==True, in which case input is (nchans, ntrials, ntimesS),
    and here will do the approppriate reshaping.
    
    [IGNORE THIS: based on value of reshape_method_that_was_used. So X is eitehr:
    --- (ntrials, nfeatures), if reshape_method_that_was_used="trials_x_chanstimes" or
    --- (nchans, ntrials, ntimes), if reshape_method_that_was_used="chans_x_trials_x_times" or
    :param reshape_method_that_was_used: str, method that was used to get components (e..g, in PA.dataextract_pca_demixed_subspace),
    presumably from data shaped same as X. This here is used to deicde whether to reshape X to match that process.]

    :return:
    - Xredu, (ntrials, n_components)
    """

    # Decide if do reshape first
    if do_additional_reshape_from_ChTrTi:
        assert len(X.shape)==3
        Xorig = X.copy()
        nchans, ntrials, ntimes = Xorig.shape
        X = np.reshape(Xorig, [nchans, ntrials * ntimes]).T # (ntrials*ntimes, nchans)        

    # Sanity checks
    if not X.shape[1] == components.shape[1]:
        print(X.shape)
        print(components.shape)
        print(do_additional_reshape_from_ChTrTi)
        print("Is this because you are doing 'scalar', and trying to use different windwos for pca space identificaiton vs. projecting? Then this must fail, since scalar doesnt slide over time... Solve by setting time windows the same.")
    
    # print("HERE", np.mean(X, axis=0))
    if False: # Need this False in order to project data that hasnt had each time bin subtracted...
        from pythonlib.tools.nptools import isnear
        assert np.all(np.mean(X, axis=0)<0.1)
    # assert isnear(np.mean(X, axis=0), np.zeros(X.shape)) # Check that X is zeroed, or else the variance calcualtion may be weird, and the result can be weird.

    # Compute projection
    Xredu = np.dot(X, components.T)

    # Compute variance explained
    # - total
    total_variance_ = np.sum((X - np.mean(X, axis=0))**2)
    # - fraction
    ncomp = components.shape[0]
    variances_ = [np.sum(np.dot(X, components[n,:].T)**2) for n in range(ncomp)]
    explained_variance_ratio_ = [v/total_variance_ for v in variances_]

    if do_additional_reshape_from_ChTrTi:
        npcs_keep = Xredu.shape[1]
        Xredu_in_orig_shape = Xredu.T # (npcs_keep, ntrials*ntimes)
        Xredu_in_orig_shape = np.reshape(Xredu_in_orig_shape, [npcs_keep, ntrials, ntimes]) # (npcs_keep, ntrials*ntimes)
    else:
        Xredu_in_orig_shape = None

    if plot_pca_explained_var_path is not None:
        fig, axes = plt.subplots(1,2, figsize=(8,3))

        ax = axes.flatten()[0]
        ax.plot(explained_variance_ratio_, "-ob")
        ax.set_title("frac var, each dim")
        ax.legend()

        ax = axes.flatten()[1]
        ax.plot(np.cumsum(explained_variance_ratio_), "-ob")
        ax.set_title("frac var, each dim")
        ax.set_title("cumulative var, each dim")
        ax.legend()

        savefig(fig, plot_pca_explained_var_path)

    stats = {
        "total_variance_":total_variance_,
        "variances_":variances_,
        "explained_variance_ratio_":explained_variance_ratio_
    }
    return Xredu, stats, Xredu_in_orig_shape

def dimredgood_pca(X, n_components=None,
                   how_decide_npcs_keep = "cumvar",
                   pca_frac_var_keep=0.85, pca_frac_min_keep=0.01,
                   plot_pca_explained_var_path=None, plot_loadings_path=None,
                   plot_loadings_feature_labels=None,
                   method="svd", npcs_keep_force=None, return_stats=False):
    """
    Holds All things related to applying PCA, and plots.
    :param X: data (ndat, nfeats)
    :param n_components:
    :param pca_frac_var_keep:
    :param plot_pca_explained_var_path:
    :param plot_loadings_path:
    :param plot_loadings_feature_labels:
    :param method:
        using "svd" so that it does not recenter.
    :param npcs_keep_force: int (take this many of the top pcs) or None (use auto method, one of the abbove
    :return: Xpcakeep (ndat, nfeats)
     - components, (n_components, n_features)
    """

    assert len(X.shape)==2

    if method=="sklearn":
        # Recenters (but not rescales)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        Xpca = pca.fit_transform(X) # (ntrials, nchans) --> (ntrials, ndims)
        explained_variance_ratio_ = pca.explained_variance_ratio_
        components_ = pca.components_
    elif method=="svd":
        # No recenter or rescale. Otehrwise owrks identically to above (tested).
        # Copied from sklearn.decomposion._pca._fit()
        if False:
            # If do this, thens hould be identical to sklearn (tested).
            X = X.copy()
            X -= np.mean(X, axis=0)

        from sklearn.utils.extmath import svd_flip
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        U, Vt = svd_flip(U, Vt) # flip sign of eigenvectors ,to force deterministic output
        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (X.shape[0] - 1)
        explained_variance_ratio_ = explained_variance_ / explained_variance_.sum()
        # singular_values_ = S.copy()  # Store the singular values.

        Xpca = np.dot(X, components_.T) # get projection

        # Store pca weights
        pca = {
            "components":components_,
            "explained_variance_ratio_":explained_variance_ratio_,
        }
    else:
        assert False

    # DEcide how many dimensions to keep
    if npcs_keep_force is None:
        if how_decide_npcs_keep=="cumvar":
            # 1. cumvar
            cumvar = np.cumsum(explained_variance_ratio_)
            npcs_keep = np.argwhere(cumvar >= pca_frac_var_keep)[0].item()+1 # the num PCs to take such that cumsum is just above thresh
        elif how_decide_npcs_keep=="minvar":
            # 2. cutoff dims that expalin less than this frac variance
            if np.all(explained_variance_ratio_ >= pca_frac_min_keep):
                npcs_keep = len(explained_variance_ratio_)
            else:
                npcs_keep = np.argwhere(explained_variance_ratio_ < pca_frac_min_keep)[0].item()
        elif how_decide_npcs_keep=="keep_all":
            npcs_keep = len(explained_variance_ratio_)
        else:
            print(how_decide_npcs_keep)
            assert False
    else:
        # Use the forced N.
        npcs_keep = min([len(explained_variance_ratio_), npcs_keep_force])

    Xpcakeep = Xpca[:, :npcs_keep]
    assert Xpcakeep.shape[1] == npcs_keep

    ### Plot?
    if plot_pca_explained_var_path is not None:
        fig, axes = plt.subplots(1,2, figsize=(8,3))

        ax = axes.flatten()[0]
        ax.plot(explained_variance_ratio_, "-ob")
        if how_decide_npcs_keep=="minvar":
            ax.axhline(pca_frac_min_keep, color="g", label="pca_frac_min_keep")
        ax.axvline(npcs_keep-1, color="r", label="last pc that is kept")
        ax.set_title("frac var, each dim")
        ax.legend()

        ax = axes.flatten()[1]
        ax.plot(np.cumsum(explained_variance_ratio_), "-ob")
        if how_decide_npcs_keep=="cumvar":
            ax.axhline(pca_frac_var_keep, color="g", label="pca_frac_var_keep")
        ax.axvline(npcs_keep-1, color="r", label="last pc that is kept")
        ax.set_title("frac var, each dim")
        ax.set_title("cumulative var, each dim")
        ax.legend()
        ax.set_xlabel(f"npcs keep = {npcs_keep}")

        savefig(fig, plot_pca_explained_var_path)

    if plot_loadings_path is not None:
        from pythonlib.tools.snstools import heatmap_mat
        fig, ax = plt.subplots(figsize=(20, 15))
        if plot_loadings_feature_labels is not None:
            assert len(plot_loadings_feature_labels)==components_.shape[1], "you inputed incorrect lables"
        heatmap_mat(components_, ax, diverge=True, annotate_heatmap=False, labels_col=plot_loadings_feature_labels) # (n_components, n_features)
        ax.set_ylabel("pcs")
        ax.set_xlabel("features (chans x twind)")
        savefig(fig, plot_loadings_path)

    if return_stats:
        return Xpcakeep, Xpca, pca, explained_variance_ratio_, components_
    else:
        return Xpcakeep, Xpca, pca


def cleanup_remove_labels_ignore(xs, ys, labels_color, labels_subplot):
    """
    Remove trials that have labels those in LABELS_IGNORE, either in
    labels_color or labels_subplot.
    PARAMS:
    - xs, (n,) array
    - ys, (n,) array
    - labels_color, list, len trials.
    - labels_subplot, list, len trials.
    RETURNS:
    - xs, ys, labels_color, labels_subplot, pruned copies.
    """

    inds_keep_1 = [i for i, val in enumerate(labels_color) if val not in LABELS_IGNORE]

    if labels_subplot is not None:
        inds_keep_2 = [i for i, val in enumerate(labels_subplot) if val not in LABELS_IGNORE]
    else:
        inds_keep_2 = []

    inds_keep = sorted(set(inds_keep_1 + inds_keep_2))
    xs = xs[inds_keep]
    ys = ys[inds_keep]
    labels_color = [labels_color[i] for i in inds_keep]
    if labels_subplot is not None:
        labels_subplot = [labels_subplot[i] for i in inds_keep]

    return xs, ys, labels_color, labels_subplot
    

# def euclidian_distance_compute
def euclidian_distance_compute_scalar(PA, LIST_VAR, LIST_VARS_OTHERS, PLOT, PLOT_MASKS, twind, tbin_dur,
                               tbin_slice, savedir,
                               PLOT_STATE_SPACE=True, COMPUTE_EUCLIDIAN=True,
                               nmin_trials_per_lev=None,
                               version_distance="euclidian_unbiased", LIST_CONTEXT=None, LIST_FILTDICT=None,
                               LIST_PRUNE_MIN_N_LEVS=None,
                               dim_red_method = "pca_umap",
                               NPCS_KEEP=None, umap_n_neighbors=30, extra_dimred_method_n_components=2,
                               superv_dpca_params = None, state_space_plot_3D=False,
                               return_PAredu=False):
    """
    Wrapper to compute all distances between levels of variables, with flecxible abilties for
    controlling context. -- see within for details ...

    :param PA:
    :param LIST_VAR:
    :param LIST_VARS_OTHERS:
    :param PLOT:
    :param PLOT_MASKS:
    :param twind:
    :param tbin_dur:
    :param tbin_slice:
    :param savedir:
    :param SHUFFLE:
    :param PLOT_STATE_SPACE:
    :param nmin_trials_per_lev:
    :param LIST_CONTEXT: list of dicts each dict has "same" and :"diff" keys, list of str vars, defining the context for the
    matchging var and vars_others.
    :return:
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    from pythonlib.cluster.clustclass import Clusters

    ######################## (1) CONSTRUCT A SINGLE SUBSPACE (that all subsequence plots and analyses will be performed on)
    # Get specific params, based on how want to do dim reduction
    if superv_dpca_params is None:
        dpca_var = None
        dpca_vars_group = None
        dpca_filtdict=None
        dpca_proj_twind = None
    else:
        dpca_var = superv_dpca_params["superv_dpca_var"]
        dpca_vars_group = superv_dpca_params["superv_dpca_vars_group"]
        dpca_filtdict = superv_dpca_params["superv_dpca_filtdict"]
        dpca_proj_twind = superv_dpca_params["dpca_proj_twind"]

    Xredu, PAredu = PA.dataextract_dimred_wrapper("scal", dim_red_method, savedir, 
                                   twind, tbin_dur=tbin_dur, tbin_slide=tbin_slice, 
                                   NPCS_KEEP = NPCS_KEEP,
                                   dpca_var = dpca_var, dpca_vars_group = dpca_vars_group, dpca_filtdict=dpca_filtdict, dpca_proj_twind = dpca_proj_twind, 
                                   raw_subtract_mean_each_timepoint=False,
                                   umap_n_components=extra_dimred_method_n_components, umap_n_neighbors=umap_n_neighbors)

    ############################ (2) Euclidian and State space plots
    if LIST_CONTEXT is not None:
        assert len(LIST_CONTEXT)==len(LIST_VAR)
    else:
        LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]

    if LIST_FILTDICT is not None:
        assert len(LIST_FILTDICT)==len(LIST_VAR)
    else:
        LIST_FILTDICT = [None for _ in range(len(LIST_VAR))]

    if LIST_PRUNE_MIN_N_LEVS is not None:
        assert len(LIST_PRUNE_MIN_N_LEVS)==len(LIST_VAR)
    else:
        LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]

    if version_distance=="euclidian_unbiased":
        DO_SHUFFLE = False
        AGG_BEFORE_DIST = False
    elif version_distance=="euclidian":
        DO_SHUFFLE = True
        AGG_BEFORE_DIST=True
        N_SHUFF = 5
    else:
        assert False

    if COMPUTE_EUCLIDIAN:
        if False: # Now this is computed below internally
            ########### Compute global distances BEFORE pruning data
            # To normalize, compute distance across datapts
            labels_rows = None
            Cl = Clusters(Xredu[:, :n_pcs_keep_euclidian], labels_rows)
            Cldistall = Cl.distsimmat_convert("euclidian")
            ma = Cldistall._rsa_matindex_generate_upper_triangular()
            dist_all = Cldistall.Xinput[ma].flatten()
            if PLOT:
                # Plot distribution
                fig, ax = plt.subplots()
                ax.hist(dist_all, bins=20)
            # get 95th percentile of distance
            DIST_NULL_50 = np.percentile(dist_all, 50)
            DIST_NULL_95 = np.percentile(dist_all, 95)
            DIST_NULL_98 = np.percentile(dist_all, 98)
            print("DIST_NULL_50", DIST_NULL_50)
            print("DIST_NULL_95", DIST_NULL_95)
            print("DIST_NULL_98", DIST_NULL_98)

    ############ SCore, for each variable
    RES = []
    vars_already_state_space_plotted = []
    var_varothers_already_plotted = []
    heatmaps_already_plotted = []

    for i_var, (var, var_others, context, filtdict, prune_min_n_levs) in enumerate(zip(LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS)):
        print("RUNNING: ", i_var,  var, " -- ", var_others)
        # Copy pa for this
        pa = PAredu.copy()
        # pa_orig_dim = PAredu_orig_dim.copy()

        var_for_name = var
        if isinstance(var, (tuple, list)):
            pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], var, "_tmp")
            # pa_orig_dim.Xlabels["trials"] = append_col_with_grp_index(pa_orig_dim.Xlabels["trials"], var, "_tmp")
            var = "_tmp"

        if filtdict is not None:
            for _var, _levs in filtdict.items():
                print("len pa bnefore filt this values (var, levs): ", _var, _levs)
                pa = pa.slice_by_labels("trials", _var, _levs, verbose=True)

        ############### PRUNE DATA, TO GET ENOUGH FOR THIS VARIABLE
        # # Prep by keeping only if enough data
        # from neuralmonkey.analyses.rsa import preprocess_rsa_prepare_popanal_wrapper
        # preprocess_rsa_prepare_popanal_wrapper(pa, )

        # Get data split by othervar
        # Return dict[levo] --> data
        # Prune data to just cases with at least 2 levels of decode var



        # dflab = pa.Xlabels["trials"]
        # dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var, var_others,
        #                                                          n_min_across_all_levs_var=prune_min_n_trials,
        #                                                          lenient_allow_data_if_has_n_levels=prune_min_n_levs,
        #                                                          prune_levels_with_low_n=True,
        #                                                          ignore_values_called_ignore=True,
        #                                                          plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
        #                                                          balance_no_missed_conjunctions=False)
        # # for levo, dfthis in dict_dfthis.items():
        # #     print(levo, len(dfthis))
        # if len(dfout)==0:
        #     print("all data pruned!!")
        #     continue
        #
        # # Only keep the indices in dfout
        # print("  Pruning for this var adn conjunction. Original length:", pa.X.shape[1], ", pruned length:", len(dfout))
        # pa = pa.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)
        if nmin_trials_per_lev is not None:
            prune_min_n_trials = nmin_trials_per_lev
        else:
            prune_min_n_trials = N_MIN_TRIALS

        if (var, tuple(var_others)) not in heatmaps_already_plotted:
            plot_counts_heatmap_savepath = f"{savedir}/{i_var}_counts_heatmap-var={var_for_name}-ovar={'|'.join(var_others)}.pdf"
            heatmaps_already_plotted.append((var, tuple(var_others)))
        else:
            plot_counts_heatmap_savepath = None

        pa_before_prune = pa.copy()
        pa, _, _= pa.slice_extract_with_levels_of_conjunction_vars(var, var_others, prune_min_n_trials, prune_min_n_levs,
                                                         plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
        if pa is None:
            print("all data pruned!!")
            continue

        ######################## COMPUTE DISTANCES between levels of var
        if PLOT:
            fig, ax = plt.subplots()
            chan = pa.Chans[0]
            pa.plotwrapper_smoothed_fr_split_by_label("trials", var, ax, chan=chan)
            plt.close("all")

        if COMPUTE_EUCLIDIAN:
            from pythonlib.tools.listtools import stringify_list

            if context is not None:
                plot_mask_path = f"{savedir}/{i_var}_MASK-var={var_for_name}-ovar={'|'.join(var_others)}-context={stringify_list(context['diff'])}.pdf" # just diff, or else too lomg. diff is main info anyway
            else:
                plot_mask_path = f"{savedir}/{i_var}_MASK-var={var_for_name}-ovar={'|'.join(var_others)}-context={context}.pdf" # just diff, or else too lomg. diff is main info anyway

            # Keep just the n dims you want
            if False: # Already done, in dim reductions
                pa_eucl = pa.slice_by_dim_indices_wrapper("chans", list(range(n_pcs_keep_euclidian)))
                print("FINAL DIMENSION OF DATA (after dimredu, before eucl):", pa_eucl.X.shape)
            else:
                pa_eucl = pa
                
            if False:
                # This passes
                assert np.all(pa.X[:n_pcs_keep_euclidian, :, :] == pa_eucl.X)
            res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98 = euclidian_distance_compute_scalar_single(pa_eucl, var, var_others, PLOT, PLOT_MASKS,
                                                          version_distance=version_distance,
                                                          AGG_BEFORE_DIST=AGG_BEFORE_DIST, context_input=context,
                                                          plot_mask_path=plot_mask_path)
            print("DIST_NULL_50", DIST_NULL_50)
            for r in res:
                r["shuffled"] = False
                r["shuffled_iter"] = -1
                r["index_var"] = i_var

            # Collect
            RES.extend(res)

            ############### SHUFFLE CONTROLS
            if DO_SHUFFLE:
                assert False, "this is old code."
                from pythonlib.tools.pandastools import shuffle_dataset_hierarchical
                for i_shuff in range(N_SHUFF):
                    print("RUNNING SHUFFLE, iter:", i_shuff)

                    # 0. Create shuffled dataset
                    PApcaSHUFF = pa.copy()
                    dflab = PApcaSHUFF.Xlabels["trials"]
                    dflabSHUFF = shuffle_dataset_hierarchical(dflab, [var], var_others)
                    PApcaSHUFF.Xlabels["trials"] = dflabSHUFF

                    res, _, _, _ = euclidian_distance_compute_scalar_single(PApcaSHUFF, var, var_others,
                                                                  version_distance=version_distance,
                                                                  AGG_BEFORE_DIST=AGG_BEFORE_DIST)
                    for r in res:
                        r["shuffled"] = True
                        r["shuffled_iter"] = i_shuff
                        r["index_var"] = i_var

                    # Collect
                    RES.extend(res)

        ######## STATE SPACE PLOTS
        if PLOT_STATE_SPACE:
            from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER
            # if pa.X.shape[0]==3:
            #     list_dims = [(0,1), (1,2)]
            # elif pa.X.shape[0]>3:
            #     list_dims = [(0,1), (2,3)]
            # else:
            #     list_dims = [(0,1)]
            if len(LIST_VAR)<15:
                if pa.X.shape[0]==3:
                    list_dims = [(0,1), (1,2)]
                elif pa.X.shape[0]>3:
                    list_dims = [(0,1), (2,3)]
                else:
                    list_dims = [(0,1)]
            else:
                # Too slow, just do 1st 2 d
                list_dims = [(0,1)]

            if (var, var_others) not in var_varothers_already_plotted:
                dflab = pa.Xlabels["trials"]
                Xthis = pa.X.squeeze(axis=2).T # (n4trials, ndims)
                trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, savedir,
                                                             vars_subplot=var_others, list_dims=list_dims,
                                                             skip_subplots_lack_mult_colors=False, save_suffix = i_var)
                var_varothers_already_plotted.append((var, tuple(var_others)))

                if state_space_plot_3D and pa.X.shape[0]>2:
                    plot_3D = True
                    alpha=0.3
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, savedir, var_others, plot_3D=plot_3D,
                                                                 skip_subplots_lack_mult_colors=False, alpha=alpha, save_suffix=f"{i_var}-3D")


            # Also plot this variable in the entire dataset.
            # - and use data that hasnt been pruend for conjunctions.
            # if (var, ("epoch",)) not in var_varothers_already_plotted:
            if var not in vars_already_state_space_plotted:
                dflab = pa_before_prune.Xlabels["trials"]
                Xthis = pa_before_prune.X.squeeze(axis=2).T # (n4trials, ndims)
                trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, savedir,
                                                             vars_subplot=["epoch"], list_dims=list_dims,
                                                             skip_subplots_lack_mult_colors=False, save_suffix = i_var)
                vars_already_state_space_plotted.append(var)

                if state_space_plot_3D and pa.X.shape[0]>2:
                    plot_3D = True
                    alpha=0.3
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, savedir, ["epoch"], plot_3D=plot_3D,
                                                                 skip_subplots_lack_mult_colors=False, alpha=alpha,
                                                                 save_suffix=f"{i_var}-3D")

            plt.close("all")

    dfres = pd.DataFrame(RES)
    if COMPUTE_EUCLIDIAN and len(dfres)>0:
        # Get score normalized against global distance
        dfres["DIST_NULL_50"] = DIST_NULL_50
        dfres["DIST_NULL_95"] = DIST_NULL_95
        dfres["DIST_NULL_98"] = DIST_NULL_98
        dfres["twind_analy"] = [twind for _ in range(len(dfres))]

    if return_PAredu:
        return dfres, PAredu
    else:
        return dfres

def euclidian_distance_compute_scalar_single(pa, var, var_others, PLOT_RSA_HEATMAP=False, PLOT_MASKS=False,
                                            version_distance="euclidian_unbiased", AGG_BEFORE_DIST=False,
                                            context_input=None, path_for_save_print_lab_each_mask=None,
                                            plot_mask_path=None):
    """
    Flexible method to compute all distances between levels of var, within each lev of var_others, and return
    results in list of dicts.
    :param pa:
    :param var:
    :param var_others:
    :param PLOT_RSA_HEATMAP:
    :param PLOT_MASKS:
    :param version_distance:
    :param AGG_BEFORE_DIST: if True, then first Agg, then compute distances between means, else Then
    compute distance between datapts, and then agg distances. Requires compatiibltiy with inputed version_distance.
    :return:
    """
    from pythonlib.cluster.clustclass import Clusters

    assert AGG_BEFORE_DIST==False, "obsolete"
    # This is not used... delete to not confused
    del version_distance


    if False: # not used...
        def _get_context_masks(Cldist, context_input, skip_mask_plot=False):
            """
            How to defince DIFFERENT context (SAME is always just the conj vars).
            :param CLdist:
            :param context_input:
            :return:
            """
            if context_input is not None and len(context_input)>0:
                print("Generating masks using context:", context_input)
                if "diff_context_ver" in context_input:
                    diff_context_ver = context_input["diff_context_ver"]
                else:
                    diff_context_ver = "diff_specific_lenient"
                # Then use inputed context
                MASKS, fig, axes = Cldist.rsa_mask_context_helper(var, var_others, diff_context_ver,
                                    context_input["same"], context_input["diff"], PLOT=PLOT_MASKS,
                                    path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
            else:
                # Called "diff" if ANY var in var_others is different.
                MASKS, fig, axes = Cldist.rsa_mask_context_helper(var, var_others, "diff_at_least_one", PLOT=PLOT_MASKS,
                                                                path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
            if plot_mask_path is not None and PLOT_MASKS and not skip_mask_plot:
                print("Saving context mask at: ", plot_mask_path)
                savefig(fig, plot_mask_path)

            return MASKS

    # CONTEXT_DIFFERENT_FOR_FIRST_STROKE = True
    # if "seqc_0_shape" in pa.Xlabels["trials"].columns:
    #     # Then this is trial-level. ignore this stroke-index based constarint
    #     CONTEXT_DIFFERENT_FOR_FIRST_STROKE = False
    # else:
    #     assert "stroke_index_is_first" in pa.Xlabels["trials"].columns

    ALSO_COLLECT_SAME_EFFECT = False # NOTE: This is incorrect - it is just "null" data.
    if ALSO_COLLECT_SAME_EFFECT:
        assert AGG_BEFORE_DIST==False, "cannot get same effect if agg first"

    assert pa.X.shape[2]==1, "must be scalar"

    assert AGG_BEFORE_DIST == False, "assuming this..."
    # if AGG_BEFORE_DIST:
    #     # Agg, then compute distances between means
    #     # 1. agg before computing distances (quicker)
    #     pa = pa.slice_and_agg_wrapper("trials", [var]+var_others)
    #     assert version_distance in ["euclidian"]
    # else:
    #     # Then compute distance between datapts, and then agg distances
    #     pa = pa
    #     assert version_distance in ["euclidian_unbiased"]

    # Create clusters
    # dflab = pa.Xlabels["trials"]
    # Xthis = pa.X.squeeze(axis=2).T # (ntrials, ndims)
    # print("  Final Scalar data (trial, dims):", Xthis.shape)

    # label_vars = [var]+var_others
    # labels_rows = dflab.loc[:, label_vars].values.tolist()
    # labels_rows = [tuple(x) for x in labels_rows] # list of tuples
    # params = {"label_vars":label_vars}
    # Cl = Clusters(Xthis, labels_rows, ver="rsa", params=params)

    ###############
    res = []

    ############################
    dat_level = "distr"
    version_distance = "euclidian_unbiased"
    # # convert to distance matrix
    # if AGG_BEFORE_DIST:
    #     Cldist = Cl.distsimmat_convert(version_distance)
    # else:
    #     Cldist = Cl.distsimmat_convert_distr(label_vars, version_distance, accurately_estimate_diagonal=ALSO_COLLECT_SAME_EFFECT)
    
    print("... dataextract_as_distance_matrix_clusters_flex")
    LIST_CLDIST, LIST_TIME = pa.dataextract_as_distance_matrix_clusters_flex([var] + var_others, 
                                                                            version_distance=version_distance)
    assert len(LIST_CLDIST)==1
    Cldist = LIST_CLDIST[0]

    if PLOT_RSA_HEATMAP:
        Cldist.rsa_plot_heatmap()

    if True:
        print("... rsa_distmat_score_same_diff_by_context")
        _res, _, _, _ = Cldist.rsa_distmat_score_same_diff_by_context(var, var_others, context_input, dat_level, PLOT_MASKS, 
                                                    plot_mask_path=plot_mask_path, path_for_save_print_lab_each_mask=path_for_save_print_lab_each_mask)
        res.extend(_res)
    else:

        ### Get masks of context
        MASKS = _get_context_masks(Cldist, context_input)

        ##################### COMPUTE SCORES.
        res = []
        # 1. Within each context, average pairwise distance between levels of effect var
        map_grp_to_mask_context_same = Cldist.rsa_mask_context_split_levels_of_conj_var(var_others, PLOT=PLOT_MASKS, exclude_diagonal=False,
                                                                                        contrast="same")
        map_grp_to_mask_vareffect = Cldist.rsa_mask_context_split_levels_of_conj_var([var], PLOT=PLOT_MASKS, exclude_diagonal=False,
                                                                                    contrast="any") # either row or col must be the given level.
        ma_ut = Cldist._rsa_matindex_generate_upper_triangular()

        # Difference between levels of var, computed within(separately) for each level of ovar
        # (NOTE: this does nto care about "context")
        for grp, ma_context_same in map_grp_to_mask_context_same.items():
            ma_final = ma_context_same & MASKS["effect_diff"] & ma_ut # same context, diff effect
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist = Cldist.Xinput[ma_final].mean()
                # sanity check
                _ma_final = ma_context_same & MASKS["effect_diff"] & ma_ut & MASKS["context_same"]
                _dist = Cldist.Xinput[_ma_final].mean()
                if not np.isclose(dist, _dist):
                    print(dist, _dist)
                    print("I thoguth that context_same is exactly identical to the vars_others... figure this out")
                    print("It is probably becuase one is nan, becuase not enougb datapts? if so, make sure all conjucntions have enough n above")
                    assert False
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"diff",
                    "context_samediff":"same",
                    "levo":grp,
                    "leveff":"ALL",
                    "dist":dist,
                    "dat_level":dat_level
                })

        #### ACROSS CONTEXTS (compute separately for each level of effect)
        for lev_effect, ma in map_grp_to_mask_vareffect.items():
            # Also collect (same effect, diff context)
            # For each level of var, get its distance to that same level of var across
            # all contexts.
            # - same effect diff context
            ma_final = ma & MASKS["effect_same"] & MASKS["context_diff"] & ma_ut
            if np.sum(ma_final)>0:
                dist = Cldist.Xinput[ma_final].mean()
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"same",
                    "context_samediff":"diff",
                    "levo":"ALL",
                    "leveff":lev_effect,
                    "dist":dist,
                    "dat_level":dat_level
                })

            # Distance for (diff effect, diff context)
            ma_final = ma & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
            if np.sum(ma_final)>0:
                dist = Cldist.Xinput[ma_final].mean()
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"diff",
                    "context_samediff":"diff",
                    "levo":"ALL",
                    "leveff":lev_effect,
                    "dist":dist,
                    "dat_level":dat_level
                })
                print("diff_diff", lev_effect)

        # Just to make sure no bleed thru to next section.
        del Cldist, MASKS, dat_level, map_grp_to_mask_context_same, map_grp_to_mask_vareffect

    #######################
    ### STUFF THAT USES Pairwise data (Get pairwise distnaces)
    version_distance = "euclidian"
    dat_level = "pts"
    _plot_masks = False # this too large (datapts...)
    if False: # Cant remmeber why did this, and it is not useful.
        assert False, "check, it is weird. Waht is the point of this? "

    LIST_CLDIST, LIST_TIME = pa.dataextract_as_distance_matrix_clusters_flex([var] + var_others, 
                                                                            version_distance=version_distance)
    assert len(LIST_CLDIST)==1
    Cldist_each_dat = LIST_CLDIST[0]

    # Cldist_each_dat = Cl.distsimmat_convert("euclidian")

    if True:
        _res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98 = Cldist_each_dat.rsa_distmat_score_same_diff_by_context(var, var_others, context_input, 
                                                                                            dat_level, _plot_masks, 
                                                                                            plot_mask_path=None, path_for_save_print_lab_each_mask=None)
        res.extend(_res)
    else:
        ### Get masks of context
        MASKS = _get_context_masks(Cldist_each_dat, context_input, skip_mask_plot=True)

        ##################### COMPUTE SCORES.
        # 1. Within each context, average pairwise distance between levels of effect var
        _plot_masks = False # this too large (datapts...)
        # map_grp_to_mask = Cldist_each_dat.rsa_mask_context_split_levels_of_conj_var(var_others, PLOT=PLOT_MASKS, exclude_diagonal=False)
        map_grp_to_mask_context = Cldist_each_dat.rsa_mask_context_split_levels_of_conj_var(var_others, PLOT=_plot_masks, exclude_diagonal=False,
                                                                                            contrast="any")
        map_grp_to_mask_vareffect = Cldist_each_dat.rsa_mask_context_split_levels_of_conj_var([var], PLOT=_plot_masks, exclude_diagonal=False,
                                                                                     contrast="any")
        ma_ut = Cldist_each_dat._rsa_matindex_generate_upper_triangular()

        # --- ALIGN TO EACH LEVEL OF CONTEXT VAR.
        # Difference between levels of var, computed within(separately) for each level of ovar
        # (NOTE: this does nto care about "context")

        for grp, ma in map_grp_to_mask_context.items():
            if np.sum(ma)==0:
                print(grp)
                assert False, "bug in Cl code."
            dist_diff_same = None
            dist_same_same = None
            dist_diff_diff = None

            ma_final = ma & MASKS["effect_diff"] & MASKS["context_same"] & ma_ut
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist_diff_same = Cldist_each_dat.Xinput[ma_final].mean()
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"diff",
                    "context_samediff":"same",
                    "levo":grp,
                    "leveff":"ALL",
                    "dist":dist_diff_same,
                    "dat_level":dat_level,
                })
            # else:
            #     fig, axes = plt.subplots(2,2)
            #
            #     ax = axes.flatten()[0]
            #     Cldist_each_dat.rsa_matindex_plot_bool_mask(ma, ax)
            #
            #     ax = axes.flatten()[1]
            #     Cldist_each_dat.rsa_matindex_plot_bool_mask(MASKS["effect_diff"], ax)
            #
            #     ax = axes.flatten()[2]
            #     Cldist_each_dat.rsa_matindex_plot_bool_mask(MASKS["context_same"], ax)
            #
            #     fig.savefig("/tmp/masks.png")
            #
            #     print(ma)
            #     print(MASKS["effect_diff"])
            #     print(MASKS["context_same"])
            #     assert False
            #     x = ma & MASKS["effect_diff"]
            #     print(sum(ma & MASKS["effect_diff"]))
            #     print(sum(ma & MASKS["context_same"]))
            #     print(grp)
            #     assert False

            # Also collect "same" effect (and same context, as above)
            ma_final = ma & MASKS["effect_same"] & MASKS["context_same"] & ma_ut
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist_same_same = Cldist_each_dat.Xinput[ma_final].mean()
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"same",
                    "context_samediff":"same",
                    "levo":grp,
                    "leveff":"ALL",
                    "dist":dist_same_same,
                    "dat_level":dat_level,
                })

            ma_final = ma & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
            if np.any(ma_final): # might not have if allow for cases with 1 level of effect var.
                dist_diff_diff = Cldist_each_dat.Xinput[ma_final].mean()
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"diff",
                    "context_samediff":"diff",
                    "levo":grp,
                    "leveff":"ALL",
                    "dist":dist_diff_diff,
                    "dat_level":dat_level,
                })
            # else:
            #     fig, axes = plt.subplots(2,2)
            #
            #     ax = axes.flatten()[0]
            #     Cldist_each_dat.rsa_matindex_plot_bool_mask(ma, ax)
            #
            #     ax = axes.flatten()[1]
            #     Cldist_each_dat.rsa_matindex_plot_bool_mask(MASKS["effect_diff"], ax)
            #
            #     ax = axes.flatten()[2]
            #     Cldist_each_dat.rsa_matindex_plot_bool_mask(MASKS["context_diff"], ax)
            #
            #     fig.savefig("/tmp/masks.pdf")
            #
            #     print(ma)
            #     print(MASKS["effect_diff"])
            #     print(MASKS["context_diff"])
            #     assert False
            #     x = ma & MASKS["effect_diff"]
            #     print(sum(ma & MASKS["effect_diff"]))
            #     print(sum(ma & MASKS["context_diff"]))
            #     print(grp)
            #     assert False

            # Normalized effect
            if (dist_diff_same is not None) and (dist_same_same is not None):
                for d, dl in [
                    [dist_diff_same/dist_same_same, "pts_yue"],
                    [np.log2(dist_diff_same/dist_same_same), "pts_yue_log"],
                    [dist_diff_same-dist_same_same, "pts_yue_diff"],
                    ]:
                    res.append({
                        "var":var,
                        "var_others":tuple(var_others),
                        "effect_samediff":"diff",
                        "context_samediff":"same",
                        "levo":grp,
                        "leveff":"ALL",
                        "dist":d,
                        "dat_level":dl,
                    })

            # (diff, diff) --> A bit arbitrary, could have n pts matching context levels (here) or
            # effect levels (below). Choose here since this is the main analysis.
            if (dist_diff_diff is not None) and (dist_same_same is not None):
                for d, dl in [
                    [dist_diff_diff/dist_same_same, "pts_yue"],
                    [np.log2(dist_diff_diff/dist_same_same), "pts_yue_log"],
                    [dist_diff_diff-dist_same_same, "pts_yue_diff"],
                    ]:
                    res.append({
                        "var":var,
                        "var_others":tuple(var_others),
                        "effect_samediff":"diff",
                        "context_samediff":"diff",
                        "levo":grp,
                        "leveff":"ALL",
                        "dist":d,
                        "dat_level":dl,
                    })

        #### ACROSS CONTEXTS (compute separately for each level of effect)
        for lev_effect, ma in map_grp_to_mask_vareffect.items():
            # Also collect (same effect, diff context)
            # For each level of var, get its distance to that same level of var across
            # all contexts.

            dist_same_same = None
            dist_same_diff = None
            # dist_diff_diff = None

            # - same effect, same context - just for normalizing.
            ma_final = ma & MASKS["effect_same"] & MASKS["context_same"] & ma_ut
            if np.sum(ma_final)>0:
                dist_same_same = Cldist_each_dat.Xinput[ma_final].mean()

            # - same effect diff context
            ma_final = ma & MASKS["effect_same"] & MASKS["context_diff"] & ma_ut
            if np.sum(ma_final)>0:
                dist_same_diff = Cldist_each_dat.Xinput[ma_final].mean()
                res.append({
                    "var":var,
                    "var_others":tuple(var_others),
                    "effect_samediff":"same",
                    "context_samediff":"diff",
                    "levo":"ALL",
                    "leveff":lev_effect,
                    "dist":dist_same_diff,
                    "dat_level":dat_level,
                })


            # Normalized effect
            if (dist_same_same is not None) and (dist_same_diff is not None):
                for d, dl in [
                    [dist_same_diff/dist_same_same, "pts_yue"],
                    [np.log2(dist_same_diff/dist_same_same), "pts_yue_log"],
                    [dist_same_diff-dist_same_same, "pts_yue_diff"],
                    ]:
                    res.append({
                        "var":var,
                        "var_others":tuple(var_others),
                        "effect_samediff":"same",
                        "context_samediff":"diff",
                        "levo":"ALL",
                        "leveff":lev_effect,
                        "dist":d,
                        "dat_level":dl,
                    })

            # Normalized effect
            if False: # Instead get these aligned to context (above).
                # Distance for (diff effect, diff context)
                ma_final = ma & MASKS["effect_diff"] & MASKS["context_diff"] & ma_ut
                if np.sum(ma_final)>0:
                    dist_diff_diff = Cldist_each_dat.Xinput[ma_final].mean()
                    res.append({
                        "var":var,
                        "var_others":tuple(var_others),
                        "effect_samediff":"diff",
                        "context_samediff":"diff",
                        "levo":"ALL",
                        "leveff":lev_effect,
                        "dist":dist_diff_diff,
                        "dat_level":dat_level,
                    })

                if (dist_same_same is not None) and (dist_diff_diff is not None):
                    res.append({
                        "var":var,
                        "var_others":tuple(var_others),
                        "effect_samediff":"diff",
                        "context_samediff":"diff",
                        "levo":"ALL",
                        "leveff":lev_effect,
                        "dist":dist_diff_diff/dist_same_same,
                        "dat_level":"pts_yue",
                    })

    return res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98

def euclidian_distance_plot_timedistmat_score_similarity(DFRES, vars_context,
                                                         DO_PLOTS=True, SAVEDIR_PLOTS=None):
    """Get the pairwise similarity between distmats (each at a specific
    event/time)

    Args:
        DFRES (_type_): EAch row holds a dist mat for a specific event/time,
            Genrated from euclidian_distance_compute_trajectories
        vars_context: grouping var. computes separated within eavh level, and averages.
            (ie each level slices out a specific distmat). Make None to compare entire distmat
            without caring about context.
    RETURNS:
    - dfres_corrs_btw_distmats, each row is corr between distats for a pair of event/time
    """

    # computes separately for each context.
    import scipy.stats as stats
    import numpy as np

    res_distmat_corrs = []
    ma_ut = None
    dict_masks = None
    for _i, row_i in DFRES.iterrows():
        print("row: ", _i)
        for _j, row_j in DFRES.iterrows():
            if _i == _j:
                distmat_corr = 0. # Set diagonal to 0, arbitrarily.
            elif _j>_i:
                Cldist1 = row_i["Cldist"]
                Cldist2 = row_j["Cldist"]
                            
                # Check matching labels
                assert Cldist1.Labels == Cldist2.Labels
                
                if ma_ut is None:
                    ma_ut = Cldist1._rsa_matindex_generate_upper_triangular(exclude_diag=True)

                if vars_context is not None:
                    # Score for each context, then combine
                    # e.g. condition on specific levels of vars_others
                    if dict_masks is None:
                        # NOTE: confirmed this correct, by setting PLOT=True
                        dict_masks = Cldist1.rsa_mask_context_split_levels_of_conj_var(vars_context, PLOT=False)
                        
                    dists = []
                    for _, ma in dict_masks.items():
                        vec1 = Cldist1.Xinput[(ma_ut & ma)].flatten()
                        vec2 = Cldist2.Xinput[(ma_ut & ma)].flatten()
                        
                        if True: # Less noisy
                            tau, _ = stats.kendalltau(vec1, vec2)
                            dists.append(tau)
                        else:
                            dists.append(np.corrcoef(vec1, vec2)[0,1])
                    distmat_corr = np.mean(dists)
                else:
                    # Just compare the distmats, without splitting by context.
                    vec1 = Cldist1.Xinput[(ma_ut)].flatten()
                    vec2 = Cldist2.Xinput[(ma_ut)].flatten()
                    tau, _ = stats.kendalltau(vec1, vec2)
                    distmat_corr = tau
            else:
                distmat_corr = None

            # Store results.
            if distmat_corr is not None:
                res_distmat_corrs.append({
                    "distmat_corr":distmat_corr,
                    "bregion_1":row_i["bregion"],
                    "event_1":row_i["event"],
                    "time_1":row_i["time"],
                    "bregion_2":row_j["bregion"],
                    "event_2":row_j["event"],
                    "time_2":row_j["time"],
                })

    dfres_corrs_btw_distmats = pd.DataFrame(res_distmat_corrs)
    dfres_corrs_btw_distmats["bregions_same"] = [row["bregion_1"] == row["bregion_2"] for _, row in dfres_corrs_btw_distmats.iterrows()]
    dfres_corrs_btw_distmats["events_same"] = [row["event_1"] == row["event_2"] for _, row in dfres_corrs_btw_distmats.iterrows()]
    
    ################### PLOTS
    if DO_PLOTS and SAVEDIR_PLOTS is not None:
        # (1) Heatmaps
        from pythonlib.tools.pandastools import plot_subplots_heatmap
        from pythonlib.tools.plottools import savefig
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.snstools import map_function_tofacet

        list_event_1= sorted(dfres_corrs_btw_distmats["event_1"].unique().tolist())
        list_event_2 = sorted(dfres_corrs_btw_distmats["event_2"].unique().tolist())

        # ZLIMS = [0., 0.8]
        ZLIMS = [0., None]
        # Across event (same bregion)
        for i, ev1 in enumerate(list_event_1):
            for j, ev2 in enumerate(list_event_2):
                if j>=i:
                    dfthis = dfres_corrs_btw_distmats[
                        (dfres_corrs_btw_distmats["event_1"] == ev1) & 
                        (dfres_corrs_btw_distmats["event_2"] == ev2) & 
                        (dfres_corrs_btw_distmats["bregions_same"] == True)].reset_index(drop=True)
                    fig, _ = plot_subplots_heatmap(dfthis, "time_1", "time_2", 
                                        "distmat_corr", "bregion_1", False, True,
                                        ZLIMS=ZLIMS);
                    path = f"{SAVEDIR_PLOTS}/heatmaps-ev1={ev1}-ev2={ev2}.pdf"
                    savefig(fig, path)

                    # (2) Timecourse plots (easier to see details)
                    fig = sns.relplot(data=dfthis, x="time_2", y="distmat_corr", hue="time_1", 
                                    col="bregion_1", col_wrap = 4, kind="line")
                    rotateLabel(fig)
                    map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
                    path = f"{SAVEDIR_PLOTS}/pointplotv1-ev1={ev1}-ev2={ev2}.pdf"
                    savefig(fig, path)

                    plt.close("all")

        # (3) separate figure for each time_1
        # Timecourse plots (easier to see details) --> one for each "from" time
        for time_1 in dfres_corrs_btw_distmats["time_1"].unique().tolist():
            for i, ev1 in enumerate(list_event_1):
                for j, ev2 in enumerate(list_event_2):
                    if j>=i:
                        dfthis = dfres_corrs_btw_distmats[
                            (dfres_corrs_btw_distmats["event_1"] == ev1) & 
                            (dfres_corrs_btw_distmats["event_2"] == ev2) & 
                            (dfres_corrs_btw_distmats["time_1"] > time_1-0.01) & (dfres_corrs_btw_distmats["time_1"] < time_1+0.01)].reset_index(drop=True)

                        fig = sns.relplot(data=dfthis, x="time_2", y="distmat_corr", hue="bregion_2", 
                                        col = "bregion_1", col_wrap = 4, kind="line")
                        map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
                        rotateLabel(fig)
                        path = f"{SAVEDIR_PLOTS}/pointplotv2-ev1={ev1}-ev2={ev2}-time1={time_1:.3f}.pdf"
                        savefig(fig, path)

                        plt.close("all")

        # (4) Nice summary plot -- average over "from" time windows to get a single curve
        event_start = "03_samp"
        twind_start = (0.1, 0.3)

        # Take average over all timecourses starting in the given time window.
        for j, ev2 in enumerate(list_event_2):
            dfthis = dfres_corrs_btw_distmats[
                (dfres_corrs_btw_distmats["event_1"] == event_start) &
                (dfres_corrs_btw_distmats["event_2"] == ev2) &
                (dfres_corrs_btw_distmats["time_1"] >= twind_start[0]-0.001) &
                (dfres_corrs_btw_distmats["time_1"] <= twind_start[1]+0.001)
                ]
            
            fig = sns.relplot(data=dfthis, x="time_2", y="distmat_corr", col="bregion_1", hue="bregion_2",
                                kind="line")
            map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
            rotateLabel(fig)

            path = f"{SAVEDIR_PLOTS}/pointplotagg_on_ev1={event_start}_t1={twind_start}-ev2={ev2}.pdf"
            savefig(fig, path)

            plt.close("all")

        # Similar, but different view
        dfthis = dfres_corrs_btw_distmats[
            (dfres_corrs_btw_distmats["event_1"] == event_start) &
            (dfres_corrs_btw_distmats["time_1"] >= twind_start[0]-0.001) &
            (dfres_corrs_btw_distmats["time_1"] <= twind_start[1]+0.001) &
            (dfres_corrs_btw_distmats["bregions_same"] == True)
            ]

        fig = sns.relplot(data=dfthis, x="time_2", y="distmat_corr", hue="event_2", col="bregion_1", 
                        col_wrap = 8, kind="line")
        map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
        rotateLabel(fig)

        path = f"{SAVEDIR_PLOTS}/pointplotagg_on_ev1_v2={event_start}_t1={twind_start}.pdf"
        savefig(fig, path)

        plt.close("all")


    return dfres_corrs_btw_distmats

def euclidian_distance_plot_timedistmat_heatmaps(DFRES, SAVEDIR,
        do_agg_over_vars=False, sort_order=None):
    """
    Plot heatmap of distance between conditions, one for each time bin.
    PARAMS:
    - DFRES, output of euclidian_distance_compute_trajectories_timedistmat
    """
    from pythonlib.tools.plottools import savefig
    from pythonlib.cluster.clustclass import Clusters

    # Sort by time
    DFRES = DFRES.sort_values("time", axis=0)

    # For each bregion and event, plot a series of RSA distmats one for each time bin.
    for bregion in DFRES["bregion"].unique():
        for event in DFRES["event"].unique():

            dfthis = DFRES[
                (DFRES["bregion"] == bregion) & 
                (DFRES["event"] == event)
                ]
            
            savedir = f"{SAVEDIR}/cldist_each_time/{bregion}-{event}"
            os.makedirs(savedir, exist_ok=True)

            print(bregion, event)
            list_time = dfthis["time"].tolist()
            list_Cldist = dfthis["Cldist"].tolist()
            
            list_Cldist_Agg = []
            if do_agg_over_vars:
                for i, (t, Cldist) in enumerate(zip(list_time, list_Cldist)):
                    # Before plot take means over levels, or else plot is too large
                    CldistAgg = Cldist.distsimmat_convert_distr_from_self()
                    list_Cldist_Agg.append(CldistAgg)
            else:
                list_Cldist_Agg = list_Cldist

            # Use same zlims for all plots
            maxes = []
            for i, (t, CldistAgg) in enumerate(zip(list_time, list_Cldist_Agg)):
                maxes.append(np.max(CldistAgg.Xinput))
            zlims = [0., max(maxes)]

            # (1) Plot time series of heat maps
            # Plot CLdist on each time

            for i, (t, CldistAgg) in enumerate(zip(list_time, list_Cldist_Agg)):
                # Make plot
                fig, ax = CldistAgg.rsa_plot_heatmap(zlims=zlims, sort_order=sort_order)
                savefig(fig, f"{savedir}/dist_heatmap-idx={i}-time={t:.2f}.pdf")
            plt.close("all")

            ### Plot average heatmap
            # 1. Times <= 0
            inds_this = [i for i, t in enumerate(list_time) if t<=0]
            if len(inds_this)>0:
                list_Cldist_Agg_this = [list_Cldist_Agg[i] for i in inds_this]
                Xinpput_mean = np.mean(np.stack([Cldist.Xinput for Cldist in list_Cldist_Agg_this], axis=0), axis=0)
                params = {
                    "label_vars":list_Cldist_Agg_this[0].Params["label_vars"],
                    "version_distance":list_Cldist_Agg_this[0].Params["version_distance"],
                    "Clraw":None,
                }
                list_lab = list_Cldist_Agg_this[0].Labels
                Cldist = Clusters(Xinpput_mean, list_lab, list_lab, ver="dist", params=params)
                fig, _ = Cldist.rsa_plot_heatmap(zlims=zlims, sort_order=sort_order)
                savefig(fig, f"{savedir}/dist_heatmap-MEAN-before_time_0.pdf")
                plt.close("all")

            # 1. Times > 0
            inds_this = [i for i, t in enumerate(list_time) if t>0]
            if len(inds_this)>0:
                list_Cldist_Agg_this = [list_Cldist_Agg[i] for i in inds_this]
                Xinpput_mean = np.mean(np.stack([Cldist.Xinput for Cldist in list_Cldist_Agg_this], axis=0), axis=0)
                params = {
                    "label_vars":list_Cldist_Agg_this[0].Params["label_vars"],
                    "version_distance":list_Cldist_Agg_this[0].Params["version_distance"],
                    "Clraw":None,
                }
                list_lab = list_Cldist_Agg_this[0].Labels
                Cldist = Clusters(Xinpput_mean, list_lab, list_lab, ver="dist", params=params)
                fig, _ = Cldist.rsa_plot_heatmap(zlims=zlims, sort_order=sort_order)
                savefig(fig, f"{savedir}/dist_heatmap-MEAN-after_time_0.pdf")
                plt.close("all")


def euclidian_distance_compute_trajectories_timedistmat(PA, var, var_others, 
                                        twind, tbin_dur, tbin_slice, savedir, 
                                        context=None, filtdict=None,
                                        PLOT_TRAJS=True, nmin_trials_per_lev=None,
                                        NPCS_KEEP=None,
                                        dim_red_method = "pca", superv_dpca_params=None,
                                        ):
    """
    Helper to get dsitance matrix between trajectories, one for each time bin. 
    This only uses a SINGLE var-var_others, since doing just one takes a long time, and the
    downstream plotting functions all assume there's just one var...

    WRapper, in that it does a bunch of things, like preprpocessing too, e.g., dim reduction.
    """
    DFRES = euclidian_distance_compute_trajectories(PA, [var], [var_others], twind, tbin_dur,
                               tbin_slice, savedir, PLOT_TRAJS, False,
                               nmin_trials_per_lev, [context], [filtdict], 
                               None, NPCS_KEEP, 
                               dim_red_method, superv_dpca_params, 
                               RETURN_EACH_TIMES_CLDIST=True)
    return DFRES


def euclidian_distance_compute_trajectories(PA, LIST_VAR, LIST_VARS_OTHERS, twind, tbin_dur,
                               tbin_slice, savedir, 
                               PLOT_TRAJS=True, PLOT_HEATMAPS=False,
                               nmin_trials_per_lev=None,
                               LIST_CONTEXT=None, LIST_FILTDICT=None,
                               LIST_PRUNE_MIN_N_LEVS=None,
                               NPCS_KEEP=None,
                               dim_red_method = "pca", superv_dpca_params=None,
                               RETURN_EACH_TIMES_CLDIST=False,
                               PLOT_CLEAN_VERSION = False,
                               COMPUTE_EUCLIDIAN=True, 
                               get_reverse_also = False,
                               PLOT_MASKS=False
                               ):
    """
    Wrapper to compute all distances between levels of variables, with flecxible abilties for
    controlling context. -- see within for details ...

    :param PA: (chans, trials, times), where len(times) can be > 1.
    :param LIST_VAR:
    :param LIST_VARS_OTHERS:
    :param PLOT:
    :param PLOT_MASKS:
    :param twind:
    :param tbin_dur:
    :param tbin_slice:
    :param savedir:
    :param SHUFFLE:
    :param PLOT_STATE_SPACE:
    :param nmin_trials_per_lev:
    :param LIST_CONTEXT: list of dicts each dict has "same" and :"diff" keys, list of str vars, defining the context for the
    matchging var and vars_others.
    :return:
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    from pythonlib.cluster.clustclass import Clusters

    assert PA.X.shape[1]>0, "empty trials..."

    PA = PA.copy()
    assert NPCS_KEEP is not None, "coded onoy for this so far -0 forceing an npcs"

    ######################## (1) CONSTRUCT A SINGLE SUBSPACE (that all subsequence plots and analyses will be performed on)
    # Get specific params, based on how want to do dim reduction
    if superv_dpca_params is None:
        dpca_var = None
        dpca_vars_group = None
        dpca_filtdict=None
        dpca_proj_twind = None
    else:
        dpca_var = superv_dpca_params["superv_dpca_var"]
        dpca_vars_group = superv_dpca_params["superv_dpca_vars_group"]
        dpca_filtdict = superv_dpca_params["superv_dpca_filtdict"]
        if "dpca_proj_twind" in superv_dpca_params:
            dpca_proj_twind = superv_dpca_params["dpca_proj_twind"]
        else:
            dpca_proj_twind = twind
        if "dpca_pca_twind" in superv_dpca_params:
            # HACKY, for nice trajs
            # This is for construting p
            twind = superv_dpca_params["dpca_pca_twind"]
        
    # if dpca_filtdict=={"task_kind":["prims_on_grid"], "stroke_index":[0]}:
    #     print("=====")
    #     print(PA.X.shape)
    #     print("input task kinds: ", PA.Xlabels["trials"]["task_kind"].unique())
        
    _, PAredu = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
                                   twind, tbin_dur=tbin_dur, tbin_slide=tbin_slice, 
                                   NPCS_KEEP = NPCS_KEEP,
                                   dpca_var = dpca_var, dpca_vars_group = dpca_vars_group, dpca_filtdict=dpca_filtdict, dpca_proj_twind = dpca_proj_twind, 
                                   raw_subtract_mean_each_timepoint=False,
                                   umap_n_components=None, umap_n_neighbors=None)
    # if dpca_filtdict=={"task_kind":["prims_on_grid"], "stroke_index":[0]}:
    #     print("=====")
    #     print(PAredu.X.shape)
    #     assert False
    
    if PAredu is None:
        # Then filtiering cleared all trials...
        print("----")
        print("Input shape: ", PA.X.shape)
        print("Lost all trials, using this fildtict:", dpca_filtdict)
        return None

    # ######################## (1) CONSTRUCT A SINGLE SUBSPACE (that all subsequence plots and analyses will be performed on)
    # # Get specific params, based on how want to do dim reduction
    # METHOD = "basic"
    # if dim_red_method is None:
    #     # Then use raw data
    #     pca_reduce = False
    #     extra_dimred_method = None
    # elif dim_red_method=="pca":
    #     pca_reduce = True
    #     extra_dimred_method = None
    # elif dim_red_method=="pca_umap":
    #     # PCA --> UMAP
    #     pca_reduce = True
    #     extra_dimred_method = "umap"
    # elif dim_red_method=="umap":
    #     # UMAP
    #     pca_reduce = False
    #     extra_dimred_method = "umap"
    # elif dim_red_method=="mds":
    #     # MDS
    #     pca_reduce = False
    #     extra_dimred_method = "mds"
    # elif dim_red_method=="superv_dpca":
    #     # Supervised, based on DPCA, find subspace for a given variable by doing PCA on the mean values.
    #     superv_dpca_var = superv_dpca_params["superv_dpca_var"]
    #     superv_dpca_vars_group = superv_dpca_params["superv_dpca_vars_group"]
    #     superv_dpca_filtdict = superv_dpca_params["superv_dpca_filtdict"]
    #     METHOD = "dpca"
    # else:
    #     print(dim_red_method)
    #     assert False

    # reshape_method = "chans_x_trials_x_times"
    # if METHOD=="basic":
    #     # 1. Dim reduction
    #     # - normalize - remove time-varying component
    #     PA = PA.norm_subtract_trial_mean_each_timepoint()
        
    #     # - PCA
    #     plot_pca_explained_var_path=f"{savedir}/pcaexp.pdf"
    #     plot_loadings_path = f"{savedir}/pcaload.pdf"
    #     umap_n_neighbors = 40
    #     _, PAredu, _, _, _ = PA.dataextract_state_space_decode_flex(twind, tbin_dur, tbin_slice, reshape_method=reshape_method,
    #                                                 pca_reduce=pca_reduce, plot_pca_explained_var_path=plot_pca_explained_var_path, 
    #                                                 plot_loadings_path=plot_loadings_path, npcs_keep_force=NPCS_KEEP,
    #                                                 extra_dimred_method=extra_dimred_method, umap_n_neighbors = umap_n_neighbors)    
    #     n_pcs_keep_euclidian = PAredu.X.shape[1]

    # elif METHOD=="dpca":
    #     # from neuralmonkey.classes.population import PopAnal
    #     print("... dpca")

    #     savedirthis = f"{savedir}/pca_construction"
    #     os.makedirs(savedirthis, exist_ok=True)
    #     PLOT_STEPS = False
    #     _, PAredu, _, _, pca = PA.dataextract_pca_demixed_subspace(
    #         superv_dpca_var, superv_dpca_vars_group, twind, tbin_dur, superv_dpca_filtdict, savedirthis,
    #         n_min_per_lev_lev_others=nmin_trials_per_lev, PLOT_STEPS=PLOT_STEPS, reshape_method=reshape_method,
    #         pca_tbin_slice=tbin_slice)

        
    #     if PAredu is None:
    #         # Then no data...
    #         return None
        
    #     # Save a version with full D, for state space
    #     # PAredu_orig_dim = PAredu.copy()

    #     # Figure out how many dimensions to keep (for euclidian).
    #     n1 = pca["nclasses_of_var_pca"] # num classes of superv_dpca_var that exist. this is upper bound on dims.
    #     n2 = PAredu.X.shape[1] # num classes to reach criterion for cumvar for pca.
    #     n3 = PAredu.X.shape[0] # num dimensions.
    #     n_pcs_keep_euclidian = min([n1, n2, n3, NPCS_KEEP])
        
    #     PAredu = PAredu.slice_by_dim_indices_wrapper("chans", list(range(n_pcs_keep_euclidian)))

    #     # # - prune data for euclidian
    #     # Xredu = Xredu[:, :n_pcs_keep_euclidian]
    #     # dflab = PAredu.Xlabels.copy()
    #     # PAredu = PopAnal(Xredu.T[:, :, None], [0])  # (ndimskeep, ntrials, 1)
    #     # PAredu.Xlabels = {dim:df.copy() for dim, df in dflab.items()}

    # else:
    #     print(METHOD)
    #     assert False    

    ############################ (2) Euclidian and State space plots
    if LIST_CONTEXT is not None:
        assert len(LIST_CONTEXT)==len(LIST_VAR)
    else:
        LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]

    if LIST_FILTDICT is not None:
        assert len(LIST_FILTDICT)==len(LIST_VAR)
    else:
        LIST_FILTDICT = [None for _ in range(len(LIST_VAR))]

    if LIST_PRUNE_MIN_N_LEVS is not None:
        assert len(LIST_PRUNE_MIN_N_LEVS)==len(LIST_VAR)
    else:
        LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]

    ############ SCore, for each variable
    RES = []
    vars_already_state_space_plotted = []
    var_varothers_already_plotted = []
    heatmaps_already_plotted = []
    for i_var, (var, var_others, context, filtdict, prune_min_n_levs) in enumerate(zip(LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS)):
        print("RUNNING: ", i_var,  var, " -- ", var_others)

        # Copy pa for this
        pa = PAredu.copy()

        ####################### Cleanup PA
        var_for_name = var
        if isinstance(var, (tuple, list)):
            pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], var, "_tmp")
            # pa_orig_dim.Xlabels["trials"] = append_col_with_grp_index(pa_orig_dim.Xlabels["trials"], var, "_tmp")
            var = "_tmp"

        if filtdict is not None:
            for _var, _levs in filtdict.items():
                print("len pa bnefore filt this values (var, levs): ", _var, _levs)
                pa = pa.slice_by_labels("trials", _var, _levs, verbose=True)
                # pa_orig_dim = pa_orig_dim.slice_by_labels("trials", _var, _levs)

        if nmin_trials_per_lev is not None:
            prune_min_n_trials = nmin_trials_per_lev
        else:
            prune_min_n_trials = N_MIN_TRIALS

        
        if (var, tuple(var_others)) not in heatmaps_already_plotted:
            plot_counts_heatmap_savepath = f"{savedir}/{i_var}_counts_heatmap-var={var_for_name}-ovar={'|'.join(var_others)}.pdf"
            heatmaps_already_plotted.append((var, tuple(var_others)))
        else:
            plot_counts_heatmap_savepath = None

        # pa_before_prune = pa.copy()
        if not PLOT_TRAJS:
            plot_counts_heatmap_savepath = None
        pa, _, _= pa.slice_extract_with_levels_of_conjunction_vars(var, var_others, prune_min_n_trials, prune_min_n_levs,
                                                         plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
        if pa is None:
            print("all data pruned!!")
            continue

        ##############################       
        if COMPUTE_EUCLIDIAN: 
            # Two options:
            if RETURN_EACH_TIMES_CLDIST:
                # Collect Cldists, one for each time bin
                version_distance = "euclidian_unbiased"
                print("... dataextract_as_distance_matrix_clusters_flex")
                LIST_CLDIST, LIST_TIME = pa.dataextract_as_distance_matrix_clusters_flex([var] + var_others, 
                                                                                        version_distance=version_distance,
                                                                                        accurately_estimate_diagonal=False)
                
                # Collect
                for cldist, time in zip(LIST_CLDIST, LIST_TIME):
                    RES.append({
                        "var":var,
                        "var_others":tuple(var_others),
                        "shuffled":False,
                        "shuffled_iter":-1,
                        "index_var":i_var,
                        "Cldist":cldist,
                        "time":time  
                    })
            else:
                # Get a single score taking mean over all time (entire traj)
                # 3. Compute euclidian
                savedir_heatmaps = f"{savedir}/heatmap_average.pdf"
                if PLOT_TRAJS:
                    dir_to_print_lab_each_mask = f"{savedir}/final_labels_data_pairs_in_masks-{i_var}-var={var}-ovar={var_others}"
                    os.makedirs(dir_to_print_lab_each_mask, exist_ok=True)
                else:
                    dir_to_print_lab_each_mask = None
                res = euclidian_distance_compute_trajectories_single(pa, var, var_others, version_distance="euclidian",
                                                                    context_input=context, PLOT_HEATMAPS=PLOT_HEATMAPS, 
                                                                    savedir_heatmaps=savedir_heatmaps,
                                                                    dir_to_print_lab_each_mask=dir_to_print_lab_each_mask,
                                                                    get_reverse_also = get_reverse_also, PLOT_MASKS=PLOT_MASKS)
                for r in res:
                    r["shuffled"] = False
                    r["shuffled_iter"] = -1
                    r["index_var"] = i_var

                RES.extend(res)

        ##############################
        # 2. Plot trajectories
        if PLOT_TRAJS:
            pathis = pa # Use the pruned data for plots.
            pathis_scalar = pa.agg_wrapper("times")
            # pathis = PAredu
            
            if len(LIST_VAR)<15:
                if PAredu.X.shape[0]==3:
                    list_dims = [(0,1), (1,2)]
                elif PAredu.X.shape[0]>3:
                    list_dims = [(0,1), (2,3)]
                else:
                    list_dims = [(0,1)]
            else:
                # Too slow, just do 1st 2 d
                list_dims = [(0,1)]

            time_bin_size = 0.05
            # savedir_this = f"{savedir}/trajectories"
            # os.makedirs(savedir_this, exist_ok=True)
            from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER

            if (var, var_others) not in var_varothers_already_plotted:
                var_varothers_already_plotted.append((var, tuple(var_others)))
                
                if PLOT_CLEAN_VERSION == False:
                    trajgood_plot_colorby_splotby_WRAPPER(pathis.X, pathis.Times, pathis.Xlabels["trials"], var, 
                                                        savedir, var_others, list_dims, 
                                                        time_bin_size=time_bin_size, save_suffix=i_var)
                    
                    # -- Plot scalar scatterplot too.
                    dflab = pathis_scalar.Xlabels["trials"]
                    Xthis = pathis_scalar.X.squeeze(axis=2).T # (n4trials, ndims)
                    sdir = f"{savedir}/SCALAR"
                    os.makedirs(sdir, exist_ok=True)
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, sdir,
                                                                    vars_subplot=var_others, list_dims=list_dims,
                                                                    skip_subplots_lack_mult_colors=False, save_suffix = i_var)
                    var_varothers_already_plotted.append((var, tuple(var_others)))

                    plt.close("all")
                else:
                    # Plot a "clean" version (Paper version), including with different x and y lims, so can compare
                    # across plots
                    
                    ssuff = f"{i_var}"
                    trajgood_plot_colorby_splotby_WRAPPER(pathis.X, pathis.Times, pathis.Xlabels["trials"], var, 
                                                        savedir, var_others, list_dims, 
                                                        time_bin_size=None, save_suffix=ssuff,
                                                        plot_dots_on_traj=False)
                    
                    for xlim_force in [
                        [-3.2, 3.2],
                        [-2.4, 2.4],
                        ]:
                        for ylim_force in [
                            [-1.5, 1.5],
                            [-2, 2],
                            [-2.5, 2.5],
                            ]:
                            ssuff = f"{i_var}--xylim={xlim_force}|{ylim_force}"
                            trajgood_plot_colorby_splotby_WRAPPER(pathis.X, pathis.Times, pathis.Xlabels["trials"], var, 
                                                                savedir, var_others, list_dims, 
                                                                time_bin_size=None, save_suffix=ssuff,
                                                                plot_dots_on_traj=False,
                                                                xlim_force = xlim_force, ylim_force=ylim_force)



            if var not in vars_already_state_space_plotted:
                vars_already_state_space_plotted.append(var)
                if PLOT_CLEAN_VERSION:
                    # Plot a "clean" version (Paper version), including with different x and y lims, so can compare
                    # across plots
                    trajgood_plot_colorby_splotby_WRAPPER(pathis.X, pathis.Times, pathis.Xlabels["trials"], var, 
                                                        savedir, None, list_dims, 
                                                        time_bin_size=None, save_suffix=i_var,
                                                        plot_dots_on_traj=False)

                    for xlim_force in [
                        [-3.2, 3.2],
                        [-2.4, 2.4],
                        ]:
                        for ylim_force in [
                            [-1.5, 1.5],
                            [-2, 2],
                            [-2.5, 2.5],
                            ]:
                            ssuff = f"{i_var}--xylim={xlim_force}|{ylim_force}"
                            trajgood_plot_colorby_splotby_WRAPPER(pathis.X, pathis.Times, pathis.Xlabels["trials"], var, 
                                                                savedir, None, list_dims, 
                                                                time_bin_size=None, save_suffix=ssuff,
                                                                plot_dots_on_traj=False,
                                                                xlim_force = xlim_force, ylim_force=ylim_force)
                else:
                    trajgood_plot_colorby_splotby_WRAPPER(pathis.X, pathis.Times, pathis.Xlabels["trials"], var, 
                                                        savedir, None, list_dims, 
                                                        time_bin_size=time_bin_size, save_suffix=i_var)
                    
                    # -- Plot scalar scatterplot too.
                    dflab = pathis_scalar.Xlabels["trials"]
                    Xthis = pathis_scalar.X.squeeze(axis=2).T # (n4trials, ndims)
                    sdir = f"{savedir}/SCALAR"
                    os.makedirs(sdir, exist_ok=True)
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xthis, dflab, var, sdir,
                                                                    vars_subplot=None, list_dims=list_dims,
                                                                    skip_subplots_lack_mult_colors=False, save_suffix = i_var)
                    var_varothers_already_plotted.append((var, tuple(var_others)))

                    plt.close("all")

                plt.close("all")
            
            # Also plot timecourse
            from neuralmonkey.analyses.state_space_good import trajgood_construct_df_from_raw, trajgood_plot_colorby_splotby_timeseries
            plot_trials_n = 5
            df = trajgood_construct_df_from_raw(pathis.X, pathis.Times, pathis.Xlabels["trials"], [var] + var_others)
            
            for dim in [0,1]:
            
                # - (i) combined, plotting means.
                fig, _ = trajgood_plot_colorby_splotby_timeseries(df, var, var_others, dim=dim,
                                                                plot_trials_n=plot_trials_n, 
                                                                SUBPLOT_OPTION="split_levs")
                path = f"{savedir}/TIMECOURSEsplit-color={var}-sub={var_others}-dim={dim}-suff={i_var}.pdf"
                print("Saving ... ", path)
                savefig(fig, path)

                # - (2) split
                fig, _ = trajgood_plot_colorby_splotby_timeseries(df, var, var_others, dim=dim, plot_trials_n=plot_trials_n,
                                                        plot_trials=False, SUBPLOT_OPTION="combine_levs")
                path = f"{savedir}/TIMECOURSEcomb-color={var}-sub={var_others}-dim={dim}-suff={i_var}.pdf"
                print("Saving ... ", path)
                savefig(fig, path)
                
                plt.close("all")


    dfres = pd.DataFrame(RES)
    if len(dfres)>0:
        dfres["twind_analy"] = [twind for _ in range(len(dfres))]
    plt.close("all")

    return dfres


def euclidian_distance_compute_trajectories_single(PA, var_effect, vars_others, context_input=None,
                                                   version_distance="euclidian",
                                                PLOT_HEATMAPS=False, savedir_heatmaps=None, PLOT_MASKS=False,
                                                get_reverse_also=True, dir_to_print_lab_each_mask=None,
                                                return_cldist=False, compute_same_diff_scores=True):
    """
    [GOOD]

    Compute distance between trajectories (each pair of trajs (i.e, triualsd) returns a 
    single scalar distances, which is the average distance over time)

    """
    from pythonlib.cluster.clustclass import Clusters
    import numpy as np

    if vars_others is None:
        vars_others = []

    if compute_same_diff_scores:
        assert len(vars_others)>0

    # # 3. Compute euclidian
    # # For each timepoint, compute
    # # - 
    # if False:
    #     tbin = 0.05
    #     pa = PAredu.agg_by_time_windows_binned(tbin, SLIDE=tbin)
    
    # Methoid 1 -- hacky for loop thru all time
    # - NOTE: this is actually same speed as other version

    if version_distance == "euclidian":
        dat_level = "pts"
    elif version_distance == "euclidian_unbiased":
        dat_level = "distr"
    else:
        assert False, "code it"

    if get_reverse_also:
        LIST_REVERSE = [False, True]
    else:
        LIST_REVERSE = [False]

    list_res = []
    Cldist_good = None
    for DO_REVERSE_CONTROL in LIST_REVERSE:
        if DO_REVERSE_CONTROL:
            niter = 3
        else:
            niter = 1

        for _i in range(niter):

            ### COMPUTE TRAJECTORY DISTANCES
            # Collect Cldists, one for each time bin
            # Collect 
            if DO_REVERSE_CONTROL:
                assert False, "Fix this, the return_as_single_mean_over_time deosnt amke sense"
                Cldist = PA.dataextract_as_distance_matrix_clusters_flex_reversed([var_effect] + vars_others, 
                                                                                            version_distance=version_distance,
                                                                                            return_as_single_mean_over_time=True)
            else:
                Cldist = PA.dataextract_as_distance_matrix_clusters_flex([var_effect] + vars_others, 
                                                                                            version_distance=version_distance,
                                                                                            return_as_single_mean_over_time=True)
                Cldist_good = Cldist

            # ### Take mean distance over time, and construct a single Clusters
            # Xinpput_mean = np.mean(np.stack([Cldist.Xinput for Cldist in LIST_CLDIST], axis=0), axis=0)
            # params = {
            #     "label_vars":LIST_CLDIST[0].Params["label_vars"],
            #     "version_distance":LIST_CLDIST[0].Params["version_distance"],
            #     "Clraw":None,
            # }
            # list_lab = LIST_CLDIST[0].Labels
            # Cldist = Clusters(Xinpput_mean, list_lab, list_lab, ver="dist", params=params)

            if PLOT_HEATMAPS and savedir_heatmaps is not None:
                fig, _ = Cldist.rsa_plot_heatmap()
                if fig is not None:
                    savefig(fig, f"{savedir_heatmaps}/heatmap_average.pdf")
                    plt.close("all")
            
            ### Compute scores (same vs. diff)
            if compute_same_diff_scores:
                # dir_to_print_lab_each_mask = # good to always save the final data pairs.
                print("var_effect --- vars_others --- context_input")
                print(var_effect, " --- ", vars_others, " --- ", context_input)
                res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98 = Cldist.rsa_distmat_score_same_diff_by_context(var_effect, vars_others, 
                                                                                            context_input, dat_level, 
                                                                                            PLOT_MASKS,
                                                                                            dir_to_print_lab_each_mask=dir_to_print_lab_each_mask)
                for r in res:
                    r["DIST_NULL_98"] = DIST_NULL_98
                    r["shuffled_time"] = DO_REVERSE_CONTROL
                    r["shuffled_time_iter"] = _i
                    
                list_res.extend(res)

    ##### Version 2 -- quicker, treat trajectoreis as single datapts [IGNORE, working, but decided above is better, intergartes with my code better]

    # DFallpa
    # var_effect = "shape"
    # vars_others = ["gridloc"]

    # from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

    # PA = DFallpa["pa"].values[2]
    # bregion = DFallpa["bregion"].values[2]
    # print(bregion)

    # # Do dim reductions

    # # - normalize - remove time-varying component
    # PA = PA.norm_subtract_trial_mean_each_timepoint()

    # SAVEDIR = f"/tmp/PCA_time_v2/{bregion}-var={var_effect}-var_others={'--'.join(vars_others)}"
    # os.makedirs(SAVEDIR, exist_ok=True)

    # reshape_method = "chans_x_trials_x_times"

    # # Just PCA
    # plot_pca_explained_var_path=f"{SAVEDIR}/pcaexp.pdf"
    # plot_loadings_path = f"{SAVEDIR}/pcaload.pdf"

    # pca_reduce = True
    # extra_dimred_method = None
    # umap_n_neighbors = 40
    # Xredu, PAredu, PAslice, pca, _ = PA.dataextract_state_space_decode_flex(twind, tbin_dur, tbin_slice, reshape_method=reshape_method,
    #                                             pca_reduce=pca_reduce, plot_pca_explained_var_path=plot_pca_explained_var_path, plot_loadings_path=plot_loadings_path, npcs_keep_force=NPCS_KEEP,
    #                                             extra_dimred_method=extra_dimred_method, umap_n_neighbors = umap_n_neighbors)    
    # # (compare to dPCA?)

    # # if False:
    # ##############################
    # # 2. Plot trajectories
    # from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER
    # savedir = f"{SAVEDIR}/trajectories"
    # os.makedirs(savedir, exist_ok=True)
    # var_color = var_effect
    # vars_subplot = vars_others
    # list_dims = [(0,1), (2,3)]
    # time_bin_size = 0.05

    # trajgood_plot_colorby_splotby_WRAPPER(PAredu.X, PAredu.Times, PAredu.Xlabels["trials"], var_color, savedir, vars_subplot, list_dims,
    #                                         time_bin_size=time_bin_size, save_suffix=bregion)


    # # Get data
    # dflab = pa.Xlabels["trials"]
    # X = pa.X
    # assert len(dflab)==X.shape[1]

    # # X = self.Xinput
    # label_vars = [var]+var_others

    # # Get grouping of labels
    # groupdict = grouping_append_and_return_inner_items(dflab, label_vars)

    # # Convert to list, which can then pass into distmat constructor
    # list_X = []
    # list_lab = []
    # list_exclude = []
    # for i, (grp, inds) in enumerate(groupdict.items()):
    #     print(grp, len(inds), X[:, inds, :].shape)
    #     list_X.append(X[:, inds, :])
    #     list_lab.append(grp)
        
    # # get pairwise distances

    # # - distance function
    # from pythonlib.tools.distfunctools import euclidian_unbiased, distmat_construct_wrapper, euclidian

    # def euclidian_unbiased_over_time(x1, x2):
    #     """
    #     Gets distance at each timepoint, and averages them
    #     :param x1: (dims, ndat_1, ntimes) 
    #     :param x2: (dims, ndat_2, ntimes)
    #     :return: scalar distance,  
    #     """
    #     import numpy as np
    #     assert x1.shape[2] == x2.shape[2]
    #     ntimes = x1.shape[2]
        
    #     list_d = []
    #     for i_time in range(ntimes):
    #         x1_slice = x1[:, :, i_time]
    #         x2_slice = x2[:, :, i_time]
    #         list_d.append(euclidian_unbiased(x1_slice.T, x2_slice.T))
        
    #     return np.mean(list_d)
        
    # accurately_estimate_diagonal = False

    # D = distmat_construct_wrapper(list_X, list_X, euclidian_unbiased_over_time,
    #                             accurately_estimate_diagonal=accurately_estimate_diagonal)

    # # Construct Cl

    # params = {
    #     "version_distance":None,
    #     "Clraw":None,
    # }
    # Cldist = Clusters(D, list_lab, list_lab, ver="dist", params=params)


    # Cldist.rsa_plot_heatmap()

    if return_cldist:
        return Cldist_good, list_res
    else:
        return list_res


def euclidian_distance_compute_AnBmCk_endpoint(PAredu, SAVEDIR):
    """ Qyiuck hapcky, to plot things relate to showing whetherh last strokes algine
    """
    # twind = (-0.1, 0.2)
    # tbin_dur = 0.1
    # tbin_slice = 0.1
    # if DO_DEMIXED:
    #     vars_subract_mean = ["epoch", "chunk_rank", "shape", "syntax_concrete", "gridloc"]
    #     var_pca = "chunk_within_rank_semantic_v2"
    #     filtdict = None
    #
    #     savedir = f"{SAVEDIR}/test_gridloc"
    #     os.makedirs(savedir, exist_ok=True)
    #     PLOT_STEPS = False
    #     Xredu, PAredu, stats_redu, Xfinal_before_redu, pca = PA.dataextract_pca_demixed_subspace(var_pca, vars_subract_mean,
    #                                                twind, tbin_dur, filtdict, savedir,
    #                                                 PLOT_STEPS=PLOT_STEPS)
    # else:
    #     # Just PCA
    #     plot_pca_explained_var_path=f"{SAVEDIR}/pcaexp.pdf"
    #     plot_loadings_path = f"{SAVEDIR}/pcaload.pdf"
    #
    #
    #     pca_reduce = True
    #     NPCS_KEEP = 10
    #     extra_dimred_method = None
    #     umap_n_neighbors = 40
    #     Xredu, PAredu, PAslice, pca, _ = PA.dataextract_state_space_decode_flex(twind, tbin_dur, tbin_slice, reshape_method="trials_x_chanstimes",
    #                                                pca_reduce=pca_reduce, plot_pca_explained_var_path=plot_pca_explained_var_path, plot_loadings_path=plot_loadings_path, npcs_keep_force=NPCS_KEEP,
    #                                               extra_dimred_method=extra_dimred_method, umap_n_neighbors = umap_n_neighbors)

    from neuralmonkey.analyses.rsa import _rsagood_convert_PA_to_Cl
    from pythonlib.tools.plottools import savefig
    from itertools import permutations
    from pythonlib.tools.plottools import savefig

    sdir = f"{SAVEDIR}"

    dflab = PAredu.Xlabels["trials"]
    Xredu = PAredu.X.squeeze(axis=2).T # (ntrials, ndims)

    ### Compute state similarity for chunk_within_indices

    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_scalar_splitmeanlines
    var_color = "chunk_within_rank_fromlast"
    # var_lines_within_subplot = ["syntax_concrete"]
    var_lines_within_subplot = ["chunk_n_in_chunk"]
    vars_subplot = ["chunk_rank", "shape"]
    n_min = 4

    # Hand enter how to color things
    from pythonlib.tools.listtools import sort_mixed_type
    fig = trajgood_plot_colorby_scalar_splitmeanlines(Xredu, dflab, var_color, var_lines_within_subplot, vars_subplot,
                                                    plot_method="overlay_mean_lines",
                                                    desired_levels_var_color_in_order = None,
                                                    map_linelev_to_color = None, dims = (0,1),
                                                    n_min_across_all_levs_var=4,
                                                    lenient_allow_data_if_has_n_levels=1,
                                                    SIZE = 5)
    savefig(fig, f"{SAVEDIR}/overlay_mean_lines.pdf")

    fig = trajgood_plot_colorby_scalar_splitmeanlines(Xredu, dflab, var_color, var_lines_within_subplot, vars_subplot,
                                                    plot_method="separate_scatters",
                                                    desired_levels_var_color_in_order = None,
                                                    map_linelev_to_color = None, dims = (0,1),
                                                    n_min_across_all_levs_var=4,
                                                    lenient_allow_data_if_has_n_levels=1,
                                                    SIZE = 5)
    savefig(fig, f"{SAVEDIR}/overlay_mean_lines_scatter.pdf")


    grouping_vars = ["chunk_rank", "shape", "chunk_n_in_chunk", "chunk_within_rank"]
    DO_AGG_TRIALS = False
    version_distance = "euclidian_unbiased"
    Clraw, Clsim = _rsagood_convert_PA_to_Cl(PAredu, grouping_vars, version_distance, DO_AGG_TRIALS)
    Clsim.Xinput
    get_all_sort_orders = False
    sort_order_force = [ 1, 2, 3, 0]
    # Plot heatmaps, in every sort order of variables.


    if get_all_sort_orders:
        if len(grouping_vars)<5:
            list_sort_order = permutations(range(len(grouping_vars)))
        else:
            list_sort_order = [list(range(len(grouping_vars)))]
    else:
        #  Just get the defautl
        list_sort_order = [list(range(len(grouping_vars)))]

    for sort_order in list_sort_order:

        if version_distance in ["_pearson_raw"]:
            diverge = True
        else:
            diverge = False
        figsim, ax = Clsim.rsa_plot_heatmap(sort_order, diverge=diverge)

        # - name this sort order
        main_var = grouping_vars[sort_order[0]]
        s = "_".join([str(i) for i in sort_order])
        s+=f"_{main_var}"

        path = f"{sdir}/heat_sim-sort_order_{s}.pdf"
        savefig(figsim, path)


        if len(Clraw.Labels) > len(Clsim.Labels):
            # then Clraw is not agged data. First agg before plotting raw
            from pythonlib.cluster.clustclass import Clusters

            PAagg, _ = PAredu.slice_and_agg_wrapper("trials", grouping_vars, return_group_dict=True)
            X = PAagg.X.squeeze().T # (ndat, nchans)
            labels_rows = PAagg.Xlabels["trials"].loc[:, grouping_vars].values.tolist()
            labels_rows = [tuple(x) for x in labels_rows] # list of tuples
            labels_cols = PAagg.Chans # list of ints
            params = {
                "label_vars":grouping_vars,
            }
            ClrawAGG = Clusters(X, labels_rows, labels_cols, ver="rsa", params=params)
            figraw, ax = ClrawAGG.rsa_plot_heatmap(sort_order, diverge=True)
            path = f"{sdir}/heat_raw_AGG-sort_order_{s}.pdf"
            savefig(figraw, path)
        else:
            figraw, ax = Clraw.rsa_plot_heatmap(sort_order, diverge=True)
            path = f"{sdir}/heat_raw-sort_order_{s}.pdf"
            savefig(figraw, path)

        plt.close("all")

    import numpy as np
    # Plot specific slices, to make easier to visualize
    from pythonlib.tools.plottools import makeColors
    MASKS = Clsim.rsa_mask_context_split_levels_of_conj_var(vars_context=["chunk_rank", "shape", "chunk_n_in_chunk"], contrast="any",
                                                            exclude_diagonal=False)

    for grp, ma in MASKS.items():

        fig, ax = Clsim.rsa_plot_heatmap(diverge=diverge, mask=ma)
        path = f"{sdir}/heat_sim-grp={grp}.pdf"

        # # Given a mask, overlay bounding box
        # x1 = 2
        # x2 = 4
        # y1 = 2
        # y2 = 4
        # Clsim.rsa_matindex_plot_bounding_box(x1, x2, y1, y2, ax)

        # HACKY - place vertical lines at bounding boxes demarcating columns that are groueped, based
        # on having same level of vars_context_bounding_box
        vars_context_bounding_box = ["chunk_rank", "shape", "chunk_n_in_chunk"]
        MASKS_bb = Clsim.rsa_mask_context_split_levels_of_conj_var(vars_context_bounding_box, exclude_diagonal=True)
        pcols = makeColors(len(MASKS_bb), cmap="winter")
        for _i, (grp_bb, ma_bb) in enumerate(MASKS_bb.items()):
            if np.any(ma_bb):
                # print(grp, np.argwhere(ma))
                # print(grp_bb, np.argwhere(ma_bb))
                x1 = min(np.argwhere(ma_bb)[:,0])
                x2 = max(np.argwhere(ma_bb)[:,1])
                # y1 = min(np.argwhere(ma_bb)[:,1])
                # y2 = max(np.argwhere(ma_bb)[:,1])
                y1 = min(np.argwhere(ma)[:,1])
                y2 = max(np.argwhere(ma)[:,1])

                # print(x1, x2, y1, y2)
                col = pcols[_i]
                Clsim.rsa_matindex_plot_bounding_box(x1, x2, y1, y2, ax, edgecolor=col)

                # assert False
        print(path)
        savefig(fig, path)
        plt.close("all")


    ##### QUantify, take different slices
    dfdists = Clsim.rsa_dataextract_with_labels_as_flattened_df()
    yvar = "dist"
    savedir = f"{SAVEDIR}"

    # Each subplot is a

    vars_figures = "shape"

    vars_subplot_cols = "chunk_n_in_chunk_col"
    var_lines_within_subplots = "chunk_within_rank_col"
    # var_lines_within_subplots = "chunk_within_rank_col"

    vars_subplot_rows = "chunk_n_in_chunk_row"
    var_x_axis_each_subplot = "chunk_within_rank_row"
    # var_x_axis_each_subplot = "chunk_within_rank_row"

    Clsim.rsa_plot_points_split_by_var_flex(var_x_axis_each_subplot, yvar,
                                              var_lines_within_subplots,
                                                savedir,
                                              vars_subplot_rows, vars_subplot_cols,
                                              vars_figures)
    # For debuggin bove, seeing that conjcuintiosn matchf igure
    a = dfdists["shape_row"] == "arcdeep-4-3-0"
    b = dfdists["shape_col"] == "arcdeep-4-3-0"
    # a = True
    # b = True
    c = dfdists["chunk_n_in_chunk_row"]==1
    d = dfdists["chunk_n_in_chunk_col"]==2
    e = dfdists["chunk_within_rank_row"]==0
    f = dfdists["chunk_within_rank_col"]==1

    dfdists[a & b & c & d & e & f]

    # Combine all chunk_n_in_chunk (columns)
    if False:
        vars_subplot_rows = None
        Clsim.rsa_plot_points_split_by_var_flex(var_x_axis_each_subplot, yvar,
                                                  var_lines_within_subplots,
                                                    savedir,
                                                  vars_subplot_rows, vars_subplot_cols,
                                                  vars_figures)
    plt.close("all")