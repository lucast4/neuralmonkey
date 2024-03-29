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

def _popanal_preprocess_normalize(PA, PLOT=False):
    """ Normalize firing rates so that similar acorss neruons (higha nd low fr) whiel
    still having higher for high fr.
    Similar to "soft normalization" used by Churchland group.
    """

    x = PA.X.reshape(PA.X.shape[0], -1)

    # STD (across all trials and times)
    normvec = np.std(x, axis=1)
    assert len(normvec.shape)==1
    normvec = np.reshape(normvec, [normvec.shape[0], 1,1]) # (chans, 1, 1) # std for each chan, across (times, trials).
    normmin = np.percentile(normvec, [2.5]) # get the min (std fr across time/conditions) across channels, add this on to still have

    # min fr, to make this a "soft" normalization
    frmean_each_chan = np.mean(x, axis=1) # (chans, 1, 1) # std for each chan, across (times, trials).
    frmin = np.min(frmean_each_chan) # (chans, 1, 1)) # min (mean fr across time/condition) across chans
    # frmin = np.min(np.mean(np.mean(PA.X, axis=1, keepdims=True), axis=2, keepdims=True)) # (chans, 1, 1)) # min (mean fr across time/condition) across chans

    # to further help making this "soft"
    abs_fr_min = 3 # any fr around this low, want to penalize drastically, effectively making it not contyribute much to population activit.

    # DENOM = (normvec+normmin)
    # DENOM = (normvec + normmin + frmin + abs_fr_min) # To further lower infleunce of low FR neurons (started 2/8/24, 2:55pm)
    DENOM = (normvec + frmin + abs_fr_min) # To make more similar across chans. (started 2/11/24)

    # Do normalization.
    PAnorm = PA.copy()
    PAnorm.X = PAnorm.X/DENOM

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
    PAnorm_pre, _ = _popanal_preprocess_normalize(PA)
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


def snippets_extract_popanals_split_bregion_twind(SP, list_time_windows, vars_extract_from_dfscalar,
                                                  SAVEDIR=None, dosave=False,
                                                  combine_into_larger_areas=False,
                                                  events_keep=None,
                                                  exclude_bad_areas=False):
    """ [GOOD] SP --> Multiple Popanals, each with speciifc (event, bregion, twind), and
    with all variables extracted into each pa.Xlabels["trials"]. The goal is that at can
    run all population analyses using these pa, without need for having beh datasets and
    all snippets in memory.
    Extraction of specific PopAnals for each conjunction of (twind, bregion).
    PARAMS:
    - list_time_windowsm, list of timw eindow, tuples .e.g, (-0.2, 0.2), each defining a specific
    extracvted PA.
    - EFFECT_VARS, list of str, vars to extract, mainly to make sure the etracted PA have all
    variables. If not SKIP_ANALY_PLOTTING, then these also determine which plots.
    - dosave, bool, def faulse since takes lots sapce, like 1-3g per wl.
    RETURNS:
    - DictBregionTwindPA, dict, mapping (bregion, twind) --> pa.
    All PAs guaradteeed to have iodentical (:, trials, times).
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # SInce this is population, make sure all channels are present (no outliers removed)
    SP.datamod_append_outliers()

    if events_keep is None or len(events_keep)==0:
        events_keep = SP.Params["list_events_uniqnames"]

    if SAVEDIR is None and dosave:
        from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
        SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS"
        os.makedirs(SAVEDIR, exist_ok=True)

    ####################### EXTRACT DATA
    # list_features_extraction = list(set(list_features_extraction + EFFECT_VARS))
    list_bregion = SP.bregion_list(combine_into_larger_areas=combine_into_larger_areas)

    if not any([e in SP.DfScalar["event"].unique().tolist() for e in events_keep]):
        events_keep = sorted(SP.DfScalar["event"].unique().tolist())

    # 1) Extract population dataras
    DictEvBrTw_to_PA = {}
    print("These events:", events_keep)
    for event in events_keep:
        if event in SP.DfScalar["event"].tolist():
            print(event)
            # assert len(SP.Params["list_events_uniqnames"])==1, "assuming is strokes, just a single event... otherwise iterate"
            # event = SP.Params["list_events_uniqnames"][0]
            PA, _ = SP.dataextract_as_popanal_statespace(SP.Sites, event,
                                                         list_features_extraction=vars_extract_from_dfscalar,
                                                      which_fr_sm = "fr_sm", max_frac_trials_lose=0.02)

            assert len(PA.X)>0
            # print("These are requested sites:", SP.Sites)
            # print("These are extracted sites:", PA.Chans)

            # Split PA based on chans (e.g., bregions), times (e.g., different time slices) BEFORE doing downstream analyses
            DictBregionTwindPA = {}
            trials = None
            xlabels_times = None
            xlabels_trials = None
            for twind in list_time_windows:
                times = None
                for bregion in list_bregion:

                    print(event, bregion, twind)

                    # Bregion
                    chans_needed = SP.sitegetter_map_region_to_sites(bregion, exclude_bad_areas=exclude_bad_areas)
                    print("Sites for this bregion ", bregion)
                    print(chans_needed)
                    if len(chans_needed)>0:
                        pa = PA.slice_by_dim_values_wrapper("chans", chans_needed)
                        # Times
                        pa = pa.slice_by_dim_values_wrapper("times", twind)

                        assert len(pa.X)>0

                        # sanity check that all pa are identical
                        if trials is not None:
                            assert pa.Trials == trials
                        if times is not None:
                            # print(list(pa.Times))
                            # print(list(times))
                            assert list(pa.Times) == list(times)
                        if xlabels_trials is not None:
                            assert pa.Xlabels["trials"].equals(xlabels_trials)
                        if xlabels_times is not None:
                            assert pa.Xlabels["times"].equals(xlabels_times)

                        # # uiseful - a conjucntionv ariable for each tw
                        # from pythonlib.tools.pandastools import append_col_with_grp_index
                        # pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"],
                        #                                                 ["which_level", "event", "twind"],
                        #                                                 "wl_ev_tw",
                        #                                                 use_strings=False)
                        #
                        # Update all
                        trials = pa.Trials
                        times = pa.Times
                        xlabels_trials = pa.Xlabels["trials"]
                        xlabels_times = pa.Xlabels["times"]

                        # DictBregionTwindPA[(bregion, twind)] = pa
                        DictEvBrTw_to_PA[(SP.Params["which_level"], event, bregion, twind)] = pa
                        print(event, " -- ", bregion, " -- ", twind, " -- (data shape:)", pa.X.shape)
                    else:
                        print("Skipping bregion (0 channels): ", bregion)

    assert len(DictEvBrTw_to_PA)>0

    # Save it as dataframe
    tmp = []
    for k, v in DictEvBrTw_to_PA.items():

        # Make sure pa itself is keeping track of the outer varibles,
        # for sanity checks once you start splitting and grouping.
        v.Xlabels["trials"]["which_level"] = k[0]
        v.Xlabels["trials"]["event"] = k[1]
        v.Xlabels["trials"]["bregion"] = k[2]
        v.Xlabels["trials"]["twind"] = [k[3] for _ in range(len(v.Xlabels["trials"]))]

        tmp.append({
            "which_level":k[0],
            "event":k[1],
            "bregion":k[2],
            "twind":k[3],
            "pa":v
        })
    DFallpa = pd.DataFrame(tmp)

    if len(DFallpa)==0:
        print(list_time_windows, vars_extract_from_dfscalar,
              combine_into_larger_areas, events_keep, exclude_bad_areas)
        assert False, "probably params not compatible with each other"

    # # Sanity check
    # for i, row in DFallpa.iterrows():
    #     a = row["twind"]
    #     b = row["pa"].Xlabels["trials"]["twind"].values[0]
    #
    #     if not a==b:
    #         print(a, b)
    #         assert False, "this is old versio before 1/28 -- delete it and regenerate DFallpa"

    # Also note down size of PA, in a column
    list_shape =[]
    for i, row in DFallpa.iterrows():
        list_shape.append(row["pa"].X.shape)
    DFallpa["pa_x_shape"] = list_shape

    ## SAVE
    if dosave:
        import pickle
        mult_sing, sessions = SP.check_if_single_or_mult_session()
        sessions_str = "_".join([str(s) for s in sessions])
        if mult_sing == "mult":
            SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/mult_session"
            path = f"{SAVEDIR}/{SP.animal()}-{SP.date()}-{SP.Params['which_level']}-{sessions_str}.pkl"
        elif mult_sing=="sing":
            SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/single_session"
            path = f"{SAVEDIR}/{SP.animal()}-{SP.date()}-{SP.Params['which_level']}-{sessions_str}.pkl"
        with open(path, "wb") as f:
            pickle.dump(DFallpa, f)
        print("Saved to: ", path)

    return DFallpa

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


def _trajgood_make_colors_discrete_var(labels, which_dim_of_labels_to_use=None):
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

    if which_dim_of_labels_to_use is None:
        labels_for_color = labels
    else:
        labels_for_color = [lab[which_dim_of_labels_to_use] for lab in labels]
    labels_color_uniq = sort_mixed_type(list(set(labels_for_color)))

    if len(set([type(x) for x in labels_color_uniq]))>1:
        # more than one type...
        color_type = "discr"
        pcols = makeColors(len(labels_color_uniq))
        _map_lev_to_color = {}
        for lev, pc in zip(labels_color_uniq, pcols):
            _map_lev_to_color[lev] = pc
    # continuous?
    elif len(labels_color_uniq)>50 and isinstance(labels_color_uniq[0], (int)):
        color_type = "cont"
        # from pythonlib.tools.plottools import map_continuous_var_to_color_range as mcv
        # valmin = min(df[var_color_by])
        # valmax = max(df[var_color_by])
        # def map_continuous_var_to_color_range(vals):
        #     return mcv(vals, valmin, valmax)
        # label_rgbs = map_continuous_var_to_color_range(df[var_color_by])
        _map_lev_to_color = None
    elif len(labels_color_uniq)>8 and isinstance(labels_color_uniq[0], (np.ndarray, float)):
        color_type = "cont"
        _map_lev_to_color = None
    else:
        color_type = "discr"
        # label_rgbs = None
        pcols = makeColors(len(labels_color_uniq))
        _map_lev_to_color = {}
        for lev, pc in zip(labels_color_uniq, pcols):
            _map_lev_to_color[lev] = pc

    # Return the color for each item
    colors = [_map_lev_to_color[lab] for lab in labels_for_color]

    return _map_lev_to_color, color_type, colors

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

        trajgood_plot_colorby_scalar_BASE(xs, ys, labels_color, ax,
                                          map_lev_to_color, color_type,
                                          overlay_mean, plot_text_over_examples,
                                          text_to_plot, alpha, SIZE)

    return fig, axes

def trajgood_plot_colorby_groupby_meanscalar_BASE(ax, xs, ys, dflab, vars_mean, colorby_ind_in_vars_mean):
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
    var_color = vars_mean[colorby_ind_in_vars_mean]
    labels_color_uniq = df[var_color].unique().tolist()
    pcols = makeColors(len(labels_color_uniq))
    map_var_to_col = {}
    for i, v in enumerate(df[var_color].unique()):
        map_var_to_col[v] = pcols[i]
    df["color"] = [map_var_to_col[v] for v in df[var_color]]

    dfmean = df.groupby(vars_mean).mean().reset_index(drop=True)
    for i, row in dfmean.iterrows():
        ax.plot(row["x"], row["y"], "s", color=row["color"], markersize=12)

def trajgood_plot_colorby_scalar_BASE(xs, ys, labels_color, ax,
                                      map_lev_to_color=None, color_type="discr",
                                      overlay_mean=False,
                                      plot_text_over_examples=False, text_to_plot=None,
                                      alpha=0.5, SIZE=5,
                                      connect_means_with_line=False, connect_means_with_line_levels=None):
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

    X = np.concatenate((xs, ys), axis=1)

    # Overwrite with user inputed
    if map_lev_to_color is None:
        map_lev_to_color, color_type, _ = _trajgood_make_colors_discrete_var(labels_color)

    plotScatterOverlay(X, labels_color, alpha=alpha, ax=ax, overlay_mean=overlay_mean,
                       overlay_ci=False,
                       plot_text_over_examples=plot_text_over_examples,
                       text_to_plot=text_to_plot, map_lev_to_color=map_lev_to_color,
                       SIZE=SIZE, color_type=color_type)

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
                Xmeans.append(np.mean(X[labels_color==lev, :], axis=0))
        if len(Xmeans)>0:
            Xmeans = np.stack(Xmeans)
            # Plot
            ax.plot(Xmeans[:,0], Xmeans[:,1], "-")

def trajgood_plot_colorby_splotby_scalar_WRAPPER(X, dflab, var_color, savedir,
                                                 vars_subplot=None, list_dims=None,
                                                 STROKES_BEH=None, STROKES_TASK=None,
                                                 n_min_per_levo=None,
                                                 overlay_mean=False, overlay_mean_var_color=None,
                                                 connect_means_with_line=False, connect_means_with_line_levels=None,
                                                 SIZE=7, alpha=0.5):
    """
    Final wrapper to make many plots, each figure showing supblots one for each levv of otehr var, colored
    by levels of var. Across figures, show different projections to dim pairs. And plot sepraerpte figuers for
    with and without strokes overlaid.

    :param X: scalar data, (ndat, nfeat)
    :param dflab:
    :param var_color:
    :param savedir:
    :param vars_subplot:
    :param list_dims:
    :param STROKES_BEH:
    :param STROKES_TASK:
    :param overlay_mean_var_color, str, which variable (in var_color, if it is list) to use for coloring. This doesnt affect
    the grouping of trails to compute means, which always uses var_color. Controls only how means are collored.
    :return:
    """
    from neuralmonkey.analyses.state_space_good import cleanup_remove_labels_ignore

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

    assert len(X.shape)==2
    assert len(X)==len(dflab)

    if list_dims is None:
        list_dims = [(0,1)]

    var_color_for_name = var_color
    if isinstance(var_color, (tuple, list)):
        dflab = append_col_with_grp_index(dflab, var_color, "_tmp")
        var_color = "_tmp"

    if isinstance(vars_subplot, str):
        vars_subplot = [vars_subplot]

    # One figure for each pair of dims
    for dim1, dim2 in list_dims:
        xs = X[:, dim1]
        ys = X[:, dim2]
        labels_color = dflab[var_color].tolist()
        # text_to_plot = labels_color
        text_to_plot = None

        if vars_subplot is None:
            labels_subplot = None
        else:
            # is a conjunctive var
            dflab = append_col_with_grp_index(dflab, vars_subplot, "_tmp", strings_compact=True)
            labels_subplot = dflab["_tmp"].tolist()
            vars_subplot_string = "|".join(vars_subplot)

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
                                                                                           overlay_mean=False,
                                                                                           text_to_plot=text_to_plot,
                                                                                           skip_subplots_lack_mult_colors=True,
                                                                                           n_min_per_levo=n_min_per_levo,
                                                                                           connect_means_with_line=connect_means_with_line,
                                                                                           connect_means_with_line_levels=connect_means_with_line_levels)

        # Overlay means, including option to use one set of variables for grouping, and a subset of those variables for coloring.
        if overlay_mean:
            for levo, ax in map_levo_to_ax.items():
                inds = map_levo_to_inds[levo]
                xsthis = xs[inds]
                ysthis = ys[inds]
                dflabthis = dflab.loc[inds, var_color_for_name]
                trajgood_plot_colorby_groupby_meanscalar_BASE(ax, xsthis, ysthis, dflabthis, var_color_for_name, colorby_ind_in_vars_mean)

        # Save
        path = f"{savedir}/color={var_color_for_name}-sub={vars_subplot_string}-dims={dim1, dim2}.pdf"
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
                                                                                               skip_subplots_lack_mult_colors=True,
                                                                                               n_min_per_levo=n_min_per_levo)
            if fig is not None:
                path = f"{savedir}/color={var_color_for_name}-sub={vars_subplot_string}-dims={dim1, dim2}-STROKES_OVERLAY.pdf"
                print("Saving ... ", path)
                savefig(fig, path)

        plt.close("all")


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
                                       connect_means_with_line_levels=None
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

    if len(xs.shape)==1:
        xs = xs[:, None]
    if len(ys.shape)==1:
        ys = ys[:, None]

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
    dfthis = pd.DataFrame(tmp)

    fig, axes, map_levo_to_ax, map_levo_to_inds = _trajgood_plot_colorby_splotby_scalar(dfthis, color_var, subplot_var,
                                         overlay_mean, plot_text_over_examples,
                                                text_to_plot, alpha, SIZE,
                                                                                        skip_subplots_lack_mult_colors=skip_subplots_lack_mult_colors,
                                                                                        n_min_per_levo=n_min_per_levo,
                                                                          connect_means_with_line=connect_means_with_line,
                                                                                        connect_means_with_line_levels=connect_means_with_line_levels)

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
                                          connect_means_with_line_levels=None):
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

    labellist = df[var_color_by].tolist()
    map_lev_to_color, color_type, _ = _trajgood_make_colors_discrete_var(labellist)

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

    max_n_subplots = 32
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

    ncols = 4
    nrows = int(np.ceil(len(levs_other)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(ncols*SIZE, nrows*SIZE))
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
        labels_color = dfthis[var_color_by].values
        trajgood_plot_colorby_scalar_BASE(xs, ys, labels_color, ax,
                                          map_lev_to_color, color_type,
                                          overlay_mean, plot_text_over_examples,
                                          text_to_plot_this, alpha, SIZE,
                                          connect_means_with_line=connect_means_with_line,
                                          connect_means_with_line_levels=connect_means_with_line_levels)

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


def trajgood_plot_colorby_splotby(df, var_color_by, var_subplots, dims=(0,1),
                                  traj_or_scalar="traj", mean_over_trials=True,
                                  times_to_mark=None,
                                   times_to_mark_markers=None,
                                   time_bin_size=None,
                                   markersize=6, marker="o",
                                   text_plot_pt1=None,
                                   alpha=0.5,
                                   ntrials=5):
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

    # One color for each level of effect var
    levs_effect = sort_mixed_type(df[var_color_by].unique().tolist())
    pcols = makeColors(len(levs_effect))
    map_lev_to_color = {}
    for lev, pc in zip(levs_effect, pcols):
        map_lev_to_color[lev] = pc

    # One subplot per othervar
    levs_other = sort_mixed_type(df[var_subplots].unique().tolist())
    ncols = 3
    nrows = int(np.ceil(len(levs_other)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))
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
                                       text_plot_pt1=text_plot_pt1, alpha=alpha)
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
                                           text_plot_pt1=text_plot_pt1, alpha=alpha)

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
        print("UMAP, Using this n_neighbors:", umap_n_neighbors, ", nsamp =", nsamp)
        min_dist = 0.1
        reducer = umap.UMAP(n_components=n_components, n_neighbors=umap_n_neighbors, min_dist=min_dist)
        # mapper = reducer.fit(X)
        Xredu = reducer.fit_transform(X)
    else:
        assert False

    return Xredu, reducer


def dimredgood_pca(X, n_components=None,
                   how_decide_npcs_keep = "cumvar",
                   pca_frac_var_keep=0.85, pca_frac_min_keep=0.01,
                   plot_pca_explained_var_path=None, plot_loadings_path=None,
                   plot_loadings_feature_labels=None,
                   method="svd", npcs_keep_force=None):
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
    :return:
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

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (X.shape[0] - 1)
        explained_variance_ratio_ = explained_variance_ / explained_variance_.sum()
        # singular_values_ = S.copy()  # Store the singular values.

        Xpca = np.dot(X, components_.T)

        pca = None
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
        else:
            print(how_decide_npcs_keep)
            assert False
    else:
        # Use the forced N.
        npcs_keep = min([len(explained_variance_ratio_), npcs_keep_force])

    Xpcakeep = Xpca[:, :npcs_keep]

    if plot_pca_explained_var_path is not None:
        fig, axes = plt.subplots(1,2, figsize=(8,3))

        ax = axes.flatten()[0]
        ax.plot(explained_variance_ratio_)
        if how_decide_npcs_keep=="minvar":
            ax.axhline(pca_frac_min_keep, color="g", label="pca_frac_min_keep")
        ax.axvline(npcs_keep, color="r", label="npcs_keep")
        ax.set_title("frac var, each dim")
        ax.legend()

        ax = axes.flatten()[1]
        ax.plot(np.cumsum(explained_variance_ratio_))
        if how_decide_npcs_keep=="cumvar":
            ax.axhline(pca_frac_var_keep, color="g", label="pca_frac_var_keep")
        ax.axvline(npcs_keep, color="r", label="npcs_keep")
        ax.set_title("frac var, each dim")
        ax.set_title("cumulative var, each dim")
        ax.legend()

        savefig(fig, plot_pca_explained_var_path)

    if plot_loadings_path is not None:
        from pythonlib.tools.snstools import heatmap_mat
        fig, ax = plt.subplots(figsize=(20, 15))
        heatmap_mat(components_, ax, diverge=True, annotate_heatmap=False, labels_col=plot_loadings_feature_labels) # (n_components, n_features)
        ax.set_ylabel("pcs")
        ax.set_xlabel("features (chans x twind)")
        savefig(fig, plot_loadings_path)

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
