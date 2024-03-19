"""
Good general-purpose script for scoring dsitance between multivar distribitions in state space,
asking about modulation at population level.

Idea is this is a graded version of decode stuff, where can see more quantitrative and small differences due to
particular variables.

Focus is on effect conditioned on othervar (i.e,, within-context).

If consider across-context, then is basicalyl same as the RSA stuff. (NOTE: already uses Cluster code written for
RSA).

Goal: to quantitatively capture intuition as can see in UMA plots.

3/14/24 - Builds on "decode" code
Notebook: Currently at end of 240127_snippets_decode_all

"""

from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
import sys
import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import extract_single_pa
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels

DEBUG = False

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])

    # DONT COMBINE, use questions.
    question = sys.argv[3]
    combine_trial_and_stroke = False
    which_level = sys.argv[4]
    dir_suffix = question

    # Load q_params
    from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
    q_params = rsagood_questions_dict(animal, date, question)[question]
    if which_level=="trial":
        events_keep = ["03_samp", "04_go_cue", "06_on_strokeidx_0"]
    elif which_level=="stroke":
        events_keep = ["00_stroke"]
    else:
        assert False

    HACK_RENAME_SHAPES = "CHAR" in question

    ############### PARAMS
    exclude_bad_areas = True
    SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
    combine_into_larger_areas = False
    list_time_windows = [(-0.6, 0.6)]
    EVENTS_IGNORE = [] # To reduce plots

    if question=="SP_novel_shape":
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=True
    else:
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=False

    ############### PARAMS (EUCLIDIAN)
    PLOT = False
    PLOT_MASKS = False
    N_MIN_TRIALS = 6
    LIST_TWIND = [
        # (-0.2, 0.2),
        # (0.05, 0.35),
        # (0, 0.2),
        (-0.1, 0.1),
    ]
    nmin_trials_per_lev = 5
    PLOT_STATE_SPACE = True
    # NPCS_KEEP = None # use auto method

    # var_loc_next = "CTXT_loc_next"
    var_loc_next = "CTXT_loconclust_next"
    var_loc_prev = "CTXT_locoffclust_prev"
    var_loc = "loc_on_clust"

    LIST_VAR = [
        "CTXT_loc_next",
        "CTXT_loc_next",
        "CTXT_loc_next",

        "CTXT_shape_next",
        "CTXT_shape_next",
        "CTXT_shape_next",

        "task_kind",

        "stroke_index",
        "stroke_index_fromlast_tskstks",
        "stroke_index_fromlast_tskstks",

        "FEAT_num_strokes_task",
        "FEAT_num_strokes_task",

        "shape",
        "shape",
        "shape",

        # "shape",
        "gridloc",
        var_loc,
        var_loc,
    ]
    # More restrictive
    LIST_VARS_OTHERS = [
        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc, "CTXT_shape_next"],
        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", "gridloc", "CTXT_shape_next"],
        ["stroke_index_is_first", "task_kind", "CTXT_loc_prev", "shape", "gridloc", "CTXT_shape_next"],

        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc, "CTXT_loc_next"],
        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", "gridloc", "CTXT_loc_next"],
        ["stroke_index_is_first", "task_kind", "CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next"],

        ["stroke_index_is_first", "shape", var_loc, var_loc_prev],

        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc],
        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc],
        ["stroke_index_is_first", "FEAT_num_strokes_task", "task_kind", var_loc_prev, "shape", var_loc],

        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc, "stroke_index"],
        ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc],

        ["stroke_index_is_first", "task_kind", var_loc, var_loc_prev, "CTXT_loc_next"],
        ["stroke_index_is_first", "task_kind", var_loc, var_loc_prev],
        ["stroke_index_is_first", "task_kind", "gridloc", var_loc_prev],

        ["stroke_index_is_first", "task_kind", "shape", var_loc_prev, "CTXT_loc_next"],
        ["stroke_index_is_first", "task_kind", "shape", var_loc_prev, "CTXT_loc_next"],
        ["stroke_index_is_first", "task_kind", "shape", var_loc_prev],
        ]

    assert len(LIST_VAR)==len(LIST_VARS_OTHERS)

    ########################################## EXTRACT NOT-NORMALIZED DATA
    from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
    DFALLPA = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows, which_level=which_level,
                                              events_keep=events_keep,
                                              combine_into_larger_areas=combine_into_larger_areas,
                                              exclude_bad_areas=exclude_bad_areas, SPIKES_VERSION=SPIKES_VERSION,
                                              HACK_RENAME_SHAPES=HACK_RENAME_SHAPES, fr_normalization_method=None,
                                              path_to_save_example_fr_normalization=None)

    ######## STANDARD PREPROCESING.
    from neuralmonkey.classes.population_mult import dfallpa_preprocess_vars_conjunctions_extract
    dfallpa_preprocess_vars_conjunctions_extract(DFALLPA, which_level=which_level)

    # for fr_normalization_method in ["across_time_bins", "each_time_bin"]:
    #     if fr_normalization_method == "across_time_bins":
    #         LIST_TBIN_DUR = [0.1, None] # None means no binning
    #     else:
    #         # if tbin is None, then same for both fr norm methods... so just do it once above.
    #         LIST_TBIN_DUR = [0.1] # None means no binning

    # for NPCS_KEEP in [5, 12]:
    for NPCS_KEEP in [10]:
        for fr_normalization_method in ["across_time_bins"]:
            LIST_TBIN_DUR = [0.1] # None means no binning

            # Make a copy of all PA before normalization
            DFallpa = DFALLPA.copy()
            list_pa = [pa.copy() for pa in DFallpa["pa"]]
            DFallpa["pa"] = list_pa

            ###########
            SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_DIST/{animal}-{date}/{dir_suffix}-fr_normalization_method={fr_normalization_method}-NPCS_KEEP={NPCS_KEEP}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            print("SAVING AT:", SAVEDIR_ANALYSIS)

            ############################ SAVE PARAMS
            writeDictToTxt({
                "events_keep":events_keep,
                "question":question,
                "q_params":q_params,
                "combine_trial_and_stroke":combine_trial_and_stroke,
                "which_level":which_level,
                "HACK_RENAME_SHAPES":HACK_RENAME_SHAPES,
                "exclude_bad_areas":exclude_bad_areas,
                "SPIKES_VERSION":SPIKES_VERSION,
                "combine_into_larger_areas":combine_into_larger_areas,
                "list_time_windows":list_time_windows,
                "fr_normalization_method":fr_normalization_method,
                "nmin_trials_per_lev":nmin_trials_per_lev,
                },
                f"{SAVEDIR_ANALYSIS}/params.txt")

            #################### Normalize PA firing rates if needed
            path_to_save_example_fr_normalization = f"{SAVEDIR_ANALYSIS}/example_fr_normalization.png"
            if fr_normalization_method is not None:
                if fr_normalization_method=="each_time_bin":
                    # Then demean in each time bin indepednently
                    subtract_mean_at_each_timepoint = True
                    subtract_mean_across_time_and_trial = False
                elif fr_normalization_method=="across_time_bins":
                    # ALl time bins subtract the same scalar --> maintains temporal moudlation.
                    subtract_mean_at_each_timepoint = False
                    subtract_mean_across_time_and_trial = True
                else:
                    print(fr_normalization_method)
                    assert False

                from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization
                list_panorm = []

                for i, pa in enumerate(DFallpa["pa"].tolist()):
                    if path_to_save_example_fr_normalization is not None and i==0:
                        plot_example_chan_number = pa.Chans[0]
                        if which_level=="trial":
                            plot_example_split_var_string = "seqc_0_shape"
                        elif which_level=="stroke":
                            plot_example_split_var_string = "shape"
                        else:
                            plot_example_split_var_string = q_params["effect_vars"][0]
                    else:
                        plot_example_chan_number = None
                    PAnorm, PAscal, PAscalagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(pa, None,
                                                                                                      DO_AGG_TRIALS=False,
                                                                                                      plot_example_chan_number=plot_example_chan_number,
                                                                                                        plot_example_split_var_string = plot_example_split_var_string,
                                                                                                      subtract_mean_at_each_timepoint=subtract_mean_at_each_timepoint,
                                                                                                      subtract_mean_across_time_and_trial=subtract_mean_across_time_and_trial)
                    if path_to_save_example_fr_normalization is not None and i==0:
                        savefig(fig, path_to_save_example_fr_normalization)
                    list_panorm.append(PAnorm)
                DFallpa["pa"] = list_panorm

            ########################################
            for twind_analy in LIST_TWIND:
                for tbin_dur in LIST_TBIN_DUR:
                    tbin_slice = tbin_dur

                    SAVEDIR = f"{SAVEDIR_ANALYSIS}/twind_analy={twind_analy}-tbin_dur={tbin_dur}"
                    os.makedirs(SAVEDIR, exist_ok=True)
                    print("THIS TBIN_DUR", SAVEDIR)

                    #### COLLECT DATA
                    from neuralmonkey.analyses.decode_good import euclidian_distance_compute
                    list_dfres = []
                    for i, row in DFallpa.iterrows():
                        br = row["bregion"]
                        tw = row["twind"]
                        ev = row["event"]
                        PA = row["pa"]

                        print("bregion, twind_analy, tbin_dur: ", br, twind_analy, tbin_dur)

                        savedir = f"{SAVEDIR}/each_region/{br}-twind_analy={twind_analy}"
                        os.makedirs(savedir, exist_ok=True)
                        dfres = euclidian_distance_compute(PA, LIST_VAR, LIST_VARS_OTHERS, PLOT, PLOT_MASKS,
                                                   twind_analy, tbin_dur, tbin_slice, savedir, PLOT_STATE_SPACE=PLOT_STATE_SPACE,
                                                           nmin_trials_per_lev=nmin_trials_per_lev,
                                                           NPCS_KEEP=NPCS_KEEP)
                        plt.close("all")
                        if len(dfres)>0:
                            dfres["bregion"] = br
                            dfres["twind"] = [tw for _ in range(len(dfres))]
                            dfres["twind_analy"] = [twind_analy for _ in range(len(dfres))]
                            dfres["event"] = ev

                            list_dfres.append(dfres)

                    DFRES = pd.concat(list_dfres).reset_index(drop=True)

                    # Compute normalized distnaces
                    from pythonlib.tools.pandastools import append_col_with_grp_index
                    DFRES["dist_norm_95"] = DFRES["dist"]/DFRES["DIST_NULL_95"]
                    DFRES["dist_norm_50"] = DFRES["dist"]/DFRES["DIST_NULL_50"]
                    DFRES["var_others"] = [tuple(x) for x in DFRES["var_others"]]
                    DFRES = append_col_with_grp_index(DFRES, ["var", "var_others"], "var_var_others")
                    DFRES = append_col_with_grp_index(DFRES, ["effect_samediff", "context_samediff"], "effect_context")

                    # SAVE
                    path = f"{SAVEDIR}/DFRES.pkl"
                    pd.to_pickle(DFRES, path)
                    print("Saved to: ", path)

                    ######################################### QUICK PLOT - SUMMARIES
                    import seaborn as sns
                    from pythonlib.tools.snstools import rotateLabel
                    from pythonlib.tools.plottools import savefig
                    from pythonlib.tools.pandastools import summarize_featurediff

                    savedir = f"{SAVEDIR}/FIGURES"
                    os.makedirs(savedir, exist_ok=True)
                    DFTHIS = DFRES

                    ########## OVERVIEWS
                    yvar = "dist_norm_95"
                    for yvarthis in [yvar, "dist", "DIST_NULL_95"]:
                        fig = sns.catplot(data=DFTHIS, x="bregion", y=yvarthis, col="var_var_others", hue="effect_context",
                                          col_wrap=3, aspect=1.5, alpha=0.4)
                        rotateLabel(fig)
                        savefig(fig, f"{savedir}/overview_scatter-{yvarthis}.pdf")

                        fig = sns.catplot(data=DFTHIS, x="bregion", y=yvarthis, col="var_var_others", hue="effect_context",
                                          col_wrap=3, aspect=1.5, kind="bar")
                        rotateLabel(fig)
                        savefig(fig, f"{savedir}/overview_bar-{yvarthis}.pdf")

                        plt.close("all")

                    ########### PLOT ALL specific conjunction levels in heatmaps
                    from pythonlib.tools.pandastools import plot_subplots_heatmap
                    print("Plotting specific conjucntions heatmaps ... ")
                    yvar = "dist"
                    list_effect_context = DFTHIS["effect_context"].unique()
                    list_shuffled = DFTHIS["shuffled"].unique()
                    for effect_context in list_effect_context:
                        for shuffled in list_shuffled:

                            dfthis = DFTHIS[(DFTHIS["effect_context"]==effect_context) & (DFTHIS["shuffled"]==shuffled)].reset_index(drop=True)
                            savedirthis = f"{savedir}/each_conjunction-effect_context={effect_context}-shuffled={shuffled}"
                            os.makedirs(savedirthis, exist_ok=True)
                            print("... ", savedirthis)

                            # 1) Scatter
                            list_vvo = dfthis["var_var_others"].unique().tolist()
                            for vvo in list_vvo:
                                dfthisthis = dfthis[dfthis["var_var_others"]==vvo]
                                fig = sns.catplot(data=dfthisthis, x=yvar, y="levo", col="bregion", alpha=0.4)
                                savefig(fig, f"{savedirthis}/allconj_scatter-vvo={vvo}.pdf")
                                plt.close("all")

                            # 2) Heatmap
                            fig, axes = plot_subplots_heatmap(dfthis, "bregion", "levo", yvar, "var_var_others",
                                                              diverge=True, ncols=None, share_zlim=True)
                            savefig(fig, f"{savedirthis}/allconj_heatmap.pdf")

                    ########## NORMALIZE DISTANCES BETWEEN DIFF (TO DISTANCE BETWEEN SAME)
                    # Normalize by same
                    try:
                        yvar = "dist_norm_95"
                        dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF, dfpivot = summarize_featurediff(DFTHIS,
                                                                   "effect_context", ["diff", "same_pt_pairs"],
                                                                   ["dist_norm_95"],
                                                           ["var_var_others", "var", "var_others", "levo", "bregion", "twind", "event"], return_dfpivot=True)
                        dfpivot["diff_minus_same"] = dfpivot[yvar]["diff"]/(dfpivot[yvar]["diff"] + dfpivot[yvar]["same_pt_pairs"])
                        dfpivot["diff_divide_same"] = dfpivot[yvar]["diff"]/dfpivot[yvar]["same_pt_pairs"]

                        for yvarthis in ["diff_minus_same", "diff_divide_same"]:
                            fig = sns.catplot(data=dfpivot, x="bregion", y=yvarthis, col="var_var_others", col_wrap=3, aspect=1.5, alpha=0.4)
                            rotateLabel(fig)
                            savefig(fig, f"{savedir}/overview_scatter-{yvarthis}.pdf")

                            fig = sns.catplot(data=dfpivot, x="bregion", y=yvarthis, col="var_var_others", col_wrap=3, aspect=1.5, kind="bar")
                            rotateLabel(fig)
                            savefig(fig, f"{savedir}/overview_bar-{yvarthis}.pdf")

                            plt.close("all")
                    except Exception as err:
                        print("SKIPPING plots of normalized distnaces -- this err")
                        print(err)
                        pass

                    print("******** DONE!")