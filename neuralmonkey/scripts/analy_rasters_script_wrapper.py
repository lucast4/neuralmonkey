"""
3/3/24 - General purpose, to plot any rasters, for set of var and other var,
wrapping all loading and preprocessing.

THis is also a good template for basic Snippets loading and preprocessing.

This uses pre-extracted Snippets, and plots entire time window.

"""

import os
import sys

from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
from neuralmonkey.classes.session import load_mult_session_helper
from neuralmonkey.classes.population_mult import snippets_extract_popanals_split_bregion_twind
# from neuralmonkey.analyses.rsa import rsagood_questions_dict
from pythonlib.tools.plottools import savefig
from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, extract_with_levels_of_conjunction_vars
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
import matplotlib.pyplot as plt

DEBUG = False

# -- Comparing samp response for char vs. pig vs. sp (need same shapes).
# OVERWRITE_n_min = 4 # n min samp per level-otherlev conjunction
# OVERWRITE_lenient_n = 1
# balance_same_levels_across_ovar = True # Then prunes levs of var so that all levels of ovar have same levels of var.

# -- Generic
OVERWRITE_n_min = 5 # n min samp per level-otherlev conjunction
OVERWRITE_lenient_n = 2
balance_same_levels_across_ovar = False # Then prunes levs of var so that all levels of ovar have same levels of var.

FR_SM_STD = 0.04

# SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
prune_low_fr_sites = True
prune_low_fr_sites_thresh = 1.5

MULTIPROCESS = False # Parallel processing of all the vars to plot.
MULTIPROCESS_N_CORES = 2

def get_events_keep(which_level):
    if which_level=="trial":
        EVENTS_KEEP = ['03_samp', '04_go_cue', '05_first_raise', '06_on_strokeidx_0']
        # EVENTS_KEEP = ['03_samp', '04_go_cue', '06_on_strokeidx_0']
        # EVENTS_KEEP = ['03_samp']
    elif which_level=="stroke":
        EVENTS_KEEP = ["00_stroke"]
    else:
        print(which_level)
        assert False
    return EVENTS_KEEP

def plotter(SP, var, vars_others, event, SAVEDIR, OVERWRITE_n_min, OVERWRITE_lenient_n):
    savedir = f"{SAVEDIR}/var={var}-others={'|'.join(vars_others)}/{event}"
    os.makedirs(savedir, exist_ok=True)

    # Plot and save conjunctions
    dfthis = SP.DfScalar[
        (SP.DfScalar["chan"] == SP.Sites[0]) &
        (SP.DfScalar["event"] == event)
        ].reset_index(drop=True)
    extract_with_levels_of_conjunction_vars(dfthis, var, vars_others, n_min_across_all_levs_var=0, lenient_allow_data_if_has_n_levels=0,
                                            PRINT_AND_SAVE_TO=f"{savedir}/00_counts_conjunction.txt",
                                            plot_counts_heatmap_savepath=f"{savedir}/00_counts_conjunction_heatmap.pdf")
    extract_with_levels_of_conjunction_vars(dfthis, var, vars_others, n_min_across_all_levs_var=OVERWRITE_n_min,
                                            lenient_allow_data_if_has_n_levels=OVERWRITE_lenient_n,
                                            PRINT_AND_SAVE_TO=f"{savedir}/00_counts_conjunction_GOOD.txt",
                                            plot_counts_heatmap_savepath=f"{savedir}/00_counts_conjunction_heatmap_GOOD.pdf")

    # Rasters
    for chan in SP.Sites:
        chan_text = SP.session_sitegetter_summarytext(chan)

        # # IGNORE - just holding repo of other plot kinds.
        # if False:
        #     # SKIP THESE - becuase shape can be very different across index_within_shape (even for PMv)
        #     # and so these are not informative. Also to speed things up.
        #
        #     # M1 with similar encoding for the same ss, no matter the shape or index
        #     var = "shape_idxwithin"
        #     vars_other = ["dist_angle"]
        #     fig, axesall = SP.plotgood_rasters_smfr_each_level_combined(chan, var, vars_other, plotvers=("smfr"));
        #     savefig(fig, f"{savedir}/{chan_text}-shape_idx-vs-substrk_dist_angle.png")
        #
        #     # [Same, but splitting into grid plot]
        #     var = "shape_idxwithin"
        #     vars_other = ["distcum_binned", "angle_binned"]
        #     fig, axesall = SP.plotgood_smfr_each_level_subplot_grid_by_vars(chan, var, vars_other[0], vars_other[1], PLOT_VER="smfr");
        #     savefig(fig, f"{savedir}/{chan_text}-shape_idx-vs-substrk_dist_angle_grid.png")
        #
        #     # LIST_PLOT_VER = ["smfr", "raster"] # to speed it up, exclude raster
        #     LIST_PLOT_VER = ["smfr"]
        #     for PLOT_VER in LIST_PLOT_VER:
        #         fig, _ = SP.plotgood_smfr_each_level_subplot_grid_by_vars(chan, "shape_oriented",
        #                                                          "gridloc", "nstk_stkidx",
        #                                                          PLOT_VER=PLOT_VER)
        #         path = f"{savedir}/{chan_text}-shape_oriented-vs-other_vars-{PLOT_VER}.png"
        #         savefig(fig, path)
        #

        plotvers = ("raster", "smfr")
        # Var vs. Vars_others (rasters and smoothed).
        # path = f"{savedir}/{chan_text}.png"
        path = f"{savedir}/{chan_text}.pdf"
        if not os.path.exists(path):
            fig, axesall = SP.plotgood_rasters_smfr_each_level_combined(chan, var, vars_others,
                                                                        event=event,
                                                                        plotvers=plotvers,
                                                                        OVERWRITE_n_min=OVERWRITE_n_min,
                                                                        OVERWRITE_lenient_n=OVERWRITE_lenient_n,
                                                                        balance_same_levels_across_ovar=balance_same_levels_across_ovar)
            if fig is not None:
                savefig(fig, path)
                print("Saved rasters to: ", path)
                plt.close("all")
            else:
                # THen  not neough data.. just skip, assuming this is true for all sites.
                print("No data found for this var/var_others: skippig it entirely.")
                break
        else:
            print("Skipped rasters (already exists): ", path)


if __name__=="__main__":

    ############# USER PARAMS
    animal = sys.argv[1]
    date = int(sys.argv[2])
    question = sys.argv[3]

    ############# USUALYL HARD-CODED PARAMS
    # Load q_params
    q_params = rsagood_questions_dict(animal, date, question)[question]
    assert len(q_params["list_which_level"])==1, "simplify it..."
    which_level = q_params["list_which_level"][0]

    ############# MODIFY THESE PARAMS
    from neuralmonkey.metadat.analy.anova_params import params_getter_raster_vars
    LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n = params_getter_raster_vars(which_level, question, OVERWRITE_lenient_n)

    # combine_trial_and_stroke = False
    EVENTS_KEEP = get_events_keep(which_level)

    ############### PARAMS
    # exclude_bad_areas = True
    HACK_RENAME_SHAPES = "CHAR" in question

    SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/RASTERS/{animal}-{date}/{question}-{which_level}"
    os.makedirs(SAVEDIR, exist_ok=True)

    ################## LOAD DATA
    MS = load_mult_session_helper(date, animal, spikes_version=SPIKES_VERSION, fr_sm_std=FR_SM_STD)
    SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level, events_keep=EVENTS_KEEP,
        DEBUG=DEBUG)

    # Prune low FR?
    if prune_low_fr_sites:
        SP.prune_low_firing_rate_sites(prune_low_fr_sites_thresh)

    # Clean up SP and extract features
    D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
        ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"],
        substrokes_plot_preprocess=False)

    ################# CLEAN JUST TO EVENTS OF INTEREST (fewer plots)
    if EVENTS_KEEP is not None:
        SP.Params["list_events_uniqnames"] = EVENTS_KEEP
        SP.DfScalar = SP.DfScalar[SP.DfScalar["event"].isin(EVENTS_KEEP)].reset_index(drop=True)

    ############## PLOTS
    if MULTIPROCESS:
        for event in EVENTS_KEEP:
            print("Starting multipporcss for event", event)
            LIST_SP = [SP for _ in range(len(LIST_VAR))]
            LIST_OVERWRITE_n_min = [OVERWRITE_n_min for _ in range(len(LIST_VAR))]
            LIST_EVENT = [event for _ in range(len(LIST_VAR))]
            LIST_SAVEDIR = [SAVEDIR for _ in range(len(LIST_VAR))]
            from multiprocessing import Pool
            with Pool(MULTIPROCESS_N_CORES) as pool:
                pool.starmap(plotter, zip(LIST_SP, LIST_VAR, LIST_VARS_OTHERS, LIST_EVENT, LIST_SAVEDIR, LIST_OVERWRITE_n_min, LIST_OVERWRITE_lenient_n))
    else:
        for var, vars_others, OVERWRITE_lenient_n in zip(LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n):
            for event in EVENTS_KEEP:
                plotter(SP, var, vars_others, event, SAVEDIR, OVERWRITE_n_min, OVERWRITE_lenient_n)