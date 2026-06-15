"""
2/3/24 - quick plotting of rasters and sm fr for shape vs. (gridloc, strokeindex)
for PIG dates. Related to RSA analysis. Aligned to strokes (uses Snippets).
Can be made flexible to work with other variables too.

OBSOLETE: use scripts/analy_rasters_script_wrapper.py instead
"""

from neuralmonkey.classes.session import load_mult_session_helper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pythonlib.tools.pandastools import convert_to_2d_dataframe, grouping_plot_n_samples_conjunction_heatmap
from neuralmonkey.classes.snippets import Snippets, extraction_helper
from pythonlib.tools.plottools import savefig
import os

# dict_dates = {
#     "Diego":["230628", "230630"],
#     "Pancho":["230623", "230626"],
# }

assert False, "must add event as argument to plotgood_rasters_smfr_each_level_combined"

if __name__=="__main__":
    dict_dates = {
        # "Pancho":["230623"],
        # "Diego":["230630"],
        "Pancho":["230626"],
        "Diego":["230628"],
    }

    ONSET_OR_OFFSET = "offset"
    # ONSET_OR_OFFSET = "onset"

    ############################
    if ONSET_OR_OFFSET=="onset":
        list_pre_dur = [-0.6]
        list_post_dur = [1.0]
        which_level = "stroke"
        SAVEDIR = "/gorilla1/analyses/recordings/main/shape_vs_loc_index_PIG_BETTER_ONSET"
    elif ONSET_OR_OFFSET=="offset":
        list_pre_dur = [-1.0]
        list_post_dur = [0.6]
        which_level = "stroke_off"
        SAVEDIR = "/gorilla1/analyses/recordings/main/shape_vs_loc_index_PIG_BETTER_OFFSET"
    else:
        assert False

    for animal, dates in dict_dates.items():
        for DATE in dates:
            # DATE = 230623
            # animal = "Diego"
            MS = load_mult_session_helper(DATE, animal, MINIMAL_LOADING=True, spikes_version="kilosort_if_exists")

            for i_sn, sn in enumerate(MS.SessionsList):

                savedir = f"{SAVEDIR}/{animal}/{DATE}/sess_{i_sn}"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)


                # == IGNORE THESE:
                strokes_only_keep_single = False # if True, then prunes dataset, removing trials "remove_if_multiple_behstrokes_per_taskstroke"
                prune_feature_levels_min_n_trials = 1
                list_events = [] # must be empty
                list_features_extraction = []
                list_features_get_conjunction = []
                trials_prune_just_those_including_events=False

                SP = Snippets(sn,
                    which_level,
                    list_events,
                    list_features_extraction,
                    list_features_get_conjunction,
                    list_pre_dur,
                    list_post_dur,
                    strokes_only_keep_single=strokes_only_keep_single,
                    tasks_only_keep_these=None,
                    prune_feature_levels_min_n_trials=prune_feature_levels_min_n_trials,
                    dataset_pruned_for_trial_analysis=None,
                    trials_prune_just_those_including_events=trials_prune_just_those_including_events,
                    fr_which_version='sqrt',
                    NEW_VERSION=True,
                    SKIP_DATA_EXTRACTION=False
                             )

                # GEt conjuction of (n strokes in task) and (stroke index)
                D = SP.datasetbeh_extract_dataset()
                D.extract_beh_features(["num_strokes_task"])
                SP.datasetbeh_append_column("FEAT_num_strokes_task", D)
                # Conjunction of stroke index and num strokes in task.
                from pythonlib.tools.pandastools import append_col_with_grp_index
                SP.DfScalar = append_col_with_grp_index(SP.DfScalar, ["FEAT_num_strokes_task", "stroke_index"], "nstk_stkidx", False)

                ## PLOT
                fig = grouping_plot_n_samples_conjunction_heatmap(SP.DfScalar, "shape_oriented", "gridloc", ["nstk_stkidx"])
                path = f"{savedir}/heatmap_counts_conjunction.pdf"
                savefig(fig, path)

                for chan in SP.Sites:
                    print("plotting, chan:", chan)

                    chan_text = SP.SN.sitegetter_summarytext(chan)

                    fig, _ = SP.plotgood_rasters_smfr_each_level_combined(chan, var="shape_oriented", vars_others=["gridloc"])
                    path = f"{savedir}/{chan_text}-shape_oriented-vs-gridloc.png"
                    savefig(fig, path)

                    fig, _ = SP.plotgood_rasters_smfr_each_level_combined(chan, var="shape_oriented", vars_others=["nstk_stkidx"])
                    path = f"{savedir}/{chan_text}-shape_oriented-vs-nstk_stkidx.png"
                    savefig(fig, path)

                    # LIST_PLOT_VER = ["smfr", "raster"] # to speed it up, exclude raster
                    LIST_PLOT_VER = ["smfr"]
                    for PLOT_VER in LIST_PLOT_VER:
                        fig, _ = SP.plotgood_smfr_each_level_subplot_grid_by_vars(chan, "shape_oriented",
                                                                         "gridloc", "nstk_stkidx",
                                                                         PLOT_VER=PLOT_VER)
                        path = f"{savedir}/{chan_text}-shape_oriented-vs-other_vars-{PLOT_VER}.png"
                        savefig(fig, path)

                    plt.close("all")

