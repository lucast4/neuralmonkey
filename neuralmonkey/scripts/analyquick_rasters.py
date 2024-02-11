"""
2/8/24 - Quickly written for substrokes analysis
- comparing substroke to stroke_shape.
"""

from neuralmonkey.classes.session import load_mult_session_helper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pythonlib.tools.pandastools import convert_to_2d_dataframe, grouping_plot_n_samples_conjunction_heatmap
from neuralmonkey.classes.snippets import Snippets, extraction_helper
from pythonlib.tools.plottools import savefig
import os
from neuralmonkey.classes.snippets import load_and_concat_mult_snippets

# dict_dates = {
#     "Diego":["230628", "230630"],
#     "Pancho":["230623", "230626"],
# }

import sys
spikes_version = "tdt"

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])

    # which_level = sys.argv[3]
    # ANALY_VER = sys.argv[4]
    #
    # dict_dates = {
    #     # "Pancho":["230623"],
    #     # "Diego":["230630"],
    #     "Pancho":["230626"],
    #     "Diego":["230628"],
    # }
    #
    # ONSET_OR_OFFSET = "offset"
    # # ONSET_OR_OFFSET = "onset"
    #
    # ############################
    # if ONSET_OR_OFFSET=="onset":
    #     list_pre_dur = [-0.6]
    #     list_post_dur = [1.0]
    #     which_level = "stroke"
    #     SAVEDIR = "/gorilla1/analyses/recordings/main/shape_vs_loc_index_PIG_BETTER_ONSET"
    # elif ONSET_OR_OFFSET=="offset":
    #     list_pre_dur = [-1.0]
    #     list_post_dur = [0.6]
    #     which_level = "stroke_off"
    #     SAVEDIR = "/gorilla1/analyses/recordings/main/shape_vs_loc_index_PIG_BETTER_OFFSET"
    # else:
    #     assert False

    savedir = f"/gorilla1/analyses/recordings/main/RASTERS/{animal}-{date}/substrokes"
    import os
    os.makedirs(savedir, exist_ok=True)


    MS = load_mult_session_helper(date, animal, MINIMAL_LOADING=True, spikes_version=spikes_version)

    which_level = "substroke"
    SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level,
        DEBUG=False)

    ##### SUBSTROKES ANALY
    SP.datasetbeh_append_column_helper(["shape_idxwithin"])

    from pythonlib.tools.plottools import savefig

    for chan in SP.Sites:

        chan_text = SP.session_sitegetter_summarytext(chan)

        print(chan_text)
        if False:
            # SKIP THESE - becuase shape can be very different across index_within_shape (even for PMv)
            # and so these are not informative. Also to speed things up.

            # M1 with similar encoding for the same ss, no matter the shape or index
            var = "shape_idxwithin"
            vars_other = ["dist_angle"]
            fig, axesall = SP.plotgood_rasters_smfr_each_level_combined(chan, var, vars_other, plotvers=("smfr"));
            savefig(fig, f"{savedir}/{chan_text}-shape_idx-vs-substrk_dist_angle.png")

            # [Same, but splitting into grid plot]
            var = "shape_idxwithin"
            vars_other = ["distcum_binned", "angle_binned"]
            fig, axesall = SP.plotgood_smfr_each_level_subplot_grid_by_vars(chan, var, vars_other[0], vars_other[1], PLOT_VER="smfr");
            savefig(fig, f"{savedir}/{chan_text}-shape_idx-vs-substrk_dist_angle_grid.png")

        # Pmv differetn across shapes
        var = "shape"
        vars_other = ["index_within_stroke", "dist_angle"]
        fig, axesall = SP.plotgood_rasters_smfr_each_level_combined(chan, var, vars_other, plotvers=("smfr"));
        savefig(fig, f"{savedir}/{chan_text}-shape-vs-substrk_idx_dist_angle.png")


        # M1 different across ss, no matter the shape or index
        # Trial by trial variability
        var = "dist_angle"
        vars_other = ["shape_idxwithin"]
        fig, axesall = SP.plotgood_rasters_smfr_each_level_combined(chan, var, vars_other, plotvers=("smfr"));
        savefig(fig, f"{savedir}/{chan_text}-substrk_dist_angle-vs-shape_idx.png")

        plt.close("all")

