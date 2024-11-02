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
Notebook: Currently in 240217_snippets_euclidian

"""

from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
import sys
import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import extract_single_pa, load_handsaved_wrapper
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels
from pythonlib.tools.pandastools import append_col_with_grp_index
from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper



def _get_list_twind_by_animal(animal, event, trajectories_method, HACK_TRAJS_JUST_FOR_PLOTTING_NICE=False):
    if trajectories_method=="scalar" and event=="00_stroke":
        if animal=="Diego":
            # LIST_TWIND = [
            #     # (-0.2, 0.2),
            #     # (0.05, 0.35),
            #     # (0, 0.2),
            #     # (-0.3, -0.1),
            #     (-0.1, 0.1),
            #     # (0.1, 0.3),
            # ]
            LIST_TWIND = [
                (-0.1, 0.2),
            ]
        elif animal=="Pancho":
            # Increaingly better compared to (-0.1, 0.1), tested up to (0.1, 0.3).
            LIST_TWIND = [
                (0.05, 0.25),
            ]
        else:
            print(animal)
            assert False
    elif trajectories_method=="scalar" and event=="03_samp":
        LIST_TWIND = [
            (0.1, 0.5),
        ]
    elif trajectories_method=="scalar" and event=="04_go_cue":
        LIST_TWIND = [
            (-0.55, -0.05), # GOOD
        ]
    elif trajectories_method=="scalar" and event=="05_first_raise":
        LIST_TWIND = [
            (-0.25, 0.2), # GOOD
        ]
    elif trajectories_method=="traj_to_scalar" and event=="03_samp":
        if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
            LIST_TWIND = [
                (-0.3, 0.5),
            ]         
        else:
            LIST_TWIND = [
                # (-0.05, 0.45),
                # (0.1, 0.5),
                (0.1, 1.0), # 7/24/24 - this should be better
            ]
    elif trajectories_method=="traj_to_scalar" and event in ["00_stroke", "06_on_strokeidx_0"]:
        if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
            LIST_TWIND = [
                (-0.3, 0.5),
            ]
        else:
            if animal=="Diego":
                LIST_TWIND = [
                    # (-0.25, 0.1), # TESTING, pretouch...
                    (-0.1, 0.3), # GOOD
                    # (-0.25, 0.75), # Just to look at size effect
                ]
            elif animal=="Pancho":
                # Increaingly better compared to (-0.1, 0.1), tested up to (0.1, 0.3).
                LIST_TWIND = [
                    # (0., 0.3),
                    # (-0.2, 0.35),
                    (0.05, 0.25),
                ]
            else:
                print(animal)
                assert False
    elif trajectories_method=="traj_to_scalar" and event in ["04_go_cue"]:
        if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
            LIST_TWIND = [
                (-0.3, 0.4),
            ]            
        else:
            LIST_TWIND = [
                (-0.35, -0.05), # GOOD
                (-0.1, 0.3), # GOOD
            ]
    elif trajectories_method=="traj_to_scalar" and event in ["05_first_raise"]:
        if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
            LIST_TWIND = [
                (-0.3, 0.4),
            ]            
        else:
            if animal=="Diego":
                LIST_TWIND = [
                    (-0.25, 0.2), # GOOD
                ]
            elif animal=="Pancho":
                LIST_TWIND = [
                    (-0.25, 0.2), # GOOD
                ]
            else:
                print(animal)
                assert False
    else:
        print(animal, event, trajectories_method)
        assert False

    if trajectories_method=="scalar":
        LIST_TBIN_DUR = [0.1] # None means no binning
        LIST_TBIN_SLIDE = [0.1] # None means no binning
    elif trajectories_method=="traj_to_scalar":
        if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
            LIST_TBIN_DUR = [0.2] 
            LIST_TBIN_SLIDE = [0.02] 
        else:
            LIST_TBIN_DUR = [0.1] # None means no binning
            LIST_TBIN_SLIDE = [0.025] # None means no binning
    else:
        print(trajectories_method)

    return LIST_TWIND, LIST_TBIN_DUR, LIST_TBIN_SLIDE

def sort_df(DF):
    """ Sort df first by var_var_others, and then by bregion --> useful for making
    consistently ordered plots
    """
    from pythonlib.tools.pandastools import sort_by_two_columns_separate_keys
    from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion_get_mapper
    map_region_to_index = datamod_reorder_by_bregion_get_mapper()

    def _key(br):
        return map_region_to_index[br]

    return sort_by_two_columns_separate_keys(DF, "var_var_others", "bregion", _key)


def params_pairwise_variables_for_plotting():
    """ Return list of params that are hand-saved for useful plotting, each relating to aspeciic question,
    as variables that should be plotted in 45 scatter to compare to each other.

    Incidnetlaly, can be used as a repo for the good, relevant params.
    RETURNS:
        - LIST_LIST_VVO_XY, list of LIST_XXO_XY, which is list of list_x and list_y, each of which is list of
        strings, var_var_others items.
        - LIST_dir_suffix, list of str, as codenames for the "question".
    """

    LIST_LIST_VVO_XY = []
    LIST_dir_suffix = []

    ############ PAIRWISE - testing specific things
    # 1) AnBmCk (two shapes --> preSMA not affected by shape)
    # GOOD (4/16/24)
    dir_suffix = "two_shape_sets"
    LIST_VVO_XY = [
        ["14|syntax_role|('syntax_concrete', 'behseq_locs_clust', 'epoch')", "16|epoch|('syntax_concrete', 'behseq_locs_clust', 'syntax_role')"],
        ["13|syntax_role|('syntax_concrete', 'epoch')", "15|epoch|('syntax_concrete', 'syntax_role')"],
    ]
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### Location vs. chunk_within semantic
    dir_suffix = "invar_location"
    list_vvo_x = [
     "17|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_loconclust_next')",
     "18|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
     "19|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')"]

    list_vvo_y = [
     "35|gridloc|('epoch', 'chunk_rank', 'shape', 'chunk_within_rank_semantic', 'CTXT_shape_prev', 'CTXT_locoffclust_prev', 'CTXT_loconclust_next', 'syntax_concrete')",
     "36|gridloc|('epoch', 'chunk_rank', 'shape', 'chunk_within_rank_semantic', 'CTXT_shape_prev', 'CTXT_locoffclust_prev', 'CTXT_loconclust_next')",
     "37|gridloc|('epoch', 'chunk_rank', 'shape', 'chunk_within_rank_semantic', 'CTXT_shape_prev', 'CTXT_locoffclust_prev')",
     "38|gridloc|('epoch', 'chunk_rank', 'shape', 'chunk_within_rank_semantic', 'CTXT_shape_prev')",
    ]
    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### Syntax concrete vs. ciwithin
    dir_suffix = "invar_syntconcr"
    list_vvo_x = [
     "17|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_loconclust_next')",
     "18|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
     "19|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')"]

    list_vvo_y = [
     "32|syntax_concrete|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'chunk_within_rank_semantic')"]

    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### shape(chunk) vs. ciwithin
    dir_suffix = "contrast_shape"
    list_vvo_x = [
        "34|chunk_rank|('epoch', 'gridloc', 'CTXT_loc_prev', 'chunk_within_rank_semantic')"]

    list_vvo_y = [
         "17|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_loconclust_next')",
         "18|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
         "19|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')"
    ]
    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### ciwithin (last) vs. ciwithin
    dir_suffix = "contrast_ci_cilast"
    list_vvo_x = [
        "22|chunk_within_rank_fromlast|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
        "23|chunk_within_rank_fromlast|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')",
    ]

    list_vvo_y = [
        "20|chunk_within_rank|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
        "21|chunk_within_rank|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')",
         "24|stroke_index|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
         "25|stroke_index|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')",
    ]
    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### ciwithin (last) vs. ci (from start) -- pitted against each other.
    dir_suffix = "contrast_cilast_cifirst"
    list_vvo_x = [
        "26|chunk_within_rank_fromlast|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'chunk_within_rank')",
        "28|chunk_within_rank_fromlast|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'loc_off_clust', 'chunk_within_rank')",
    ]
    list_vvo_y = [
        "27|chunk_within_rank|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'chunk_within_rank_fromlast')",
        "29|chunk_within_rank|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'loc_off_clust', 'chunk_within_rank_fromlast')",
    ]
    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### ciwithin (last) vs. si -- pitted against each other.
    dir_suffix = "contrast_cilast_si"
    list_vvo_x = [
        "30|chunk_within_rank_fromlast|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'loc_off_clust', 'stroke_index')",
    ]
    list_vvo_y = [
        "31|stroke_index|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'loc_off_clust', 'chunk_within_rank_fromlast')",
    ]
    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    ### nprims (onset vs. offset)
    dir_suffix = "contrast_nprims_onset_offset"
    list_vvo_x = [
         "39|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_shape_next')",
         "40|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
         "41|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')",
    ]

    list_vvo_y = [
         "42|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_shape_next')",
         "43|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
         "44|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')",
    ]
    LIST_VVO_XY = []
    for vvo_x in list_vvo_x:
        for vvo_y in list_vvo_y:
            LIST_VVO_XY.append([vvo_x, vvo_y])
    LIST_LIST_VVO_XY.append(LIST_VVO_XY)
    LIST_dir_suffix.append(dir_suffix)

    return LIST_LIST_VVO_XY, LIST_dir_suffix

def plot_pairwise_all_wrapper(DFRES, SAVEDIR):
    """
    Wrapper to make ALL plots that are scatter, pairwise between variables, to
    make specific points

    NOTE: all updated as of 4/16/24
    :param DFRES:
    :return:
    """
    for VERSION in ["nosup_vs_sup", "shape_vs_dir", "nocol_vs_col"]:
        try:
            _plot_pairwise_btw_levels_for_seqsup(DFRES, SAVEDIR, VERSION=VERSION)
        except Exception as err:
            print(err)
            print("Skipping plot_pairwise_btw_levels_for_seqsup... version:", VERSION)

    LIST_LIST_VVO_XY, LIST_dir_suffix = params_pairwise_variables_for_plotting()

    for LIST_VVO_XY, dir_suffix in zip(LIST_LIST_VVO_XY, LIST_dir_suffix):
        _plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix=dir_suffix)

def _plot_pairwise_btw_vvo_general_MULT(DFRES_MULT, ythis, SAVEDIR, LIST_VVO_XY,
                                        dir_suffix=None, plot_text=True,
                                        version="one_subplot_per_bregion",
                                        map_subplot_var_to_new_subplot_var=None):
    """
    Like plot_pairwise_btw_vvo_general but each datapt is a date (average) and each
    subplot is a brain region --> for summarizing across dates (experiemnts).
    PARAMS:
    - ythis, e.g., "dist_norm_95"
    """
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping


    # subplot_levels_to_separate = ["preSMA_a", "preSMA_p"]
    # map_subplot_var_to_new_subplot_var = {
    #     "preSMA_a":"preSMA_a",
    #     "preSMA_p":"preSMA_p",
    # }
    # def F(x):
    #     if x["bregion"] in map_subplot_var_to_new_subplot_var:
    #         return map_subplot_var_to_new_subplot_var[x["bregion"]]
    #     else:
    #         return "LEFTOVER"
    # DFRES_MULT = applyFunctionToAllRows(DFRES_MULT, F, "subplot_set")

    from pythonlib.tools.pandastools import stringify_values
    DFRES_MULT = stringify_values(DFRES_MULT)

    from pythonlib.tools.pandastools import append_col_with_grp_index, applyFunctionToAllRows
    DFRES_MULT = append_col_with_grp_index(DFRES_MULT, ["date", "bregion"], "date_bregion")
    if version=="one_subplot_per_bregion":
        # Standrad
        var_subplot = "bregion"
        var_datapt = "date_bregion"
    elif version=="single_subplot":
        # Useful for comparing across regions.
        var_subplot = None
        var_datapt = "date_bregion"
    else:
        assert False

    if dir_suffix is None:
        savedir = f"{SAVEDIR}/PAIRWISE_VVO_MULT"
    else:
        savedir = f"{SAVEDIR}/PAIRWISE_VVO-MULT-{dir_suffix}"

    shuffled = False
    list_dat_lev = DFRES_MULT["dat_level"].unique().tolist()
    list_effect_context = DFRES_MULT["effect_context"].unique().tolist()
    for dat_lev in list_dat_lev:
        for effect_context in list_effect_context:
            print(dat_lev, list_effect_context)

            # Keep just this.
            dfthis = DFRES_MULT[
                (DFRES_MULT["dat_level"] == dat_lev) & (DFRES_MULT["shuffled"] == shuffled) & (DFRES_MULT["effect_context"]==effect_context)
            ].reset_index(drop=True)

            for vvo_x, vvo_y in LIST_VVO_XY:
                if vvo_x in dfthis["var_var_others"].tolist() and vvo_y in dfthis["var_var_others"].tolist():
                    savedirthis = f"{savedir}/dat_lev={dat_lev}-effect_context={effect_context}-vvo_x={vvo_x[:40]}-vvo_y={vvo_y[:40]}"
                    os.makedirs(savedirthis, exist_ok=True)

                    # print("---- Plotting for dat_lev = ", dat_lev)
                    # print("x = ", vvo_x)
                    # print("y = ", vvo_y)
                    print("Saving at .. ", savedirthis)

                    _, fig = plot_45scatter_means_flexible_grouping(dfthis, "var_var_others", vvo_x, vvo_y,
                                                            var_subplot, ythis, var_datapt,
                                                            shareaxes=True, plot_text=plot_text, SIZE=4,
                                                            map_subplot_var_to_new_subplot_var=map_subplot_var_to_new_subplot_var)
                    if fig is not None:
                        path = f"{savedirthis}/ythis={ythis}.pdf"
                        savefig(fig, path)

                    plt.close("all")
    plt.close("all")

def _plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix=None):
    """
    Good - plot scatter to compare effects across pairs of vvo (var_vars_others) -- .e,g,
    compare effect of syntax (holding shape constant) vs. shape (holding syntax constant), each of those
    being a vvo.
    (Wrote this for AnBmCk, two shapes, showing low effect of shape and high effect of syntax role, for preSMA.
    :param DFRES:
    :param SAVEDIR:
    :param LIST_VVO_XY:
    :return:
    """
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    if dir_suffix is None:
        savedir = f"{SAVEDIR}/PAIRWISE_VVO"
    else:
        savedir = f"{SAVEDIR}/PAIRWISE_VVO-{dir_suffix}"

    # Condition.
    DFRES, _ = compute_normalized_distances(DFRES)

    # Compute "yue" neural distnac emetrics.
    DFRES_PIVOT_YUE = plot_all_results_yue(DFRES, "/tmp", PLOT=False)

    # Regular  metrics
    ythis_done = [] # hacky, to maek sure dont overwrite plot with same ythis name.
    for DFTHIS, dat_lev, ythis in [
        [DFRES, "distr", "dist_norm_95"],
        # [DFRES, "pts_yue_log", "dist"],
        [DFRES, "pts_yue_diff", "dist_norm_95"],
        # [DFRES_PIVOT_YUE, "dist_yue_diff"],
        # [DFRES_PIVOT_YUE, "dist_yue"],
        # [DFRES_PIVOT_YUE, "dist_yue_log"],
        ]:

        assert (dat_lev, ythis) not in ythis_done, "this hacky solution sodesnt work - have to also make filenmae use DFTHIs version"
        ythis_done.append((dat_lev, ythis))

        # ----------
        shuffled = False
        dfthis = DFTHIS[
            (DFTHIS["dat_level"] == dat_lev) & (DFTHIS["shuffled"] == shuffled)
        ].reset_index(drop=True)

        for vvo_x, vvo_y in LIST_VVO_XY:
            if vvo_x in dfthis["var_var_others"].tolist() and vvo_y in dfthis["var_var_others"].tolist():
                savedirthis = f"{savedir}/dat_lev={dat_lev}-vvo_x={vvo_x[:50]}-vvo_y={vvo_y[:50]}"
                os.makedirs(savedirthis, exist_ok=True)

                print("---- Plotting for dat_lev = ", dat_lev)
                print("x = ", vvo_x)
                print("y = ", vvo_y)
                # print("at .. ", savedirthis)

                if "effect_context" in dfthis.columns:
                    var_subplot = "effect_context"
                else:
                    var_subplot = None
                _, fig = plot_45scatter_means_flexible_grouping(dfthis, "var_var_others", vvo_x, vvo_y,
                                                        var_subplot, ythis, "bregion",
                                                        shareaxes=False, SIZE=4)
                if fig is not None:
                    path = f"{savedirthis}/ythis={ythis}.pdf"
                    savefig(fig, path)

                plt.close("all")
            else:
                print("--- Skipping, did not find in data (var_var_others):")
                print(vvo_x)
                print(vvo_y)
                # print("Unique values...:", dfthis["var_var_others"].unique().tolist())
    plt.close("all")


def _plot_pairwise_btw_levels_for_seqsup(DFRES, SAVEDIR, VERSION="nosup_vs_sup",
                                         one_subplot_per_bregion=False):
    """
    Specific (and a bit hacky) to test for whether preSMA stroke index represntation collapses during sequence supervision.
    Plots 45deg scatter between (without supervision, x axi) and (with sup, yaxis), for tasks controlled to have same
    motor beh --> predicttion is below unity line for preSMA but not for others.

    "Hacky" becuase it is veruy specific to ovar having this struvture:
     - musst having the variable superv_is_seq_sup,
     - first variabel must be either "epochset_dir" or "epochset_shape" (to ensure "same motor" across levels). Will
     use only cases that are not ("LEFTOVER",).

    :param DFRES:
    :param VERSION: whcih comaprison to make plots for, str,
    -- nosup_vs_sup
    -- shape_vs_dir
    :param one_subplot_per_bregion: bool, then datapt is "date". This is useful for multi-day combined data...
    :param SAVEDIR:
    :return:
    """
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping


    savedir = f"{SAVEDIR}/pairwise_btw_levels_SEQSUP--VER={VERSION}--subplot_bregion={one_subplot_per_bregion}"
    os.makedirs(savedir, exist_ok=True)
    print("Saving to.. ", savedir)

    if "var_others_tuple" in DFRES.columns:
        _var = "var_others_tuple"
    else:
        _var = "var_others"
    # First,
    if VERSION=="nosup_vs_sup":
        inds_keep = [i for i, var_others in enumerate(DFRES[_var]) if (var_others[0] in ["epochset_shape", "epochset_dir"]) and ("superv_is_seq_sup" in var_others)]
    elif VERSION=="nocol_vs_col":
        inds_keep = [i for i, var_others in enumerate(DFRES[_var]) if (var_others[0] in ["epochset_shape", "epochset_dir"]) and ("INSTRUCTION_COLOR" in var_others)]
    elif VERSION=="shape_vs_dir":
        inds_keep = [i for i, var_others in enumerate(DFRES[_var]) if (var_others[0] in ["epochset_shape", "epochset_dir"])]
    else:
        print(VERSION)
        assert False

    DFRES = DFRES.iloc[inds_keep].reset_index(drop=True)

    ### HACKY Cleanup string names in levo, since diff levo can be same but diff named...

    # (1)
    # See within for the things that are converted.
    conversions = {}
    list_levo_simp = []
    for levo in DFRES["levo"].tolist():
        if isinstance(levo, str):
            levo_simp = levo
        elif isinstance(levo, tuple) and len(levo)==1:
            levo_simp = levo
        else:

            if levo[1][-4:] == "|0|S":
                # 'llCV3|0|S' --> 'llCV3|S'
                levo_simp = list(levo)
                levo_simp[1] = levo[1][:-4] + levo[1][-2:]
                levo_simp = tuple(levo_simp)
                if levo not in conversions:
                    conversions[levo] = levo_simp
                else:
                    assert conversions[levo] == levo_simp
            elif (levo[1][-2:] == "|0") and (levo[1][-4] != "|"):
                # 'llCV3|0' --> 'llCV3'
                levo_simp = list(levo)
                levo_simp[1] = levo[1][:-2]
                levo_simp = tuple(levo_simp)
                if levo not in conversions:
                    conversions[levo] = levo_simp
                else:
                    assert conversions[levo] == levo_simp
            else:
                levo_simp = levo

        list_levo_simp.append(levo_simp)
    print("Made these levo name conversions...")
    for k, v in conversions.items():
        print(k, "-->", v)
    DFRES = DFRES.copy()
    DFRES["levo"] = list_levo_simp

    # (2) Making the epochnames generic.
    # - remove epoch
    # - replace epochset with "sahpe" or "direction"
    # --> both allow agging across expts.
    # e.g, (('LEFTOVER',), 'L', 0, 'line-6-1-0', False, True) --> (('LEFTOVER',), 0, 'line-6-1-0', False, True)
    from pythonlib.dataset.modeling.discrete import MAP_EPOCH_EPOCHKIND
    # Merge all shape-related rules
    tmp = {}
    for k, v in MAP_EPOCH_EPOCHKIND.items():
        if v in ["AnBm", "AnBmDir"]:
            tmp[k] = "shape"
        else:
            tmp[k] = v
    MAP_EPOCH_EPOCHKIND = tmp

    conversions = {}
    list_levo_simp = []
    list_ovar_tuple = []
    for i, row in DFRES.iterrows():

        if isinstance(row["levo"], str):
            levo_new = row["levo"]
            ovar_tuple = row[_var]
        else:
            # (1) Remove the epoch from levo...
            if "epoch_rand" in row[_var]:
                idx_epoch = row[_var].index("epoch_rand")
            else:
                idx_epoch = row[_var].index("epoch")
            levo_new = list(row["levo"][:idx_epoch]) + list(row["levo"][idx_epoch+1:])
            ovar_tuple = tuple(list(row[_var][:idx_epoch]) + list(row[_var][idx_epoch+1:]))

            # (2) Replace the epochset with a generic term
            epochset = levo_new[0]
            assert row[_var][0] in ["epochset_dir", "epochset_shape", "epochset"], "then taking levo_new[0] is incorrect"
            try:
                if isinstance(epochset, tuple) and len(epochset)==0:
                    # This happens e,,g for epoch=="presetrand", and epochset_dir = ()
                    pass
                elif epochset == ('LEFTOVER',):
                    # will get exlcuded later.
                    pass
                elif epochset[0] in MAP_EPOCH_EPOCHKIND.keys():
                    assert len(epochset)==1
                    levo_new[0] = tuple([MAP_EPOCH_EPOCHKIND[epochset[0]]])
                else:
                    print(row["levo"])
                    print(levo_new)
                    print(MAP_EPOCH_EPOCHKIND.keys())
                    assert False, "hand enter this epoch..."
            except Exception as err:
                print("epochset: ", epochset)
                for k, v in MAP_EPOCH_EPOCHKIND.items():
                    print(k, " -- ", v)
                print(row["levo"])
                print(levo_new)
                print(row[_var])
                raise err
            levo_new = tuple(levo_new)

        list_levo_simp.append(levo_new)
        list_ovar_tuple.append(ovar_tuple)
        if row["levo"] not in conversions:
            conversions[row["levo"]] = levo_new
        else:
            assert conversions[row["levo"]] == levo_new

    DFRES = DFRES.copy()
    
    DFRES["levo"] = list_levo_simp
    DFRES[_var] = list_ovar_tuple

    print("Made these levo name conversions...")
    for k, v in conversions.items():
        print(k, "-->", v)

    #### Decide which variables are x and y.
    for var_var_others in DFRES["var_var_others"].unique():

        # Need to do this up here.
        DFTHIS = DFRES[
            (DFRES["var_var_others"] == var_var_others) & (DFRES["context_samediff"] == "same")
        ].reset_index(drop=True)

        if VERSION=="nosup_vs_sup":
            # Determine which is the "no sup" and "yes sup" levo
            # - This keeps only levels that are (i) matched motor across epochs and (ii) splits them
            # into two lists based on whether is sequenevc sup.
            possible_lev_x = []
            possible_lev_y = []
            possible_levels = []
            for i, row in DFTHIS.iterrows():

                # WHich index holds the variable indication if this is sequence supervision?
                if "superv_is_seq_sup" in row[_var]:
                    idx_check = row[_var].index("superv_is_seq_sup")
                else:
                    idx_check = None

                # Determoine if this is to keeop
                try:
                    if idx_check is None:
                        # seuqence supevision is not even a relevnat part of this variables.
                        pass
                    elif row["levo"][0] == ("LEFTOVER",):
                        # Is not motor controlled (same beh)
                        pass
                    elif row["levo"][idx_check] == False:
                        if row["levo"] not in possible_lev_x:
                            possible_lev_x.append(row["levo"])
                    elif row["levo"][idx_check] == True:
                        if row["levo"] not in possible_lev_y:
                            possible_lev_y.append(row["levo"])
                    else:
                        print(row["levo"])
                        assert False
                except Exception as err:
                    print(idx_check)
                    print(row["levo"])
                    print(len(row["levo"]))
                    print(row[_var])
                    raise err

            # combine all levels
            possible_levels = possible_lev_x + possible_lev_y
        elif VERSION in ["nocol_vs_col", "shape_vs_dir"]:
            # QUicka nd simple, just plot each pair of vars without subselecting x and y vars.
            possible_levels = []
            for i, row in DFTHIS.iterrows():

                # WHich index holds the variable indication if this is sequence supervision?
                # Determoine if this is to keeop
                if row["levo"][0] == ("LEFTOVER",):
                    # Is not motor controlled (same beh)
                    pass
                else:
                    if row["levo"] not in possible_levels:
                        possible_levels.append(row["levo"])

            possible_lev_x = possible_levels
            possible_lev_y = possible_levels

        # pick one (arbitrary)
        if False: # Instead, use each lev_x
            lev_x = sorted(possible_lev_x)[0]
            # The others make list of possible ys
            list_lev_y = [lev for lev in possible_lev_x + possible_lev_y if not lev_x == lev]

            print("The fixed x level:", lev_x)
            print("The possible y levels:", list_lev_y)

        # Make plots
        savedirthis = f"{savedir}/{var_var_others}"
        os.makedirs(savedirthis, exist_ok=True)
        pairs_already_done = []

        ##### PLOTS
        # for x in possible_lev_x:
        #     print(x)
        # for y in possible_lev_y:
        #     print(y)
        # assert False
        list_yvar_dat_level = [
            ["dist_norm_95", "distr"],
            ["dist_norm_95", "pts_yue_diff"],
            # ["dist", "pts_yue_log"],
        ]
        for lev_x in possible_lev_x:
            for lev_y in possible_lev_y:
                if lev_x != lev_y:
                    if tuple(sorted([lev_x, lev_y])) not in pairs_already_done:
                        if lev_x[0] == lev_y[0]: # Only plot if they are same epochset
                            pairs_already_done.append(tuple(sorted([lev_x, lev_y])))

                            # 3 compare effect across levels -->
                            for yvarthis, dat_lev in list_yvar_dat_level:
                            # list_dat_lev = DFTHIS["dat_level"].unique().tolist()
                            # for dat_lev in list_dat_lev:
                                if one_subplot_per_bregion==False:
                                    # Standard --> one pt per region. Useful for a single session plot.
                                    dfthis = DFTHIS[DFTHIS["dat_level"] == dat_lev].reset_index(drop=True)
                                    _, fig = plot_45scatter_means_flexible_grouping(dfthis, "levo", lev_x, lev_y,
                                                                    "effect_context", yvarthis, "bregion",
                                                                           shareaxes=True)
                                    path = f"{savedirthis}/dat_lev={dat_lev}--lev_x={lev_x}--lev_y={lev_y}.pdf"
                                    savefig(fig, path)
                                else:
                                    # Multi-session plot.
                                    effect_context = "diff|same" # hacky...
                                    dfthis = DFTHIS[
                                        (DFTHIS["dat_level"] == dat_lev) & (DFTHIS["effect_context"]==effect_context)
                                        ].reset_index(drop=True)
                                    if len(dfthis)>0:
                                        _, fig = plot_45scatter_means_flexible_grouping(dfthis, "levo", lev_x, lev_y,
                                                                        "bregion", yvarthis, "date",
                                                                               shareaxes=True)
                                        path = f"{savedirthis}/dat_lev={dat_lev}--eff_cont={effect_context}--lev_x={lev_x}--lev_y={lev_y}.pdf"
                                        savefig(fig, path)

                                plt.close("all")
                        else:
                            pass # 9/5/24 - wasnt sure how to fix this bug, just let is pass, shoudl be fine.
                            # print(lev_x[0], lev_y[0])
                            # assert False

def compute_normalized_distances(DFRES):
    """
    QUickly add columns and do quick normalziation of data to width of distribution.
    Is OK to run repeatedly on same DFRES, will check that it's been done...
    :param DFRES:
    :return:
    """
    from pythonlib.tools.pandastools import stringify_values

    # What overall distance to normalize all distances to (i.e, "width of distribution").
    DIST_NULL = "DIST_NULL_98"

    if "done_compute_normalized_distances" in DFRES.columns:
        assert np.all(DFRES["done_compute_normalized_distances"] == True)
        # if isinstance(DFRES["var_others"].values[0], tuple):
        #     # Already been run ...
        return DFRES, DIST_NULL

    # Compute normalized distnaces
    DFRES["dist_norm_95"] = DFRES["dist"]/DFRES[DIST_NULL]
    # DFRES["dist_norm_95"] = DFRES["dist"]/DFRES["DIST_NULL_95"]
    # DFRES["dist_norm_50"] = DFRES["dist"]/DFRES["DIST_NULL_50"]
    DFRES["var_others"] = [tuple(x) for x in DFRES["var_others"]]

    # convert from 4 --> 04 (for sorting properly)
    list_idx = []
    for i, row in DFRES.iterrows():
        idx_old = str(row["index_var"])
        if len(idx_old)==1:
            idx = f"0{idx_old}"
        elif len(idx_old)==2:
            idx = f"{idx_old}"
        else:
            print(idx_old)
            assert False, "extend to length 3..."
        list_idx.append(idx)
    DFRES["index_var_str"] = list_idx

    DFRES = append_col_with_grp_index(DFRES, ["index_var_str", "var", "var_others"], "var_var_others")
    DFRES = append_col_with_grp_index(DFRES, ["effect_samediff", "context_samediff"], "effect_context")

    # If no animal and date, give it dumym (will be assumed to exist later)
    if "animal" not in DFRES.columns:
        DFRES["animal"] = "dummy"

    if "date" not in DFRES.columns:
        DFRES["date"] = -1

    DFRES = sort_df(DFRES)

    # Stringify, and keep tuples.
    if False:
        # Only do this right before plotting, to retain var as tuples useful.
        var_others_tuple = DFRES["var_others"].tolist()
        DFRES = stringify_values(DFRES)
        DFRES["var_others_tuple"] = var_others_tuple

    DFRES["done_compute_normalized_distances"] = True
    return DFRES, DIST_NULL

def plot_all_results_yue(DFRES, SAVEDIR, PLOT=True):
    """
    PLot results (effects mainly), using yue's Nueral modulation index and related metrics.
    :param DFRES:
    :param SAVEDIR:
    :return:
    """
    from pythonlib.tools.pandastools import pivot_table
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.snstools import map_function_tofacet

    DFRES, _ = compute_normalized_distances(DFRES)

    savedir = f"{SAVEDIR}/FIGURES_YUE"
    os.makedirs(savedir, exist_ok=True)

    #################### YUE scores
    # Get dataframe with each row being a specific set of variables.
    if True:
        DFRES = stringify_values(DFRES)
    DFRES_THIS = DFRES[(DFRES["dat_level"] == "pts") & (DFRES["context_samediff"] == "same")]
    DFRES_PIVOT = pivot_table(DFRES_THIS, ["animal", "date", "var", "var_others", "shuffled", "bregion", "twind", "event",
                                           "var_var_others", "dat_level", "levo", "leveff", "twind_analy"],
                              ["effect_context"], ["dist_norm_95"],
                              flatten_col_names=True).reset_index(drop=True)
    DFRES_PIVOT["effect_context"] = "IGNORE"

    # keep only if have both diff|same and same|same
    # print(len(DFRES_PIVOT))
    DFRES_PIVOT = DFRES_PIVOT[~DFRES_PIVOT["dist_norm_95-diff|same"].isna()]
    # print(len(DFRES_PIVOT))
    DFRES_PIVOT = DFRES_PIVOT[~DFRES_PIVOT["dist_norm_95-same|same"].isna()]
    # print(len(DFRES_PIVOT))
    DFRES_PIVOT = DFRES_PIVOT.reset_index(drop=True)

    # Compute scores, two versions, either divide or diff.
    DFRES_PIVOT["dist_yue"] = DFRES_PIVOT["dist_norm_95-diff|same"] / DFRES_PIVOT["dist_norm_95-same|same"]
    DFRES_PIVOT["dist_yue_log"] = np.log2(DFRES_PIVOT["dist_norm_95-diff|same"] / DFRES_PIVOT["dist_norm_95-same|same"])
    DFRES_PIVOT["dist_yue_diff"] = DFRES_PIVOT["dist_norm_95-diff|same"] - DFRES_PIVOT["dist_norm_95-same|same"]

    # Sort
    DFRES_PIVOT = sort_df(DFRES_PIVOT)

    if PLOT:
        # for yvarthis in ["dist_yue", "dist_yue_log", "dist_yue_diff"]:
        for yvarthis in ["dist_yue", "dist_yue_log", "dist_yue_diff"]:
            fig = sns.catplot(data=DFRES_PIVOT, x="bregion", y=yvarthis, col="var_var_others",
                              col_wrap=3, aspect=1.57, alpha=0.4, height=6)
            if yvarthis=="dist_yue":
                map_function_tofacet(fig, lambda ax: ax.axhline(1, color="k", alpha=0.4))
            else:
                map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))

            rotateLabel(fig)
            savefig(fig, f"{savedir}/overview_scatter-YUE-{yvarthis}.pdf")

            if yvarthis in ["dist_yue_log", "dist_yue_diff"]: # aling to 0
                fig = sns.catplot(data=DFRES_PIVOT, x="bregion", y=yvarthis, col="var_var_others",
                                  col_wrap=3, aspect=1.7, kind="bar", height=6, errorbar=('ci', 68))
                rotateLabel(fig)
                map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))
                savefig(fig, f"{savedir}/overview_bar-YUE-{yvarthis}.pdf")
                plt.close("all")
            elif yvarthis=="dist_yue":
                fig = sns.catplot(data=DFRES_PIVOT, x="bregion", y=yvarthis, col="var_var_others",
                                  col_wrap=3, aspect=1.7, kind="point", height=6, errorbar=('ci', 68))
                map_function_tofacet(fig, lambda ax: ax.axhline(1, color="k", alpha=0.4))
                rotateLabel(fig)
                savefig(fig, f"{savedir}/overview_bar-YUE-{yvarthis}.pdf")
                plt.close("all")
            else:
                assert False
        plt.close("all")

    return DFRES_PIVOT

def compute_all_derived_metrics(DFRES):
    """
    Key principle:
    - DFRES that is returned remains unchanged, but is modded in here to compute derived stuff.
    Unchange is important --> retains vars as list of tuples, not as string. Then do stringify before each ploting.
    :param DFRES:
    :return:
    Copy of DFRES, and other derived metrics in different dataframes.
    """
    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.pandastools import stringify_values

    from pythonlib.tools.pandastools import _check_index_reseted
    _check_index_reseted(DFRES)

    if "shuffled" not in DFRES:
        DFRES["shuffled"] = False
    if "twind" not in DFRES:
        DFRES["twind"] = "dummy"
    if "event" not in DFRES:
        DFRES["event"] = "dummy"

    dat_level_gen_indices = "pts" # This is correct -- don't use pts_yue diff, since the below codes does those diffing.
    yvar_gen_indices = "dist_norm_95"

    # Compute normalized distnaces
    DFRES, DIST_NULL = compute_normalized_distances(DFRES)


    # Stringify, or else will fail groupby step
    DFRES_ORIG = DFRES.copy()
    if True:
        # Just to compute derived stuff
        DFRES = stringify_values(DFRES)

    ###########################################################################
    # Get dataframe with each row being a specific set of variables.
    DFRES_PIVOT = pivot_table(DFRES, ["animal", "date", "var", "var_others", "shuffled", "bregion", "twind",
                                      "event", "var_var_others", "dat_level"], ["effect_context"],
                              [yvar_gen_indices], flatten_col_names=True).reset_index(drop=True)
    DFRES_PIVOT["effect_context"] = "IGNORE"

    # Compute effects tha DFRES_PIVOT[DFRES_PIVOT["dat_level"] == "pts"].reset_index(drop=True)t require inputs from multiple distance metrics.
    DFRES_PIVOT_DISTR = DFRES_PIVOT[DFRES_PIVOT["dat_level"] == "distr"].reset_index(drop=True)
    DFRES_PIVOT_DISTR["effect_index"] = DFRES_PIVOT_DISTR["dist_norm_95-diff|same"] / (DFRES_PIVOT_DISTR["dist_norm_95-diff|same"] + DFRES_PIVOT_DISTR["dist_norm_95-same|diff"])

    # Keep only the data using pairwise distances
    DFRES_PIVOT_PAIRWISE = DFRES_PIVOT[DFRES_PIVOT["dat_level"] == dat_level_gen_indices].reset_index(drop=True)

    DFRES_PIVOT_PAIRWISE["effect_index"] = DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|same"] / (DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|same"] + DFRES_PIVOT_PAIRWISE["dist_norm_95-same|diff"])

    DFRES_PIVOT_PAIRWISE["norm_dist_effect"] = DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|same"]-DFRES_PIVOT_PAIRWISE["dist_norm_95-same|same"]
    # This makes less sense --> diff|diff can be different for many reasons, emprticlaly doesnt match intuition that well
    # DFRES_PIVOT_PAIRWISE["norm_dist_context"] = DFRES_PIVOT_PAIRWISE["dist_norm_95-same|diff"] - DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|diff"]
    DFRES_PIVOT_PAIRWISE["norm_dist_context"] = DFRES_PIVOT_PAIRWISE["dist_norm_95-same|diff"] - DFRES_PIVOT_PAIRWISE["dist_norm_95-same|same"]
    DFRES_PIVOT_PAIRWISE["norm_dist_both"] = DFRES_PIVOT_PAIRWISE["norm_dist_effect"] - DFRES_PIVOT_PAIRWISE["norm_dist_context"]
    # DFRES_PIVOT_PAIRWISE["norm_dist_effect"] = DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|same"]/DFRES_PIVOT_PAIRWISE["dist_norm_95-same|same"]
    # DFRES_PIVOT_PAIRWISE["norm_dist_context"] = DFRES_PIVOT_PAIRWISE["dist_norm_95-same|diff"]/DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|diff"]
    # DFRES_PIVOT_PAIRWISE["norm_dist_both"] = DFRES_PIVOT_PAIRWISE["norm_dist_effect"]/DFRES_PIVOT_PAIRWISE["norm_dist_context"]

    ################### Good normalization method...
    # - First, cap everything by min and max (normalize all do (diff, diff) (so max is 1))
    SS = DFRES_PIVOT_PAIRWISE["dist_norm_95-same|same"].values
    DD = DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|diff"].values
    DS = DFRES_PIVOT_PAIRWISE["dist_norm_95-diff|same"].values
    SD = DFRES_PIVOT_PAIRWISE["dist_norm_95-same|diff"].values
    MIN = SS
    MAX = DD

    # Requiored, or else faiols downstream beacuse A will be neg
    SS[SS > DD] = 0.99*DD[SS > DD]

    # - clamp
    DS[DS < MIN] = MIN[DS < MIN]
    DS[DS > MAX] = MAX[DS > MAX]
    SD[SD < MIN] = MIN[SD < MIN]
    SD[SD > MAX] = MAX[SD > MAX]

    def _compute_scores(A, B, C, D, ignore_division=False):

        for _x in [A, B, C, D]:
            # print(_x[~np.isnan(_x)])
            # print(_x[~np.isnan(_x)]>=0)
            if not np.all(_x[~np.isnan(_x)]>=0.):
                assert False
        # assert np.all(A>=0)
        # assert np.all(B>=0)
        # assert np.all(C>=0)
        # assert np.all(D>=0)

        if ignore_division:
            s1, s2 = None, None
        else:
            s1 = (A/B) * (C/D) # aka (A*C)/(B*D)
            s2 = (A*C) - (B*D)

        ################
        s3 = 0.5 * (A - B) + (C - D)

        a = A - B
        b = C - D
        a[a<0] = 0. # to make sure dont multiply neg by neg.
        b[b<0] = 0.
        s4 = a * b

        a = A - D
        b = C - B
        a[a<0] = 0. # to make sure dont multiply neg by neg.
        b[b<0] = 0.
        s5 = a * b

        return s1, s2, s3, s4, s5

    # SCores that use ratios
    A = DS/SS
    B = DD/DS
    C = DD/SD
    D = SD/SS
    # -- good ones:
    s1, s2, s3, s4, s5 = _compute_scores(A, B, C, D)
    DFRES_PIVOT_PAIRWISE["gen_idx_ratio_1"] = s1
    DFRES_PIVOT_PAIRWISE["gen_idx_ratio_2"] = s2
    # -- Just testing
    DFRES_PIVOT_PAIRWISE["gen_idx_ratio_3"] = s3
    DFRES_PIVOT_PAIRWISE["gen_idx_ratio_4"] = s4
    DFRES_PIVOT_PAIRWISE["gen_idx_ratio_5"] = s5

    # Scores that use differences
    A = DS - SS
    B = DD - DS
    C = DD - SD
    D = SD - SS
    s1, s2, s3, s4, s5 = _compute_scores(A, B, C, D, ignore_division=True)

    # -- Just testing:
    # DFRES_PIVOT_PAIRWISE["gen_idx_diff_1"] = s1 # Skip, they can fail.
    # DFRES_PIVOT_PAIRWISE["gen_idx_diff_2"] = s2
    # -- good ones
    DFRES_PIVOT_PAIRWISE["gen_idx_diff_3"] = s3
    DFRES_PIVOT_PAIRWISE["gen_idx_diff_4"] = s4
    DFRES_PIVOT_PAIRWISE["gen_idx_diff_5"] = s5

    ############## OLDER VERSION OF GENERLAZATION INDEX
    # 1. normalize all do (diff, diff) (so max is 1)
    yvar = "dist_norm_95"
    for ef in ["same", "diff"]:
        for ctxt in ["same", "diff"]:
            DFRES_PIVOT_PAIRWISE[f"DIST-{ef}|{ctxt}"] = DFRES_PIVOT_PAIRWISE[f"{yvar}-{ef}|{ctxt}"]/DFRES_PIVOT_PAIRWISE[f"{yvar}-diff|diff"]

    # 2.
    A = DFRES_PIVOT_PAIRWISE[f"DIST-diff|same"] - DFRES_PIVOT_PAIRWISE[f"DIST-same|same"]
    B = DFRES_PIVOT_PAIRWISE[f"DIST-diff|diff"] - DFRES_PIVOT_PAIRWISE[f"DIST-same|diff"]
    C = (DFRES_PIVOT_PAIRWISE[f"DIST-diff|diff"] - DFRES_PIVOT_PAIRWISE[f"DIST-same|same"]) + 0.02 # 0.02 is to reduce noise.
    DFRES_PIVOT_PAIRWISE["generalization_index"] = A*B
    DFRES_PIVOT_PAIRWISE["generalization_index_scaled"] = (A/C) * (B/C)

    # Sort, for consistent plotting across expts.

    DFRES_PIVOT_PAIRWISE = sort_df(DFRES_PIVOT_PAIRWISE)
    # DFRES = sort_df(DFRES)
    DFRES_ORIG = sort_df(DFRES_ORIG)
    DFRES_PIVOT_DISTR = sort_df(DFRES_PIVOT_DISTR)

    ################## YUE RELATED METRICS
    DFRES_PIVOT_YUE = plot_all_results_yue(DFRES_ORIG, None, False)

    plot_params = {
        "yvar":yvar,
        "DIST_NULL":DIST_NULL
    }

    return DFRES_ORIG, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, plot_params

def plot_histograms_clean_wrapper(DFRES, SAVEDIR, dat_level = "pts_yue_diff", effect_context = "diff|same",
                                  ythis = "dist_norm_95",
                                  bregions_plot=None):
    """
    Good, all plots summarizing specific vars and bregions.
    :param DFRES:
    :param dat_level:
    :param effect_context:
    :return:
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import params_pairwise_variables_for_plotting, _plot_histograms_clean
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import stringify_values

    # (1) Auto get the variables to plot, each set is a single set of figures
    LIST_LIST_VVO_XY, LIST_dir_suffix = params_pairwise_variables_for_plotting()

    ########################## PLOTS
    # (1) Grand mean
    for LIST_VVO_XY, dir_suffix in zip(LIST_LIST_VVO_XY, LIST_dir_suffix):
        list_vvo = sorted(set([xx for x in LIST_VVO_XY for xx in x])) # list of str

        savedir = f"{SAVEDIR}/HISTOGRAMS_CLEAN_GRAND_MEAN/{dir_suffix}"
        os.makedirs(savedir, exist_ok=True)
        print(savedir)

        _plot_histograms_clean(DFRES, list_vvo, savedir, effect_context=effect_context, dat_level=dat_level,
                               bregions_plot=bregions_plot)


    if False: # Skip, since is doing above in grand_mean --> the plot that separates into days
        # (2) Aggregate so each day is one pt.
        DFRES_STR = stringify_values(DFRES)
        DFRES_AGG = aggregGeneral(DFRES_STR, ["animal", "date", "effect_context", "question", "var",
                                              "var_others", "shuffled", "bregion", "twind", "event",
                                              "var_var_others", "dat_level", "leveff", "twind_analy"],
                                  values=["dist_norm_95"])
        for LIST_VVO_XY, dir_suffix in zip(LIST_LIST_VVO_XY, LIST_dir_suffix):
            list_vvo = sorted(set([xx for x in LIST_VVO_XY for xx in x])) # list of str

            savedir = f"{SAVEDIR}/HISTOGRAMS_CLEAN_VVO_DAY_PTS/{dir_suffix}"
            os.makedirs(savedir, exist_ok=True)
            print(savedir)

            _plot_histograms_clean(DFRES_AGG, list_vvo, savedir, effect_context=effect_context, dat_level=dat_level,
                                   bregions_plot=bregions_plot)

    # (2) Separate each day
    for date in DFRES["date"].unique().tolist():
        DFTHIS = DFRES[(DFRES["date"] == date)]

        for LIST_VVO_XY, dir_suffix in zip(LIST_LIST_VVO_XY, LIST_dir_suffix):
            list_vvo = sorted(set([xx for x in LIST_VVO_XY for xx in x])) # list of str
            savedir = f"{SAVEDIR}/HISTOGRAMS_CLEAN_VVO_EACH_DATE/{date}/{dir_suffix}"
            os.makedirs(savedir, exist_ok=True)
            print(savedir)

            _plot_histograms_clean(DFTHIS, list_vvo, savedir, effect_context=effect_context, dat_level=dat_level,
                                   bregions_plot=bregions_plot, ythis=ythis)

def _plot_histograms_clean(DFRES, list_vvo, savedir, effect_context="diff|same",
                           dat_level="pts_yue_diff", bregions_plot=None,
                           ythis = "dist_norm_95"):
    """
    Helper to plot clean plots of many kinds, focusing on speicifc variables, including specific brain regions
    to compare.

    :param DFRES:
    :param list_vvo: list of var_var_others (strings)
    :param savedir: The immediate dir to save figs
    :param effect_context:
    :param dat_level:
    :param bregions_plot: list of bregions (srings)
    :return:
    """
    from pythonlib.tools.snstools import map_function_tofacet, rotateLabel

    from pythonlib.tools.pandastools import stringify_values
    DFRES = stringify_values(DFRES)

    # FIlter input
    DFTHIS = DFRES[
        (DFRES["effect_context"] == effect_context) & (DFRES["dat_level"] == dat_level) & (DFRES["var_var_others"].isin(list_vvo))
    ]

    if len(DFTHIS)==0:
        return

    ######## JUST SUBSET OF BRAIN REGIONS
    # (1) Each subplot a var (overlaying areas)
    if bregions_plot is not None:
        dfthis = DFTHIS[(DFTHIS["bregion"].isin(bregions_plot))]
        if len(dfthis)>0:
            fig = sns.displot(data=dfthis, x=ythis, hue="bregion", col="var_var_others", col_wrap=4, element="step",
                              fill=True, bins=20)
            map_function_tofacet(fig, lambda ax: ax.axvline(0, color="k", alpha=0.4))
            rotateLabel(fig)
            savefig(fig, f"{savedir}/subset_bregions-subplot=var.pdf")

    ######### ALL REGIONS
    # (2) Each subplot = bregion
    fig = sns.displot(data=DFTHIS, x=ythis, hue="var_var_others", col="bregion", col_wrap=4, element="step", fill=True, bins=20)
    map_function_tofacet(fig, lambda ax: ax.axvline(0, color="k", alpha=0.5))
    rotateLabel(fig)
    savefig(fig, f"{savedir}/all_regions-subplot=region.pdf")

    # (3) Separate plots for each area and var
    fig = sns.displot(data=DFTHIS, x=ythis, col="bregion", row="var_var_others", element="step", fill=True, bins=20)
    map_function_tofacet(fig, lambda ax: ax.axvline(0, color="k", alpha=0.5))
    rotateLabel(fig)
    savefig(fig, f"{savedir}/all_regions-subplot=region_var.pdf")

    # (4) Show all bregions
    fig = sns.catplot(data=DFTHIS, x="bregion", y=ythis, col="var_var_others", jitter=True,
                      aspect=1.57, alpha=0.4, height=6)
    map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
    rotateLabel(fig)
    savefig(fig, f"{savedir}/all_regions-scatter.pdf")

    if False: # boxen is better
        fig = sns.catplot(data=DFTHIS, x="bregion", y=ythis, col="var_var_others", col_wrap=4,
                          aspect=1.57, kind="bar", errorbar=('ci', 68), height=6)
        map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
        rotateLabel(fig)
        savefig(fig, f"{savedir}/all_regions-bar.pdf")

    fig = sns.catplot(data=DFTHIS, x="bregion", y=ythis, col="var_var_others", col_wrap=4,
                      aspect=1.57, kind="boxen", height=6)
    map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
    rotateLabel(fig)
    savefig(fig, f"{savedir}/all_regions-boxen.pdf")

    # (5) Split by day
    fig = sns.catplot(data=DFTHIS, x="bregion", y=ythis, col="var_var_others", kind="point",
                      aspect=1.57, errorbar=('ci', 68), height=6, hue="date")
    map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.5))
    rotateLabel(fig)
    savefig(fig, f"{savedir}/all_regions-day_means.pdf")

    ##### OVERLAY ON BREGION
    from neuralmonkey.neuralplots.brainschematic import plot_df_from_longform
    savedirthis = f"{savedir}/brain_schematics"
    os.makedirs(savedirthis, exist_ok=True)
    plot_df_from_longform(DFTHIS, ythis, "var_var_others",
                          savedirthis)

    # Also plot with diff heat limits
    savedirthis = f"{savedir}/brain_schematics_each_var"
    os.makedirs(savedirthis, exist_ok=True)
    for vvo in DFTHIS["var_var_others"].unique().tolist():
        dfthis = DFTHIS[DFTHIS["var_var_others"]==vvo]
        plot_df_from_longform(dfthis, ythis, None,
                              savedirthis, savesuffix=vvo)
    plt.close("all")


def plot_all_results_time_trajectories(DFRES, SAVEDIR):
    """
    Run this in addition to plot_all_results, if this expt is taking
    distance between time-varying trajectories.

    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import compute_all_derived_metrics
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import summarize_featurediff
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.snstools import map_function_tofacet



    # Prepare DFRES
    if "shuffled" not in DFRES:
        DFRES["shuffled"] = False
    if "twind" not in DFRES:
        DFRES["twind"] = "dummy"
    if "event" not in DFRES:
        DFRES["event"] = "dummy"
    assert "shuffled_time" in DFRES, "this is not time-varying..."
    DFRES, _, _, _, plot_params = compute_all_derived_metrics(DFRES)
    DFRES = append_col_with_grp_index(DFRES, ["effect_samediff", "context_samediff", "shuffled_time"], "effect_context_shuff")
    
    # Always run this once, if in low-level plots.
    from pythonlib.tools.pandastools import stringify_values
    DFRES = stringify_values(DFRES)

    savedir = f"{SAVEDIR}/FIGURES_TIME_TRAJ"
    os.makedirs(savedir, exist_ok=True)

    # Sammary bar plots
    for vvo in DFRES["var_var_others"].unique():
        for y in ["dist", "dist_norm_95"]:
            dfthis = DFRES[DFRES["var_var_others"] == vvo]

            # fig = sns.catplot(data=dfthis, x="bregion", y=y, hue="effect_context", kind="bar", aspect=2, col="shuffled_time", 
            #             row="dat_level", sharey=False)

            fig = sns.catplot(data=dfthis, x="bregion", y=y, hue="effect_context_shuff", kind="bar", aspect=1.7, 
                        height=6, errorbar=('ci', 68), row="dat_level", sharey=False)
            rotateLabel(fig)
            map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))
            savefig(fig, f"{savedir}/overview_bar-vvo={vvo}-{y}.pdf")        
        plt.close("all")

    # Plot scatter (compare shuffled time to not)
    for dat_level in DFRES["dat_level"].unique():
        for vvo in DFRES["var_var_others"].unique():
            dfthis = DFRES[(DFRES["dat_level"]==dat_level) & (DFRES["var_var_others"]==vvo)]
            for y in ["dist", "dist_norm_95"]:
                _, fig = plot_45scatter_means_flexible_grouping(dfthis, "shuffled_time", True, False,
                                                    "effect_context", "dist", "bregion");
                if fig is not None:
                    savefig(fig, f"{savedir}/scatterbyshuff-dat_level={dat_level}-vvo={vvo}-y={y}.pdf")
        plt.close("all")
        

    # Summarize in 45 deg plot
    for vvo in DFRES["var_var_others"].unique():
        for shuff in [True, False]:
            for y in ["dist", "dist_norm_95"]:
                dfthis = DFRES[(DFRES["var_var_others"]==vvo) & (DFRES["shuffled_time"]==shuff)]
                _, fig = plot_45scatter_means_flexible_grouping(dfthis, "effect_context", "same|diff", "diff|same",
                                                    "dat_level", y, "bregion");
                if fig is not None:
                    savefig(fig, f"{savedir}/scatterbyvar-vvo={vvo}-shuff={shuff}-y={y}.pdf")
        plt.close("all")

    ######## NORMALIZING BY SUBTRACTING TIME-SHUFFLED
    if (True in DFRES["shuffled_time"].unique().tolist()) and (False in DFRES["shuffled_time"].unique().tolist()):
        # Only possible if you have extracted time-shuffled data
        # Normalize against time-shuffled
        #TODO: first, get all on same scale against the mean 95th
        # aggregate
        yvar = "dist_norm_95"
        dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = summarize_featurediff(DFRES, 
                                                        "shuffled_time", [True, False], 
                                                        [yvar], ["index_var", "var", "var_others", "var_var_others", 
                                                                "effect_samediff", "context_samediff", 
                                                                "effect_context", "leveff", "levo", "dat_level", "bregion",
                                                                "twind_analy", "twind", "event"])

        fig = sns.catplot(data=dfsummaryflat, x="bregion", y="value", hue="effect_context", kind="bar", aspect=2, 
                    col="var_var_others", row="dat_level", sharey=False)
        rotateLabel(fig)
        map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))
        savefig(fig, f"{savedir}/MINUSTIMESHUFF-overview_bar-y={yvar}.pdf")
        plt.close("all")
        
        for vvo in dfsummaryflat["var_var_others"].unique():
            dfthis = dfsummaryflat[dfsummaryflat["var_var_others"]==vvo]
            _, fig = plot_45scatter_means_flexible_grouping(dfthis, "effect_context", "same|diff", "diff|same",
                                                "dat_level", "value", "bregion")
            if fig is not None:
                savefig(fig, f"{savedir}/MINUSTIMESHUFF-scatter-vvo={vvo}-y={yvar}.pdf")
        plt.close("all")
        

def plot_all_results(DFRES, SAVEDIR):
    """
    Wrapper to make all main plots of reusults.
    :param DFRES:
    :param SAVEDIR:
    :return:
    """

    DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, plot_params = compute_all_derived_metrics(DFRES)
    return _plot_all_results(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, plot_params, SAVEDIR)

def _plot_all_results(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, plot_params, SAVEDIR,
                      ONLY_ESSENTIALS=False):
    """
    Low-level plotting (seee plot_all_results).
    :param DFRES:
    :param DFRES_PIVOT_DISTR:
    :param DFRES_PIVOT_PAIRWISE:
    :param plot_params:
    :param SAVEDIR:
    :return:
    """
    ######################################### QUICK PLOT - SUMMARIES
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.snstools import map_function_tofacet

    if "shuffled" not in DFRES:
        DFRES["shuffled"] = False

    # Always run this once, if in low-level plots.
    from pythonlib.tools.pandastools import stringify_values
    DFRES = stringify_values(DFRES)

    yvar = plot_params["yvar"]
    DIST_NULL = plot_params["DIST_NULL"]

    savedir = f"{SAVEDIR}/FIGURES"
    os.makedirs(savedir, exist_ok=True)

    list_yvar_dat_level = [
        [DIST_NULL, "dist"],
        ["dist_norm_95", "distr"],
        ["dist_norm_95", "pts"],
        # ["dist", "pts_yue_log"],
        ["dist_norm_95", "pts_yue_diff"],
    ]

    ########## OVERVIEWS
    print("..Plotting overview bar and scatter plots")
    for yvarthis, dat_level in list_yvar_dat_level:
        dfthis = DFRES[DFRES["dat_level"]==dat_level]
        if len(dfthis)>0:
            if ONLY_ESSENTIALS:
                if not yvarthis == yvar:
                    continue
                if not dat_level in ["pts_yue_diff", "distr"]:
                    continue

            fig = sns.catplot(data=dfthis, x="bregion", y=yvarthis, col="var_var_others", hue="effect_context",
                              col_wrap=3, aspect=1.57, alpha=0.4, height=6)
            rotateLabel(fig)
            if dat_level=="pts_yue":
                map_function_tofacet(fig, lambda ax: ax.axhline(1, color="k", alpha=0.4))
            else:
                map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))
            savefig(fig, f"{savedir}/overview_scatter-{dat_level}-{yvarthis}.pdf")

            fig = sns.catplot(data=dfthis, x="bregion", y=yvarthis, col="var_var_others", hue="effect_context",
                              col_wrap=3, aspect=1.7, kind="bar", height=6, errorbar=('ci', 68))
            rotateLabel(fig)
            savefig(fig, f"{savedir}/overview_bar-{dat_level}-{yvarthis}.pdf")

            plt.close("all")

    ########## OVERVIEWS (OLD - effect index)
    if False: # Never checekd.
        if not ONLY_ESSENTIALS:
            yvarthis = "effect_index"
            fig = sns.catplot(data=DFRES_PIVOT_DISTR, x="bregion", y=yvarthis, hue="var_var_others",  aspect=1.7, col="dat_level",
                              height=6, kind="bar", errorbar=('ci', 68))
            rotateLabel(fig)
            savefig(fig, f"{savedir}/effect_index-bar.pdf")

    ########## OVERVIEWS (dat_level = pts)
    if ONLY_ESSENTIALS:
        list_yvarthis = ["gen_idx_diff_3", "gen_idx_diff_5",
                     "gen_idx_ratio_1", "gen_idx_ratio_2", "gen_idx_ratio_3", "gen_idx_ratio_5"]
    else:
        list_yvarthis = ["norm_dist_effect", "norm_dist_context", "norm_dist_both", "generalization_index", "generalization_index_scaled",
                     "gen_idx_diff_1", "gen_idx_diff_2", "gen_idx_diff_3", "gen_idx_diff_4", "gen_idx_diff_5",
                     "gen_idx_ratio_1", "gen_idx_ratio_2", "gen_idx_ratio_3", "gen_idx_ratio_4", "gen_idx_ratio_5"]

    for yvarthis in list_yvarthis:
        if yvarthis in DFRES_PIVOT_PAIRWISE.columns:

            print("..Plotting pairwise for one specific pair of variables:")
            print(yvarthis)

            if not ONLY_ESSENTIALS:
                fig = sns.catplot(data=DFRES_PIVOT_PAIRWISE, x="bregion", y=yvarthis, hue="var_var_others",  aspect=1.7, height=6,
                                  kind="bar", errorbar=('ci', 68))
                rotateLabel(fig)
                savefig(fig, f"{savedir}/FINAL-{yvarthis}-bar.pdf")

            # Also plot splitting by yvar
            fig = sns.catplot(data=DFRES_PIVOT_PAIRWISE, x="bregion", y=yvarthis, col="var_var_others",
                              col_wrap = 6, aspect=1, height=6, kind="bar", sharey=False, errorbar=('ci', 68))
            rotateLabel(fig)
            savefig(fig, f"{savedir}/FINAL-{yvarthis}-bar-splitby_yvar.pdf")

            # Also plot splitting by bregion
            if not ONLY_ESSENTIALS:
                fig = sns.catplot(data=DFRES_PIVOT_PAIRWISE, x="var_var_others", y=yvarthis, col="bregion",
                                  col_wrap = 6, aspect=1, height=6, kind="bar", errorbar=('ci', 68))
                rotateLabel(fig)
                savefig(fig, f"{savedir}/FINAL-{yvarthis}-bar-splitby_bregion.pdf")
        plt.close("all")

    ########### PLOT ALL specific conjunction levels in heatmaps
    if not ONLY_ESSENTIALS:
        sns.set_context("paper", rc={"axes.labelsize":5})

        # list_yvar_dat_level = [
        #     [DIST_NULL, "dist"],
        #     ["dist_norm_95", "distr"],
        #     ["dist_norm_95", "pts"],
        #     ["dist", "pts_yue_log"],
        #     ["dist_norm_95", "pts_yue_diff"],
        # ]
        list_yvar_dat_level = [
            ["dist_norm_95", "pts_yue_diff"],
        ]
        for yvarthis, dat_level in list_yvar_dat_level:
        # for dat_level in DFRES["dat_level"].unique():
            DFTHIS = DFRES[DFRES["dat_level"] == dat_level].reset_index(drop=True)

            # Plot histograms
            if False:
                # not usefl
                savedirthis = f"{savedir}/histograms-dat_level={dat_level}-yvar={yvarthis}"
                os.makedirs(savedirthis, exist_ok=True)
                print("... ", savedirthis)

                fig = sns.displot(data=DFTHIS, x=yvarthis, hue="effect_context", col="bregion", row="var_var_others", element="step", fill=True, bins=20)
                savefig(fig, f"{savedirthis}/step.pdf")
                fig = sns.displot(data=DFTHIS, x=yvarthis, hue="effect_context", col="bregion", row="var_var_others", kind="kde", fill=False)
                savefig(fig, f"{savedirthis}/kde.pdf")

            # Plot heatmaps
            print("Plotting specific conjucntions heatmaps ... ")
            list_effect_context = DFTHIS["effect_context"].unique()
            list_shuffled = DFTHIS["shuffled"].unique()
            for effect_context in list_effect_context:
                for shuffled in list_shuffled:

                    dfthis = DFTHIS[(DFTHIS["effect_context"]==effect_context) & (DFTHIS["shuffled"]==shuffled)].reset_index(drop=True)
                    if len(dfthis)>0:
                        savedirthis = f"{savedir}/each_conjunction-effect_context={effect_context}-shuffled={shuffled}-dat_level={dat_level}-yvar={yvarthis}"
                        os.makedirs(savedirthis, exist_ok=True)
                        print("... ", savedirthis)

                        # # 1) Scatter
                        # list_vvo = dfthis["var_var_others"].unique().tolist()
                        # for vvo in list_vvo:
                        #     dfthisthis = dfthis[dfthis["var_var_others"]==vvo]
                        #     fig = sns.catplot(data=dfthisthis, x=yvar, y="levo", col="bregion", alpha=0.4)
                        #     savefig(fig, f"{savedirthis}/allconj_scatter-vvo={vvo}.pdf", height=6)
                        #     plt.close("all")

                        # 2) Heatmap
                        try:
                            fig, axes = plot_subplots_heatmap(dfthis, "bregion", "levo", yvarthis, "var_var_others",
                                                              diverge=True, ncols=None, share_zlim=True)
                            savefig(fig, f"{savedirthis}/allconj_heatmap.pdf")
                            plt.close("all")
                        except Exception as err:
                            print(dfthis["levo"])
                            print(dfthis["var_var_others"])
                            raise err

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])
    question = sys.argv[3]
    which_level = sys.argv[4]
    # if len(sys.argv)>5:
    combine_into_larger_areas = int(sys.argv[5])==1
    # else:
        # combine_into_larger_areas = False
    assert len(sys.argv)<7, "you made mistake with args"

    # LIST_TWIND = _get_list_twind_by_animal(animal)

    ############## HIGH-LEVEL PARAMS
    # TRAJECTORIES_METHOD = "scalar"
    TRAJECTORIES_METHOD = "traj_to_scalar"
    # TRAJECTORIES_METHOD = "traj_to_timeseries"

    HACK_TRAJS_JUST_FOR_PLOTTING_NICE = False # To plot with wider time windows, clean version of state space plots, and wont spend time computing euclidian distance.

    if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
        COMPUTE_EUCLIDIAN = False
        PLOT_CLEAN_VERSION = True
    else:
        COMPUTE_EUCLIDIAN = True
        PLOT_CLEAN_VERSION = False

    ############### PARAMS (EUCLIDIAN)
    PLOT = False
    PLOT_MASKS = False
    PLOT_CHUNK_END_STATE_ALIGNMENT = False
    LOAD_AND_PLOT_RESULTS_ONLY = False
    SKIP_EUCL_PLOTS = False

    ############### PARAMS
    PLOT_STATE_SPACE = True
    # NPCS_KEEP = None # use auto method

    DEBUG = False
    # SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
    # combine_into_larger_areas = True
    # if combine_into_larger_areas:
    SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
    # else:
    #     SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks

    nmin_trials_per_lev = 4 # e.g. some cases liek (shape x loc x size)...
    LIST_FR_NORMALIZATION = ["across_time_bins"]

    # exclude_bad_areas = True
    exclude_bad_areas = True
    list_time_windows = [(-0.8, 1.2)]
    EVENTS_IGNORE = [] # To reduce plots

    # DONT COMBINE, use questions.
    combine_trial_and_stroke = False
    dir_suffix = question

    # # HACKY - hard code event...
    # if which_level in ["stroke", "stroke_off"]:
    #     EVENT_KEEP = "00_stroke"
    # elif which_level == "trial":
    #     EVENT_KEEP = "03_samp"
    #     # EVENT_KEEP = "06_on_strokeidx_0"
    # else:
    #     assert False

    # Final version, this is best
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question)

    assert len(LIST_VAR)==len(LIST_VARS_OTHERS)
    # Convert from tuple to list
    LIST_VARS_OTHERS = [list(var_others) for var_others in LIST_VARS_OTHERS]
    LIST_VAR = [list(var) if isinstance(var, tuple) else var for var in LIST_VAR]

    if False:
        if len(sys.argv)>5:
            # Test hypoerparams, e.g, iterate over diff ways to do dim reduction.
            HYPERPARAM_MODE = int(sys.argv[5])==1
        else:
            HYPERPARAM_MODE = False
    else:
        HYPERPARAM_MODE = True    
    
    ###############################################################
    ###############################################################
    ## DIM REDUCTION PARAMS
    if TRAJECTORIES_METHOD == "traj_to_scalar":
        # Very hacky, 4/29/24 - to try out traj methods, for SP and PIG
        LIST_DIMRED_METHODS = [
            # ["pca", 6, None, None, True, None, None]
            # ["pca", 8, None, None, True, None, None]
        ]
        
        # - Append dPCA params
        LIST_SAVEDIR_SUFFIX = []
        LIST_SUPERV_DPCA_PARAMS = []
        # NPCS_KEEP = 3
        NPCS_KEEP = 6
        
        if question in ["SP_shape_loc", "SP_BASE_trial"]:

            # - Append dPCA params
            LIST_SAVEDIR_SUFFIX = []
            LIST_SUPERV_DPCA_PARAMS = []
            NPCS_KEEP = 3

            # (1) gridloc
            savedir_suffix = f"seqc_0_loc"
            superv_dpca_params = {
                "superv_dpca_var":"seqc_0_loc",
                "superv_dpca_vars_group":["seqc_0_shape", "gridsize"],
                "superv_dpca_filtdict":None
            }
            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            savedir_suffix = f"seqc_0_shape"
            superv_dpca_params = {
                "superv_dpca_var":"seqc_0_shape",
                "superv_dpca_vars_group":["seqc_0_loc", "gridsize"],
                "superv_dpca_filtdict":None
            }
            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        elif question in ["SP_BASE_stroke"]:
            savedir_suffix = f"shape"
            superv_dpca_params = {
                "superv_dpca_var":"shape",
                "superv_dpca_vars_group":["gridloc", "gridsize"],
                "superv_dpca_filtdict":None
            }
            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # (1) gridloc
            savedir_suffix = f"gridloc"
            superv_dpca_params = {
                "superv_dpca_var":"gridloc",
                "superv_dpca_vars_group":["shape", "gridsize"],
                "superv_dpca_filtdict":None
            }
            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)
            
        elif question in ["PIG_BASE_stroke"]:

            savedir_suffix = f"shape_prims_single"
            superv_dpca_params={
                "superv_dpca_var":"shape",
                "superv_dpca_vars_group":["gridloc"],
                "superv_dpca_filtdict":{"task_kind":["prims_single"]}
            }

            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # TEst generlaization across tgask_kind
            LIST_VAR = [
                "shape",
            ]
            LIST_VARS_OTHERS = [
                ["task_kind", "stroke_index", "gridloc"],
            ]
            LIST_CONTEXT = [
                {"same":["gridloc"], "diff":["task_kind"]}
            ]
            LIST_PRUNE_MIN_N_LEVS = [2]
            LIST_FILTDICT = [None]

            assert False, "OLDER stuff, moved here, merge with above"
            # savedir_suffix = f"shape"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"shape",
            #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "gridloc", "stroke_index"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"stroke_index"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"stroke_index",
            #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "gridloc", "shape"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"gridloc"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"gridloc",
            #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "stroke_index", "shape"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"CTXT_loc_next"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"CTXT_loc_next",
            #     "superv_dpca_vars_group":["task_kind", "stroke_index", "CTXT_loc_prev", "shape", "gridloc"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"CTXT_shape_next"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"CTXT_shape_next",
            #     "superv_dpca_vars_group":["task_kind", "stroke_index", "CTXT_loc_prev", "shape", "gridloc"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # ###############
            # LIST_VAR = [
            #     "CTXT_shape_next",
            #     "CTXT_loc_next",
            #     "shape",
            #     "gridloc",
            #     "stroke_index",
            #     "task_kind",
            # ]
            # # More restrictive
            # LIST_VARS_OTHERS = [
            #     ["task_kind", "stroke_index_is_first", "CTXT_loc_prev", "shape", "gridloc"],
            #     ["task_kind", "stroke_index_is_first", "CTXT_loc_prev", "shape", "gridloc"],
            #     ["task_kind", "stroke_index_is_first", "gridloc"],
            #     ["task_kind", "stroke_index_is_first", "shape"],
            #     ["task_kind", "shape", "gridloc"],
            #     ["CTXT_loc_prev", "shape", "gridloc"],
            # ]

            # LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]

            # LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]

            # filtdict = None
            # LIST_FILTDICT = [
            #     filtdict for _ in range(len(LIST_VAR))
            # ]
            
                        
        elif question in ["CHAR_BASE_stroke"]:

            # Char  
            if False: # OK, but removed to run things quicker
                
                # Char  
                savedir_suffix = f"shape_char_strokesothers"
                dim_red_method = "superv_dpca"
                superv_dpca_params={
                    "superv_dpca_var":"shape_semantic",
                    "superv_dpca_vars_group":["task_kind"],
                    "superv_dpca_filtdict":{"task_kind":["character"], "stroke_index":[1,2,3,4,5,6,7,8]}
                }

                LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            assert False, "use shape_semantic_grp instead of shape_semantic"
            savedir_suffix = f"shape_prims_single"
            dim_red_method = "superv_dpca"
            superv_dpca_params={
                "superv_dpca_var":"shape_semantic",
                "superv_dpca_vars_group":["task_kind"],
                "superv_dpca_filtdict":{"task_kind":["prims_single"]}
            }

            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # PIG (0)  
            savedir_suffix = f"shape_PIG_stroke0"
            dim_red_method = "superv_dpca"
            superv_dpca_params={
                "superv_dpca_var":"shape_semantic",
                "superv_dpca_vars_group":["task_kind"],
                "superv_dpca_filtdict":{"task_kind":["prims_on_grid"], "stroke_index":[0]}
            }

            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # Char  
            savedir_suffix = f"shape_char_stroke0"
            dim_red_method = "superv_dpca"
            superv_dpca_params={
                "superv_dpca_var":"shape_semantic",
                "superv_dpca_vars_group":["task_kind"],
                "superv_dpca_filtdict":{"task_kind":["character"], "stroke_index":[0]}
            }

            LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)


            # TEst generlaization across tgask_kind
            LIST_VAR = [
                "shape_semantic",
                "shape_semantic",
                "shape_semantic",
                "shape_semantic",

                "shape_semantic",
                "shape_semantic",
                "shape_semantic",
                "shape_semantic",
            ]
            LIST_VARS_OTHERS = [
                ["task_kind"],
                ["task_kind", "stroke_index"],
                ["task_kind", "stroke_index"],
                ["task_kind", "loc_on_clust"],

                ["task_kind"],
                ["task_kind"],
                ["task_kind", "stroke_index"],
                ["task_kind", "stroke_index"],
            ]
            LIST_CONTEXT = [
                {"same":[], "diff":["task_kind"]},
                {"same":["stroke_index"], "diff":["task_kind"]},
                {"same":["stroke_index"], "diff":["task_kind"]},
                {"same":["loc_on_clust"], "diff":["task_kind"]},

                {"same":[], "diff":["task_kind"]},
                {"same":[], "diff":["task_kind"]},
                {"same":["stroke_index"], "diff":["task_kind"]},
                {"same":["stroke_index"], "diff":["task_kind"]},
            ]
            LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
            LIST_FILTDICT = [
                {"task_kind":["prims_single", "character"]},
                {"task_kind":["prims_single", "character"], "stroke_index":[0]},
                {"task_kind":["prims_single", "character"]}, # Just for visualization of char on other stroke indices
                {"task_kind":["prims_single", "character"]},

                {"task_kind":["prims_on_grid", "character"]},
                {"task_kind":["prims_on_grid", "character"], "stroke_index":[1,2,3,4,5,6,7]}, 
                {"task_kind":["prims_on_grid", "character"], "stroke_index":[0]},
                {"task_kind":["prims_on_grid", "character"]}, 
                ]
            nmin_trials_per_lev = 3 # chracters, not many trials for each shape.

            # TODO: "OLDER stuff, moved here, merge with above"
            # savedir_suffix = f"shape"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"shape",
            #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "gridloc", "stroke_index"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"stroke_index"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"stroke_index",
            #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "gridloc", "shape"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"gridloc"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"gridloc",
            #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "stroke_index", "shape"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"CTXT_loc_next"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"CTXT_loc_next",
            #     "superv_dpca_vars_group":["task_kind", "stroke_index", "CTXT_loc_prev", "shape", "gridloc"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # savedir_suffix = f"CTXT_shape_next"
            # # (1) var rank, condition on everything else.
            # superv_dpca_params = {
            #     "superv_dpca_var":"CTXT_shape_next",
            #     "superv_dpca_vars_group":["task_kind", "stroke_index", "CTXT_loc_prev", "shape", "gridloc"],
            #     "superv_dpca_filtdict":None
            # }
            # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            # ###############
            # LIST_VAR = [
            #     "CTXT_shape_next",
            #     "CTXT_loc_next",
            #     "shape",
            #     "gridloc",
            #     "stroke_index",
            #     "task_kind",
            # ]
            # # More restrictive
            # LIST_VARS_OTHERS = [
            #     ["task_kind", "stroke_index_is_first", "CTXT_loc_prev", "shape", "gridloc"],
            #     ["task_kind", "stroke_index_is_first", "CTXT_loc_prev", "shape", "gridloc"],
            #     ["task_kind", "stroke_index_is_first", "gridloc"],
            #     ["task_kind", "stroke_index_is_first", "shape"],
            #     ["task_kind", "shape", "gridloc"],
            #     ["CTXT_loc_prev", "shape", "gridloc"],
            # ]

            # LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]

            # LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]

            # filtdict = None
            # LIST_FILTDICT = [
            #     filtdict for _ in range(len(LIST_VAR))
            # ]
            
        elif question in ["RULE_ANBMCK_STROKE", "RULESW_ANY_SEQSUP_STROKE"]:
            SYNTAX_PARAMS_VERSION = "default"
            if SYNTAX_PARAMS_VERSION == "default":
                if False:
                    # chunk_within_rank
                    savedir_suffix = f"chunk_within_rank_semantic_v2"
                    # (1) var rank, condition on everything else.
                    superv_dpca_params = {
                        "superv_dpca_var":"chunk_within_rank_semantic_v2",
                        "superv_dpca_vars_group":["epoch", "chunk_rank", "syntax_concrete", "shape"],
                        "superv_dpca_filtdict":None
                    }
                    LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                    LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                    # chunk_within_rank
                    savedir_suffix = f"chunk_within_rank_semantic_v2_gridloc"
                    # (1) var rank, condition on everything else.
                    superv_dpca_params = {
                        "superv_dpca_var":"chunk_within_rank_semantic_v2",
                        "superv_dpca_vars_group":["epoch", "chunk_rank", "shape", "syntax_concrete", "gridloc"],
                        "superv_dpca_filtdict":None
                    }
                    LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                    LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                    # (3) shape
                    savedir_suffix = f"shape"
                    superv_dpca_params = {
                        "superv_dpca_var":"shape",
                        "superv_dpca_vars_group":["epoch", "chunk_within_rank_semantic", "syntax_concrete"], # This seems better...
                        "superv_dpca_filtdict":None
                    }
                    LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                    LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                    # (3) shape
                    savedir_suffix = f"shape_gridloc"
                    superv_dpca_params = {
                        "superv_dpca_var":"shape",
                        "superv_dpca_vars_group":["epoch", "chunk_within_rank_semantic", "gridloc"],
                        "superv_dpca_filtdict":None
                    }
                    LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                    LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                    # # - Supervised
                    savedir_suffix = f"syntax_role_gridloc"
                    # (1) var rank, condition on everything else.
                    superv_dpca_params = {
                        "superv_dpca_var":"syntax_role",
                        "superv_dpca_vars_group":["epoch", "syntax_concrete", "gridloc"],
                        "superv_dpca_filtdict":None
                    }
                    LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                    LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                    for cr in [0,1,2]:
                        savedir_suffix = f"chunk_within_rank_{cr}"
                        superv_dpca_params = {
                            "superv_dpca_var":"chunk_within_rank_semantic_v2",
                            "superv_dpca_vars_group":["epoch", "shape", "syntax_concrete"],
                            "superv_dpca_filtdict":{"chunk_rank":[cr]}
                        }
                        LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                        LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                # # - Supervised
                savedir_suffix = f"syntax_role"
                # (1) var rank, condition on everything else.
                superv_dpca_params = {
                    "superv_dpca_var":"syntax_role",
                    "superv_dpca_vars_group":["epoch", "syntax_concrete"],
                    "superv_dpca_filtdict":None
                }
                LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                HACKTHIS = False
                if HACKTHIS:
                    ############### QUICK HACK, TO PLOT STATE SPACE FOR GOOD EXAMPLE DAYS (before lab meeting, 5/17, mainly goal 
                    # is to have plots that split by n_in_chunk). 
                    # THis removes all the range of plots.
                    PLOT_STATE_SPACE = True
                    COMPUTE_EUCLIDIAN = False

                    LIST_VAR = [
                        "shape",
                        "chunk_within_rank_semantic",
                        "chunk_within_rank_fromlast",
                        "chunk_within_rank_fromlast",
                        "chunk_within_rank",
                        "chunk_within_rank_fromlast",
                    ]
                    # More restrictive
                    LIST_VARS_OTHERS = [
                        ["epoch", "chunk_within_rank_semantic"],
                        ["epoch", "chunk_rank", "shape"],
                        ["epoch", "chunk_rank", "shape", "syntax_concrete"],
                        ["epoch", "chunk_rank", "shape", "syntax_concrete", "behseq_locs_clust"],
                        ["epoch", "chunk_rank", "shape", "chunk_n_in_chunk"],
                        ["epoch", "chunk_rank", "shape", "chunk_n_in_chunk"],
                    ]

                    LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]
                    LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]
                    filtdict = None
                    LIST_FILTDICT = [
                        filtdict for _ in range(len(LIST_VAR))
                    ]
                    ############################### (end HACK)

                    # # Append PCA
                    # LIST_DIMRED_METHODS.append(["pca", 10, None, None, PLOT_STATE_SPACE, None, None])

                # Append dPCA
                for savedir_suffix, superv_dpca_params in zip(LIST_SAVEDIR_SUFFIX, LIST_SUPERV_DPCA_PARAMS):
                    LIST_DIMRED_METHODS.append(["superv_dpca", NPCS_KEEP, None, None, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params])

            elif SYNTAX_PARAMS_VERSION == "end_state_alignment":
                # 4/24/24 just to focus of question of alignemnet to end

                # This is best?
                savedir_suffix = f"chunk_within_rank_semantic_v3"
                # (1) var rank, condition on everything else.
                superv_dpca_params = {
                    "superv_dpca_var":"chunk_within_rank_semantic_v3",
                    "superv_dpca_vars_group":["epoch", "chunk_rank", "shape", "syntax_concrete", "gridloc"],
                    "superv_dpca_filtdict":None
                }
                LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
                LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

                LIST_VAR = [
                    "chunk_within_rank_semantic_v3",
                    "chunk_within_rank_semantic",
                    "chunk_within_rank",
                    "chunk_within_rank_fromlast",
                ]
                # More restrictive
                LIST_VARS_OTHERS = [
                    ["chunk_rank", "shape", "chunk_n_in_chunk"],
                    ["chunk_rank", "shape", "chunk_n_in_chunk"],
                    ["chunk_rank", "shape", "chunk_n_in_chunk"],
                    ["chunk_rank", "shape", "chunk_n_in_chunk"],
                ]

                LIST_CONTEXT = [
                    {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
                    {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
                    {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
                    {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
                    ]

                LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
                # Use 1 for things that use syntax role as effect. or else will throw out cases with 1 item in given chunk.

                filtdict = {
                    "stroke_index": list(range(1, 10, 1)), # [1, ..., ]
                }
                LIST_FILTDICT = [
                    filtdict for _ in range(len(LIST_VAR))
                ]

                ########
                PLOT_CHUNK_END_STATE_ALIGNMENT = True
        else:
            print("Code it for this question: ", question)
            assert False

        for savedir_suffix, superv_dpca_params in zip(LIST_SAVEDIR_SUFFIX, LIST_SUPERV_DPCA_PARAMS):
            LIST_DIMRED_METHODS.append(["superv_dpca", NPCS_KEEP, None, None, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params])

    # else:
    #     if True: # 5/16/24 - prep for lab meeting, want to run all Euclidian distances, but first doing dpca to syntax role.

    #         LIST_SAVEDIR_SUFFIX = []
    #         LIST_SUPERV_DPCA_PARAMS = []
    #         LIST_DIMRED_METHODS = []
    #         NPCS_KEEP = 6
    #         PLOT_STATE_SPACE = True
    #         COMPUTE_EUCLIDIAN = True
            
            # if question in ["RULE_ANBMCK_STROKE", "RULESW_ANY_SEQSUP_STROKE"]:

            #     # # - Supervised
            #     savedir_suffix = f"syntax_role"
            #     # (1) var rank, condition on everything else.
            #     superv_dpca_params = {
            #         "superv_dpca_var":"syntax_role",
            #         "superv_dpca_vars_group":["epoch", "syntax_concrete"],
            #         "superv_dpca_filtdict":None
            #     }
            #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            #     HACKTHIS = False
            #     if HACKTHIS:
            #         ############### QUICK HACK, TO PLOT STATE SPACE FOR GOOD EXAMPLE DAYS (before lab meeting, 5/17, mainly goal 
            #         # is to have plots that split by n_in_chunk)
            #         PLOT_STATE_SPACE = True
            #         COMPUTE_EUCLIDIAN = False

            #         LIST_VAR = [
            #             "shape",
            #             "chunk_within_rank_semantic",
            #             "chunk_within_rank_fromlast",
            #             "chunk_within_rank",
            #             "chunk_within_rank_fromlast",
            #         ]
            #         # More restrictive
            #         LIST_VARS_OTHERS = [
            #             ["epoch", "chunk_within_rank_semantic"],
            #             ["epoch", "chunk_rank", "shape"],
            #             ["epoch", "chunk_rank", "shape", "syntax_concrete"],
            #             ["epoch", "chunk_rank", "shape", "chunk_n_in_chunk"],
            #             ["epoch", "chunk_rank", "shape", "chunk_n_in_chunk"],
            #         ]

            #         LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]
            #         LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]
            #         filtdict = None
            #         LIST_FILTDICT = [
            #             filtdict for _ in range(len(LIST_VAR))
            #         ]
            #         ############################### (end HACK)

            #         # # Append PCA
            #         # LIST_DIMRED_METHODS.append(["pca", 10, None, None, PLOT_STATE_SPACE, None, None])

            #     # Append dPCA
            #     for savedir_suffix, superv_dpca_params in zip(LIST_SAVEDIR_SUFFIX, LIST_SUPERV_DPCA_PARAMS):
            #         LIST_DIMRED_METHODS.append(["superv_dpca", NPCS_KEEP, None, None, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params])

            # if question in ["SP_shape_loc", "SP_BASE_trial"]:
            #     # Very hacky
            #     LIST_DIMRED_METHODS = [
            #         ["pca", 6, None, None, True, None, None]
            #     ]
                
            #     # - Append dPCA params
            #     LIST_SAVEDIR_SUFFIX = []
            #     LIST_SUPERV_DPCA_PARAMS = []
            #     NPCS_KEEP = 3

            #     savedir_suffix = f"seqc_0_shape"
            #     superv_dpca_params = {
            #         "superv_dpca_var":"seqc_0_shape",
            #         "superv_dpca_vars_group":["seqc_0_loc", "gridsize"],
            #         "superv_dpca_filtdict":None
            #     }
            #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            #     # (1) gridloc
            #     savedir_suffix = f"seqc_0_loc"
            #     superv_dpca_params = {
            #         "superv_dpca_var":"seqc_0_loc",
            #         "superv_dpca_vars_group":["seqc_0_shape", "gridsize"],
            #         "superv_dpca_filtdict":None
            #     }
            #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
            #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

            #     for savedir_suffix, superv_dpca_params in zip(LIST_SAVEDIR_SUFFIX, LIST_SUPERV_DPCA_PARAMS):
            #         LIST_DIMRED_METHODS.append(["superv_dpca", NPCS_KEEP, None, None, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params])



            # else:
            #     print(question)
            #     assert False, "you prob dont want to run this hacky params"

            # assert len(LIST_DIMRED_METHODS)>0
        # elif False:
        #     if HYPERPARAM_MODE:
        #         if TRAJECTORIES_METHOD =="traj_to_scalar":
        #             LIST_DIMRED_METHODS = [ # (dim_red_method, NPCS_KEEP, extra_dimred_method_n_components, umap_n_neighbors, PLOT_STATE_SPACEC)
        #                 ["pca", 6, None, None, PLOT_STATE_SPACE], # pca (low D)
        #             ]
        #         elif TRAJECTORIES_METHOD == "scalar":
        #             # LIST_DIMRED_METHODS = [ # (dim_red_method, NPCS_KEEP, extra_dimred_method_n_components, umap_n_neighbors, PLOT_STATE_SPACEC)
        #             #     # ["pca", None, None, None, False], # pca (high D)
        #             #     ["pca", 10, None, None, True], # pca (low D)
        #             #     # ["pca", 4, None, None, False], # pca (low D)
        #             #     # ["pca_umap", 10, 2, 40], # pca --> umap [old version]
        #             #     # ["pca_umap", None, 2, 40], # pca --> umap [new version]
        #             #     ["pca_umap", None, 4, 40, False], # pca --> umap [new version]
        #             #     # ["umap", None, 6, 40, False], # umap (high D)
        #             #     # ["umap", None, 2, 40, False], # umap (low D)
        #             #     # ["umap", None, 4, 40, True], # umap (high D)
        #             #     # [None, None, None, None, False], # raw data
        #             # ]
        #             LIST_DIMRED_METHODS = [ # (dim_red_method, NPCS_KEEP, extra_dimred_method_n_components, umap_n_neighbors, PLOT_STATE_SPACEC)
        #                 ["pca", 10, None, None, PLOT_STATE_SPACE], # pca (low D)
        #             ]
        #         else:
        #             assert False
        #     else:
        #         # The best params, for main run
        #         LIST_DIMRED_METHODS = [ # (dim_red_method, NPCS_KEEP, extra_dimred_method_n_components, umap_n_neighbors)
        #             ["pca_umap", 10, 2, 40, True], # pca --> umap [old version]
        #     ]

        #     LIST_DIMRED_METHODS = [x + [None, None] for x in LIST_DIMRED_METHODS]
        # else:
        #     ########### DPCA
        #     LIST_SAVEDIR_SUFFIX = []
        #     LIST_SUPERV_DPCA_PARAMS = []
        #     LIST_DIMRED_METHODS = []

        #     if question in ["RULE_ANBMCK_STROKE"]:

        #         # if False: # Original set -- still good.
        #             #  # chunk_within_rank
        #             # savedir_suffix = f"chunk_within_rank_semantic_v2"
        #             # # (1) var rank, condition on everything else.
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"chunk_within_rank_semantic_v2",
        #             #     "superv_dpca_vars_group":["epoch", "chunk_rank", "syntax_concrete", "shape"],
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # # chunk_within_rank
        #             # savedir_suffix = f"chunk_within_rank_semantic_v2_gridloc"
        #             # # (1) var rank, condition on everything else.
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"chunk_within_rank_semantic_v2",
        #             #     "superv_dpca_vars_group":["epoch", "chunk_rank", "shape", "syntax_concrete", "gridloc"],
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # # (3) shape
        #             # savedir_suffix = f"shape"
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"shape",
        #             #     "superv_dpca_vars_group":["epoch", "chunk_within_rank_semantic", "syntax_concrete"], # This seems better...
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # # (3) shape
        #             # savedir_suffix = f"shape_gridloc"
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"shape",
        #             #     "superv_dpca_vars_group":["epoch", "chunk_within_rank_semantic", "gridloc"],
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # # # - Supervised
        #             # savedir_suffix = f"syntax_role"
        #             # # (1) var rank, condition on everything else.
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"syntax_role",
        #             #     "superv_dpca_vars_group":["epoch", "syntax_concrete"],
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # # # - Supervised
        #             # savedir_suffix = f"syntax_role_gridloc"
        #             # # (1) var rank, condition on everything else.
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"syntax_role",
        #             #     "superv_dpca_vars_group":["epoch", "syntax_concrete", "gridloc"],
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             ################## AnBmCk [good]
        #             # LIST_VAR = [
        #             #     "shape",
        #             #     "chunk_within_rank_semantic",
        #             #     "chunk_within_rank_fromlast",
        #             #     "chunk_within_rank_fromlast",
        #             #     "chunk_within_rank",
        #             #     "chunk_within_rank_fromlast",
        #             # ]
        #             # # More restrictive
        #             # LIST_VARS_OTHERS = [
        #             #     ["epoch", "chunk_within_rank_semantic"],
        #             #     ["epoch", "chunk_rank", "shape"],
        #             #     ["epoch", "chunk_rank", "shape", "syntax_concrete"],
        #             #     ["epoch", "chunk_rank", "shape", "syntax_concrete", "behseq_locs_clust"],
        #             #     ["epoch", "chunk_rank", "shape", "chunk_n_in_chunk"],
        #             #     ["epoch", "chunk_rank", "shape", "chunk_n_in_chunk"],
        #             # # ]

        #             # for cr in [0,1,2]:
        #             #     savedir_suffix = f"chunk_within_rank_{cr}"
        #             #     superv_dpca_params = {
        #             #         "superv_dpca_var":"chunk_within_rank_semantic_v2",
        #             #         "superv_dpca_vars_group":["epoch", "shape", "syntax_concrete"],
        #             #         "superv_dpca_filtdict":{"chunk_rank":[cr]}
        #             #     }
        #             #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]
        #             # LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]
        #             # Use 1 for things that use syntax role as effect. or else will throw out cases with 1 item in given chunk.

        #             # filtdict = {
        #             #     # "stroke_index": list(range(1, 10, 1)), # [1, ..., ]
        #             # }
        #             # filtdict = {"task_kind":["character"]}
        #             # filtdict = None
        #             # LIST_FILTDICT = [
        #             #     filtdict for _ in range(len(LIST_VAR))
        #             # ]

        #         # else:
        #             # 4/24/24 just to focus of question of alignemnet to end

        #             # # This is best?
        #             # savedir_suffix = f"chunk_within_rank_semantic_v3"
        #             # # (1) var rank, condition on everything else.
        #             # superv_dpca_params = {
        #             #     "superv_dpca_var":"chunk_within_rank_semantic_v3",
        #             #     "superv_dpca_vars_group":["epoch", "chunk_rank", "shape", "syntax_concrete", "gridloc"],
        #             #     "superv_dpca_filtdict":None
        #             # }
        #             # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #             # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #             # LIST_VAR = [
        #             #     "chunk_within_rank_semantic_v3",
        #             #     "chunk_within_rank_semantic",
        #             #     "chunk_within_rank",
        #             #     "chunk_within_rank_fromlast",
        #             # ]
        #             # # More restrictive
        #             # LIST_VARS_OTHERS = [
        #             #     ["chunk_rank", "shape", "chunk_n_in_chunk"],
        #             #     ["chunk_rank", "shape", "chunk_n_in_chunk"],
        #             #     ["chunk_rank", "shape", "chunk_n_in_chunk"],
        #             #     ["chunk_rank", "shape", "chunk_n_in_chunk"],
        #             # ]

        #             # LIST_CONTEXT = [
        #             #     {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
        #             #     {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
        #             #     {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
        #             #     {"same":["chunk_rank", "shape"], "diff":["chunk_n_in_chunk"]},
        #             #     ]

        #             # LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
        #             # # Use 1 for things that use syntax role as effect. or else will throw out cases with 1 item in given chunk.

        #             # filtdict = {
        #             #     "stroke_index": list(range(1, 10, 1)), # [1, ..., ]
        #             # }
        #             # LIST_FILTDICT = [
        #             #     filtdict for _ in range(len(LIST_VAR))
        #             # ]

        #             # ########
        #             # PLOT_CHUNK_END_STATE_ALIGNMENT = True

        #     # elif question in ["SP_BASE_stroke"]:
        #     #     ################# SP
        #     #     # (1) Shape.
        #     #     savedir_suffix = f"shape"
        #     #     superv_dpca_params = {
        #     #         "superv_dpca_var":"shape",
        #     #         "superv_dpca_vars_group":["gridloc", "gridsize"],
        #     #         "superv_dpca_filtdict":None
        #     #     }
        #     #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #     #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #     #     # (1) gridloc
        #     #     savedir_suffix = f"gridloc"
        #     #     superv_dpca_params = {
        #     #         "superv_dpca_var":"gridloc",
        #     #         "superv_dpca_vars_group":["shape", "gridsize"],
        #     #         "superv_dpca_filtdict":None
        #     #     }
        #     #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #     #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #     #     # (1) Size
        #     #     savedir_suffix = f"gridsize"
        #     #     superv_dpca_params = {
        #     #         "superv_dpca_var":"gridsize",
        #     #         "superv_dpca_vars_group":["shape", "gridloc"],
        #     #         "superv_dpca_filtdict":None
        #     #     }
        #     #     LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #     #     LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #     #     ################ sometimes only few trials, and clean enough...
        #     #     nmin_trials_per_lev = 3
        #     #     LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]

        #     # elif question in ["PIG_BASE_stroke", "CHAR_BASE_stroke"]:
        #         # savedir_suffix = f"shape"
        #         # # (1) var rank, condition on everything else.
        #         # superv_dpca_params = {
        #         #     "superv_dpca_var":"shape",
        #         #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "gridloc", "stroke_index"],
        #         #     "superv_dpca_filtdict":None
        #         # }
        #         # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #         # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #         # savedir_suffix = f"stroke_index"
        #         # # (1) var rank, condition on everything else.
        #         # superv_dpca_params = {
        #         #     "superv_dpca_var":"stroke_index",
        #         #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "gridloc", "shape"],
        #         #     "superv_dpca_filtdict":None
        #         # }
        #         # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #         # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #         # savedir_suffix = f"gridloc"
        #         # # (1) var rank, condition on everything else.
        #         # superv_dpca_params = {
        #         #     "superv_dpca_var":"gridloc",
        #         #     "superv_dpca_vars_group":["task_kind", "CTXT_loc_prev", "stroke_index", "shape"],
        #         #     "superv_dpca_filtdict":None
        #         # }
        #         # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #         # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #         # savedir_suffix = f"CTXT_loc_next"
        #         # # (1) var rank, condition on everything else.
        #         # superv_dpca_params = {
        #         #     "superv_dpca_var":"CTXT_loc_next",
        #         #     "superv_dpca_vars_group":["task_kind", "stroke_index", "CTXT_loc_prev", "shape", "gridloc"],
        #         #     "superv_dpca_filtdict":None
        #         # }
        #         # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #         # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #         # savedir_suffix = f"CTXT_shape_next"
        #         # # (1) var rank, condition on everything else.
        #         # superv_dpca_params = {
        #         #     "superv_dpca_var":"CTXT_shape_next",
        #         #     "superv_dpca_vars_group":["task_kind", "stroke_index", "CTXT_loc_prev", "shape", "gridloc"],
        #         #     "superv_dpca_filtdict":None
        #         # }
        #         # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        #         # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        #         # ###############
        #         # LIST_VAR = [
        #         #     "CTXT_shape_next",
        #         #     "CTXT_loc_next",
        #         #     "shape",
        #         #     "gridloc",
        #         #     "stroke_index",
        #         #     "task_kind",
        #         # ]
        #         # # More restrictive
        #         # LIST_VARS_OTHERS = [
        #         #     ["task_kind", "stroke_index_is_first", "CTXT_loc_prev", "shape", "gridloc"],
        #         #     ["task_kind", "stroke_index_is_first", "CTXT_loc_prev", "shape", "gridloc"],
        #         #     ["task_kind", "stroke_index_is_first", "gridloc"],
        #         #     ["task_kind", "stroke_index_is_first", "shape"],
        #         #     ["task_kind", "shape", "gridloc"],
        #         #     ["CTXT_loc_prev", "shape", "gridloc"],
        #         # ]

        #         # LIST_CONTEXT = [None for _ in range(len(LIST_VAR))]

        #         # LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(len(LIST_VAR))]

        #         # filtdict = None
        #         # LIST_FILTDICT = [
        #         #     filtdict for _ in range(len(LIST_VAR))
        #         # ]

        #     else:
        #         print(question)
        #         assert False

        #     for savedir_suffix, superv_dpca_params in zip(LIST_SAVEDIR_SUFFIX, LIST_SUPERV_DPCA_PARAMS):
        #         LIST_DIMRED_METHODS.append(["superv_dpca", None, None, None, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params])

        #     assert len(LIST_DIMRED_METHODS)>0

    for x in LIST_DIMRED_METHODS:
        assert len(x)==7

    # Load q_params
    from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
    q_params = rsagood_questions_dict(animal, date, question)[question]
    if which_level=="trial":
        # events_keep = ["03_samp", "04_go_cue", "05_first_raise", "06_on_strokeidx_0"]
        # events_keep = ["03_samp", "04_go_cue", "05_first_raise", "06_on_strokeidx_0"]
        events_keep = ["03_samp", "05_first_raise", "06_on_strokeidx_0"]
    elif which_level in ["stroke", "stroke_off"]:
        events_keep = ["00_stroke"]
    else:
        print(question)
        assert False

    ############### HACK
    HACK_RENAME_SHAPES = "CHAR" in question

    if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
        # To make plots cleaner, e.g., for char projected on SP subspaces
        nmin_trials_per_lev = 6

    if question=="SP_novel_shape":
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=True
    else:
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=False

    if LOAD_AND_PLOT_RESULTS_ONLY == False:
        # First try to load. If fails, then extract
        DFALLPA = load_handsaved_wrapper(animal=animal, date=date, version=which_level, combine_areas=combine_into_larger_areas,
                                        return_none_if_no_exist=True, question=question)
        if DFALLPA is None:
            DFALLPA = extract_dfallpa_helper(animal, date, question, combine_into_larger_areas, events_keep=events_keep, do_save=True)

            # ########################################## EXTRACT NOT-NORMALIZED DATA
            # from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
            # DFALLPA = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows, which_level=which_level,
            #                                         events_keep=events_keep,
            #                                         combine_into_larger_areas=combine_into_larger_areas,
            #                                         exclude_bad_areas=exclude_bad_areas, SPIKES_VERSION=SPIKES_VERSION,
            #                                         HACK_RENAME_SHAPES=HACK_RENAME_SHAPES, fr_normalization_method=None,
            #                                         path_to_save_example_fr_normalization=None)

            ######## STANDARD PREPROCESING.
            from neuralmonkey.classes.population_mult import dfallpa_preprocess_vars_conjunctions_extract
            dfallpa_preprocess_vars_conjunctions_extract(DFALLPA, which_level=which_level)
    else:
        # Just a DUMMY VARiable
        DFALLPA = pd.DataFrame([])


    # Iterate over all dim reduction methods
    # for NPCS_KEEP in LIST_NPCS_KEEP:
    for EVENT_KEEP in events_keep:
        for dim_red_method, NPCS_KEEP, extra_dimred_method_n_components, umap_n_neighbors, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params in LIST_DIMRED_METHODS:
            for fr_normalization_method in LIST_FR_NORMALIZATION:
                
                if HACK_TRAJS_JUST_FOR_PLOTTING_NICE and EVENT_KEEP == "03_samp":
                    # hacky, make the PCA construction window start AFTER iamge onset.
                    assert "dpca_proj_twind" not in superv_dpca_params, "nto sure what to do -- avoiding overwriting."
                    superv_dpca_params = {k:v for k, v in superv_dpca_params.items()} # make copy
                    superv_dpca_params["dpca_pca_twind"] = [0.1, 0.5]
                    superv_dpca_params["dpca_proj_twind"] = [-0.3, 0.5]
                
                ############ PREPROCESSING
                from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
                # Make a copy of all PA before normalization
                DFallpa = DFALLPA.copy()
                list_pa = [pa.copy() for pa in DFallpa["pa"]]
                DFallpa["pa"] = list_pa

                dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

                ###########
                if HACK_TRAJS_JUST_FOR_PLOTTING_NICE:
                    tmp = "NICE_TRAJ--"
                else:
                    tmp = ""
                if TRAJECTORIES_METHOD == "scalar":
                    SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_DIST/{animal}-{date}/{tmp}{TRAJECTORIES_METHOD}-wl={which_level}-ev={EVENT_KEEP}-spks={SPIKES_VERSION}-combarea={combine_into_larger_areas}/{dir_suffix}-norm={fr_normalization_method}-dr={dim_red_method}-NPC={NPCS_KEEP}-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={savedir_suffix}"
                elif TRAJECTORIES_METHOD == "traj_to_scalar":
                    SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_DIST/{animal}-{date}/{tmp}{TRAJECTORIES_METHOD}-wl={which_level}-ev={EVENT_KEEP}-spks={SPIKES_VERSION}-combarea={combine_into_larger_areas}/{dir_suffix}-norm={fr_normalization_method}-dr={dim_red_method}-NPC={NPCS_KEEP}-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={savedir_suffix}"
                else:
                    print(TRAJECTORIES_METHOD)
                    assert False, "code it"

                os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
                print("SAVING AT:", SAVEDIR_ANALYSIS)

                if LOAD_AND_PLOT_RESULTS_ONLY==False:
                    # # Make a copy of all PA before normalization
                    # DFallpa = DFALLPA.copy()
                    # list_pa = [pa.copy() for pa in DFallpa["pa"]]
                    # DFallpa["pa"] = list_pa

                    ########### HACK - Testing generalization of shapes to char, then keep only the shapes presenta cross SP and char
                    if question in ["CHAR_BASE_stroke"]:
                        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

                        # [Optional, for char] Keep only shapes that exist in every context
                        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
                        dflab = DFallpa["pa"].values[0].Xlabels["trials"]

                        ### (0) Plot original tabulation of shape vs task_klind
                        fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "shape_semantic", "task_kind")
                        path = f"{SAVEDIR_ANALYSIS}/CHAR_HACK_COUNTS-orig.pdf"
                        fig.savefig(path)


                        ##### (1) Try to keep only shapes present in both primsinsgle and CHAR
                        task_kinds_keep = ["character", "prims_single"]
                        dflabthis = dflab[dflab["task_kind"].isin(task_kinds_keep)].reset_index(drop=True)
                        _, dict_dfthis = extract_with_levels_of_conjunction_vars_helper(dflabthis, "task_kind", ["shape_semantic"], 
                                                                                        n_min_per_lev=nmin_trials_per_lev,
                                                                    plot_counts_heatmap_savepath=f"{SAVEDIR_ANALYSIS}/CHAR_HACK_COUNTS-{'|'.join(task_kinds_keep)}.png")
                        shapes_keep = dict_dfthis.keys()
                        shapes_keep = [x[0] for x in list(dict_dfthis.keys())]

                        #### (2) If this is empty, then try getting shapes matched across PIG and CHAR
                        if len(shapes_keep)==0:
                            task_kinds_keep = ["character", "prims_on_grid"]
                            dflabthis = dflab[dflab["task_kind"].isin(task_kinds_keep)].reset_index(drop=True)
                            _, dict_dfthis = extract_with_levels_of_conjunction_vars_helper(dflabthis, "task_kind", ["shape_semantic"], 
                                                                                            n_min_per_lev=nmin_trials_per_lev,
                                                                        plot_counts_heatmap_savepath=f"{SAVEDIR_ANALYSIS}/CHAR_HACK_COUNTS-{'|'.join(task_kinds_keep)}.png")
                            shapes_keep = dict_dfthis.keys()
                            shapes_keep = [x[0] for x in list(dict_dfthis.keys())]
                        
                        #### (3) If still empty, then you dont have any shapes matched between SP/PIG vs. CHAR. Cannot run this analysis.
                        if len(shapes_keep)==0:
                            fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "shape_semantic", "task_kind")
                            fig.savefig("/tmp/CONJUNCTIONS.pdf")
                            assert False, "No shape exists across task_kinds... see /tmp/CONJUNCTIONS.pdf"

                        ##### (4) Apply filter to data
                        for filtdict in LIST_FILTDICT:
                            filtdict["shape_semantic"] = shapes_keep

                        # HACKY - filter to just keep these shapes, and relevant task kind , so that
                        # This is done BEFORE dim reduction, to maximally separate them
                        list_pa_new = []
                        for pa in DFallpa["pa"].values:
                            pa_new = pa.slice_by_labels_filtdict({
                                "task_kind":task_kinds_keep,
                                "shape_semantic":shapes_keep,
                                })
                            assert len(pa_new.Xlabels["trials"]["task_kind"].unique())>1, "you need at least one of SP/PIG along with char... problem with filtering?"
                            list_pa_new.append(pa_new)
                            if pa_new.X.shape[1]==0:
                                assert False, "Shold not be possible. see above."
                                # from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
                                # fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "shape_semantic", "task_kind")
                                # fig.savefig("/tmp/CONJUNCTIONS.pdf")
                                # assert False, "No shape exists across task_kinds... see /tmp/CONJUNCTIONS.pdf"
                        DFallpa["pa"] = list_pa_new

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
                        "savedir_suffix":savedir_suffix,
                        "superv_dpca_params":superv_dpca_params
                        },
                        f"{SAVEDIR_ANALYSIS}/params.txt")

                    #################### Normalize PA firing rates if needed
                    # IGNORE - it is done above now.
                    # path_to_save_example_fr_normalization = f"{SAVEDIR_ANALYSIS}/example_fr_normalization.png"
                    # if fr_normalization_method is not None:
                    #     if fr_normalization_method=="each_time_bin":
                    #         # Then demean in each time bin indepednently
                    #         subtract_mean_at_each_timepoint = True
                    #         subtract_mean_across_time_and_trial = False
                    #     elif fr_normalization_method=="across_time_bins":
                    #         # ALl time bins subtract the same scalar --> maintains temporal moudlation.
                    #         subtract_mean_at_each_timepoint = False
                    #         subtract_mean_across_time_and_trial = True
                    #     else:
                    #         print(fr_normalization_method)
                    #         assert False

                    #     from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization
                    #     list_panorm = []

                    #     for i, pa in enumerate(DFallpa["pa"].tolist()):
                    #         if path_to_save_example_fr_normalization is not None and i==0:
                    #             plot_example_chan_number = pa.Chans[0]
                    #             if which_level=="trial":
                    #                 plot_example_split_var_string = "seqc_0_shape"
                    #             elif which_level=="stroke":
                    #                 plot_example_split_var_string = "shape"
                    #             else:
                    #                 plot_example_split_var_string = q_params["effect_vars"][0]
                    #         else:
                    #             plot_example_chan_number = None
                    #         PAnorm, PAscal, PAscalagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(pa, None,
                    #                                                                                         DO_AGG_TRIALS=False,
                    #                                                                                         plot_example_chan_number=plot_example_chan_number,
                    #                                                                                             plot_example_split_var_string = plot_example_split_var_string,
                    #                                                                                         subtract_mean_at_each_timepoint=subtract_mean_at_each_timepoint,
                    #                                                                                         subtract_mean_across_time_and_trial=subtract_mean_across_time_and_trial)
                    #         if path_to_save_example_fr_normalization is not None and i==0:
                    #             savefig(fig, path_to_save_example_fr_normalization)
                    #         list_panorm.append(PAnorm)
                    #     DFallpa["pa"] = list_panorm

                ########################################
                LIST_TWIND, LIST_TBIN_DUR, LIST_TBIN_SLIDE = _get_list_twind_by_animal(animal, EVENT_KEEP, 
                                                                                    TRAJECTORIES_METHOD, HACK_TRAJS_JUST_FOR_PLOTTING_NICE=HACK_TRAJS_JUST_FOR_PLOTTING_NICE)
                for twind_analy in LIST_TWIND:
                    for tbin_dur, tbin_slice in zip(LIST_TBIN_DUR, LIST_TBIN_SLIDE):

                        SAVEDIR = f"{SAVEDIR_ANALYSIS}/twinda={twind_analy}-tbin={tbin_dur}"

                        ################# GET DFRES
                        if LOAD_AND_PLOT_RESULTS_ONLY:
                            # LOAD pre-computed results
                            path = f"{SAVEDIR}/DFRES.pkl"
                            DFRES = pd.read_pickle(path)
                        else:
                            # Compute results new
                            os.makedirs(SAVEDIR, exist_ok=True)
                            print("THIS TBIN_DUR", SAVEDIR)

                            # Save all the params
                            from pythonlib.tools.expttools import writeDictToTxtFlattened
                            path = f"{SAVEDIR}/params_var.txt"
                            writeDictToTxtFlattened({
                                "LIST_VAR":{i:x for i, x in enumerate(LIST_VAR)},
                                "LIST_VARS_OTHERS":{i:x for i, x in enumerate(LIST_VARS_OTHERS)},
                                "LIST_CONTEXT":{i:x for i, x in enumerate(LIST_CONTEXT)},
                                "LIST_PRUNE_MIN_N_LEVS":{i:x for i, x in enumerate(LIST_PRUNE_MIN_N_LEVS)},
                                "LIST_FILTDICT":{i:x for i, x in enumerate(LIST_FILTDICT)},
                                "NPCS_KEEP":NPCS_KEEP,
                                "fr_normalization_method":fr_normalization_method,
                                "tbin_dur":tbin_dur,
                                "tbin_slice":tbin_slice,
                                "twind_analy":twind_analy,
                                "dim_red_method":dim_red_method,
                                "extra_dimred_method_n_components":extra_dimred_method_n_components,
                                "umap_n_neighbors":umap_n_neighbors,
                                "TRAJECTORIES_METHOD":TRAJECTORIES_METHOD,
                                "event_keep":EVENT_KEEP
                            }, path)


                            #### COLLECT DATA
                            # from neuralmonkey.analyses.decode_good import euclidian_distance_compute
                            from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_scalar, euclidian_distance_compute_AnBmCk_endpoint, euclidian_distance_compute_trajectories
                            list_dfres = []
                            for i, row in DFallpa.iterrows():
                                br = row["bregion"]
                                tw = row["twind"]
                                if "event" not in DFallpa.columns:
                                    print(DFallpa)
                                    print(DFallpa.columns)
                                    assert False
                                ev = row["event"]
                                PA = row["pa"]

                                if not ev == EVENT_KEEP:
                                    continue

                                print("bregion, twind_analy, tbin_dur: ", br, twind_analy, tbin_dur)

                                savedir = f"{SAVEDIR}/each_region/{br}-twind_analy={twind_analy}"
                                os.makedirs(savedir, exist_ok=True)

                                if TRAJECTORIES_METHOD == "scalar":
                                    if dim_red_method is None:
                                        # Then this is just plotting raw data...
                                        _plot_state_space = False
                                    else:
                                        _plot_state_space = PLOT_STATE_SPACE
                                    dfres, PAredu = euclidian_distance_compute_scalar(PA, LIST_VAR, LIST_VARS_OTHERS, PLOT, PLOT_MASKS,
                                                                    twind_analy, tbin_dur, tbin_slice, savedir,
                                                                    PLOT_STATE_SPACE=_plot_state_space,
                                                                    nmin_trials_per_lev=nmin_trials_per_lev,
                                                                    LIST_CONTEXT=LIST_CONTEXT, LIST_FILTDICT=LIST_FILTDICT,
                                                                    LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS,
                                                                    dim_red_method = dim_red_method,
                                                                    extra_dimred_method_n_components = extra_dimred_method_n_components,
                                                                    NPCS_KEEP=NPCS_KEEP,
                                                                    umap_n_neighbors=umap_n_neighbors,
                                                                    superv_dpca_params=superv_dpca_params,
                                                                    return_PAredu=True)
                                    
                                    try:
                                        if PLOT_CHUNK_END_STATE_ALIGNMENT:
                                            # HACKY - plots related to chunks, testing alignement to endpoint of chunk.
                                            savedir = f"{SAVEDIR}/END_STATE_ALIGNMENT/{br}"
                                            os.makedirs(savedir, exist_ok=True)
                                            euclidian_distance_compute_AnBmCk_endpoint(PAredu, savedir)
                                    except Exception as err:
                                        print("************ FAILED euclidian_distance_compute_AnBmCk_endpoint")
                                        print(err)
                                        pass

                                elif TRAJECTORIES_METHOD == "traj_to_scalar":
                                    # PLOT_TRAJS = False
                                    PLOT_HEATMAPS = False
                                    dfres = euclidian_distance_compute_trajectories(PA, LIST_VAR, LIST_VARS_OTHERS, twind_analy, tbin_dur,
                                                            tbin_slice, savedir, PLOT_TRAJS=PLOT_STATE_SPACE, PLOT_HEATMAPS=PLOT_HEATMAPS,
                                                            nmin_trials_per_lev=nmin_trials_per_lev,
                                                            LIST_CONTEXT=LIST_CONTEXT, LIST_FILTDICT=LIST_FILTDICT,
                                                            LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS,
                                                            NPCS_KEEP=NPCS_KEEP,
                                                            dim_red_method = dim_red_method, superv_dpca_params=superv_dpca_params,
                                                            COMPUTE_EUCLIDIAN = COMPUTE_EUCLIDIAN, PLOT_CLEAN_VERSION = PLOT_CLEAN_VERSION)
                                else:
                                    print(TRAJECTORIES_METHOD)
                                    assert False

                                plt.close("all")

                                if dfres is not None and len(dfres)>0:
                                    dfres["bregion"] = br
                                    dfres["twind"] = [tw for _ in range(len(dfres))]
                                    dfres["twind_analy"] = [twind_analy for _ in range(len(dfres))]
                                    dfres["event"] = ev
                                    dfres["savedir_suffix"] = savedir_suffix
                                    list_dfres.append(dfres)

                            # Concat
                            if COMPUTE_EUCLIDIAN and len(list_dfres)>0:
                                DFRES = pd.concat(list_dfres).reset_index(drop=True)

                                # SAVE
                                path = f"{SAVEDIR}/DFRES.pkl"
                                pd.to_pickle(DFRES, path)
                                print("Saved to: ", path)
                            else:
                                # SKip saving
                                DFRES = None

                        ############ PLOTS
                        if DFRES is not None and not SKIP_EUCL_PLOTS:
                            # try:
                            # (1) All main plots, scores, distributions
                            if "shuffled_time" in DFRES.columns:
                                # Results from euclidian_distance_compute_trajectories()
                                DFRES_NOSHUFF = DFRES[DFRES["shuffled_time"] == False].reset_index(drop=True)
                                plot_all_results(DFRES_NOSHUFF, SAVEDIR)
                            else:
                                plot_all_results(DFRES, SAVEDIR)

                            # (2) Like above, but a few, using Yue neural moduluation
                            # try:
                            DFRES_PIVOT_YUE = plot_all_results_yue(DFRES, SAVEDIR)
                            # except Exception as err:
                            #     print("FAILED plot_all_results_yue, this err:")
                            #     print(err)

                            # (3) Pairwise plots
                            plot_pairwise_all_wrapper(DFRES, SAVEDIR)

                            if TRAJECTORIES_METHOD == "traj_to_scalar":
                                plot_all_results_time_trajectories(DFRES, SAVEDIR)
                            # except Exception as err:
                            #     print("------------------------")
                            #     print("THIS ERROR: ", err)
                            #     print("SKIPPING PLOTS!!")
                            #     pass
                            
