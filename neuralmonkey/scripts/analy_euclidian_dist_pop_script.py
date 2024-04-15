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
from neuralmonkey.classes.population_mult import extract_single_pa
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels
from pythonlib.tools.pandastools import append_col_with_grp_index

############### PARAMS (EUCLIDIAN)
PLOT = False
PLOT_MASKS = False
N_MIN_TRIALS = 6
LIST_TWIND = [
    # (-0.2, 0.2),
    # (0.05, 0.35),
    # (0, 0.2),
    # (-0.3, -0.1),
    (-0.1, 0.1),
    # (0.1, 0.3),
]
PLOT_STATE_SPACE = True
# NPCS_KEEP = None # use auto method

DEBUG = False
SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
nmin_trials_per_lev = 6
LIST_NPCS_KEEP = [10]
extra_dimred_method = "umap"
umap_n_neighbors = 40
LIST_FR_NORMALIZATION = ["across_time_bins"]
LIST_TBIN_DUR = [0.1] # None means no binning

# Load and plot saved values?
LOAD_AND_PLOT_RESULTS_ONLY = False

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

def plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix=None):
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
    for DFTHIS, ythis in [
        [DFRES, "dist_norm_95"],
        [DFRES_PIVOT_YUE, "dist_yue_diff"],
        [DFRES_PIVOT_YUE, "dist_yue"],
        ]:

        assert ythis not in ythis_done, "this hacky solution sodesnt work - have to also make filenmae use DFTHIs version"
        ythis_done.append(ythis)

        # ----------
        shuffled = False
        list_dat_lev = DFTHIS["dat_level"].unique().tolist()
        for dat_lev in list_dat_lev:
            # Keep just this.
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
                    path = f"{savedirthis}/ythis={ythis}.pdf"
                    savefig(fig, path)

                    plt.close("all")
                else:
                    print("--- Skipping, did not find in data (var_var_others):")
                    print(vvo_x)
                    print(vvo_y)
                    print("Unique values...:", dfthis["var_var_others"].unique().tolist())
    plt.close("all")


def plot_pairwise_btw_levels_for_seqsup(DFRES, SAVEDIR, VERSION="nosup_vs_sup"):
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
    :param SAVEDIR:
    :return:
    """
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping


    savedir = f"{SAVEDIR}/pairwise_btw_levels_SEQSUP--VER={VERSION}"
    os.makedirs(savedir, exist_ok=True)

    # First,
    if VERSION=="nosup_vs_sup":
        inds_keep = [i for i, var_others in enumerate(DFRES["var_others"]) if (var_others[0] in ["epochset_shape", "epochset_dir"]) and ("superv_is_seq_sup" in var_others)]
    if VERSION=="nocol_vs_col":
        inds_keep = [i for i, var_others in enumerate(DFRES["var_others"]) if (var_others[0] in ["epochset_shape", "epochset_dir"]) and ("INSTRUCTION_COLOR" in var_others)]
    elif VERSION=="shape_vs_dir":
        inds_keep = [i for i, var_others in enumerate(DFRES["var_others"]) if (var_others[0] in ["epochset_shape", "epochset_dir"])]
    else:
        print(VERSION)
        assert False

    DFRES = DFRES.iloc[inds_keep].reset_index(drop=True)

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
                if "superv_is_seq_sup" in row["var_others"]:
                    idx_check = row["var_others"].index("superv_is_seq_sup")
                else:
                    idx_check = None

                # Determoine if this is to keeop
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

            # combine all levels
            possible_levels = possible_lev_x + possible_lev_y
        elif version in ["nocol_vs_col", "shape_vs_dir"]:
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
        for lev_x in possible_lev_x:
            for lev_y in possible_lev_y:
                if lev_x != lev_y:
                    if tuple(sorted([lev_x, lev_y])) not in pairs_already_done:

                        pairs_already_done.append(tuple(sorted([lev_x, lev_y])))

                        # 3 compare effect across levels -->
                        list_dat_lev = DFTHIS["dat_level"].unique().tolist()
                        for dat_lev in list_dat_lev:
                            dfthis = DFTHIS[DFTHIS["dat_level"] == dat_lev].reset_index(drop=True)
                            _, fig = plot_45scatter_means_flexible_grouping(dfthis, "levo", lev_x, lev_y,
                                                            "effect_context", "dist_norm_95", "bregion",
                                                                   shareaxes=True)
                            path = f"{savedirthis}/dat_lev={dat_lev}--lev_x={lev_x}--lev_y={lev_y}.pdf"
                            savefig(fig, path)

                            plt.close("all")

def compute_normalized_distances(DFRES):
    """
    QUickly add columns and do quick normalziation of data to width of distribution.
    :param DFRES:
    :return:
    """

    # Compute normalized distnaces
    DIST_NULL = "DIST_NULL_98"
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

    DFRES = sort_df(DFRES)

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
    DFRES = stringify_values(DFRES)
    DFRES_THIS = DFRES[(DFRES["dat_level"] == "pts") & (DFRES["context_samediff"] == "same")]
    DFRES_PIVOT = pivot_table(DFRES_THIS, ["var", "var_others", "shuffled", "bregion", "twind", "event", "var_var_others", "dat_level", "levo", "leveff", "twind_analy"], ["effect_context"], ["dist_norm_95"], flatten_col_names=True).reset_index(drop=True)

    # keep only if have both diff|same and same|same
    # print(len(DFRES_PIVOT))
    DFRES_PIVOT = DFRES_PIVOT[~DFRES_PIVOT["dist_norm_95-diff|same"].isna()]
    # print(len(DFRES_PIVOT))
    DFRES_PIVOT = DFRES_PIVOT[~DFRES_PIVOT["dist_norm_95-same|same"].isna()]
    # print(len(DFRES_PIVOT))
    DFRES_PIVOT = DFRES_PIVOT.reset_index(drop=True)

    # Compute scores, two versions, either divide or diff.
    DFRES_PIVOT["dist_yue"] = DFRES_PIVOT["dist_norm_95-diff|same"] / DFRES_PIVOT["dist_norm_95-same|same"]
    DFRES_PIVOT["dist_yue_diff"] = DFRES_PIVOT["dist_norm_95-diff|same"] - DFRES_PIVOT["dist_norm_95-same|same"]

    # Sort
    DFRES_PIVOT = sort_df(DFRES_PIVOT)

    if PLOT:
        for yvarthis in ["dist_yue", "dist_yue_diff"]:
            fig = sns.catplot(data=DFRES_PIVOT, x="bregion", y=yvarthis, col="var_var_others",
                              col_wrap=3, aspect=1.57, alpha=0.4, height=6)
            if yvarthis=="dist_yue_diff": # aling to 0
                map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))
            elif yvarthis=="dist_yue":
                map_function_tofacet(fig, lambda ax: ax.axhline(1, color="k", alpha=0.4))
            else:
                assert False

            rotateLabel(fig)
            savefig(fig, f"{savedir}/overview_scatter-YUE-{yvarthis}.pdf")

            if yvarthis=="dist_yue_diff": # aling to 0
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

def plot_all_results(DFRES, SAVEDIR):
    """
    Wrapper to make all main plots of reusults.
    :param DFRES:
    :param SAVEDIR:
    :return:
    """
    from pythonlib.tools.pandastools import pivot_table
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import summarize_featurediff, plot_subplots_heatmap, stringify_values
    from pythonlib.tools.snstools import map_function_tofacet

    # Compute normalized distnaces
    DFRES, DIST_NULL = compute_normalized_distances(DFRES)
    # DIST_NULL = "DIST_NULL_98"
    # DFRES["dist_norm_95"] = DFRES["dist"]/DFRES[DIST_NULL]
    # # DFRES["dist_norm_95"] = DFRES["dist"]/DFRES["DIST_NULL_95"]
    # # DFRES["dist_norm_50"] = DFRES["dist"]/DFRES["DIST_NULL_50"]
    # DFRES["var_others"] = [tuple(x) for x in DFRES["var_others"]]
    # DFRES = append_col_with_grp_index(DFRES, ["index_var", "var", "var_others"], "var_var_others")
    # DFRES = append_col_with_grp_index(DFRES, ["effect_samediff", "context_samediff"], "effect_context")

    # Stringify, or else will fail groupby step
    DFRES = stringify_values(DFRES)

    ###########################################################################
    # Get dataframe with each row being a specific set of variables.
    DFRES_PIVOT = pivot_table(DFRES, ["var", "var_others", "shuffled", "bregion", "twind", "event", "var_var_others", "dat_level"], ["effect_context"], ["dist_norm_95"], flatten_col_names=True).reset_index(drop=True)

    # Compute effects tha DFRES_PIVOT[DFRES_PIVOT["dat_level"] == "pts"].reset_index(drop=True)t require inputs from multiple distance metrics.
    DFRES_PIVOT_DISTR = DFRES_PIVOT[DFRES_PIVOT["dat_level"] == "distr"].reset_index(drop=True)
    DFRES_PIVOT_DISTR["effect_index"] = DFRES_PIVOT_DISTR["dist_norm_95-diff|same"] / (DFRES_PIVOT_DISTR["dist_norm_95-diff|same"] + DFRES_PIVOT_DISTR["dist_norm_95-same|diff"])

    # Keep only the data using pairwise distances
    DFRES_PIVOT_PAIRWISE = DFRES_PIVOT[DFRES_PIVOT["dat_level"] == "pts"].reset_index(drop=True)

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

    ######################################### QUICK PLOT - SUMMARIES
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import summarize_featurediff

    # Sort, for consistent plotting across expts.
    DFRES_PIVOT_PAIRWISE = sort_df(DFRES_PIVOT_PAIRWISE)
    DFRES = sort_df(DFRES)
    DFRES_PIVOT_DISTR = sort_df(DFRES_PIVOT_DISTR)

    savedir = f"{SAVEDIR}/FIGURES"
    os.makedirs(savedir, exist_ok=True)

    ########## OVERVIEWS
    for yvarthis in [yvar, "dist", DIST_NULL]:
        for dat_level in DFRES["dat_level"].unique():
            dfthis = DFRES[DFRES["dat_level"]==dat_level]

            # Only plot speciifc ones that are useful
            if dat_level=="pts_yue" and yvarthis not in ["dist"]: # i.e. do not nomralize it is already normed.
                continue
            elif yvarthis == DIST_NULL and dat_level not in ["pts"]:
                # just plot oncse
                continue

            fig = sns.catplot(data=dfthis, x="bregion", y=yvarthis, col="var_var_others", hue="effect_context",
                              col_wrap=3, aspect=1.57, alpha=0.4, height=6)
            rotateLabel(fig)
            if dat_level=="distr":
                map_function_tofacet(fig, lambda ax: ax.axhline(0, color="k", alpha=0.4))
            elif dat_level=="pts_yue":
                map_function_tofacet(fig, lambda ax: ax.axhline(1, color="k", alpha=0.4))
            savefig(fig, f"{savedir}/overview_scatter-{yvarthis}-{dat_level}.pdf")

            fig = sns.catplot(data=dfthis, x="bregion", y=yvarthis, col="var_var_others", hue="effect_context",
                              col_wrap=3, aspect=1.7, kind="bar", height=6, errorbar=('ci', 68))
            rotateLabel(fig)
            savefig(fig, f"{savedir}/overview_bar-{yvarthis}-{dat_level}.pdf")

            plt.close("all")

    ########## OVERVIEWS (OLD - effect index)
    yvarthis = "effect_index"
    fig = sns.catplot(data=DFRES_PIVOT_DISTR, x="bregion", y=yvarthis, hue="var_var_others",  aspect=1.7, col="dat_level",
                      height=6, kind="bar", errorbar=('ci', 68))
    rotateLabel(fig)
    savefig(fig, f"{savedir}/effect_index-bar.pdf")

    ########## OVERVIEWS (dat_level = pts)
    for yvarthis in ["norm_dist_effect", "norm_dist_context", "norm_dist_both", "generalization_index", "generalization_index_scaled",
                     "gen_idx_diff_1", "gen_idx_diff_2", "gen_idx_diff_3", "gen_idx_diff_4", "gen_idx_diff_5",
                     "gen_idx_ratio_1", "gen_idx_ratio_2", "gen_idx_ratio_3", "gen_idx_ratio_4", "gen_idx_ratio_5"]:
        if yvarthis in DFRES_PIVOT_PAIRWISE.columns:
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
            fig = sns.catplot(data=DFRES_PIVOT_PAIRWISE, x="var_var_others", y=yvarthis, col="bregion",
                              col_wrap = 6, aspect=1, height=6, kind="bar", errorbar=('ci', 68))
            rotateLabel(fig)
            savefig(fig, f"{savedir}/FINAL-{yvarthis}-bar-splitby_bregion.pdf")
        plt.close("all")

    ########### PLOT ALL specific conjunction levels in heatmaps
    sns.set_context("paper", rc={"axes.labelsize":5})

    for dat_level in DFRES["dat_level"].unique():
        DFTHIS = DFRES[DFRES["dat_level"] == dat_level].reset_index(drop=True)

        # Plot histograms
        savedirthis = f"{savedir}/histograms-dat_level={dat_level}"
        os.makedirs(savedirthis, exist_ok=True)
        print("... ", savedirthis)

        fig = sns.displot(data=DFTHIS, x="dist_norm_95", hue="effect_context", col="bregion", row="var_var_others", element="step", fill=True, bins=20)
        savefig(fig, f"{savedirthis}/step.pdf")
        fig = sns.displot(data=DFTHIS, x="dist_norm_95", hue="effect_context", col="bregion", row="var_var_others", kind="kde", fill=False)
        savefig(fig, f"{savedirthis}/kde.pdf")

        print("Plotting specific conjucntions heatmaps ... ")
        yvar = "dist"
        list_effect_context = DFTHIS["effect_context"].unique()
        list_shuffled = DFTHIS["shuffled"].unique()
        for effect_context in list_effect_context:
            for shuffled in list_shuffled:

                dfthis = DFTHIS[(DFTHIS["effect_context"]==effect_context) & (DFTHIS["shuffled"]==shuffled)].reset_index(drop=True)
                if len(dfthis)>0:
                    savedirthis = f"{savedir}/each_conjunction-effect_context={effect_context}-shuffled={shuffled}-dat_level={dat_level}"
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
                    fig, axes = plot_subplots_heatmap(dfthis, "bregion", "levo", yvar, "var_var_others",
                                                      diverge=True, ncols=None, share_zlim=True)
                    savefig(fig, f"{savedirthis}/allconj_heatmap.pdf")
                    plt.close("all")

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
    combine_into_larger_areas = False
    list_time_windows = [(-0.6, 0.6)]
    EVENTS_IGNORE = [] # To reduce plots

    if question=="SP_novel_shape":
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=True
    else:
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=False

    if True:
        # Final version, this is best
        from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
        LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question)
    else:
        if question in ["PIG_BASE_stroke", "CHAR_BASE_stroke"]:

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

        elif question in ["RULE_BASE_stroke", "RULESW_BASE_stroke", "RULE_COLRANK_STROKE"]:
            # Syntax role encoding

            LIST_VAR = [
                # PMv should not get this
                "chunk_within_rank",
                "chunk_within_rank",
                "chunk_within_rank",
                "chunk_within_rank",
                "chunk_within_rank",

                # Like above, but mixing across syntax_concrete.
                "chunk_within_rank",
                "chunk_within_rank",

                # Roles
                "syntax_role",
                "syntax_role",
                "syntax_role",
                "syntax_role",
                "syntax_role",
                "syntax_role",

                "syntax_role",
                "syntax_role",

                # Trying this.
                "syntax_concrete",

                # "chunk_n_in_chunk", #
                "chunk_n_in_chunk", #
                "chunk_n_in_chunk", #

                # This is testing for "hierarchy" in that stroke index should not be consistent across concrete symtaxes.
                "stroke_index",
                "stroke_index",
            ]
            # More restrictive
            LIST_VARS_OTHERS = [
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "chunk_rank", "behseq_shapes_clust", "behseq_locs_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "chunk_rank", "behseq_locs_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "chunk_rank"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch"],

                # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],

                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "behseq_shapes_clust", "behseq_locs_clust", "shape"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "behseq_shapes_clust", "behseq_locs_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "behseq_shapes_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "behseq_locs_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch"],

                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev"],
                # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],

                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev", "FEAT_num_strokes_task"],

                # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust", "CTXT_shape_prev"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust", "CTXT_loc_next", "CTXT_shape_prev"],

                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete"],
                ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "behseq_shapes_clust", "behseq_locs_clust"],
            ]

            # LIST_VAR = [
            #     ("chunk_rank", "chunk_within_rank_semantic"), # strongest test of index within chunk, and chunk index.
            #     ("chunk_rank", "chunk_within_rank_semantic"), # strongest test of index within chunk, and chunk index.
            #     ("chunk_rank", "chunk_within_rank_semantic"), # strongest test of index within chunk, and chunk index.
            #
            #     "CTXT_shape_prev", # var = 2-motifs, conditioned on the shape of 2nd stroke.
            #     "CTXT_shape_prev", # var = 2-motifs, conditioned on the shape of 2nd stroke.
            #     "CTXT_shape_prev", # var = 2-motifs, conditioned on the shape of 2nd stroke.
            #
            #     "chunk_n_in_chunk", #
            #     "chunk_n_in_chunk", #
            #     "chunk_n_in_chunk", #
            #
            #     "chunk_within_rank_semantic", #
            #     "chunk_within_rank_semantic", #
            #     "chunk_within_rank_semantic", #
            #     "chunk_within_rank_semantic", #
            # ]
            #
            # LIST_VARS_OTHERS = [
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],
            #
            #     ("shape", "CTXT_loc_prev", "loc_on_clust", "loc_off_clust"), # Good -- tight control!
            #     ("shape", "CTXT_loc_prev", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next"), # Good -- tight control!
            #     ("shape", "CTXT_loc_prev", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next", "CTXT_shape_next"), # Good -- tight control!
            #
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust", "CTXT_loc_next"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust", "CTXT_loc_next", "CTXT_shape_prev"],
            #
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next", "CTXT_shape_prev"],
            #     ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next", "CTXT_shape_next"],


        else:
            print(question)
            assert False

    assert len(LIST_VAR)==len(LIST_VARS_OTHERS)
    # Convert from tuple to list
    LIST_VARS_OTHERS = [list(var_others) for var_others in LIST_VARS_OTHERS]
    LIST_VAR = [list(var) if isinstance(var, tuple) else var for var in LIST_VAR]

    if LOAD_AND_PLOT_RESULTS_ONLY == False:
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
    else:
        # Just a DUMMY VARiable
        DFALLPA = pd.DataFrame([])

    # for NPCS_KEEP in [5, 12]:
    for NPCS_KEEP in LIST_NPCS_KEEP:
        for fr_normalization_method in LIST_FR_NORMALIZATION:

            ###########
            SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_DIST/{animal}-{date}/{dir_suffix}-fr_normalization_method={fr_normalization_method}-NPCS_KEEP={NPCS_KEEP}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            print("SAVING AT:", SAVEDIR_ANALYSIS)

            if LOAD_AND_PLOT_RESULTS_ONLY==False:
                # Make a copy of all PA before normalization
                DFallpa = DFALLPA.copy()
                list_pa = [pa.copy() for pa in DFallpa["pa"]]
                DFallpa["pa"] = list_pa

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
                            "twind_analy":twind_analy,
                        }, path)

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
                                                               nmin_trials_per_lev=nmin_trials_per_lev, LIST_CONTEXT=LIST_CONTEXT,
                                                               LIST_FILTDICT=LIST_FILTDICT, NPCS_KEEP=NPCS_KEEP, extra_dimred_method=extra_dimred_method,
                                                               umap_n_neighbors = umap_n_neighbors, LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS)

                            plt.close("all")
                            if len(dfres)>0:
                                dfres["bregion"] = br
                                dfres["twind"] = [tw for _ in range(len(dfres))]
                                dfres["twind_analy"] = [twind_analy for _ in range(len(dfres))]
                                dfres["event"] = ev

                                list_dfres.append(dfres)

                        DFRES = pd.concat(list_dfres).reset_index(drop=True)

                        # SAVE
                        path = f"{SAVEDIR}/DFRES.pkl"
                        pd.to_pickle(DFRES, path)
                        print("Saved to: ", path)

                    ############### PLOT RESULTS
                    # Compute normalized distnaces
                    # DFRES, DIST_NULL = compute_normalized_distances(DFRES)
                    # DFRES["dist_norm_95"] = DFRES["dist"]/DFRES["DIST_NULL_95"]
                    # DFRES["dist_norm_50"] = DFRES["dist"]/DFRES["DIST_NULL_50"]
                    # DFRES["var_others"] = [tuple(x) for x in DFRES["var_others"]]
                    # DFRES = append_col_with_grp_index(DFRES, ["index_var", "var", "var_others"], "var_var_others")
                    # DFRES = append_col_with_grp_index(DFRES, ["effect_samediff", "context_samediff"], "effect_context")
                    # DFRES = append_col_with_grp_index(DFRES, ["dat_level", "effect_samediff", "context_samediff"], "dl_eff_ctxt")

                    ############ PLOTS
                    plot_all_results(DFRES, SAVEDIR)

                    for VERSION in ["nosup_vs_sup", "shape_vs_dir", "nocol_vs_col"]:
                        try:
                            plot_pairwise_btw_levels_for_seqsup(DFRES, SAVEDIR, VERSION=VERSION)
                        except Exception as err:
                            print(err)
                            print("Skipping plot_pairwise_btw_levels_for_seqsup... version:", VERSION)

                    try:
                        DFRES_PIVOT_YUE = plot_all_results_yue(DFRES, SAVEDIR)
                    except Exception as err:
                        print("FAILED plot_all_results_yue, this err:")
                        print(err)

                    ############ PAIRWISE - testing specific things
                    # 1) AnBmCk (two shapes --> preSMA not affected by shape)
                    dir_suffix = "two_shape_sets"
                    LIST_VVO_XY = [
                        ["14|syntax_role|('syntax_concrete', 'behseq_locs_clust', 'epoch')", "16|epoch|('syntax_concrete', 'behseq_locs_clust', 'syntax_role')"],
                        ["13|syntax_role|('syntax_concrete', 'epoch')", "15|epoch|('syntax_concrete', 'syntax_role')"],
                    ]
                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix=dir_suffix)

                    ### Location vs. chunk_within semantic
                    dir_suffix = "invar_location"
                    list_vvo_x = [
                     "17|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_loconclust_next')",
                     "18|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
                     "19|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')"]

                    list_vvo_y = [
                     "36|gridloc|('epoch', 'chunk_rank', 'shape', 'chunk_within_rank_semantic', 'CTXT_shape_prev', 'CTXT_locoffclust_prev')",
                     "37|gridloc|('epoch', 'chunk_rank', 'shape', 'chunk_within_rank_semantic', 'CTXT_shape_prev')"]

                    LIST_VVO_XY = []
                    for vvo_x in list_vvo_x:
                        for vvo_y in list_vvo_y:
                            LIST_VVO_XY.append([vvo_x, vvo_y])

                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix)

                    ### Syntax concrete vs. ciwithin
                    dir_suffix = "invar_syntconcr"
                    list_vvo_x = [
                     "17|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'CTXT_loconclust_next')",
                     "18|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
                     "19|chunk_within_rank_semantic|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')"]

                    list_vvo_y = [
                     "33|syntax_concrete|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'chunk_within_rank_semantic')"]

                    LIST_VVO_XY = []
                    for vvo_x in list_vvo_x:
                        for vvo_y in list_vvo_y:
                            LIST_VVO_XY.append([vvo_x, vvo_y])

                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix)


                    ### shape(chunk) vs. ciwithin
                    dir_suffix = "contrast_shape"
                    list_vvo_x = [
                     "35|chunk_rank|('epoch', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'chunk_within_rank_semantic')"]

                    list_vvo_y = [
                     "33|syntax_concrete|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev', 'chunk_within_rank_semantic')"
                    ]

                    LIST_VVO_XY = []
                    for vvo_x in list_vvo_x:
                        for vvo_y in list_vvo_y:
                            LIST_VVO_XY.append([vvo_x, vvo_y])

                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix)

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

                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix)


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

                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix)


                    ### nprims (onset vs. offset)
                    dir_suffix = "contrast_nprims_onset_offset"
                    list_vvo_x = [
                         "38|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')",
                         "39|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust')",
                    ]

                    list_vvo_y = [
                        "40|chunk_n_in_chunk|('epoch', 'chunk_rank', 'shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 'CTXT_shape_prev')"
                    ]
                    LIST_VVO_XY = []
                    for vvo_x in list_vvo_x:
                        for vvo_y in list_vvo_y:
                            LIST_VVO_XY.append([vvo_x, vvo_y])

                    plot_pairwise_btw_vvo_general(DFRES, SAVEDIR, LIST_VVO_XY, dir_suffix)
