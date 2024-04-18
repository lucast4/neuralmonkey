"""
Loading and summary plots for euclidian stuff, made esp for syntax plots.
4/16/24
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



def load_preprocess_concat_mult_sessions(animal, save_suffix):
    """ Loads, preprocesses (each individualy) and concats
    """
    # Get params
    if animal=="Diego":
        twind_analy = (-0.1, 0.1)

        if save_suffix=="AnBmCk_general":
            dates = [230724, 230726, 230730, 230817, 230913, 231116, 231118]
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_seqsup":
            dates = [230922, 230920, 230924, 230925]
            question = "RULESW_ANY_SEQSUP_STROKE"
            fr_normalization_method = "across_time_bins"
            twind_analy = (-0.1, 0.1)
        elif save_suffix=="sh_vs_dir":
            dates = [230719, 230823, 230804, 230827, 230919]
            question = "RULESW_ANBMCK_DIR_STROKE"
            fr_normalization_method = "across_time_bins"
            twind_analy = (-0.1, 0.1)
        elif save_suffix=="sh_vs_col":
            dates = [230910, 230912, 230927]
            question = "RULESW_ANBMCK_COLRANK_STROKE"
            fr_normalization_method = "across_time_bins"
            twind_analy = (-0.1, 0.1)
        else:
            print(animal, save_suffix)
            assert False, "add this."
    elif animal=="Pancho":
        twind_analy = (0.05, 0.25)

        if save_suffix=="two_shape_sets":
            # AnBkCk, two shape sets.
            dates = [220906, 220907, 220908, 220909]
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="AnBmCk_general":
            dates = [220906, 220907, 220908, 220909, 230811, 230829]
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_seqsup":
            assert False, "have not run all yet"
            dates = [230921, 230920, 231019]
            question = "RULESW_ANY_SEQSUP_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_dir":
            dates = [221023, 230910, 230914, 230919]
            question = "RULESW_ANBMCK_DIR_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_col":
            dates = [230928, 230929]
            question = "RULESW_ANBMCK_COLRANK_STROKE"
            fr_normalization_method = "across_time_bins"
        else:
            print(animal, save_suffix)
            assert False
    else:
        print(animal, save_suffix)
        assert False, "add this."

    # Load data across dates
    list_dfres = []
    list_dfres_pivot_distr = []
    list_dfres_pivot_pair = []
    list_dfres_pivot_yue = []
    for d in dates:
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{question}-fr_normalization_method={fr_normalization_method}-NPCS_KEEP=10/twind_analy={twind_analy}-tbin_dur=0.1"

        print("Loading... ", SAVEDIR)
        path = f"{SAVEDIR}/DFRES.pkl"

        DFRES = pd.read_pickle(path)

        DFRES["animal"] = animal
        DFRES["date"] = d
        DFRES["question"] = question
        DFRES["fr_normalization_method"] = fr_normalization_method

        # Derive all metrics within each animal, then concat
        from neuralmonkey.scripts.analy_euclidian_dist_pop_script import compute_all_derived_metrics
        DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, plot_params = compute_all_derived_metrics(DFRES)

        # print(DFRES["var_others"])
        # print(DFRES["dist_norm_95"])
        # assert False
        list_dfres.append(DFRES)
        list_dfres_pivot_distr.append(DFRES_PIVOT_DISTR)
        list_dfres_pivot_pair.append(DFRES_PIVOT_PAIRWISE)
        list_dfres_pivot_yue.append(DFRES_PIVOT_YUE)
    DFRES = pd.concat(list_dfres).reset_index(drop=True)
    DFRES_PIVOT_DISTR = pd.concat(list_dfres_pivot_distr).reset_index(drop=True)
    DFRES_PIVOT_PAIRWISE = pd.concat(list_dfres_pivot_pair).reset_index(drop=True)
    DFRES_PIVOT_YUE = pd.concat(list_dfres_pivot_yue).reset_index(drop=True)

    # Initialize savedir and save params
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
    params = {
            "animal":animal,
            "dates":dates,
            "question":question,
            "fr_normalization_method":fr_normalization_method,
            "twind_analy":twind_analy,
            "save_suffix":save_suffix
        }
    SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_DIST/MULT/{animal}-{question}-{save_suffix}-{min(dates)}-{max(dates)}"
    print(SAVEDIR_ANALYSIS)
    os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened(
        params,
        f"{SAVEDIR_ANALYSIS}/params.txt"
    )

    return DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, SAVEDIR_ANALYSIS, params, plot_params


def plot_all(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, plot_params, SAVEDIR_ANALYSIS):
    """

    :param DFRES:
    :param DFRES_PIVOT_DISTR:
    :param DFRES_PIVOT_PAIRWISE:
    :param plot_params:
    :param SAVEDIR_ANALYSIS:
    :return:
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _plot_all_results, plot_pairwise_all_wrapper
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _plot_pairwise_btw_vvo_general_MULT
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import params_pairwise_variables_for_plotting, plot_pairwise_btw_levels_for_seqsup

    # (1) All the geneeral plots (combining all data, i.e., datapt = levo, not rigorous)
    _plot_all_results(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, plot_params, SAVEDIR_ANALYSIS, ONLY_ESSENTIALS=True)

    # (2) Pairwise (combining all data, i.e., datapt = levo, not rigorous)
    plot_pairwise_all_wrapper(DFRES, SAVEDIR_ANALYSIS)

    # (3) Plot pairwise data, (one datapt per date, rigorous)
    LIST_LIST_VVO_XY, LIST_dir_suffix = params_pairwise_variables_for_plotting()

    # Have preSMA separate from other (syntax).
    map_subplot_var_to_new_subplot_var = {
        "preSMA_a":"preSMA_a",
        "preSMA_p":"preSMA_p",
    }
    # Make all the plots
    for LIST_VVO_XY, dir_suffix in zip(LIST_LIST_VVO_XY, LIST_dir_suffix):
        for version, plot_text, map_subplot in [
            ["single_subplot", True, None],
            ["one_subplot_per_bregion", True, None],
            ["one_subplot_per_bregion", False, map_subplot_var_to_new_subplot_var],
            ]:
            dir_suffix_this = f"{dir_suffix}--ver={version}-text={plot_text}-map_subplot={map_subplot is not None}"
            for DFTHIS, ythis in [
                [DFRES_PIVOT_YUE, "dist_yue_diff"],
                [DFRES, "dist_norm_95"],
                ]:
                _plot_pairwise_btw_vvo_general_MULT(DFTHIS, ythis, SAVEDIR_ANALYSIS, LIST_VVO_XY, dir_suffix_this,
                                                    plot_text=plot_text,
                                                    version=version,
                                                    map_subplot_var_to_new_subplot_var=map_subplot)
                plt.close("all")

    # (4) Pairwise, superv vs. nonsuperv, shape vs dir, etc.
    # NOTE: the agged version is already done above in plot_pairwise_all_wrapper. Here is doing split by bregion, one
    # dot per date.
    for VERSION in ["nosup_vs_sup", "shape_vs_dir", "nocol_vs_col"]:
    # for VERSION in ["nocol_vs_col"]:
        plot_pairwise_btw_levels_for_seqsup(DFRES, SAVEDIR_ANALYSIS, VERSION=VERSION, one_subplot_per_bregion=True)


    # (5) Clean histograms
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import plot_histograms_clean_wrapper
    dat_level = "distr"
    effect_context = "diff|same"
    # bregions_plot = ["M1_m", "PMv_m", "PMd_p", "dlPFC_a", "vlPFC_p", ]
    # bregions_plot = ["M1_m", "dlPFC_a", "preSMA_a"]
    bregions_plot = ["M1_m", "preSMA_a"]
    plot_histograms_clean_wrapper(DFRES, SAVEDIR_ANALYSIS, dat_level, effect_context,
                                      bregions_plot)



if __name__=="__main__":

    animal = sys.argv[1]
    save_suffix = sys.argv[2] # a code name for this analysis (e.g., two_shape_sets)

    # (1) Load all data
    DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, SAVEDIR_ANALYSIS, params, plot_params = load_preprocess_concat_mult_sessions(
        animal, save_suffix)

    # (2) Plot
    plot_all(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, plot_params, SAVEDIR_ANALYSIS)