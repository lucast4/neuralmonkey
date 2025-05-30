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


def load_preprocess_get_dates(animal, save_suffix, new_varied_hyperparams=True):
    """
    Load dates that are relevant for a given question (in grammar analyses)
    """

    if animal=="Diego":
        ### Time window
        if new_varied_hyperparams:
            twind_analy = (-0.1, 0.2)
        else:
            twind_analy = (-0.1, 0.1)

        ### Dates
        if save_suffix=="two_shape_sets":
            # AnBkCk, two shape sets.
            dates = [240822, 240827, 250319, 250321]
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="AnBmCk_general":
            # dates = [230724, 230726, 230730, 230817, 230913, 231116, 231118]
            # dates = [230724, 230726, 230817, 230913, 231116, 231118] # skipping 230730 most of the time.
            dates = [230723, 230724, 230726, 230727, 230728, 230730, 230815, 230816, 230817, 230913, 230914, 230915, 231116, 231118, 240822, 240827, 250319, 250321] # ALL (3/18/25)
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_seqsup":
            dates = [230920, 230921, 230922, 230924, 230925, 250320] # ALL (3/18/25)
            question = "RULESW_ANY_SEQSUP_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_dir":
            dates = [230719, 230823, 230804, 230827, 230919]
            question = "RULESW_ANBMCK_DIR_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_col":
            # dates = [230910, 230912, 230927, 231001]
            dates = [230910, 230912, 230927] # 231001 need to figure our neural preprpcessing.
            question = "RULESW_ANBMCK_COLRANK_STROKE"
            fr_normalization_method = "across_time_bins"
        else:
            print(animal, save_suffix)
            assert False, "add this."
    elif animal=="Pancho":
        ### Time window
        twind_analy = (0.05, 0.25)

        ### Dates
        if save_suffix=="two_shape_sets":
            # AnBkCk, two shape sets.
            # dates = [220906, 220907, 220908, 220909]
            dates = [220902, 220906, 220907, 220908, 220909, 240830, 250321, 250322] # ALL (3/18/25)
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="AnBmCk_general":
            # dates = [220906, 220907, 220908, 220909, 230811, 230829]
            dates = [220831, 220901, 230810, 230811, 230824, 230826, 230829, 231114, 231116, 220902, 220906, 220907, 220908, 220909, 240830, 250321, 250322] # ALL (4/6/25)
            question = "RULE_ANBMCK_STROKE"
            fr_normalization_method = "across_time_bins"
        elif save_suffix=="sh_vs_seqsup":
            # dates = [230920, 230921, 230923, 231019, 231020, 240828, 240829]
            # dates = [230920, 230921, 230923, 231019, 240828, 240829, 250324, 250325] # # ALL (3/18/25) -- 231020 failed for extraction.
            dates = [230920, 230921, 230923, 231019, 240828, 240829] # 250324, 250325  not yet kilosorted
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

    return dates, question, twind_analy, fr_normalization_method

def load_preprocess_concat_mult_sessions(animal, save_suffix, new_varied_hyperparams=True,
                                         dim_red_method=None, NPCS_KEEP=None, extra_dimred_method_n_components=None,
                                         umap_n_neighbors=None, savedir_suffix=None,
                                         skip_dates_dont_exist=False,
                                         new_mult_savedir_suffix = None, 
                                         HACK=False):
    """ 
    Loads, preprocesses (each individualy) and concats all sessions
    """
    # Get params
    which_level = "stroke"
    event = "00_stroke"
    spks = "kilosort_if_exists"
    combarea = False

    dates, question, twind_analy, fr_normalization_method = load_preprocess_get_dates(animal, save_suffix, new_varied_hyperparams=new_varied_hyperparams)

    # Load data across dates
    list_dfres = []
    list_dfres_pivot_distr = []
    list_dfres_pivot_pair = []
    list_dfres_pivot_yue = []
    dates_exist = []

    HACK_PANCHO_240825 = True # Just to get shape vs. seq, to decide on tasks to make for today.

    traj_or_scalar = "traj_to_scalar"
    # traj_or_scalar = "scalar"

    for d in dates:
        
        if HACK_PANCHO_240825 and int(d)<240827:
            SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/traj_to_scalar-wl=stroke-ev=00_stroke-spks=kilosort_if_exists-combarea=False/{question}-norm={fr_normalization_method}-dr=superv_dpca-NPC=6-nc=None-un=None-suff=syntax_role/twinda={twind_analy}-tbin=0.1"
            if not os.path.exists(f"{SAVEDIR}/DFRES.pkl"):
                SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{question}-fr_normalization_method={fr_normalization_method}-NPCS_KEEP=10/twind_analy={twind_analy}-tbin_dur=0.1"
                if not os.path.exists(f"{SAVEDIR}/DFRES.pkl"):
                    print(animal, d)
                    assert False
        else:
            if new_varied_hyperparams==False:
                # Old method, using pca-->umap, before adding variation in hyperparams
                SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{question}-fr_normalization_method={fr_normalization_method}-NPCS_KEEP=10/twind_analy={twind_analy}-tbin_dur=0.1"
            else:
                # - new version, look for this. if doesnt exist, then use old versoin

                if isinstance(NPCS_KEEP, (list, tuple)):
                    # Then iterate over thema nd take the first one that matches
                    _found = False
                    for _NPCS_KEEP in NPCS_KEEP:
                        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{traj_or_scalar}-wl={which_level}-ev={event}-spks={spks}-combarea={combarea}/{question}-norm={fr_normalization_method}-dr={dim_red_method}-NPC={_NPCS_KEEP}-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={savedir_suffix}/twinda={twind_analy}-tbin=0.1"
                        if os.path.exists(f"{SAVEDIR}/DFRES.pkl"):
                            print("FOund, using this NPCS_KEEP:", _NPCS_KEEP)
                            _found = True
                            break
                    if not _found and HACK:
                        # THen jkust use PCA, not DPCA
                        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{traj_or_scalar}-wl={which_level}-ev={event}-spks={spks}-combarea={combarea}/{question}-norm={fr_normalization_method}-dr=pca-NPC=10-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={None}/twinda={twind_analy}-tbin=0.1"
                else:
                    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{traj_or_scalar}-wl={which_level}-ev={event}-spks={spks}-combarea={combarea}/{question}-norm={fr_normalization_method}-dr={dim_red_method}-NPC={NPCS_KEEP}-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={savedir_suffix}/twinda={twind_analy}-tbin=0.1"
                    if not os.path.exists(f"{SAVEDIR}/DFRES.pkl"):        
                        # Old version
                        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/EUCLIDIAN_DIST/{animal}-{d}/{question}-norm={fr_normalization_method}-dr={dim_red_method}-NPC={NPCS_KEEP}-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={savedir_suffix}/twinda={twind_analy}-tbin=0.1"

        if not os.path.exists(f"{SAVEDIR}/DFRES.pkl"):
            if skip_dates_dont_exist:
                print("******** SKIPPING, doesnt exist:", d, SAVEDIR)
                continue
            else:
                print("******** MISSING, doesnt exist:", d, SAVEDIR)
                assert False    

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
        dates_exist.append(d)

    DFRES = pd.concat(list_dfres).reset_index(drop=True)
    DFRES_PIVOT_DISTR = pd.concat(list_dfres_pivot_distr).reset_index(drop=True)
    DFRES_PIVOT_PAIRWISE = pd.concat(list_dfres_pivot_pair).reset_index(drop=True)
    DFRES_PIVOT_YUE = pd.concat(list_dfres_pivot_yue).reset_index(drop=True)

    # Initialize savedir and save params
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
    params = {
            "animal":animal,
            "dates":dates,
            "dates_exist":dates_exist,
            "question":question,
            "fr_normalization_method":fr_normalization_method,
            "twind_analy":twind_analy,
            "save_suffix":save_suffix
        }

    SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_DIST/MULT/{animal}-{question}-{save_suffix}-{min(dates_exist)}-{max(dates_exist)}"
    if new_mult_savedir_suffix is not None:
        SAVEDIR_ANALYSIS += f"-{new_mult_savedir_suffix}"

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
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import params_pairwise_variables_for_plotting, _plot_pairwise_btw_levels_for_seqsup

    # (1) All the geneeral plots (combining all data, i.e., datapt = levo, not rigorous)
    _plot_all_results(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, plot_params, SAVEDIR_ANALYSIS, ONLY_ESSENTIALS=True)

    # (2) Pairwise (combining all data, i.e., datapt = levo, not rigorous)
    plot_pairwise_all_wrapper(DFRES, SAVEDIR_ANALYSIS)

    # (5) Clean histograms
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import plot_histograms_clean_wrapper
    # dat_level = "distr"
    effect_context = "diff|same"
    # bregions_plot = ["M1_m", "PMv_m", "PMd_p", "dlPFC_a", "vlPFC_p", ]
    # bregions_plot = ["M1_m", "dlPFC_a", "preSMA_a"]
    bregions_plot = ["M1_m", "preSMA_a"]

    list_yvar_dat_level = [
        ["dist_norm_95", "distr"],
        ["dist_norm_95", "pts_yue_diff"],
    ]
    for yvar, dat_level in list_yvar_dat_level:
        plot_histograms_clean_wrapper(DFRES, SAVEDIR_ANALYSIS, dat_level, effect_context,
                                      ythis=yvar, bregions_plot=bregions_plot)

    # (3) Plot pairwise data, (one datapt per date, rigorous)
    LIST_LIST_VVO_XY, LIST_dir_suffix = params_pairwise_variables_for_plotting()

    # (4) Pairwise, superv vs. nonsuperv, shape vs dir, etc.
    # NOTE: the agged version is already done above in plot_pairwise_all_wrapper. Here is doing split by bregion, one
    # dot per date.
    for VERSION in ["nosup_vs_sup", "shape_vs_dir", "nocol_vs_col"]:
    # for VERSION in ["nocol_vs_col"]:
        _plot_pairwise_btw_levels_for_seqsup(DFRES, SAVEDIR_ANALYSIS, VERSION=VERSION, one_subplot_per_bregion=True)

    # Slow - do this last.
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
            for DFTHIS, dat_lev, ythis in [
                # [DFRES, "distr", "dist_norm_95"],
                # [DFRES, "pts_yue_log", "dist"],
                [DFRES, "pts_yue_diff", "dist_norm_95"]
                ]:

                dfthis = DFTHIS[(DFTHIS["dat_level"] == dat_lev)]

                dir_suffix_this = f"{dir_suffix}--ver={version}-text={plot_text}-map_subplot={map_subplot is not None}-datlev={dat_lev}"
                _plot_pairwise_btw_vvo_general_MULT(dfthis, ythis, SAVEDIR_ANALYSIS, LIST_VVO_XY, dir_suffix_this,
                                                    plot_text=plot_text,
                                                    version=version,
                                                    map_subplot_var_to_new_subplot_var=map_subplot)
                plt.close("all")


if __name__=="__main__":

    animal = sys.argv[1]
    save_suffix = sys.argv[2] # a code name for this analysis (e.g., two_shape_sets)
    HACK = True
    skip_dates_dont_exist = False

    if False: # Good, PCA
        dim_red_method = "pca"
        NPCS_KEEP = 10
        extra_dimred_method_n_components = None
        umap_n_neighbors = None
        savedir_suffix = None
        new_mult_savedir_suffix = "PCA"
    else:
        dim_red_method = "superv_dpca"
        NPCS_KEEP = (6, 5, None)
        extra_dimred_method_n_components = None
        umap_n_neighbors = None
        savedir_suffix = "syntax_role"
        new_mult_savedir_suffix = f"{dim_red_method}-{save_suffix}-{NPCS_KEEP}"

    # (1) Load all data
    # DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, SAVEDIR_ANALYSIS, params, plot_params = load_preprocess_concat_mult_sessions(
    #     animal, save_suffix)    
    DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, SAVEDIR_ANALYSIS, params, plot_params =\
        load_preprocess_concat_mult_sessions(animal, save_suffix, new_varied_hyperparams=True,
                                            dim_red_method=dim_red_method, NPCS_KEEP=NPCS_KEEP, 
                                            extra_dimred_method_n_components=extra_dimred_method_n_components,
                                            umap_n_neighbors=umap_n_neighbors, savedir_suffix=savedir_suffix,
                                            skip_dates_dont_exist=skip_dates_dont_exist, new_mult_savedir_suffix=new_mult_savedir_suffix,
                                            HACK=HACK)


    # (2) Plot
    plot_all(DFRES, DFRES_PIVOT_DISTR, DFRES_PIVOT_PAIRWISE, DFRES_PIVOT_YUE, plot_params, SAVEDIR_ANALYSIS)