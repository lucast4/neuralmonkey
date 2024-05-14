"""
Good script for anlaysis comparing distmats across time adn events, where distmats are 
(RSA) eucldiidant distances between beh states.
E.g,, question is whether PMv has same population geometry across time, even from samp to stroke
epochs. 

This is derived from analy_euclidian_dist_pop_script.py, and
notebook: 240217_snippets_euclidian...
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


PLOT_TRAJS = False
# SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
nmin_trials_per_lev = 3
DEBUG = False

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])

    if animal=="Diego":
        combine_into_larger_areas = True
    else:
        # becuase PMdA and preSMAp are diff from other one
        combine_into_larger_areas = False
        
    if combine_into_larger_areas:
        SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
    else:
        SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks

    exclude_bad_areas = False

    ############## HIGH-LEVEL PARAMS
    if DEBUG:
        LIST_TWIND = [(-0.3, 0.3)]
        LIST_TBIN_DUR = [0.2]
        LIST_TBIN_SLIDE = [0.2]
        EVENTS_KEEP = ["03_samp", "06_on_strokeidx_0"]
    else:
        LIST_TWIND = [(-0.6, 0.6)]
        # LIST_TBIN_DUR = [0.15]
        # LIST_TBIN_SLIDE = [0.05]
        LIST_TBIN_DUR = [0.1]
        LIST_TBIN_SLIDE = [0.025]
        EVENTS_KEEP = ["03_samp", "04_go_cue", "05_first_raise", "06_on_strokeidx_0"]


    ############### PARAMS (EUCLIDIAN)
    # DONT COMBINE, use questions.
    question = sys.argv[3]
    combine_trial_and_stroke = False
    which_level = sys.argv[4]
    dir_suffix = question

    HACK_RENAME_SHAPES = "CHAR" in question
    list_time_windows = LIST_TWIND
    

    ################ sanity check of params
    assert which_level == "trial", "HACKY, in progress"


    ###############################################################
    ## DIM REDUCTION PARAMS
    LIST_DIMRED_METHODS = [
        ["pca", 6, None, None, True, None, None]
    ]

    # - Append dPCA params
    LIST_SAVEDIR_SUFFIX = []
    LIST_SUPERV_DPCA_PARAMS = []

    savedir_suffix = f"shape"
    if which_level == "stroke":
        superv_dpca_params = {
            "superv_dpca_var":"shape",
            "superv_dpca_vars_group":["gridloc", "gridsize"],
            "superv_dpca_filtdict":None
        }
        LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        # # (1) gridloc
        # savedir_suffix = f"gridloc"
        # superv_dpca_params = {
        #     "superv_dpca_var":"gridloc",
        #     "superv_dpca_vars_group":["shape", "gridsize"],
        #     "superv_dpca_filtdict":None
        # }
        # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)
    elif which_level == "trial":
        savedir_suffix = f"seqc_0_shape"
        superv_dpca_params = {
            "superv_dpca_var":"seqc_0_shape",
            "superv_dpca_vars_group":["seqc_0_loc", "gridsize"],
            "superv_dpca_filtdict":None
        }
        LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)

        # # (1) gridloc
        # savedir_suffix = f"seqc_0_loc"
        # superv_dpca_params = {
        #     "superv_dpca_var":"seqc_0_loc",
        #     "superv_dpca_vars_group":["seqc_0_shape", "gridsize"],
        #     "superv_dpca_filtdict":None
        # }
        # LIST_SAVEDIR_SUFFIX.append(savedir_suffix)
        # LIST_SUPERV_DPCA_PARAMS.append(superv_dpca_params)
        
    else:
        assert False
    for savedir_suffix, superv_dpca_params in zip(LIST_SAVEDIR_SUFFIX, LIST_SUPERV_DPCA_PARAMS):
        LIST_DIMRED_METHODS.append(["superv_dpca", 4, None, None, True, savedir_suffix, superv_dpca_params])


    ###############################################
    # Load q_params
    from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
    q_params = rsagood_questions_dict(animal, date, question)[question]

    ############################################################
    ## VARIABLES
    if question in ["SP_shape_loc", "SP_shape_size"]:
        # which_level == "trial":
        VAR_EFFECT = "seqc_0_shape"
        VARS_OTHERS = ["seqc_0_loc", "gridsize"]
        CONTEXT = None
        FILTDICT = None

        # LIST_VAR_EFFECT = [
        #     "seqc_0_shape",
        #     "seqc_0_shape"
        # ]

        # LIST_VARS_OTHERS = []

    else:
        print(question)
        assert False

    ########################################## EXTRACT NORMALIZED DATA
    from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
    fr_normalization_method = "across_time_bins"
    DFALLPA = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows, which_level=which_level,
                                                events_keep=EVENTS_KEEP,
                                                combine_into_larger_areas=combine_into_larger_areas,
                                                exclude_bad_areas=exclude_bad_areas, SPIKES_VERSION=SPIKES_VERSION,
                                                HACK_RENAME_SHAPES=HACK_RENAME_SHAPES, fr_normalization_method=fr_normalization_method,
                                                path_to_save_example_fr_normalization=None)

    ######## STANDARD PREPROCESING.
    from neuralmonkey.classes.population_mult import dfallpa_preprocess_vars_conjunctions_extract
    dfallpa_preprocess_vars_conjunctions_extract(DFALLPA, which_level=which_level)

    ########## RUN
    # Iterate over all dim reduction methods
    # for NPCS_KEEP in LIST_NPCS_KEEP:
    for dim_red_method, NPCS_KEEP, extra_dimred_method_n_components, umap_n_neighbors, PLOT_STATE_SPACE, savedir_suffix, superv_dpca_params in LIST_DIMRED_METHODS:

        ###########
        SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/EUCLIDIAN_TIMEDISTMAT/{animal}-{date}/wl={which_level}-spks={SPIKES_VERSION}-combarea={combine_into_larger_areas}/{dir_suffix}-dr={dim_red_method}-NPC={NPCS_KEEP}-nc={extra_dimred_method_n_components}-un={umap_n_neighbors}-suff={savedir_suffix}"

        os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
        print("SAVING AT:", SAVEDIR_ANALYSIS)

        # Make a copy of all PA before normalization
        DFallpa = DFALLPA.copy()
        list_pa = [pa.copy() for pa in DFallpa["pa"]]
        DFallpa["pa"] = list_pa

        ########################################
        for twind_analy, tbin_dur, tbin_slice in zip(LIST_TWIND, LIST_TBIN_DUR, LIST_TBIN_SLIDE):

            SAVEDIR = f"{SAVEDIR_ANALYSIS}/twinda={twind_analy}-tbin={tbin_dur}/var={VAR_EFFECT}-ovar={'|'.join(VARS_OTHERS)}"

            ################# GET DFRES
            # Compute results new
            os.makedirs(SAVEDIR, exist_ok=True)
            print("THIS TBIN_DUR", SAVEDIR)

            # Save all the params
            from pythonlib.tools.expttools import writeDictToTxtFlattened
            path = f"{SAVEDIR}/params_var.txt"
            writeDictToTxtFlattened({
                "question":question,
                "q_params":q_params,
                "which_level":which_level,
                "HACK_RENAME_SHAPES":HACK_RENAME_SHAPES,
                "exclude_bad_areas":exclude_bad_areas,
                "SPIKES_VERSION":SPIKES_VERSION,
                "combine_into_larger_areas":combine_into_larger_areas,
                "fr_normalization_method":fr_normalization_method,
                "nmin_trials_per_lev":nmin_trials_per_lev,
                "savedir_suffix":savedir_suffix,
                "superv_dpca_params":superv_dpca_params,
                "VAR_EFFECT":VAR_EFFECT,
                "VARS_OTHERS":VARS_OTHERS,
                "CONTEXT":CONTEXT,
                "FILTDICT":FILTDICT,
                "NPCS_KEEP":NPCS_KEEP,
                "tbin_dur":tbin_dur,
                "tbin_slice":tbin_slice,
                "twind_analy":twind_analy,
                "dim_red_method":dim_red_method,
                "extra_dimred_method_n_components":extra_dimred_method_n_components,
                "umap_n_neighbors":umap_n_neighbors}, path)

            #### COLLECT DATA
            # from neuralmonkey.analyses.decode_good import euclidian_distance_compute
            from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_timedistmat
            list_dfres = []
            for i, row in DFallpa.iterrows():
                br = row["bregion"]
                tw = row["twind"]
                ev = row["event"]
                PA = row["pa"]

                print("bregion, twind_analy, tbin_dur: ", br, twind_analy, tbin_dur)

                savedir = f"{SAVEDIR}/each_region/{br}-twind_analy={twind_analy}"
                os.makedirs(savedir, exist_ok=True)

                PLOT_HEATMAPS = False
                dfres = euclidian_distance_compute_trajectories_timedistmat(PA, VAR_EFFECT, VARS_OTHERS, 
                                                    twind_analy, tbin_dur, tbin_slice, savedir, 
                                                    context=CONTEXT, filtdict=FILTDICT,
                                                    PLOT_TRAJS=PLOT_TRAJS, nmin_trials_per_lev=nmin_trials_per_lev,
                                                    NPCS_KEEP=NPCS_KEEP,
                                                    dim_red_method = dim_red_method, superv_dpca_params=superv_dpca_params)

                plt.close("all")
                if dfres is not None and len(dfres)>0:
                    dfres["bregion"] = br
                    dfres["twind"] = [tw for _ in range(len(dfres))]
                    dfres["twind_analy"] = [twind_analy for _ in range(len(dfres))]
                    dfres["event"] = ev
                    list_dfres.append(dfres)

            # Concat
            if len(list_dfres)>0:
                DFRES = pd.concat(list_dfres).reset_index(drop=True)

                # SAVE
                path = f"{SAVEDIR}/DFRES.pkl"
                pd.to_pickle(DFRES, path)
                print("Saved to: ", path)
            else:
                DFRES = None


            #####################################
            ##### PLOTS
            from neuralmonkey.analyses.state_space_good import euclidian_distance_plot_timedistmat_heatmaps
            euclidian_distance_plot_timedistmat_heatmaps(DFRES, SAVEDIR, sort_order=(0, 1))                

            # Get pairwise corr between distmats
            from neuralmonkey.analyses.state_space_good import euclidian_distance_plot_timedistmat_score_similarity
            for context_this in [VARS_OTHERS, None]:
                SAVEDIR_PLOTS = f"{SAVEDIR}/similarity_between_distmats-context={context_this}"
                print("RUNNING:", SAVEDIR_PLOTS)
                os.makedirs(SAVEDIR_PLOTS, exist_ok=True)
                dfres_corrs_btw_distmats = euclidian_distance_plot_timedistmat_score_similarity(DFRES, context_this,
                                                                                                DO_PLOTS=True, 
                                                                                                SAVEDIR_PLOTS=SAVEDIR_PLOTS)
