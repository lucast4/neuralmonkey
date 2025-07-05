"""
Goal: general purpose approach for plotting state space (scalar and trajc) variety of dim redutcion methods, all using
same plots.

This builds on analy_dpca_plot_script, which was specific for dpca

NOTEBOOK: 230227_snippets_statespace_tsne

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
from neuralmonkey.classes.population_mult import extract_single_pa, load_handsaved_wrapper
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels
from neuralmonkey.analyses.state_space_good import dimredgood_nonlinear_embed_data
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    fr_normalization_method = "across_time_bins" # Slightly better

    if animal=="Pancho":
        COMBINE_AREAS = False
        twind_stroke = (0., 0.3)
    elif animal == "Diego":
        COMBINE_AREAS = True
        twind_stroke = (-0.15, 0.35)
    else:
        assert False

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_TI/{animal}-{date}-combine={COMBINE_AREAS}"
    os.makedirs(SAVEDIR, exist_ok=True)

    #####################################
    # Method 2 - Combine two dfallpa
    DFallpa = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS)

    #################### PREPROCESSING
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)


    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import dfallpa_preprocess_condition
    shape_var_suff = "shape"
    loc_var_suff = "loc"
    dfallpa_preprocess_condition(DFallpa, shape_var_suff, loc_var_suff)

    list_bregion = DFallpa["bregion"].unique().tolist()
    
    for bregion in list_bregion:
        for raw_subtract_mean_each_timepoint in [False, True]:
        # for raw_subtract_mean_each_timepoint in [True]:
            # for scalar_or_traj in ["scal", "traj"]:
            for scalar_or_traj in ["scal"]:
                for event, twind_pca in [
                    ("03_samp", (0, 1.6)),
                    ("06_on_strokeidx_0", twind_stroke),
                    ]:
                # for event, twind_pca in [
                    # ("03_samp", (0, 1.6)),
                    # ]:
                    # for dim_red_method in ["pca", "dpca"]:
                    for dim_red_method in ["dpca"]:

                        PA = extract_single_pa(DFallpa, bregion, None, "trial", event)

                        # only successful trials
                        dflab = PA.Xlabels["trials"]
                        shapes_good = dflab["seqc_0_shape"].value_counts()[dflab["seqc_0_shape"].value_counts()>10].index.tolist()
                        # trialcode_good = dflab[dflab["FEAT_num_strokes_task"] == dflab["FEAT_num_strokes_beh"]]["trialcode"].tolist()
                        # trialcode_fail = dflab[dflab["FEAT_num_strokes_task"] > dflab["FEAT_num_strokes_beh"]]["trialcode"].tolist()

                        dpca_filtdict={
                            "seqc_0_shape":shapes_good,
                            # "trialcode":trialcode_good,
                            # "task_kind":["prims_on_grid"],
                            "success_binary_quick":[True],
                            }


                        # dim_red_method = "pca"
                        # dim_red_method = "dpca"

                        savedir = f"{SAVEDIR}/bregion={bregion}-subtrmean={raw_subtract_mean_each_timepoint}-scaltraj={scalar_or_traj}-event={event}-dimred={dim_red_method}"
                        os.makedirs(savedir, exist_ok=True)

                        umap_n_components = 3
                        dpca_var = "seqc_0_shape"
                        dpca_vars_group = None
                        dpca_proj_twind = None

                        # dpca_var = "seqc_1_shape"
                        # dpca_vars_group = ["seqc_0_shape"]

                        # dpca_filtdict={"epoch":["SSD3"]}
                        # dpca_filtdict=None

                        tbin_dur = 0.1
                        tbin_slide = 0.05

                        Xredu, PAredu = PA.dataextract_dimred_wrapper(scalar_or_traj, dim_red_method, savedir, 
                                                        twind_pca, tbin_dur, tbin_slide, 
                                                        NPCS_KEEP = 10,
                                                        dpca_var = dpca_var, dpca_vars_group = dpca_vars_group, dpca_filtdict=dpca_filtdict, dpca_proj_twind = dpca_proj_twind,
                                                        umap_n_components =umap_n_components,
                                                        raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)


                        ################## PLOTS
                        LIST_VAR = ["seqc_0_shape", "seqc_0_shape", "seqc_0_shape", "seqc_1_shape", "seqc_1_shape"]
                        LIST_VARS_OTHERS = [
                            ("task_kind", "FEAT_num_strokes_task", "success_binary_quick",), 
                            ("task_kind", "success_binary_quick",), 
                            ("task_kind",),
                            ("task_kind", "FEAT_num_strokes_task", "success_binary_quick",), 
                            ("task_kind", "FEAT_num_strokes_task", "seqc_0_shape", "success_binary_quick",), 
                            ]
                        LIST_FILTDICT = [
                            {"seqc_0_shape":shapes_good} for _ in range(len(LIST_VAR))
                        ]
                        PLOT_CLEAN_VERSION = False
                        time_bin_size = 0.05
                        PAredu.plot_state_space_good_wrapper(savedir, LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, 
                                                            PLOT_CLEAN_VERSION=PLOT_CLEAN_VERSION,
                                                            time_bin_size=time_bin_size)