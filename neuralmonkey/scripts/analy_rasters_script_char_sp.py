"""
11/5/24 - Rasters, spefifically for chars vs. sp/pig, doing carefuly pruning of shapes (similar across task_kinds, and
high quality) and ony plotting shapes with enough n across task_kind.

Does stroke level, makes two plots:
- SP vs. CHAR (all stroke index)
- PIG vs. CHAR (all stroke index)

Notebook: 241105_char_sp_rasters_snippets
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
from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

if __name__=="__main__":

    ############# USER PARAMS
    animal = sys.argv[1]
    date = int(sys.argv[2])

    shape_var = "shape_semantic_grp"
    n_min_trials_per_shape = 4

    ############# USUALYL HARD-CODED PARAMS

    ############# MODIFY THESE PARAMS

    ################## LOAD DATA
    MS = load_mult_session_helper(date, animal)
    from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
    which_level = "stroke"
    PRE_DUR = -0.5
    POST_DUR = 0.5
    EVENTS_KEEP = ["00_stroke"]
    DEBUG = False
    SP, _ = load_and_concat_mult_snippets(MS, which_level, EVENTS_KEEP, DEBUG = DEBUG, 
                                        prune_low_fr_sites=True, 
                                        REGENERATE_SNIPPETS=True, 
                                        PRE_DUR=PRE_DUR, POST_DUR=POST_DUR)

    # Clean up SP and extract features
    from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
    question = "CHAR_BASE_stroke"
    q_params = rsagood_questions_dict(animal, date, question)[question]

    D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
        ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"],
        substrokes_plot_preprocess=False)


    ############ CLEAN UP DATA
    ### Remove bad shapes and trials
    from neuralmonkey.scripts.analy_euclidian_chars_sp import behstrokes_map_clustshape_to_thresh, params_shapes_remove
    # (1) shapes that are clearly bad (e.g., wrong direction)
    # Hard coded shapes to remove
    shapes_remove = params_shapes_remove(animal, date, shape_var)
    print("Also removing tese shapes. by hand: ", shapes_remove)
    SP.DfScalar = SP.DfScalar[~SP.DfScalar[shape_var].isin(shapes_remove)]

    # (2) clust_sim_max threshold.
    map_clustshape_to_thresh = behstrokes_map_clustshape_to_thresh(animal)
    def good(x):
        sh = x["clust_sim_max_colname"]
        return x["clust_sim_max"] > map_clustshape_to_thresh[sh]
    SP.DfScalar["clust_sim_max_GOOD"] = [good(row) for i, row in SP.DfScalar.iterrows()]
    SP.DfScalar = SP.DfScalar[SP.DfScalar["clust_sim_max_GOOD"]==True].reset_index(drop=True)

    # Prune so that SP and CHAR have same shapes.
    DFSCALAR = SP.DfScalar.copy()
    for prune_version in ["sp_char", "pig_char"]:
        SP.DfScalar = DFSCALAR.copy()
        if prune_version == "sp_char_0":
            task_kinds = ["prims_single", "character"]
            fd = {"task_kind":task_kinds, "stroke_index":[0]}
        elif prune_version == "sp_char":
            task_kinds = ["prims_single", "character"]
            fd = {"task_kind":task_kinds}
        elif prune_version == "sp_pig":
            task_kinds = ["prims_single", "prims_on_grid"]
            fd = {"task_kind":task_kinds}            
        elif prune_version == "pig_char":
            task_kinds = ["prims_on_grid", "character"]
            fd = {"task_kind":task_kinds}
        elif prune_version == "pig_char_0":
            task_kinds = ["prims_on_grid", "character"]
            fd = {"task_kind":task_kinds, "stroke_index":[0]}
        elif prune_version == "pig_char_1plus":
            task_kinds = ["prims_on_grid", "character"]
            fd = {"task_kind":task_kinds, "stroke_index":list(range(1, 10))}
        else:
            assert False

        # (1) Prune to just the desired tasks
        SP.DfScalar = SP.DfScalar[SP.DfScalar["task_kind"].isin(task_kinds)].reset_index(drop=True)

        # # (2) Keep only shapes that appear across all task kinds
        plot_counts_heatmap_savepath = None
        df = SP.DfScalar[SP.DfScalar["chan"] == SP.Sites[0]].reset_index(drop=True)
        _dfout,_  = extract_with_levels_of_conjunction_vars_helper(df, "task_kind", [shape_var], 
                                                                n_min_per_lev=n_min_trials_per_shape,
                                                    plot_counts_heatmap_savepath=plot_counts_heatmap_savepath, 
                                                    levels_var=task_kinds)

        # Get the list of good shapes, and prune SP
        shapes_keep = _dfout[shape_var].unique().tolist()
        SP.DfScalar = SP.DfScalar[SP.DfScalar["shape_semantic_grp"].isin(shapes_keep)].reset_index(drop=True)

        ############## PLOTS
        from neuralmonkey.scripts.analy_rasters_script_wrapper import plotter
        import os
        vars_others = ["task_kind", "stroke_index"]
        event = "00_stroke"
        OVERWRITE_n_min = n_min_trials_per_shape
        OVERWRITE_lenient_n = 1

        SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/RASTERS/{animal}-{date}/CHAR_SP-prune={prune_version}"
        os.makedirs(SAVEDIR, exist_ok=True)
        plotter(SP, shape_var, vars_others, event, SAVEDIR, OVERWRITE_n_min, OVERWRITE_lenient_n)

        # ############## PLOTS
        # if MULTIPROCESS:
        #     for event in EVENTS_KEEP:
        #         print("Starting multipporcss for event", event)
        #         LIST_SP = [SP for _ in range(len(LIST_VAR))]
        #         LIST_OVERWRITE_n_min = [OVERWRITE_n_min for _ in range(len(LIST_VAR))]
        #         LIST_EVENT = [event for _ in range(len(LIST_VAR))]
        #         LIST_SAVEDIR = [SAVEDIR for _ in range(len(LIST_VAR))]
        #         from multiprocessing import Pool
        #         with Pool(MULTIPROCESS_N_CORES) as pool:
        #             pool.starmap(plotter, zip(LIST_SP, LIST_VAR, LIST_VARS_OTHERS, LIST_EVENT, LIST_SAVEDIR, LIST_OVERWRITE_n_min, LIST_OVERWRITE_lenient_n))
        # else:
        #     for var, vars_others, OVERWRITE_lenient_n in zip(LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n):
        #         for event in EVENTS_KEEP:
        #             plotter(SP, var, vars_others, event, SAVEDIR, OVERWRITE_n_min, OVERWRITE_lenient_n)