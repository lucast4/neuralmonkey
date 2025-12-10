"""
11/5/24 - Rasters, spefifically for grammar

Notebook: 251111_syntaxgrammar_rasters_snippets.ipynb
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
    n_min_trials_per_shape = 4

    ################## LOAD DATA
    MS = load_mult_session_helper(date, animal)
    from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
    which_level = "stroke"
    PRE_DUR = -0.8
    POST_DUR = 0.8
    EVENTS_KEEP = ["00_stroke"]
    DEBUG = False
    SP, _ = load_and_concat_mult_snippets(MS, which_level, EVENTS_KEEP, DEBUG = DEBUG, 
                                        prune_low_fr_sites=False, 
                                        PRE_DUR=PRE_DUR, POST_DUR=POST_DUR)

    # Clean up SP and extract features
    from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
    question = "RULE_ANBMCK_STROKE"
    q_params = rsagood_questions_dict(animal, date, question)[question]

    D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
        ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"],
        substrokes_plot_preprocess=False)

    ############## PLOTS
    from neuralmonkey.scripts.analy_rasters_script_wrapper import plotter
    import os

    event = "00_stroke"
    OVERWRITE_n_min = n_min_trials_per_shape
    OVERWRITE_lenient_n = 1

    LIST_VAR = [
        "chunk_n_in_chunk",
        "chunk_within_rank_fromlast",
    ]
    LIST_VARS_OTHERS = [
        ("shape", "chunk_within_rank_fromlast"),
        ("shape", "chunk_n_in_chunk"),
    ]

    # For some, should allow even if just 1 class.
    LIST_OVERWRITE_lenient_n = [OVERWRITE_lenient_n for _ in range(len(LIST_VARS_OTHERS))]

    SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/RASTERS/{animal}-{date}/{question}"
    os.makedirs(SAVEDIR, exist_ok=True)
    
    for var, vars_others, OVERWRITE_lenient_n in zip(LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n):
        for event in EVENTS_KEEP:
            plotter(SP, var, vars_others, event, SAVEDIR, OVERWRITE_n_min, OVERWRITE_lenient_n)
