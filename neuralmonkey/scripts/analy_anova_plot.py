""" 
New ANOVA, aligned to strokes.
Load SP previously saved by 230418_script_chunk_modulation
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
from pythonlib.tools.exceptions import NotEnoughDataException
import sys

DEBUG = False

if __name__=="__main__":

    animal = sys.argv[1]    
    DATE = int(sys.argv[2])
    which_level = sys.argv[3]
    ANALY_VER = sys.argv[4]
    anova_interaction = sys.argv[5]
    if anova_interaction=="y":
        anova_interaction = True
    elif anova_interaction=="n":
        anova_interaction = False
    else:
        assert False

    # DATE = 220709
    # animal = "Pancho"
    # which_level="trial"
    # ANALY_VER = "seqcontext"

    # score_ver='r2smfr_zscore'

    ################################### LOAD PARAMS
    from neuralmonkey.metadat.analy.anova_params import params_getter_plots
    params = params_getter_plots(animal, DATE, which_level, ANALY_VER, anova_interaction=anova_interaction)

    ######################################## RUN
    # assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)

    # %matplotlib inline
    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal)
    from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
    # SAVEDIR = "/gorilla1/analyses/recordings/main/chunks_modulation"
    SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level,
        DEBUG=DEBUG)
    
    from pythonlib.tools.expttools import writeDictToYaml
    path = f"{SAVEDIR_ALL}/params_plot.yaml"
    writeDictToYaml(params, path)

    if DEBUG:
        SP.Sites = SP.Sites[::20]
        print("new sites (subsampled): ", SP.Sites)
        # SAVEDIR_ALL = SAVEDIR_ALL + "-DEBUG"
        # print("NEW SAVE DIR (DEBUG):", SAVEDIR_ALL)

    #######################
    for var, vars_conjuction in zip(params["LIST_VAR"], params["LIST_VARS_CONJUNCTION"]):
        try:
            SP.modulationgood_compute_plot_ALL(var, vars_conjuction, 
                    params["score_ver"], SAVEDIR=SAVEDIR_ALL, 
                    PRE_DUR_CALC=params["PRE_DUR_CALC"], 
                    POST_DUR_CALC=params["POST_DUR_CALC"],
                    list_events=params["list_events"], list_pre_dur=params["list_pre_dur"], 
                    list_post_dur=params["list_post_dur"],
                    globals_nmin = params["globals_nmin"],
                    globals_lenient_allow_data_if_has_n_levels = params["globals_lenient_allow_data_if_has_n_levels"],
                    get_z_score=params["get_z_score"])
        except NotEnoughDataException as err:
            print("!! SKIPPING: ", var, vars_conjuction)
            pass