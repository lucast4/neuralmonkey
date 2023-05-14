""" New ANOVA, aligned to strokes.
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
import sys


LIST_SESSIONS = None
DEBUG = False # runs fast
PLOT = False

assert PLOT == False, "do all plots using analy_anova_"

if __name__=="__main__":

    animal = sys.argv[1]    
    DATE = int(sys.argv[2])
    which_level = sys.argv[3]
    ANALY_VER = sys.argv[4]

    # DATE = 220709
    # animal = "Pancho"
    # which_level = "trial"
    # ANALY_VER = "trial_seqcontext"

    # # LIST_SESSIONS = [0, 1, 2, 3]
    # # SCORE_SEQUENCE_VER = "matlab"
    # # ASSIGN_CHUNK_INFO = False
    # SCORE_SEQUENCE_VER = "matlab"

    ####################### EXTRACT PARAMS
    from neuralmonkey.metadat.analy.anova_params import params_getter_extraction
    params = params_getter_extraction(animal, DATE, which_level, ANALY_VER)

    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal)

    # for session in range(len(MS.SessionsList)):
    if LIST_SESSIONS is None:
        LIST_SESSIONS = range(len(MS.SessionsList))

    for session in LIST_SESSIONS:
        sn = MS.SessionsList[session]

        ###################################
        D = sn.Datasetbeh

        if params["DO_SCORE_SEQUENCE_VER"]=="parses":
            D.grammar_successbinary_score_parses()
        elif params["DO_SCORE_SEQUENCE_VER"]=="matlab":
            D.grammar_successbinary_score_matlab()
        else:
            # dont score
            assert params["DO_SCORE_SEQUENCE_VER"] is None

        ######################## 
        if params["DO_EXTRACT_CONTEXT"]:
            # D.behclass_preprocess_wrapper()
            D.seqcontext_preprocess()

        # if DEBUG:
        #     SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/bytrial/{animal}-{DATE}-sess_{session}-DEBUG"
        # else:
        SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/bytrial/{animal}-{DATE}-sess_{session}"
        os.makedirs(SAVEDIR, exist_ok=True)

        from pythonlib.tools.expttools import writeDictToYaml
        path = f"{SAVEDIR}/params_extraction.yaml"
        writeDictToYaml(params, path)

        from neuralmonkey.classes.snippets import _dataset_extract_prune_general
        list_superv_keep = ["off|1|rank|0", "off|0||0", "off|1|solid|0"]
        # if ANALY_VER in ["trial_rulesw"]:
        #     preprocess_steps_append = ["correct_sequencing_binary_score", "one_to_one_beh_task_strokes"]
        # elif ANALY_VER in ["trial_seqcontext"]:
        #     preprocess_steps_append = ["one_to_one_beh_task_strokes"]
        # else:
        #     assert False
        dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(sn, 
            list_superv_keep=list_superv_keep, 
            preprocess_steps_append=params["preprocess_steps_append"])    

        if DEBUG:
            sn._DEBUG_PRUNE_SITES = True
            dataset_pruned_for_trial_analysis.subsampleTrials(10, 1)

        from neuralmonkey.classes.snippets import Snippets, extraction_helper
        PRE_DUR = -0.6
        POST_DUR = 0.6
        SP = extraction_helper(sn, "trial", params["list_features_modulation_append"], 
                               dataset_pruned_for_trial_analysis=dataset_pruned_for_trial_analysis, 
                               NEW_VERSION=True, 
                               PRE_DUR = PRE_DUR, POST_DUR = POST_DUR)

        SP.save_v2(SAVEDIR)

        if PLOT:
            ######## PLOTS
            var = "epoch"
            vars_conjuction = ['taskgroup'] # list of str, vars to take conjunction over
            # PRE_DUR_CALC = -0.25
            # POST_DUR_CALC = 0.25
            PRE_DUR_CALC = None
            POST_DUR_CALC = None
            score_ver='fracmod_smfr_minshuff'
            # score_ver='r2smfr_zscore'
            list_events = ["00_fix_touch", "01_samp", "03_first_raise"]
            list_pre_dur = [-0.5, 0.05, -0.1]
            list_post_dur = [-0, 0.6, 0.5]

            SP.modulationgood_compute_plot_ALL(var, vars_conjuction, score_ver, SAVEDIR=SAVEDIR, 
                                               PRE_DUR_CALC=PRE_DUR_CALC, 
                                               POST_DUR_CALC=POST_DUR_CALC,
                                              list_events=list_events, list_pre_dur=list_pre_dur, list_post_dur=list_post_dur)
