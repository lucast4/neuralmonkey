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
from neuralmonkey.metadat.analy.anova_params import params_getter_plots
from neuralmonkey.classes.snippets import load_and_concat_mult_snippets

DEBUG = False

if __name__=="__main__":

    animal = sys.argv[1]    
    DATE = int(sys.argv[2])
    which_level = sys.argv[3]
    ANALY_VER = sys.argv[4]
    if len(sys.argv)>5:
        anova_interaction = sys.argv[5]
    else:
        anova_interaction = "n"

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
    params = params_getter_plots(animal, DATE, which_level, ANALY_VER, anova_interaction=anova_interaction)

    ######################################## RUN
    # assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)

    # %matplotlib inline
    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal)
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

    # Prune data based on beh trials.
    from neuralmonkey.classes.snippets import _dataset_extract_prune_general
    from pythonlib.dataset.analy_dlist import concatDatasets
    list_dataset = []
    for sn in MS.SessionsList:

        sn.Datasetbeh.behclass_preprocess_wrapper()

        if params["DO_SCORE_SEQUENCE_VER"]=="parses":
            sn.Datasetbeh.grammar_successbinary_score_parses()
        elif params["DO_SCORE_SEQUENCE_VER"]=="matlab":
            sn.Datasetbeh.grammar_successbinary_score_matlab()
        else:
            # dont score
            assert params["DO_SCORE_SEQUENCE_VER"] is None

        # THis moved to original extraction.
        # if params["taskgroup_reassign_simple_neural"]:
        #     # do here, so the new taskgroup can be used as a feature.
        #     sn.Datasetbeh.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
        #     print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
        #     sn.Datasetbeh.grouping_print_n_samples(["taskgroup", "probe"])

        dat_pruned = _dataset_extract_prune_general(sn, 
            list_superv_keep=params["list_superv_keep"], 
            preprocess_steps_append=params["preprocess_steps_append"],
            remove_aborts=params["remove_aborts"],
            list_superv_keep_full=params["list_superv_keep_full"], 
            )    

        # Sanity check that didn't remove too much data.
        npre = len(sn.Datasetbeh.Dat)
        npost = len(dat_pruned.Dat)
        if npost/npre<0.25:
            print(params)
            assert False, "dataset pruning removed >0.75 of data. Are you sure correct? Maybe removing a supervisiuon stage that is actually important?"

        list_dataset.append(dat_pruned)
    
    # concat the datasets 
    dataset_pruned_for_trial_analysis = concatDatasets(list_dataset)

    # Only keep these trialcodes
    trialcodes_keep = dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist()

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
                    get_z_score=params["get_z_score"],
                    trialcodes_keep=trialcodes_keep,
                    ANALY_VER=ANALY_VER,
                    params_to_save=params)
            if SP.DfScalarBeforeRemoveSuperv is not None:
                # then pruned. replace the original
                SP.DfScalar = SP.DfScalarBeforeRemoveSuperv
        except NotEnoughDataException as err:
            print("!! SKIPPING: ", var, vars_conjuction)
            if SP.DfScalarBeforeRemoveSuperv is not None:
                # then pruned. replace the original
                SP.DfScalar = SP.DfScalarBeforeRemoveSuperv
            pass