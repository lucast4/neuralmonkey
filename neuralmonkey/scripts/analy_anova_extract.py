""" New ANOVA, aligned to strokes.
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
import sys
from neuralmonkey.classes.snippets import load_snippet_single

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
    MS = load_mult_session_helper(DATE, animal,
        units_metadat_fail_if_no_exist=True)

    # for session in range(len(MS.SessionsList)):
    if LIST_SESSIONS is None:
        LIST_SESSIONS = range(len(MS.SessionsList))

    for session in LIST_SESSIONS:
        sn = MS.SessionsList[session]

        try:
            sp = load_snippet_single(sn, which_level)
            SKIP_EXTRACTION = True
        except Exception as err:
            # Then recompute
            SKIP_EXTRACTION = False

        if SKIP_EXTRACTION:
            print("** SKIPPING EXTRACTION, since was able to load snippets, for: ")
            print("(animal, DATE, which_level, session)")
            print(animal, DATE, which_level, session)
            continue
        else:
            print("** NOT SKIPPING EXTRACTION, since was not able to load snippets, for: ")
            print("(animal, DATE, which_level, session)")
            print(animal, DATE, which_level, session)

        ###################################
        D = sn.Datasetbeh

        if False:
                # Do all these before plotting... not here.
            if params["DO_SCORE_SEQUENCE_VER"]=="parses":
                D.grammar_successbinary_score_parses()
            elif params["DO_SCORE_SEQUENCE_VER"]=="matlab":
                D.grammar_successbinary_score_matlab()
            else:
                # dont score
                assert params["DO_SCORE_SEQUENCE_VER"] is None

            ######################## 
            D.behclass_preprocess_wrapper()
            if params["DO_EXTRACT_CONTEXT"]:
                D.seqcontext_preprocess()

            if params["taskgroup_reassign_simple_neural"]:
                # do here, so the new taskgroup can be used as a feature.
                D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
                print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
                D.grouping_print_n_samples(["taskgroup", "probe"])
                    
            # Merge epochs (i.e., rename them)
            for this in params["list_epoch_merge"]:
                D.supervision_epochs_merge_these(this[0], this[1], key=params["epoch_merge_key"],
                    assert_list_epochs_exist=False)

            # Assign each row of D a char_seq
            if params["DO_CHARSEQ_VER"] is not None:
                D.sequence_char_taskclass_assign_char_seq(ver=params["DO_CHARSEQ_VER"])

            # Extract epochsets
            if params["EXTRACT_EPOCHSETS"]:
                D.epochset_extract_common_epoch_sets(
                    trial_label=params["EXTRACT_EPOCHSETS_trial_label"],
                    n_max_epochs=params["EXTRACT_EPOCHSETS_n_max_epochs"],
                    merge_sets_with_only_single_epoch=params["EXTRACT_EPOCHSETS_merge_sets"],
                    merge_sets_with_only_single_epoch_name = ("LEFTOVER",))
                
            if params["DO_EXTRACT_EPOCHKIND"]:
                D.supervision_epochs_extract_epochkind()

        ##############################
        # if DEBUG:
        #     SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/bytrial/{animal}-{DATE}-sess_{session}-DEBUG"
        # else:
        SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/by{which_level}/{animal}-{DATE}-sess_{session}"
        # SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/bytrial/{animal}-{DATE}-sess_{session}"
        os.makedirs(SAVEDIR, exist_ok=True)

        # if detects already extracted, and can successfully load, then skips.

        from pythonlib.tools.expttools import writeDictToYaml
        path = f"{SAVEDIR}/params_extraction.yaml"
        writeDictToYaml(params, path)

        from neuralmonkey.classes.snippets import _dataset_extract_prune_general
        if False:
            # Old method, where pruned dataset before extraction. Instead, now I run on 
            # entire dataset, then do pruning before each plot.
            if True:
                list_superv_keep = "all" # DONT PRUNE!
            else:
                list_superv_keep = None # keep only not_training
            dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(sn, 
                list_superv_keep=list_superv_keep, 
                preprocess_steps_append=params["preprocess_steps_append"],
                remove_aborts=True)    
        else:
            # THIS DOES NOTHIGN. just copies dataset and does sanity_gridloc_identical
            preprocess_steps_append = ["sanity_gridloc_identical"]
            dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(sn, 
                list_superv_keep="all", 
                preprocess_steps_append=preprocess_steps_append,
                remove_aborts=False)    
            assert sn.Datasetbeh.Dat["trialcode"].tolist()==dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist(), "shold not modify for snuippets extraction"

        if DEBUG:
            sn._DEBUG_PRUNE_SITES = True
            dataset_pruned_for_trial_analysis.subsampleTrials(10, 1)

        from neuralmonkey.classes.snippets import Snippets, extraction_helper
        
        if True:
            # do this only before making plots
            list_features_modulation_append = None
        else:
            list_features_modulation_append = params["list_features_modulation_append"]

        # DEbugging, was missing a trialcode.
        # print("HASDSADASDASD")
        # TC = "230613-1-308"
        # print(sum(dataset_pruned_for_trial_analysis.Dat["trialcode"]==TC))
        # print(len(dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist()), dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist())
        # print(len(sn.Datasetbeh.Dat["trialcode"].tolist()), sn.Datasetbeh.Dat["trialcode"].tolist())
        # dataset_pruned_for_trial_analysis.Dat = dataset_pruned_for_trial_analysis.Dat[dataset_pruned_for_trial_analysis.Dat["trialcode"]==TC]

        SP = extraction_helper(sn, which_level, list_features_modulation_append, 
                               dataset_pruned_for_trial_analysis=dataset_pruned_for_trial_analysis, 
                               NEW_VERSION=True, 
                               PRE_DUR = params["PRE_DUR"], POST_DUR = params["POST_DUR"],
                               PRE_DUR_FIXCUE=params["PRE_DUR_FIXCUE"])

        SP.save_v2(SAVEDIR)

        # Delete from memory, causes OOM error.
        import gc
        del SP
        del sn
        del D
        del dataset_pruned_for_trial_analysis
        gc.collect()
