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
from neuralmonkey.metadat.analy.anova_params import params_getter_plots, params_getter_extraction
from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
from neuralmonkey.classes.snippets import _dataset_extract_prune_general, _dataset_extract_prune_general_dataset
from pythonlib.dataset.analy_dlist import concatDatasets
from pythonlib.tools.expttools import writeDictToYaml

DEBUG = False
DO_ONLY_PRINT_CONJUNCTIONS = False

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
    params_extraction = params_getter_extraction(animal, DATE, which_level, ANALY_VER)

    ######################################## RUN
    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal)
    SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level,
        DEBUG=DEBUG)
    
    # Save params
    path = f"{SAVEDIR_ALL}/params_plot.yaml"
    writeDictToYaml(params, path)

    if DEBUG:
        SP.Sites = SP.Sites[::20]
        print("new sites (subsampled): ", SP.Sites)
        # SAVEDIR_ALL = SAVEDIR_ALL + "-DEBUG"
        # print("NEW SAVE DIR (DEBUG):", SAVEDIR_ALL)

    ################# BEH DATASET
    # First, concatenate all D.
    list_dataset = []
    for sn in MS.SessionsList:
        D = sn.Datasetbeh

        # THINGs that must be done by each individual D
        D.behclass_preprocess_wrapper()

        # Second, do preprocessing to concatted D
        if params_extraction["DO_SCORE_SEQUENCE_VER"]=="parses":
            D.grammar_successbinary_score_parses()
        elif params_extraction["DO_SCORE_SEQUENCE_VER"]=="matlab":
            D.grammar_successbinary_score_matlab()
        else:
            # dont score
            assert params_extraction["DO_SCORE_SEQUENCE_VER"] is None

        if params_extraction["taskgroup_reassign_simple_neural"]:
            # do here, so the new taskgroup can be used as a feature.
            D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
            print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
            D.grouping_print_n_samples(["taskgroup", "probe"])

        if params_extraction["DO_CHARSEQ_VER"] is not None:
            D.sequence_char_taskclass_assign_char_seq(ver=params_extraction["DO_CHARSEQ_VER"])

        list_dataset.append(D.copy())
    # concat the datasets 
    Dall = concatDatasets(list_dataset)

    ################ DO SAME THING AS IN EXTRACTION (these dont fail, when use concatted)
    if params_extraction["DO_EXTRACT_CONTEXT"]:
        Dall.seqcontext_preprocess()

    for this in params_extraction["list_epoch_merge"]:
        # D.supervision_epochs_merge_these(["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
        Dall.supervision_epochs_merge_these(this[0], this[1], key=params_extraction["epoch_merge_key"],
            assert_list_epochs_exist=False)


    if params_extraction["EXTRACT_EPOCHSETS"]:
        Dall.epochset_extract_common_epoch_sets(
            trial_label=params_extraction["EXTRACT_EPOCHSETS_trial_label"],
            n_max_epochs=params_extraction["EXTRACT_EPOCHSETS_n_max_epochs"],
            merge_sets_with_only_single_epoch=params_extraction["EXTRACT_EPOCHSETS_merge_sets"],
            merge_sets_with_only_single_epoch_name = ("LEFTOVER",))

    # Sanity check that didn't remove too much data.
    if False:
        if "wrong_sequencing_binary_score" not in params["preprocess_steps_append"]:
            # Skip if is error trials.
            npre = len(D.Dat)
            npost = len(dat_pruned.Dat)
            if npost/npre<0.25 and len(sn.Datasetbeh.Dat)>200: # ie ignore this if it is a small session...
                print(params)
                print("THis has no wrong_sequencing_binary_score: ",  params['preprocess_steps_append'])
                assert False, "dataset pruning removed >0.75 of data. Are you sure correct? Maybe removing a supervisiuon stage that is actually important?"
    

    ####################### REEXTRACT VARIABLES
    # First collect all variables that you might need (before deleting dataset).
    # By default, recompute them, since concatting datasets might change some variables.
    def _reextract_var(SP, varthis):
        """ Repopulate SP.DfScalar[varthis] with the new values in
        dataset.
        Modifies SP in place
        """
        from pythonlib.tools.pandastools import slice_by_row_label
        
        trialcodesthis = SP.DfScalar["trialcode"].tolist()

        # keep onlyt the trialcodes that exist in dataset (after pruning)
        # tmp = Dall.Dat["trialcode"].tolist()
        # trialcodesthis = [tc for tc in trialcodesthis if tc in tmp]

        # Get the sliced dataframe
        dfslice = slice_by_row_label(Dall.Dat, "trialcode", trialcodesthis,
            reset_index=True, assert_exactly_one_each=True)
        
        # sanity check
        if False:
            # is now done in slice_by_row_label
            assert dfslice["trialcode"].tolist() == trialcodesthis

        # Assign the values to SP
        print("Updating this column of SP.DfScalar with Dataset beh:")
        print(varthis)
        SP.DfScalar[varthis] = dfslice[varthis].tolist()

    vars_already_extracted =[]
    for var, vars_conjuction in zip(params["LIST_VAR"], params["LIST_VARS_CONJUNCTION"]):

        # If any of these vars dont exist, try to extract them again from dataset
        # if var not in SP.DfScalar.columns:
        #     valuesthis = _reextract_var(SP, var)
        #     SP.DfScalar[var] = valuesthis
        # for v in vars_conjuction:
        #     if v not in SP.DfScalar.columns:
        #         valuesthis = _reextract_var(SP, v)
        #         SP.DfScalar[v] = valuesthis

        if var not in vars_already_extracted:
            _reextract_var(SP, var)
            vars_already_extracted.append(var)

        for v in vars_conjuction:
            if v not in vars_already_extracted:
                _reextract_var(SP, v)
                vars_already_extracted.append(v)

    ###### PRUNE DATASET TO GET SUBSET TRIALCODES
    # Only keep subset these trialcodes
    dataset_pruned_for_trial_analysis = _dataset_extract_prune_general_dataset(Dall, 
        list_superv_keep=params["list_superv_keep"], 
        preprocess_steps_append=params["preprocess_steps_append"],
        remove_aborts=params["remove_aborts"],
        list_superv_keep_full=params["list_superv_keep_full"], 
        )    
    TRIALCODES_KEEP = dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist()

    ###### SANITY CHECK, the type of each item for each var, must be the same across levels.
    # or else errors, e.g., seaborn fails.
    for var, vars_conjuction in zip(params["LIST_VAR"], params["LIST_VARS_CONJUNCTION"]):
        tmp = SP.DfScalar[var].unique().tolist()
        if len(set([type(x) for x in tmp]))>1:
            print(tmp)
            print([type(x) for x in tmp])
            print(var)
            assert False, "levels are not all same type..."

        for v in vars_conjuction:
            tmp = SP.DfScalar[v].unique().tolist()
            if len(set([type(x) for x in tmp]))>1:
                print(tmp)
                print([type(x) for x in tmp])
                print(v)
                assert False, "levels are not all same type..."


    # Delete MS from memory, causes OOM error.
    import gc
    del MS
    del sn
    del D
    del dataset_pruned_for_trial_analysis
    del list_dataset
    del Dall
    # del dat_pruned
    gc.collect()

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
                    trialcodes_keep=TRIALCODES_KEEP,
                    ANALY_VER=ANALY_VER,
                    params_to_save=params,
                    do_only_print_conjunctions=DO_ONLY_PRINT_CONJUNCTIONS)
            if SP.DfScalarBeforeRemoveSuperv is not None:
                # then pruned. replace the original
                SP.DfScalar = SP.DfScalarBeforeRemoveSuperv
        except NotEnoughDataException as err:
            print("!! SKIPPING: ", var, vars_conjuction)
            if SP.DfScalarBeforeRemoveSuperv is not None:
                # then pruned. replace the original
                SP.DfScalar = SP.DfScalarBeforeRemoveSuperv
            pass