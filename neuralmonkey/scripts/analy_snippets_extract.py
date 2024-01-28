""" One goal - extraction of snippets, which can then be used
for any analyses.
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
import sys
from neuralmonkey.classes.snippets import load_snippet_single
from pythonlib.tools.expttools import writeDictToYaml
from neuralmonkey.classes.snippets import Snippets, extraction_helper
from neuralmonkey.classes.snippets import _dataset_extract_prune_general

LIST_SESSIONS = None
DEBUG = False # runs fast

LIST_WHICH_LEVEL = ["trial", "stroke", "stroke_off"]

if __name__=="__main__":

    animal = sys.argv[1]
    DATE = int(sys.argv[2])

    if len(sys.argv)>3:
        which_level = sys.argv[3]
        LIST_WHICH_LEVEL = [which_level]

    ####################### EXTRACT PARAMS
    # from neuralmonkey.metadat.analy.anova_params import params_getter_extraction
    # params = params_getter_extraction(animal, DATE, which_level, ANALY_VER)

    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal,
        units_metadat_fail_if_no_exist=True)

    # for session in range(len(MS.SessionsList)):
    if LIST_SESSIONS is None:
        LIST_SESSIONS = range(len(MS.SessionsList))

    for session in LIST_SESSIONS:
        sn = MS.SessionsList[session]

        for which_level in LIST_WHICH_LEVEL:

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

            ##############################
            SAVEDIR = f"/gorilla1/analyses/recordings/main/anova/by{which_level}/{animal}-{DATE}-sess_{session}"
            os.makedirs(SAVEDIR, exist_ok=True)

            # if detects already extracted, and can successfully load, then skips.
            if False:
                path = f"{SAVEDIR}/params_extraction.yaml"
                writeDictToYaml(params, path)

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

            SP = extraction_helper(sn, which_level)

            SP.save_v2(SAVEDIR)

            # Delete from memory, causes OOM error.
            import gc
            del SP
            del sn
            del D
            del dataset_pruned_for_trial_analysis
            gc.collect()
