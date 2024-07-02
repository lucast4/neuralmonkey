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
from pythonlib.tools.exceptions import NotEnoughDataException

#### USER PARAMS
SPIKES_VERSION = "tdt" # Still improving ks curation.
LIST_WHICH_LEVEL = ["trial", "stroke", "stroke_off"]
FORCE_EXTRACT = 0
# PRE_DUR = -0.8
# POST_DUR = 0.8
SAVEDIR_BASE = "/gorilla1/analyses/recordings/main/anova"



def extract_snippets_all_sessions(MS, which_level, EVENTS_KEEP=None, FORCE_EXTRACT=0, do_save=True,
                                  LIST_SESSIONS=None, DEBUG=False, PRE_DUR = None, POST_DUR = None):
    """ To extract SP for each sn in MS, optioanlly loading or extracting from
    scratch
    PARAMS:
    - FORCE_EXTRACT, int, if 1 then reextracts even if finds saved version.
    - do_save, bool, save
    """

    
    if PRE_DUR is None:
        PRE_DUR = -0.8
    if POST_DUR is None:
        POST_DUR = 0.8

    animal = MS.animal()
    DATE = MS.date()

    # for session in range(len(MS.SessionsList)):
    if LIST_SESSIONS is None:
        LIST_SESSIONS = range(len(MS.SessionsList))

    LIST_SP = []
    for session in LIST_SESSIONS:
        sn = MS.SessionsList[session]

        if FORCE_EXTRACT==0:
            try:
                sp = load_snippet_single(sn, which_level)
                # Only skip extraction if the sites all match. e.g., maybe Kilosort
                # has been completed recently, then should redo Snippets.

                if sorted(sp.Sites) == sn.sitegetterKS_map_region_to_sites_MULTREG():
                    print("Saved SP has identical sites to SN -- no need to reextract!")
                    SKIP_EXTRACTION = True
                else:
                    print("Saved SP has DIFFERENT sites to SN -- DO REEXTRACT of SP!")
                    SKIP_EXTRACTION = False

                # Make sure have large enough window.
                if (sp.Params["list_pre_dur"][0] > PRE_DUR) or (sp.Params["list_post_dur"][0] < POST_DUR):
                    SKIP_EXTRACTION = False
                else:
                    SKIP_EXTRACTION = True

            except FileNotFoundError as err:
                # Then recompute
                print("No saved SP found -- DO EXTRACT of SP!")
                SKIP_EXTRACTION = False
            except NotEnoughDataException as err:
                # Then SP exists, but it is incompativale with SN, lkekly becuase
                # SN is now kiloosrt.
                print("SP incompatioble with SN (kilosoert?) -- DO EXTRACT of SP!")
                SKIP_EXTRACTION = False
            except Exception as err:
                raise err

        else:
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
        if False:
            D = sn.Datasetbeh

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

        ######### DO EXTRACTION
        SP = extraction_helper(sn, which_level, PRE_DUR=PRE_DUR, POST_DUR=POST_DUR,
                               EVENTS_KEEP=EVENTS_KEEP) 

        ##############################
        if do_save:
            SAVEDIR = f"{SAVEDIR_BASE}/by{which_level}/{animal}-{DATE}-sess_{session}"
            os.makedirs(SAVEDIR, exist_ok=True)

            # if detects already extracted, and can successfully load, then skips.
            if False:
                path = f"{SAVEDIR}/params_extraction.yaml"
                writeDictToYaml(params, path)
            SP.save_v2(SAVEDIR)

            # Delete from memory, causes OOM error.
            import gc
            del SP
            del sn
            del D
            del dataset_pruned_for_trial_analysis
            gc.collect()

        ####
        LIST_SP.append(SP)

    return LIST_SP

if __name__=="__main__":
    # To extract SP and save.

    #### USUALLY NOT CHANGE PARAMS
    LIST_SESSIONS = None
    DEBUG = False # runs fast

    animal = sys.argv[1]
    DATE = int(sys.argv[2])
    if len(sys.argv)>3:
        which_level = sys.argv[3]
        LIST_WHICH_LEVEL = [which_level]
    if len(sys.argv)>4:
        FORCE_EXTRACT = int(sys.argv[4])

    ####################### EXTRACT PARAMS
    # from neuralmonkey.metadat.analy.anova_params import params_getter_extraction
    # params = params_getter_extraction(animal, DATE, which_level, ANALY_VER)

    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal,
        units_metadat_fail_if_no_exist=True,
        spikes_version=SPIKES_VERSION)

    for which_level in LIST_WHICH_LEVEL:
        extract_snippets_all_sessions(MS, which_level, FORCE_EXTRACT=0, do_save=True,
                                      LIST_SESSIONS=LIST_SESSIONS, DEBUG=DEBUG)