""" One goal - extraction and saving of a DFallpa object, which can then be used
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
import pandas as pd
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion

############### PARAMS
exclude_bad_areas = True
SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
# SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
# combine_into_larger_areas = False
# Diego_use_main_21 = False # For char clust, which set of labels to use

# events_keep = None
# events_keep = ["03_samp", "05_first_raise", "06_on_strokeidx_0"]
# events_keep = ["03_samp", "05_first_raise", "06_on_strokeidx_0"]
fr_normalization_method = None
# fr_normalization_method = "across_time_bins"


def extract_dfallpa_helper(animal, date, question, combine_into_larger_areas, 
                           events_keep=None, do_save=True,
                           replace_fr_sm_with_spike_counts=False, 
                           spike_counts_bin_size=0.01):

    print("INPUT PARAMS:")
    print("animal, date, question, combine_into_larger_areas")
    print(animal, date, question, combine_into_larger_areas)

    print("... getting these events:")
    print(events_keep)

    if question=="PIG_BASE_saccade_fix_on":
        list_time_windows = [(-0.4, 0.4)] # to slice data including just within this time window (realtive to events)
    elif question=="SP_BASE_stroke":
        # Then want to get entire stroke (shorter, since some datasets did not use done button)
        list_time_windows = [(-0.5, 2.1)]
    elif question=="CHAR_BASE_stroke":
        # Then want to get entire stroke
        list_time_windows = [(-0.6, 2.5)]
    else:
        list_time_windows = [(-1.0, 1.8)]
        # list_time_windows = [(-0.8, 1.4)]
        # list_time_windows = [(-0.8, 1.25)]
        # list_time_windows = [(-1., 1.6)]


    ### Hard coded params
    do_combine = False

    print(":EHRE")
    if do_combine:
        assert False, "this doesnt work generally... something in code within..."
        # COMBINE trial and stroke
        dir_suffix = "test"
        question = None
        # q_params = None
        which_level = None
        q_params = {
            "effect_vars": ["seqc_0_shape", "seqc_0_loc"]
        }
        
        combine_trial_and_stroke = True
        
        # PIG
        # question_trial = "PIG_BASE_trial"
        # question_stroke = "PIG_BASE_stroke"
        # check_that_locs_match = True
        
        # CHAR
        question_trial = "PIG_BASE_trial"
        question_stroke = "PIG_BASE_stroke"
        check_that_locs_match = False
        check_that_shapes_match = False

        HACK_RENAME_SHAPES = "CHAR" in question_trial
        
        events_keep_trials = ["03_samp", "05_first_raise"]

    else:
        # DONT COMBINE, use questions.
        # question = "SS_shape"
        # question = "CHAR_BASE_stroke"
        # question = "CHAR_BASE_trial"
        # question = "SP_shape_loc"
        # question = "SP_BASE_stroke"
        # question = "SP_BASE_trial"
        # question = "PIG_BASE_trial"
        # question = "RULE_BASE_trial"
        # question = "PIG_BASE_stroke"
        # question = "RULE_ANBMCK_STROKE"
        
        # which_level = "trial" # Doesnt matter
        # which_level = "stroke" # Doesnt matter
        # which_level = "substroke" # Doesnt matter

        dir_suffix = question
        combine_trial_and_stroke = False
        # Load q_params
        from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
        q_params = rsagood_questions_dict(animal, date, question)[question]
        
        assert len(q_params["list_which_level"])==1, "then cant use this for auto..."
        which_level = q_params["list_which_level"][0]

        HACK_RENAME_SHAPES = "CHAR" in question
        # events_keep = ["00_stroke"]

        if events_keep is None:
            if which_level == "trial":
                events_keep = ["03_samp", "05_first_raise", "06_on_strokeidx_0"]
            elif which_level == "stroke":
                events_keep = ["00_stroke"]
            elif which_level == "saccade_fix_on":
                events_keep = ["00_fixon_preparation"]
            else:
                assert False

    if animal=="Diego":
        exclude_bad_areas = True
    elif animal=="Pancho" and combine_into_larger_areas==False:
        exclude_bad_areas = False
    elif animal=="Pancho" and combine_into_larger_areas==True:
        # exclude_bad_areas = True
        exclude_bad_areas = False # Decided to always include PMvl. Can filter after based on fr stats.
    else:
        print(animal)
        assert False

    ########################################## RUN

    if combine_trial_and_stroke:
        from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper_combine_trial_strokes
        DFallpa = dfallpa_extraction_load_wrapper_combine_trial_strokes(animal, date, question_trial,
                                                                        question_stroke,
                                                    list_time_windows, events_keep_trials=events_keep_trials,
                                                combine_into_larger_areas = combine_into_larger_areas,
                                                exclude_bad_areas=exclude_bad_areas,
                                                    SPIKES_VERSION=SPIKES_VERSION,
                                                    HACK_RENAME_SHAPES = HACK_RENAME_SHAPES,
                                                fr_normalization_method=fr_normalization_method,
                                                        check_that_shapes_match=check_that_shapes_match,
                                                    check_that_locs_match=check_that_locs_match)
    else:
        from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
        DFallpa = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows,
                                                which_level=which_level, events_keep=events_keep,
                                                combine_into_larger_areas = combine_into_larger_areas,
                                                exclude_bad_areas = exclude_bad_areas,
                                                SPIKES_VERSION = SPIKES_VERSION,
                                                HACK_RENAME_SHAPES = HACK_RENAME_SHAPES,
                                                fr_normalization_method=fr_normalization_method,
                                                replace_fr_sm_with_spike_counts=replace_fr_sm_with_spike_counts,
                                                spike_counts_bin_size=spike_counts_bin_size)
    
    if do_save:
        t1 = list_time_windows[0][0]
        t2 = list_time_windows[0][1]
        if replace_fr_sm_with_spike_counts:
            path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{which_level}-{SPIKES_VERSION}-norm={fr_normalization_method}-combine={combine_into_larger_areas}-t1={t1}-t2={t2}-quest={question}-spkcnts_binsz={spike_counts_bin_size}.pkl"
        else:
            path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{which_level}-{SPIKES_VERSION}-norm={fr_normalization_method}-combine={combine_into_larger_areas}-t1={t1}-t2={t2}-quest={question}.pkl"

        pd.to_pickle(DFallpa, path)
        print("*** Saved to:", path)        

    return DFallpa


if __name__=="__main__":
    # e..g, python analy_dfallpa_extract.py Pancho 240619 PIG_BASE_trial 0
    animal = sys.argv[1]
    date = int(sys.argv[2])
    question = sys.argv[3]
    combine_into_larger_areas = bool(int(sys.argv[4])) # 0, 1

    if len(sys.argv)>=6:
        get_all_events = int(sys.argv[5])==1
    else:
        get_all_events = False

    spike_counts_bin_size = 0.01
    if len(sys.argv)>=7:
        replace_fr_sm_with_spike_counts = bool(int(sys.argv[6])) # 
    else:
        replace_fr_sm_with_spike_counts = False

    # - To get fixations.
    # question = "PIG_BASE_saccade_fix_on" # holds variety of prepropoessing steps to clean data, specificalyl for PIG data.
    # which_level = "saccade_fix_on"

    FORCE_REEXTRACT = True
    which_level = "trial"

    if get_all_events:
        events_keep = ["03_samp", "go_cue", "05_first_raise", "06_on_strokeidx_0"]
    else:
        events_keep = None

    if FORCE_REEXTRACT:
        DFallpa = None
    else:
        # First try to load. If fails, then extract
        DFallpa = load_handsaved_wrapper(animal=animal, date=date, version=which_level, combine_areas=combine_into_larger_areas,
                                        return_none_if_no_exist=True)
    if DFallpa is None:
        DFallpa = extract_dfallpa_helper(animal, date, question, combine_into_larger_areas, events_keep=events_keep,
                                         do_save=True,
                                         replace_fr_sm_with_spike_counts=replace_fr_sm_with_spike_counts,
                                         spike_counts_bin_size=spike_counts_bin_size)

    # Save it
    # path = "/gorilla4/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa.pkl"
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa.pkl"
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa_2.pkl" # (tdt) (no norm)
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa_3.pkl" # (no norm)
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa_4.pkl" # (tdt)
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa_KS.pkl" # (kilosort)
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa_KS_nonorm.pkl" # (kilosort)

    # Dan: tough decoding, syntax stuff.
    # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa-{animal}-{date}-{which_level}-tdt_nonorm.pkl"

    # Xuan: Diego, char, good for testing tough shape decoding
    # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{which_level}-ks_nonorm.pkl"

    if False: # saving within above.
        t1 = list_time_windows[0][0]
        t2 = list_time_windows[0][1]
        path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{which_level}-{SPIKES_VERSION}-norm={fr_normalization_method}-combine={combine_into_larger_areas}-t1={t1}-t2={t2}.pkl"

        pd.to_pickle(DFallpa, path)
        print("*** Saved to:", path)

    # Also extract sitedirty preprocessing (e..g, fr drift metrics)
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_wrapper
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date, do_sitesdirty_extraction=True, spikes_version=SPIKES_VERSION)