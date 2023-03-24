""" Site-level analysis of variance explained, 
flexibly use different kinds of params
Relates to notebook: 220713_prims_state_space.ipynb
"""
import numpy as np
from neuralmonkey.classes.snippets import _dataset_extract_prune_general, _dataset_extract_prune_sequence, _dataset_extract_prune_rulesw

def params_database_extract(MS, QUESTION, EVENTS_SIMPLE, 
    DATASET_PRUNE_SAME_BEH_ONLY, REMOVE_BASELINE_EPOCHS, 
    n_min_trials_in_each_epoch, first_stroke_same_only,
    DEBUG=False, MINIMAL_PLOTS=False):
    """ Hold hard-coded params for different kidns of experimemnts.
    """
    # Which version to use?
    # QUESTION = "stroke"
    # QUESTION = "stroke"
    # QUESTION = "sequence"
    # QUESTION = "stroke"
    # QUESTION = "motor"
    # DEBUG = True
    # EVENTS_SIMPLE = True # simple (baseline, planning, exec)
    SESSIONS = range(len(MS.SessionsList))
    # LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_allvars", 
    #             "eachsite_smfr", "eachsite_rasters"]
    # THINGS_TO_COMPUTE = ("modulation", "fr")
    THINGS_TO_COMPUTE = ("modulation")
    if MINIMAL_PLOTS:
        LIST_PLOTS = ["summarystats", "heatmaps"]
    else:
        # LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_smfr", "eachsite_rasters"]
        LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_smfr", "eachsite_rasters"]
    
    HACK = True
    if HACK:
        # LIST_PLOTS = ["eachsite_allvars"]
        # LIST_PLOTS = ["summarystats", "heatmaps"]
        LIST_PLOTS = ["eachsite_smfr", "eachsite_rasters", "eachsite_allvars"]
        # LIST_PLOTS = ["eachsite_rasters"] 
        # LIST_PLOTS = [] # just save

    #######################################
    list_features_extraction = ["trialcode", "epoch", "character", "taskgroup", "supervision_stage_concise"]
    list_possible_features_datstrokes = None
    sn = MS.SessionsList[0]
    if SESSIONS is None:
        SESSIONS = range(len(MS.SessionsList))

    if QUESTION=="motor":
        ##### PARAMAS
        list_possible_features_datstrokes = ["shape_oriented", "velmean_x", "velmean_y"]
        list_events = ["samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        list_pre_dur = [-0.4, 0.1, -0.55, -0.55, 0.05]
        list_post_dur = [-0.05, 0.65, -0.05, -0.05, 0.55]
        
        if EVENTS_SIMPLE:
            list_events = ["samp", "samp", "on_strokeidx_0"]
            list_pre_dur = [-0.4, 0.1, 0.05]
            list_post_dur = [-0.05, 0.65, 0.55]
                
    #     list_features_get_conjunction = ["shape_oriented", "velmean_x", "velmean_y"]
        list_features_get_conjunction = ["velmean_x", "velmean_y"]
        
        LIST_PLOTS = ["summarystats", "heatmaps"] # cant do yet the others, they assume specific 
        # categorical levels.
        THINGS_TO_COMPUTE = ["modulation"]
        
    elif QUESTION=="stroke":
        ##### PARAMAS
        list_possible_features_datstrokes = ["gridsize", "shape_oriented", "gridloc"]
        list_events = ["samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        list_pre_dur = [-0.4, 0.1, -0.55, -0.55, 0.05]
        list_post_dur = [-0.05, 0.65, -0.05, -0.05, 0.55]
        
        if EVENTS_SIMPLE:
            list_events = ["samp", "samp", "on_strokeidx_0"]
            list_pre_dur = [-0.4, 0.1, 0.05]
            list_post_dur = [-0.05, 0.65, 0.55]
            
        # list_events = ["fixcue", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        # list_pre_dur = [-0.5, -0.3, 0.1, -0.55, -0.55, 0.05]
        # list_post_dur = [0., 0., 0.65, -0.05, 0.05, 0.55]
        
        list_features_get_conjunction = ["gridsize", "shape_oriented", "gridloc"]

    elif QUESTION=="rulesw":
        if np.any(sn._behcode_extract_times(132)):
            # Then there are separated in time: cue alone --> strokes alone
            list_events = ["fixcue", "fixcue", "fix_touch", "fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
            list_pre_dur = [-0.8, 0.05, -0.45, 0.1, -0.75, 0.1, -0.55, -0.55, 0.05]
            list_post_dur = [-0.1, 0.75, -0,  0.8, -0.05, 0.95, -0.05, -0.05, 0.55]
        else:
            list_events = ["fixcue", "fixcue", "fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
            list_pre_dur = [-0.8, 0.05, -0.45, -0.65, 0.1, -0.55, -0.55, 0.05]
            list_post_dur = [-0.1, 0.75, -0, -0.05, 0.95, -0.05, -0.05, 0.55]
        
        # list_features_get_conjunction = ["epoch", "character"]
        list_features_get_conjunction = ["epoch"]


    elif QUESTION=="sequence":
        list_events = ["fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_1"]
        list_pre_dur = [-0.45, -0.65, 0.1, -0.55, -0.25, -0.25]
        list_post_dur = [-0, -0.05, 0.95, -0.05, 0.25, 0.25]
        # list_features_get_conjunction = ["1_loc", "1_shape"]
        list_features_get_conjunction = ["0_shape", "0_loc", "1_shape"]
    else:
        assert False


    if DEBUG:
        list_events = ["fixcue", "fix_touch"]
        list_pre_dur = [-0.8, -0.45]
        list_post_dur = [-0.1, -0]
    #     list_features_get_conjunction = ["0_shape", "0_loc", "1_shape", "1_loc", "2_shape", "2_loc"]
    #     list_features_get_conjunction = ["samp"]

    # make sure do extraction of relevant features.
    list_features_extraction = list(set(list_features_extraction + list_features_get_conjunction))
    print("list_features_extraction: ", list_features_extraction)

    if QUESTION=="rulesw":
        # Run this to simlate what the rueslting data would be.
        from pythonlib.tools.pandastools import grouping_print_n_samples
        sn = MS.SessionsList[0]
        D = _dataset_extract_prune_rulesw(sn, same_beh_only=DATASET_PRUNE_SAME_BEH_ONLY,
            n_min_trials_in_each_epoch=n_min_trials_in_each_epoch, 
            remove_baseline_epochs=REMOVE_BASELINE_EPOCHS,
            first_stroke_same_only=first_stroke_same_only)
        # grouping_print_n_samples(D.Dat, ["character", "epoch"]);

    # SAVE PARAMS
    PARAMS = {
        "QUESTION":QUESTION,
        "DEBUG":DEBUG, 
        "SESSIONS":SESSIONS,
        "LIST_PLOTS":LIST_PLOTS,
        "THINGS_TO_COMPUTE":THINGS_TO_COMPUTE,
        "list_events":list_events,
        "list_pre_dur":list_pre_dur,
        "list_post_dur":list_post_dur,
        "list_features_get_conjunction":list_features_get_conjunction,
        "list_features_extraction":list_features_extraction,
        "list_possible_features_datstrokes":list_possible_features_datstrokes,
    }
    return PARAMS



def save_all(SP, RES_ALL_CHANS, SAVEDIR_SCALAR):
    """ Help save data
    """
    import pickle
    # 1) SP
    SP.save(SAVEDIR_SCALAR)

    # 2) RES_ALL_CHANS
    path = f"{SAVEDIR_SCALAR}/RES_ALL_CHANS.pkl"
    print("SAving: ", path)
    with open(path, "wb") as f:
        pickle.dump(RES_ALL_CHANS, f)

    print("Done...!")
