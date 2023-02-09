from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper


# -- motor kinematics (controlling for position of first touch)
# LIST_FEATURES_THIS = ["velmean_x", "velmean_y", "velmean_thbin", "velmean_norm"]
# LIST_FEATURES_THIS = ["velmean_x", "velmean_y", "velmean_thbin", "velmean_norm"]
# LIST_FEATURES_THIS = ["velmean_x", "velmean_y", "shape_oriented", "velmean_norm"]
# LIST_FEATURES_THIS = ["velmean_thbin", "velmean_normbin"]
# LIST_FEATURES_THIS = ["shape_oriented", "velmean_thbin", "velmean_normbin"]
LIST_FEATURES_THIS = ["shape_oriented", "velmean_thbin"]
PREFIX_SAVE = "primsingle_shape_thbin"

#################
DATE = "220719" 
dataset_beh_expt = None
animal = "Pancho"

# to help debug if times are misaligned.

# MS = load_mult_session_helper(DATE, animal, dataset_beh_expt)
MS = load_mult_session_helper(DATE, animal)
    
animal = "Pancho"
# SAVEDIR = f"/data2/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}"
if PREFIX_SAVE is None:
    SAVEDIR = f"/gorilla1/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}"
else:
    SAVEDIR = f"/gorilla1/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}_{PREFIX_SAVE}"
    
import os
print(SAVEDIR)


# [OPTIONAL] import dataset
# for sn in MS.SessionsList:
#     sn.datasetbeh_load_helper(dataset_beh_expt)
for sn in MS.SessionsList:
    sn.datasetbeh_load_helper(None)

# Get dataset
# D = MS.datasetbeh_extract()


DATASET_PRUNE_SAME_BEH_ONLY = False

if DATASET_PRUNE_SAME_BEH_ONLY:
    PREFIX_SAVE = "rules_samebeh"
else:
    PREFIX_SAVE = "rules_all"



from neuralmonkey.classes.snippets import datasetstrokes_extract
import os
import numpy as np
import seaborn as sns


# Which version to use?
# QUESTION = "stroke"
# QUESTION = "stroke"
# QUESTION = "sequence"
# QUESTION = "stroke"
QUESTION = "motor"
DEBUG = False
EVENTS_SIMPLE = False # simple (baseline, planning, exec)
SESSIONS = range(len(MS.SessionsList))
LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_allvars", 
            "eachsite_smfr", "eachsite_rasters"]
THINGS_TO_COMPUTE = ("modulation", "fr")

#######################################
list_features = ["trialcode", "epoch", "character", "taskgroup", "supervision_stage_concise"]
sn = MS.SessionsList[0]
if SESSIONS is None:
    SESSIONS = range(len(MS.SessionsList))

if QUESTION=="motor":
    ##### PARAMAS
    list_possible_features = LIST_FEATURES_THIS
    list_events = ["samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
    list_pre_dur = [-0.4, 0.1, -0.55, -0.55, 0.05]
    list_post_dur = [-0.05, 0.65, -0.05, -0.05, 0.55]
    
    if EVENTS_SIMPLE:
        list_events = ["samp", "samp", "on_strokeidx_0"]
        list_pre_dur = [-0.4, 0.1, 0.05]
        list_post_dur = [-0.05, 0.65, 0.55]
            
#     list_features_get_conjunction = ["shape_oriented", "velmean_x", "velmean_y"]
    list_features_get_conjunction = LIST_FEATURES_THIS
    
    LIST_PLOTS = ["summarystats", "heatmaps"] # cant do yet the others, they assume specific 
    # categorical levels.
    THINGS_TO_COMPUTE = ["modulation"]
    
elif QUESTION=="stroke":
    ##### PARAMAS
    list_possible_features = ["gridsize", "shape_oriented", "gridloc"]
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
    list_features_get_conjunction = ["epoch", "character"]
elif QUESTION=="sequence":
    if np.any(sn._behcode_extract_times(132)):
        # Then there are separated in time: cue alone --> strokes alone
        list_events = ["fixcue", "fix_touch", "fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        list_pre_dur = [-0.8, -0.45, 0.1, -0.75, 0.1, -0.55, -0.55, 0.05]
        list_post_dur = [-0.1, -0,  0.8, -0.05, 0.95, -0.05, -0.05, 0.55]
    else:
        list_events = ["fixcue", "fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        list_pre_dur = [-0.8, -0.45, -0.65, 0.1, -0.55, -0.55, 0.05]
        list_post_dur = [-0.1, -0, -0.05, 0.95, -0.05, -0.05, 0.55]
    list_features_get_conjunction = ["nstrok", "0_shape", "0_loc", "1_shape", "1_loc", "2_shape", "2_loc"]
else:
    assert False


if DEBUG:
    list_events = ["fixcue", "fix_touch"]
    list_pre_dur = [-0.8, -0.45]
    list_post_dur = [-0.1, -0]
#     list_features_get_conjunction = ["0_shape", "0_loc", "1_shape", "1_loc", "2_shape", "2_loc"]
#     list_features_get_conjunction = ["samp"]

# make sure do extraction of relevant features.
list_features = list(set(list_features + list_features_get_conjunction))
print(list_features)

if QUESTION=="rulesw":
    from pythonlib.tools.pandastools import grouping_print_n_samples
    sn = MS.SessionsList[0]
    D = _dataset_extract_prune_rulesw(sn, same_beh_only=DATASET_PRUNE_SAME_BEH_ONLY)
    grouping_print_n_samples(D.Dat, ["character", "epoch"]);


from neuralmonkey.classes.snippets import Snippets
for sess in SESSIONS:
    sn = MS.SessionsList[sess]

    # pass in already-pruned/preprocessed dataset?
    if QUESTION=="rulesw":
        dataset_pruned_for_trial_analysis = _dataset_extract_prune_rulesw(sn, same_beh_only=DATASET_PRUNE_SAME_BEH_ONLY)
    else:
        dataset_pruned_for_trial_analysis = None

    SAVEDIR_SCALAR = f'{SAVEDIR}/sess_{sess}/scalar_plots'
    os.makedirs(SAVEDIR_SCALAR, exist_ok=True)

    # list_events = ["fixcue", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
    # list_pre_dur = [-0.5, -0.3, 0.1, -0.55, -0.55, 0.05]
    # list_post_dur = [0., 0., 0.65, -0.05, 0.05, 0.55]

    ##### 1) First automatically figure out what features to use
    # - extract datstrokes, and check what fetures it has
    if QUESTION in ["motor", "stroke"]:
        strokes_only_keep_single=True
        tasks_only_keep_these=["prims_single"]
        prune_feature_levels_min_n_trials=20
        DS = datasetstrokes_extract(sn.Datasetbeh, strokes_only_keep_single,
                                    tasks_only_keep_these,
                                    prune_feature_levels_min_n_trials, list_possible_features)
        
        if len(DS.Dat)<50:
            continue

        # check which features exist
        list_features = []
        for feat in list_possible_features:
            levels = DS.Dat[feat].unique().tolist()
            if len(levels)>1:
                # keep
                list_features.append(feat)
        print("=== USING THESE FEATURES:", list_features)
        assert len(list_features)>0

    elif QUESTION in ["sequence", "rulesw"]:
        # DOesnt need DS
        pass 
    else:
        print(QUESTION)
        assert False
    

    ##### 2) Do everything
    if QUESTION in ["motor", "stroke"]:
        list_features_get_conjunction = list_features
        strokes_only_keep_single = True
        which_level = "stroke"
    elif QUESTION in ["sequence", "rulesw"]:
        strokes_only_keep_single = False
        which_level = "trial"
    else:
        assert False
        
    print("Extracvting snips..")
    SP = Snippets(SN=sn, which_level=which_level, list_events=list_events, 
                list_features_extraction=list_features, list_features_get_conjunction=list_features_get_conjunction, 
                list_pre_dur=list_pre_dur, list_post_dur=list_post_dur,
                strokes_only_keep_single=strokes_only_keep_single,
                  prune_feature_levels_min_n_trials=prune_feature_levels_min_n_trials,
                  dataset_pruned_for_trial_analysis = dataset_pruned_for_trial_analysis
                 )
    

    if dataset_pruned_for_trial_analysis is not None:
        tc1 = sorted(SP.DfScalar["trialcode"].unique().tolist())
        tc2 = sorted(dataset_pruned_for_trial_analysis.Dat["trialcode"].unique().tolist())
        assert tc1 == tc2, "Pruned dataset did not work correctly in Snippets"
        
    ##########################################
    # 1) Just the ones during no-supervision (well-trined)
#     if QUESTION=="rulesw":
#         # should to this to D above
#         SP.PAscalar = SP.PAscalar.slice_by_label("trials", "supervision_stage_concise", "off|1|solid|0")
#         SP.pascal_convert_to_dataframe(fr_which_version="sqrt")

    # Compute summary stats
    RES_ALL_CHANS = SP.modulation_compute_each_chan(things_to_compute=THINGS_TO_COMPUTE)
    

    OUT = SP.modulation_compute_higher_stats(RES_ALL_CHANS)

    # Plot and save
    if False:
        # Dont need to do this, since rules have character as an explicit variable
        if QUESTION=="rulesw":
            # because interested in whether activity encodes the rule or the 
            # action plan (character)
            list_plots = list_plots + ["eachsite_smfr_splitby_character", "eachsite_raster_splitby_character"]
    SP.modulation_plot_all(RES_ALL_CHANS, OUT, SAVEDIR_SCALAR, list_plots=LIST_PLOTS)
