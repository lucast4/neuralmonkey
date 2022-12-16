"""
Running a few specific days for:
    220713_prims_state_space
        (Size, shape, location) for single prims
In prep for monthly meeting (12/14/22).
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
from neuralmonkey.utils.monkeylogic import _load_session_mapper
import numpy as np


LIST_DATE = ["220715", "220716", "220717"]
# LIST_DATE = ["220717"]
dataset_beh_expt = "priminvar3"
animal = "Pancho"
# LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_allvars", "eachsite_smfr", "eachsite_rasters"]
LIST_PLOTS = ["eachsite_allvars", "eachsite_smfr", "eachsite_rasters"]

for DATE in LIST_DATE:

    # MS = load_mult_session_helper(DATE, animal, dataset_beh_expt)
    MS = load_mult_session_helper(DATE, animal, dataset_beh_expt)
        
        
    animal = "Pancho"
    # SAVEDIR = f"/data2/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}"
    SAVEDIR = f"/gorilla1/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{DATE}"

    import os
    print(SAVEDIR)

    from neuralmonkey.classes.snippets import datasetstrokes_extract
    import os

    ##### PARAMAS
    list_possible_features = ["gridsize", "shape_oriented", "gridloc"]
    list_sess = range(len(MS.SessionsList))
    list_events = ["samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
    list_pre_dur = [-0.4, 0.1, -0.55, -0.55, 0.05]
    list_post_dur = [-0.05, 0.65, -0.05, -0.05, 0.55]

    # list_events = ["fixcue", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
    # list_pre_dur = [-0.5, -0.3, 0.1, -0.55, -0.55, 0.05]
    # list_post_dur = [0., 0., 0.65, -0.05, 0.05, 0.55]

    for sess in list_sess:
        sn = MS.SessionsList[sess]

        SAVEDIR_SCALAR = f'/gorilla1/analyses/recordings/NOTEBOOKS/220713_prims_state_space/{animal}/{sn.Date}/sess_{sess}/scalar_plots'
        os.makedirs(SAVEDIR_SCALAR, exist_ok=True)

        # list_events = ["fixcue", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        # list_pre_dur = [-0.5, -0.3, 0.1, -0.55, -0.55, 0.05]
        # list_post_dur = [0., 0., 0.65, -0.05, 0.05, 0.55]

        ##### 1) First automatically figure out what features to use
        # - extract datstrokes, and check what fetures it has
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

        ##### 2) Do everything
        from neuralmonkey.classes.snippets import Snippets
        list_features_get_conjunction = list_features
        SP = Snippets(SN=sn, which_level="stroke", 
                      list_events=list_events, 
                     list_features_extraction=list_features, list_features_get_conjunction=list_features_get_conjunction,
                      list_pre_dur = list_pre_dur, list_post_dur=list_post_dur,
                     strokes_only_keep_single=True)

        # Compute summary stats
        RES_ALL_CHANS = SP.modulation_compute_each_chan(DEBUG=False)
        OUT = SP.modulation_compute_higher_stats(RES_ALL_CHANS)

        # Plot and save
        SP.modulation_plot_all(RES_ALL_CHANS, OUT, SAVEDIR_SCALAR, list_plots=LIST_PLOTS)
        