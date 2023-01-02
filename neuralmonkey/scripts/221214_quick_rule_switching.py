"""
Running a few specific days for:
    220815_rule_switching_biasdir
        (direction vs. shape)
In prep for monthly meeting (12/14/22).
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
from neuralmonkey.utils.monkeylogic import _load_session_mapper
import numpy as np

LIST_DATE = ["221020", "221031"]
LIST_EXPT = ["dirshapecolor1b", "dirshapecolor1h"]
animal = "Pancho"
# LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_allvars", "eachsite_smfr", "eachsite_rasters"]
LIST_PLOTS = ["summarystats", "heatmaps", "eachsite_allvars", "eachsite_smfr", "eachsite_rasters"]

# good but not fully preprocessed
for DATE, dataset_beh_expt in zip(LIST_DATE, LIST_EXPT):
    # DATE = "221020"
    # dataset_beh_expt = "dirshapecolor1b"

    # MS = load_mult_session_helper(DATE, animal, dataset_beh_expt)
    MS = load_mult_session_helper(DATE, animal, dataset_beh_expt)

    animal = "Pancho"
    SAVEDIR = f"/gorilla1/analyses/recordings/NOTEBOOKS/220815_rule_switching_biasdir/{animal}/{DATE}"
    import os
    os.makedirs(SAVEDIR, exist_ok=True)
    print(SAVEDIR)


    from neuralmonkey.classes.snippets import datasetstrokes_extract
    import os

    ##### PARAMAS
    # list_possible_features = ["gridsize", "shape_oriented", "gridloc"]
    list_sess = range(len(MS.SessionsList))
    sn = MS.SessionsList[0]
    if np.any(sn._behcode_extract_times(132)):
        # Then there are separated in time: cue alone --> strokes alone
        list_events = ["fixcue", "fix_touch", "fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        list_pre_dur = [-0.8, -0.45, 0.1, -0.75, 0.1, -0.55, -0.55, 0.05]
        list_post_dur = [-0.1, -0,  0.8, -0.05, 0.95, -0.05, -0.05, 0.55]
    else:
        list_events = ["fixcue", "fix_touch", "samp", "samp", "first_raise", "on_strokeidx_0", "on_strokeidx_0"]
        list_pre_dur = [-0.8, -0.45, -0.65, 0.1, -0.55, -0.55, 0.05]
        list_post_dur = [-0.1, -0, -0.05, 0.95, -0.05, -0.05, 0.55]
    list_features = ["trialcode", "epoch", "character", "taskgroup", "supervision_stage_concise"]
    list_features_get_conjunction = ["epoch"]

    for sess in list_sess:
        sn = MS.SessionsList[sess]
        
        SAVEDIR_SCALAR = f'{SAVEDIR}/sess_{sess}/scalar_plots'
        os.makedirs(SAVEDIR_SCALAR, exist_ok=True)
    #     savedir = f"{SAVEDIR_SCALAR}/modulation_by_features"
    #     os.makedirs(savedir, exist_ok=True)

        ##### 2) Do everything
        from neuralmonkey.classes.snippets import Snippets
        SP = Snippets(SN=sn, which_level="trial", list_events=list_events, 
                    list_features_extraction=list_features, list_features_get_conjunction=list_features_get_conjunction, 
                    list_pre_dur=list_pre_dur, list_post_dur=list_post_dur,
                    strokes_only_keep_single=False,
                    tasks_only_keep_these=None,
                    prune_feature_levels_min_n_trials=None)
        

        ##########################################
        # 1) Just the ones during no-supervision (well-trined)
        SP.PAscalar = SP.PAscalar.slice_by_label("trials", "supervision_stage_concise", "off|1|solid|0")
        SP.pascal_convert_to_dataframe(fr_which_version="sqrt")
        
        # Compute summary stats
        RES_ALL_CHANS = SP.modulation_compute_each_chan()
        OUT = SP.modulation_compute_higher_stats(RES_ALL_CHANS)

        SP.modulation_plot_all(RES_ALL_CHANS, OUT, SAVEDIR_SCALAR, 
            list_plots=LIST_PLOTS)
        
        
        ##########################################
        # 2) Extract takss done the same in mult epochs
        D = sn.Datasetbeh
        from pythonlib.dataset.dataset_preprocess.probes import _generate_map_taskclass
        mapper_taskname_epoch_to_taskclass = _generate_map_taskclass(D)

        list_task = D.Dat["character"].unique().tolist()
        list_epoch = D.Dat["epoch"].unique().tolist()
        print(list_task)
        print(list_epoch)

        dict_task_orders = {}
        for task in list_task:

            list_inds_each_epoch = []
            INCLUDE_THIS_TASK = True
            for epoch in list_epoch:

                if (task, epoch) not in mapper_taskname_epoch_to_taskclass.keys():
                    # Then this task has at least one epoch for which it doesnet havet rials.
                    # print((task, epoch))
                    # print(mapper_taskname_epoch_to_taskclass)
                    INCLUDE_THIS_TASK = False
                    # assert False, "should first exclude tasks that are not present across all epochs"
                else:
                    Task = mapper_taskname_epoch_to_taskclass[(task, epoch)]
                    inds_ordered = Task.ml2_objectclass_extract_active_chunk(return_as_inds=True)
                    list_inds_each_epoch.append(tuple(inds_ordered))

            # ax = Task.plotStrokes(ordinal=True)
            # ax.set_title(f"{task}")
            if INCLUDE_THIS_TASK:
                dict_task_orders[task] = list_inds_each_epoch


        # pull out tasks which have same sequence
        tasks_same_sequence = []
        for taskname, list_seqs in dict_task_orders.items():
            if len(list(set(list_seqs)))==1:
                tasks_same_sequence.append(taskname)

        print("\nTHese tasks foudn to have same seuqence across epochs: ")
        print(tasks_same_sequence)

        #### 2) Just those tasks with same beh in the different rules.
        SP.PAscalar = SP.PAscalar.slice_by_labels("trials", "character", tasks_same_sequence)
        SP.pascal_convert_to_dataframe(fr_which_version="sqrt")

        # Compute summary stats
        RES_ALL_CHANS = SP.modulation_compute_each_chan()
        OUT = SP.modulation_compute_higher_stats(RES_ALL_CHANS)
        
        # Plot
        SP.modulation_plot_all(RES_ALL_CHANS, OUT, SAVEDIR_SCALAR, 
            suffix="tasks_same_seq", list_plots=LIST_PLOTS)