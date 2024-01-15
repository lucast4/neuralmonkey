from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper


for DATASET_PRUNE_SAME_BEH_ONLY in [False]:
    # DATASET_PRUNE_SAME_BEH_ONLY = False

    if DATASET_PRUNE_SAME_BEH_ONLY:
        PREFIX_SAVE = "rules_samebeh"
    else:
        PREFIX_SAVE = "rules_all"


    ### RULES
    DATE = "221020"
    dataset_beh_expt = "dirshapecolor1b"
    animal = "Pancho"


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
    for sn in MS.SessionsList:
        sn.datasetbeh_load_helper(dataset_beh_expt)

    # Get dataset
    # D = MS.datasetbeh_extract()


    def _dataset_extract_prune_rulesw(sn, plot_tasks=False, same_beh_only=True, n_min_trials_in_each_epoch=2):
        """
        Return pruned dataset (copy), to do rule swtiching analy
        matching motor beh, etc.
        """
        D = sn.Datasetbeh.copy()

        ##### Clean up tasks
        # 1) only tasks that are done in all epochs (generally, that span all combo of variables).
        # 2) only tasks with completed trials (not failures).
        # 3) only tasks that are correct (especially for probes).

        ##### Plot just tasks with common first stroke

        ##### Plot just for the tasks with common stim, diff behavior
        # Extract it
    #     df = SP.DfScalar
    #     # save it
    #     dfORIG = SP.DfScalar.copy()

        # only "correct" trials (based on actual sequence executed)
        print("############ TAKING ONLY CORRECT TRIALS")
        inds_correct = []
        inds_incorrect = []
        for ind in range(len(D.Dat)):
            dat = D.sequence_extract_beh_and_task(ind)
            taskstroke_inds_beh_order = dat["taskstroke_inds_beh_order"]
            taskstroke_inds_correct_order = dat["taskstroke_inds_correct_order"]

            if taskstroke_inds_beh_order==taskstroke_inds_correct_order:
                # correct
                inds_correct.append(ind)
            else:
                inds_incorrect.append(ind)

        print("correct, incorrect:", len(inds_correct), len(inds_incorrect))
        D.subsetDataframe(inds_correct)
        print("Dataset len:", len(D.Dat))

        print("############ TAKING ONLY NO SUPERVISION TRIALS")
        # Only during no-supervision blocks
        D.filterPandas({"supervision_stage_concise":["off|1|solid|0"]}, "modify")
        print("Dataset len:", len(D.Dat))

        # Only characters with same beh across rules.
        # 2) Extract takss done the same in mult epochs
        if same_beh_only:
            print("############ TAKING ONLY CHAR WITH SAME BEH ACROSS TRIALS")
            from pythonlib.dataset.dataset_preprocess.probes import _generate_map_taskclass
            mapper_taskname_epoch_to_taskclass = _generate_map_taskclass(D)

            list_task = D.Dat["character"].unique().tolist()
            list_epoch = D.Dat["epoch"].unique().tolist()
            print(list_task)
            print(list_epoch)

            dict_task_orders = {}
            for task in list_task:

                list_inds_each_epoch = []
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

                if plot_tasks:
                    ax = Task.plotStrokes(ordinal=True)
                    ax.set_title(f"{task}")

                dict_task_orders[task] = list_inds_each_epoch

            # pull out tasks which have same sequence
            tasks_same_sequence = []
            for taskname, list_seqs in dict_task_orders.items():
                if len(list(set(list_seqs)))==1:
                    tasks_same_sequence.append(taskname)
            print("\nTHese tasks foudn to have same seuqence across epochs: ")
            print(tasks_same_sequence)
            D.filterPandas({"character":tasks_same_sequence}, "modify")
            print("Dataset len:", len(D.Dat))

        # Only if have at least N trials per epoch
        print("############ TAKING ONLY CHAR with min n trials in each epoch")
        df = D.prune_min_ntrials_across_higher_levels("epoch", "character", n_min=n_min_trials_in_each_epoch)
        D.Dat = df

        # Final length of D.Dat
        print("Final n trials", len(D.Dat))
        return D


    from neuralmonkey.classes.snippets import datasetstrokes_extract
    import os
    import numpy as np
    import seaborn as sns


    # Which version to use?
    # QUESTION = "stroke"
    # QUESTION = "stroke"
    # QUESTION = "sequence"
    QUESTION = "rulesw"
    DEBUG = False
    EVENTS_SIMPLE = False # simple (baseline, planning, exec)
    SESSIONS = range(len(MS.SessionsList))

    #######################################
    list_features = ["trialcode", "epoch", "character", "taskgroup", "supervision_stage_concise"]
    sn = MS.SessionsList[0]
    if SESSIONS is None:
        SESSIONS = range(len(MS.SessionsList))

    if QUESTION=="stroke":
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
        list_features_get_conjunction = ["epoch"]
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
        list_features_get_conjunction = ["0_shape", "0_loc", "1_shape", "1_loc", "2_shape", "2_loc"]

    # make sure do extraction of relevant features.
    list_features = list(set(list_features + list_features_get_conjunction))
    print(list_features)

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
        if QUESTION in ["stroke"]:
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
        from neuralmonkey.classes.snippets import Snippets
        if QUESTION=="stroke":
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
        RES_ALL_CHANS = SP.modulation_compute_each_chan()
        

        OUT = SP.modulation_compute_higher_stats(RES_ALL_CHANS)

        # Plot and save
        # list_plots = ["summarystats", "heatmaps", "eachsite_allvars", 
        #             "eachsite_smfr", "eachsite_rasters"]
        list_plots = ["eachsite_rasters", "eachsite_raster_splitby_character", "eachsite_smfr_splitby_character"]
        # if QUESTION=="rulesw":
        #     # because interested in whether activity encodes the rule or the 
        #     # action plan (character)
        #     list_plots = list_plots + ["eachsite_smfr_splitby_character", "eachsite_raster_splitby_character"]
        SP.modulation_plot_all(RES_ALL_CHANS, OUT, SAVEDIR_SCALAR, list_plots=list_plots)
