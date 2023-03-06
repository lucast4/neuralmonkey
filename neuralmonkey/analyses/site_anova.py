""" Site-level analysis of variance explained, 
flexibly use different kinds of params
Relates to notebook: 220713_prims_state_space.ipynb
"""
import numpy as np

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


def _dataset_extract_prune_sequence(sn, n_strok_max = 2):
    """ Prep beh dataset before extracting snippets.
    Add columns that are necessary and
    return pruned datsaet for sequence analysis.
    """

    import pandas as pd
    D = sn.Datasetbeh

#     # 1) Convert to dataset strokes (task variant)
#     from pythonlib.dataset.dataset_strokes import DatStrokes
#     # DSbeh = DatStrokes(D, version="beh")
#     DStask = DatStrokes(D, version="task") 

#     # 1) only prims in grid
#     DStask.dataset_append_column("supervision_stage_concise")
#     filtdict = {"task_kind":["prims_on_grid"], "supervision_stage_concise":["off|1|solid|0"]}
#     DStask.filter_dataframe(filtdict, True)


    # Method 1 - level of trial
    # for each trial, extract 
    datall = []
    list_dat = []
    for i in range(len(D.Dat)):

        dat = {}
        # trial-level info
        # trialcode = D.Dat.iloc[i]["trialcode"]   
        tokens_behft = D.taskclass_tokens_extract_wrapper(i)
        if False:
            dat["nstrokes_beh"] = len(D.Dat.iloc[i]["strokes_beh"])
        else:
            # Better, beucase there can be mismatch sometimes.
            dat["nstrokes_beh"] = len(tokens_behft)

        dat["nstrokes_task"] = len(D.Dat.iloc[i]["strokes_task"])

        # shapes in order
        for i in range(n_strok_max):
            if i<len(tokens_behft):
                tok = tokens_behft[i]
                dat[f"{i}_shape"] = tok["shape"]
                dat[f"{i}_loc"] = tok["gridloc"]
            else:
                dat[f"{i}_shape"] = None
                dat[f"{i}_loc"] = None

        list_dat.append(dat)

    # Put back into D.Dat
    dfdat = pd.DataFrame(list_dat)
    D.Dat["nstrokes_beh"] = dfdat["nstrokes_beh"]
    D.Dat["nstrokes_task"] = dfdat["nstrokes_task"]
    for i in range(n_strok_max):
        D.Dat[f"{i}_shape"] = dfdat[f"{i}_shape"]
        D.Dat[f"{i}_loc"] = dfdat[f"{i}_loc"]
        
    Dcopy = sn.Datasetbeh.copy()

    # Remove if aborted
    D.filterPandas({"aborted":[False], "task_kind":["prims_on_grid"]}, "modify")
    # Filter trials that dont have enough strokes
    D.Dat = D.Dat[D.Dat["nstrokes_beh"]>=n_strok_max].reset_index(drop=True)

    # Print final
    print("Pruned dataset for _dataset_extract_prune_sequence")
    print(D.Dat["nstrokes_beh"].value_counts())
    print(D.Dat["nstrokes_task"].value_counts())
    print(D.Dat["0_shape"].value_counts())
    print(D.Dat["0_loc"].value_counts())
    print(D.Dat["1_shape"].value_counts())
    print(D.Dat["1_loc"].value_counts())

    return D


def _dataset_extract_prune_rulesw(sn, same_beh_only, 
        n_min_trials_in_each_epoch, remove_baseline_epochs, 
        first_stroke_same_only,
        plot_tasks=False):
    """
    Prep beh dataset before extracting snippets.
    Return pruned dataset (copy), to do rule swtiching analy
    matching motor beh, etc.
    """

    if first_stroke_same_only:
        assert same_beh_only==False, "this will throw out cases with same first stroke, but nto all strokes same."
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
    list_correctness = []
    for ind in range(len(D.Dat)):
        dat = D.sequence_extract_beh_and_task(ind)
        taskstroke_inds_beh_order = dat["taskstroke_inds_beh_order"]
        taskstroke_inds_correct_order = dat["taskstroke_inds_correct_order"]

        if taskstroke_inds_beh_order==taskstroke_inds_correct_order:
            # correct
            inds_correct.append(ind)
            list_correctness.append(True)
        else:
            inds_incorrect.append(ind)
            list_correctness.append(False)

    D.Dat["sequence_correct"] = list_correctness
    print("-- Correctness:")
    D.grouping_print_n_samples(["character", "epoch", "sequence_correct"])

    print("correct, incorrect:", len(inds_correct), len(inds_incorrect))
    D.subsetDataframe(inds_correct)
    print("Dataset len:", len(D.Dat))

    print("############ TAKING ONLY NO SUPERVISION TRIALS")
    LIST_NO_SUPERV = ["off|0||0", "off|1|solid|0", "off|1|rank|0"]
    # Only during no-supervision blocks
    print(D.Dat["supervision_stage_concise"].value_counts())
    D.filterPandas({"supervision_stage_concise":LIST_NO_SUPERV}, "modify")
    print("Dataset len:", len(D.Dat))

    # Only characters with same beh across rules.
    # 2) Extract takss done the same in mult epochs
    if first_stroke_same_only:
        # find lsit of tasks that have same first stroke across epochs.
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        list_tasks = D.Dat["character"].unique().tolist()
        list_epoch = D.Dat["epoch"].unique().tolist()
        info = grouping_append_and_return_inner_items(D.Dat, ["character", "epoch"])
        list_tasks_keep = []
        print("n unique first stroke identities for each task:")
        for task in list_tasks:
            list_orders = []
            for ep in list_epoch:
                if (task, ep) in info.keys():
                    idx = info[(task, ep)][0] # take first task in epoch
                    sdict = D.sequence_extract_beh_and_task(idx)
                    order = sdict["taskstroke_inds_correct_order"]
                    list_orders.append(order)
            n_orders_first_sroke = len(list(set([o[0] for o in list_orders])))
            n_epochs_with_data = len(list_orders)
            print("task, n_orders_first_sroke, n_epochs_with_data ... ")
            print(task, n_orders_first_sroke, n_epochs_with_data)
            if n_orders_first_sroke==1 and n_epochs_with_data>1:
                list_tasks_keep.append(task)
        print("These tasks keep (same first stroke): ", list_tasks_keep)
        D.filterPandas({"character":list_tasks_keep}, "modify")
        print("New len of D:", len(D.Dat))

    if same_beh_only:
        print("############ TAKING ONLY CHAR WITH SAME BEH ACROSS TRIALS")
        if True:
            # Already extracted in taskgroups (probes.py)
            print("Using this filter on taskgroup: same_beh, same_beh-P]")
            print("Starting task groups:")
            print("N chars per group:")    
            for grp, inds in D.grouping_get_inner_items("taskgroup", "character").items():
                print(grp, ":", len(inds))    
                
            print("----")
            print("N trials per group:")
            for grp, inds in D.grouping_get_inner_items("taskgroup").items():
                print(grp, ":", len(inds))
            print("---------------------")    
            D.grouping_print_n_samples(["taskgroup", "epoch", "character"])

            D.filterPandas({"taskgroup":["same_beh", "same_beh-P"]}, "modify")
        else:
            # Old method, obsolste, this is done to define taskgroup in probes.py
            from pythonlib.dataset.dataset_preprocess.probes import _generate_map_taskclass
            mapper_taskname_epoch_to_taskclass = _generate_map_taskclass(D)

            list_task = D.Dat["character"].unique().tolist()
            list_epoch = D.Dat["epoch"].unique().tolist()
            if len(list_epoch)<2:
                print("maybe failed to preprocess dataset to reextract epochs?")
                assert False
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
            
            print(dict_task_orders)
            # pull out tasks which have same sequence
            tasks_same_sequence = []
            for taskname, list_seqs in dict_task_orders.items():
                if len(list(set(list_seqs)))==1:
                    tasks_same_sequence.append(taskname)
            print("\nTHese tasks foudn to have same seuqence across epochs: ")
            print(tasks_same_sequence)
            D.filterPandas({"character":tasks_same_sequence}, "modify")
        print("Dataset len:", len(D.Dat))

    # Remove epoch = baseline
    if remove_baseline_epochs:
        # D.Dat[~(D.Dat["epoch"].isin(["base", "baseline"]))]
        indskeep = D.Dat[~(D.Dat["epoch"].isin(["base", "baseline"]))].index.tolist()
        D.subsetDataframe(indskeep)

    # Only if have at least N trials per epoch
    print("############ TAKING ONLY CHAR with min n trials in each epoch")
    # print(D.Dat["character"].value_counts())
    # print(D.Dat["epoch"].value_counts())
    D.grouping_print_n_samples(["taskgroup", "epoch", "character"])
    df = D.prune_min_ntrials_across_higher_levels("epoch", "character", n_min=n_min_trials_in_each_epoch)
    D.Dat = df

    # Final length of D.Dat
    print("Final n trials", len(D.Dat))
    print("Dataset conjunctions:")
    D.grouping_print_n_samples(["taskgroup", "epoch", "character"])

    return D


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
