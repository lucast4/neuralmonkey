""" Holds params for anava analysis for each date
"""

ONLY_ESSENTIAL_VARS = False # then just the first var, assuemd to be most important, for quick analys

##################
LIST_ANALYSES = ["rulesw", "ruleswERROR", "seqcontext", "singleprim"] # repo of possible analses, 

def exptlist_getter(self):

    LIST_EXPTS = [
        ("Pancho", 220709, "trial", "seqcontext"),
        ("Pancho", 230105, "trial", "seqcontext"),
    ]

    return LIST_EXPTS

def _params_score_sequence_ver(animal, DATE, ANALY_VER):
    """ Decide how to score each trial's sequence success, either
    comparing beh to matlab task seuqence or to parses"""
    if ANALY_VER in ["rulesw", "ruleswERROR"]:
        if animal=="Pancho" and DATE in [220913]:
            DO_SCORE_SEQUENCE_VER = "parses"
        elif animal=="Pancho" and DATE in [220812, 220814, 220815, 220816, 220827,
            220921, 220928, 220929, 221014, 221020, 
            221031, 221107, 221114, 221119, 221125]:
            # determenistic (single solution)
            DO_SCORE_SEQUENCE_VER = "matlab"
        else:
            print(animal, DATE)
            assert False
    else:
        DO_SCORE_SEQUENCE_VER = None
    return DO_SCORE_SEQUENCE_VER

def params_getter_plots(animal, DATE, which_level, ANALY_VER, anova_interaction=False):
    """
    Params for use in plotting of anova, for each var and conjunction of vars.
    Like a repo of thise.
    Used in analy_anova_plot.py
    PARAMS:
    - trials_subset_ver, str, key to pick out subset of trials, e.g., "error"
    """

    assert ANALY_VER in LIST_ANALYSES, "add this to list"

    PRE_DUR_CALC = None # None, since below uses diff ones for each event.
    POST_DUR_CALC = None
    globals_nmin = 8
    globals_lenient_allow_data_if_has_n_levels = 2

    ################## SUPERVISION LEVELS TO KEEP
    list_superv_keep = None # None means just keep all cases that are "no supervision"

    ################# OPTIONALLY KEEP ONLY SPECIFIC "FULL" SUPERVISION NAMES
    if animal=="Pancho" and DATE==220921:
        # This day has "randomseq-sequence mask"... (rare)
        # Therefore want to keep sequence mask
        # (An epoch is random sequence using sequence mask)
        list_superv_keep_full = ["mask|0.5|0||0|none",
                                  "off|1.0|0||0|none",
                                  "off|1.0|0||0|none"]
    else:
        list_superv_keep_full = None


    ################ SCORE PERFORMANCE?
    DO_SCORE_SEQUENCE_VER = _params_score_sequence_ver(animal, DATE, ANALY_VER)

    ################ BEH DATASET PREPROCESS STEPS
    # THESE ARE ONLY used for deciding which trials ot keep.
    if ANALY_VER in ["rulesw"]:
        preprocess_steps_append = ["remove_repeated_trials", "correct_sequencing_binary_score", 
            "one_to_one_beh_task_strokes"]
    elif ANALY_VER in ["ruleswERROR"]:
        # error trials
        preprocess_steps_append = ["remove_repeated_trials", "wrong_sequencing_binary_score"]
    elif ANALY_VER in ["seqcontext", "singleprim"]:
        preprocess_steps_append = ["remove_repeated_trials", "one_to_one_beh_task_strokes"]
    else:
        assert False

    # Remove aborts
    if ANALY_VER in ["ruleswERROR"]:
        # error trials
        remove_aborts = False
    else:        
        # correct trials
        remove_aborts = True

    if anova_interaction:
        # score_ver='r2smfr_running_maxtime_twoway'
        score_ver='r2_maxtime_2way_mshuff'
        get_z_score = False
    else:
        # score_ver='fracmod_smfr_minshuff'
        # score_ver='r2smfr_running_maxtime_oneway'
        score_ver='r2_maxtime_1way_mshuff'
        get_z_score=True

    if which_level=="trial" and ANALY_VER=="seqcontext":

        LIST_VAR = [
            "seqc_nstrokes_beh",
            "seqc_nstrokes_beh",
            "seqc_nstrokes_beh",
            "seqc_0_shape",
            "seqc_0_loc",
            "seqc_1_shape",
            "seqc_1_loc",
            # "seqc_2_shape",
            # "seqc_2_loc",
            ]
        LIST_VARS_CONJUNCTION = [
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc", "seqc_2_shape", "seqc_2_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            ["seqc_0_shape", "seqc_0_loc"],
            ["seqc_0_loc"],
            ["seqc_0_shape"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape"],
            # ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            # ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
        ]           

        list_events = ["03_samp", "03_samp", "05_first_raise", "06_on_strokeidx_0", "09_post", "10_reward_all"]
        list_pre_dur = [-0.6, 0.05, -0.6, -0.1, 0.05, 0.05]
        list_post_dur = [-0.05, 0.6, -0.05, 0.6, 0.6, 0.6]

    elif which_level=="trial" and ANALY_VER in ["rulesw", "ruleswERROR"]:
        # Rule switching.

        if DATE in [220928, 220929]:
            # grmamar vs. color rank. should do epochsets, but decided to try
            # this becuase epochsets would throw out like 1/2 the data (keeping only
            # epochset spanning both epochs)
            # - possibly try both meothds.
            # strategy here is to get "same beh" as those with matched first stroke.
            LIST_VAR = [
                "epoch",
            ]
            LIST_VARS_CONJUNCTION = [
                ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            ]
        elif DATE in [221014]:
            # Then this uses epochsets
            LIST_VAR = [
                "epoch",
                "seqc_0_shape",
                "seqc_0_loc",
            ]
            LIST_VARS_CONJUNCTION = [
                ["epochset"],
                ["epoch", "seqc_0_loc"],
                ["epoch", "seqc_0_shape"],
            ]
        else:
            LIST_VAR = [
                "epoch",
                "seqc_0_shape",
                "seqc_0_loc",
                "probe",
            ]
            LIST_VARS_CONJUNCTION = [
                ["taskgroup", "probe"],
                ["epoch", "seqc_0_loc"],
                ["epoch", "seqc_0_shape"],
                ["seqc_0_loc", "seqc_0_shape", "epoch"] 
            ]

        if ONLY_ESSENTIAL_VARS or (ANALY_VER in ["ruleswERROR"]): 
            # just test epoch, for error trials
            LIST_VAR = LIST_VAR[:1]
            LIST_VARS_CONJUNCTION = LIST_VARS_CONJUNCTION[:1]

        # list_events = ["00_fixcue", "00_fixcue", "01_fix_touch", "02_samp", "03_go_cue", "05_on_strokeidx_0", ""]
        # list_pre_dur = [-0.6, 0.05, -0.4, 0.05, -0.6, -0.2]
        # list_post_dur = [0,   0.38, 0.4, 0.6, -0.05, 0.45]

        if DATE in [220812, 220814, 220815, 220816, 220827, 220913, 220921, 220928, 220929]:
            # NO COLOR (blocks)
            list_events = ["03_samp", "03_samp", "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            list_pre_dur = [-0.6, 0.05, -0.6, -0.1, -0.5, 0.05, 0.05]
            list_post_dur = [-0.05, 0.6, -0.05, 0.6, 0.3, 0.6, 0.6]
        elif DATE in [221014, 221020]:
            # fixcue[colored] --> fixtouch --> image[colored] --> go...
            list_events = ["00_fixcue", "00_fixcue", "03_samp", "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            list_pre_dur = [-0.6, 0.05, 0.05, -0.6, -0.1, -0.5, 0.05, 0.05]
            list_post_dur = [-0.05, 0.6, 0.6, -0.05, 0.6, 0.3, 0.6, 0.6]
        elif DATE in [221031, 221107, 221114, 221119, 221125]:
            # fuxcue --> fixtouch --> rulecue2[e.g, fixcue_color_change] --> samp + cue_color_off

            # OLD, post-rule_cue was not immediately after rule cue, missed transient stuff.
            # list_events = ["02_rulecue2",   "03_samp",  "03_samp", "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            # list_pre_dur = [-0.6,           -0.6,       0.05, -0.6, -0.1, -0.5, 0.05, 0.05]
            # list_post_dur = [-0.04,         -0.04,      0.6, -0.05, 0.6, 0.3, 0.6, 0.6]
            list_events = ["02_rulecue2",   "02_rulecue2",  "03_samp", "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            list_pre_dur = [-0.6,           0.04,       0.05, -0.6, -0.1, -0.5, 0.05, 0.05]
            list_post_dur = [-0.04,         0.6,      0.6, -0.05, 0.6, 0.3, 0.6, 0.6]
        else:
            print(DATE)
            assert False

    elif which_level=="trial" and ANALY_VER=="singleprim":
        # single prim (the first stroke)
        LIST_VAR = [
            "gridsize",
            "seqc_0_shape",
            "seqc_0_loc",
        ]
        LIST_VARS_CONJUNCTION = [
            ["seqc_0_loc", "seqc_0_shape"],
            ["seqc_0_loc", "gridsize"],
            ["gridsize", "seqc_0_shape"],
        ]

        list_events = ["03_samp", "03_samp", "05_first_raise", "06_on_strokeidx_0", "09_post", "10_reward_all"]
        list_pre_dur = [-0.6, 0.05, -0.6, -0.1, 0.05, 0.05]
        list_post_dur = [-0.05, 0.6, -0.05, 0.6, 0.6, 0.6]

    # ###################################
    # LIST_VAR = [
    #     # "seqc_nstrokes_beh",
    #     # "seqc_nstrokes_beh",
    #     # "seqc_nstrokes_beh",
    #     # "seqc_nstrokes_beh",
    #     # "seqc_0_shape",
    #     # "seqc_1_shape",
    #     # "seqc_2_shape",
    #     # "seqc_3_shape",
    #     # "seqc_0_loc",
    #     # "seqc_1_loc",
    #     # "seqc_2_loc",
    #     # "seqc_3_loc",
    #     # 'CTXT_shape_next',
    #     # 'CTXT_loc_next',
    #     # 'CTXT_shape_prev',
    #     # 'CTXT_loc_prev',
    #     # "chunk_within_rank",
    #     # "chunk_within_rank",
    #     # "chunk_within_rank",
    #     # "chunk_within_rank",
    #     # "chunk_within_rank",
    #     # "supervision_stage_concise",
    #     # "stroke_index",
    #     # "stroke_index",
    #     # "shape_oriented",
    #     # "gridloc",
    #     ]
    # LIST_VARS_CONJUNCTION = [
    #     # ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc", "seqc_2_shape", "seqc_2_loc"],
    #     # ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
    #     # ["seqc_0_shape", "seqc_0_loc"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ["epoch"],
    #     # ['CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next', 'gridloc', 'shape_oriented'],
    #     # ['stroke_index', 'CTXT_loc_prev', 'CTXT_shape_prev', 'gridloc', 'shape_oriented'],
    #     # ['shape_oriented', 'gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next'],
    #     # ['shape_oriented', 'gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_shape_next'],
    #     # ['shape_oriented', 'gridloc', 'CTXT_loc_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    #     # ['shape_oriented', 'gridloc', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    #     # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev'],
    #     # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev', 'supervision_stage_concise'],
    #     # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'supervision_stage_concise'],
    #     # ['gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev', 'supervision_stage_concise'],
    #     # ['gridloc', 'chunk_rank', 'CTXT_loc_prev', 'supervision_stage_concise'],
    #     # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev'],
    #     # ['gridloc', 'shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev'],
    #     # ['gridloc', 'shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next'],
    #     # ['gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    #     # ['shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    # ]

    assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)
    assert len(LIST_VAR)>0

    params = {
        "LIST_VAR":LIST_VAR,
        "LIST_VARS_CONJUNCTION":LIST_VARS_CONJUNCTION,
        "PRE_DUR_CALC":PRE_DUR_CALC,
        "POST_DUR_CALC":POST_DUR_CALC,
        "globals_nmin":globals_nmin,
        "globals_lenient_allow_data_if_has_n_levels":globals_lenient_allow_data_if_has_n_levels,
        "score_ver":score_ver,
        "list_events":list_events,
        "list_pre_dur":list_pre_dur,
        "list_post_dur":list_post_dur,
        "ANALY_VER":ANALY_VER,
        "which_level":which_level,
        "DATE":DATE,
        "animal":animal,
        "get_z_score":get_z_score, 
        "list_superv_keep":list_superv_keep,
        "preprocess_steps_append":preprocess_steps_append,
        "remove_aborts":remove_aborts,
        "DO_SCORE_SEQUENCE_VER":DO_SCORE_SEQUENCE_VER,
        "list_superv_keep_full":list_superv_keep_full
    }

    return  params

def params_getter_extraction(animal, DATE, which_level, ANALY_VER):
    """ PArams for use with extraction of snippets, e.g., 
    in analy_anova_extraction.py
    """

    assert isinstance(DATE, int)
    assert ANALY_VER in LIST_ANALYSES, "add this to list"

    PRE_DUR = -0.65
    POST_DUR = 0.65
    # PRE_DUR_FIXCUE = -0.5
    PRE_DUR_FIXCUE = None # leavea as None, since having different predur for fix leads
    # to error when try to concat into frmat (e.,gm in SP.save_v2)

    ################ SCORE PERFORMANCE?
    DO_SCORE_SEQUENCE_VER = _params_score_sequence_ver(animal, DATE, ANALY_VER)

    ############### Reassign name to "taskgroup", siimplifying things, especialyl good
    # for grid sequence tasks with different probes, want to merge them for larger N.
    if ANALY_VER in ["ruleswERROR", "rulesw"]:
        taskgroup_reassign_simple_neural = True
    else:
        taskgroup_reassign_simple_neural = False

    ################ SEQUENCE CONTEXT?
    DO_EXTRACT_CONTEXT = True    

    # ################ BEH DATASET PREPROCESS STEPS
    # if ANALY_VER in ["rulesw"]:
    #     preprocess_steps_append = ["sanity_gridloc_identical", "correct_sequencing_binary_score", "one_to_one_beh_task_strokes"]
    # elif ANALY_VER in ["seqcontext", "singleprim"]:
    #     preprocess_steps_append = ["sanity_gridloc_identical", "one_to_one_beh_task_strokes"]
    # else:
    #     assert False

    ############### RENAME EPOCHS
    if animal=="Pancho" and DATE in  [220928, 220929, 221014]:
        list_epoch_merge = [
            (["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
        ]
        DO_CHARSEQ_VER = "task_matlab"
        EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "char_seq"
        EXTRACT_EPOCHSETS_n_max_epochs = 2
    else:
        list_epoch_merge = []
        DO_CHARSEQ_VER = None
        EXTRACT_EPOCHSETS = False
        EXTRACT_EPOCHSETS_trial_label = None
        EXTRACT_EPOCHSETS_n_max_epochs = None

    ################ FEATURES TO EXTRACT
    if ANALY_VER in ["ruleswERROR", "rulesw", "seqcontext", "singleprim"]:
        list_features_modulation_append = ["probe", "taskgroup", "character", "trialcode", "epoch", 
                                            "task_kind", "supervision_stage_concise"]
    else:
        assert False

    if DO_EXTRACT_CONTEXT:
        # These are features that are gotten from extracting context
        list_features_modulation_append = list_features_modulation_append + ["seqc_nstrokes_beh", "seqc_nstrokes_task",
                                           "seqc_0_shape", "seqc_0_loc", "seqc_1_shape", 
                                           "seqc_1_loc", "seqc_2_shape", "seqc_2_loc", 
                                           "seqc_3_shape", "seqc_3_loc", 
                                           "gridsize"]

    if DO_CHARSEQ_VER:
        list_features_modulation_append.append("char_seq")                                        
    if EXTRACT_EPOCHSETS:
        list_features_modulation_append.append("epochset")                                        

    # make sure all vars that you will use for plots are included in extraction
    params_plots = params_getter_plots(animal, DATE, which_level, ANALY_VER)
    for var in params_plots["LIST_VAR"]:
        if var not in list_features_modulation_append:
            print(var)
            print(list_features_modulation_append)
            assert False, "need to extract this feature"
    for lv in params_plots["LIST_VARS_CONJUNCTION"]:
        for var in lv:
            if var not in list_features_modulation_append:
                print(lv)
                print(var)
                print(list_features_modulation_append)
                assert False, "need to extract this feature"

    params = {
        "PRE_DUR":PRE_DUR,
        "POST_DUR":POST_DUR,
        "PRE_DUR_FIXCUE":PRE_DUR_FIXCUE,
        "DATE":DATE,
        "animal":animal,
        "which_level":which_level,
        "DO_SCORE_SEQUENCE_VER":DO_SCORE_SEQUENCE_VER,
        "DO_EXTRACT_CONTEXT":DO_EXTRACT_CONTEXT,
        # "preprocess_steps_append":preprocess_steps_append,
        "list_features_modulation_append":list_features_modulation_append,
        "ANALY_VER":ANALY_VER,
        "taskgroup_reassign_simple_neural":taskgroup_reassign_simple_neural,
        "list_epoch_merge":list_epoch_merge,
        "DO_CHARSEQ_VER":DO_CHARSEQ_VER,
        "EXTRACT_EPOCHSETS":EXTRACT_EPOCHSETS,
        "EXTRACT_EPOCHSETS_trial_label":EXTRACT_EPOCHSETS_trial_label,
        "EXTRACT_EPOCHSETS_n_max_epochs":EXTRACT_EPOCHSETS_n_max_epochs
        }

    return params