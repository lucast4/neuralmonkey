""" Holds params for anava analysis for each date
"""


def exptlist_getter(self):

    LIST_EXPTS = [
        ("Pancho", 220709, "trial", "seqcontext"),
        ("Pancho", 230105, "trial", "seqcontext"),
    ]

    return LIST_EXPTS

def params_getter_plots(animal, DATE, which_level, ANALY_VER, anova_interaction=False):
    """
    Params for use in plotting of anova, for each var and conjunction of vars.
    Like a repo of thise.
    Used in analy_anova_plot.py
    """

    PRE_DUR_CALC = None # None, since below uses diff ones for each event.
    POST_DUR_CALC = None
    globals_nmin = 8 
    globals_lenient_allow_data_if_has_n_levels = 2
    if anova_interaction:
        score_ver='r2smfr_running_maxtime_twoway'
        get_z_score = False
    else:
        # score_ver='fracmod_smfr_minshuff'
        score_ver='r2smfr_running_maxtime_oneway'
        get_z_score=True


    if which_level=="trial" and ANALY_VER=="seqcontext":
        # prims in grid
        # if animal=="Pancho" and DATE in [220709]:
        LIST_VAR = [
            "seqc_nstrokes_beh",
            "seqc_nstrokes_beh",
            "seqc_nstrokes_beh",
            "seqc_0_shape",
            "seqc_0_loc",
            "seqc_1_shape",
            "seqc_1_loc",
            "seqc_2_shape",
            "seqc_2_loc",
            # "seqc_1_shape",
            # "seqc_2_shape",
            # "seqc_3_shape",
            # "seqc_1_loc",
            # "seqc_2_loc",
            # "seqc_3_loc",
            ]
        LIST_VARS_CONJUNCTION = [
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc", "seqc_2_shape", "seqc_2_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            ["seqc_0_shape", "seqc_0_loc"],
            ["epoch"],
            ["epoch"],
            ["seqc_0_shape", "seqc_0_loc"],
            ["seqc_0_shape", "seqc_0_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            # ["epoch"],
            # ["epoch"],
            # ["epoch"],
            # ["epoch"],
            # ["epoch"],
            # ["epoch"],
        ]           
        list_events = ["01_fix_touch", "02_samp", "04_first_raise"]
        list_pre_dur = [-0.4, 0.05, -0.1]
        list_post_dur = [0.4, 0.6, 0.5]

    elif which_level=="trial" and ANALY_VER=="rulesw":
        # Rule switching.
        LIST_VAR = [
            "epoch",
            "seqc_0_shape",
            "seqc_0_loc",
        ]
        LIST_VARS_CONJUNCTION = [
            ["taskgroup"],
            ["epoch", "seqc_0_loc"],
            ["epoch", "seqc_0_shape"],
        ]

        list_events = ["00_fixcue", "00_fixcue", "01_fix_touch", "02_samp", "03_go_cue", "05_on_strokeidx_0"]
        list_pre_dur = [-0.6, 0.05, -0.4, 0.05, -0.6, -0.2]
        list_post_dur = [0,   0.38, 0.4, 0.6, -0.05, 0.45]

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
        "get_z_score":get_z_score
    }

    return  params


def params_getter_extraction(animal, DATE, which_level, ANALY_VER):
    """ PArams for use with extraction of snippets, e.g., 
    in analy_anova_extraction.py
    """

    assert isinstance(DATE, int)
    assert ANALY_VER in ["rulesw", "seqcontext"] # repo of possible analses

    ################ SCORE PERFORMANCE?
    if ANALY_VER in ["rulesw"]:
        # if animal=="Pancho" and DATE in [221020]:
        #     DO_SCORE_SEQUENCE_VER = "matlab"
        # else:
        #     DO_SCORE_SEQUENCE_VER = "parses"
        if animal=="Pancho" and DATE in []:
            DO_SCORE_SEQUENCE_VER = "parses"
        else:
            DO_SCORE_SEQUENCE_VER = "matlab"
    else:
        DO_SCORE_SEQUENCE_VER = None

    ################ SEQUENCE CONTEXT?
    if ANALY_VER in []:
        DO_EXTRACT_CONTEXT = False
    elif ANALY_VER in ["seqcontext", "rulesw"]:
        DO_EXTRACT_CONTEXT = True
    else:
        assert False

    ################ BEH DATASET PREPROCESS STEPS
    if ANALY_VER in ["rulesw"]:
        preprocess_steps_append = ["sanity_gridloc_identical", "correct_sequencing_binary_score", "one_to_one_beh_task_strokes"]
    elif ANALY_VER in ["seqcontext"]:
        preprocess_steps_append = ["sanity_gridloc_identical", "one_to_one_beh_task_strokes"]
    else:
        assert False

    ################ FEATURES TO EXTRACT
    if ANALY_VER in ["rulesw", "seqcontext"]:
        list_features_modulation_append = ["probe", "taskgroup", "character", "trialcode", "epoch", 
                                            "task_kind", "supervision_stage_concise"]
    else:
        assert False
    if DO_EXTRACT_CONTEXT:
        list_features_modulation_append = list_features_modulation_append + ["seqc_nstrokes_beh", "seqc_nstrokes_task",
                                           "seqc_0_shape", "seqc_0_loc", "seqc_1_shape", 
                                           "seqc_1_loc", "seqc_2_shape", "seqc_2_loc", 
                                           "seqc_3_shape", "seqc_3_loc"]

    params = {
        "DATE":DATE,
        "animal":animal,
        "which_level":which_level,
        "DO_SCORE_SEQUENCE_VER":DO_SCORE_SEQUENCE_VER,
        "DO_EXTRACT_CONTEXT":DO_EXTRACT_CONTEXT,
        "preprocess_steps_append":preprocess_steps_append,
        "list_features_modulation_append":list_features_modulation_append,
        "ANALY_VER":ANALY_VER
    }

    return params