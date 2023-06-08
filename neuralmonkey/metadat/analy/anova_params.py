""" Holds params for anava analysis for each date
"""

ONLY_ESSENTIAL_VARS = False # then just the first var, assuemd to be most important, for quick analys

##################
LIST_ANALYSES = ["rulesw", "ruleswALLDATA", "ruleswERROR", "seqcontext", "singleprim"] # repo of possible analses, 

# def exptlist_getter(self):

#     LIST_EXPTS = [
#         ("Pancho", 220709, "trial", "seqcontext"),
#         ("Pancho", 230105, "trial", "seqcontext"),
#     ]

#     return LIST_EXPTS

def _params_score_sequence_ver(animal, DATE, ANALY_VER):
    """ Decide how to score each trial's sequence success, either
    comparing beh to matlab task seuqence or to parses"""
    if ANALY_VER in ["rulesw", "ruleswERROR", "ruleswALLDATA"]:
        if animal=="Pancho" and DATE in [220913]:
            DO_SCORE_SEQUENCE_VER = "parses"
        elif animal=="Pancho" and DATE in [220812, 220814, 220815, 220816, 220827,
            220921, 220928, 220929, 220930, 221001, 221002, 221014, 221020, 221021, 221023, 221024,
            221031, 221102, 221107, 221112, 221113, 221114, 221118, 221119, 221121, 221125]:
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
    globals_nmin = 5
    globals_lenient_allow_data_if_has_n_levels = 2

    ################## SUPERVISION LEVELS TO KEEP
    ################# OPTIONALLY KEEP ONLY SPECIFIC "FULL" SUPERVISION NAMES
    if animal=="Pancho" and DATE==220921:
        # This day has "randomseq-sequence mask"... (rare)
        # Therefore want to keep sequence mask
        # (An epoch is random sequence using sequence mask)
        list_superv_keep_full = ["mask|0.5|0||0|none",
                                  "off|1.0|0||0|none",
                                  "off|1.0|0||0|none"]
        list_superv_keep = "all" # None means just keep all cases that are "no supervision"
    else:
        # Prune so is just "no supervision" (i.e., list_superv_keep is None)
        list_superv_keep_full = None
        list_superv_keep = None # None means just keep all cases that are "no supervision"


    ################ SCORE PERFORMANCE?
    DO_SCORE_SEQUENCE_VER = _params_score_sequence_ver(animal, DATE, ANALY_VER)

    ######### interaction?
    if anova_interaction:
        # score_ver='r2smfr_running_maxtime_twoway'
        score_ver='r2_maxtime_2way_mshuff'
        get_z_score = False
    else:
        # score_ver='fracmod_smfr_minshuff'
        # score_ver='r2smfr_running_maxtime_oneway'
        score_ver='r2_maxtime_1way_mshuff'
        get_z_score=True

    ############################# VARIABLES
    if which_level=="stroke" and ANALY_VER in ["rulesw"]:
        # Rule switching.

        LIST_VAR = [
            "gridloc", # 1) state space expansion
            "shape_oriented",
            "gridloc",
            "epoch", # 2) epoch encoidng
            # "shape_oriented",
            # "gridloc",
            "shape_oriented", # 3) prim encoding.
            "gridloc",
            "stroke_index",
            # "character"
        ]
        LIST_VARS_CONJUNCTION = [
            ["shape_oriented", "epochkind"], # this allows combining multiple "direction" epochs into a single epochkind (state space expansion).
            ["gridloc", "epoch"], # removed stroke_index, since this is not dissociated from both location and shape (for a given rule).
            ["shape_oriented", "epoch"], # removed stroke_index, since this is not dissociated from both location and shape (for a given rule).
            ["epochset", "stroke_index"],
            # ["gridloc", "epoch", "stroke_index"], # useful for comparison across areas
            # ["shape_oriented", "epoch", "stroke_index"], 
            ["gridloc", "epoch", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"], # useful for comparison across areas
            ["shape_oriented", "epoch", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"], 
            ["gridloc", "shape_oriented", "epoch"], 
            # ["epoch", "epochset", "stroke_index"],
        ]

        # # Consistency of loc and shape activity in sequence.
        # LIST_VAR = [
        #     "gridloc",
        #     "shape_oriented",
        # ]
        # LIST_VARS_CONJUNCTION = [
        #     ["shape_oriented", "stroke_index"],
        #     ["gridloc", "stroke_index"], # removed stroke_index, since this is not dissociated from both location and shape (for a given rule).
        # ]

        WINDOWS_DEFAULT = [
            ('00_stroke', -0.25, 0.35)]
        list_events = [x[0] for x in WINDOWS_DEFAULT]
        list_pre_dur = [x[1] for x in WINDOWS_DEFAULT]
        list_post_dur = [x[2] for x in WINDOWS_DEFAULT]

    elif which_level=="stroke" and ANALY_VER=="seqcontext":

        LIST_VAR = [
            # "gridloc",
            # "shape_oriented",
            "gridloc",
            "shape_oriented",
            "stroke_index",
        ]
        LIST_VARS_CONJUNCTION = [
            # ["shape_oriented", "stroke_index"],
            # ["gridloc", "stroke_index"], 
            ["shape_oriented", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"],
            ["gridloc", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"], 
            ["gridloc", "shape_oriented", "CTXT_loc_prev", "CTXT_shape_prev"], 
        ]

        WINDOWS_DEFAULT = [
            ('00_stroke', -0.25, 0.35)]
        list_events = [x[0] for x in WINDOWS_DEFAULT]
        list_pre_dur = [x[1] for x in WINDOWS_DEFAULT]
        list_post_dur = [x[2] for x in WINDOWS_DEFAULT]

    elif which_level=="trial" and ANALY_VER=="seqcontext":

        LIST_VAR = [
            "seqc_2_shape",
            "seqc_2_loc",
            "seqc_2_loc_shape",

            "seqc_1_shape",
            "seqc_1_loc",
            "seqc_1_loc_shape",

            "seqc_0_shape",
            "seqc_0_loc",

            "seqc_nstrokes_beh",
            "seqc_nstrokes_beh",
            "seqc_nstrokes_beh",
            # "seqc_2_shape",
            # "seqc_2_loc",
            ]
        LIST_VARS_CONJUNCTION = [
            ["seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc"], 
            ["seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_shape"], 
            ["seqc_0_loc_shape", "seqc_1_loc_shape"], 

            ["seqc_0_shape", "seqc_0_loc", "seqc_1_loc"], 
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape"],
            ["seqc_0_shape", "seqc_0_loc"],

            ["seqc_0_loc"],
            ["seqc_0_shape"],

            ["seqc_0_shape", "seqc_0_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc", "seqc_2_shape", "seqc_2_loc"],
            # ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
            # ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc"],
        ]           

        # list_events = ["03_samp", "03_samp", "05_first_raise", "06_on_strokeidx_0", "09_post", "10_reward_all"]
        # list_pre_dur = [-0.6, 0.05, -0.6, -0.1, 0.05, 0.05]
        # list_post_dur = [-0.05, 0.6, -0.05, 0.6, 0.6, 0.6]

        # list_events = ["03_samp",   "03_samp", "04_go_cue",  "05_first_raise",   "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
        # list_pre_dur = [-0.6,       0.05,      -0.6,        -0.6,               -0.25, -0.5, 0.05, 0.05]
        # list_post_dur = [-0.04,     0.6,       -0.04,       -0.05,              0.35, 0.3, 0.6, 0.6]
    
        WINDOWS_DEFAULT = [
            ('03_samp', -0.6, -0.04),
            ('03_samp', 0.04, 0.24),
            ('03_samp', 0.26, 0.6),
            ('04_go_cue', -0.6, -0.04),
            ('05_first_raise', -0.6, -0.05),
            # ('06_on_strokeidx_0', -0.25, 0.35),
            ('06_on_strokeidx_0', -0.45, -0.05),
            ('06_on_strokeidx_0', -0.05, 0.35),
            ('08_doneb', -0.5, 0.3),
            ('09_post', 0.05, 0.6),
            ('10_reward_all', 0.05, 0.6)]
        list_events = [x[0] for x in WINDOWS_DEFAULT]
        list_pre_dur = [x[1] for x in WINDOWS_DEFAULT]
        list_post_dur = [x[2] for x in WINDOWS_DEFAULT]

    elif which_level=="trial" and ANALY_VER in ["rulesw", "ruleswERROR", "ruleswALLDATA"]:
        # Rule switching.

        if ANALY_VER == "ruleswALLDATA":
            # Everything else, especially trial by trial,
            LIST_VAR = [
                "success_binary_quick",
                "success_binary_quick",
                # "success_binary_quick",
                "success_binary_quick",
                # "success_binary_quick",
                # "success_binary_quick",
                "seqc_0_loc_shape"
            ]
            LIST_VARS_CONJUNCTION = [
                ["epoch", "character"],
                ["epoch", "epochset"],
                # ["epoch", "epochset"],
                ["epoch", "seqc_0_loc_shape"], 
                # ["epoch", "epochset", "seqc_0_loc_shape"], # not enough data...
                # ["epoch", "epochset", "seqc_0_loc_shape"]
                ["epoch", "character"],
            ]
        else:
            ### NOTE: commented out is left here just for notes re: what's 
            # special about each day.
            # if DATE in [220921]:
            #     # grmamar vs. sequence mask rank. This must use epoch_superv, since
            #     # epoch does not indiciate whether is using superv.
            #     # LIST_VAR = [
            #     #     "epoch_superv",
            #     #     "epoch_superv",
            #     # ]
            #     # LIST_VARS_CONJUNCTION = [
            #     #     ["epochset"],
            #     #     ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            #     # ]

            #     LIST_VAR = [
            #         "epoch",
            #         "epoch",
            #         "character",
            #         "seqc_0_loc_shape",
            #         "seqc_0_loc",
            #         "seqc_1_loc_shape",
            #     ]
            #     LIST_VARS_CONJUNCTION = [
            #         ["epochset"],
            #         ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset", "seqc_0_loc_shape"]
            #     ]

            # elif DATE in [221102]:
            #     # combines blocks and trial cues within blocks.
            #     # (e.g., rapid switching blcoks, and rule vs. rand within block, with siwtchingin
            #     # bnetween 2 rules between blocks)
            #     # Thus want to get both matched tasks (trial) and across blcoks.
            #     # LIST_VAR = [
            #     #     "epoch",
            #     #     "epoch",
            #     #     "epoch",
            #     # ]
            #     # LIST_VARS_CONJUNCTION = [
            #     #     ["epochset"],
            #     #     ["taskgroup"],
            #     #     ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            #     # ]
            #     LIST_VAR = [
            #         "epoch",
            #         "epoch",
            #         "character",
            #         "seqc_0_loc_shape",
            #         "seqc_0_loc",
            #         "seqc_1_loc_shape",
            #     ]
            #     LIST_VARS_CONJUNCTION = [
            #         ["epochset"],
            #         ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset", "seqc_0_loc_shape"]
            #     ]

            # elif DATE in [220928, 220929, 221001, 221014, 221023, 221024, 221113, 221021, 221118]:
            #     # grmamar vs. color rank (where color rank mixes random + grammar ssecretly). should do epochsets, but decided to try
            #     # this becuase epochsets would throw out like 1/2 the data (keeping only
            #     # epochset spanning both epochs)
            #     # - possibly try both meothds.
            #     # strategy here is to get "same beh" as those with matched first stroke.
                
            #     # LIST_VAR = [
            #     #     "epoch",
            #     #     "epoch",
            #     # ]
            #     # LIST_VARS_CONJUNCTION = [
            #     #     ["epochset"],
            #     #     ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            #     # ]
                        
            #     LIST_VAR = [
            #         "epoch",
            #         "epoch",
            #         "character",
            #         "seqc_0_loc_shape",
            #         "seqc_0_loc",
            #         "seqc_1_loc_shape",
            #     ]
            #     LIST_VARS_CONJUNCTION = [
            #         ["epochset"],
            #         ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset"],
            #         ["epoch", "epochset", "seqc_0_loc_shape"]
            #     ]

            # Everything else, especially trial by trial,
            LIST_VAR = [
                "epoch",
                "character",
                "seqc_0_loc",
                "seqc_0_shape",
                "seqc_1_loc_shape",
                "seqc_1_loc_shape",
                "epoch",
                "epoch", # abstract rule?
            ]
            LIST_VARS_CONJUNCTION = [
                ["epochset"],
                ["epoch", "epochset"],
                ["epoch", "seqc_0_shape"],
                ["epoch", "seqc_0_loc"],
                ["epoch"],
                ["epoch", "seqc_0_loc_shape"],
                ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
                ["epochset", "character"],
            ]

        #### EVENTS
        if (DATE in [220812, 220814, 220815, 220816, 220827, 220913, 220921, 220928, 220929, 220930]) or (DATE in [221001]):
            # NO COLOR (blocks)
            # - OR - 
            # grammar vs. color_rank (no color cue on fixation, since the strokes are colored).
            # list_events = ["03_samp",   "03_samp", "04_go_cue",  "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            # list_pre_dur = [-0.6,       0.05,      -0.6,        -0.6, -0.25, -0.5, 0.05, 0.05]
            # list_post_dur = [-0.04,     0.6,       -0.04,       -0.05, 0.35, 0.3, 0.6, 0.6]

            WINDOWS_DEFAULT = [
                ('03_samp', -0.6, -0.04),
                ('03_samp', 0.04, 0.24),
                ('03_samp', 0.26, 0.6),
                ('04_go_cue', -0.6, -0.04),
                ('05_first_raise', -0.6, -0.05),
                ('06_on_strokeidx_0', -0.25, 0.35),
                ('08_doneb', -0.5, 0.3),
                ('09_post', 0.05, 0.6),
                ('10_reward_all', 0.05, 0.6)]

        elif DATE in [221002, 221014, 221020, 221021, 221023, 221024]:
            # fixcue[colored] --> fixtouch --> image[colored] --> go...
            # list_events = ["00_fixcue", "00_fixcue", "03_samp",       "03_samp", "04_go_cue",   "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            # list_pre_dur = [-0.6, 0.05,              -0.6,       0.05,           -0.6,    -0.6, -0.1, -0.5, 0.05, 0.05]
            # list_post_dur = [-0.05, 0.6,             -0.04,       0.6,           -0.04,    -0.05, 0.6, 0.3, 0.6, 0.6]

            # Updated 5/26/23: (1) including fast and slow visual. (2) using list of tuples format.
            WINDOWS_DEFAULT = [
                ('00_fixcue', -0.6, -0.05),
                ('00_fixcue', 0.04, 0.24),
                ('00_fixcue', 0.26, 0.6),
                ('03_samp', -0.6, -0.04),
                ('03_samp', 0.04, 0.24),
                ('03_samp', 0.26, 0.6),
                ('04_go_cue', -0.6, -0.04),
                ('05_first_raise', -0.6, -0.05),
                ('06_on_strokeidx_0', -0.25, 0.35),
                ('08_doneb', -0.5, 0.3),
                ('09_post', 0.05, 0.6),
                ('10_reward_all', 0.05, 0.6)]
        elif DATE in [221031, 221102, 221107, 221112, 221113, 221114, 221118, 221119, 221121, 221125]:
            # fuxcue[nocolor] --> fixtouch --> rulecue2[e.g, fixcue_color_change] --> samp + cue_color_off

            # OLD, post-rule_cue was not immediately after rule cue, missed transient stuff.
            # list_events = ["02_rulecue2",   "03_samp",  "03_samp", "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            # list_pre_dur = [-0.6,           -0.6,       0.05, -0.6, -0.1, -0.5, 0.05, 0.05]
            # list_post_dur = [-0.04,         -0.04,      0.6, -0.05, 0.6, 0.3, 0.6, 0.6]

            # if False:
            #     list_events = ["02_rulecue2",   "02_rulecue2",  "03_samp", "05_first_raise", "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            #     list_pre_dur = [-0.6,           0.04,       0.05, -0.6, -0.1, -0.5, 0.05, 0.05]
            #     list_post_dur = [-0.04,         0.6,      0.6, -0.05, 0.6, 0.3, 0.6, 0.6]
            # else:
            #     # 5/21/23 - adding to get slower response.
            #     list_events = ["02_rulecue2",   "02_rulecue2",  "03_samp", "03_samp",   "04_go_cue",    "05_first_raise",   "06_on_strokeidx_0",    "08_doneb", "09_post", "10_reward_all"]
            #     list_pre_dur = [-0.6,           0.04,           -0.6,       0.04,       -0.6,           -0.6,               -0.25,                  -0.5, 0.05, 0.05]
            #     list_post_dur = [-0.04,         0.6,            -0.04,      0.6,        -0.05,          -0.05,              0.35,                   0.3, 0.6, 0.6]
            
            WINDOWS_DEFAULT = [
                ('02_rulecue2', -0.6, -0.05),
                ('02_rulecue2', 0.04, 0.24),
                ('02_rulecue2', 0.26, 0.6),
                ('03_samp', -0.6, -0.04),
                ('03_samp', 0.04, 0.24),
                ('03_samp', 0.26, 0.6),
                ('04_go_cue', -0.6, -0.04),
                ('05_first_raise', -0.6, -0.05),
                ('06_on_strokeidx_0', -0.25, 0.35),
                ('08_doneb', -0.5, 0.3),
                ('09_post', 0.05, 0.6),
                ('10_reward_all', 0.05, 0.6)]

        else:
            print(DATE)
            assert False
        list_events = [x[0] for x in WINDOWS_DEFAULT]
        list_pre_dur = [x[1] for x in WINDOWS_DEFAULT]
        list_post_dur = [x[2] for x in WINDOWS_DEFAULT]


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

        if False:
            list_events = ["03_samp",   "03_samp", "04_go_cue",  "05_first_raise",   "06_on_strokeidx_0", "08_doneb", "09_post", "10_reward_all"]
            list_pre_dur = [-0.6,       0.05,      -0.6,        -0.6,               -0.25, -0.5, 0.05, 0.05]
            list_post_dur = [-0.04,     0.6,       -0.04,       -0.05,              0.35, 0.3, 0.6, 0.6]
        else:
            WINDOWS_DEFAULT = [
                ('03_samp', -0.6, -0.04),
                ('03_samp', 0.04, 0.24),
                ('03_samp', 0.26, 0.6),
                ('04_go_cue', -0.6, -0.04),
                ('05_first_raise', -0.6, -0.05),
                ('06_on_strokeidx_0', -0.45, -0.05),
                ('06_on_strokeidx_0', -0.05, 0.35),
                ('08_doneb', -0.5, 0.3),
                ('09_post', 0.05, 0.6),
                ('10_reward_all', 0.05, 0.6)]
            list_events = [x[0] for x in WINDOWS_DEFAULT]
            list_pre_dur = [x[1] for x in WINDOWS_DEFAULT]
            list_post_dur = [x[2] for x in WINDOWS_DEFAULT]

    ####################### CLEAN UP VARS
    print("Got these LIST_VAR and LIST_VARS_CONJUNCTION:")
    print(LIST_VAR)
    print(LIST_VARS_CONJUNCTION)
    assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)
    assert len(LIST_VAR)>0

    if ONLY_ESSENTIAL_VARS or (ANALY_VER in ["ruleswERROR"]): 
        # just test epoch, for error trials
        LIST_VAR = LIST_VAR[:2]
        LIST_VARS_CONJUNCTION = LIST_VARS_CONJUNCTION[:2]

    ################ BEH DATASET PREPROCESS STEPS
    # THESE ARE ONLY used for deciding which trials ot keep.
    if ANALY_VER in ["rulesw"]:
        preprocess_steps_append = ["remove_repeated_trials", "correct_sequencing_binary_score", 
            "one_to_one_beh_task_strokes"]
    elif ANALY_VER in ["ruleswALLDATA"]:
        # keep all trials
        preprocess_steps_append = ["remove_repeated_trials"]
    elif ANALY_VER in ["ruleswERROR"]:
        # error trials
        preprocess_steps_append = ["remove_repeated_trials", "wrong_sequencing_binary_score"]
    elif ANALY_VER in ["seqcontext", "singleprim"]:
        preprocess_steps_append = ["remove_repeated_trials", "one_to_one_beh_task_strokes"]
    else:
        assert False
    preprocess_steps_append.append("beh_strokes_at_least_one")

    # Remove aborts
    if ANALY_VER in ["ruleswERROR", "ruleswALLDATA"]:
        # error trials
        remove_aborts = False
    else:        
        # correct trials
        remove_aborts = True


    ## If you have success as a variable then you cannot prune to only keep success...
    def _vars_including_success_binary(var, vars_others):
        """ Return True if any var or other var cares about performance succes...
        """
        if "success_binary" in var:
            return True
        if any(["success_binary" in v for v in vars_others]):
            return True
        return False


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

    assert len(list_events) == len(list_pre_dur)
    assert len(list_events) == len(list_post_dur)

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
    if ANALY_VER in ["ruleswERROR", "rulesw", "ruleswALLDATA"]:
        taskgroup_reassign_simple_neural = True
        DO_EXTRACT_EPOCHKIND = True
    else:
        taskgroup_reassign_simple_neural = False
        DO_EXTRACT_EPOCHKIND = False

    ################ SEQUENCE CONTEXT?
    DO_EXTRACT_CONTEXT = True    

    # ################ BEH DATASET PREPROCESS STEPS
    # if ANALY_VER in ["rulesw"]:
    #     preprocess_steps_append = ["sanity_gridloc_identical", "correct_sequencing_binary_score", "one_to_one_beh_task_strokes"]
    # elif ANALY_VER in ["seqcontext", "singleprim"]:
    #     preprocess_steps_append = ["sanity_gridloc_identical", "one_to_one_beh_task_strokes"]
    # else:
    #     assert False

    ############### RENAME EPOCHS (to help merge)
    if animal=="Pancho" and DATE in  [220928, 220929, 220930, 221002, 221014]:
        # Color-supervision -- ie single epochs which combine
        # random sequence + structured sequence. merge those since
        # the subject doesnt know.
        list_epoch_merge = [
            (["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
        ]
        epoch_merge_key = "epoch"
    elif animal=="Pancho" and DATE in  [221102]:
        # Color-supervision, just differnet set of rules/epochs
        list_epoch_merge = [
            (["rndstr", "llV1|1", "L|1"], "rank|1")
        ]
        epoch_merge_key = "epoch"
    elif animal =="Pancho" and DATE in [220921]:
        # Sequence mask supervision, i..e, an old version before
        # designed the color supervision mask (so it is rare). ie.. 
        # in single epoch mixing random + structured sequence. merge those since
        # the subject doesnt know.
        _epochs_to_merge = [("AnBmTR", "mask|0||0"), ("TR", "mask|0||0"), ("rndstr", "mask|0||0")]
        _new_epoch_name = "rank_mask"
        list_epoch_merge = [
            (_epochs_to_merge, _new_epoch_name)
        ]
        # epoch_merge_key = "epoch_superv"
        epoch_merge_key = "epoch"
    else:
        list_epoch_merge = []
        epoch_merge_key = None


    if False:
        # this all folded into section below. Keeping bevause it has good notes on the dates.
        ############# LOOK FOR "SAME_BEH" BASED ON EPOCHSETS?
        # (Useful if >2 epochs, then same_beh would be confused)
        # (Or if each task can be done different ways, such as for random sequence rank)
        if animal=="Pancho" and DATE in  [220928, 220929, 220930, 221002, 221014, 221001, 221102]:
            # Color-supervision -- ie single epochs which combine
            # random sequence + structured sequence. merge those since
            # the subject doesnt know.
            DO_CHARSEQ_VER = "task_matlab"
            EXTRACT_EPOCHSETS = True
            EXTRACT_EPOCHSETS_trial_label = "char_seq"
            EXTRACT_EPOCHSETS_n_max_epochs = 3
        elif animal =="Pancho" and DATE in [220921]:
            # Sequence mask supervision, i..e, an old version before
            # designed the color supervision mask (so it is rare). ie.. 
            # in single epoch mixing random + structured sequence. merge those since
            # the subject doesnt know.
            DO_CHARSEQ_VER = "task_matlab"
            EXTRACT_EPOCHSETS = True
            EXTRACT_EPOCHSETS_trial_label = "char_seq"
            EXTRACT_EPOCHSETS_n_max_epochs = 3
        elif animal =="Pancho" and DATE in [221021]:
            # Cases with >2 epochs, then need to use trial-level definition of "same-beh",
            # which is done here
            DO_CHARSEQ_VER = "task_matlab"
            EXTRACT_EPOCHSETS = True
            EXTRACT_EPOCHSETS_trial_label = "char_seq"
            EXTRACT_EPOCHSETS_n_max_epochs = 3
        else:
            DO_CHARSEQ_VER = None
            EXTRACT_EPOCHSETS = False
            EXTRACT_EPOCHSETS_trial_label = None
            EXTRACT_EPOCHSETS_n_max_epochs = None

    ############ rules will generally need to use this.
    if ANALY_VER in ["ruleswERROR", "rulesw", "ruleswALLDATA"]:
        DO_CHARSEQ_VER = "task_matlab"
        EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "char_seq"
        EXTRACT_EPOCHSETS_n_max_epochs = 3
        EXTRACT_EPOCHSETS_merge_sets = True
    else:
        DO_CHARSEQ_VER = None
        EXTRACT_EPOCHSETS = False
        EXTRACT_EPOCHSETS_trial_label = None
        EXTRACT_EPOCHSETS_n_max_epochs = None
        EXTRACT_EPOCHSETS_merge_sets = None

    ################ FEATURES TO EXTRACT
    if which_level=="trial":
        if ANALY_VER in ["ruleswALLDATA", "ruleswERROR", "rulesw", "seqcontext", "singleprim"]:
            list_features_modulation_append = ["probe", "taskgroup", "character", "trialcode", "epoch",
                                                "epoch_superv",  
                                                "task_kind", "supervision_stage_concise"]
        else:
            assert False

        if DO_EXTRACT_CONTEXT:
            # These are features that are gotten from extracting context
            list_features_modulation_append = list_features_modulation_append + ["seqc_nstrokes_beh", "seqc_nstrokes_task",
                                               "seqc_0_shape", "seqc_0_loc", "seqc_0_loc_shape",
                                               "seqc_1_shape", "seqc_1_loc", "seqc_1_loc_shape",
                                               "seqc_2_shape", "seqc_2_loc", "seqc_2_loc_shape",
                                               "seqc_3_shape", "seqc_3_loc", "seqc_3_loc_shape",
                                               "gridsize"]
    elif which_level=="stroke":
        if ANALY_VER in ["ruleswALLDATA", "ruleswERROR", "rulesw", "seqcontext", "singleprim"]:
            list_features_modulation_append = ["probe", "taskgroup", "character", "trialcode", "epoch",
                                                "epoch_superv",  
                                                "task_kind", "supervision_stage_concise",
                                                "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next", "CTXT_shape_next",
                                                "shape_oriented", "gridloc", "stroke_index"]
        else:
            assert False        
    else:
        print(which_level)
        assert False, "typo?"                                               

    if DO_CHARSEQ_VER:
        list_features_modulation_append.append("char_seq")                                        
    if EXTRACT_EPOCHSETS:
        list_features_modulation_append.append("epochset")               
    if DO_SCORE_SEQUENCE_VER:
        list_features_modulation_append.append("success_binary_quick")                         
    if DO_EXTRACT_EPOCHKIND:
        list_features_modulation_append.append("epochkind")                         

    # print(list_features_modulation_append)
    # assert False

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
        "list_features_modulation_append":list_features_modulation_append,
        "ANALY_VER":ANALY_VER,
        "taskgroup_reassign_simple_neural":taskgroup_reassign_simple_neural,
        "list_epoch_merge":list_epoch_merge,
        "DO_CHARSEQ_VER":DO_CHARSEQ_VER,
        "EXTRACT_EPOCHSETS":EXTRACT_EPOCHSETS,
        "EXTRACT_EPOCHSETS_trial_label":EXTRACT_EPOCHSETS_trial_label,
        "EXTRACT_EPOCHSETS_n_max_epochs":EXTRACT_EPOCHSETS_n_max_epochs,
        "EXTRACT_EPOCHSETS_merge_sets":EXTRACT_EPOCHSETS_merge_sets,
        "epoch_merge_key":epoch_merge_key,
        "DO_EXTRACT_EPOCHKIND":DO_EXTRACT_EPOCHKIND
        }

    return params