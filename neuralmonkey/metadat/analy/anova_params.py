""" Holds params for anava analysis for each date
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile

ONLY_ESSENTIAL_VARS = False # then just the first var, assuemd to be most important, for quick analys

##################
LIST_ANALYSES = ["rulesw", "rulesingle", "ruleswALLDATA", "ruleswERROR", "singleprimvar", "seqcontextvar",
                 "seqcontext", "singleprim", "charstrokes", "chartrial", "substrokes_sp", "PIG_BASE"] # repo of possible analses,

# - rulesw, rule switching
# - rulesingle, a single rule, look for draw program representations.

# def exptlist_getter(self):

#     LIST_EXPTS = [
#         ("Pancho", 220709, "trial", "seqcontext"),
#         ("Pancho", 230105, "trial", "seqcontext"),
#     ]

#     return LIST_EXPTS

def _params_score_sequence_ver(animal, DATE, ANALY_VER):
    """ Decide how to score each trial's sequence success, either
    comparing beh to matlab task seuqence or to parses"""
    if ANALY_VER in ["rulesw", "ruleswERROR", "ruleswALLDATA", "rulesingle"]:
        if animal=="Pancho" and DATE in [220913]:
            DO_SCORE_SEQUENCE_VER = "parses"
        elif animal=="Pancho" and DATE in [220812, 220814, 220815, 220816, 220827,
            220921, 220928, 220929, 220930, 221001, 221002, 221014, 221020, 221021, 221023, 221024,
            221031, 221102, 221107, 221112, 221113, 221114, 221118, 221119, 221121, 221125]:
            # determenistic (single solution)
            # DO_SCORE_SEQUENCE_VER = "matlab"
            DO_SCORE_SEQUENCE_VER = "parses" # This is better!! It is required if you want to extract chunk features (e.g, chunk rank)
            # which is done below and so "matlab" would run into error.
        elif animal=="Diego" and DATE in [230701, 230702, 230703, 230704, 230705, 230706, 230707, 230713, 230717, 230719, 230802]:
            # A single correct sequence
            # DO_SCORE_SEQUENCE_VER = "matlab"
            DO_SCORE_SEQUENCE_VER = "parses" # This is better!! It is required if you want to extract chunk features (e.g, chunk rank)
            # which is done below and so "matlab" would run into error.
        elif animal in ["Diego", "Pancho"]:
            # For chunks analysis (e.g., single rule). Lately (Feb 2024) extracting cjhunks
            # seems to require parse version...
            DO_SCORE_SEQUENCE_VER = "parses"
        else:
            print(animal, DATE)
            assert False
    elif ANALY_VER in ["singleprimvar", "seqcontext", "singleprim", "seqcontextvar", "charstrokes", "chartrial", "substrokes_sp", "PIG_BASE"]:
        DO_SCORE_SEQUENCE_VER = None
    else:
        assert False
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
    globals_nmin = 7
    globals_lenient_allow_data_if_has_n_levels = 2
    VARS_GROUP_PLOT = []

    ################## SUPERVISION LEVELS TO KEEP
    ################# OPTIONALLY KEEP ONLY SPECIFIC "FULL" SUPERVISION NAMES
    # if animal=="Pancho" and DATE==220921:
    #     # This day has "randomseq-sequence mask"... (rare)
    #     # Therefore want to keep sequence mask
    #     # (An epoch is random sequence using sequence mask)
    #     assert False, "check this days supervision_stage_new, make sure the 1111 is correct."
    #     list_superv_keep_full = ["mask|0.5|0|1111||0|none",
    #                               "off|1.0|0|1111||0|none",
    #                               "off|1.0|0|1111}|0|none"]
    #     list_superv_keep = "all" # None means just keep all cases that are "no supervision"
    # else:
    #     # Prune so is just "no supervision" (i.e., list_superv_keep is None)
    #     list_superv_keep_full = None
    #     list_superv_keep = None # None means just keep all cases that are "no supervision"


    ################ SCORE PERFORMANCE?
    # DO_SCORE_SEQUENCE_VER = _params_score_sequence_ver(animal, DATE, ANALY_VER)

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
            "epoch", # 2) epoch encoidng
            "epoch", #
            "epoch", #
            "gridloc", # 1) state space expansion
            "shape_oriented",
            # "gridloc", # 1) shape vs. loc
            # "shape_oriented", #
            # "shape_oriented",
            # "gridloc",
            "shape_oriented", # 3) prim encoding.
            "gridloc",
            "shape_oriented",
            "gridloc",
            "stroke_index",
            # "character"
        ]
        LIST_VARS_CONJUNCTION = [
            ["shape_oriented", "gridloc", "CTXT_locshape_prev", "CTXT_locshape_next"],
            ["shape_oriented", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev"],
            ["epochset", "shape_oriented", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev"],
            ["shape_oriented", "epochkind"], # this allows combining multiple "direction" epochs into a single epochkind (state space expansion).
            ["gridloc", "epochkind"], # this allows combining multiple "direction" epochs into a single epochkind (state space expansion).
            # ["shape_oriented", "epoch"], # removed stroke_index, since this is not dissociated from both location and shape (for a given rule).
            # ["gridloc", "epoch"], # removed stroke_index, since this is not dissociated from both location and shape (for a given rule).
            # ["gridloc", "epoch", "stroke_index"], # useful for comparison across areas
            # ["shape_oriented", "epoch", "stroke_index"], 
            ["gridloc", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"], # useful for comparison across areas
            ["shape_oriented", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"], # useful for comparison across areas
            ["gridloc", "epoch", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"], # useful for comparison across areas
            ["shape_oriented", "epoch", "stroke_index", "CTXT_loc_prev", "CTXT_shape_prev"],
            ["gridloc", "shape_oriented", "stroke_index", "epoch"],
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

        if False:
            # Older, separating shape and location. This takes too long?
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
            ]           
        else:

            LIST_VAR = [

                # "seqc_3_loc_shape", # same n strokes, just diff sequence
                # "seqc_3_loc_shape", #  
                # "seqc_3_loc_shape", # same shape config

                "seqc_3_loc_shape", # DIFF SHAPE/LOC, same stim entirely
                "seqc_3_shape", # DIFF SHAPE, same loc
                "seqc_3_loc", # DIFF LOC, same shape

                "seqc_2_loc_shape", # DIFF SHAPE/LOC, same stim entirely
                "seqc_2_shape", # DIFF SHAPE, same loc
                "seqc_2_loc", # DIFF LOC, same shape

                "seqc_0_loc_shape", # DIFF SHAPE/LOC, same stim entirely
                "seqc_0_loc", # DIFF SHAPE/LOC, same stim entirely
                "seqc_0_shape", # DIFF SHAPE/LOC, same stim entirely

                "seqc_1_loc_shape", # DIFF SHAPE/LOC, same stim entirely
                "seqc_1_shape", # DIFF SHAPE, same loc
                "seqc_1_loc", # DIFF LOC, same shape

                "seqc_nstrokes_beh", # DIFF N STROKES, same seq/loc sequence
                "seqc_nstrokes_beh",
                "seqc_nstrokes_beh",

                "seqc_3_loc_shape", # DIFF SHAPE/LOC
                "seqc_2_loc_shape", # DIFF SHAPE/LOC
                "seqc_1_loc_shape", # DIFF SHAPE/LOC
                ]
            LIST_VARS_CONJUNCTION = [

                # ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"], 
                # ["taskconfig_loc", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
                # ["taskconfig_shp", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
                
                ["taskconfig_shploc", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape", "seqc_3_loc"], 
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape", "seqc_3_shape"], 

                ["taskconfig_shploc", "seqc_0_loc_shape", "seqc_1_loc_shape"],
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc"], 
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_shape"], 

                ["taskconfig_shploc"],
                ["taskconfig_shploc", "seqc_0_shape"],
                ["taskconfig_shploc", "seqc_0_loc"],

                ["taskconfig_shploc", "seqc_0_loc_shape"],
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc"], 
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_shape"], 

                ["seqc_0_loc_shape"], # diff n strokes.
                ["seqc_0_loc_shape", "seqc_1_loc_shape"],
                ["seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"],
                
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape", "seqc_2_loc_shape"], 
                ["seqc_nstrokes_beh", "seqc_0_loc_shape", "seqc_1_loc_shape"], 
                ["seqc_nstrokes_beh", "seqc_0_loc_shape"], 
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
                "epoch", # abstract rule?
                "character", # visually repsonsive?
                "seqc_0_loc",
                "seqc_0_shape",
                "epoch",
                "seqc_1_loc_shape",
                "seqc_1_loc_shape",
            ]
            LIST_VARS_CONJUNCTION = [
                ["epochset"],
                ["epochset", "character"],
                ["epoch", "epochset"],
                ["epoch", "seqc_0_shape"],
                ["epoch", "seqc_0_loc"],
                ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
                ["epoch"],
                ["epoch", "seqc_0_loc_shape"],
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

        elif DATE in [221002, 221014, 221020, 221021, 221023, 221024] or DATE in [230701, 230702, 230703, 230704, 230705, 230706, 230707, 230713, 230717, 230719, 230802]:
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

    elif which_level=="trial" and ANALY_VER in ["seqcontextvar", "singleprimvar"]:
        # NOTE: for seqcontextvar this is temporary, ideal;ly consider entie sequence context, not just first prim.
        LIST_VAR = [
            "seqc_0_shape", # Controlled: exact same tasks (defined by shape/loc) across epochs
            "seqc_0_loc",

            "seqc_0_shape", # Less well controlled, tasks not identical acorss epihcs. but useful for analysis before onset of stim
            "seqc_0_loc",
        ]
        LIST_VARS_CONJUNCTION = [
            ["epochset", "epoch", "seqc_0_loc"],
            ["epochset", "epoch", "seqc_0_shape"],

            ["epoch", "seqc_0_loc"],
            ["epoch", "seqc_0_shape"],
        ]

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

        VARS_GROUP_PLOT = [
            ["epochset", "epoch"]
        ]
    else:
        print(which_level, ANALY_VER)
        assert False

    ####################### CLEAN UP VARS
    if ONLY_ESSENTIAL_VARS or (ANALY_VER in ["ruleswERROR"]): 
        # just test epoch, for error trials
        LIST_VAR = LIST_VAR[:2]
        LIST_VARS_CONJUNCTION = LIST_VARS_CONJUNCTION[:2]

    ## If you have success as a variable then you cannot prune to only keep success...
    def _vars_including_success_binary(var, vars_others):
        """ Return True if any var or other var cares about performance succes...
        """
        if "success_binary" in var:
            return True
        if any(["success_binary" in v for v in vars_others]):
            return True
        return False

    ##### Auto decide what to extract based on vars
    LIST_VAR_ALL = [x for x in LIST_VAR]
    for vars_conj in LIST_VARS_CONJUNCTION:
        LIST_VAR_ALL.extend(vars_conj)

    #### AUTO EXTRACT BASED ON WHAT VARS ARE EXTRACTED
    assert False, "get it from params_getter_dataset_preprocess below..."
    # DO_EXTRACT_TASKCONFIG = False
    # if "taskconfig_loc" in LIST_VAR_ALL or "taskconfig_shp" in LIST_VAR_ALL or "taskconfig_shploc" in LIST_VAR_ALL:
    #     DO_EXTRACT_TASKCONFIG = True
    # if ANALY_VER=="seqcontextvar":
    #     DO_EXTRACT_TASKCONFIG = True

    
    print("Got these LIST_VAR and LIST_VARS_CONJUNCTION:")
    print(LIST_VAR)
    print(LIST_VARS_CONJUNCTION)
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
        "VARS_GROUP_PLOT":VARS_GROUP_PLOT,
        "DO_EXTRACT_TASKCONFIG":DO_EXTRACT_TASKCONFIG,
        # "DO_SCORE_SEQUENCE_VER":DO_SCORE_SEQUENCE_VER,
        # "list_superv_keep_full":list_superv_keep_full,
        # "list_superv_keep":list_superv_keep,
        # "preprocess_steps_append":preprocess_steps_append,
        # "remove_aborts":remove_aborts,
    }

    ### APPEND PARAMS FOR PREPROCESSING DATASET
    params_dataset = params_getter_dataset_preprocess(ANALY_VER)
    for k, v in params_dataset.items():
        assert k not in params
        params[k] = v

    assert len(list_events) == len(list_pre_dur)
    assert len(list_events) == len(list_post_dur)

    return params

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

    assert False, "get this from params_getter_dataset_preprocess"

    ################ SEQUENCE CONTEXT?
    DO_EXTRACT_CONTEXT = True    


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

    assert False, "get from pgdp."

    ######### FINALLY, FLIP SOME "DO_" FLAGS BASD ONW HAT VAR YOU WANT.
    params_plots = params_getter_plots(animal, DATE, which_level, ANALY_VER)

    ##### Auto decide what to extract based on vars (AUTO EXTRACT BASED ON WHAT VARS ARE EXTRACTED)
    LIST_VAR_ALL = [x for x in params_plots["LIST_VAR"]]
    for vars_conj in params_plots["LIST_VARS_CONJUNCTION"]:
        LIST_VAR_ALL.extend(vars_conj)
    # if "epochkind" in LIST_VAR_ALL:
    #     DO_EXTRACT_EPOCHKIND = True
    # else:
    #     DO_EXTRACT_EPOCHKIND = False

    ###### Feature to get, based on flags.
    if DO_CHARSEQ_VER:
        list_features_modulation_append.append("char_seq")                                        
    if EXTRACT_EPOCHSETS:
        list_features_modulation_append.append("epochset")               
    if DO_SCORE_SEQUENCE_VER:
        list_features_modulation_append.append("success_binary_quick")                         
    list_features_modulation_append.append("epochkind")


    # make sure all vars that you will use for plots are included in extraction
    if False: # STOPPED, since I always reextract during plotting...
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
        # "DO_EXTRACT_EPOCHKIND":DO_EXTRACT_EPOCHKIND
        }

    return params


# def dataset_apply_params_OLD(ListD, animal, DATE, which_level, ANALY_VER, anova_interaction=False):
#     """Preprocess dataset in all ways, including pruning, appending/modifying columns, etc.
#     PARAMS:
#     - ListD, list of Datasets. will operate onb
#     ## each, then concatenate.
#     RETURNS:
#     - Dall, concated datasets, processed but not yet trial-pruned [COPY OF D]
#     - dataset_pruned_for_trial_analysis, same, but trial-pruend. this is final.
#     - TRIALCODES_KEEP, lsit of unique trialcodes in dataset_pruned_for_trial_analysis
#     - params, dict, params for ploting
#     - params_extraction,. dict params for extafcvtion data.
#     """
#     from pythonlib.dataset.analy_dlist import concatDatasets
#     from neuralmonkey.classes.snippets import _dataset_extract_prune_general, dataset_extract_prune_general_dataset
#
#     ################################### LOAD PARAMS
#     params = params_getter_plots(animal, DATE, which_level, ANALY_VER,
#         anova_interaction=anova_interaction)
#     params_extraction = params_getter_extraction(animal, DATE, which_level, ANALY_VER)
#
#     assert False, "first concat, then apply..."
#
#     ################# BEH DATASET
#     # First, concatenate all D.
#     list_dataset = []
#     for i, D in enumerate(ListD):
#         # if which_level=="trial":
#         # use the dataset here, since it is not saved
#         # D = sn.Datasetbeh
#         # else:
#         #     # use the dataset linked to DS, since it is saved
#         #     D = SP.DSmult[i].Dataset
#         #     assert len(D.Dat)==len(sn.Datasetbeh.Dat), "a sanity check. desnt have to be, but I am curious why it is not..."
#
#         # Becasue dataset can be locked, just replace it with copy
#         D = D.copy()
#
#
#
#
#         # THINGs that must be done by each individual D
#         D.behclass_preprocess_wrapper()
#
#         # Second, do preprocessing to concatted D
#         if params_extraction["DO_SCORE_SEQUENCE_VER"]=="parses":
#             D.grammarparses_successbinary_score()
#         elif params_extraction["DO_SCORE_SEQUENCE_VER"]=="matlab":
#             D.grammarmatlab_successbinary_score()
#         else:
#             # dont score
#             assert params_extraction["DO_SCORE_SEQUENCE_VER"] is None
#
#         if params_extraction["taskgroup_reassign_simple_neural"]:
#             # do here, so the new taskgroup can be used as a feature.
#             D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)
#             print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
#             D.grouping_print_n_samples(["taskgroup", "probe"])
#
#         if params_extraction["DO_CHARSEQ_VER"] is not None:
#             D.sequence_char_taskclass_assign_char_seq(ver=params_extraction["DO_CHARSEQ_VER"])
#
#
#
#
#         list_dataset.append(D.copy())
#     # concat the datasets
#     Dall = concatDatasets(list_dataset)
#
#     ################ DO SAME THING AS IN EXTRACTION (these dont fail, when use concatted)
#     if params_extraction["DO_EXTRACT_CONTEXT"]:
#         Dall.seqcontext_preprocess()
#
#     if params["DO_EXTRACT_TASKCONFIG"]:
#         Dall.taskclass_shapes_loc_configuration_assign_column()
#
#     for this in params_extraction["list_epoch_merge"]:
#         # D.supervision_epochs_merge_these(["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
#         Dall.supervision_epochs_merge_these(this[0], this[1], key=params_extraction["epoch_merge_key"],
#             assert_list_epochs_exist=False)
#
#
#     if params_extraction["EXTRACT_EPOCHSETS"]:
#         if params_extraction["EXTRACT_EPOCHSETS_trial_label"] == "char_seq":
#             # This useful to separate into cases with same first stroke, and also chars present across contexts,
#             # separating out single prims if they exist into their own epochset.
#             versions_ordered = ["char", "same_beh_first_stroke", "same_beh"]
#             Dall.epochset_apply_sequence_wrapper(versions_ordered=versions_ordered)
#         else:
#             # Only apply epochset extraction once.
#             Dall.epochset_extract_common_epoch_sets(
#                 trial_label=params_extraction["EXTRACT_EPOCHSETS_trial_label"],
#                 n_max_epochs=params_extraction["EXTRACT_EPOCHSETS_n_max_epochs"],
#                 merge_sets_with_only_single_epoch=params_extraction["EXTRACT_EPOCHSETS_merge_sets"],
#                 merge_sets_with_only_single_epoch_name = ("LEFTOVER",))
#
#     if params_extraction["DO_EXTRACT_EPOCHKIND"]:
#         Dall.supervision_epochs_extract_epochkind()
#
#
#     # Sanity check that didn't remove too much data.
#     if False:
#         if "wrong_sequencing_binary_score" not in params["preprocess_steps_append"]:
#             # Skip if is error trials.
#             npre = len(D.Dat)
#             npost = len(dat_pruned.Dat)
#             if npost/npre<0.25 and len(sn.Datasetbeh.Dat)>200: # ie ignore this if it is a small session...
#                 print(params)
#                 print("THis has no wrong_sequencing_binary_score: ",  params['preprocess_steps_append'])
#                 assert False, "dataset pruning removed >0.75 of data. Are you sure correct? Maybe removing a supervisiuon stage that is actually important?"
#
#     ###### PRUNE DATASET TO GET SUBSET TRIALCODES
#     # Only keep subset these trialcodes
#     dataset_pruned_for_trial_analysis = _dataset_extract_prune_general_dataset(Dall,
#         list_superv_keep=params["list_superv_keep"],
#         preprocess_steps_append=params["preprocess_steps_append"],
#         remove_aborts=params["remove_aborts"],
#         list_superv_keep_full=params["list_superv_keep_full"],
#         )
#     TRIALCODES_KEEP = dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist()
#
#     return Dall, dataset_pruned_for_trial_analysis, TRIALCODES_KEEP, params, params_extraction


def dataset_apply_params(D, DS, ANALY_VER, animal, DATE, save_substroke_preprocess_figures=True):
    """Preprocess dataset in all ways, including pruning, appending/modifying columns, etc.
    PARAMS:
    - ANALY_VER, str, params to use.
    - ListD, list of Datasets. will operate onb
    ## each, then concatenate.
    RETURNS:
    - Dall, concated datasets, processed but not yet trial-pruned [COPY OF D]
    - dataset_pruned_for_trial_analysis, same, but trial-pruend. this is final.
    - TRIALCODES_KEEP, lsit of unique trialcodes in dataset_pruned_for_trial_analysis
    - params, dict, params for ploting
    - params_extraction,. dict params for extafcvtion data.
    """
    from pythonlib.dataset.analy_dlist import concatDatasets
    from neuralmonkey.classes.snippets import dataset_extract_prune_general_dataset

    D = D.copy()

    if ANALY_VER == "MINIMAL":
        # This is for when have not inlucded this in params_getter_dataset_preprocess yet, but do minimal cleaning.
        # (Did for kedar, fixation stuff).
        D = dataset_extract_prune_general_dataset(D,
                                                  list_superv_keep=None,
                                                  preprocess_steps_append=None,
                                                  remove_aborts=None,
                                                  list_superv_keep_full=None,
                                                  )
        return D, DS, None
    else:

        ################################### LOAD PARAMS
        params = params_getter_dataset_preprocess(ANALY_VER, animal, DATE)
        # params = params_getter_plots(animal, DATE, which_level, ANALY_VER,
        #     anova_interaction=anova_interaction)
        # params_extraction = params_getter_extraction(animal, DATE, which_level, ANALY_VER)

        # assert False, "first concat, then apply..."
        print("Dataset preprocess, these params:")
        print(params)

        ################# BEH DATASET
        # First, concatenate all D.
        # Becasue dataset can be locked, just replace it with copy
        # print("5 dfafasf", D.TokensVersion)

        # Second, do preprocessing to concatted D
        if params["DO_SCORE_SEQUENCE_VER"]=="parses":
            D.grammarparses_successbinary_score_wrapper()
        elif params["DO_SCORE_SEQUENCE_VER"]=="matlab":
            D.grammarmatlab_successbinary_score()
            assert params["datasetstrokes_extract_chunks_variables"]==False, "if True, you need to use parses method."
        else:
            # dont score
            assert params["DO_SCORE_SEQUENCE_VER"] is None

        if params["taskgroup_reassign_simple_neural"]:
            # do here, so the new taskgroup can be used as a feature.
            D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)
            print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
            D.grouping_print_n_samples(["taskgroup", "probe"])

        if params["DO_CHARSEQ_VER"] is not None:
            D.sequence_char_taskclass_assign_char_seq(ver=params["DO_CHARSEQ_VER"])

        ################ DO SAME THING AS IN EXTRACTION (these dont fail, when use concatted)
        # D.tokens_append_to_dataframe_column()
        # NOTE: This might take time, as it requires extract DS...
        if False: # done in general prerpocessing
            D.shapesemantic_classify_novel_shape()
            D.strokes_onsets_offsets_location_append()
            D.tokens_cluster_touch_onset_loc_across_all_data() # cluster location of storke onests --> tok["loc_on_clust"]
            D.tokens_cluster_touch_offset_loc_across_all_data() # Cluster offset locations --> tok["loc_off_clust"]
        if False:
            # Older, works, but decide to do waht I do for characters (below) more genrelaiy.
            D.seqcontext_preprocess()
        else:
            ########################### CHARACTERS, stuff related to (i) image parse and (ii) beh sequence
            # (1) Extract Vraibles defined within each stroke
            # (1) Extract Vraibles defined within each stroke
            if False: # done in general prerpocessing
                for ind in range(len(D.Dat)):
                    # Beh strokes
                    Tk_behdata = D.taskclass_tokens_extract_wrapper(ind, "beh_using_beh_data", return_as_tokensclass=True)
                    Tk_behdata.features_extract_wrapper(["loc_on", "angle"])

                    Tk_behdata = D.taskclass_tokens_extract_wrapper(ind, "beh_using_task_data", return_as_tokensclass=True)
                    Tk_behdata.features_extract_wrapper(["shape_semantic"])

                    # Task strokes (ignore beh)
                    Tk_taskdata = D.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True)
                    Tk_taskdata.features_extract_wrapper(["shape_semantic"])

                # (2) Compute all binned data, using beh data
                PLOT = False
                nbins = 3 # 2 or 3...
                D.tokens_bin_feature_across_all_data("loc_on", "beh_using_beh_data", nbins=nbins, PLOT=PLOT)
                D.tokens_bin_feature_across_all_data("angle", "beh_using_beh_data", nbins=nbins, PLOT=PLOT)

                D.tokens_bin_feature_across_all_data("center", "beh_using_beh_data", nbins=nbins, PLOT=PLOT)
                D.tokens_bin_feature_across_all_data("center", "beh_using_task_data", nbins=nbins, PLOT=PLOT)
                D.tokens_bin_feature_across_all_data("center", "task", nbins=nbins, PLOT=PLOT)

                # Get locon_bin_in_loc
                # D.tokens_sequence_bin_location_within_gridloc()

                # Replace loc, for char, with loc within gridloc.
                # And then get shape_loc conjunctions
                D.tokens_gridloc_replace_with_recomputed_loc_chars()

                # (3) IMAGE PARSE
                D.taskclass_shapes_loc_configuration_assign_column()
                # 1. specific
                D.taskclass_shapes_loc_configuration_assign_column(version="char", shape_ver="shape_semantic", suffix="SHSEM", plot_examples=PLOT)
                # 2. more lenient
                D.taskclass_shapes_loc_configuration_assign_column(version="char", shape_ver="shape_semantic_cat", suffix="SHSEMCAT", plot_examples=PLOT)

                # (4) LAST: Extract new seq context variables, based on variables in tokens.
                D.seqcontext_preprocess(plot_examples=PLOT, force_run=True)
            ########################### (end)

        for this in params["list_epoch_merge"]:
            # D.supervision_epochs_merge_these(["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
            D.supervision_epochs_merge_these(this[0], this[1], key=params["epoch_merge_key"],
                assert_list_epochs_exist=False)

        # Always extract epochset, simply so that can hard-code extract "epochset" to Snippets.
        # if params["EXTRACT_EPOCHSETS"]:
        if params["EXTRACT_EPOCHSETS_trial_label"] == "char_seq":
            # This useful to separate into cases with same first stroke, and also chars present across contexts,
            # separating out single prims if they exist into their own epochset.
            versions_ordered = ["char", "same_beh_first_stroke", "same_beh"]
            D.epochset_apply_sequence_wrapper(versions_ordered=versions_ordered)
        else:
            # Only apply epochset extraction once.
            D.epochset_extract_common_epoch_sets(
                trial_label=params["EXTRACT_EPOCHSETS_trial_label"],
                n_max_epochs=params["EXTRACT_EPOCHSETS_n_max_epochs"],
                merge_sets_with_only_single_epoch=params["EXTRACT_EPOCHSETS_merge_sets"],
                merge_sets_with_only_single_epoch_name = ("LEFTOVER",))

        if params["DO_EXTRACT_EPOCHKIND"]:
            D.supervision_epochs_extract_epochkind()

        # # Sanity check that didn't remove too much data.
        # if False:
        #     if "wrong_sequencing_binary_score" not in params["preprocess_steps_append"]:
        #         # Skip if is error trials.
        #         npre = len(D.Dat)
        #         npost = len(dat_pruned.Dat)
        #         if npost/npre<0.25 and len(sn.Datasetbeh.Dat)>200: # ie ignore this if it is a small session...
        #             print(params)
        #             print("THis has no wrong_sequencing_binary_score: ",  params['preprocess_steps_append'])
        #             assert False, "dataset pruning removed >0.75 of data. Are you sure correct? Maybe removing a supervisiuon stage that is actually important?"

        # Append variables by hand
        # D = self.datasetbeh_extract_dataset()
        if "FEAT_num_strokes_task" not in D.Dat.columns:
            D.extract_beh_features()
        # if "char_seq" not in D.Dat.columns:
        #     D.sequence_char_taskclass_assign_char_seq()
        if "seqc_nstrokes_task" not in D.Dat.columns:
            D.seqcontext_preprocess()

        # Load character clust labels (do this BEFORE prune anything)
        # This also replaces all seqc_{}_shape labels in D.Dat
        if params["charclust_dataset_extract_shapes"]:
            assert False, "IGNORE -- now this is done in dataset defautl preprocesing."
            D.charclust_shape_labels_extract_presaved_from_DS()

        ###### PRUNE DATASET TO GET SUBSET TRIALCODES
        # Only keep subset these trialcodes
        D = dataset_extract_prune_general_dataset(D,
                                                  list_superv_keep=params["list_superv_keep"],
                                                  preprocess_steps_append=params["preprocess_steps_append"],
                                                  remove_aborts=params["remove_aborts"],
                                                  list_superv_keep_full=params["list_superv_keep_full"],
                                                  )

        ################### PRUNE STROKES --> THEN PRUNE DATASET GIVEN THAT
        from neuralmonkey.classes.snippets import datasetstrokes_extract
        if params["datasetstrokes_extract_to_prune_trial"] is not None:
            # Only keep good strokes
            ds = datasetstrokes_extract(D, params["datasetstrokes_extract_to_prune_trial"])

            # Remove these trialcodes from original dataset
            list_tc = ds._dataset_find_trialcodes_incomplete_data(D=D)
            inds_remove = [D.index_by_trialcode(tc) for tc in list_tc]
            D.Dat = D.Dat.drop(index = inds_remove).reset_index(drop=True)

        if params["datasetstrokes_extract_to_prune_stroke_and_get_features"] is not None:
            # Only keep good strokes
            dsprun = datasetstrokes_extract(D, params["datasetstrokes_extract_to_prune_stroke_and_get_features"])
            DS = dsprun # Replace with newly extracted.

        ############### SUBSTROKES
        if params["substrokes_features_do_extraction"]:
            from pythonlib.dataset.substrokes import features_motor_extract_and_bin
            assert params["datasetstrokes_extract_to_prune_stroke_and_get_features"] is None, "they would overwrite each other"

            # Save in substrokes preprocess folder.
            if save_substroke_preprocess_figures: # Takes too long
                SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("substrokes_preprocess")
                plot_save_dir = f"{SAVEDIR}/plots_during_anova_params"
                os.makedirs(plot_save_dir, exist_ok=True)
            else:
                plot_save_dir = None

            # from pythonlib.tools.expttools import writeDictToTxt

            # Extract motor variables (DS)
            features_motor_extract_and_bin(DS, plot_save_dir=plot_save_dir)

        ################ CHUNKS, STROKES (e.g., singlerule, AnBm)
        if params["datasetstrokes_extract_chunks_variables"]:
            # Extract chunk variables from Dataset
            for i in range(len(D.Dat)):
                D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(i)

            # Also extract "syntax parse" e.g., (3,1,1) for A3B1C1.
            # Also called "taskcat_by_rule"
            D.grammarparses_classify_tasks_categorize_based_on_rule()
            print("These are the SYNTAX PARSES (i.e., 'taskcat_by_rule'):")
            print(D.Dat["taskcat_by_rule"].value_counts())

            ###################################################
            ################# SNTAX-RELATED VARIABLES...
            # And extract syntax_concrete column
            D.grammarparses_syntax_concrete_append_column()

            # For each sequence kind (e.g. shapes) split into concrete variations (classes).
            savedir_preprocess = D.make_savedir_for_analysis_figures_BETTER("preprocess_general")
            sdir = f"{savedir_preprocess}/seqcontext_behorder_cluster_concrete_variation"
            os.makedirs(sdir, exist_ok=True)
            D.seqcontext_behorder_cluster_concrete_variation(SAVEDIR=sdir,
                                                 LIST_VAR_BEHORDER=["behseq_shapes", "behseq_locs",
                                                                    "behseq_locs_x", "behseq_locs_diff", "behseq_locs_diff_x"])

            if True:
                # Wrapper to get all info about epochs
                D.grammarparses_rules_epochs_superv_summarize_wrapper(PRINT=True)
            else:
                # Add column "epoch_rand", which collects random + color instruction
                print(" *** RUNNING: grammarparses_rules_random_sequence()")
                D.grammarparses_rules_random_sequence(PRINT=True)

            # Need to run this here, and not in preprocess/general.py (can't remember why...)
            if D.animals(force_single=True)[0]=="Pancho" and int(D.dates(True)[0])>=220902 and int(D.dates(True)[0])<=220909:
                # AnBm, with two shape ses switching by trail in same day.
                # Replace epoch and syntax_concrete so shapaes are diff epoch, but same synta concrete.
                list_epoch = []
                list_syntax_concrete = []
                for i, row in D.Dat.iterrows():
                    tmp = [x>0 for x in row["syntax_concrete"][:4]]
                    epoch_orig = row["epoch_orig"]
                    if tmp == [True, False, True, False]:
                        list_epoch.append(f"{epoch_orig}|A")
                        list_syntax_concrete.append((row["syntax_concrete"][0], row["syntax_concrete"][2]))
                    elif tmp == [False, True, False, True]:
                        list_epoch.append(f"{epoch_orig}|B")
                        list_syntax_concrete.append((row["syntax_concrete"][1], row["syntax_concrete"][3]))
                    else:
                        print(tmp)
                        print(row["syntax_concrete"])
                        print(row["epoch_orig"])
                        assert False
                # D.Dat["epoch_orig"] = list_epoch # NO!! this leads to rulestring problems (old note: must update epoch_orig (and not epoch) or else grammarparses_syntax_role_append_to_tokens)
                D.Dat["epoch"] = list_epoch # must update epoch_orig (and not epoch) or else grammarparses_syntax_role_append_to_tokens
                # will incorrectly count within all data.
                D.Dat["syntax_concrete"] = list_syntax_concrete

            # Further bin trials based on variation in gap duration --> longer gaps means difference in preSMA state space?
            sdir = f"{savedir_preprocess}/grammarparses_chunk_transitions_gaps_extract_batch"
            os.makedirs(sdir, exist_ok=True)
            D.grammarparses_chunk_transitions_gaps_extract_batch(plot_savedir=sdir)

            # For each token, assign a new key called "syntax role" -- good.
            D.grammarparses_syntax_role_append_to_tokens()

            plt.close("all")

        ###### MAKE SURE DS is None, if you dont want to use it to prune SP.
        if params["datasetstrokes_extract_to_prune_stroke_and_get_features"] is None and params["substrokes_features_do_extraction"] is None:
            assert DS is None, "expect this, since the input should have been DS=None..."

        ####### PREPROCESSING FOR DS THAT SHOULD ALWAYS RUN:
        if DS is not None:
            # append Tkbeh_stktask
            DS.tokens_append(ver="beh_using_task_data")

        # Shape sequence for each trial:
        D.seqcontext_extract_shapes_in_beh_order_append_column()

        return D, DS, params


def conjunctions_print_plot_all(ListD, SAVEDIR, ANALY_VER, which_level="trial"):
    """
    Wrapper for all printing and plotting (saving) related to conjuicntions of varaibles that matter for PIG>
    Think of these as the conjucntiosn that care about for neural analysis. Here help assess each beahvior quickly.
    PARAMS:
    - ListD, lsit of Datasets
    - ANALY_VER, string code for which analysis. e.g., ANALY_VER = "singleprim"
    - which_level, string, what data? e.g. {"stroke", "trial"}
    RETURNS:
    - saves figures in f"{SAVEDIR}/conjunctions"
    """
    # from neuralmonkey.metadat.analy.anova_params import dataset_apply_params, params_getter_plots
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from pythonlib.tools.plottools import savefig

    sdir = f"{SAVEDIR}/conjunctions"
    os.makedirs(sdir, exist_ok=True)

    # ListD = [D]
    list_animals = []
    list_dates = []
    for D in ListD:
        assert len(D.animals())==1
        animal = D.animals()[0]
        _dates = D.Dat["date"].unique()
        assert len(_dates)==1
        DATE = int(_dates[0])

        list_animals.append(animal)
        list_dates.append(DATE)
    assert len(set(list_animals))==1
    assert len(set(list_dates))==1
    animal = list_animals[0]
    DATE = list_dates[0]

    from pythonlib.dataset.analy_dlist import concatDatasets
    D = concatDatasets(ListD)

    ### Prep dataset, and extract params
    assert False, "fix this call to dataset_apply_params"
    Dpruned, DSprun, params = dataset_apply_params(ListD, DS, animal, DATE, which_level, ANALY_VER)
    # _, Dpruned, TRIALCODES_KEEP, params, params_extraction = dataset_apply_params_OLD(ListD,
    #                                                                                   animal, DATE, which_level, ANALY_VER)
    assert len(Dpruned.Dat)>0
    
    ### Print and plot all conjucntions
    LIST_VAR = params["LIST_VAR"]
    LIST_VARS_CONJUNCTION = params["LIST_VARS_CONJUNCTION"]         

    _conjunctions_print_plot_all(Dpruned.Dat, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, 
        params["globals_nmin"], Dpruned)  
    # list_n = []
    # for var, vars_others in zip(LIST_VAR, LIST_VARS_CONJUNCTION):
        
    #     print(var, "vs", vars_others)
        
    #     # All data
    #     path = f"{sdir}/{var}|vs|{'-'.join(vars_others)}.txt"
    #     plot_counts_heatmap_savedir = f"{sdir}/heatmap-{var}|vs|{'-'.join(vars_others)}.pdf"
    #     Dpruned.grouping_conjunctions_print_variables_save(var, vars_others, path, n_min=0, 
    #                                                       plot_counts_heatmap_savedir=plot_counts_heatmap_savedir)
    #     # Passing nmin
    #     path = f"{sdir}/goodPassNmin-{var}|vs|{'-'.join(vars_others)}.txt"
    #     plot_counts_heatmap_savedir = f"{sdir}/goodPassNmin-heatmap-{var}|vs|{'-'.join(vars_others)}.pdf"
    #     dfout, dict_dfs = Dpruned.grouping_conjunctions_print_variables_save(var, vars_others, path, n_min=params["globals_nmin"], 
    #                                                       plot_counts_heatmap_savedir=plot_counts_heatmap_savedir)
    #     plt.close("all")
        
    #     # Count
    #     list_n.append(len(dict_dfs))
        
    # ### Print summary across conjucntions
    # strings = []
    # strings.append("n good levels of othervar | var |vs| othervars")
    # for var, vars_others, n in zip(LIST_VAR, LIST_VARS_CONJUNCTION, list_n):
    #     s = f"{n} -- {var}|vs|{'-'.join(vars_others)}"
    #     strings.append(s)
    # path = f"{sdir}/summary_n_levels_of_othervar_with_min_data.txt"
    # writeStringsToFile(path, strings)  

    ### STROKE LEVEL - heatmaps of (shape, location) vs. index
    from pythonlib.dataset.dataset_strokes import DatStrokes
    DS = DatStrokes(Dpruned)
    for task_kind in ["prims_single", "prims_on_grid"]:
        dfthis = DS.Dat[DS.Dat["task_kind"]==task_kind]
        
        if len(dfthis)>0:
            fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", vars_others=["stroke_index"])
            path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc-task_kind_{task_kind}.pdf"
            savefig(fig, path)

            # Dissociate stroke index from remaining num strokes.
            fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="stroke_index", 
                                                              var2="stroke_index_fromlast", vars_others=["shape", "gridloc"])
            path = f"{sdir}/STROKELEVEL-conjunctions_stroke_index-task_kind_{task_kind}.pdf"
            savefig(fig, path)

            plt.close("all")

def _conjunctions_print_plot_all(DF, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, 
        N_MIN, Dplotter):
    """ Low-level code to plot and print info for var conjucntions
    PARAMS:
    - LIST_VAR, list of str
    - LIST_VARS_CONJUNCTION, list of list of str, len outer list matches LIST_VAR,
    - N_MIN, min n data for a given level of var, for defining that as "enough" or "good" data.
    - Dplotter, dataset, for plotting methods, won't use its data.
    """

    os.makedirs(sdir, exist_ok=True)
    # N_MIN = params["globals_nmin"]

    list_n = []
    for var, vars_others in zip(LIST_VAR, LIST_VARS_CONJUNCTION):

        print(var, "vs", vars_others)

        # All data
        path = f"{sdir}/{var}|vs|{'-'.join(vars_others)}.txt"
        plot_counts_heatmap_savedir = f"{sdir}/heatmap-{var}|vs|{'-'.join(vars_others)}.pdf"
        Dplotter.grouping_conjunctions_print_variables_save(var, vars_others, path, n_min=0, 
                                                          plot_counts_heatmap_savedir=plot_counts_heatmap_savedir,
                                                          DF = DF)
        

        # Passing nmin
        path = f"{sdir}/goodPassNmin-{var}|vs|{'-'.join(vars_others)}.txt"
        plot_counts_heatmap_savedir = f"{sdir}/goodPassNmin-heatmap-{var}|vs|{'-'.join(vars_others)}.pdf"
        dfout, dict_dfs = Dplotter.grouping_conjunctions_print_variables_save(var, vars_others, path, n_min=N_MIN, 
                                                          plot_counts_heatmap_savedir=plot_counts_heatmap_savedir,
                                                            DF = DF)
        plt.close("all")

        # Count
        list_n.append(len(dict_dfs))

    ### Print summary across conjucntions
    strings = []
    strings.append("n good levels of othervar | var |vs| othervars")
    for var, vars_others, n in zip(LIST_VAR, LIST_VARS_CONJUNCTION, list_n):
        s = f"{n} -- {var}|vs|{'-'.join(vars_others)}"
        strings.append(s)
    path = f"{sdir}/summary_n_levels_of_othervar_with_min_data.txt"
    writeStringsToFile(path, strings)  

    if False:
        #### IN PROGRESS - NOT NEEDED. This should be outside reall,y and implemented
        # by passing in strokes dataset.
        ### Two variables each subplot. Subplots a third variable.
        fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", vars_others=["stroke_index"])
        path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc-task_kind_{task_kind}.pdf"
        savefig(fig, path)


        ####### OLD
        for task_kind in ["prims_single", "prims_on_grid"]:
            dfthis = DS.Dat[DS.Dat["task_kind"]==task_kind]

            if len(dfthis)>0:
                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", vars_others=["stroke_index"])
                path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc-task_kind_{task_kind}.pdf"
                savefig(fig, path)

                # Dissociate stroke index from remaining num strokes.
                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="stroke_index", 
                                                                  var2="stroke_index_fromlast", vars_others=["shape", "gridloc"])
                path = f"{sdir}/STROKELEVEL-conjunctions_stroke_index-task_kind_{task_kind}.pdf"
                savefig(fig, path)

                plt.close("all")


def params_getter_dataset_preprocess(ANALY_VER, animal, DATE):
    """ [GOOD] Extract params just for preprocessing Dataset and/or
    DatasetStrokes, depending on ANALY_VER.
    Should be useful across neural and beh.
    """

    ############### Reassign name to "taskgroup", siimplifying things, especialyl good
    # for grid sequence tasks with different probes, want to merge them for larger N.
    if ANALY_VER in ["ruleswERROR", "rulesw", "ruleswALLDATA"]:
        taskgroup_reassign_simple_neural = True
    elif ANALY_VER in ["rulesingle"]:
        # Skip for now, since it takes long time and not necessary, but in future maybe move to True
        taskgroup_reassign_simple_neural = False
    else:
        taskgroup_reassign_simple_neural = False

    ################ BEH DATASET PREPROCESS STEPS
    # THESE ARE ONLY used for deciding which trials ot keep.
    if ANALY_VER in ["rulesw"]:
        preprocess_steps_append = ["correct_sequencing_binary_score",
            "one_to_one_beh_task_strokes_allow_unfinished"]
    elif ANALY_VER in ["rulesingle"]:
        # preprocess_steps_append = ["correct_sequencing_binary_score",
        #     "one_to_one_beh_task_strokes"]
        preprocess_steps_append = ["correct_sequencing_binary_score",
            "one_to_one_beh_task_strokes_allow_unfinished"]
    elif ANALY_VER in ["ruleswALLDATA"]:
        # keep all trials
        preprocess_steps_append = []
    elif ANALY_VER in ["ruleswERROR"]:
        # error trials
        preprocess_steps_append = ["wrong_sequencing_binary_score"]
    elif ANALY_VER in ["seqcontext", "singleprimvar", "seqcontextvar", "PIG_BASE"]:
        preprocess_steps_append = ["one_to_one_beh_task_strokes_allow_unfinished"]
    elif ANALY_VER in ["singleprim"]:
        preprocess_steps_append = ["one_to_one_beh_task_strokes", "remove_online_abort"]
    elif ANALY_VER in ["charstrokes", "chartrial"]:
        # Dont carea bout match btw beh and task strokes
        preprocess_steps_append = ["remove_online_abort"]
    elif ANALY_VER in ["substrokes_sp"]:
        # Dont do much, since already pruend in extraction of substrokes,a nd these
        # wont be accurate anyway
        preprocess_steps_append = ["remove_online_abort"]
    else:
        print(ANALY_VER)
        assert False
    preprocess_steps_append.append("beh_strokes_at_least_one")
    if False:
        preprocess_steps_append.append("remove_repeated_trials")

    ################ CHARACTER CLUSTER LABELS
    # if ANALY_VER in ["charstrokes", "chartrial"]:
    #     # Then, in D, extract shape labels for each stroke (in "character" task_kind only)
    #     charclust_dataset_extract_shapes = True
    # else:
    #     charclust_dataset_extract_shapes = False

    # Should always be false, since now loading D automatically replaces shapes.
    # INSTEAD, now re-extract Snippets whenever you change the clust labels.
    # OTherwise is very difficult to update Datset after laoding from Snippets (Session
    # class holds it)...
    charclust_dataset_extract_shapes = False

    ################## SUPERVISION LEVELS TO KEEP
    ################# OPTIONALLY KEEP ONLY SPECIFIC "FULL" SUPERVISION NAMES
    if animal=="Pancho" and DATE==220921:
        # This day has "randomseq-sequence mask"... (rare)
        # Therefore want to keep sequence mask
        # (An epoch is random sequence using sequence mask)
        assert False, "check this days supervision_stage_new, make sure the 1111 is correct."
        list_superv_keep_full = ["mask|0.5|0|1111||0|none",
                                  "off|1.0|0|1111||0|none",
                                  "off|1.0|0|1111}|0|none"]
        list_superv_keep = "all" # None means just keep all cases that are "no supervision"
    else:
        # Prune so is just "no supervision" (i.e., list_superv_keep is None)
        list_superv_keep_full = None
        list_superv_keep = None # None means just keep all cases that are "no supervision"


    ############ rules will generally need to use this.
    DO_EXTRACT_EPOCHKIND = False
    if "rule" in ANALY_VER:
        # Label each trial based on its conjunction of character and correct beh sequence.
        DO_CHARSEQ_VER = "task_matlab"
        # EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "char_seq"
        EXTRACT_EPOCHSETS_n_max_epochs = 3
        EXTRACT_EPOCHSETS_merge_sets = True
        DO_EXTRACT_EPOCHKIND = True
    elif ANALY_VER in ["singleprimvar"]:
        # Label each trial based on its (shape/loc).
        DO_CHARSEQ_VER = None
        # EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "seqc_0_loc_shape"
        EXTRACT_EPOCHSETS_n_max_epochs = 10 # make this higher, since these are usually clean expts.
        EXTRACT_EPOCHSETS_merge_sets = True
    elif ANALY_VER in ["seqcontextvar"]:
        # Label each trial based on its (task config).
        DO_CHARSEQ_VER = None
        # EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "taskconfig_shploc"
        EXTRACT_EPOCHSETS_n_max_epochs = 10 # make this higher, since these are usually clean expts.
        EXTRACT_EPOCHSETS_merge_sets = True
    elif ANALY_VER in ["singleprim", "seqcontext", "charstrokes", "chartrial", "substrokes_sp", "PIG_BASE"]:
        # DO_CHARSEQ_VER = None
        # EXTRACT_EPOCHSETS = False
        # EXTRACT_EPOCHSETS_trial_label = None
        # EXTRACT_EPOCHSETS_n_max_epochs = None
        # EXTRACT_EPOCHSETS_merge_sets = None
        DO_CHARSEQ_VER = None
        # EXTRACT_EPOCHSETS = False
        EXTRACT_EPOCHSETS_trial_label = None
        EXTRACT_EPOCHSETS_n_max_epochs = None
        EXTRACT_EPOCHSETS_merge_sets = False
    else:
        print(ANALY_VER)
        assert False

    ########################
    # DO_EXTRACT_TASKCONFIG = False
    # if "taskconfig_loc" in LIST_VAR_ALL or "taskconfig_shp" in LIST_VAR_ALL or "taskconfig_shploc" in LIST_VAR_ALL:
    #     DO_EXTRACT_TASKCONFIG = True
    # if ANALY_VER=="seqcontextvar":
    #     DO_EXTRACT_TASKCONFIG = True

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
            (["rndstr", "llV1a|1", "L|1"], "rank|1"),
            (["rndstr", "llV1b|1", "L|1"], "rank|1"),
            (["rndstr", "llV1c|1", "L|1"], "rank|1"),
            (["rndstr", "llV1d|1", "L|1"], "rank|1"),
        ]
        epoch_merge_key = "epoch"
    elif animal =="Pancho" and DATE in [220921]:
        # Sequence mask supervision, i..e, an old version before
        # designed the color supervision mask (so it is rare). ie..
        # in single epoch mixing random + structured sequence. merge those since
        # the subject doesnt know.
        assert False, "include the color strokes binary in the supev string - check this day for how to type it."
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

    ########## WHETHER TO USE DATASETSTROKES to help prune
    # see dataset_strokes.preprocess_dataset_to_datstrokes
    # ONLY DO THIS If you want to use info about pruned strokes to
    # then prune entire trials....
    if ANALY_VER in ["substrokes_sp"]:
        # Ignore, since already pruned at substrokes level.
        datasetstrokes_extract_to_prune_trial = None
        datasetstrokes_extract_to_prune_stroke_and_get_features = None
    elif ANALY_VER in ["singleprim", "singleprimvar"]:
        # Single prim -- most stringent
        datasetstrokes_extract_to_prune_trial = "singleprim"
        datasetstrokes_extract_to_prune_stroke_and_get_features = None
    elif ANALY_VER in ["seqcontext", "PIG_BASE"]:
        # PIG.
        # Skip this, since you dont want to remove entire trial. This
        # Will be run automatically with snippet extraction.
        datasetstrokes_extract_to_prune_trial = None
        datasetstrokes_extract_to_prune_stroke_and_get_features = "clean_one_to_one"
    elif ANALY_VER in ["charstrokes", "chartrial"]:
        # Character strokes.
        # Shape labels must be saved and loadable (e..g, from clustering).
        datasetstrokes_extract_to_prune_trial = None
        # datasetstrokes_extract_to_prune_stroke_and_get_features = "clean_chars_load_clusters" # This would remove all strokes that are not "character" tasks.
        # datasetstrokes_extract_to_prune_stroke_and_get_features = "chars_load_clusters" #
        # datasetstrokes_extract_to_prune_stroke_and_get_features = "clean_chars" # now defualt is to preload in D
        datasetstrokes_extract_to_prune_stroke_and_get_features = "clean_chars_clusters_without_reloading" # Generates
        # DS anew from D, and uses the clust scores within D to prune strokes.

    elif "rule" in ANALY_VER:
        # Rule grid tasks, ignore stroke qualtiy
        datasetstrokes_extract_to_prune_trial = None
        datasetstrokes_extract_to_prune_stroke_and_get_features = None
    else:
        print(ANALY_VER)
        assert False

    ############ SUBSTROKES, WHETHER TO EXTRACT MOTOR FEATURES
    if ANALY_VER in ["substrokes_sp"]:
        substrokes_features_do_extraction = True
    elif ANALY_VER in ["singleprim", "singleprimvar", "seqcontext",
                       "charstrokes", "chartrial", "PIG_BASE",
                       "rulesingle", "rulesw"]:
        substrokes_features_do_extraction = False
    else:
        print(ANALY_VER)
        assert False

    ######## CHUNKS, e.g., AnBm, wherther to extract state, e.g.,, n in chunk, and so on.
    if "rule" in ANALY_VER:
        # This operates on DS only.
        datasetstrokes_extract_chunks_variables = True
    else:
        datasetstrokes_extract_chunks_variables = False


    params = {
        "DO_CHARSEQ_VER":DO_CHARSEQ_VER,
        # "EXTRACT_EPOCHSETS":EXTRACT_EPOCHSETS,
        "EXTRACT_EPOCHSETS_trial_label":EXTRACT_EPOCHSETS_trial_label,
        "EXTRACT_EPOCHSETS_n_max_epochs":EXTRACT_EPOCHSETS_n_max_epochs,
        "EXTRACT_EPOCHSETS_merge_sets":EXTRACT_EPOCHSETS_merge_sets,
        "taskgroup_reassign_simple_neural":taskgroup_reassign_simple_neural,
        "preprocess_steps_append":preprocess_steps_append,
        "remove_aborts":False, # Never remove. Use preprocessing isntead.
        "list_superv_keep":list_superv_keep,
        "list_superv_keep_full":list_superv_keep_full,
        "DO_SCORE_SEQUENCE_VER":_params_score_sequence_ver(animal, DATE, ANALY_VER),
        "list_epoch_merge":list_epoch_merge,
        "epoch_merge_key":epoch_merge_key,
        "DO_EXTRACT_EPOCHKIND":DO_EXTRACT_EPOCHKIND,
        "datasetstrokes_extract_to_prune_trial":datasetstrokes_extract_to_prune_trial,
        "datasetstrokes_extract_to_prune_stroke_and_get_features":datasetstrokes_extract_to_prune_stroke_and_get_features,
        "substrokes_features_do_extraction":substrokes_features_do_extraction,
        "charclust_dataset_extract_shapes":charclust_dataset_extract_shapes,
        "datasetstrokes_extract_chunks_variables":datasetstrokes_extract_chunks_variables
    }

    return params


def params_getter_raster_vars(which_level, question, OVERWRITE_lenient_n=2):
    """
    Params for rasters, focusing on the minimal plots for getting across qusritons. Mniamrl since takes a logn time to plot.
    :param which_level:
    :param ANALY_VER:
    :return:
    """

    print("****", which_level, question)
    LIST_OVERWRITE_lenient_n = None

    if which_level=="trial" and "CHAR_BASE_trial" in question:
        # Often not enough data for each shape x loc for single prims...
        ################## - PIG (trial, decode first stroke)
        # LIST_VAR = [
        #     "seqc_0_shape",
        # ]
        # LIST_VARS_OTHERS = [
        #     ["task_kind", "seqc_0_loc"],
        # ]
        # - 3/12/24 - Parse (vlPFC?)
        LIST_VAR = [
            "seqc_1_shape",
            "seqc_1_shape",
            "taskconfig_shp_SHSEM",
            "taskconfig_shp_SHSEM",
        ]
        LIST_VARS_OTHERS = [
            ["character", "task_kind"], # control for image.
            ["seqc_0_shape", "seqc_0_loc", "task_kind"],
            ["taskconfig_loc", "seqc_0_shape", "seqc_0_loc", "task_kind"], # control for first action.
            ["character"],
        ]
    elif which_level=="stroke" and question in ["PIG_BASE_stroke"]:

        ################## - PIG decode online context [3/12/24]
        LIST_VAR = [
            "CTXT_loc_next",
            "CTXT_shape_next",
            "task_kind",
            # "stroke_index_fromlast_tskstks",
            "stroke_index",
            "FEAT_num_strokes_task",
            "shape",
        ]
        # LIST_VARS_OTHERS = [
        #     ["task_kind", "CTXT_loc_prev", "shape", "gridloc", "CTXT_shape_next"],
        #     ["task_kind", "CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next"],
        #     ["shape", "gridloc", "CTXT_loc_prev"],
        #     ["task_kind", "FEAT_num_strokes_task", "CTXT_loc_prev", "shape", "gridloc"],
        #     ["task_kind", "FEAT_num_strokes_task", "CTXT_loc_prev", "shape", "gridloc"],
        #     ["CTXT_loc_prev", "shape", "gridloc"],
        #     ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["task_kind", "CTXT_shape_prev", "CTXT_loc_prev", "shape", "gridloc", "CTXT_shape_next"],
            ["task_kind", "CTXT_shape_prev", "CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next"],
            ["shape", "gridloc", "CTXT_shape_prev", "CTXT_loc_prev"],
            # ["task_kind", "FEAT_num_strokes_task", "CTXT_shape_prev", "CTXT_loc_prev", "shape", "gridloc"],
            ["task_kind", "FEAT_num_strokes_task", "CTXT_shape_prev", "CTXT_loc_prev", "shape", "gridloc"],
            ["CTXT_shape_prev", "CTXT_loc_prev", "shape", "gridloc"],
            ["task_kind", "FEAT_num_strokes_task", "stroke_index_fromlast_tskstks"],
            ]

    elif which_level == "stroke" and question in ["RULE_BASE_stroke"]:
        # "gridloc", # subplots = indices within chunk

        LIST_VAR = [
            ("chunk_rank", "chunk_within_rank_semantic"), # strongest test of index within chunk, and chunk index.
            # "chunk_within_rank_semantic", # Good, but slightly redudant with ("chunk_rank", "chunk_within_rank_semantic"). Consider activating

            # Encoding of shape even during gap?
            "CTXT_shape_prev", # var = 2-motifs, conditioned on the shape of 2nd stroke.

            # -- N in chunk (analysis)
            "chunk_n_in_chunk", # n at start or end of chunk [general across n_in_chunk]


            # These dont need to do chunk_within_rank_fromlast, since it would be very similar to this (chunk_within_rank)
            "chunk_within_rank_semantic", #

            # -- N in chunk (visualtions)
            "chunk_n_in_chunk", # subplots = indices within chunk (color by n in chunk)
            "chunk_n_in_chunk", # subplots = indices within chunk (color by n in chunk)
        ]

        # ["chunk_rank", "chunk_within_rank", "chunk_within_rank_fromlast"], # NEW - effect of onset
        LIST_VARS_OTHERS = [
            # This, checked closely by hand, as the minimum for getting very clean M1
            ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next"],

            ("shape", "CTXT_loc_prev", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next", "CTXT_shape_next"), # Good -- tight control!

            # GOOD - 2 variations, 2nd is more controlled.
            ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust", "CTXT_loc_next"],
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "chunk_within_rank_semantic", "shape", "loc_on_clust", "CTXT_loc_next", "CTXT_shape_prev"],

            # GOOD - 4 variations, each with stronger control.
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust"]
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next"]
            ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next", "CTXT_shape_prev"],
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev", "shape", "loc_on_clust", "loc_off_clust", "CTXT_loconclust_next", "CTXT_shape_next"]

            ("chunk_rank", "shape", "chunk_within_rank"), # [vislauzations] NEW: coloring by n in chunk --> consistent across those?
            ("chunk_rank", "shape", "chunk_within_rank_fromlast"), # same, but align to last stroke in chunk.
        ]

        # For some, should allow even if just 1 class.
        LIST_OVERWRITE_lenient_n = [OVERWRITE_lenient_n for _ in range(len(LIST_VARS_OTHERS))]
        LIST_OVERWRITE_lenient_n[4] = 1
        LIST_OVERWRITE_lenient_n[5] = 1

    elif which_level == "stroke" and question in ["RULEVSCOL_BASE_stroke"]:
        # Switchign between two rules.

        # Use all of those that exist for single rule, but addition a conjucntion on epoch
        LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n = params_getter_raster_vars("stroke",
                                                                                            "RULE_BASE_stroke")
        # LIST_VARS_OTHERS = [list(var_others) + ["epoch"] for var_others in LIST_VARS_OTHERS]
        LIST_VARS_OTHERS = [list(var_others) + ["superv_COLOR_METHOD"] for var_others in LIST_VARS_OTHERS]

        # Add those that use "epoch" as effect
        LIST_VAR.append("epoch")
        LIST_VARS_OTHERS.append(["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev",
                                 "shape", "loc_on_clust", "CTXT_shape_prev"])

        LIST_VAR.append("epoch")
        LIST_VARS_OTHERS.append(["stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev",
                                 "shape", "loc_on_clust", "CTXT_shape_prev", "CTXT_loconclust_next", "chunk_rank",
                                 "chunk_within_rank_semantic"])

    elif which_level == "stroke" and question in ["RULESW_BASE_stroke"]:
        # Switchign between two rules.

        # Use all of those that exist for single rule, but addition a conjucntion on epoch
        LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n = params_getter_raster_vars("stroke",
                                                                                            "RULE_BASE_stroke")
        LIST_VARS_OTHERS = [list(var_others) + ["epoch"] for var_others in LIST_VARS_OTHERS]

        # Add those that use "epoch" as effect
        LIST_VAR.append("epoch")
        LIST_VARS_OTHERS.append(["epochset", "stroke_index_is_first", "stroke_index_is_last_tskstks", "CTXT_locoffclust_prev",
                                 "shape", "loc_on_clust", "CTXT_shape_prev", "CTXT_loconclust_next", "chunk_rank",
                                 "chunk_within_rank_semantic"])

    elif which_level=="trial" and question in ["RULESW_BASE_trial"]:
        LIST_VAR = [
            "epoch",
            # "epoch", # abstract rule?
            # "character", # visually repsonsive?
            # "seqc_0_loc",
            # "seqc_0_shape",
            "epoch",
            # "seqc_1_loc_shape",
            # "seqc_1_loc_shape",
        ]
        LIST_VARS_OTHERS = [
            ["epochset"],
            # ["epochset", "character"],
            # ["epoch", "epochset"],
            # ["epoch", "seqc_0_shape"],
            # ["epoch", "seqc_0_loc"],
            ["seqc_0_loc", "seqc_0_shape", "seqc_nstrokes_beh"],
            # ["epoch"],
            # ["epoch", "seqc_0_loc_shape"],
        ]

    else:
        print(which_level)
        print(question)
        assert False

    if LIST_OVERWRITE_lenient_n is None:
        LIST_OVERWRITE_lenient_n = [OVERWRITE_lenient_n for _ in range(len(LIST_VARS_OTHERS))]

    assert len(LIST_VAR)==len(LIST_VARS_OTHERS)

    # Convert from tuple to list
    LIST_VARS_OTHERS = [list(var_others) for var_others in LIST_VARS_OTHERS]
    LIST_VAR = [list(var) if isinstance(var, tuple) else var for var in LIST_VAR]

    return LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n

def params_getter_decode_vars(which_level):
    """
    Holds variables for decoding and other popujation-level plots, especialyl relating var and conjunction vars,
    both for cross- and within-decoding.
    Currently used for both DECODE and UMAP analyses.
    """

    LIST_VAR_DECODE = []
    LIST_VARS_CONJ = []
    LIST_SEPARATE_BY_TASK_KIND = []
    LIST_FILTDICT = []
    LIST_SUFFIX = []

    if which_level=="trial":
        ########### Action (first stroke)
        suffix = "action_first"
        list_var_decode = [
            "seqc_0_shape",
            "seqc_0_shape",
            # "seqc_0_locon_bin_in_loc",
            # "gridsize",
            # "seqc_0_locon_binned",
            # "seqc_0_shape",
            "seqc_0_shapesemcat", # Usually does nothing different.
            "shape_is_novel_all"
        ]
        list_vars_conj = [
            ["seqc_0_loc", "seqc_0_loc_on_clust", "gridsize", "task_kind"], # ["seqc_0_center_binned", "gridsize", "task_kind"],
            ["taskconfig_shp_SHSEM", "seqc_0_loc", "seqc_0_loc_on_clust", "gridsize", "task_kind"],
            # ["seqc_0_shape", "gridsize", "task_kind"],
            ["seqc_0_shape", "seqc_0_loc", "task_kind"],
            # ["seqc_0_loc", "gridsize", "task_kind"],
            # ["seqc_0_center_binned", "gridsize", "task_kind"],
            ["seqc_0_loc", "task_kind"],
            ["seqc_0_loc", "task_kind"],
            ]
        separate_by_task_kind = True
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        ########### Action (sequence)
        suffix = "action_seq"
        list_var_decode = [
            "seqc_1_shape",
            "seqc_1_loc",
            "seqc_nstrokes_beh",
            "seqc_1_shape",
            "seqc_1_loc",
            "seqc_nstrokes_beh",
            "task_kind"
        ]
        list_vars_conj = [
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_loc", "task_kind"],
            ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "task_kind"],
            ["seqc_0_shape", "seqc_0_loc", "task_kind"],
            ["taskconfig_shp_SHSEM", "seqc_0_shape", "seqc_0_loc", "seqc_1_loc", "task_kind"],
            ["taskconfig_shp_SHSEM", "seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "task_kind"],
            ["taskconfig_shp_SHSEM", "seqc_0_shape", "seqc_0_loc", "task_kind"],
            ["seqc_0_shape", "seqc_0_loc"],
            ]
        separate_by_task_kind = True
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)

        ########## PARSE
        suffix = "parse"
        list_var_decode = [
            "taskconfig_shp_SHSEM",
            "taskconfig_shp_SHSEM",
            "taskconfig_shp_SHSEM",
            "taskconfig_shp_SHSEM",
            # "taskconfig_shploc_SHSEM",
            # "taskconfig_shploc_SHSEM",
            # "taskconfig_shploc_SHSEM",
            "seqc_1_shape",
            "seqc_2_shape",
            "seqc_3_shape",
            "seqc_1_shape",
        ]
        list_vars_conj = [
            ["taskconfig_loc", "task_kind"], # minimal control
            ["taskconfig_loc", "seqc_0_shape", "seqc_0_loc", "task_kind"], # control for first action.
            ["seqc_0_shape", "seqc_0_loc", "task_kind"], # control for first action.
            ["character", "task_kind"], # control for image.
            # ["taskconfig_loc", "task_kind"], # minimal control
            # ["taskconfig_loc", "seqc_0_shape", "seqc_0_loc", "task_kind"], # control for first action.
            # ["taskconfig_loc", "character", "task_kind"], # control for image.
            ["seqc_0_shape", "seqc_0_loc", "task_kind"],
            ["seqc_0_shape", "seqc_0_loc", "task_kind"],
            ["seqc_0_shape", "seqc_0_loc", "task_kind"],
            ["character", "task_kind"], # control for image.
            ]
        separate_by_task_kind = True
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        ##### Same image, diff action
        # Same image --> diff sequence
        # list_var_color_var_subplot.append(["seqc_0_shapeloc", ("character", "task_kind")])
        # list_var_color_var_subplot.append(["FEAT_num_strokes_beh", ("character", "task_kind")])
        # list_var_color_var_subplot.append(["FEAT_num_strokes_beh", "task_kind"])
        #
        ### SAME IMAGe, diff parse.
    elif which_level=="stroke":
        ######### SEQUENCE
        ## SEQUENCE PREDICTION

        # Context
        # NOTE: using CTXT_loc_prev isntad fo CTXT_shapeloc_prev since the latter is too constarined --> not muich data...
        suffix = "context"
        list_var_decode = [
            "CTXT_loc_next",
            "CTXT_loc_next",
            "CTXT_loc_next",
            # "CTXT_shapeloc_next",
            "CTXT_shape_next",
            "CTXT_shape_next",
            # "CTXT_loc_next",
        ]
        list_vars_conj = [
            ["shape", "gridloc", "task_kind"],
            ["shape", "gridloc", "gridloc_within", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "CTXT_shape_next", "task_kind"],
            # ["CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            ["shape", "gridloc", "CTXT_loc_next", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next", "task_kind"],
            # ["CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shape_next", "task_kind"],
            ]
        separate_by_task_kind = True
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)

        # Context
        suffix = "context_excludefirst"
        list_var_decode = [
            "CTXT_loc_next",
            "CTXT_loc_next",
            "CTXT_loc_next",
            # "CTXT_shapeloc_next",
            "CTXT_shape_next",
            "CTXT_shape_next",
            # "CTXT_loc_next",
        ]
        list_vars_conj = [
            ["shape", "gridloc", "task_kind"],
            ["shape", "gridloc", "gridloc_within", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "CTXT_shape_next", "task_kind"],
            # ["CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            ["shape", "gridloc", "CTXT_loc_next", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next", "task_kind"],
            # ["CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shape_next", "task_kind"],
            ]

        separate_by_task_kind = True
        filtdict = {
            "stroke_index":[1,2,3,4,5,6,7,8],
        }
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        # Task kind
        suffix = "taskkind"
        list_var_decode = [
            "task_kind",
            "task_kind"
        ]
        list_vars_conj = [
            ["shape", "gridloc", "CTXT_shapeloc_prev"],
            ["shape", "gridloc", "gridloc_within", "CTXT_shapeloc_prev"],
            ]
        separate_by_task_kind = False
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        # Shape, invariant to context
        suffix = "shape"
        list_var_decode = [
            "shape",
            "shape",
            "shape",
            "shape",
        ]
        list_vars_conj = [
            ["CTXT_loc_prev", "gridloc", "CTXT_loc_next", "gridsize", "task_kind"],
            ["CTXT_loc_prev", "gridloc", "gridloc_within", "gridsize", "CTXT_loc_next", "task_kind"],
            ["CTXT_loc_prev", "gridloc", "gridloc_within", "gridsize","task_kind"],
            ["gridloc", "gridloc_within", "gridsize","task_kind"],
            ["gridloc", "gridsize", "task_kind"],
            ]
        separate_by_task_kind = True
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        # Location, invariant to context
        suffix = "location"
        list_var_decode = [
            "gridloc",
            # "gridloc",
            # "gridloc",
        ]
        # list_vars_conj = [
        #     # ["gap_from_prev_angle_binned", "shape", "task_kind"],
        #     ["gap_to_next_angle_binned", "shape", "task_kind"],
        #     # ["CTXT_shapeloc_prev", "shape", "CTXT_shapeloc_next", "task_kind"],
        #     # ["CTXT_loc_prev", "shape", "CTXT_loc_next", "task_kind"],
        #     # ["CTXT_shapeloc_prev", "shape", "CTXT_loc_next", "task_kind"],
        #     ]
        list_vars_conj = [
            ["CTXT_loc_prev", "shape", "gridsize", "task_kind"],
            ]
        separate_by_task_kind = True
        filtdict = {
            "stroke_index":[1,2,3,4,5,6,7,8], # Avoid effect of reach direction at onset.
        }
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        if False:
            # Reach direction
            suffix = "reachnext"
            list_var_decode = [
                "gap_to_next_angle_binned",
            ]
            list_vars_conj = [
                ["CTXT_loc_prev", "shape", "gridloc", "gridloc_within", "task_kind"],
                ]
            separate_by_task_kind = True
            filtdict = {
                "stroke_index_fromlast_tskstks":[-7, -6, -5, -4, -3, -2]
            }
            # ------
            LIST_VAR_DECODE.append(list_var_decode)
            LIST_VARS_CONJ.append(list_vars_conj)
            LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
            LIST_FILTDICT.append(filtdict)
            LIST_SUFFIX.append(suffix)

            # Reach direction
            suffix = "reachprev"
            list_var_decode = [
                "gap_from_prev_angle_binned",
            ]
            list_vars_conj = [
                ["shape", "gridloc", "gridloc_within", "task_kind"],
                ]
            separate_by_task_kind = True
            filtdict = {
                "stroke_index":[1,2,3,4,5,6,7,8],
            }
            # ------
            LIST_VAR_DECODE.append(list_var_decode)
            LIST_VARS_CONJ.append(list_vars_conj)
            LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
            LIST_FILTDICT.append(filtdict)
            LIST_SUFFIX.append(suffix)

            # # Strong, test context vs. SI
            # list_var_decode = ["CTXT_ALL_MAX"]
            # list_vars_conj = [
            #     ["stroke_index_fromlast_tskstks", "task_kind"],
            #     ]
            #

        # Stroke index
        suffix = "strokeindex_exclude_first_stroke"
        list_var_decode = [
            # "stroke_index_fromlast_tskstks",
            # "stroke_index",
            "stroke_index_fromlast_tskstks",
            "stroke_index",
            "stroke_index_fromlast_tskstks",
            "stroke_index",
            "stroke_index",
            # "stroke_index_fromlast_tskstks",
            # "stroke_index",
        ]
        list_vars_conj = [
            # ["CTXT_shapeloc_prev", "shape", "gridloc", "task_kind"],
            # ["CTXT_shapeloc_prev", "shape", "gridloc", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            ["shape", "gridloc", "gridloc_within", "task_kind"],
            ["shape", "gridloc", "gridloc_within", "task_kind"],
            ["FEAT_num_strokes_task", "CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            # ["CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shapeloc_next", "task_kind"],
            # ["CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shapeloc_next", "task_kind"],
            ]
        separate_by_task_kind = True
        filtdict = {
            "stroke_index":[1,2,3,4,5,6,7,8],
        }
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


        # Stroke index
        suffix = "strokeindex"
        list_var_decode = [
            "stroke_index_fromlast_tskstks",
            "stroke_index",
            "stroke_index_fromlast_tskstks",
            "stroke_index",
        ]
        list_vars_conj = [
            ["CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            ["CTXT_loc_prev", "shape", "gridloc", "task_kind"],
            ["shape", "gridloc", "gridloc_within", "task_kind"],
            ["shape", "gridloc", "gridloc_within", "task_kind"],
            ]
        separate_by_task_kind = True
        filtdict = None
        # ------
        LIST_VAR_DECODE.append(list_var_decode)
        LIST_VARS_CONJ.append(list_vars_conj)
        LIST_SEPARATE_BY_TASK_KIND.append(separate_by_task_kind)
        LIST_FILTDICT.append(filtdict)
        LIST_SUFFIX.append(suffix)


    else:
        assert False

    return LIST_VAR_DECODE, LIST_VARS_CONJ, LIST_SEPARATE_BY_TASK_KIND, LIST_FILTDICT, LIST_SUFFIX

