""" Holds params for anava analysis for each date
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile

ONLY_ESSENTIAL_VARS = False # then just the first var, assuemd to be most important, for quick analys

##################
LIST_ANALYSES = ["rulesw", "rulesingle", "rulesingleALLDATA", "ruleswALLDATA", "ruleswERROR", "singleprimvar", "seqcontextvar",
                 "seqcontext", "singleprim", "charstrokes", "chartrial", "substrokes_sp", "PIG_BASE", "singleprim_psycho"] # repo of possible analses,

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
    if ANALY_VER in ["rulesw", "ruleswERROR", "ruleswALLDATA", "rulesingle", "rulesingleALLDATA"]:
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
    elif ANALY_VER in ["singleprimvar", "seqcontext", "singleprim", "singleprim_psycho", "seqcontextvar", "charstrokes", "chartrial", "substrokes_sp", "PIG_BASE"]:
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


def dataset_apply_params(D, DS, ANALY_VER, animal, DATE, save_substroke_preprocess_figures=True,
                         SKIP_PLOTS=False):
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
            assert False, "should always use parse version (right?)"
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
        # if "FEAT_num_strokes_task" not in D.Dat.columns:
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
        if False: # This is done in Snippets now.
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

            savedir_preprocess = D.make_savedir_for_analysis_figures_BETTER("preprocess_general")
            if params["datasetstrokes_extract_chunks_behseq_clusts"]:
                # For each sequence kind (e.g. shapes) split into concrete variations (classes).
                if not SKIP_PLOTS:
                    sdir = f"{savedir_preprocess}/seqcontext_behorder_cluster_concrete_variation"
                    os.makedirs(sdir, exist_ok=True)
                    D.seqcontext_behseq_cluster_concrete_variation(SAVEDIR=sdir,
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
                assert len(D.Dat.iloc[0]["syntax_concrete"])<=3, "should probably runt he code below. See this note:"
                # If sc is like (2,3,0), then  epochs arelady split into two epochs, with same representation of sc.
                # If sc is like (2,3,0,0) vs. (0, 0, 2,3), then need to run the code her:
                if False:
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
            if SKIP_PLOTS:
                PLOT = False
                sdir = None
            else:
                PLOT = True
                sdir = f"{savedir_preprocess}/grammarparses_chunk_transitions_gaps_extract_batch"
                os.makedirs(sdir, exist_ok=True)
            D.grammarparses_chunk_transitions_gaps_extract_batch(plot_savedir=sdir, PLOT=PLOT)
            plt.close("all")

            # For each token, assign a new key called "syntax role" -- good.
            D.grammarparses_syntax_role_append_to_tokens()

            # Epochsets, figure out if is same or diff motor beahvior (quick and dirty)
            D.grammarparses_syntax_epochset_quick_classify_same_diff_motor()

            # Define separate epochsets based on matching motor beh within each epoch_orig
            D.epochset_extract_matching_motor_wrapper()
            # D.grammarparses_rules_epochs_superv_summarize_wrapper(PRINT=True, include_epochset=True)
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

        # Rename so I remember that this is NOT the final DS. This is just used for pruning D
        DS_for_pruning_D = DS

        return D, DS_for_pruning_D, params


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
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars

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

    assert len(D.TokensStrokesBeh)>0, "sanity check, lost in concat?"

    ### Prep dataset, and extract params
    Dpruned, _, params = dataset_apply_params(D, None, ANALY_VER, animal, DATE)
    assert len(Dpruned.Dat)>0
    assert len(Dpruned.TokensStrokesBeh)>0
    
    ### Print and plot all conjucntions
    if False: # NOT WORKING -- need the params for which_level = trial
        LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars("PIG_BASE_stroke")
        # LIST_VAR = params["LIST_VAR"]
        # LIST_VARS_CONJUNCTION = params["LIST_VARS_CONJUNCTION"]         

        _conjunctions_print_plot_all(Dpruned.Dat, LIST_VAR, LIST_VARS_CONJUNCTION, sdir, 
            params["globals_nmin"], Dpruned)  

    ### STROKE LEVEL - heatmaps of (shape, location) vs. index
    from pythonlib.dataset.dataset_strokes import DatStrokes
    # D.sketchpad_fixation_append_as_string()
    DS = DatStrokes(Dpruned)
    # DS.dataset_append_column("epoch") 
    # DS.dataset_append_column("origin_string") 

    DS.dataset_append_column("block")
    list_block = DS.Dat["block"].unique().tolist()

    for task_kind in ["prims_single", "prims_on_grid"]:
        dfthis = DS.Dat[DS.Dat["task_kind"]==task_kind]
        
        if len(dfthis)>0:
            fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", vars_others=["stroke_index"])
            path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc-task_kind_{task_kind}.pdf"
            savefig(fig, path)

            # Dissociate stroke index from remaining num strokes.
            if len(dfthis["shape"].unique())<30:
                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="stroke_index", 
                                                                var2="stroke_index_fromlast", vars_others=["shape", "gridloc"])
                path = f"{sdir}/STROKELEVEL-conjunctions_stroke_index-task_kind_{task_kind}.pdf"
                savefig(fig, path)

            plt.close("all")
        
        # Also split by block
        for bk in list_block:
            dfthis = DS.Dat[(DS.Dat["task_kind"]==task_kind) & (DS.Dat["block"]==bk)]
            
            if len(dfthis)>10:
                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="shape", var2="gridloc", vars_others=["stroke_index"])
                path = f"{sdir}/STROKELEVEL-conjunctions_shape_gridloc-task_kind_{task_kind}-BLOCK_{bk}.pdf"
                savefig(fig, path)

                # Dissociate stroke index from remaining num strokes.
                if len(dfthis["shape"].unique())<30 and not task_kind=="prims_single":
                    fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, var1="stroke_index", 
                                                                    var2="stroke_index_fromlast", vars_others=["shape", "gridloc"])
                    path = f"{sdir}/STROKELEVEL-conjunctions_stroke_index-task_kind_{task_kind}-BLOCK_{bk}.pdf"
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
    elif ANALY_VER in ["rulesingle", "rulesingleALLDATA"]:
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
        preprocess_steps_append = ["correct_sequencing_binary_score",
            "one_to_one_beh_task_strokes_allow_unfinished"]
    elif ANALY_VER in ["rulesingleALLDATA"]:
        preprocess_steps_append = ["one_to_one_beh_task_strokes_allow_unfinished"]
    elif ANALY_VER in ["ruleswALLDATA"]:
        # keep all trials
        preprocess_steps_append = []
    elif ANALY_VER in ["ruleswERROR"]:
        # error trials
        preprocess_steps_append = ["wrong_sequencing_binary_score"]
    elif ANALY_VER in ["seqcontext", "singleprimvar", "seqcontextvar", "PIG_BASE"]:
        preprocess_steps_append = ["one_to_one_beh_task_strokes_allow_unfinished"]
    elif ANALY_VER in ["singleprim"]:
        preprocess_steps_append = ["beh_strokes_one", "remove_online_abort"]
    elif ANALY_VER in ["singleprim_psycho"]:
        preprocess_steps_append = []
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
    elif ANALY_VER in ["singleprim_psycho", "singleprim", "seqcontext", "charstrokes", "chartrial", "substrokes_sp", "PIG_BASE"]:
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
    elif ANALY_VER in ["singleprim_psycho"]:
        # Single prim -- most stringent
        # datasetstrokes_extract_to_prune_trial = "singleprim_psycho"
        datasetstrokes_extract_to_prune_trial = "singleprim_psycho_noabort"
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
                       "rulesingle", "rulesingleALLDATA", "rulesw", "singleprim_psycho"]:
        substrokes_features_do_extraction = False
    else:
        print(ANALY_VER)
        assert False

    ######## CHUNKS, e.g., AnBm, wherther to extract state, e.g.,, n in chunk, and so on.
    if "rule" in ANALY_VER:
        if "ALLDATA" in ANALY_VER or "ERROR" in ANALY_VER:
            # An exception, if you are collecting all data, then avoid things that will fail, because not correct number
            # of strokes.
            datasetstrokes_extract_chunks_variables = True
            datasetstrokes_extract_chunks_behseq_clusts = False
        else:
            # This operates on DS only.
            datasetstrokes_extract_chunks_variables = True
            datasetstrokes_extract_chunks_behseq_clusts = True
    else:
        datasetstrokes_extract_chunks_variables = False
        datasetstrokes_extract_chunks_behseq_clusts = False
    
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
        "datasetstrokes_extract_chunks_variables":datasetstrokes_extract_chunks_variables,
        "datasetstrokes_extract_chunks_behseq_clusts":datasetstrokes_extract_chunks_behseq_clusts,
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

    elif which_level == "stroke" and question in ["RULE_BASE_stroke", "RULE_ANBMCK_STROKE", "RULE_COLRANK_STROKE", "RULE_DIR_STROKE", "RULE_ROWCOL_STROKE"]:
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

    elif which_level == "stroke" and question in ["RULESW_ANBMCK_COLRANK_STROKE", "RULESW_ANY_SEQSUP_STROKE"]:
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

    elif which_level == "stroke" and question in ["RULESW_BASE_stroke", "RULESW_ANBMCK_DIR_STROKE",
                                                  "RULESW_ANBMCK_COLRANK_STROKE", "RULESW_ANBMCK_ABN_STROKE"]:
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

    elif which_level=="trial" and question in ["SP_BASE_trial"]:
        LIST_VAR = [
            "seqc_0_shape",
            "seqc_0_loc",
            "gridsize",
        ]
        LIST_VARS_OTHERS = [
            ["seqc_0_loc", "gridsize"],
            ["seqc_0_shape", "gridsize"],
            ["seqc_0_shape", "seqc_0_loc"],
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

def params_getter_umap_vars(question):
    """
    For visualization, so there are many... THese are like a superset of the vars in 
    params_getter_euclidian...
    :param question:
    :return:
    """
    # 3/20/24 - Made this for stroke AnBm Context. Should generalyl use this (over decode params, below?)
    list_var_color_var_subplot = []
    from neuralmonkey.metadat.analy.anova_params import params_getter_raster_vars
    LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n = params_getter_raster_vars(which_level, question)

    single_vars_done = []
    for var_decode, vars_conj in zip(LIST_VAR, LIST_VARS_OTHERS):

        if isinstance(var_decode, list):
            var_decode = tuple(var_decode)

        list_var_color_var_subplot.append([var_decode, tuple(vars_conj)])
        if var_decode not in single_vars_done:
            list_var_color_var_subplot.append([var_decode, "task_kind"])
            single_vars_done.append(var_decode)

    if question in ["RULE_BASE_stroke", "RULE_ANBMCK_STROKE", "RULE_COLRANK_STROKE", "RULE_DIR_STROKE", "RULE_ROWCOL_STROKE"]:
        # Sequence context, chunks
        # Usually AnBmCk
        list_var_color_var_subplot.append([("chunk_rank", "shape"), ("chunk_n_in_chunk", "task_kind")])
        list_var_color_var_subplot.append([("chunk_rank", "shape"), ("chunk_within_rank_semantic", "task_kind")])
        list_var_color_var_subplot.append([("chunk_rank", "shape"), ("chunk_within_rank_semantic", "chunk_n_in_chunk", "task_kind")]) # ** GOOD
        list_var_color_var_subplot.append(["chunk_within_rank_semantic", ("chunk_rank", "shape", "task_kind")])
        list_var_color_var_subplot.append(["chunk_within_rank_semantic", ("chunk_n_in_chunk", "task_kind")])
        list_var_color_var_subplot.append(["chunk_within_rank_semantic", ("chunk_rank", "shape", "chunk_n_in_chunk", "task_kind")]) # ** GOOD
        list_var_color_var_subplot.append(["chunk_n_in_chunk", ("chunk_within_rank_semantic", "task_kind")])
        list_var_color_var_subplot.append(["chunk_n_in_chunk", ("chunk_rank", "shape", "task_kind")])
        list_var_color_var_subplot.append(["chunk_n_in_chunk", ("chunk_rank", "shape", "chunk_within_rank_semantic", "task_kind")]) # ** GOOD

        # Each subplot is a syntax parse. Color by
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank"), ("taskcat_by_rule", "behseq_shapes", "task_kind")]) # ** GOOD
        list_var_color_var_subplot.append([("chunk_within_rank", "chunk_rank"), ("taskcat_by_rule", "behseq_shapes", "task_kind")])
        list_var_color_var_subplot.append([("shape", "chunk_within_rank"), ("taskcat_by_rule", "behseq_shapes", "task_kind")])
        list_var_color_var_subplot.append([("chunk_within_rank", "shape"), ("taskcat_by_rule", "behseq_shapes", "task_kind")])

        # Show that it is not trivially explained by location or shape
        list_var_color_var_subplot.append(["gridloc", "task_kind"])
        list_var_color_var_subplot.append(["gridloc", ("chunk_rank", "chunk_within_rank_semantic", "task_kind")])
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("gridloc", "task_kind")])
        list_var_color_var_subplot.append(["shape", "task_kind"])

        # Goal: show that chunk structure is more strongly represented compared to stroke index.
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task")])
        list_var_color_var_subplot.append(["stroke_index_fromlast", ("task_kind", "FEAT_num_strokes_task")])
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("FEAT_num_strokes_task", "task_kind", "stroke_index")])

        # Also add those for predicting seuqence
        LIST_VAR, LIST_VARS_OTHERS, LIST_OVERWRITE_lenient_n = params_getter_raster_vars(which_level, "PIG_BASE_stroke")

        single_vars_done = []
        for var_decode, vars_conj in zip(LIST_VAR, LIST_VARS_OTHERS):

            if isinstance(var_decode, list):
                var_decode = tuple(var_decode)

            list_var_color_var_subplot.append([var_decode, tuple(vars_conj)])
            if var_decode not in single_vars_done:
                list_var_color_var_subplot.append([var_decode, "task_kind"])
                single_vars_done.append(var_decode)

    elif question in ["RULESW_BASE_stroke"]:
        # Switching between rules (e.g., AnBm vs. DIR)
        # Currneetly assuming is just shapes vs. locations... (i.e. deterministic shape).
        list_var_color_var_subplot = []
        # list_var_color_var_subplot.append(["gridloc", "task_kind"])
        list_var_color_var_subplot.append(["gridloc", ("epoch", "task_kind")])
        list_var_color_var_subplot.append(["gridloc", ("epoch", "task_kind", "shape")])
        list_var_color_var_subplot.append(["gridloc_x", ("epoch", "task_kind")])
        list_var_color_var_subplot.append(["gridloc_x", ("epoch", "task_kind", "shape")])
        list_var_color_var_subplot.append(["gridloc_y", ("epoch", "task_kind")])
        list_var_color_var_subplot.append(["gridloc_y", ("epoch", "task_kind", "shape")])
        # list_var_color_var_subplot.append(["shape", "task_kind"])
        list_var_color_var_subplot.append(["shape", ("epoch", "task_kind")])
        list_var_color_var_subplot.append(["shape", ("epoch", "task_kind", "gridloc")])
        list_var_color_var_subplot.append(["shape", ("epoch", "task_kind", "gridloc_x")])
        list_var_color_var_subplot.append(["shape", ("epoch", "task_kind", "gridloc_y")])
        # list_var_color_var_subplot.append(["stroke_index", "task_kind"])
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epoch")])
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epoch", "gridloc_x")])
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epoch", "gridloc_y")])
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epoch", "shape")])

        # Check --> Effect of shape higher during shape epoch?
        list_var_color_var_subplot.append(["gridloc_x", ("stroke_index", "epoch")])
        list_var_color_var_subplot.append(["gridloc_y", ("stroke_index", "epoch")])
        list_var_color_var_subplot.append(["shape", ("stroke_index", "epoch")])

        list_var_color_var_subplot.append(["shape", ("stroke_index", "gridloc_x", "epoch")])
        list_var_color_var_subplot.append(["gridloc_x", ("stroke_index", "shape", "epoch")])

        list_var_color_var_subplot.append(["gridloc_x", ("epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["gridloc_y", ("epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["gridloc", ("epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["shape", ("epochset", "epoch")]) # ** GOOD

        # Check --> Controlling for context as much as possible.
        list_var_color_var_subplot.append(["shape", ("gridloc", "CTXT_loc_prev", "epoch")])
        list_var_color_var_subplot.append(["gridloc_x", ("shape", "CTXT_loc_prev", "epoch")])
        list_var_color_var_subplot.append(["gridloc_y", ("shape", "CTXT_loc_prev", "epoch")])

        # DIR vs. DIR, effect of x still present when control for y?
        list_var_color_var_subplot.append(["gridloc_x", ("epochset", "epoch", "gridloc_y")]) # ** GOOD (for DIR vs. DIR, that is correlated, e.g, U vs L)
        list_var_color_var_subplot.append(["gridloc_y", ("epochset", "epoch", "gridloc_x")])

        # Effect of epoch
        list_var_color_var_subplot.append(["epoch", ("epochset", "shape", "gridloc", "CTXT_loc_prev")])
        list_var_color_var_subplot.append(["epoch", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev")]) # ** GOOD
        list_var_color_var_subplot.append(["epoch", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next")])
        list_var_color_var_subplot.append(["epoch", "epochset"]) #

    elif question in ["RULESW_ANBMCK_COLRANK_STROKE"]:
        # Switching between grammar (usualyl AnBmCk) and color_rank, including both random and those using
        # same sequence as grammar.

        list_var_color_var_subplot = []

        # (1) Important ones copied from SHAPE vs DIR (above)
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epoch")])
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["gridloc_x", ("epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["gridloc", ("epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["shape", ("epochset", "epoch")]) # ** GOOD
        list_var_color_var_subplot.append(["epoch", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev")]) # ** GOOD
        list_var_color_var_subplot.append(["epoch", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next")])

        # # (2) Copies of above (1) but using color, not epoch.
        # list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "INSTRUCTION_COLOR")])
        # list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epochset", "INSTRUCTION_COLOR")]) # ** GOOD
        # list_var_color_var_subplot.append(["gridloc_x", ("epochset", "INSTRUCTION_COLOR")]) # ** GOOD
        # list_var_color_var_subplot.append(["gridloc", ("epochset", "INSTRUCTION_COLOR")]) # ** GOOD
        # list_var_color_var_subplot.append(["shape", ("epochset", "INSTRUCTION_COLOR")]) # ** GOOD
        # list_var_color_var_subplot.append(["INSTRUCTION_COLOR", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev")]) # ** GOOD
        # list_var_color_var_subplot.append(["INSTRUCTION_COLOR", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next")])

        # (2) Copies of above (1) but using color, not epoch.
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epoch_rand")])
        list_var_color_var_subplot.append(["stroke_index", ("task_kind", "FEAT_num_strokes_task", "epochset", "epoch_rand")]) # ** GOOD
        list_var_color_var_subplot.append(["gridloc_x", ("epochset", "epoch_rand")]) # ** GOOD
        list_var_color_var_subplot.append(["gridloc", ("epochset", "epoch_rand")]) # ** GOOD
        list_var_color_var_subplot.append(["shape", ("epochset", "epoch_rand")]) # ** GOOD
        list_var_color_var_subplot.append(["epoch_rand", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev")]) # ** GOOD
        list_var_color_var_subplot.append(["epoch_rand", ("epochset", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next")])

        # (2) Chunk rank stuff. This is possible becuase color trials also have extraction of chunk rank.
        list_var_color_var_subplot.append([("chunk_rank", "shape"), ("chunk_within_rank_semantic", "chunk_n_in_chunk", "task_kind", "epochset", "INSTRUCTION_COLOR")])
        list_var_color_var_subplot.append(["chunk_within_rank_semantic", ("chunk_rank", "shape", "chunk_n_in_chunk", "task_kind", "epochset", "INSTRUCTION_COLOR")])
        list_var_color_var_subplot.append(["chunk_within_rank_semantic", ("chunk_rank", "shape", "chunk_n_in_chunk", "task_kind", "epochset", "epoch")])
        list_var_color_var_subplot.append(["chunk_n_in_chunk", ("chunk_rank", "shape", "chunk_within_rank_semantic", "task_kind", "epochset", "INSTRUCTION_COLOR")])

        list_var_color_var_subplot.append([("chunk_rank", "shape"), ("chunk_within_rank_semantic", "chunk_n_in_chunk", "task_kind", "epoch_rand")])
        list_var_color_var_subplot.append(["chunk_within_rank_semantic", ("chunk_rank", "shape", "chunk_n_in_chunk", "task_kind", "epoch_rand")])
        list_var_color_var_subplot.append(["chunk_n_in_chunk", ("chunk_rank", "shape", "chunk_within_rank_semantic", "task_kind", "epoch_rand")])

        # (4) Stuff from SINGLE RULE (AnBm), but adding conditioning on epoch or INSTRUCTION_COLOR
        # Each subplot is a syntax parse. Color by
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("task_kind", "epochset", "epoch", "taskcat_by_rule", "behseq_shapes")]) # ** GOOD
        # list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("task_kind", "epochset", "INSTRUCTION_COLOR", "taskcat_by_rule", "behseq_shapes")]) # ** GOOD
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("task_kind", "epochset", "epoch_rand", "taskcat_by_rule", "behseq_shapes")]) # ** GOOD
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("task_kind", "epochset", "epoch")]) # ** GOOD
        # list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("task_kind", "epochset", "INSTRUCTION_COLOR")]) # ** GOOD
        list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank_semantic"), ("task_kind", "epochset", "epoch_rand")]) # ** GOOD

        # Each subplot is a syntax parse. Color by
        if False: # Skip for now, since taskcat_by_rule does not describe beh accurately for color instruction trials (it depends on the image).
            list_var_color_var_subplot.append([("chunk_rank", "chunk_within_rank"), ("taskcat_by_rule", "behseq_shapes", "task_kind", "epochset", "INSTRUCTION_COLOR")])
            list_var_color_var_subplot.append([("chunk_within_rank", "chunk_rank"), ("taskcat_by_rule", "behseq_shapes", "task_kind", "epochset", "INSTRUCTION_COLOR")])
            list_var_color_var_subplot.append([("shape", "chunk_within_rank"), ("taskcat_by_rule", "behseq_shapes", "task_kind", "epochset", "INSTRUCTION_COLOR")])
            list_var_color_var_subplot.append([("chunk_within_rank", "shape"), ("taskcat_by_rule", "behseq_shapes", "task_kind", "epochset", "INSTRUCTION_COLOR")])
    elif question in ["PIG_BASE_trial", "CHAR_BASE_trial"]:
        # list_var_color_var_subplot.append(["seqc_0_shape", "task_kind"])
        # if len(dflab["gridsize"].unique())>1:
        #     list_var_color_var_subplot.append(["seqc_0_shape", "gridsize"])
        #     list_var_color_var_subplot.append(["gridsize", "seqc_0_shape"])
        #     list_var_color_var_subplot.append(["gridsize", "task_kind"])
        # if len(dflab["seqc_0_loc"].unique())>1:
        #     list_var_color_var_subplot.append(["seqc_0_shape", "seqc_0_loc"])
        #     list_var_color_var_subplot.append(["seqc_0_loc", "seqc_0_shape"])
        #     list_var_color_var_subplot.append(["seqc_0_loc", "task_kind"])
        if len(dflab["shape_is_novel_all"].unique())>1:
            list_var_color_var_subplot.append(["seqc_0_shape", "shape_is_novel_all"])
            list_var_color_var_subplot.append(["shape_is_novel_all", "seqc_0_shape"])
            list_var_color_var_subplot.append(["seqc_0_shapesemcat", ("shape_is_novel_all", "task_kind")])
            list_var_color_var_subplot.append(["seqc_0_angle", ("shape_is_novel_all", "seqc_0_shape")]) # One subplot per shape, use seqc_0_angle is hack-- need a variation in var for this to not be skipped.
            # list_var_color_var_subplot.append(["shape_is_novel_all", "task_kind"])
        if "seqc_0_locx" in dflab.columns:
            list_var_color_var_subplot.append(["seqc_0_locx", "seqc_0_shapeloc"])
            list_var_color_var_subplot.append(["seqc_0_locy", "seqc_0_shapeloc"])
            list_var_color_var_subplot.append(["seqc_0_locx", "task_kind"])
            list_var_color_var_subplot.append(["seqc_0_locy", "task_kind"])
        if "seqc_0_angle" in dflab.columns:
            list_var_color_var_subplot.append(["seqc_0_angle", "seqc_0_shapeloc"])
            list_var_color_var_subplot.append(["seqc_0_angle", "task_kind"])
        if "seqc_0_angle_binned" in dflab.columns:
            list_var_color_var_subplot.append(["seqc_0_shape", "seqc_0_angle_binned"])
        # color by shape semantic category (vlPFC?)
        # list_var_color_var_subplot.append(["seqc_0_shapesemcat", "task_kind"])

        # (for character)
        # list_var_color_var_subplot.append(["seqc_0_shape", ("taskconfig_shp_SHSEM", "seqc_0_center_binned", "gridsize", "task_kind")]) # Same image --> diff sequence

        # sequence predictions
        if False:
            # list_var_color_var_subplot.append(["seqc_1_shapeloc", ("seqc_0_shapeloc", "task_kind")])
            list_var_color_var_subplot.append(["seqc_1_shape", ("seqc_0_shapeloc", "seqc_1_loc", "task_kind")])
            list_var_color_var_subplot.append(["seqc_1_loc", ("seqc_0_shapeloc", "seqc_1_shape", "task_kind")])
            # list_var_color_var_subplot.append(["seqc_2_shapeloc", ("seqc_0_shapeloc", "seqc_1_shapeloc", "task_kind")])
            list_var_color_var_subplot.append(["seqc_2_shape", ("seqc_0_shapeloc", "seqc_1_shapeloc", "seqc_2_loc", "task_kind")])
            list_var_color_var_subplot.append(["seqc_2_loc", ("seqc_0_shapeloc", "seqc_1_shapeloc", "seqc_2_shape", "task_kind")])
        # (for character)
        # list_var_color_var_subplot.append(["seqc_1_shape", ("seqc_0_shape", "seqc_0_center_binned", "seqc_1_locon_binned", "task_kind")])
        # list_var_color_var_subplot.append(["seqc_1_locon_binned", ("seqc_0_shape", "seqc_0_center_binned", "seqc_1_shape", "task_kind")])
        # list_var_color_var_subplot.append(["seqc_nstrokes_beh", ("seqc_0_shape", "seqc_0_center_binned", "task_kind")])
        # list_var_color_var_subplot.append(["seqc_nstrokes_beh", ("taskconfig_shp_SHSEM", "seqc_0_shape", "seqc_0_center_binned", "task_kind")]) # Same image --> diff sequence
        # list_var_color_var_subplot.append(["seqc_nstrokes_beh", "task_kind"])

        # Same image --> diff sequence
        if False:
            list_var_color_var_subplot.append(["seqc_0_shapeloc", ("character", "task_kind")])
            list_var_color_var_subplot.append(["FEAT_num_strokes_beh", ("character", "task_kind")])
            # list_var_color_var_subplot.append(["FEAT_num_strokes_beh", "task_kind"])

        # Image properties (controlling for beh).
        if False:
            list_var_color_var_subplot.append(["taskconfig_shp", ("taskconfig_loc", "seqc_0_shapeloc", "task_kind")])
            list_var_color_var_subplot.append(["taskconfig_shploc", ("FEAT_num_strokes_task", "seqc_0_shapeloc", "task_kind")])
            list_var_color_var_subplot.append(["FEAT_num_strokes_task", ("seqc_0_shapeloc", "task_kind")])
            list_var_color_var_subplot.append(["FEAT_num_strokes_task", "task_kind"])

        # Parse (characters)
        # list_var_color_var_subplot.append(["taskconfig_shp_SHSEM", ("taskconfig_loc", "task_kind")])
        # list_var_color_var_subplot.append(["taskconfig_shp_SHSEM", ("taskconfig_loc", "seqc_0_shape", "seqc_0_center_binned", "task_kind")])
        # list_var_color_var_subplot.append(["taskconfig_shp_SHSEM", ("character", "task_kind")])
        # list_var_color_var_subplot.append(["taskconfig_shploc_SHSEM", ("taskconfig_loc", "task_kind")])
        # list_var_color_var_subplot.append(["taskconfig_shploc_SHSEM", ("taskconfig_loc", "seqc_0_shape", "seqc_0_center_binned", "task_kind")])
        # list_var_color_var_subplot.append(["taskconfig_shploc_SHSEM", ("character", "task_kind")])
        # list_var_color_var_subplot.append(["taskconfig_shp_SHSEM", "task_kind"])
        # list_var_color_var_subplot.append(["taskconfig_shploc_SHSEM", "task_kind"])
        # list_var_color_var_subplot.append(["seqc_1_shape", ("seqc_0_shape", "seqc_0_center_binned", "task_kind")]) # actually, is testing for parsing.
        # list_var_color_var_subplot.append(["seqc_2_shape", ("seqc_0_shape", "seqc_0_center_binned", "task_kind")]) # actually, is testing for parsing.

        from neuralmonkey.metadat.analy.anova_params import params_getter_decode_vars
        LIST_VAR_DECODE, LIST_VARS_CONJ, LIST_SEPARATE_BY_TASK_KIND, LIST_FILTDICT, LIST_SUFFIX = params_getter_decode_vars(which_level)
        single_vars_done = []
        for list_var_decode, list_vars_conj in zip(LIST_VAR_DECODE, LIST_VARS_CONJ):
            for var_decode, vars_conj in zip(list_var_decode, list_vars_conj):
                list_var_color_var_subplot.append([var_decode, tuple(vars_conj)])
                if var_decode not in single_vars_done:
                    list_var_color_var_subplot.append([var_decode, "task_kind"])
                    single_vars_done.append(var_decode)

    elif question in ["PIG_BASE_stroke", "CHAR_BASE_stroke"]:
        # shape (effect of shape)
        # list_var_color_var_subplot.append(["shape", ("CTXT_shapeloc_prev", "gridloc", "CTXT_shapeloc_next", "task_kind")]) # (1) context --> after account for context, not much shape encoding.
        # list_var_color_var_subplot.append(["shape", ("CTXT_shapeloc_prev", "gridloc", "CTXT_loc_next", "task_kind")])
        list_var_color_var_subplot.append(["shape", ("CTXT_loc_prev", "gridloc", "task_kind")])
        # list_var_color_var_subplot.append(["shape", ("stroke_index", "gridloc", "task_kind")]) # effect of stroke index
        list_var_color_var_subplot.append(["shape", ("stroke_index", "task_kind")]) # important
        list_var_color_var_subplot.append(["shape", ("stroke_index", "stroke_index_fromlast_tskstks", "task_kind")]) # (1) PMv, shape is invariant, but different for first stroke, (2) preSMA, encode SI indep of shape.
        list_var_color_var_subplot.append(["shape", "task_kind"])

        # shape (invariance)
        list_var_color_var_subplot.append(["stroke_index_semantic", ("shape", "task_kind")]) # # also useful to see consistent for shape across contexts.
        list_var_color_var_subplot.append(["CTXT_ALL_shape", ("shape", "stroke_index_semantic", "task_kind")]) # (1) also useful to see consistent for shape across contexts. (in PMv, not in M1).
        list_var_color_var_subplot.append(["gridloc", ("shape", "stroke_index_semantic", "task_kind")]) # if many shapes, then this is easier to see if shape is invariant.
        # list_var_color_var_subplot.append(["task_kind", ("shape", "stroke_index_semantic")]) # if many shapes, then this is easier to see if shape is invariant.

        # location
        # list_var_color_var_subplot.append(["gridloc", ("CTXT_shapeloc_prev", "shape", "CTXT_shapeloc_next", "task_kind")])
        # list_var_color_var_subplot.append(["gridloc", ("CTXT_shapeloc_prev", "shape", "CTXT_loc_next", "task_kind")])
        list_var_color_var_subplot.append(["gridloc", ("CTXT_loc_prev", "shape", "task_kind")])
        list_var_color_var_subplot.append(["gridloc", ("stroke_index", "shape", "task_kind")])
        list_var_color_var_subplot.append(["gridloc", "task_kind"])

        # loc (vs. reach direction)
        list_var_color_var_subplot.append(["gridloc", ("gap_from_prev_angle_binned", "shape", "stroke_index_semantic", "task_kind")]) # control for SIS, since onset reach and offset can be different.
        list_var_color_var_subplot.append(["gridloc", ("gap_to_next_angle_binned", "shape", "stroke_index_semantic", "task_kind")])

        # state (seq context) (also: prediction)
        # list_var_color_var_subplot.append(["CTXT_shapeloc_next", ("CTXT_shapeloc_prev", "shape", "gridloc", "task_kind")])
        # list_var_color_var_subplot.append(["CTXT_shape_next", ("CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_loc_next", "task_kind")])
        # list_var_color_var_subplot.append(["CTXT_loc_next", ("CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shape_next", "task_kind")])
        # list_var_color_var_subplot.append(["CTXT_shapeloc_prev", ("CTXT_shapeloc_next", "shape", "gridloc", "task_kind")])
        # list_var_color_var_subplot.append(["CTXT_loc_next", "task_kind"])
        # list_var_color_var_subplot.append(["CTXT_shape_next", "task_kind"])

        # list_var_color_var_subplot.append(["CTXT_shapeloc_next", ("CTXT_shapeloc_prev", "shape", "gridloc", "stroke_index_semantic", "task_kind")]) # important to have SIS, to separate (shapeloc) from END.
        list_var_color_var_subplot.append(["CTXT_shapeloc_next", ("CTXT_loc_prev", "shape", "gridloc", "stroke_index_semantic", "task_kind")])  # important to have SIS, to separate (shapeloc) from END.
        list_var_color_var_subplot.append(["CTXT_loc_next", ("CTXT_loc_prev", "shape", "gridloc", "stroke_index_semantic", "CTXT_shape_next", "task_kind")])
        if False: # just get fewer plots
            list_var_color_var_subplot.append(["CTXT_shape_next", ("CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_loc_next", "stroke_index_semantic", "task_kind")]) # important to have SIS, to separate (shapeloc) from END.
            list_var_color_var_subplot.append(["CTXT_loc_next", ("CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shape_next", "stroke_index_semantic", "task_kind")]) # important to have SIS, to separate (shapeloc) from END.

        # list_var_color_var_subplot.append(["CTXT_shapeloc_prev", ("CTXT_shapeloc_next", "shape", "gridloc", "stroke_index_semantic", "task_kind")]) # important to have SIS, to separate (shapeloc) from END.
        list_var_color_var_subplot.append(["CTXT_shapeloc_prev", ("shape", "gridloc", "stroke_index_semantic", "task_kind")]) # important to have SIS, to separate (shapeloc) from END.
        # list_var_color_var_subplot.append(["CTXT_shapeloc_prev", ("shape", "gridloc", "task_kind")]) # also useful to see consistent for shape across contexts.

        list_var_color_var_subplot.append(["CTXT_shapeloc_prev", "task_kind"])
        list_var_color_var_subplot.append(["CTXT_shapeloc_next", "task_kind"])
        list_var_color_var_subplot.append(["CTXT_loc_next", "task_kind"])

        # stroke index (effect)
        # list_var_color_var_subplot.append(["stroke_index", ("CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shapeloc_next", "task_kind")])
        # list_var_color_var_subplot.append(["stroke_index", ("shape", "gridloc", "CTXT_shapeloc_next", "task_kind")])
        list_var_color_var_subplot.append(["stroke_index", ("CTXT_loc_prev", "shape", "gridloc", "task_kind")])
        list_var_color_var_subplot.append(["stroke_index", ("stroke_index_semantic", "task_kind")]) # Important: showing that PMv has no stroke effect if exclude first stroke
        list_var_color_var_subplot.append(["stroke_index", ("FEAT_num_strokes_task", "task_kind")]) # Important: is counting, or internal ,etc.
        list_var_color_var_subplot.append(["stroke_index", "task_kind"])

        # list_var_color_var_subplot.append(["stroke_index_fromlast_tskstks", ("CTXT_shapeloc_prev", "shape", "gridloc", "CTXT_shapeloc_next", "task_kind")]) # SI - Good (strongest control)
        list_var_color_var_subplot.append(["stroke_index_fromlast_tskstks", ("CTXT_loc_prev", "shape", "gridloc", "task_kind")])
        list_var_color_var_subplot.append(["stroke_index_fromlast_tskstks", ("stroke_index_semantic", "task_kind")]) # Important: showing that PMv has no stroke effect if exclude first stroke
        list_var_color_var_subplot.append(["stroke_index_fromlast_tskstks", "task_kind"])

        list_var_color_var_subplot.append(["CTXT_ALL_MAX", ("stroke_index_fromlast_tskstks", "task_kind")]) # Strong, test context vs. SI
        list_var_color_var_subplot.append(["CTXT_shapeloc_next", ("CTXT_shapeloc_prev", "stroke_index_fromlast_tskstks", "task_kind")]) # Strong, test context vs. SI

        # stroke index (invariance)
        list_var_color_var_subplot.append(["shape_loc", ("stroke_index", "task_kind")]) # (1) Stroke index invariant to shape/loc (2) Consistent across task_kind

        # contrast stroke index vs. stroke index from last
        if False: # too messy
            list_var_color_var_subplot.append(["stroke_index", ("shape", "gridloc", "stroke_index_fromlast_tskstks", "task_kind")])
            list_var_color_var_subplot.append(["stroke_index_fromlast_tskstks", ("shape", "gridloc", "stroke_index", "task_kind")])

        # task kind
        list_var_color_var_subplot.append(["task_kind", ("shape", "gridloc", "CTXT_shapeloc_prev")])

        # num strokes in task
        list_var_color_var_subplot.append(["FEAT_num_strokes_task", ("shape", "gridloc", "CTXT_shapeloc_prev")])

    else:
        print(question)
        assert False

    # Cleanup, if forgot to add taskkind
    tmp = []
    for var_color, var_subplot in list_var_color_var_subplot:
        if not var_color == "task_kind":
            if isinstance(var_subplot, (list, tuple)) and not any([v=="task_kind" for v in var_subplot]):
                var_subplot = tuple(list(var_subplot) + ["task_kind"])
            elif isinstance(var_subplot, str) and not var_subplot=="task_kind":
                var_subplot = tuple([var_subplot, "task_kind"])
            elif var_subplot is None:
                var_subplot = "task_kind"
        tmp.append([var_color, var_subplot])
    list_var_color_var_subplot = tmp

    return list_var_color_var_subplot

def params_getter_euclidian_vars(question, context_version="new"):
    """
    GOOD - these are the most carefully constructed, in terms of high control, and testing specific hypothes
    Helper to get variables for euclidian distnace when this involves specific hand-pikced variables to test speciifc
    hypotheses.
    Written for syntax analyses.
    """

    if question is None:
        all_questions = ["RULE_ROWCOL_STROKE", "RULE_DIR_STROKE", "RULE_ANBMCK_STROKE", "RULESW_ANBMCK_DIR_STROKE",
                         "RULE_COLRANK_STROKE", "RULESW_ANBMCK_COLRANK_STROKE", "RULESW_ANY_SEQSUP_STROKE",
                         "RULESW_ANBMCK_ABN_STROKE", "SP_BASE_stroke", "SP_BASE_trial", "PIG_BASE_stroke", "CHAR_BASE_stroke"]
        # Do quick check that lengths match up (hand entered correctly)
        for q in all_questions:
            print("... testing: ", q)
            LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(q)

            # Test that context and ovar match
            for i, (vars_others, context) in enumerate(zip(LIST_VARS_OTHERS, LIST_CONTEXT)):
                if context is not None:
                    try:
                        assert all([v in vars_others for v in context["same"]])
                        assert all([v in vars_others for v in context["diff"]])
                        assert all([v in context["same"] + context["diff"] for v in vars_others])
                    except Exception as err:
                        print(vars_others)
                        print(context)
                        print("index:", i)
                        raise err

        print("PAssed all tests!")
        return (None for _ in range(5))


    if question in ["PIG_BASE_stroke" , "CHAR_BASE_stroke"]:
        # OLD -- should be ok, but shoudl update this file
        # Mainly about sequence stuff.

        if question == "PIG_BASE_stroke":
            # task_kind_keep = "prims_on_grid"
            task_kind_keep = None
            var_loc_next = "CTXT_loc_next"
            var_loc_prev = "CTXT_loc_prev"
            var_loc = "gridloc"
        elif question == "CHAR_BASE_stroke":
            # task_kind_keep = "character"
            task_kind_keep = None
            var_loc_next = "CTXT_loconclust_next"
            var_loc_prev = "CTXT_locoffclust_prev"
            var_loc = "loc_on_clust"
        else:
            assert False


        LIST_VAR = [
            # [Predict sequence]
            "CTXT_loc_next",
            "CTXT_loc_next",
            "CTXT_loc_next",

            # [Predict sequence]
            "CTXT_shape_next",
            "CTXT_shape_next",
            "CTXT_shape_next",

            # [Task kind]
            "task_kind",

            # [Stroke index]
            "stroke_index",
            "stroke_index_fromlast_tskstks",
            "stroke_index_fromlast_tskstks",

            # [Num strokes]
            "FEAT_num_strokes_task",
            "FEAT_num_strokes_task",

            # [Shape]
            "shape",
            "shape",
            "shape",
            "shape",

            "shape",
            "shape",
            "shape",
            "shape",
            "shape",
            "shape",

            # [Location]
            "gridloc",
            var_loc,
            var_loc,
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc, "CTXT_shape_next"],
            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", "gridloc", "CTXT_shape_next"],
            ["stroke_index_is_first", "task_kind", "CTXT_loc_prev", "shape", "gridloc", "CTXT_shape_next"],

            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc, "CTXT_loc_next"],
            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", "gridloc", "CTXT_loc_next"],
            ["stroke_index_is_first", "task_kind", "CTXT_loc_prev", "shape", "gridloc", "CTXT_loc_next"],

            ["stroke_index_is_first", "shape", var_loc, var_loc_prev],

            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc],
            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc],
            ["stroke_index_is_first", "FEAT_num_strokes_task", "task_kind", var_loc_prev, "shape", var_loc],

            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc, "stroke_index"],
            ["stroke_index_is_first", "task_kind", var_loc_prev, "shape", var_loc],

            ["task_kind", "stroke_index_is_first", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next"],
            ["task_kind", "stroke_index_is_first", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev"],
            ["task_kind", "stroke_index_is_first", "gridloc", "CTXT_loc_prev", "CTXT_loc_next"],
            ["task_kind", "stroke_index_is_first", "gridloc", "CTXT_loc_prev"],

            ["task_kind", "stroke_index_is_first", "loc_on_clust", "CTXT_locoffclust_prev"],
            ["task_kind", "stroke_index_is_first", "loc_on_clust", "CTXT_locoffclust_prev", "CTXT_shape_prev"],
            ["task_kind", "stroke_index_is_first", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["task_kind", "stroke_index_is_first", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["task_kind", "stroke_index_is_first", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_loconclust_next"],
            ["task_kind", "stroke_index_is_first", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],

            ["stroke_index_is_first", "task_kind", "shape", var_loc_prev, "CTXT_loc_next"],
            ["stroke_index_is_first", "task_kind", "shape", var_loc_prev, "CTXT_loc_next"],
            ["stroke_index_is_first", "task_kind", "shape", var_loc_prev],
            ]
        LIST_CONTEXT = [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,

            {"same":["task_kind", "stroke_index_is_first"], "diff":["gridloc", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["gridloc", "CTXT_loc_prev", "CTXT_shape_prev"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["gridloc", "CTXT_loc_prev", "CTXT_loc_next"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["gridloc", "CTXT_loc_prev"], "diff_context_ver":"diff_specific_lenient"},

            {"same":["task_kind", "stroke_index_is_first"], "diff":["loc_on_clust", "CTXT_locoffclust_prev"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["loc_on_clust", "CTXT_locoffclust_prev", "CTXT_shape_prev"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_loconclust_next"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["task_kind", "stroke_index_is_first"], "diff":["loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"], "diff_context_ver":"diff_specific_lenient"},

            {"same":["stroke_index_is_first", "task_kind"], "diff":["shape", var_loc_prev, "CTXT_loc_next"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["stroke_index_is_first", "task_kind"], "diff":["shape", var_loc_prev, "CTXT_loc_next"], "diff_context_ver":"diff_specific_lenient"},
            {"same":["stroke_index_is_first", "task_kind"], "diff":["shape", var_loc_prev], "diff_context_ver":"diff_specific_lenient"},
            ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]

        if task_kind_keep is not None:
            f = {"task_kind":[task_kind_keep]}
        else:
            f = None
        LIST_FILTDICT = [
            f for _ in range(len(LIST_VAR))
        ]

    elif question == "SP_BASE_stroke":
        # Standard stuff, related to single strokes
        LIST_VAR = [
            # [Shape]
            "shape", # (gen across size)
            "shape", # (gen across loc)
            "shape", # (more strict)
            "shape", # (more strict)

            # [Shape semantic]
            "shape_semantic_cat",
            "shape_semantic_cat",
            "shape_semantic_cat", # (more strict)
            "shape_semantic_cat", # (more strict)

            # [Location]
            "gridloc",
            "loc_on_clust",
            
            # [Size]
            "gridsize",
            # TODO angle of stroke onset.
        ]
        LIST_VARS_OTHERS = [
            ["gridloc", "gridsize"],
            ["gridloc", "gridsize"],
            ["loc_on_clust", "gridsize"],
            ["loc_on_clust", "loc_off_clust", "gridsize"],

            ["gridloc", "gridsize"],
            ["gridloc", "gridsize"],
            ["loc_on_clust", "gridsize"],
            ["loc_on_clust", "loc_off_clust", "gridsize"],

            ["shape", "gridsize"],
            ["shape", "gridsize"],

            ["shape", "gridloc"],
            ]
        LIST_CONTEXT = [
            {"same":["gridloc"], "diff":["gridsize"]},
            {"same":["gridsize"], "diff":["gridloc"]},
            None,
            None,

            {"same":["gridloc"], "diff":["gridsize"]},
            {"same":["gridsize"], "diff":["gridloc"]},
            None,
            None,

            {"same":["gridsize"], "diff":["shape"]},
            {"same":["gridsize"], "diff":["shape"]},

            {"same":["gridloc"], "diff":["shape"]},
            ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        LIST_FILTDICT = [
            None for _ in range(len(LIST_VAR))
        ]

    elif question == "SP_BASE_trial":
        #TODO: this is temporaly
        # Standard stuff, related to single strokes
        LIST_VAR = [
            # [Shape]
            "seqc_0_shape", # (gen across size)
            "seqc_0_shape", # (gen across loc)

            # [Location]
            "seqc_0_loc",
            "seqc_0_loc",
            
            # [Size]
            "gridsize",
        ]
        LIST_VARS_OTHERS = [
            ["seqc_0_loc", "gridsize"],
            ["seqc_0_loc", "gridsize"],

            ["seqc_0_shape", "gridsize"],
            ["seqc_0_shape", "gridsize"],

            ["seqc_0_shape", "seqc_0_loc"],
            ]
        LIST_CONTEXT = [
            {"same":["seqc_0_loc"], "diff":["gridsize"]},
            {"same":["gridsize"], "diff":["seqc_0_loc"]},

            {"same":["gridsize"], "diff":["seqc_0_shape"]},
            {"same":["gridsize"], "diff":["seqc_0_shape"]},

            {"same":["seqc_0_loc"], "diff":["seqc_0_shape"]},
            ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        LIST_FILTDICT = [
            None for _ in range(len(LIST_VAR))
        ]

    elif question == "RULESW_ANBMCK_ABN_STROKE":
        LIST_VAR = [
            #### SPECIFIC TO (AB)n
            "chunk_rank", # in (AB)n, same shape, diff "chunk" **

            #### EFFECT OF EPOCH
            "epoch", # Epoch diff (diff motor)

            #### BELOW ARE ALL taken from rowcol (except where marked with @@)
            # Chunk_within_rank generalize across chunk_rank
            "chunk_within_rank", # generalizing across rows **
            "chunk_within_rank", # (more lenient)
            "chunk_within_rank_semantic", # (same, but semantic)

            "chunk_within_rank", # Generalize across syntax_concrete **

            # Syntax role, generalize across other things.
            "syntax_role", # Syntax role generalize across syntax concrete (and shape, as control)
            "syntax_role", # Syntax role generalize across shape **
            "syntax_role", # (more leneint)

            "syntax_role", # Syntax role generalize across shape seq **
            "syntax_role", # Generalize across location sequence **
            "syntax_role", # Generalize across shape and location sequence **

            # Syntax role, single-stroke level
            "syntax_role", # See effect of syntax even after controlling for shape and location
            "syntax_role", # See effect of syntax even after controlling for shape and location and context

            # Syntax role is not just stroke index.
            "syntax_role", # "Hierarchical" - See effect of syntax even after controlling for shape and stroke index?
            "syntax_role", # "Hierarchical" - See effect of syntax even after controlling for shape and stroke index?
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["epoch", "shape", "chunk_within_rank"],

            ["epochset", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],

            ["epoch", "syntax_concrete", "chunk_rank"], #
            ["epoch", "chunk_rank"], #
            ["epoch", "chunk_rank"], #

            ["epoch", "syntax_concrete", "chunk_rank"], #

            ["epoch", "syntax_concrete", "shape"],
            ["epoch", "syntax_concrete", "shape"],
            ["epoch", "shape"],

            ["epoch", "syntax_concrete", "behseq_shapes_clust"],
            ["epoch", "syntax_concrete", "behseq_locs_clust"],
            ["epoch", "shape", "syntax_concrete", "behseq_locs_clust"],

            ["epoch", "shape", "gridloc"],
            ["epoch", "shape", "gridloc", "CTXT_loc_prev"],

            ["epoch", "shape", "stroke_index"],
            ["epoch", "stroke_index"],
            ]
        LIST_CONTEXT = [
            {"same":["epoch", "shape"], "diff":["chunk_within_rank"]},

            None,

            {"same":["epoch", "syntax_concrete"], "diff":["chunk_rank"]},
            {"same":["epoch"], "diff":["chunk_rank"]},
            {"same":["epoch"], "diff":["chunk_rank"]},

            {"same":["epoch", "chunk_rank"], "diff":["syntax_concrete"]},

            {"same":["epoch", "shape"], "diff":["syntax_concrete"]},
            {"same":["epoch", "syntax_concrete"], "diff":["shape"]},
            {"same":["epoch"], "diff":["shape"]},

            {"same":["epoch", "syntax_concrete"], "diff":["behseq_shapes_clust"]},
            {"same":["epoch", "syntax_concrete"], "diff":["behseq_locs_clust"]},
            {"same":["epoch", "syntax_concrete", "shape"], "diff":["behseq_locs_clust"]},

            {"same":["epoch"], "diff":["shape", "gridloc"]},
            {"same":["epoch"], "diff":["shape", "gridloc", "CTXT_loc_prev"]},

            {"same":["epoch", "shape"], "diff":["stroke_index"]},
            {"same":["epoch"], "diff":["stroke_index"]},
            ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        filtdict = {
            "stroke_index": list(range(1, 10, 1)), # [1, ..., ]
        }
        LIST_FILTDICT = [
            filtdict for _ in range(len(LIST_VAR))
        ]

    elif question == "RULESW_ANY_SEQSUP_STROKE":
        # Any vs. sequence superviosion, where "any" could be one or more grammars.

        # Printed variables:
        # vars = ["epochset_diff_motor", "epoch_orig", "epoch", "epoch_rand", "INSTRUCTION_COLOR", "superv_is_seq_sup", "epoch_orig_rand_seq", "epoch_is_AnBmCk"]
        # (False, 'UL', 'UL', 'UL', False, False, False, False) :     900
        # (False, 'UL', 'UL|S', 'UL|S', False, True, False, False) :     402
        # (False, 'llCV2FstStk', 'llCV2FstStk|S', 'presetrand', False, True, True, False) :     6
        # (False, 'llCV3', 'llCV3', 'llCV3', False, False, False, True) :     894
        # (False, 'llCV3', 'llCV3|S', 'llCV3|S', False, True, False, False) :     438
        # (True, 'llCV2FstStk', 'llCV2FstStk|S', 'presetrand', False, True, True, False) :     402
        # (True, 'rndstr', 'rndstr|S', 'presetrand', False, True, True, False) :     78

        LIST_VAR = [
            ## preSMA, similar stroke_index for rule VS. sup_seq (on random seqsup only)
            "stroke_index", #
            "stroke_index", # preSMA, similar stroke_index for rule VS. sup_seq
            "stroke_index", # preSMA, similar stroke_index for rule VS. sup_seq

            "stroke_index", # stroke index (during sup_seq) doesnt care about behseq
            "stroke_index", # stroke index (during sup_seq) doesnt care about behseq

            ## preSMA loses SI structure during seqsup (motor matched) ***NEW
            "stroke_index", # (dont care if is motor matched -- get all data)
            "stroke_index", #
            "stroke_index", # (motor matched)
            "stroke_index", #

            "stroke_index", # (cleanest, by exclude last stroke... not motor matched)
            "stroke_index", #

            # ==+ Effect of supervision... WELL CONTROLED
            "superv_is_seq_sup", # color_rank (better approach, by testing color_rank directly) **
            "superv_is_seq_sup", # (more lenient)
            "superv_is_seq_sup", # (more lenient)
            "superv_is_seq_sup", # (more lenient)

            # === preSMA, strucrture collapses during seqsup?
            "chunk_within_rank_semantic",
            "chunk_within_rank_semantic",
            "stroke_index",
            "stroke_index",
            "stroke_index",
            "stroke_index",

        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["epoch_rand"], #
            ["superv_is_seq_sup"], #
            ["superv_is_seq_sup", "epoch_orig"], #

            ["epoch_rand", "FEAT_num_strokes_task", "behseq_shapes_clust", "behseq_locs_clust"], #
            ["superv_is_seq_sup", "FEAT_num_strokes_task", "behseq_shapes_clust", "behseq_locs_clust"], #

            ["epochset", "epoch_orig", "superv_is_seq_sup"],
            ["epochset", "epoch_rand"],
            ["epochset", "epoch_orig", "superv_is_seq_sup"],
            ["epochset", "epoch_rand"],

            ["epochset", "epoch_orig", "superv_is_seq_sup"],
            ["epochset", "epoch_rand"],

            # ==+ WELL CONTROLED
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],

            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"],
            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"],
            ["epochset_dir", "epoch_rand", "superv_is_seq_sup"],
            ["epochset_shape", "epoch_rand", "superv_is_seq_sup"],
            ["epochset_dir", "epoch_rand", "superv_is_seq_sup"],
            ["epochset_shape", "epoch_rand", "superv_is_seq_sup"],

            ]
        LIST_CONTEXT = [
            {"same":[], "diff":["epoch_rand"]},
            {"same":[], "diff":["superv_is_seq_sup"]},
            {"same":["epoch_orig"], "diff":["superv_is_seq_sup"]},

            {"same":["epoch_rand", "FEAT_num_strokes_task"], "diff":["behseq_shapes_clust", "behseq_locs_clust"]},
            {"same":["superv_is_seq_sup", "FEAT_num_strokes_task"], "diff":["behseq_shapes_clust", "behseq_locs_clust"]},

            {"same":["epochset", "epoch_orig"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset"], "diff":["epoch_rand"]},
            {"same":["epochset", "epoch_orig"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset"], "diff":["epoch_rand"]},

            {"same":["epochset", "epoch_orig"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset"], "diff":["epoch_rand"]},

            # ==+ WELL CONTROLED
            None,
            None,
            None,
            None,

            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset_dir", "epoch_rand"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset_shape", "epoch_rand"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset_dir", "epoch_rand"], "diff":["superv_is_seq_sup"]},
            {"same":["epochset_shape", "epoch_rand"], "diff":["superv_is_seq_sup"]},

        ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        LIST_FILTDICT = [
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(True, True), (False, False)]}, # if superv, then shoudl be rand
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(True, True), (False, False)]}, # if superv, then shoudl be rand
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(True, True), (False, False)]}, # if superv, then shoudl be rand

            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(True, True)]},
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(True, True)]},

            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(False, True), (False, False)]}, # exclude True,True, which is random seq sup.
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(False, True), (False, False)]}, # exclude True,True, which is random seq sup.

            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(False, True), (False, False)]}, # exclude True,True, which is random seq sup.
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1)), ("epoch_orig_rand_seq", "superv_is_seq_sup"):[(False, True), (False, False)]}, # exclude True,True, which is random seq sup.

            # ==+ WELL CONTROLED
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[False]},

            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
        ]

    elif question == "RULE_ROWCOL_STROKE":
        ################## ROWCOL
        LIST_VAR = [
            # Chunk_within_rank generalize across chunk_rank
            "chunk_within_rank", # generalizing across rows **
            "chunk_within_rank", # (more lenient)
            "chunk_within_rank_semantic", # (same, but semantic)

            "chunk_within_rank", # Generalize across syntax_concrete **

            # Syntax role, generalize across other things.
            "syntax_role", # Syntax role generalize across syntax concrete (and shape, as control)
            "syntax_role", # Syntax role generalize across shape **
            "syntax_role", # (more leneint)

            "syntax_role", # Syntax role generalize across shape seq **
            "syntax_role", # Generalize across shape and location sequence **

            # Syntax role, single-stroke level
            "syntax_role", # See effect of syntax even after controlling for shape and location
            "syntax_role", # See effect of syntax even after controlling for shape and location and context

            # Syntax role is not just stroke index.
            "syntax_role", # "Hierarchical" - See effect of syntax even after controlling for shape and stroke index?
            "syntax_role", # "Hierarchical" - See effect of syntax even after controlling for shape and stroke index?
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["epoch", "syntax_concrete", "chunk_rank"], #
            ["epoch", "chunk_rank"], #
            ["epoch", "chunk_rank"], #

            ["epoch", "syntax_concrete", "chunk_rank"], #

            ["epoch", "syntax_concrete", "shape"],
            ["epoch", "syntax_concrete", "shape"],
            ["epoch", "shape"],

            ["epoch", "syntax_concrete", "behseq_shapes_clust"],
            ["epoch", "shape", "syntax_concrete", "behseq_locs_clust"],

            ["epoch", "shape", "gridloc"],
            ["epoch", "shape", "gridloc", "CTXT_loc_prev"],

            ["epoch", "shape", "stroke_index"],
            ["epoch", "stroke_index"],
            ]
        LIST_CONTEXT = [
            {"same":["epoch", "syntax_concrete"], "diff":["chunk_rank"]},
            {"same":["epoch"], "diff":["chunk_rank"]},
            {"same":["epoch"], "diff":["chunk_rank"]},

            {"same":["epoch", "chunk_rank"], "diff":["syntax_concrete"]},

            {"same":["epoch", "shape"], "diff":["syntax_concrete"]},
            {"same":["epoch", "syntax_concrete"], "diff":["shape"]},
            {"same":["epoch"], "diff":["shape"]},

            {"same":["epoch", "syntax_concrete"], "diff":["behseq_shapes_clust"]},
            {"same":["epoch", "syntax_concrete", "shape"], "diff":["behseq_locs_clust"]},

            {"same":["epoch"], "diff":["shape", "gridloc"]},
            {"same":["epoch"], "diff":["shape", "gridloc", "CTXT_loc_prev"]},

            {"same":["epoch", "shape"], "diff":["stroke_index"]},
            {"same":["epoch"], "diff":["stroke_index"]},
            ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        filtdict = {
            "stroke_index": list(range(1, 10, 1)), # [1, ..., ]
        }
        LIST_FILTDICT = [
            filtdict for _ in range(len(LIST_VAR))
        ]

    elif question == "RULE_DIR_STROKE":
        # Single rule (direction)

        LIST_VAR = [
            # [Stroke index generlaizes]
            "stroke_index", # preSMA reflect stroke index no matter the shape
            "stroke_index", # (more strict - controlling loc context)
            "stroke_index", # (more strict -- generalize across gridloc_x or gridloc_y)
            "stroke_index", # (more strict -- generalize across gridloc_x or gridloc_y)
            "stroke_index", # [vs. location] (might not have enough data)

            # [SI generalizes across sequence]
            "stroke_index", # preSMA reflect stroke index no matter the shape sequence

            # [preSMA reflects locaiton, not shape]
            "gridloc",
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["epoch", "shape"], # (note: do not put sequence context here)
            ["epoch", "shape", "CTXT_loc_prev"], # (note: do not put sequence context here)
            ["epoch", "shape", "gridloc_x"], # .
            ["epoch", "shape", "gridloc_y"], # .
            ["epoch", "gridloc"], # .

            ["epoch", "FEAT_num_strokes_task", "behseq_shapes_clust"], # .

            ["epoch", "shape"], # .
            ]

        LIST_CONTEXT = [
            {"same":["epoch"], "diff":["shape"]},
            {"same":["epoch", "CTXT_loc_prev"], "diff":["shape"]},
            {"same":["epoch"], "diff":["shape", "gridloc_x"], "diff_context_ver":"diff_specific"},
            {"same":["epoch"], "diff":["shape", "gridloc_y"], "diff_context_ver":"diff_specific"},
            {"same":["epoch"], "diff":["gridloc"]},

            {"same":["epoch", "FEAT_num_strokes_task"], "diff":["behseq_shapes_clust"]},

            {"same":["epoch"], "diff":["shape"]},
        ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        LIST_FILTDICT = [
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
        ]

    elif question == "RULE_ANBMCK_STROKE":
        # All cases of single rule AnBmCk, including when multiple epochs with diff shapes (e.g.,, Pancho), same syntaxes
        # across epochs.

        ################## AnBmCk [good]
        LIST_VAR = [
            ############################  SINGLE EPOCH
            # [SHAPES] Chunk_wtihin (semantic), generalize across sjhapes/chunks
            "chunk_within_rank_semantic", # generalize across chunks (same syntax concrete)
            "chunk_within_rank_semantic", # (more lenient)

            # [LOCATION] Chunk_wtihin (semantic), generalize across concrete sequences (beh loc)
            "chunk_within_rank_semantic", # generalize across location sequences (cond on shape) ** (strict).
            "chunk_within_rank_semantic", # generalize across location sequences and syntax concrete

            # [INDEX] For exact same stroke index, see effect of chunk_within_rank_semantic?
            "chunk_within_rank_semantic", # Contrast: chunk_within_semantic vs. stroke index? **
            "chunk_within_rank_semantic", # (more lenient, allow for diff shapes across stroke indices)

            # [SHAPES] Chunk_wtihin (semantic), generalize across sjhapes/chunks
            "chunk_within_rank", # effect of rank within chunk (vs. across shapes)
            "chunk_within_rank", # (more lenient)

            # [SYNTAX CONCRETE] (Chunk within rank) --> generalize across syntax concrete
            "chunk_within_rank_semantic", # effect of rank within chunk (vs. across syntax_concrete) *** (NEW)
            "chunk_within_rank", # effect of rank within chunk (vs. across syntax_concrete)

            # # [INDEX] For exact same stroke index, see effect of chunk_within_rank_semantic?
            # "syntax_role", # See effect of syntax even after controlling for stroke index?

            ############################  Multiple epochs (diff shapes, same syntax).
            # [Gen across epoch]
            "syntax_role", # syntax role, generalizes across diff shape sets
            "syntax_role", # (more strict)
            "syntax_role", # (more strict)

            # [Directly test effect of epoch]
            "syntax_role", # syntax role, generalizes across diff shape sets
            "syntax_role", # syntax role, generalizes across diff shape sets
            "epoch", # effect of epoch (shape), conditioned on specifici syntax role (high for M1)
            "epoch", # (more strict))

            # [Pure epoch] (Ingore, since here epochs mean diff shapes, which diff even in M1)
            # "epoch", # effect of epoch (shape), conditioned on specifici syntax role (high for M1)

            ############################ STRONG CONTROL
            # CONTRAST SET A - [CHunk within rank]
            "chunk_within_rank_semantic", # strongest test of "within chunk". Also tests (within vs. between chunks).
            "chunk_within_rank_semantic", # (more lenient)
            "chunk_within_rank_semantic", # (more lenient)

            # [Counting from onset or offset of chunk]
            "chunk_within_rank", # (rank, exclude 0th stroke)
            "chunk_within_rank", # (lenient + allowing 0th stroke within each chunk)) *** (NEW)
            "chunk_within_rank_fromlast", # (rank from last, exclude 0th)
            "chunk_within_rank_fromlast", # (lenient + allowing 0th stroke within each chunk)
            "stroke_index", # (stroke index less effect compared to chunk rank?)
            "stroke_index",

            # [Pitting chunk_rank_within, from last, and stroke index against each other)
            "chunk_within_rank_fromlast", # (pit "from start" vs. "from last") *** (NEW)
            "chunk_within_rank", # (pit "from start" vs. "from last") *** (NEW)
            "chunk_within_rank_fromlast", # (more lenient)
            "chunk_within_rank", # (more lenient)
            "chunk_within_rank_fromlast",
            "stroke_index",

            # [Generic role]
            # "syntax_role", # strongest test of "within chunk" [just in case this is diff from chunk_within_rank_semantic]

            # CONTRAST SET A - [Syntax concrete] [GOOD - contrast this to chunk_within_rank_semantic above]
            "syntax_concrete", # Encoding of syntax

            # [CHunk rank]
            "chunk_rank", # In some cases, same shape in different chunk ranks...
            "chunk_rank", # (allowing diff shapes)

            # CONTRAST SET A - [Location] [GOOD - low-level effect]
            "gridloc", # (strict, vary just loc)
            "gridloc", # (lenient, vary loc and seq context)
            "gridloc", #
            "gridloc", #

            # [N in chunk]
            "chunk_n_in_chunk", # (allowing first stroke in chunk) *** (NEW)
            "chunk_n_in_chunk", # (allowing first stroke in chunk) *** (NEW)
            "chunk_n_in_chunk", # (allowing first stroke in chunk) *** (NEW)
            "chunk_n_in_chunk", # (allowing first stroke in chunk) *** (NEW)
            "chunk_n_in_chunk", # (allowing first stroke in chunk) *** (NEW)
            "chunk_n_in_chunk", # (allowing first stroke in chunk) *** (NEW)
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "shape"], # Hierarchy --> within_chunk vs across chunk
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "shape"], # Hierarchy -->< within_chunk vs. across chunk (across syntaxes)
            # ["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "shape", "stroke_index"],
            # ["stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "shape"], # Hierarchy --> within_chunk vs across chunk
            # ["stroke_index_is_last_tskstks", "epoch", "syntax_concrete", "shape"], # Hierarchy -->< within_chunk vs. across chunk (across syntaxes)
            # ["stroke_index_is_last_tskstks", "epoch", "shape", "stroke_index"],

            ["epoch", "syntax_concrete", "shape"], # Hierarchy --> within_chunk vs across chunk
            ["epoch", "shape"],

            ["epoch", "syntax_concrete", "shape", "behseq_locs_clust"], # Hierarchy --> within_chunk vs across chunk
            ["epoch", "syntax_concrete", "shape", "behseq_locs_clust"], # Hierarchy --> within_chunk vs across chunk

            ["epoch", "shape", "stroke_index"],
            ["epoch", "stroke_index"],

            ["epoch", "syntax_concrete", "shape"], # Hierarchy --> within_chunk vs across chunk
            ["epoch", "shape"], # Hierarchy --> within_chunk vs across chunk

            ["epoch", "syntax_concrete", "shape"], # Hierarchy -->< within_chunk vs. across chunk (across syntaxes)
            ["epoch", "syntax_concrete", "shape"], # Hierarchy -->< within_chunk vs. across chunk (across syntaxes)

            # ["epoch", "shape", "stroke_index"],

            ["epoch"],
            ["syntax_concrete", "epoch"], # IGNORE (is repeated 2 steps below)
            ["syntax_concrete", "epoch", "behseq_locs_clust"], # IGNORE (is repeated 2 steps below)

            ["syntax_concrete", "epoch"],
            ["syntax_concrete", "behseq_locs_clust", "epoch"],
            ["syntax_concrete", "syntax_role"],
            ["syntax_concrete", "behseq_locs_clust", "syntax_role"],

            # == STRONG CONTROL
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],

            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],

            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "chunk_within_rank"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "chunk_within_rank_fromlast"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust", "chunk_within_rank"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust", "chunk_within_rank_fromlast"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust", "stroke_index"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust", "chunk_within_rank_fromlast"],

            # ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],

            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "chunk_within_rank_semantic"],

            # ["epoch", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            # ["epoch", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "chunk_within_rank_semantic"],
            ["epoch", "shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev"],
            ["epoch", "gridloc", "CTXT_loc_prev", "chunk_within_rank_semantic"],

            ["epoch", "chunk_rank", "shape", "chunk_within_rank_semantic", "CTXT_shape_prev", "CTXT_locoffclust_prev", "CTXT_loconclust_next", "syntax_concrete"],
            ["epoch", "chunk_rank", "shape", "chunk_within_rank_semantic", "CTXT_shape_prev", "CTXT_locoffclust_prev", "CTXT_loconclust_next"],
            ["epoch", "chunk_rank", "shape", "chunk_within_rank_semantic", "CTXT_shape_prev", "CTXT_locoffclust_prev"],
            ["epoch", "chunk_rank", "shape", "chunk_within_rank_semantic", "CTXT_shape_prev"],

            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_shape_next"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_shape_next"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ]
        LIST_CONTEXT = [
            # {"same":["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "syntax_concrete"], "diff":["shape"]},
            # {"same":["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "shape"], "diff":["syntax_concrete"]},
            # {"same":["stroke_index_is_first", "stroke_index_is_last_tskstks", "epoch", "shape"], "diff":["stroke_index"]},
            # {"same":["stroke_index_is_last_tskstks", "epoch", "syntax_concrete"], "diff":["shape"]},
            # {"same":["stroke_index_is_last_tskstks", "epoch", "shape"], "diff":["syntax_concrete"]},
            # {"same":["stroke_index_is_last_tskstks", "epoch", "shape"], "diff":["stroke_index"]},

            {"same":["epoch", "syntax_concrete"], "diff":["shape"]},
            {"same":["epoch"], "diff":["shape"]},

            {"same":["epoch", "syntax_concrete", "shape"], "diff":["behseq_locs_clust"]},
            {"same":["epoch", "shape"], "diff":["syntax_concrete", "behseq_locs_clust"], "diff_context_ver":"diff_specific_lenient"},

            {"same":["epoch", "shape"], "diff":["stroke_index"]},
            {"same":["epoch"], "diff":["stroke_index"]},

            {"same":["epoch", "syntax_concrete"], "diff":["shape"]},
            {"same":["epoch"], "diff":["shape"]},

            {"same":["epoch", "shape"], "diff":["syntax_concrete"]},
            {"same":["epoch", "shape"], "diff":["syntax_concrete"]},

            # {"same":["epoch", "shape"], "diff":["stroke_index"]},

            {"same":[], "diff":["epoch"]},
            {"same":["syntax_concrete"], "diff":["epoch"]},
            {"same":["syntax_concrete", "behseq_locs_clust"], "diff":["epoch"]},

            {"same":["syntax_concrete"], "diff":["epoch"]},
            {"same":["syntax_concrete", "behseq_locs_clust"], "diff":["epoch"]},
            {"same":["syntax_concrete"], "diff":["syntax_role"]},
            {"same":["syntax_concrete", "behseq_locs_clust"], "diff":["syntax_role"]},

            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,

            {"same":["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"], "diff":["chunk_within_rank"]},
            {"same":["epoch", "chunk_rank", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"], "diff":["chunk_within_rank_fromlast"]},
            {"same":["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust"], "diff":["chunk_within_rank"]},
            {"same":["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust"], "diff":["chunk_within_rank_fromlast"]},
            {"same":["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust"], "diff":["stroke_index"]},
            {"same":["epoch", "chunk_rank", "shape", "loc_on_clust", "loc_off_clust"], "diff":["chunk_within_rank_fromlast"]},

            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        LIST_PRUNE_MIN_N_LEVS = [1 for _ in range(10)] + [2 for _ in range(35)]
        # Use 1 for things that use syntax role as effect. or else will throw out cases with 1 item in given chunk.

        filtdict = {
            "stroke_index": list(range(1, 10, 1)), # [1, ..., ]
            # "stroke_index_fromlast_tskstks": list(range(-10, -1, 1)), # [-10, ..., -2]
        }
        LIST_FILTDICT = [
            filtdict for _ in range(len(LIST_VAR))
        ]

        # For n in chunk, replace filter
        # LIST_FILTDICT[-3] = {"chunk_within_rank": [0]}
        # LIST_FILTDICT[-6] = {"stroke_index": list(range(1, 10, 1)), "chunk_within_rank": [0]}
        # LIST_FILTDICT[-5] = {"stroke_index": list(range(1, 10, 1)), "chunk_within_rank": [0]}
        # LIST_FILTDICT[-4] = {"stroke_index": list(range(1, 10, 1)), "chunk_within_rank": [0]}
        # LIST_FILTDICT[-3] = {"stroke_index": list(range(1, 10, 1)), "chunk_within_rank_fromlast": [-1]}
        # LIST_FILTDICT[-2] = {"stroke_index": list(range(1, 10, 1)), "chunk_within_rank_fromlast": [-1]}
        # LIST_FILTDICT[-1] = {"stroke_index": list(range(1, 10, 1)), "chunk_within_rank_fromlast": [-1]}
        LIST_FILTDICT[-6] = {"chunk_within_rank": [0]}
        LIST_FILTDICT[-5] = {"chunk_within_rank": [0]}
        LIST_FILTDICT[-4] = {"chunk_within_rank": [0]}
        LIST_FILTDICT[-3] = {"chunk_within_rank_fromlast": [-1]}
        LIST_FILTDICT[-2] = {"chunk_within_rank_fromlast": [-1]}
        LIST_FILTDICT[-1] = {"chunk_within_rank_fromlast": [-1]}

    elif question == "RULESW_ANBMCK_DIR_STROKE":
        # AnBmCk vs. DIR [DONE]
        # Includes (i) dir-specific and (ii) across dir-shape.
        # Why no include shape-speciifc variables? Not sure.

        # During dir, is more related to stroke index than shape (preSMA)
        # Stroke index: similar across epochs.
        LIST_VAR = [
            # == For DIR epoch
            "shape",
            # "shape",
            "stroke_index",

            # == For SHAPES epoch
            "gridloc",
            "stroke_index",

            # === Compare DIR and SHAPES
            "stroke_index",

            # ==+ WELL CONTROLED
            "epoch", # Epoch diff (diff motor)
            "epoch", # Epoch diff (same motor). **
            "epoch", # (not using epochset, but tighter control) *** NEW
            "epoch", # (not using epochset, but tighter control) *** NEW

            "epoch", # (not using epochset, but tighter control) *** NEW
            "epoch", # (not using epochset, but tighter control) *** NEW
            "epoch", # (not using epochset, but tighter control) *** NEW

            # === preSMA, strucrture collapses during dir?
            "chunk_within_rank_semantic",
            "chunk_within_rank_semantic",
            "stroke_index",
            "stroke_index",
            "stroke_index",
            "stroke_index",
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["epochset", "epoch", "stroke_index"], # Prediction: DIR epoch, effect of shape in M1 (cond on stroke index), but not preSMA
            # ["epochset", "epoch", "stroke_index"], # Prediction: DIR epoch, effect of shape in M1 (cond on stroke index), but not preSMA
            ["epochset", "epoch", "shape"], # Prediction: DIR epoch, effect of stroke index in preSMA even if condition on shape.

            ["epochset", "epoch", "stroke_index"], # Prediction: SHAPES epoch, effect of location in M1 (cond on stroke index), but not preSMA
            ["epochset", "epoch", "gridloc"], # Prediction: SHAPES epoch, effect of stroke index in preSMA even if condition on location.

            ["epochset", "epoch"], # Similarity of stroke index across DI r a --> within_chunk vs across chunk

            ["epochset", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["epochset", "shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_loconclust_next"],

            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["shape", "gridloc", "CTXT_loc_prev", "CTXT_shape_prev"],

            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "epoch_is_AnBmCk", "epoch_is_DIR"],
            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "epoch_is_AnBmCk", "epoch_is_DIR"],
            ["epochset_dir", "epoch_rand", "epoch_is_AnBmCk", "epoch_is_DIR"],
            ["epochset_shape", "epoch_rand", "epoch_is_AnBmCk", "epoch_is_DIR"],
            ["epochset_dir", "epoch_rand", "epoch_is_AnBmCk", "epoch_is_DIR"],
            ["epochset_shape", "epoch_rand", "epoch_is_AnBmCk", "epoch_is_DIR"],
            ]
        LIST_CONTEXT = [
            {"same":["epochset", "epoch"], "diff":["stroke_index"]},
            # {"same":["epochset", "epoch"], "diff":["stroke_index"]},
            {"same":["epochset", "epoch"], "diff":["shape"]},

            {"same":["epochset", "epoch"], "diff":["stroke_index"]},
            {"same":["epochset", "epoch"], "diff":["gridloc"]},

            {"same":["epochset"], "diff":["epoch"]},

            None,
            None,
            None,
            None,

            None,
            None,
            None,

            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape"], "diff":["epoch_is_AnBmCk", "epoch_is_DIR"]},
            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape"], "diff":["epoch_is_AnBmCk", "epoch_is_DIR"]},
            {"same":["epochset_dir", "epoch_rand"], "diff":["epoch_is_AnBmCk", "epoch_is_DIR"]},
            {"same":["epochset_shape", "epoch_rand"], "diff":["epoch_is_AnBmCk", "epoch_is_DIR"]},
            {"same":["epochset_dir", "epoch_rand"], "diff":["epoch_is_AnBmCk", "epoch_is_DIR"]},
            {"same":["epochset_shape", "epoch_rand"], "diff":["epoch_is_AnBmCk", "epoch_is_DIR"]},
        ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        # LIST_FILTDICT = [
        #     {"stroke_index": list(range(1, 10, 1)), "epochset":[("char", "UL", "llCV3")], "epoch":["UL"]},
        #     {"stroke_index": list(range(1, 10, 1)), "epochset":[("char", "UL", "llCV3")], "epoch":["llCV3"]},
        #     {"stroke_index": list(range(1, 10, 1)), "epochset":[("char", "UL", "llCV3")]},
        # ]
        LIST_FILTDICT = [
            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True], ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(False, False, False)]},
            # {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True], ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(False, True, False)]},
            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True], ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(False, False, False)]},

            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True], ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(False, True, False)]},
            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True], ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(False, True, False)]},

            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True]},

            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True]},
            {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[False]},

            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1))},
            {},
            {},
            {},
            {},
            {},

            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
        ]
    elif question=="RULE_COLRANK_STROKE":
        # COL RANK [GOOD]
        # Questions:
        # - preSMA encodes stroke index for COL_RANK
        # - preSMA, stroke index during COL_RANK doesnt care about behseq.
        # - preSMA, doesnt encode location or shape during COL_RANK.

        LIST_VAR = [
            # [LOC AND SHAPE specific sequences]
            "stroke_index", # Effect of stroke index --> gen across location sequences
            "stroke_index", # Effect of stroke index --> gen across shape sequences
            "stroke_index", # (exclude 1st stroke)   **
            "stroke_index", # (exclude 1st stroke)  **

            # [Well controlled, SHAPE, LOC, and combo]
            "stroke_index", # (strong control) **
            "stroke_index", # (strongest control) **
            "stroke_index", # (weaker)
            "stroke_index", # (weakest) ***
            "stroke_index", # (strong control) (exclude first storke) **
            "stroke_index", # (strongest control) (exclude first storke) **

            # [Shape and location encoding, not about sequence]
            "shape", # effect of shape, holding others constant.
            "gridloc", # effect of location, holding others constant.
            "shape", # (more strict)
            "gridloc", # (more strict)
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["FEAT_num_strokes_task", "behseq_locs_clust"], #
            ["FEAT_num_strokes_task", "behseq_shapes_clust"],
            ["FEAT_num_strokes_task", "behseq_locs_clust"], #
            ["FEAT_num_strokes_task", "behseq_shapes_clust"],

            ["shape", "gridloc", "CTXT_loc_prev"],
            ["shape", "gridloc", "CTXT_loc_prev"],
            ["shape", "gridloc"],
            ["shape", "gridloc"],
            ["shape", "gridloc", "CTXT_loc_prev"],
            ["shape", "gridloc", "CTXT_loc_prev"],

            ["gridloc", "stroke_index"],
            ["shape", "stroke_index"],
            ["gridloc", "stroke_index", "CTXT_loc_prev"],
            ["shape", "stroke_index", "CTXT_loc_prev"],
            ]
        LIST_CONTEXT = [
            {"same":["FEAT_num_strokes_task"], "diff":["behseq_locs_clust"]},
            {"same":["FEAT_num_strokes_task"], "diff":["behseq_shapes_clust"]},
            {"same":["FEAT_num_strokes_task"], "diff":["behseq_locs_clust"]},
            {"same":["FEAT_num_strokes_task"], "diff":["behseq_shapes_clust"]},

            {"same":["gridloc", "CTXT_loc_prev"], "diff":["shape"]},
            {"same":["CTXT_loc_prev"], "diff":["shape", "gridloc"], "diff_context_ver":"diff_specific"},
            {"same":[], "diff":["shape", "gridloc"], "diff_context_ver":"diff_specific"},
            {"same":["gridloc"], "diff":["shape"]},
            {"same":["gridloc", "CTXT_loc_prev"], "diff":["shape"]},
            {"same":["CTXT_loc_prev"], "diff":["shape", "gridloc"], "diff_context_ver":"diff_specific"},

            {"same":["gridloc"], "diff":["stroke_index"]},
            {"same":["shape"], "diff":["stroke_index"]},
            {"same":["gridloc", "CTXT_loc_prev"], "diff":["stroke_index"]},
            {"same":["shape", "CTXT_loc_prev"], "diff":["stroke_index"]},
        ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        LIST_FILTDICT = [
            {}, {},
            {"stroke_index": list(range(1, 10, 1))}, {"stroke_index": list(range(1, 10, 1))},
            {}, {}, # Dont remove first stroke, just not enough data
            {}, {},
            {"stroke_index": list(range(1, 10, 1))}, {"stroke_index": list(range(1, 10, 1))},
            {}, {},
            {}, {},
        ]

    elif question == "RULESW_ANBMCK_COLRANK_STROKE":
        ################# AnBmCk vs. COL_RANK [GOOD]
        # Notes:
        # - epochsets are a mess (too many)
        # - use epoch_rand, not epoch, to lump all random ones.
        # - Ignore epoch|1, since this is very similar to epoch...
        LIST_VAR = [

            # [Stroke index generalizes across shapes and col_rank]
            "stroke_index", # preSMA, similar stroke_index for SHAPES VS. COL_RANK
            "stroke_index", # (more strict))
            "stroke_index", # (more strict))

            # [SI gen across sequences]
            "stroke_index", # stroke index (during COL_RANK) doesnt care about behseq

            # ==+ WELL CONTROLED
            "epoch", # Epoch diff (diff motor)
            "epoch", # Epoch diff (same motor).

            "INSTRUCTION_COLOR", # color_rank (better approach, by testing color_rank directly) **
            "INSTRUCTION_COLOR", # (more lenient)
            "INSTRUCTION_COLOR", # (more lenient)
            "INSTRUCTION_COLOR", # (more lenient)

            # === preSMA, strucrture collapses during seqsup?
            "chunk_within_rank_semantic",
            "chunk_within_rank_semantic",
            "stroke_index",
            "stroke_index",
        ]
        # More restrictive
        LIST_VARS_OTHERS = [
            ["epoch_rand"], #
            ["epoch_rand", "gridloc"],
            ["epoch_rand", "shape"],

            ["epoch_rand", "FEAT_num_strokes_task", "behseq_shapes_clust", "behseq_locs_clust"], #

            # ==+ WELL CONTROLED
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],

            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev", "CTXT_loconclust_next"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust", "CTXT_shape_prev"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],
            ["shape", "loc_on_clust", "CTXT_locoffclust_prev", "loc_off_clust"],

            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "INSTRUCTION_COLOR"],
            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "INSTRUCTION_COLOR"],
            ["epochset_shape", "epoch_rand", "INSTRUCTION_COLOR"],
            ["epochset_shape", "epoch_rand", "INSTRUCTION_COLOR"],
            ]

        LIST_CONTEXT = [
            {"same":[], "diff":["epoch_rand"]},
            {"same":["gridloc"], "diff":["epoch_rand"]},
            {"same":["shape"], "diff":["epoch_rand"]},

            {"same":["epoch_rand", "FEAT_num_strokes_task"], "diff":["behseq_shapes_clust", "behseq_locs_clust"]},

            # ==+ WELL CONTROLED
            None,
            None,

            None,
            None,
            None,
            None,

            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape"], "diff":["INSTRUCTION_COLOR"]},
            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape"], "diff":["INSTRUCTION_COLOR"]},
            {"same":["epochset_shape", "epoch_rand"], "diff":["INSTRUCTION_COLOR"]},
            {"same":["epochset_shape", "epoch_rand"], "diff":["INSTRUCTION_COLOR"]},
        ]
        LIST_PRUNE_MIN_N_LEVS = [
            2 for _ in range(len(LIST_VAR))
        ]
        LIST_FILTDICT = [
            # {"stroke_index": list(range(1, 10, 1)), "epoch":["llCV2FstStk", "llCV2|0"]},
            # {"stroke_index": list(range(1, 10, 1)), "epoch":["llCV2FstStk", "llCV2|0"]},
            # {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(True, False, True), (False, True, False)]}, # if superv, then shoudl be rand
            # {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(True, False, True), (False, True, False)]},
            # {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(True, False, True), (False, True, False)]},
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "INSTRUCTION_COLOR"):[(True, True), (False, False)]}, # if superv, then shoudl be rand
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "INSTRUCTION_COLOR"):[(True, True), (False, False)]},
            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "INSTRUCTION_COLOR"):[(True, True), (False, False)]},

            {"stroke_index": list(range(1, 10, 1)), ("epoch_orig_rand_seq", "epoch_is_AnBmCk", "INSTRUCTION_COLOR"):[(True, False, True)]},

            # ==+ WELL CONTROLED
            # {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[True]},
            # {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[False]},
            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1))},
            # {"stroke_index": list(range(1, 10, 1)), "epochset_diff_motor":[False]},
            {"epochset_diff_motor":[True]},
            {"epochset_diff_motor":[False]},
            {},
            {},
            {},
            {"epochset_diff_motor":[False]},

            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
            {"stroke_index": list(range(1, 10, 1))},
            {"stroke_index": list(range(1, 10, 1)), "stroke_index_fromlast_tskstks": list(range(-10, -1, 1))},
        ]
    else:
        print(question)
        assert False

    if not len(LIST_VAR)==len(LIST_VARS_OTHERS)==len(LIST_CONTEXT)==len(LIST_PRUNE_MIN_N_LEVS)==len(LIST_FILTDICT):
        print(LIST_VAR)
        print(len(LIST_VAR))
        print(LIST_VARS_OTHERS)
        print(len(LIST_VARS_OTHERS))
        print(LIST_CONTEXT)
        print(len(LIST_CONTEXT))
        print(LIST_PRUNE_MIN_N_LEVS)
        print(len(LIST_PRUNE_MIN_N_LEVS))
        print(LIST_FILTDICT)
        print(len(LIST_FILTDICT))
        assert False    

    ### WHether is old or new version of context
    # old -- using the time-varying eucl. context is confusing combination of vars_others and contet
    # new -- using the faster method. vars_others and context remain separate (i.e, context always applies)
    if context_version=="old":
        pass
    elif context_version == "new":
        # Update context, so that it is compatible with new approahc, where vars_others and context independent.
        # To see what is wrong with old approach, see docs in Cl.rsa_mask_context_helper() -- basicalyl, it is 
        # confusing beucase vars_others, and context interact.

        for i, (var, vars_others, context) in enumerate(zip(LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT)):
            if context is not None:
                assert sorted(vars_others) == sorted(context["same"] + context["diff"])
                print(i, " == ", var, " -- ", vars_others, " -- ", context)

                # - context["diff"] is only used for "diff" pairs, and so it is redundant with vars_others.

                # - check that it is in vars_others
                if len(context["diff"])>0:
                    assert all([c in vars_others for c in context["diff"]])
                
                # - Remove it
                context["diff"] = []

                if "diff_context_ver" in context:
                    # if context["diff_context_ver"] == "diff_specific_lenient":
                    #     diff_context_ver = "diff_at_least_one"
                    # elif context["diff_context_ver"] == "diff_specific":
                    #     diff_context_ver = "diff_complete"
                    #     assert False, "need to tell the code somehow that should diff vars_others shold be complete"
                    # else:
                    #     # Default mode.
                    #     diff_context_ver = "diff_at_least_one"
                    if context["diff_context_ver"] == "diff_specific":
                        diff_context_ver = "diff_complete"
                        assert False, "need to tell the code somehow that should diff vars_others shold be complete"

                # Check for cases where no "diff var_other" is possible (mistakenly)
                print()
                if sorted(context["same"]) == sorted(vars_others):
                    print(i, " == ", var, " -- ", vars_others, " -- ", context)
                    assert False
        print("context and vars_others are identical...")
        print("LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT are ready to use!")

    # Standardize the output format.
    def _context_cleanup(context):
        """
        Returns (potentialyl) copy
        """

        if context is None:
            return context
        else:    
            assert isinstance(context, dict)

            # Must have same and diff
            if "same" not in context:
                context["same"] = None
            if "diff" not in context:
                context["diff"] = None
            
            # empy --> None
            if (context["same"] is not None) and (len(context["same"]))==0:
                context["same"] = None
            if (context["diff"] is not None) and (len(context["diff"]))==0:
                context["diff"] = None

            # If both same and diff are None, then make the whole thing None
            if context["same"] is None and context["diff"] is None:
                context = None
            
            return context
    LIST_CONTEXT = [_context_cleanup(context) for context in LIST_CONTEXT]

    return LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT

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

