""" Holds params for anava analysis for each date
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from pythonlib.tools.expttools import writeStringsToFile

ONLY_ESSENTIAL_VARS = False # then just the first var, assuemd to be most important, for quick analys

##################
LIST_ANALYSES = ["rulesw", "ruleswALLDATA", "ruleswERROR", "singleprimvar", "seqcontextvar", "seqcontext", "singleprim"] # repo of possible analses, 

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
        elif animal=="Diego" and DATE in [230701, 230702, 230703, 230704, 230705, 230706, 230707, 230713, 230717, 230719, 230802]:
            # A single correct sequence
            DO_SCORE_SEQUENCE_VER = "matlab"
        else:
            print(animal, DATE)
            assert False
    elif ANALY_VER in ["singleprimvar", "seqcontext", "singleprim", "seqcontextvar"]:
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
    elif ANALY_VER in ["seqcontext", "singleprim", "singleprimvar", "seqcontextvar"]:
        preprocess_steps_append = ["remove_repeated_trials", "one_to_one_beh_task_strokes"]
    else:
        assert False
    preprocess_steps_append.append("beh_strokes_at_least_one")

    # Remove aborts
    if False:
        if ANALY_VER in ["ruleswERROR", "ruleswALLDATA"]:
            # error trials
            remove_aborts = False
        else:        
            # correct trials
            remove_aborts = True
    else:
        # Never remove aborts. Instead already pruning based on things like one to one, or correct
        # sequence. This so includes cases where complete but abort on the last stroke simply for 
        # shape quality.
        remove_aborts = False

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
    DO_EXTRACT_TASKCONFIG = False
    if "taskconfig_loc" in LIST_VAR_ALL or "taskconfig_shp" in LIST_VAR_ALL or "taskconfig_shploc" in LIST_VAR_ALL:
        DO_EXTRACT_TASKCONFIG = True
    if ANALY_VER=="seqcontextvar":
        DO_EXTRACT_TASKCONFIG = True

    
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
        "list_superv_keep":list_superv_keep,
        "preprocess_steps_append":preprocess_steps_append,
        "remove_aborts":remove_aborts,
        "DO_SCORE_SEQUENCE_VER":DO_SCORE_SEQUENCE_VER,
        "list_superv_keep_full":list_superv_keep_full,
        "DO_EXTRACT_TASKCONFIG":DO_EXTRACT_TASKCONFIG,
        "VARS_GROUP_PLOT":VARS_GROUP_PLOT
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
        # DO_EXTRACT_EPOCHKIND = True
    else:
        taskgroup_reassign_simple_neural = False
        # DO_EXTRACT_EPOCHKIND = False

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
        # Label each trial based on its conjunction of character and correct beh sequence.
        DO_CHARSEQ_VER = "task_matlab"
        EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "char_seq"
        EXTRACT_EPOCHSETS_n_max_epochs = 3
        EXTRACT_EPOCHSETS_merge_sets = True
    elif ANALY_VER in ["singleprimvar"]:
        # Label each trial based on its (shape/loc).
        DO_CHARSEQ_VER = None
        EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "seqc_0_loc_shape"
        EXTRACT_EPOCHSETS_n_max_epochs = 10 # make this higher, since these are usually clean expts.
        EXTRACT_EPOCHSETS_merge_sets = True
    elif ANALY_VER in ["seqcontextvar"]:
        # Label each trial based on its (task config).
        DO_CHARSEQ_VER = None
        EXTRACT_EPOCHSETS = True
        EXTRACT_EPOCHSETS_trial_label = "taskconfig_shploc"
        EXTRACT_EPOCHSETS_n_max_epochs = 10 # make this higher, since these are usually clean expts.
        EXTRACT_EPOCHSETS_merge_sets = True
    else:
        DO_CHARSEQ_VER = None
        EXTRACT_EPOCHSETS = False
        EXTRACT_EPOCHSETS_trial_label = None
        EXTRACT_EPOCHSETS_n_max_epochs = None
        EXTRACT_EPOCHSETS_merge_sets = None

    ################ FEATURES TO EXTRACT
    if which_level=="trial":
        if ANALY_VER in ["ruleswALLDATA", "ruleswERROR", "rulesw", "seqcontext", "singleprim", "singleprimvar", "seqcontextvar"]:
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

    ######### FINALLY, FLIP SOME "DO_" FLAGS BASD ONW HAT VAR YOU WANT.
    params_plots = params_getter_plots(animal, DATE, which_level, ANALY_VER)
    ##### Auto decide what to extract based on vars (AUTO EXTRACT BASED ON WHAT VARS ARE EXTRACTED)
    LIST_VAR_ALL = [x for x in params_plots["LIST_VAR"]]
    for vars_conj in params_plots["LIST_VARS_CONJUNCTION"]:
        LIST_VAR_ALL.extend(vars_conj)
    if "epochkind" in LIST_VAR_ALL:
        DO_EXTRACT_EPOCHKIND = True
    else:
        DO_EXTRACT_EPOCHKIND = False

    ###### Feature to get, based on flags.
    if DO_CHARSEQ_VER:
        list_features_modulation_append.append("char_seq")                                        
    if EXTRACT_EPOCHSETS:
        list_features_modulation_append.append("epochset")               
    if DO_SCORE_SEQUENCE_VER:
        list_features_modulation_append.append("success_binary_quick")                         
    if DO_EXTRACT_EPOCHKIND:
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
        "DO_EXTRACT_EPOCHKIND":DO_EXTRACT_EPOCHKIND
        }

    return params


def dataset_apply_params(ListD, animal, DATE, which_level, ANALY_VER, anova_interaction=False):
    """Preprocess dataset in all ways, including pruning, appending/modifying columns, etc.
    PARAMS:
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
    from neuralmonkey.classes.snippets import _dataset_extract_prune_general, _dataset_extract_prune_general_dataset

    ################################### LOAD PARAMS
    params = params_getter_plots(animal, DATE, which_level, ANALY_VER, 
        anova_interaction=anova_interaction)
    params_extraction = params_getter_extraction(animal, DATE, which_level, ANALY_VER)


    ################# BEH DATASET
    # First, concatenate all D.
    list_dataset = []
    for i, D in enumerate(ListD):
        # if which_level=="trial":
        # use the dataset here, since it is not saved
        # D = sn.Datasetbeh
        # else:
        #     # use the dataset linked to DS, since it is saved
        #     D = SP.DSmult[i].Dataset
        #     assert len(D.Dat)==len(sn.Datasetbeh.Dat), "a sanity check. desnt have to be, but I am curious why it is not..."

        # Becasue dataset can be locked, just replace it with copy
        D = D.copy()

        # THINGs that must be done by each individual D
        D.behclass_preprocess_wrapper()

        # Second, do preprocessing to concatted D
        if params_extraction["DO_SCORE_SEQUENCE_VER"]=="parses":
            D.grammarparses_successbinary_score()
        elif params_extraction["DO_SCORE_SEQUENCE_VER"]=="matlab":
            D.grammarmatlab_successbinary_score()
        else:
            # dont score
            assert params_extraction["DO_SCORE_SEQUENCE_VER"] is None

        if params_extraction["taskgroup_reassign_simple_neural"]:
            # do here, so the new taskgroup can be used as a feature.
            D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
            print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
            D.grouping_print_n_samples(["taskgroup", "probe"])

        if params_extraction["DO_CHARSEQ_VER"] is not None:
            D.sequence_char_taskclass_assign_char_seq(ver=params_extraction["DO_CHARSEQ_VER"])

        list_dataset.append(D.copy())
    # concat the datasets 
    Dall = concatDatasets(list_dataset)

    ################ DO SAME THING AS IN EXTRACTION (these dont fail, when use concatted)
    if params_extraction["DO_EXTRACT_CONTEXT"]:
        Dall.seqcontext_preprocess()

    if params["DO_EXTRACT_TASKCONFIG"]:
        Dall.taskclass_shapes_loc_configuration_assign_column()

    for this in params_extraction["list_epoch_merge"]:
        # D.supervision_epochs_merge_these(["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
        Dall.supervision_epochs_merge_these(this[0], this[1], key=params_extraction["epoch_merge_key"],
            assert_list_epochs_exist=False)


    if params_extraction["EXTRACT_EPOCHSETS"]:
        if params_extraction["EXTRACT_EPOCHSETS_trial_label"] == "char_seq":
            # This useful to separate into cases with same first stroke, and also chars present across contexts,
            # separating out single prims if they exist into their own epochset.
            versions_ordered = ["char", "same_beh_first_stroke", "same_beh"]
            Dall.epochset_apply_sequence_wrapper(versions_ordered=versions_ordered)
        else:
            # Only apply epochset extraction once.
            Dall.epochset_extract_common_epoch_sets(
                trial_label=params_extraction["EXTRACT_EPOCHSETS_trial_label"],  
                n_max_epochs=params_extraction["EXTRACT_EPOCHSETS_n_max_epochs"],
                merge_sets_with_only_single_epoch=params_extraction["EXTRACT_EPOCHSETS_merge_sets"],
                merge_sets_with_only_single_epoch_name = ("LEFTOVER",))

    if params_extraction["DO_EXTRACT_EPOCHKIND"]:
        Dall.supervision_epochs_extract_epochkind()

        
    # Sanity check that didn't remove too much data.
    if False:
        if "wrong_sequencing_binary_score" not in params["preprocess_steps_append"]:
            # Skip if is error trials.
            npre = len(D.Dat)
            npost = len(dat_pruned.Dat)
            if npost/npre<0.25 and len(sn.Datasetbeh.Dat)>200: # ie ignore this if it is a small session...
                print(params)
                print("THis has no wrong_sequencing_binary_score: ",  params['preprocess_steps_append'])
                assert False, "dataset pruning removed >0.75 of data. Are you sure correct? Maybe removing a supervisiuon stage that is actually important?"
    
    ###### PRUNE DATASET TO GET SUBSET TRIALCODES
    # Only keep subset these trialcodes
    dataset_pruned_for_trial_analysis = _dataset_extract_prune_general_dataset(Dall, 
        list_superv_keep=params["list_superv_keep"],  
        preprocess_steps_append=params["preprocess_steps_append"],
        remove_aborts=params["remove_aborts"],
        list_superv_keep_full=params["list_superv_keep_full"], 
        )    
    TRIALCODES_KEEP = dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist()

    return Dall, dataset_pruned_for_trial_analysis, TRIALCODES_KEEP, params, params_extraction



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

    ### Prep dataset, and extract params
    _, Dpruned, TRIALCODES_KEEP, params, params_extraction = dataset_apply_params(ListD, 
        animal, DATE, which_level, ANALY_VER)
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