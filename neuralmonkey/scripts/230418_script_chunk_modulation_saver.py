""" just for saving snippets for each session
"""

assert False, 'use chunk_moudlation!'
from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os


DATE = 220929
animal = "Pancho"
LIST_SESSIONS = [0,1,2]
PRE_DUR = -0.6
POST_DUR = 0.6
lenient_allow_data_if_has_n_levels=2
n_min = 5
# score_ver='r2smfr_minshuff'
PRE_DUR_CALC = -0.1
POST_DUR_CALC = 0.25
# score_ver='r2smfr_minshuff'
score_ver='r2smfr_zscore'
DEBUG = False # runs fast

# LIST_VAR = [
#     "chunk_within_rank"
#     # "chunk_rank"
#     ]

# LIST_VARS_CONJUNCTION = [
#     ['chunk_rank'] # list of str, vars to take conjunction over
#     # ['gridloc', 'chunk_within_rank'] # list of str, vars to take conjunction over
# ]

# LIST_VAR = [
#     "shape_oriented",
#     "gridloc"
#     ]

# LIST_VARS_CONJUNCTION = [
#     ['gridloc'], # list of str, vars to take conjunction over
#     ['shape_oriented'] # list of str, vars to take conjunction over
# ]

# LIST_VAR = [
#     "chunk_within_rank",
#     "chunk_within_rank",
#     "shape_oriented",
#     "gridloc",
#     "stroke_index",
#     ]

# LIST_VARS_CONJUNCTION = [
#     ['gridloc', 'chunk_rank'],
#     ['stroke_index', 'gridloc', 'chunk_rank'],
#     ['gridloc', 'chunk_within_rank'],
#     ['stroke_index', 'shape_oriented'],
#     ['gridloc', 'shape_oriented']
# ]

# LIST_VAR = [
#     "gridloc",
#     "stroke_index",
#     ]

# LIST_VARS_CONJUNCTION = [
#     ['stroke_index', 'shape_oriented'],
#     ['gridloc', 'shape_oriented']
# ]

LIST_VAR = [
    # "chunk_within_rank",
    "supervision_stage_concise",
    ]

LIST_VARS_CONJUNCTION = [
    # ['stroke_index', 'gridloc', 'chunk_rank', 'supervision_stage_concise'],
    ['stroke_index', 'gridloc', 'chunk_rank', 'chunk_within_rank'],
]


assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)

# %matplotlib inline
# to help debug if times are misaligned.
MS = load_mult_session_helper(DATE, animal)

# for session in range(len(MS.SessionsList)):
for session in LIST_SESSIONS:
    sn = MS.SessionsList[session]

    SAVEDIR = f"/gorilla1/analyses/recordings/main/chunks_modulation/{animal}-{DATE}-sess_{session}"
    os.makedirs(SAVEDIR, exist_ok=True)


    ###################################
    EPOCH = "AnBmTR"

    D = sn.Datasetbeh
    from pythonlib.dataset.modeling.discrete import rules_map_rule_to_ruledict_extract_auto
    map_epoch_rulestring = rules_map_rule_to_ruledict_extract_auto(D)
    # map shapes to their abstract "role"
    # epoch = D.Dat.iloc[ind]["epoch_orig"]
    ruledict = map_epoch_rulestring[EPOCH]
    shapes_in_order_of_roles = ruledict["params_good"][0]
    map_shape_to_rank = {}
    for i, shape in enumerate(shapes_in_order_of_roles):
        map_shape_to_rank[shape] = i
    print(map_shape_to_rank)

    # for each trial, determine the chunks, and assign it to each stroke
    PLOT = False
    for ind in range(len(D.Dat)):
        tokens = D.taskclass_tokens_extract_wrapper(ind, plot=PLOT, return_as_tokensclass=True)
        tokens.chunks_update_by_shaperank(map_shape_to_rank)



    #####################################
    from neuralmonkey.classes.snippets import _dataset_extract_prune_general
    list_superv_keep = ["off|1|rank|0", "off|0||0"]
    dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(sn, list_superv_keep=list_superv_keep)



    ####################################
    from neuralmonkey.classes.snippets import Snippets, extraction_helper

    list_features_modulation_append = ["chunk_rank", "chunk_within_rank", "supervision_stage_concise"]
    SP = extraction_helper(sn, "stroke", list_features_modulation_append, 
                           dataset_pruned_for_trial_analysis=dataset_pruned_for_trial_analysis, NEW_VERSION=True, 
                           PRE_DUR = PRE_DUR, POST_DUR = POST_DUR)
    SP.globals_update(n_min, lenient_allow_data_if_has_n_levels)

    # SAVE
    SP.save_v2(SAVEDIR)


    # if DEBUG:
    #     SP.Sites = SP.Sites[::15]
    #     print("new sites (subsampled): ", SP.Sites)


    # #######################
    # for var, vars_conjuction in zip(LIST_VAR, LIST_VARS_CONJUNCTION):
    #     SP.modulationgood_compute_plot_ALL(var, vars_conjuction, score_ver, SAVEDIR=SAVEDIR, 
    #                                        PRE_DUR_CALC=PRE_DUR_CALC, 
    #                                        POST_DUR_CALC=POST_DUR_CALC)
