""" New ANOVA, aligned to strokes.
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os


DATE = 221020
animal = "Pancho"
# LIST_SESSIONS = [0, 1, 2, 3]
LIST_SESSIONS = None
SCORE_SEQUENCE_VER = "matlab"
ASSIGN_CHUNK_INFO = False

####################################
PRE_DUR = -0.6
POST_DUR = 0.6
lenient_allow_data_if_has_n_levels=2
n_min = 5

# score_ver='r2smfr_minshuff'
PRE_DUR_CALC = -0.25
POST_DUR_CALC = 0.25
# score_ver='r2smfr_minshuff'
score_ver='fracmod_smfr_minshuff'
DEBUG = False # runs fast
SKIP_PLOTTING = True

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
    "epoch",
    ]
LIST_VARS_CONJUNCTION = [
    ['gridloc', 'shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev'],
]



assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)

# %matplotlib inline
# to help debug if times are misaligned.
MS = load_mult_session_helper(DATE, animal)

# for session in range(len(MS.SessionsList)):
if LIST_SESSIONS is None:
    LIST_SESSIONS = range(len(MS.SessionsList))

for session in LIST_SESSIONS:
    sn = MS.SessionsList[session]

    SAVEDIR = f"/gorilla1/analyses/recordings/main/chunks_modulation/{animal}-{DATE}-sess_{session}"
    os.makedirs(SAVEDIR, exist_ok=True)


    ###################################
    D = sn.Datasetbeh
    if SCORE_SEQUENCE_VER=="parses":
        D.grammar_successbinary_score_parses()
    elif SCORE_SEQUENCE_VER=="matlab":
        D.grammar_successbinary_score_matlab()

    ###################################
    if ASSIGN_CHUNK_INFO:
        list_epoch = D.Dat["epoch"].unique().tolist()
        if len(list_epoch)==1:
            EPOCH = list_epoch[0]
        else:
            print(list_epoch)
            assert False, "how deal with this?"

        from pythonlib.dataset.modeling.discrete import rules_map_rule_to_ruledict_extract_auto
        map_epoch_rulestring = rules_map_rule_to_ruledict_extract_auto(D)
        map_epoch_rulestring

        # map shapes to their abstract "role"
        # epoch = D.Dat.iloc[ind]["epoch_orig"]
        ruledict = map_epoch_rulestring[EPOCH]

        if ruledict["categ"]=="ss" and ruledict["subcat"]=="rank":
            shapes_in_order_of_roles = ruledict["params_good"]
        else:
            shapes_in_order_of_roles = ruledict["params_good"][0]
        map_shape_to_rank = {}
        for i, shape in enumerate(shapes_in_order_of_roles):
            map_shape_to_rank[shape] = i
        print(map_shape_to_rank)

        # clean up ranks. They should reflect rank only within the set of shapes that are ever presented otgether
        # e.g, if ranks are {'line-8-4-0': 0, 'line-11-1-0': 1, 'line-8-3-0': 2, 'line-11-2-0': 3},
        # but line-8 are never presented with line-11, then really want:
        # Update shape ranks:  {'line-8-4-0': 0, 'line-11-1-0': 0, 'line-8-3-0': 1, 'line-11-2-0': 1}

        list_shape_sets = []
        for i in range(len(D.Dat)):
            shapes_used = D.taskclass_shapes_extract(i)
            ADDED = False
            for SET in list_shape_sets:
                if any([sh in SET for sh in shapes_used]):
                    # Then at least one shape in shapes_useed is par tof this SET
                    ADDED = True
                    SET.extend(shapes_used)
            if not ADDED:
                # start a new set
                list_shape_sets.append(shapes_used)
            
            # prune to just unique
            list_shape_sets = [list(set(SET)) for SET in list_shape_sets]

        # check that each shape is in a unqque set
        this = [xx for x in list_shape_sets for xx in x]
        assert len(list(set(this))) == len(this), "bug in code"

        print(list_shape_sets)

        # for each shape, reassign it a new rank based on only the shapes in its set.
        import numpy as np
        list_shape_sets_ranks = [[map_shape_to_rank[sh] for sh in SET] for SET in list_shape_sets]
        for i, set_idx in enumerate(list_shape_sets_ranks):
            print(set_idx, np.argsort(set_idx))
            list_shape_sets_ranks[i] = np.argsort(set_idx).tolist()
            
        # map back the rank to the shape
        for i, SET in enumerate(list_shape_sets):
            for j, shape in enumerate(SET):
                rank_within_set = list_shape_sets_ranks[i][j]
                map_shape_to_rank[shape] = rank_within_set
                
        print("Updated shape ranks: ", map_shape_to_rank)

        # for each trial, determine the chunks, and assign it to each stroke
        PLOT = False
        for ind in range(len(D.Dat)):
            tokens = D.taskclass_tokens_extract_wrapper(ind, plot=PLOT, return_as_tokensclass=True)
            tokens.chunks_update_by_shaperank(map_shape_to_rank)



    #####################################
    from neuralmonkey.classes.snippets import _dataset_extract_prune_general

    list_superv_keep = ["off|1|rank|0", "off|0||0", "off|1|solid|0"]
    preprocess_steps_append = ["correct_sequencing_binary_score", "one_to_one_beh_task_strokes"]
    dataset_pruned_for_trial_analysis = _dataset_extract_prune_general(sn, list_superv_keep=list_superv_keep,
                                                                      preprocess_steps_append=preprocess_steps_append)


    ####################################
    from neuralmonkey.classes.snippets import Snippets, extraction_helper

    list_features_modulation_append = ["epoch", "CTXT_loc_prev", "CTXT_shape_prev", "CTXT_loc_next", 
        "CTXT_shape_next", "supervision_stage_concise"]
    if ASSIGN_CHUNK_INFO:
        list_features_modulation_append = list_features_modulation_append + ["chunk_rank", "chunk_within_rank"]
    SP = extraction_helper(sn, "stroke", list_features_modulation_append, 
                           dataset_pruned_for_trial_analysis=dataset_pruned_for_trial_analysis, NEW_VERSION=True, 
                           PRE_DUR = PRE_DUR, POST_DUR = POST_DUR)
    SP.globals_update(n_min, lenient_allow_data_if_has_n_levels, PRE_DUR_CALC, POST_DUR_CALC)

    # SAVE
    SP.save_v2(SAVEDIR)

    if not SKIP_PLOTTING:

        if DEBUG:
            SP.Sites = SP.Sites[::15]
            print("new sites (subsampled): ", SP.Sites)


        #######################
        for var, vars_conjuction in zip(LIST_VAR, LIST_VARS_CONJUNCTION):
            SP.modulationgood_compute_plot_ALL(var, vars_conjuction, score_ver, SAVEDIR=SAVEDIR, 
                                               PRE_DUR_CALC=PRE_DUR_CALC, 
                                               POST_DUR_CALC=POST_DUR_CALC)
