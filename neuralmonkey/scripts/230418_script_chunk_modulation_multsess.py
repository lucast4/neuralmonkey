""" 
New ANOVA, aligned to strokes.
Load SP previously saved by 230418_script_chunk_modulation
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os


DATE = 221020
animal = "Pancho"
DEBUG = False
# score_ver='r2smfr_zscore'
score_ver='fracmod_smfr_minshuff'
# PRE_DUR_CALC = -0.1
# POST_DUR_CALC = 0.25
PRE_DUR_CALC = -0.25
POST_DUR_CALC = 0.25

###################################
LIST_VAR = [
    "epoch",
    # 'CTXT_shape_next',
    # 'CTXT_loc_next',
    # 'CTXT_shape_prev',
    # 'CTXT_loc_prev',
    # "chunk_within_rank",
    # "chunk_within_rank",
    # "chunk_within_rank",
    # "chunk_within_rank",
    # "chunk_within_rank",
    # "supervision_stage_concise",
    # "stroke_index",
    # "stroke_index",
    # "shape_oriented",
    # "gridloc",
    ]
LIST_VARS_CONJUNCTION = [
    # ['CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next', 'gridloc', 'shape_oriented'],
    ['stroke_index', 'CTXT_loc_prev', 'CTXT_shape_prev', 'gridloc', 'shape_oriented'],
    # ['shape_oriented', 'gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next'],
    # ['shape_oriented', 'gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_shape_next'],
    # ['shape_oriented', 'gridloc', 'CTXT_loc_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    # ['shape_oriented', 'gridloc', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev'],
    # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev', 'supervision_stage_concise'],
    # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'supervision_stage_concise'],
    # ['gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev', 'supervision_stage_concise'],
    # ['gridloc', 'chunk_rank', 'CTXT_loc_prev', 'supervision_stage_concise'],
    # ['stroke_index', 'gridloc', 'chunk_rank', 'CTXT_loc_prev', 'CTXT_shape_prev'],
    # ['gridloc', 'shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev'],
    # ['gridloc', 'shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next'],
    # ['gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
    # ['shape_oriented', 'CTXT_loc_prev', 'CTXT_shape_prev', 'CTXT_loc_next', 'CTXT_shape_next'],
]

######################################## RUN
assert len(LIST_VAR)==len(LIST_VARS_CONJUNCTION)

# %matplotlib inline
# to help debug if times are misaligned.
MS = load_mult_session_helper(DATE, animal)
from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
SAVEDIR = "/gorilla1/analyses/recordings/main/chunks_modulation"
SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, SAVEDIR=SAVEDIR)

if DEBUG:
    SP.Sites = SP.Sites[::20]
    print("new sites (subsampled): ", SP.Sites)

#######################
for var, vars_conjuction in zip(LIST_VAR, LIST_VARS_CONJUNCTION):
    # try:
    SP.modulationgood_compute_plot_ALL(var, vars_conjuction, 
            score_ver, SAVEDIR=SAVEDIR_ALL, 
            PRE_DUR_CALC=PRE_DUR_CALC, 
            POST_DUR_CALC=POST_DUR_CALC)
    # except Exception as err:
    #     print("!! FAILED: ", var, vars_conjuction)
    #     pass