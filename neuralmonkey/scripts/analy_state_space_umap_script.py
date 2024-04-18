"""
Goal: general purpose approach for plotting state space (scalar and trajc) variety of dim redutcion methods, all using
same plots.

This builds on analy_dpca_plot_script, which was specific for dpca

NOTEBOOK: 230227_snippets_statespace_tsne

"""


from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
import sys
import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import extract_single_pa
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels
from neuralmonkey.analyses.state_space_good import dimredgood_nonlinear_embed_data
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

fr_normalization_method = "across_time_bins"
# LIST_METHOD = ["umap", "tsne"]
LIST_METHOD = ["umap"]
# LIST_UMAP_N_NEIGHBORS = [10, 30, 50]
exclude_bad_areas = True
SPIKES_VERSION = "kilosort_if_exists" # since Snippets not yet extracted for ks
combine_into_larger_areas = False
list_time_windows = [(-0.6, 0.6)]
pca_frac_var_keep = 0.8
n_min_per_levo= 5
HACK_CHUNKS = True

# LIST_UMAP_N_NEIGHBORS = [30]
# TWIND_STROKE = [-0.1, 0.1]

# # 3/24/24 - Params, trying to get more global structure, for chunk variables.
# LIST_UMAP_N_NEIGHBORS = [45]
# TWIND_STROKE = [-0.15, 0.15]

# 3/25/24 - Params, less influence of context
LIST_UMAP_N_NEIGHBORS = [45]
# TWIND_STROKE = [-0.1, 0.1]

# 4/??/24 - Trying to look at in between strokes (e..g, syntax represnetation here?)
TWIND_STROKE = [-0.4, -0.1]
#
def get_list_twind_overall(ev):
    """ Get list of twindows that will iterate over, each time doing one analysis.
    """
    if ev=="03_samp":
        list_twind_overall = [
            [0.2, 0.6]
        ]
    elif ev in ["06_on_strokeidx_0", "00_stroke"]:
        # list_twind_overall = [
        #     [-0.35, -0.05],
        #     [0, 0.3]
        # ]
        list_twind_overall = [
            TWIND_STROKE,
        ]
        # list_twind_overall = [
        #     [-0.3, -0.1],
        # ]
    elif ev == "04_go_cue":
        list_twind_overall = [
            [-0.45, -0.05]
        ]
    else:
        print(ev)
        assert False
    return list_twind_overall

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])
    do_combine = int(sys.argv[3]) # 0, 1 (combine trial and stroke PA)

    if do_combine == 1:
        # COMBINE trial and stroke
        dir_suffix = sys.argv[4]
        question = None
        q_params = None
        combine_trial_and_stroke = True
        which_level = None
        q_params = {
            "effect_vars": ["seqc_0_shape", "seqc_0_loc"]
        }
        if dir_suffix=="PIG":
            question_trial = "PIG_BASE_trial"
            question_stroke = "PIG_BASE_stroke"
        elif dir_suffix=="CHAR":
            question_trial = "CHAR_BASE_trial"
            question_stroke = "CHAR_BASE_stroke"
        else:
            print(dir_suffix)
            assert False

        # Since chars have small N per shape, do clumping.
        HACK_RENAME_SHAPES = "CHAR" in question_trial
    else:
        # DONT COMBINE, use questions.
        question = sys.argv[4]
        combine_trial_and_stroke = False
        which_level = sys.argv[5]
        # which_level = "trial"
        dir_suffix = question
        if which_level=="trial":
            # events_keep = ["03_samp", "04_go_cue", "06_on_strokeidx_0"]
            events_keep = ["03_samp", "04_go_cue", "06_on_strokeidx_0"]
        elif which_level=="stroke":
            events_keep = ["00_stroke"]
        else:
            print(which_level)
            assert False

        # Load q_params
        from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
        q_params = rsagood_questions_dict(animal, date, question)[question]

        HACK_RENAME_SHAPES = "CHAR" in question

    # if len(sys.argv)>=6:
    #     plots_do = str(sys.argv[5]) # 1111 means do all 4 plots. 1101 means skip 3rd.
    #     assert len(plots_do)==4
    #     PLOT_1 = plots_do[0]=="1"
    #     PLOT_2 = plots_do[1]=="1"
    #     PLOT_3 = plots_do[2]=="1"
    #     PLOT_4 = plots_do[3]=="1"
    # else:
    #     PLOT_1, PLOT_2, PLOT_3, PLOT_4 = True, True, True, True

    if len(sys.argv)>=7:
        SINGLE_TWIND = int(sys.argv[6])
    else:
        SINGLE_TWIND = 0

    ############### PARAMS
    # N_MIN_TRIALS = 5
    # EVENTS_IGNORE = ["04_go_cue"] # To reduce plots
    EVENTS_IGNORE = [] # To reduce plots

    # SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/STATE_SPACE_GENERAL/{animal}-{date}"
    SAVEDIR_ANALYSIS = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/STATE_SPACE_GENERAL/{animal}-{date}/{dir_suffix}-{which_level}-combine_{do_combine}-combarea_{combine_into_larger_areas}"
    os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

    if question=="SP_novel_shape":
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=True
    else:
        TASK_KIND_RENAME_AS_NOVEL_SHAPE=False

    ########################################## RUN
    if combine_trial_and_stroke:
        from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper_combine_trial_strokes
        DFallpa = dfallpa_extraction_load_wrapper_combine_trial_strokes(animal, date, question_trial,
                                                                           question_stroke,
                                                    list_time_windows,
                                                   combine_into_larger_areas = combine_into_larger_areas,
                                                   exclude_bad_areas=exclude_bad_areas,
                                                    SPIKES_VERSION=SPIKES_VERSION,
                                                    HACK_RENAME_SHAPES = HACK_RENAME_SHAPES,
                                                   do_fr_normalization=fr_normalization_method)
    else:
        from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
        DFallpa = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows, which_level=which_level,
                                                  events_keep=events_keep,
                                                  combine_into_larger_areas=combine_into_larger_areas,
                                                  exclude_bad_areas=exclude_bad_areas, SPIKES_VERSION=SPIKES_VERSION,
                                                  HACK_RENAME_SHAPES=HACK_RENAME_SHAPES, fr_normalization_method=fr_normalization_method)

    ################################ PLOTS
    # list_br = DFallpa["bregion"].unique().tolist()
    # list_tw = DFallpa["twind"].unique().tolist()
    # list_ev = DFallpa["event"].unique().tolist()
    #
    # # Prune events, to reduce size of plots.
    # list_ev = [ev for ev in list_ev if ev not in EVENTS_IGNORE]


    #################### HACKY, for wl=="trial", extract motor and other variables for first stroke.
    from neuralmonkey.classes.population_mult import dfallpa_preprocess_vars_conjunctions_extract
    dfallpa_preprocess_vars_conjunctions_extract(DFallpa, which_level=which_level)

    from neuralmonkey.analyses.state_space_good import trajgood_construct_df_from_raw, trajgood_plot_colorby_splotby, trajgood_plot_colorby_splotby_scalar
    from pythonlib.tools.plottools import savefig
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
    import os

    ### PARAMS
    tbin_dur = 0.1
    tbin_slide = 0.1
    if SINGLE_TWIND==1:
        tbin_dur = 0.4
        tbin_slide = 0.4
    reshape_method = "trials_x_chanstimes"
    RES = []

    for i, row in DFallpa.iterrows():
        pa = row["pa"]
        br = row["bregion"]
        wl = row["which_level"]
        ev = row["event"]
        tw = row["twind"]

        list_twind_overall = get_list_twind_overall(ev)

        # if br=="PMv_m" and ev=="06_on_strokeidx_0":
        for twind_overall in list_twind_overall:

            # Extract data
            X, _, pathis, _ = pa.dataextract_state_space_decode_flex(twind_overall, tbin_dur, tbin_slide, reshape_method,
                                                               pca_reduce=True, pca_frac_var_keep=pca_frac_var_keep)
            dflab = pathis.Xlabels["trials"]

            # Extract strokes
            if which_level=="trial":
                # Extract the first stroke...
                STROKES_BEH = [tk.Tokens[0]["Prim"].Stroke() for tk in dflab["Tkbeh_stkbeh"]]
                STROKES_TASK = [tk.Tokens[0]["Prim"].Stroke() for tk in dflab["Tkbeh_stktask"]]
            elif which_level=="stroke":
                STROKES_BEH = [stk() for stk in dflab["Stroke"]]
                STROKES_TASK = [tk.Tokens[0]["Prim"].Stroke() for tk in dflab["TokTask"]]
            else:
                assert False

            # Embed data
            for METHOD in LIST_METHOD:
                if METHOD == "umap":
                    list_tsne_perp = ["auto" for _ in range(len(LIST_UMAP_N_NEIGHBORS))]
                    list_umap_n_neighbors = LIST_UMAP_N_NEIGHBORS
                else:
                    assert False, "add LIST PARAMS for tsne"

                for tsne_perp, umap_n_neighbors in zip(list_tsne_perp, list_umap_n_neighbors):
                    Xredu, _ = dimredgood_nonlinear_embed_data(X, METHOD=METHOD, n_components=2, tsne_perp=tsne_perp, umap_n_neighbors=umap_n_neighbors)
                    # Xredu = dimredgood_nonlinear_embed_data(X, METHOD=METHOD, n_components=2, tsne_perp="auto", umap_n_neighbors="auto")

                    ##### Plot scalars
                    savedir = f"{SAVEDIR_ANALYSIS}/{br}-{ev}-twind={'_'.join([str(t) for t in twind_overall])}-METHOD={METHOD}-T_perp={tsne_perp}-U_nn={umap_n_neighbors}"
                    print(savedir)
                    os.makedirs(savedir, exist_ok=True)

                    # Save params
                    from pythonlib.tools.expttools import writeDictToTxt
                    params_this = {
                        "exclude_bad_areas":exclude_bad_areas,
                        "TASK_KIND_RENAME_AS_NOVEL_SHAPE":TASK_KIND_RENAME_AS_NOVEL_SHAPE,
                        "tbin_dur":tbin_dur,
                        "tbin_slide":tbin_slide,
                        "reshape_method":reshape_method,
                        "X_dim_before_redu_dim": X.shape,
                        "pca_frac_var_keep":pca_frac_var_keep
                    }
                    writeDictToTxt(params_this, f"{savedir}/params.txt")

                    if True:
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

                        if question in ["RULESW_BASE_stroke"]:
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

                        if question in ["RULESW_ANBMCK_COLRANK_STROKE"]:
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

                    else:
                        # Older, which I used for PIG strokes... SHould replace with avbove?

                        #### DECIDE WHAT VARIABLES TO PLOT
                        # List of plots to make
                        list_var_color_var_subplot = []
                        if which_level=="trial":
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

                        elif which_level=="stroke":
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
                            print(which_level)
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


                    if False:
                        ##### Create meaned data to plot
                        effect_vars = ["seqc_0_shape", "seqc_0_loc", "shape_is_novel_all", "seqc_0_shapesem", "seqc_0_locon_binned", "seqc_0_angle_binned"]
                        pathis_meaned = pathis.slice_and_agg_wrapper("trials", effect_vars)

                    for var_color, var_subplot in list_var_color_var_subplot:
                        from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER
                        trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu, dflab, var_color, savedir,
                                                 vars_subplot=var_subplot, list_dims=[(0,1)],
                                                 STROKES_BEH=STROKES_BEH, STROKES_TASK=STROKES_TASK, n_min_per_levo=n_min_per_levo)

                    ########################## grouping trials by the duration of bins between chunks
                    VARS_GAP_DUR_BINS = ["chunkgap_(0, 1)_durbin", "chunkgap_(1, 2)_durbin"]
                    list_subplot = [
                        ["taskcat_by_rule"],
                        ["taskcat_by_rule", "behseq_shapes_clust"],
                        ["taskcat_by_rule", "behseq_locs_clust"],
                        ["taskcat_by_rule", "behseq_locs_diff_clust"],
                        ]
                    for t in VARS_GAP_DUR_BINS:
                        if t in dflab.columns:
                            list_subplot.append(["taskcat_by_rule", "behseq_locs_clust", t])

                    if HACK_CHUNKS:
                        # AnBmCk
                        for var_subplot in list_subplot:
                            var_color = ["chunk_rank", "chunk_within_rank"]
                            # var_subplot = ["taskcat_by_rule", "behseq_shapes", "task_kind"]
                            overlay_mean_var_color = "chunk_rank"
                            connect_means_with_line = True
                            dflab = append_col_with_grp_index(dflab, var_color, "_tmp")
                            connect_means_with_line_levels = sorted(dflab["_tmp"].unique().tolist())

                            # Only keep cases that have low and high bin durations (otherwise too many plots).
                            FOUND=False
                            for _var_effect in VARS_GAP_DUR_BINS:
                                if _var_effect in var_subplot:
                                    assert FOUND==False, "only one of the vars in VARS_GAPDUR.. should be in var_supblot."
                                    _vars_others = [v for v in var_subplot if not v==_var_effect]
                                    dflab_this, _ = extract_with_levels_of_conjunction_vars_helper(dflab, _var_effect, _vars_others)
                                    Xredu_this = Xredu[dflab_this["_index"].tolist(), :]
                                    FOUND=True
                            if not FOUND:
                                Xredu_this = Xredu
                                dflab_this = dflab
                            trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu_this, dflab_this, var_color, savedir,
                                                     vars_subplot=var_subplot, list_dims=[(0,1)], connect_means_with_line=connect_means_with_line,
                                                                  connect_means_with_line_levels=connect_means_with_line_levels,
                                                                         overlay_mean=True, overlay_mean_var_color=overlay_mean_var_color,
                                                                         alpha=0.3)

                    # Split by epochs
                    list_subplot = [
                        ["taskcat_by_rule", "epochset", "epoch"],
                        ["taskcat_by_rule", "behseq_shapes_clust", "epochset", "epoch"],
                        ["taskcat_by_rule", "behseq_locs_clust", "epochset", "epoch"],
                        ["taskcat_by_rule", "behseq_locs_diff_clust", "epochset", "epoch"],
                        ]
                    for t in VARS_GAP_DUR_BINS:
                        if t in dflab.columns:
                            list_subplot.append(["taskcat_by_rule", "behseq_locs_clust", t])
                    if HACK_CHUNKS and question in ["RULESW_BASE_stroke", "RULESW_ANBMCK_DIR_STROKE", "RULESW_ANBMCK_ABN_STROKE"]:
                        for var_subplot in list_subplot:
                            var_color = ["chunk_rank", "chunk_within_rank"]
                            # var_subplot = ["taskcat_by_rule", "behseq_shapes", "task_kind", "epochset", "epoch"]
                            overlay_mean_var_color = "chunk_rank"
                            connect_means_with_line = True
                            dflab = append_col_with_grp_index(dflab, var_color, "_tmp")
                            connect_means_with_line_levels = sorted(dflab["_tmp"].unique().tolist())

                            # Only keep cases that have low and high bin durations (otherwise too many plots).
                            FOUND=False
                            for _var_effect in VARS_GAP_DUR_BINS:
                                if _var_effect in var_subplot:
                                    assert FOUND==False, "only one of the vars in VARS_GAPDUR.. should be in var_supblot."
                                    _vars_others = [v for v in var_subplot if not v==_var_effect]
                                    dflab_this, _ = extract_with_levels_of_conjunction_vars_helper(dflab, _var_effect, _vars_others)
                                    Xredu_this = Xredu[dflab_this["_index"].tolist(), :]
                                    FOUND=True
                            if not FOUND:
                                Xredu_this = Xredu
                                dflab_this = dflab
                            trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu_this, dflab_this, var_color, savedir,
                                                     vars_subplot=var_subplot, list_dims=[(0,1)], connect_means_with_line=connect_means_with_line,
                                                                  connect_means_with_line_levels=connect_means_with_line_levels,
                                                                         overlay_mean=True, overlay_mean_var_color=overlay_mean_var_color,
                                                                         alpha=0.3)

                    if HACK_CHUNKS and question in ["RULESW_ANBMCK_COLRANK_STROKE"]:
                        tmp = list_subplot + [["taskcat_by_rule", "behseq_shapes_clust", "epoch_rand"],
                                                    ["taskcat_by_rule", "behseq_locs_clust", "epoch_rand"],
                                                    ["taskcat_by_rule", "behseq_locs_diff_clust", "epoch_rand"]]
                        for var_subplot in tmp:
                            var_color = ["chunk_rank", "chunk_within_rank"]
                            # var_subplot = ["taskcat_by_rule", "behseq_shapes", "task_kind", "epochset", "epoch"]
                            overlay_mean_var_color = "chunk_rank"
                            connect_means_with_line = True
                            dflab = append_col_with_grp_index(dflab, var_color, "_tmp")
                            connect_means_with_line_levels = sorted(dflab["_tmp"].unique().tolist())

                            # Only keep cases that have low and high bin durations (otherwise too many plots).
                            FOUND=False
                            for _var_effect in VARS_GAP_DUR_BINS:
                                if _var_effect in var_subplot:
                                    assert FOUND==False, "only one of the vars in VARS_GAPDUR.. should be in var_supblot."
                                    _vars_others = [v for v in var_subplot if not v==_var_effect]
                                    dflab_this, _ = extract_with_levels_of_conjunction_vars_helper(dflab, _var_effect, _vars_others)
                                    Xredu_this = Xredu[dflab_this["_index"].tolist(), :]
                                    FOUND=True
                            if not FOUND:
                                Xredu_this = Xredu
                                dflab_this = dflab
                            trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu_this, dflab_this, var_color, savedir,
                                                     vars_subplot=var_subplot, list_dims=[(0,1)], connect_means_with_line=connect_means_with_line,
                                                                  connect_means_with_line_levels=connect_means_with_line_levels,
                                                                         overlay_mean=True, overlay_mean_var_color=overlay_mean_var_color,
                                                                         alpha=0.3)


