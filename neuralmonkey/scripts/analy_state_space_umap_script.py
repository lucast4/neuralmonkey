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
            X, _, pathis, _, _ = pa.dataextract_state_space_decode_flex(twind_overall, tbin_dur, tbin_slide, reshape_method,
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

                    ### Get variables to plot
                    from neuralmonkey.metadat.analy.anova_params import params_getter_umap_vars
                    list_var_color_var_subplot = params_getter_umap_vars(question)

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


