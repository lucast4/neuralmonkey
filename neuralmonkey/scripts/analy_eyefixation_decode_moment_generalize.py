"""
Decoder, asking about generalization of seqc_0_shape space to shape-fixation spce.
Doing it using moment decoding, which allows more fine measures of decoding strenght.
"""

from pythonlib.tools.plottools import savefig
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys
import seaborn as sns
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from pythonlib.tools.snstools import rotateLabel
from pythonlib.tools.pandastools import aggregGeneral, stringify_values
from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
from neuralmonkey.analyses.decode_moment import pipeline_get_dataset_params_from_codeword, train_decoder_helper, _test_decoder_helper


def plot_all(DFallpa, bregion, SAVEDIR, plot_each_other_var=True):
    """
    All plots
    PARAMS:
    - plot_each_other_var, bool, if False, skips the plots for each specific other_var. useful if you just
    want to get to summary plots quickly.
    """
    ########## TRAIN DECODER
    PLOT_DECODER = True
    n_min_per_var = 6
    VAR_EARLY_LATE = "is-first-macrosaccade"
    savedir_this = f"{SAVEDIR}/decoder_training"
    os.makedirs(savedir_this, exist_ok=True)

    # Train a single decoder
    # event_train, twind_train, filterdict_train, _, which_level_train = pipeline_get_dataset_params_from_codeword(train_dataset_name)
    if False: # If just have loaded fixation data
        event_train = "fixon_preparation"
        twind_train = (-0.4, 0.4)
        filterdict_train = {
            "FEAT_num_strokes_task":[1],
        }
        which_level_train = "flex"
    else:
        # Better, train on SP data
        event_train = "03_samp"
        twind_train = (0.05, 1.2)
        filterdict_train = {
            "FEAT_num_strokes_task":[1],
        }
        which_level_train = "trial"

    var_train = "seqc_0_shape"
    _, Dc = train_decoder_helper(DFallpa, bregion, var_train, event_train, 
                            twind_train, PLOT=PLOT_DECODER, include_null_data=False,
                            n_min_per_var=n_min_per_var, filterdict_train=filterdict_train,
                            which_level=which_level_train, decoder_method_index=None,
                            savedir=savedir_this, do_upsample_balance=True, do_upsample_balance_fig_path_nosuff=None,
                            downsample_trials=False)

    ########## DECODER, TESGING ALL CONJUCNTIONS (CONTROLLING SEQC_0_SHAPE)
    # - get a conjucntion
    vars_others = [VAR_EARLY_LATE, "seqc_0_shape"]
    # vars_others = ["seqc_0_shape"]
    var_test = "shape-fixation"
    # list_twind_test = [(-0.3, 0), (0, 0.3), (0.3, 0.6)]
    list_twind_test = [(-0.4, -0.2), (-0.2, 0), (0, 0.2), (0.2, 0.4)]
    # list_twind_test = [(0, 0.3)]
    
    filterdict_test = {
        "FEAT_num_strokes_task":list(range(2,10)),
    }

    which_level_test = "flex"
    event_test = "fixon_preparation"

    savedir = f"{SAVEDIR}/vartest={var_test}-varsothers={vars_others}"
    os.makedirs(savedir, exist_ok=True)

    PAtest = extract_single_pa(DFallpa, bregion, None, which_level=which_level_test, event=event_test)
    plot_counts_heatmap_savepath = f"{savedir}/counts_vars_others.pdf"
    dict_pa = PAtest.slice_extract_with_levels_of_conjunction_vars_as_dictpa(var_test, vars_others,
                                                                             plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
    # dfscores["vars_others_grp"].unique()
    list_dfscores = []
    list_dfclasses = []
    list_dfaccuracy =[]
    accuracy_res = []
    metrics_res = []
    for grp, pa_test in dict_pa.items():
        
        savedir_this = f"{savedir}/grp={grp}"
        os.makedirs(savedir, exist_ok=True)
        
        if plot_each_other_var:
            PLOT=True
        else:
            PLOT=False
        dfscores, pa_test = _test_decoder_helper(Dc, pa_test, var_test, list_twind_test, subtract_baseline=False, 
                                        PLOT=PLOT, savedir=savedir_this,
                                        prune_labels_exist_in_train_and_test=True,
                                        filterdict_test=filterdict_test)
        dfscores["vars_others_grp"] = [grp for _ in range(len(dfscores))]
        dfscores["var_test"] = var_test

        # Condition dfscores BEFORE concat
        from neuralmonkey.analyses.decode_moment import analy_eyefixation_dfscores_condition
        dfscores = analy_eyefixation_dfscores_condition(dfscores, pa_test.Xlabels["trials"], var_test)

        assert vars_others == [VAR_EARLY_LATE, "seqc_0_shape"], "assumes so for this line"
        dfscores[VAR_EARLY_LATE] = [grp[0] for grp in dfscores["vars_others_grp"]]

        # Plot test scores, using score_norm
        if plot_each_other_var:
            sdir = f"{savedir_this}/test_plots-var_score=score_norm"
            os.makedirs(sdir, exist_ok=True)
            Dc.scalar_score_df_plot_summary(dfscores, sdir, var_score="score_norm")

        # Collect scores, one for each (grp, twind)
        map_twind_to_score, _, _ = Dc.scalar_score_compute_metric(dfscores)
        for twind, score in map_twind_to_score.items():
            metrics_res.append({
                "score":score,
                "twind":twind,
                "grp":grp,
            })

        # Get classification accuracy scores.
        for var_score in ["score", "score_norm"]:
            if plot_each_other_var:
                sdir = f"{savedir_this}/classifier-test-var_score={var_score}"
                os.makedirs(sdir, exist_ok=True)
            else:
                sdir = None
            score, score_adjusted, dfclasses, dfaccuracy = Dc.scalar_score_convert_to_classification_accuracy(dfscores, 
                                                                                                            var_score=var_score, 
                                                                                                            plot_savedir=sdir)

            dfclasses["vars_others_grp"] = [grp for _ in range(len(dfclasses))]
            dfclasses["var_test"] = var_test
            dfclasses["var_score_for_classify"] = var_score

            dfaccuracy["vars_others_grp"] = [grp for _ in range(len(dfaccuracy))]
            dfaccuracy["var_test"] = var_test
            dfaccuracy["var_score_for_classify"] = var_score

            list_dfscores.append(dfscores)
            list_dfclasses.append(dfclasses)
            list_dfaccuracy.append(dfaccuracy)
            accuracy_res.append({
                "score":score,
                "score_adjusted":score_adjusted,
                "grp":grp,
            })

    ### COLLECT data and save
    dfmetrics = pd.DataFrame(metrics_res)
    DFSCORES = pd.concat(list_dfscores).reset_index(drop=True)
    pd.to_pickle(dfmetrics, f"{savedir}/dfmetrics.pkl")
    pd.to_pickle(DFSCORES, f"{savedir}/DFSCORES.pkl")

    # Plot metrics
    fig = sns.catplot(data=dfmetrics, x="twind", y="score", hue="grp", kind="point")
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/metrics-1.pdf")

    fig = sns.catplot(data=dfmetrics, x="twind", y="score", kind="bar")
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/metrics-2.pdf")

    plt.close("all")

    ### Summary of continuous scores
    savedir_this = f"{savedir}/ALL"
    _plot_all_dfscores(Dc, DFSCORES, savedir_this, VAR_EARLY_LATE)
    
    # Plot for ealry and late
    DFSCORES_EARLY = DFSCORES[DFSCORES["is-first-macrosaccade"]==True].reset_index(drop=True)
    savedir_this = f"{savedir}/EARLY"
    os.makedirs(savedir_this, exist_ok=True)
    _plot_all_dfscores(Dc, DFSCORES_EARLY, savedir_this, VAR_EARLY_LATE)

    DFSCORES_LATE = DFSCORES[DFSCORES["is-first-macrosaccade"]==False].reset_index(drop=True)
    savedir_this = f"{savedir}/LATE"
    os.makedirs(savedir_this, exist_ok=True)
    _plot_all_dfscores(Dc, DFSCORES_LATE, savedir_this, VAR_EARLY_LATE)

    ### Classifier scores.
    DFACCURACY = pd.concat(list_dfaccuracy).reset_index(drop=True)
    DFCLASSES = pd.concat(list_dfclasses).reset_index(drop=True)
    _plot_all_classifier_accuracy(Dc, DFACCURACY, DFCLASSES, savedir)


def _plot_all_dfscores(Dc, DFSCORES, savedir, VAR_EARLY_LATE):

    from pythonlib.tools.pandastools import summarize_featurediff
    DFSCORES_AGG = aggregGeneral(DFSCORES, ["twind", "var_test", "vars_others_grp", "decoder_class", "pa_class"], values=["score", "score_norm"], 
                nonnumercols=["seqc_0_shape", VAR_EARLY_LATE, "same_class", "decoder_class_good", 'decoder_class_was_fixated', 
                              "decoder_idx", "pa_class_is_in_decoder", "decoder_class_semantic", "decoder_class_is_in_pa", "decoder_class_semantic_str"])
    DFSCORES_NOT_DRAWN = DFSCORES[DFSCORES["decoder_class_was_first_drawn"]==False].reset_index(drop=True)
    
    if False:
        from pythonlib.tools.pandastools import grouping_print_n_samples
        grouping_print_n_samples(DFSCORES, ["vars_others_grp", "decoder_class", "pa_class", "decoder_class_was_fixated"])
        grouping_print_n_samples(DFSCORES, ["vars_others_grp", "seqc_0_shape", "decoder_class", "decoder_class_was_first_drawn"])
    # Determine if decoder is same as seqc_0_shape
    # [decoder_class == pa_class for decoder_class in DFSCORES]

    # from pythonlib.tools.pandastools import stringify_values
    # dfscores_str = stringify_values(dfscores)
    # dfscores_str_success = stringify_values(dfscores_success)

    # Summary plots

    for dfscores_ver, dfscores_this in [
        ("trials", DFSCORES),
        ("agg", DFSCORES_AGG),
        ("trials_not_drawing_what_fixate", DFSCORES_NOT_DRAWN)]:
        
        dfscores_this = stringify_values(dfscores_this)
        if len(dfscores_this)>0:
            for var_score in ["score", "score_norm"]:
                
                tmp = ["decoder_class_good", "decoder_class_is_in_pa", "pa_class", 
                                    "pa_class_is_in_decoder", "twind", "seqc_0_shape", VAR_EARLY_LATE
                                    ]
                if dfscores_ver != "agg":
                    # THen is trial-level for data.
                    tmp += ["pa_idx", "trialcode"]

                ##### Compare early vs. late
                # - first, get a metric
                try:
                    dfsummary_this, _, _, _, _ = summarize_featurediff(dfscores_this, "same_class", [False, True], [var_score], tmp)
                    var_score_this= f"{var_score}-TrueminFalse"
                except Exception as err:
                    print(len(dfscores_this))
                    print(dfscores_this.columns)
                    print(var_score)
                    assert False, "why>?"

                savedirthis = f"{savedir}/COMBINED-{dfscores_ver}-{var_score}"
                os.makedirs(savedirthis, exist_ok=True)

                fig = sns.catplot(data=dfsummary_this, x=VAR_EARLY_LATE, y=var_score_this, hue="twind", alpha=0.2, jitter=True)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                rotateLabel(fig)
                savefig(fig, f"{savedirthis}/early_vs_late-1.pdf")

                fig = sns.catplot(data=dfsummary_this, x=VAR_EARLY_LATE, y=var_score_this, hue="twind", kind="bar", errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                rotateLabel(fig)
                savefig(fig, f"{savedirthis}/early_vs_late-2.pdf")

                fig = sns.catplot(data=dfsummary_this, x=VAR_EARLY_LATE, y=var_score_this, hue="seqc_0_shape", col="twind", kind="point", errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                rotateLabel(fig)
                savefig(fig, f"{savedirthis}/early_vs_late-3.pdf")

                #### All regular plots
                savedirthis = f"{savedir}/COMBINED-{dfscores_ver}-{var_score}"
                os.makedirs(savedirthis, exist_ok=True)
                Dc.scalar_score_df_plot_summary(dfscores_this, savedirthis, var_score=var_score)

    if False: # done above
        # Keep only cases where he is not fixating on what will draw first
        DFSCORES_THIS = DFSCORES[DFSCORES["decoder_class_was_first_drawn"]==False].reset_index(drop=True)
        for var_score in ["score", "score_norm"]:
            # Summary plots
            savedirthis = f"{savedir}/COMBINED-not_drawing_what_will_fixate-varscore-{var_score}"
            os.makedirs(savedirthis, exist_ok=True)

            # - heatmap
            Dc.scalar_score_df_plot_summary(DFSCORES_THIS, savedirthis, var_score=var_score)

    plt.close("all")

def _plot_all_classifier_accuracy(Dc, DFACCURACY, DFCLASSES, savedir):
    ### Summary of classifier accuracies
    chance = 1/(len(Dc.LabelsUnique))
    for var_score in ["score", "score_norm"]:
        
        savedirthis = f"{savedir}/COMBINED_classify-using_varscore={var_score}"
        os.makedirs(savedirthis, exist_ok=True)

        dfaccuracy_all = DFACCURACY[DFACCURACY["var_score_for_classify"] == var_score].reset_index(drop=True)
        dfclasses_all = DFCLASSES[DFCLASSES["var_score_for_classify"] == var_score].reset_index(drop=True)
        
        # Catplot, scores, versus grps

        fig = sns.catplot(data=dfaccuracy_all, x="label_actual", y="accuracy", hue="vars_others_grp", col="var_test", row="twind", kind="point")
        for ax in fig.axes.flatten():
            ax.axhline(chance, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{savedirthis}/accuracy-1.pdf")

        fig = sns.catplot(data=dfaccuracy_all, x="label_actual", y="accuracy", col="var_test", row="twind", kind="bar", errorbar=("ci", 68))
        for ax in fig.axes.flatten():
            ax.axhline(chance, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{savedirthis}/accuracy-2.pdf")

        fig = sns.catplot(data=dfaccuracy_all, x="vars_others_grp", y="accuracy", col="var_test", row="twind", kind="bar", errorbar=("ci", 68))
        for ax in fig.axes.flatten():
            ax.axhline(chance, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{savedirthis}/accuracy-3.pdf")

        fig = sns.catplot(data=dfaccuracy_all, x="twind", y="accuracy", col="var_test", kind="bar", errorbar=("ci", 68))
        for ax in fig.axes.flatten():
            ax.axhline(chance, color="k", alpha=0.5)
        rotateLabel(fig)
        savefig(fig, f"{savedirthis}/accuracy-3.pdf")

        plt.close("all")

        # Agg, then plot all, including heatmap
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        for norm_method in [None, "row_sub", "col_sub"]:
            fig = grouping_plot_n_samples_conjunction_heatmap(dfclasses_all, "label_actual", "label_predicted", ["vars_others_grp", "twind"], 
                                                            norm_method=norm_method)            
            savefig(fig, f"{savedirthis}/accuracy_heatmap-norm={norm_method}.pdf")

            fig = grouping_plot_n_samples_conjunction_heatmap(dfclasses_all, "label_actual", "label_predicted", ["twind"], 
                                                            norm_method=norm_method)            
            savefig(fig, f"{savedirthis}/accuracy_heatmap-norm={norm_method}.pdf")

        plt.close("all")

def plot_scatter_dissociate_draw_fix(DFallpa, bregion, SAVEDIR_ALL):
    """ Find cases where draw and fixation shapes are dissocaited, and compare the decoders for these two shapes, 
    and how this progresses over course of trial
    --> Early on, strong fixation effect. Later on, strong draw effect.
    #TODO: restrict to just trials that have both early and late fixations.
    """
    from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score
    from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score

    ########## TRAIN DECODER
    PLOT_DECODER = True
    n_min_per_var = 6

    # Train a single decoder
    # event_train, twind_train, filterdict_train, _, which_level_train = pipeline_get_dataset_params_from_codeword(train_dataset_name)
    # Better, train on SP data
    event_train = "03_samp"
    twind_train = (0.05, 1.2)
    filterdict_train = {
        "FEAT_num_strokes_task":[1],
    }
    which_level_train = "trial"
    var_train = "seqc_0_shape"

    var_test = "shape-fixation"
    # list_twind_test = [(-0.3, 0), (0, 0.3)]
    # list_twind_test = [(-0.35, -0.05)]
    which_level_test = "flex"
    event_test = "fixon_preparation"
    filterdict_test = {
        "FEAT_num_strokes_task":list(range(2,10)),
    }

    list_twind_test = [(0, 0.3), (-0.3, 0)]
    for twind_test in list_twind_test:
        SAVEDIR = f"{SAVEDIR_ALL}/twind={twind_test}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print("SAVING AT: ", SAVEDIR)

        savedir = f"{SAVEDIR}/decoder_training"
        os.makedirs(savedir, exist_ok=True)
        DFSCORES, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, 
                                            var_train, event_train, twind_train, filterdict_train,
                                            var_test, event_test, [twind_test], filterdict_test,
                                            savedir, prune_labels_exist_in_train_and_test=True, PLOT=PLOT_DECODER,
                                            which_level_train=which_level_train, which_level_test=which_level_test, 
                                            n_min_per_var=n_min_per_var,
                                            allow_multiple_twind_test=True)

        # Condition dfscores BEFORE concat
        from neuralmonkey.analyses.decode_moment import analy_eyefixation_dfscores_condition
        DFSCORES = analy_eyefixation_dfscores_condition(DFSCORES, PAtest.Xlabels["trials"], var_test)

        ###### PLOTS

        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        for var_color in ["event_idx_within_trial", "shape-macrosaccade-index"]:
            for only_if_trial_has_early_and_late_fixations in [False, True]:
                if only_if_trial_has_early_and_late_fixations:
                    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
                    dfscores, _ = extract_with_levels_of_conjunction_vars_helper(DFSCORES, "early_late_by_eidx", ["trialcode", "shape_draw_fix", "twind"])
                else:
                    dfscores = DFSCORES.copy()

                savedir = f"{SAVEDIR}/PLOTS-var={var_color}-only_if_trial_has_early_late={only_if_trial_has_early_and_late_fixations}"
                os.makedirs(savedir, exist_ok=True)

                fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "shape-fixation", "seqc_0_shape", [var_color])
                savefig(fig, f"{savedir}/counts-{var_color}-1.pdf")
                fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "shape-fixation", var_color, ["seqc_0_shape"])
                savefig(fig, f"{savedir}/counts-{var_color}-2.pdf")
                fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "seqc_0_shape", var_color, ["shape-fixation"])
                savefig(fig, f"{savedir}/counts-{var_color}-3.pdf")

                for PLOT_VER in ["scatter_decoder_vs_decoder", "point", "point_and_scatter", "regr"]:
                    dfregr, map_decoder_ver_to_col = _plot_scatter_dissociate_draw_fix(dfscores, PLOT_VER, twind_test, savedir, var_color =var_color)
                    plt.close("all")

                    dfregr["shape_draw_fix_diff"] = [row["shape_draw"] != row["shape_fix"] for _, row in dfregr.iterrows()]
                    dfregr = dfregr[dfregr["shape_draw_fix_diff"]==True].reset_index(drop=True)

                    if PLOT_VER == "regr":
                        from pythonlib.tools.snstools import rotateLabel

                        for y in ["intercept", "slope"]:
                            fig = sns.catplot(data=dfregr, x="decoder_ver", y=y, hue="shape_draw_fix", kind="point")
                            for ax in fig.axes.flatten():
                                ax.axhline(0)
                            rotateLabel(fig)
                            savefig(fig, f"{savedir}/regression-{y}-1.pdf")

                            fig = sns.catplot(data=dfregr, x="decoder_ver", y=y, kind="bar", errorbar=("ci", 68))
                            for ax in fig.axes.flatten():
                                ax.axhline(0)
                            rotateLabel(fig)
                            savefig(fig, f"{savedir}/regression-{y}-2.pdf")
                        plt.close("all")

                        ### Overlay of regression lines
                        # Make plot overlaying all regression lines
                        fig, axes = plt.subplots(2,2, figsize = (8,8), sharex=True, sharey=True)
                        ax = axes.flatten()[0]
                        for ax, decoder_ver in zip(axes.flatten(), dfregr["decoder_ver"].unique().tolist()):
                            ax.set_title(decoder_ver)
                            ax.set_xlabel(var_color)
                            ax.set_ylabel("decode")
                            for _, row in dfregr.iterrows():
                                if row["shape_draw"] != row["shape_fix"]:
                                    if row["decoder_ver"] == decoder_ver:
                                        xvals = [min(row["xvals"]), max(row["xvals"])]
                                        slope = row["slope"]
                                        yvals = [row["intercept"], row["intercept"] + row["slope"]*xvals[1]]
                                        ax.plot(xvals, yvals, "o-", alpha=0.8, color=map_decoder_ver_to_col[row["decoder_ver"]])
                        savefig(fig, f"{savedir}/regression-lines.pdf")
                        plt.close("all")

def _plot_scatter_dissociate_draw_fix(dfscores, PLOT_VER, twind, savedir, var_color = "event_idx_within_trial"):
    """
    twind = (0, 0.3)
    PLOT_VER = "regr"
    """

    from pythonlib.tools.pandastools import pivot_table
    from pythonlib.tools.pandastools import slice_by_row_label
    from pythonlib.tools.pandastools import merge_subset_indices_prioritizing_second
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from pythonlib.tools.statstools import statsmodel_ols

    list_seqc_0_shape = sorted(dfscores["seqc_0_shape"].unique().tolist())
    list_shape_fixation = sorted(dfscores["shape-fixation"].unique().tolist())
    assert np.all(dfscores["shape-fixation"] == dfscores["pa_class"]), "sanity cehck"

    if PLOT_VER == "scatter_decoder_vs_decoder":
        overlay_point = False
        overlay_point_baseline = False
        overlay_scatter = False
        overlay_regr = False
    elif PLOT_VER == "point":
        overlay_point = True
        overlay_point_baseline = True
        overlay_scatter = False
        overlay_regr = False
    elif PLOT_VER == "point_and_scatter":
        overlay_point = True
        overlay_point_baseline = False
        overlay_scatter = True
        overlay_regr = False
    elif PLOT_VER == "regr":
        overlay_point = False
        overlay_point_baseline = False
        overlay_scatter = False
        overlay_regr = True
    else:
        assert False

    # twind = (0, 0.3)
    # twind = (-0.35, -0.05)
    vars_trial = ["pa_idx"]

    if var_color in ["event_idx_within_trial", "shape-macrosaccade-index"]:
        palette = "plasma"
    elif var_color in ["early_late_by_smi", "is-first-macrosaccade"]:
        palette = "deep"
    else:
        print(var_color)
        assert False

    nrows = len(list_shape_fixation)
    ncols = len(list_seqc_0_shape)
    SIZE = 5
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    n_min_per_x = 3

    color_draw = [1, 0, 0, 1]
    color_fix = [0, 0, 1, 1]
    color_draw_baseline = [1, 0.7, 0.7, 0.25]
    color_fix_baseline = [0.7, 0.7, 1, 0.25]
    map_decoder_ver_to_col = {
                            "decoder_match_fix":color_fix,
                            "decoder_match_draw":color_draw,
                            "decoder_match_fix_control":color_fix_baseline,
                            "decoder_match_draw_control":color_draw_baseline}
    def _extract_data(dfscores, shape_draw, shape_fix, twind, version, n_min_per_x):
        if version == "decoder_match_draw":
            dfthis = dfscores[
                (dfscores["seqc_0_shape"] == shape_draw) & (dfscores["shape-fixation"] == shape_fix) & 
                (dfscores["twind"] == twind) & (dfscores["decoder_class"] == shape_draw)
                ].reset_index(drop=True)
            # plot_color = color_draw
        elif version == "decoder_match_fix":
            dfthis = dfscores[
                (dfscores["seqc_0_shape"] == shape_draw) & (dfscores["shape-fixation"] == shape_fix) & 
                (dfscores["twind"] == twind) & (dfscores["decoder_class"] == shape_fix)
                ].reset_index(drop=True)
            # plot_color = color_fix
        elif version == "decoder_match_draw_control":
            # control: i.e., not drawing what decoder
            dfthis = dfscores[
                (dfscores["seqc_0_shape"] != shape_draw) & (dfscores["shape-fixation"] == shape_fix) & 
                (dfscores["twind"] == twind) & (dfscores["decoder_class"] == shape_draw)
                ].reset_index(drop=True)
            # plot_color = color_draw_baseline
        elif version == "decoder_match_fix_control":
            # control: i.e., not looking at fix
            dfthis = dfscores[
                (dfscores["seqc_0_shape"] == shape_draw) & (dfscores["shape-fixation"] != shape_fix) & 
                (dfscores["twind"] == twind) & (dfscores["decoder_class"] == shape_fix)
                ].reset_index(drop=True)
            # plot_color = color_fix_baseline
        # elif version == "decoder_match_fix_control":
        #     # control: i.e., not looking at fix
        #     dfthis = dfscores[
        #         (dfscores["seqc_0_shape"] == shape_fix) & (dfscores["shape-fixation"] != shape_fix) & 
        #         (dfscores["twind"] == twind) & (dfscores["decoder_class"] == shape_fix)
        #         ].reset_index(drop=True)
        #     plot_color = color_fix_baseline
        else:
            assert False

        dfthis, _ = extract_with_levels_of_var_good(dfthis, [var_color], n_min_per_var=n_min_per_x)
        dfthis = dfthis.sort_values(var_color) # need this, so plot x values are in order
        x = [str(i) for i in dfthis[var_color].values] # Or else pointplot and scatter will not align.
        y = dfthis["score"].values

        return dfthis, x, y, map_decoder_ver_to_col[version]
    
    sharex = False
    assert sharex==False, "otherwise misaligns scatter and pointplot, not sure why"
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=sharex, sharey=True)

    res_regr = [] # Regresson results.
    for i, shape_draw in enumerate(list_seqc_0_shape):
        for j, shape_fix in enumerate(list_shape_fixation):

            ax = axes[j][i]
            print("draw, fix:", shape_draw, shape_fix)
            ax.set_title(f"draw {shape_draw}, fix {shape_fix}")
            
            if PLOT_VER == "scatter_decoder_vs_decoder":
                dfscores_this = dfscores[
                    (dfscores["shape-fixation"] == shape_fix) & (dfscores["seqc_0_shape"] == shape_draw) & (dfscores["twind"] == twind)
                    ].reset_index(drop=True)
                
                if len(dfscores_this)>0:
                    dfscores_this_wide = pivot_table(dfscores_this, vars_trial, ["decoder_class"], ["score"], flatten_col_names=True)
                    
                    cols_keep = ["shape_draw_fix", "event_idx_within_trial", "is-first-macrosaccade", "early_late_by_smi", "shape-macrosaccade-index", "seqc_0_shape", "shape-fixation", "decoder_class_was_first_drawn", "trialcode"]
                    list_pa_idx = dfscores_this_wide["pa_idx"].tolist()
                    dfscores_this_single = dfscores_this.groupby(["pa_idx"]).first().reset_index() # Take the first indstance
                    dfscores_this_sub = slice_by_row_label(dfscores_this_single, "pa_idx", list_pa_idx, assert_exactly_one_each=True)

                    dfscores_this_wide = merge_subset_indices_prioritizing_second(dfscores_this_wide, dfscores_this_sub.loc[:, cols_keep])

                    # For each fixation, recode to get score for fixated shape, and score for draw shape
                    dfscores_this_wide["score-seqc_0_shape"] = [row[f"score-{row['seqc_0_shape']}"] for i, row in dfscores_this_wide.iterrows()]
                    dfscores_this_wide["score-shape-fixation"] = [row[f"score-{row['shape-fixation']}"] for i, row in dfscores_this_wide.iterrows()]

                    try:
                        x = dfscores_this_wide[f"score-{shape_draw}"]
                        y = dfscores_this_wide[f"score-{shape_fix}"]
                        assert np.all(x==dfscores_this_wide["score-seqc_0_shape"])
                        assert np.all(y==dfscores_this_wide["score-shape-fixation"])
                    except Exception as err:
                        print(dfscores_this_wide)
                        raise err
                    # msi = [str(i) for i in dfscores_this_wide[f"shape-macrosaccade-index"]]
                    msi = [i for i in dfscores_this_wide[var_color]]

                    # SCATTER
                    import seaborn as sns
                    sns.scatterplot(x=x, y=y, hue=msi, hue_order = sorted(set(msi)), palette=palette, ax=ax, alpha=0.4)
                    ax.set_xlabel(f"draw={shape_draw}")
                    ax.set_ylabel(f"fix={shape_fix}")
            elif PLOT_VER in ["point", "point_and_scatter", "regr"]:

                ### OVERLAY DATA
                # - (fixated)
                decoder_ver = "decoder_match_fix"
                dfthis, x, y, plot_color = _extract_data(dfscores, shape_draw, shape_fix, twind, decoder_ver, n_min_per_x)
                if len(dfthis)>0:
                    if overlay_point:
                        try:
                            sns.pointplot(x=x, y=y, ax=ax, color=plot_color)
                        except Exception as err:
                            print(x)
                            print(y)
                            print(plot_color)
                            raise err
                    if overlay_scatter:
                        sns.scatterplot(x=x, y=y, ax=ax, color=plot_color, alpha=0.2)
                    if overlay_regr:
                        # Overlay regression fit.
                        sns.regplot(dfthis, x=var_color, y="score", ax=ax, color=plot_color)
                        if len(dfthis)>7: # n min for ols
                            results = statsmodel_ols(dfthis[var_color].values, dfthis["score"].values)
                            # ax.text(0, -0.1, f"r2={results.rsquared:.2f}|p={results.pvalues[0]:.3f}, {results.pvalues[1]:.3f}", color=color_fix, fontsize=12)
                            try:
                                ax.text(0, -0.1, f"r2={results.rsquared:.2f}|p={results.pvalues[1]:.3f}", color=plot_color, fontsize=12)
                            except Exception as err:
                                print(dfthis[var_color].values.shape)
                                print(print(results.summary()))
                                raise err

                            res_regr.append({
                                "xvals":sorted(set(dfthis[var_color].values)),
                                "shape_draw":shape_draw,
                                "shape_fix":shape_fix,
                                "shape_draw_fix":(shape_draw, shape_fix),
                                "decoder_ver":decoder_ver,
                                "slope":results.params[1],
                                "intercept":results.params[0],
                            })

                # - (draw)
                decoder_ver = "decoder_match_draw"
                dfthis, x, y, plot_color = _extract_data(dfscores, shape_draw, shape_fix, twind, decoder_ver, n_min_per_x)
                if len(dfthis)>0:
                    if overlay_point:
                        sns.pointplot(x=x, y=y, ax=ax, color=plot_color)
                    if overlay_scatter:
                        sns.scatterplot(x=x, y=y, ax=ax, color=plot_color, alpha=0.2)
                    if overlay_regr:
                        # Overlay regression fit.
                        sns.regplot(dfthis, x=var_color, y="score", ax=ax, color=plot_color)
                        if len(dfthis)>7: # n min for ols
                            results = statsmodel_ols(dfthis[var_color].values, dfthis["score"].values)
                            # ax.text(0, -0.1, f"r2={results.rsquared:.2f}|p={results.pvalues[0]:.3f}, {results.pvalues[1]:.3f}", color=color_fix, fontsize=12)
                            ax.text(0, -0.03, f"r2={results.rsquared:.2f}|p={results.pvalues[1]:.3f}", color=plot_color, fontsize=12)
                            res_regr.append({
                                "xvals":sorted(set(dfthis[var_color].values)),
                                "shape_draw":shape_draw,
                                "shape_fix":shape_fix,
                                "shape_draw_fix":(shape_draw, shape_fix),
                                "decoder_ver":decoder_ver,
                                "slope":results.params[1],
                                "intercept":results.params[0],
                            })

                # if shape_draw=="arcdeep-4-3-0" and shape_fix == "V-2-2-0":
                #     assert False

                ### Overlay controls (baseline)
                # - (not fixated)
                decoder_ver = "decoder_match_fix_control"
                dfthis, x, y, plot_color = _extract_data(dfscores, shape_draw, shape_fix, twind, decoder_ver, n_min_per_x)
                if len(dfthis)>0:
                    ax.axhline(np.mean(y), color=plot_color, alpha=0.5, linestyle="--")
                    if overlay_point_baseline:
                        sns.pointplot(x=x, y=y, ax=ax, color=plot_color)
                    # if overlay_scatter:
                    #     sns.scatterplot(x=x, y=y, ax=ax, color=plot_color, alpha=0.2)
                    if overlay_regr:
                        # Overlay regression fit.
                        # sns.regplot(dfthis, x=var_color, y="score", ax=ax, color=plot_color)
                        if len(dfthis)>7: # n min for ols
                            results = statsmodel_ols(dfthis[var_color].values, dfthis["score"].values)
                            # ax.text(0, -0.1, f"r2={results.rsquared:.2f}|p={results.pvalues[0]:.3f}, {results.pvalues[1]:.3f}", color=color_fix, fontsize=12)
                            res_regr.append({
                                "xvals":sorted(set(dfthis[var_color].values)),
                                "shape_draw":shape_draw,
                                "shape_fix":shape_fix,
                                "shape_draw_fix":(shape_draw, shape_fix),
                                "decoder_ver":decoder_ver,
                                "slope":results.params[1],
                                "intercept":results.params[0],
                            })


                # - (not drawn)
                decoder_ver = "decoder_match_draw_control"
                dfthis, x, y, plot_color = _extract_data(dfscores, shape_draw, shape_fix, twind, decoder_ver, n_min_per_x)
                if len(dfthis)>0:
                    ax.axhline(np.mean(y), color=plot_color, alpha=0.5, linestyle="--")
                    if overlay_point_baseline:
                        sns.pointplot(x=x, y=y, ax=ax, color=plot_color)
                    # if overlay_scatter:
                    #     sns.scatterplot(x=x, y=y, ax=ax, color=plot_color, alpha=0.2)
                    if overlay_regr:
                        # Overlay regression fit.
                        # sns.regplot(dfthis, x=var_color, y="score", ax=ax, color=plot_color)
                        if len(dfthis)>7: # n min for ols
                            results = statsmodel_ols(dfthis[var_color].values, dfthis["score"].values)
                            # ax.text(0, -0.1, f"r2={results.rsquared:.2f}|p={results.pvalues[0]:.3f}, {results.pvalues[1]:.3f}", color=color_fix, fontsize=12)
                            res_regr.append({
                                "xvals":sorted(set(dfthis[var_color].values)),
                                "shape_draw":shape_draw,
                                "shape_fix":shape_fix,
                                "shape_draw_fix":(shape_draw, shape_fix),
                                "decoder_ver":decoder_ver,
                                "slope":results.params[1],
                                "intercept":results.params[0],
                            })

    savefig(fig, f"{savedir}/temporal-{PLOT_VER}-var_color={var_color}-twind={twind}.pdf")
    dfregr = pd.DataFrame(res_regr)
    return dfregr, map_decoder_ver_to_col


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    combine_areas = int(sys.argv[3])==1

    # PLOTS_DO = ["plot_all", "plot_scatter_dissociate_draw_fix"]
    # PLOTS_DO = ["plot_scatter_dissociate_draw_fix"]
    PLOTS_DO = ["plot_all"]
    
    ### PARAMS
    fr_normalization_method = "across_time_bins"
    # if animal == "Pancho":
    #     combine_areas = False
    # else:
    #     combine_areas = True

    # Method 2 - Combine two dfallpa
    DFallpa1 = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=combine_areas, use_time=True)
    which_level = "saccade_fix_on"
    DFallpa2 = load_handsaved_wrapper(animal=animal, date=date, version=which_level, combine_areas=combine_areas, use_time=True)
    DFallpa = pd.concat([DFallpa1, DFallpa2]).reset_index(drop=True)

    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_wrapper
    dfpa_concatbregion_preprocess_wrapper(DFallpa, fr_mean_subtract_method=fr_normalization_method)

    ### PLOTS
    SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/eyetracking_analyses/DECODE_MOMENT_shape-fixation|seqc_0_shape/{animal}-{date}-combine={combine_areas}-wl={which_level}-norm={fr_normalization_method}"
    os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

    LIST_BREGION = DFallpa["bregion"].unique().tolist()
    for bregion in LIST_BREGION:
        
        if "plot_all" in PLOTS_DO:
            SAVEDIR = f"{SAVEDIR_ANALYSIS}/plot_all/{bregion}"
            os.makedirs(SAVEDIR, exist_ok=True)
            plot_all(DFallpa, bregion, SAVEDIR)
        
        if "plot_scatter_dissociate_draw_fix" in PLOTS_DO:
            SAVEDIR = f"{SAVEDIR_ANALYSIS}/plot_scatter_dissociate_draw_fix/{bregion}"
            os.makedirs(SAVEDIR, exist_ok=True)
            plot_scatter_dissociate_draw_fix(DFallpa, bregion, SAVEDIR)
