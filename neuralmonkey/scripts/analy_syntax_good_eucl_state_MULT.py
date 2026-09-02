"""
Organizing good plots for syntax, espeically:
- euclidian dist
- state space

NOTEBOOK: 250510_syntax_good.ipynb

"""

from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
import sys
import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.tools.plottools import savefig
import pickle
from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER
import pickle
import os
import matplotlib.pyplot as plt

# General map used throughout syntax paper.
map_bregion_to_color = {
    "M1":"#587467eb",
    "PMd":"#077026ec",
    "PMv":"#397721ed",
    "SMA":"#98be3eec",
    "preSMA":"#f80f0f",
    "dlPFC":"#8675AF",
    "vlPFC":"#414994",
    "FP":"#287AC7",
}

def final_plot_state_space(analysis, RUN, subspace, LIST_VAR_VAR_OTHERS, iternum=0):
    """
    Make good state space plots for paper. 
    
    LT Checked
    """
    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED

    ### Params
    save_suffix = "AnBmCk_general"
    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}"
    # yvar = "dist_yue_diff"
    
    HACK = False # Just to plot subset, for quicker analy
    LIST_DIMS = [(0,1), (2,3), (4,5)]
    # LIST_DIMS = [(4,5)]
    # force_continuous = True # color using continuous colormap
    force_continuous = False # color using continuous colormap

    ### Iterate
    for animal in ["Diego", "Pancho"]:
        list_dates, _, _, _ = load_preprocess_get_dates(animal, save_suffix)
        list_dates = list(set(list_dates))

        if analysis=="two_shapes":
            assert animal == "Diego", "dont need to do this."
            list_dates = [240827, 240822]

        if HACK:
            # Just to plot a hand picked set of dates
            list_dates = [230816, 230913, 231118, 230726]
            # list_dates = [230726]
        
        for date in list_dates:
            savedir= f"{SAVEDIR}/MULT/final-{animal}-ss={save_suffix}-analy={analysis}/neural_state_space/{animal}-{date}"
            os.makedirs(savedir, exist_ok=True)

            for bregion in _REGIONS_IN_ORDER_COMBINED:
                
                if HACK and bregion not in ["preSMA", "PMv", "M1"]:
                    continue

                # Make improved state space plots
                path = f"{SAVEDIR}/{animal}-{date}-q=RULE_ANBMCK_STROKE/bregion={bregion}/FITTING_subspc={subspace}-iter={iternum}/pa_subspace.pkl"

                if os.path.exists(path):
                    print("loading path:", path)
                    with open(path, "rb") as f:
                        pa_subspace = pickle.load(f)

                    ### Plot all subspaces
                    dflab = pa_subspace.Xlabels["trials"]
                    dflab["date"] = date
                    
                    # In general, remove SP
                    if analysis != "pig_vs_sp":
                        inds = dflab[
                            ~(dflab["epoch"]=="base") &
                            (dflab["task_kind"]=="prims_on_grid")
                            ].index.tolist() # beucase "base" messes up global cr
                        pa_subspace = pa_subspace.slice_by_dim_indices_wrapper("trials", inds)
                        dflab = pa_subspace.Xlabels["trials"]

                    if analysis == "two_shapes":
                        from pythonlib.dataset.dataset_analy.grammar import chunk_rank_global_extract
                        assert animal=="Diego" and date in [240822, 240827], "only checked for these expts."
                        # To set up for other dates, run this and make sure that global chunk rank is aligned
                        # with shapes.
                        # from pythonlib.tools.pandastools import grouping_print_n_samples
                        # grouping_print_n_samples(dflab, ["epoch", "chunk_rank_global", "shape", "chunk_rank"])
                        dflab = pa_subspace.Xlabels["trials"]
                        chunk_rank_global_extract(dflab)

                    Xredu = pa_subspace.X # (chans, trials, 1)
                    x = Xredu.squeeze().T # (trials, chans)
                    dflab = pa_subspace.Xlabels["trials"]
                    
                    ### Second, plot scalar neural data
                    # for plot_kde in [True, False]:
                    for plot_kde in [True, False]:
                        for var_effect, vars_others in LIST_VAR_VAR_OTHERS:
                            trajgood_plot_colorby_splotby_scalar_WRAPPER(x, dflab, var_effect, savedir,
                                                                            vars_subplot=vars_others, list_dims=LIST_DIMS,
                                                                            overlay_mean_orig=True, plot_kde=plot_kde,
                                                                            save_suffix=f"kde={plot_kde}-{bregion}-iter={iternum}",
                                                                            force_continuous=force_continuous)
                            plt.close("all")

def final_dfeffect_plots_WRAPPER(RUN, save_suffix, yvar, analysis, return_debug=False, skip_ratio_stats=False,
        n_shuff_ratio=1000, HACK_RETURN_DFEFFECT=False, HACK_ANIMALS=None):
    """
    Wrapper for all final plots of euclidean distance analysis of syntax.

    # ### MAIN ONES
    # RUN = 29
    # save_suffix = "AnBmCk_general"
    # yvar = "dist_yue_diff"
    # analysis = "pig_vs_sp"

    # # RUN = 27
    # # save_suffix = "AnBmCk_general"
    # # yvar = "dist_yue_diff"
    # # analysis = "two_shapes"

    # # RUN = 27 
    # # save_suffix = "AnBmCk_general"
    # # yvar = "dist_yue_diff"
    # # analysis = "rank_within"

    # ### OTHER ONES (not part of paper)
    # # RUN = 27 
    # # save_suffix = "AnBmCk_general"
    # # yvar = "dist_yue_diff"
    # # analysis = "rank_up_vs_down"

    # # RUN = 27 
    # # save_suffix = "AnBmCk_general"
    # # yvar = "dist_yue_diff"
    # # analysis = "n_in_chunk"

    LT CHECKED (at least for run 29)
    - Also checked for the section relating ord encoding to gap duration (ie. rank_up_vs_down)
    """

    from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import targeted_pca_MULT_3_combined_plots
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping, aggregGeneral
    from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import final_dfeffect_postprocess
    from pythonlib.tools.pandastools import append_col_with_grp_index, aggregGeneral
    from pythonlib.tools.pandastools import pivot_table
    import os
    from pythonlib.tools.pandastools import savefig

    # yvar = "dist_yue_diff"

    if analysis=="two_shapes":
        # True/False gives very similar results, so just stick with False
        # list_remove_probe = [False, True]
        list_remove_probe = [False]
    else:
        # Then doesnt matter whats in list_remove_probe, it doesnt do anything.
        # list_remove_probe = [True]
        list_remove_probe = ["ignore"]

    if False:
        # During testing
        list_n_min_trials = [2,3,4]
    else:
        # This is the final param
        list_n_min_trials = [2]

    if HACK_ANIMALS is None:
        list_animal = ["Pancho", "Diego"]
    else:
        list_animal = HACK_ANIMALS

    # for animal in ["Diego"]:
    # for animal in ["Pancho"]:
    # for animal in ["Diego", "Pancho"]:
    for animal in list_animal:
    # for animal in ["Diego"]:
        DFEFFECT_ALL, _ = targeted_pca_MULT_3_combined_plots(animal, RUN, save_suffix, return_dfeffect=True)
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}/MULT/final-{animal}-ss={save_suffix}-analy={analysis}"
        os.makedirs(SAVEDIR, exist_ok=True)

        for n_min_trials_per_label in list_n_min_trials:

            if animal == "Pancho":
                if analysis == "pig_vs_sp":
                    list_h_dates = [None, 0, 1, 2]
                else:
                    list_h_dates = [None]
            else:
                list_h_dates = [None]

            for HACK_dates in list_h_dates:
                for remove_probe in list_remove_probe:
                    savedir = f"{SAVEDIR}/nmintrials={n_min_trials_per_label}-H={HACK_dates}-remove_prb={remove_probe}"
                    os.makedirs(savedir, exist_ok=True)

                    ### POSTPROCESS
                    DFEFFECT, eff1, eff2 = final_dfeffect_postprocess(DFEFFECT_ALL, animal, analysis, savedir, 
                                                                    n_min_trials_per_label=n_min_trials_per_label, 
                                                                    HACK_dates=HACK_dates,
                                                                    two_shapes_remove_probe_trials=remove_probe)
                    from pythonlib.tools.pandastools import integerify_values
                    integerify_values(DFEFFECT, "chunk_rank_1")
                    integerify_values(DFEFFECT, "chunk_rank_2")
                    DFEFFECT = append_col_with_grp_index(DFEFFECT, ["chunk_rank_1", "chunk_rank_2"], "chunk_rank_12")                
                    DFEFFECT = append_col_with_grp_index(DFEFFECT, ["date", "chunk_rank_1", "chunk_rank_2", "shape_1", "shape_2"], "da_cr_sh_12")

                    # if return_debug:
                    #     return DFEFFECT, eff1, eff2, savedir

                    # Agg. Datapt = date
                    DFEFFECT_AGG = aggregGeneral(DFEFFECT, ["effect", "animal", "date", "bregion", "question", "subspace"], ["dist_yue_diff"])


                    ########################################
                    ### Classic plots
                    if False:
                        fig = sns.catplot(data=DFEFFECT, x="bregion", y="dist_yue_diff", hue="effect", col="date", jitter=True, alpha=0.5)
                        for ax in fig.axes.flatten():
                            ax.axhline(0, color="k", alpha=0.5)

                    # First, get both directions, in case you want to over labels_1
                    dftmp = DFEFFECT.copy()
                    dftmp["labels_1"] = DFEFFECT["labels_2"]
                    dftmp["labels_2"] = DFEFFECT["labels_1"]
                    dfeffect_full = pd.concat([dftmp, DFEFFECT], axis=0).reset_index(drop=True)

                    if HACK_RETURN_DFEFFECT:
                        # Just for devo/debugging.
                        return DFEFFECT, DFEFFECT_AGG, dfeffect_full, eff1, eff2

                    # == Bar plots
                    # fig = sns.catplot(data=DFEFFECT, x="bregion", y=yvar, hue="effect", col="date", col_wrap=6, kind="bar", errorbar="se")
                    # savefig(fig, f"{savedir}/catplot-data=label-1.pdf")

                    for var_datapt in ["labels_1", "da_cr_sh_12"]:
                        _vars = ["effect", "animal", "date", "bregion", "question", "subspace"] + [var_datapt]
                        dfeffect_agg = aggregGeneral(dfeffect_full, _vars, [yvar])

                        fig = sns.catplot(data=dfeffect_agg, x="bregion", y=yvar, hue="effect", col="date", col_wrap=6, kind="bar", errorbar="se")
                        savefig(fig, f"{savedir}/catplot-data={var_datapt}-1.pdf")
                        
                        fig = sns.catplot(data=dfeffect_agg, x="effect", y=yvar, col="bregion", jitter=True, alpha=0.25)
                        for ax in fig.axes.flatten():
                            ax.axhline(0, color="k")
                        savefig(fig, f"{savedir}/catplot-data={var_datapt}-2.pdf")

                        fig = sns.catplot(data=dfeffect_agg, x="effect", y=yvar, col="bregion", kind="bar", errorbar="se")
                        savefig(fig, f"{savedir}/catplot-data={var_datapt}-3.pdf")

                    # == Scatterplots
                    _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effect", eff1, eff2, "date", yvar, "bregion", shareaxes=True);
                    savefig(fig, f"{savedir}/scatter-data=label-1.pdf")

                    _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effect", eff1, eff2, None, yvar, "bregion", shareaxes=True);
                    savefig(fig, f"{savedir}/scatter-data=label-2.pdf")

                    _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_AGG, "effect", eff1, eff2, None, yvar, "bregion", shareaxes=True);
                    savefig(fig, f"{savedir}/scatter-data=date-2.pdf")

                    plt.close("all")

                    # Also plot each date
                    # Plot scatter for each date, comparing two bregions.
                    # from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
                    from pythonlib.tools.pandastools import plot_45scatter_color_by_var                       

                    ############################# COMPARE y/x RATIOS
                    if analysis in ["pig_vs_sp", "two_shapes"] and skip_ratio_stats==False:
                    # if analysis == "two_shapes" and skip_ratio_stats==False:
                        from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import final_dfeffect_compute_plot_ratio_stats
                        dfres, savedir_this = final_dfeffect_compute_plot_ratio_stats(DFEFFECT, eff1, eff2, 
                            savedir, analysis, n_shuff=n_shuff_ratio)
                        dfres.to_pickle(f"{savedir_this}/dfres.pkl")
                        del dfres

                    ##########
                    if analysis == "rank_up_vs_down": # Scatter, each datapt a (date, cr, shape) and color by things (e.g, cr)
                        from pythonlib.dataset.dataset_analy.grammar import chunk_rank_global_extract
                        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping_color_mapper
                        _HACK = True # Untru this after fix the epoch naems for cross-AB dates.

                        # Will assume these, so verify
                        assert all(DFEFFECT["chunk_rank_same"] == True)
                        assert all(DFEFFECT["shape_same"] == True)
                        assert all(DFEFFECT["epoch_1"] == "none")
                        assert all(DFEFFECT["epoch_2"] == "none")
                        DFEFFECT["shape"] = DFEFFECT["shape_1"]
                        DFEFFECT["chunk_rank"] = DFEFFECT["chunk_rank_1"]
                        DFEFFECT["epoch"] = DFEFFECT["epoch_1"]

                        # This date epochs not set well, will fail
                        if _HACK:
                            dfeffect = DFEFFECT[~(DFEFFECT["date"].isin([240822, 240827]))].reset_index(drop=True)
                        else:
                            dfeffect = DFEFFECT.reset_index(drop=True)

                        ### Analyses the require chunk_rank_global.
                        if True: # Good code. I'm just testing.
                            # Get chunk rank global
                            # try:
                            assert all(dfeffect["task_kind_12"] == "prims_on_grid|prims_on_grid")
                            dfeffect["task_kind"] = "prims_on_grid" # Hacky, to get the followng to not break
                            _ = chunk_rank_global_extract(dfeffect, shape_ratio_max= 0.99)
                            list_var_to_color = [None, "chunk_rank_12", "chunk_rank_global"]
                            # except Exception as err:
                            #     list_var_to_color = [None, "chunk_rank_12"]

                            #################################################################
                            # (1) Scatterplot
                            var_datapt = "da_cr_sh_12"
                            for var_to_color in list_var_to_color:
                                
                                if var_to_color is not None:
                                    map_datapt_lev_to_colorlev, colorlevs_that_exist = plot_45scatter_means_flexible_grouping_color_mapper(
                                        dfeffect, var_datapt, var_to_color)
                                else:
                                    map_datapt_lev_to_colorlev, colorlevs_that_exist = None, None
                                _, fig = plot_45scatter_means_flexible_grouping(dfeffect, "effect", eff1, eff2, "bregion", yvar, var_datapt, 
                                                                                plot_text=False, shareaxes=True, alpha=0.3,
                                                                                map_datapt_lev_to_colorlev=map_datapt_lev_to_colorlev,
                                                                                colorlevs_that_exist=colorlevs_that_exist)
                                savefig(fig, f"{savedir}/scatter-data={var_datapt}-colorby={var_to_color}.pdf")            

                        
                        #################################################################
                        # (2) Scatterplot, coloring by gap durations.
                        from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import final_dfeffect_load_dfgaps

                        ### (1) Load gaps data
                        DFGAPS, map_gapsemantic_to_dfgaps = final_dfeffect_load_dfgaps(dfeffect, animal, savedir)

                        ### (2) Plot scatter, overlaying gaps durations
                        from pythonlib.tools.pandastools import pivot_table
                        from pythonlib.tools.pandastools import slice_by_row_label
                        from pythonlib.tools.plottools import set_axis_lims_square_bounding_data_45line

                        var_datapt = "da_cr_sh_12"
                        # _xvar = "dist_yue_diff-14_rankwithin_dn"
                        # _yvar = "dist_yue_diff-14_rankwithin_up"
                        for which_gap_semantic_higher, dfgaps_agg in map_gapsemantic_to_dfgaps.items():

                            # Assign gap duration values to dfeffect
                            if which_gap_semantic_higher == "within_chk":
                                dfeffect_this = dfeffect.copy() # These effects will already be within the same (cr, shape)
                            else:
                                # Then have to prune neural data to just those neural chunks that trans from or to (depending on analysis)
                                dfeffect_this = dfeffect[dfeffect["da_cr_sh_12"].isin(dfgaps_agg["da_cr_sh_12"].unique().tolist())].reset_index(drop=True)
                            row_values = dfeffect_this["da_cr_sh_12"].tolist()
                            dftmp = slice_by_row_label(dfgaps_agg, "da_cr_sh_12", row_values, assert_exactly_one_each=True)
                            dfeffect_this["gap_dur"] = dftmp["gap_dur"]

                            # Plot!
                            from pythonlib.tools.pandastools import plot_45scatter_color_by_var
                            var_subplot = "bregion"
                            var_manip = "effect"
                            x_lev_manip = "14_rankwithin_dn"
                            y_lev_manip = "14_rankwithin_up"
                            var_color = "gap_dur"
                            fig1, fig2, fig3, dfeffect_pivot = plot_45scatter_color_by_var(dfeffect_this, var_manip, x_lev_manip, y_lev_manip,
                                                                    var_subplot, yvar, var_datapt, var_color, return_dfpivot=True)
                            savefig(fig1, f"{savedir}/scatter-data={var_datapt}-colorby=gap_dur-{which_gap_semantic_higher}.pdf")            
                            savefig(fig2, f"{savedir}/scatter-data={var_datapt}-x={x_lev_manip}_MIN_{y_lev_manip}-{which_gap_semantic_higher}-1.pdf")            
                            savefig(fig3, f"{savedir}/scatter-data={var_datapt}-x={x_lev_manip}_MIN_{y_lev_manip}-{which_gap_semantic_higher}-2.pdf") # MS FIGURE
                            plt.close("all")

                            ### Also do regression to get p-values
                            from pythonlib.tools.statstools import statsmodel_linregress_ols_each_conj_grp
                            _dfres = statsmodel_linregress_ols_each_conj_grp(dfeffect_pivot, "var_color", "x_min_y", var_subplot)                            
                            _dfres.to_csv(f"{savedir}/scatter-data={var_datapt}-x={x_lev_manip}_MIN_{y_lev_manip}-{which_gap_semantic_higher}-stats.csv")

                            if return_debug:
                                return dfeffect_this, var_manip, x_lev_manip, y_lev_manip, var_subplot, yvar, var_datapt, var_color

                            # Also plot versus controls:
                            # TODO: Also plot vs. n in chunk. Prob best to use dfgaps to infer this, similar to how I did for gap_dur

                    ########################################
                    ### Bootstrapped plot
                    # Each datapt a unique (label_1, label_2), ie each a single datapt (acafter taking mean over all trials).
                    vars_conj = ["bregion", "date", "effect"]
                    nboot = 100
                    DFSCORE_BOOT, effect_div_name, eff1_name, eff2_name = final_dfeffect_mean_simple_PIGvsSP_bootstrap(DFEFFECT, 
                                                                eff1, eff2, vars_conj, nboot)
                    DFSCORE_BOOT_WIDE = pivot_table(DFSCORE_BOOT, ["bregion", "i_boot"], ["effect"], [yvar], flatten_col_names=True)
                    effect_div_name = f"{yvar}-{effect_div_name}"
                    assert effect_div_name in DFSCORE_BOOT_WIDE

                    # Plot scatter using actual data, not bootstrap
                    if False:
                        DFSCORE_WIDE = pivot_table(DFSCORE, ["bregion", "i_boot"], ["effect"], ["dist_yue_diff"], flatten_col_names=True)
                        DFSCORE_WIDE_ORIG = pivot_table(DFEFFECT, ["bregion", "shapeloc12"], ["effect"], ["dist_yue_diff"], flatten_col_names=True)
                        fig, ax = plot_class_kde(
                            DFSCORE_WIDE,
                            x="dist_yue_diff-shapePIG",
                            y="dist_yue_diff-dist_yue_diff-shapeSPdivshapePIG",
                            label="bregion",
                            levels=12,
                            # levels=(0.05, 0.5, 0.95),
                            normalize="per_class",
                            scatter=True
                        )
                        fig, ax = plot_class_kde(
                            DFSCORE_WIDE,
                            x="dist_yue_diff-shapePIG",
                            y="dist_yue_diff-shapeSP",
                            label="bregion",
                            levels=12,
                            # levels=(0.05, 0.5, 0.95),
                            normalize="per_class",
                            scatter=True
                        )

                    try:
                        from pythonlib.tools.pandastools import plot_class_kde
                        from pythonlib.tools.plottools import set_axis_lims_square_bounding_data_45line

                        DFSCORE_BOOT_WIDE_AGG = aggregGeneral(DFSCORE_BOOT_WIDE, ["bregion"], [eff1_name, eff2_name, effect_div_name])

                        # --- 
                        fig, ax = plot_class_kde(DFSCORE_BOOT_WIDE, x=eff1_name, y=eff2_name, label="bregion", levels=10, scatter=False,
                                    cmap_per_class=map_bregion_to_color, ellipses=True)
                        savefig(fig, f"{savedir}/kdescatter-1.pdf")
                        
                        # - also plot with 45 deg axis
                        xs = DFSCORE_BOOT_WIDE_AGG[eff1_name].values
                        ys = DFSCORE_BOOT_WIDE_AGG[eff2_name].values
                        set_axis_lims_square_bounding_data_45line(ax, xs, ys, dotted_lines="unity")
                        savefig(fig, f"{savedir}/kdescatter-1-box.pdf")

                        # --- 
                        fig, ax = plot_class_kde(DFSCORE_BOOT_WIDE, x=eff1_name, y=effect_div_name, label="bregion", levels=10, scatter=False,
                                    cmap_per_class=map_bregion_to_color, ellipses=True)
                        savefig(fig, f"{savedir}/kdescatter-2.pdf")

                        # --- Plot ratios, without FP
                        df = DFSCORE_BOOT_WIDE[~(DFSCORE_BOOT_WIDE["bregion"]=="FP")]
                        fig, ax = plot_class_kde(df, x=eff1_name, y=effect_div_name, label="bregion", levels=10, scatter=False,
                                    cmap_per_class=map_bregion_to_color, ellipses=True)
                        savefig(fig, f"{savedir}/kdescatter-2-noFP.pdf")

                    except Exception as err:
                        pass

                    if False:
                        # sns.jointplot(DFSCORE_WIDE_ORIG, x="dist_yue_diff-shapePIG", y="dist_yue_diff-dist_yue_diff-shapeSPdivshapePIG", hue="bregion", kind="kde")
                        sns.jointplot(DFSCORE_WIDE_ORIG, x="dist_yue_diff-shapePIG", y="dist_yue_diff-shapeSP", hue="bregion", kind="kde", fill=True)
                        sns.jointplot(DFSCORE_WIDE, x="dist_yue_diff-shapePIG", y="dist_yue_diff-dist_yue_diff-shapeSPdivshapePIG", hue="bregion", kind="kde")
                        sns.jointplot(DFSCORE_WIDE, x="dist_yue_diff-shapePIG", y="dist_yue_diff-shapeSP", hue="bregion", kind="kde", fill=True)

                    ########################################
                    #### Stats -- compare regions to each other
                    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_2br_scatter_wrapper, euclidianshuff_stats_linear_plot_wrapper
                    import os
                    var_same_same = "effect"
                    if analysis == "pig_vs_sp":
                        var_datapt = "shapeloc_12"
                    elif analysis == "two_shapes":
                        var_datapt = "cr_and_w_12"
                    elif analysis == "rank_within":
                        var_datapt = "shape_12"
                    elif analysis == "rank_up_vs_down":
                        var_datapt = "shape_12"
                    elif analysis == "n_in_chunk":
                        var_datapt = "shape_12"
                    else:
                        assert False
                    savedir_this = f"{savedir}/stats"
                    os.makedirs(savedir_this, exist_ok=True)
                    plot_heatmap_counts=False
                    plot_catplots=True
                    plot_results_scatter=True
                    var_same_same_levels = [eff1, eff2]
                    vars_needed = ["subspace|twind", "event", "metaparams"]
                    for var in vars_needed:
                        if var not in DFEFFECT:
                            DFEFFECT[var] = "none"
                    _ = _euclidianshuff_stats_linear_2br_scatter_wrapper(DFEFFECT, var_same_same, var_datapt, savedir_this, 
                                                                        plot_heatmap_counts, plot_catplots,
                                                                        plot_results_scatter, var_same_same_levels)
                    
                    ########################################
                    ### Plots comparing bregions directly (now datapt is usllay date or other thing)
                    # (1) Plot each expt, showing effect
                    # Single-effect plots.
                    DFEFFECT["date_str"] = DFEFFECT["date"].astype("str")
                    DFEFFECT_AGG["date_str"] = DFEFFECT_AGG["date"].astype("str")

                    # Example color mapping (replace with your dict)
                    g = sns.catplot(
                        data=DFEFFECT,
                        y="date_str",
                        x="dist_yue_diff",
                        hue="bregion",
                        col="effect",
                        kind="point",
                        errorbar="se",          # show SEM
                        join=False,             # no connecting lines
                        dodge=True,             # separate groups
                        palette=map_bregion_to_color,  # use your dict
                        markers="o",            # circle markers
                        linestyles="none",       # extra safety
                    )
                    for ax in g.axes.flatten():
                        ax.axvline(0, color="k", alpha=0.5)
                        
                    savefig(g, f"{savedir}/compare_regions-sessions-catplot-1.pdf")

                    # (2) Scatterplot -- Given preSMA, compare it to every other region.
                    # TODO, write a versoin above that plots all bregions.
                    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
                    plot_45scatter_means_flexible_grouping(DFEFFECT, "bregion", "SMA", "preSMA", "effect", yvar, "date", plot_text=True, shareaxes=True)

                    if False:
                        # These methods dont work. they get diff between variables, not diff between regions.
                        from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import final_dfeffect_mean_simple_PIGvsSP
                        dfmerge_long, dfmerge_wide, effect_div_name, eff1_name, eff2_name = final_dfeffect_mean_simple_PIGvsSP(DFEFFECT_AGG, eff1, eff2, doplot=True);
                        from neuralmonkey.analyses.euclidian_distance import dfdist_compute_effects_diff_wideform
                        dfsummary, dfpivot, effect_div_name, eff1_name, eff2_name = dfdist_compute_effects_diff_wideform(DFEFFECT_AGG, "effect", eff1, eff2, ["bregion", "date"])
                        dfsummary, dfpivot, effect_div_name, eff1_name, eff2_name = dfdist_compute_effects_diff_wideform(
                            DFEFFECT_AGG, "effect", eff1, eff2, ["bregion", "date"])    


                    ########################################################
                    #### Plots -- compare preSMA to the others 
                    from neuralmonkey.analyses.euclidian_distance import dfdist_compute_regions_diff
                    # vars_datapt = ["effect", "animal", "date", "bregion", "question", "subspace", "chunk_rank_12", "shape_12"]
                    # vars_datapt = ["effect", "animal", "date", "bregion", "question", "subspace"]
                    vars_datapt = ["effect", "animal", "date", "question", "subspace"]
                    var_value = yvar
                    bregion_1 = "preSMA"
                    do_plot = False
                    dfdifference_long, _ = dfdist_compute_regions_diff(DFEFFECT, vars_datapt, 
                                                                                    var_value, bregion_1, do_plot)

                    fig = sns.catplot(data=dfdifference_long, y="bregion_1", x=var_value, col="effect", jitter=True, alpha=0.5)
                    for ax in fig.axes.flatten():
                        ax.axvline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/compare_regions-vs={bregion_1}-catplot_1.pdf")

                    fig = sns.catplot(data=dfdifference_long, y="bregion_1", x=var_value, col="effect", kind="boxen")
                    for ax in fig.axes.flatten():
                        ax.axvline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/compare_regions-vs={bregion_1}-catplot_2.pdf")

                    fig = sns.catplot(data=dfdifference_long, y="bregion_1", x=var_value, col="effect", kind="bar", errorbar="se")
                    for ax in fig.axes.flatten():
                        ax.axvline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/compare_regions-vs={bregion_1}-catplot_3.pdf")

                    plt.close("all")


def final_dfeffect_compute_plot_ratio_stats(DFEFFECT, eff1, eff2, savedir, analysis, n_shuff=10000):
    """
    Finally summary stats for two shapes, compare pairwise bregions, each bregion one value, which is
    its ratio (y/x axis)

    PARAMS:
    - eff1, eff2, the values in the "effect" column which each contributes one value to make the ratio
    eff2/eff1
    """

    from pythonlib.tools.pandastools import replace_None_with_string
    from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import final_dfeffect_mean_simple_PIGvsSP
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

    savedir_this = f"{savedir}/stats_compare_ratios"
    os.makedirs(savedir_this, exist_ok=True)

    ### First, prep dataste
    DFEFFECT = replace_None_with_string(DFEFFECT)

    if analysis == "two_shapes":
        # - What grouping for shuffling? Idea is that you flip the labels for the two brain regions
        # within each level of grouping. Choose a grouping that gives many levels but still reasonable
        # to consider independnet comparisons
        # vars_datapt = ["animal", "date", "question", "subspace", "labels_1", "labels_2", "effect"] # _index
        # vars_datapt = ["animal", "date", "question", "subspace", "effect", "labels_1"] # ok

        # # This is probably most reasonable, each (chunk, shape, gridloc_1)
        # vars_datapt = ["animal", "date", "question", "subspace", "effect", 
        #             "epoch_1", "chunk_within_rank_1", "chunk_rank_1", "shape_1", "gridloc_1"] # _index
        
        # This gets stronger stats (Older version)
        vars_datapt = ["animal", "date", "question", "subspace", "effect", 
                    "epoch_1", "chunk_within_rank_1", "chunk_rank_1", "shape_1", "gridloc_1", "CTXT_loc_prev_1"] # Originaly good. (n=593 groups)

        # This is alternative, not great.
        # vars_datapt = ["animal", "date", "question", "subspace", "effect", 
        #             "epoch_1", "chunk_within_rank_1", "shape_1", "gridloc_1"] # _index

        # # This gets even stronger stats (useful for Pancho) [good on run 29]
        # vars_datapt = ["animal", "date", "question", "subspace", "effect", 
        #     "epoch_12", "chunk_within_rank_12", "chunk_rank_12", "shape_12", "gridloc_12", "CTXT_loc_prev_12"] # (n=682)

    elif analysis == "pig_vs_sp":
        # vars_datapt = ["animal", "date", "question", "subspace", "effect", 
        #     "task_kind_12", "shape_12", "gridloc_12"] # Too strong. Everything is significaint, includng PFC less than preSMA.
        # vars_datapt = ["animal", "date", "question", "subspace", "effect", 
        #     "task_kind_12", "shape_12"] # Good for D, not enough data for P
        vars_datapt = ["animal", "date", "question", "subspace", "effect", 
            "task_kind_12", "shape_1", "gridloc_1"] # This is strong for D, but perfect for P
    else:
        print(analysis)
        assert False, "What is this?"

    DFEFFECT = append_col_with_grp_index(DFEFFECT, vars_datapt, "_grp")

    print("[ratio stats] This many unique grps exist: ", len(DFEFFECT["_grp"].unique()))
    
    # get preSMA against every other area
    list_bregion_1 = ["preSMA"]
    list_bregion_2 = ["M1", "PMd", "PMv", "SMA", "vlPFC", "dlPFC"] 

    # list_bregion = ["preSMA", "PMd", "PMv", "M1", "SMA"] # The areas in contention with preSMA (not blowing up ratio)
    # list_bregion = ["preSMA", "M1"] # The areas in contention with preSMA (not blowing up ratio)

    # Permutation test functions
    def funshuff(df):
        from pythonlib.tools.pandastools import shuffle_dataset_hierarchical_remap, append_col_with_grp_index
        df_shuff = shuffle_dataset_hierarchical_remap(df, "bregion", "_grp", col_name_shuffed="bregion")
        return df_shuff

    res = []
    for _, bregion1 in enumerate(list_bregion_1):
        for _, bregion2 in enumerate(list_bregion_2):
            # if j>i:
            print(bregion1, bregion2)
            # Slice to just these, for doing shuffle between them
            dfeffect = DFEFFECT[DFEFFECT["bregion"].isin([bregion1, bregion2])].reset_index(drop=True)

            # Make sure you actually have data to shuffle. Each group should have both brain regions.
            print("Cleaning up pairs. Starting n = ", len(dfeffect))
            dfeffect, _ = extract_with_levels_of_conjunction_vars_helper(dfeffect, "bregion", 
                vars_datapt, 1, None, 2, [bregion1, bregion2], False, None)
            print("... Resulting n = ", len(dfeffect))

            # Get actual value
            if False:
                dfmerge_long, dfmerge_wide, effect_div_name, eff1_name, eff2_name = final_dfeffect_mean_simple_PIGvsSP(dfeffect, 
                                                                                                            eff1, eff2, doplot=False)

            # # Iterate and shuffle
            # nshuff = 10
            # list_df = []
            # for i_shuff in range(nshuff):
            #     if i_shuff%20==0:
            #         print(i_shuff)
            #     # Shuffle
            #     # dfeffect_shuff = shuffle_dataset_hierarchical(dfeffect, ["bregion"], vars_datapt)

            #     from pythonlib.tools.pandastools import shuffle_dataset_hierarchical_remap, append_col_with_grp_index
            #     dfeffect_shuff = shuffle_dataset_hierarchical_remap(dfeffect, "bregion", "_grp", col_name_shuffed="bregion")

            #     # Compute effect using shuff
            #     dfmerge_long, dfmerge_wide, effect_div_name, eff1_name, eff2_name = final_dfeffect_mean_simple_PIGvsSP(dfeffect_shuff, 
            #                                                                                                 eff1, eff2, doplot=False)
            #     dfmerge_long["i_shuff"] = i_shuff
            #     list_df.append(dfmerge_long)

            # Permutation test functions
            def funstat(df):
                _, dfmerge_wide, effect_div_name, _, _ = final_dfeffect_mean_simple_PIGvsSP(df, eff1, eff2, doplot=False)
                val_bregion2 = dfmerge_wide[dfmerge_wide["bregion"] == bregion2][effect_div_name].values[0] # ratio of effects, for this bregion.
                val_bregion1 = dfmerge_wide[dfmerge_wide["bregion"] == bregion1][effect_div_name].values[0] # ...
                return val_bregion2 - val_bregion1

            # Plot shuff and overlay actual
            from pythonlib.tools.statstools import permutationTest
            p, stat_actual, stats_shuff, fig = permutationTest(dfeffect, funstat, funshuff, 
                                                                                n_shuff, force_return_stats=True)

            # Save things
            savefig(fig, f"{savedir_this}/{bregion2}-minus-{bregion1}.pdf")
            plt.close("all")
            
            res.append({
                "p":p,
                "stat_actual_2_min_1":stat_actual,
                "stats_shuff":stats_shuff,
                "bregion1":bregion1,
                "bregion2":bregion2
            })

    dfres = pd.DataFrame(res)
    return dfres, savedir_this

def final_dfeffect_compute_ratio_OLS_deviation(DFEFFECT, eff1, eff2, FIT_MODE = "loo",
    FIT_THROUGH_ORIGIN = True, RESCALE_BY_MAX_PROJECTION = True, var_datapt=None,
    FIT_ON="bregion_means", DIAG_SQUARE=True, DIAG_SHARE_AXES=False,
    RESID_KIND="orthogonal", savedir=None, SAVE_FIGS=False,
    LL_DODGE_BY_ANIMAL=True, ALPHA_DATAPT=0.35, ALPHA_DATE=0.55):
    """
    OLS residual vs. projection of (eff1, eff2) effect means across brain regions.

    For each (animal, date) session, fit OLS y ~ x, then for each region compute:
      - projection: signed coordinate along the OLS line
      - residual: signed deviation from the line (see RESID_KIND)
        (negative ⇒ below the line ⇒ lower y/x)

    Scientific goal: preSMA should have more negative residuals than other regions.

    PARAMS
    ------
    DFEFFECT : DataFrame
        Must have columns: animal, date, bregion, effect, dist_yue_diff.
        If var_datapt is set, that column must exist too.
    eff1, eff2 : hashable
        Values of the "effect" column for the x and y axes (ratio ≈ eff2/eff1).
    FIT_MODE : {"loo", "all"}
        "loo" — for each region X, fit OLS excluding that region (leave-one-out).
        "all" — single OLS fit on all regions; project every region onto that line.
    FIT_THROUGH_ORIGIN : bool
        True  → y = b*x (ratio-native, no intercept).
        False → y = a + b*x.
    RESCALE_BY_MAX_PROJECTION : bool
        If True, also plot residual-vs-projection after dividing projection and residual
        (and their SEMs) by max(projection) within each (animal, date).
    var_datapt : str or None
        Low-level condition column (e.g. "labels_1"). Required if FIT_ON="low_level".
        If set, also project each low-level datapoint onto the session OLS line and run LME stats.
    FIT_ON : {"bregion_means", "low_level"}
        What goes into the per-(animal,date) OLS regression:
        "bregion_means" — one (x,y) per bregion (≈8 points).
        "low_level" — all low-level (var_datapt) points that date (requires var_datapt).
        Session-level summaries always evaluate residuals on bregion means projected
        onto that fitted line.
    DIAG_SQUARE : bool
        If True, per-date regression diagnostic panels use equal aspect and square axes.
    DIAG_SHARE_AXES : bool
        If True, share x and y limits across all per-date diagnostic panels.
    RESID_KIND : {"orthogonal", "vertical"}
        Deviation metric used for residual plots, heatmap, and stats:
        "orthogonal" — signed perpendicular distance to the OLS line (geometry in xy).
        "vertical" — classical OLS residual y - (a + b*x).
    savedir : str or None
        Directory for saved figures when SAVE_FIGS is True.
    SAVE_FIGS : bool
        If True, save all figures into savedir (created if needed) with descriptive names,
        and save related stats/tables as CSVs with matching stems
        (e.g. 06_..._stats.csv next to 06_....pdf).
    LL_DODGE_BY_ANIMAL : bool
        For low-level plots that combine animals: if True, x-dodge points by animal within
        each bregion and draw one summary marker (mean±SE) per animal; if False, use the
        previous pooled bar + overlapping jitter style.
    ALPHA_DATAPT : float
        Scatter alpha for low-level (var_datapt) datapoints.
    ALPHA_DATE : float
        Scatter / errorbar alpha for date-level means (one point per date).


    Returns
    -------
    dict with keys:
      DF, DF_sc, DF_plot, DF_diff, stats_date, DF_FIT_INFO
      and if var_datapt is set: DF_ll, DF_diff_ll, stats_ll_resid, stats_ll_diff, mdf_ll_resid
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wilcoxon, sem
    from matplotlib.lines import Line2D

    yvar = "dist_yue_diff"
    dfeffect = DFEFFECT.copy()
    assert FIT_ON in ("bregion_means", "low_level")
    assert RESID_KIND in ("orthogonal", "vertical")
    if FIT_ON == "low_level":
        assert var_datapt is not None, "FIT_ON='low_level' requires var_datapt"
    if SAVE_FIGS:
        assert savedir is not None, "SAVE_FIGS=True requires savedir"
        os.makedirs(savedir, exist_ok=True)

    # One (x,y) per group — same aggregation as the 45° scatter
    def _build_df_xy(dfeffect, datapt_cols=None):
        """datapt_cols: extra grouping cols beyond (animal, date, bregion), e.g. ['labels_1']."""
        grp_cols = ["animal", "date", "bregion"] + (datapt_cols or [])
        rows = []
        for grp, g in dfeffect.groupby(grp_cols):
            animal, date, bregion = grp[:3]
            datapt_vals = {c: v for c, v in zip(datapt_cols or [], grp[3:])}
            gx = g[g["effect"] == eff1]
            gy = g[g["effect"] == eff2]
            if len(gx) == 0 or len(gy) == 0:
                continue
            row = {
                "animal": animal,
                "date": date,
                "bregion": bregion,
                "x": gx[yvar].mean(),
                "y": gy[yvar].mean(),
                "x_sem": sem(gx[yvar]) if len(gx) > 1 else 0.0,
                "y_sem": sem(gy[yvar]) if len(gy) > 1 else 0.0,
            }
            row.update(datapt_vals)
            rows.append(row)
        return pd.DataFrame(rows)

    df_xy = _build_df_xy(dfeffect)
    if var_datapt is not None:
        assert var_datapt in dfeffect.columns, f"{var_datapt} not in DFEFFECT"
        df_xy_ll = _build_df_xy(dfeffect, datapt_cols=[var_datapt])
    else:
        df_xy_ll = None

    df_fit = df_xy_ll if FIT_ON == "low_level" else df_xy

    def _ols_slope_intercept(x_fit, y_fit, through_origin):
        if through_origin:
            denom_xx = np.sum(x_fit ** 2)
            if denom_xx == 0:
                return None, None
            b = float(np.sum(x_fit * y_fit) / denom_xx)
            a = 0.0
        else:
            b, a = np.polyfit(x_fit, y_fit, 1)
            a, b = float(a), float(b)
        return a, b

    def _r2(x_fit, y_fit, a, b, through_origin):
        yhat = a + b * np.asarray(x_fit, dtype=float)
        y_fit = np.asarray(y_fit, dtype=float)
        ss_res = np.sum((y_fit - yhat) ** 2)
        ss_tot = np.sum(y_fit ** 2) if through_origin else np.sum((y_fit - y_fit.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    def _proj_resid(x0, y0, a, b):
        denom = np.sqrt(1.0 + b ** 2)
        residual_ortho = (y0 - a - b * x0) / denom
        projection = (x0 + (y0 - a) * b) / denom
        resid_vertical = y0 - (a + b * x0)
        return projection, residual_ortho, resid_vertical

    def _foot_of_perpendicular(x0, y0, a, b):
        x_f = (x0 + b * (y0 - a)) / (1.0 + b ** 2)
        y_f = a + b * x_f
        return x_f, y_f

    def _proj_resid_sem(x_sem, y_sem, a, b):
        """Error propagation for orthogonal / vertical residuals and projection (a,b fixed)."""
        denom = np.sqrt(1.0 + b ** 2)
        dp_dx, dp_dy = 1.0 / denom, b / denom
        dro_dx, dro_dy = -b / denom, 1.0 / denom
        drv_dx, drv_dy = -b, 1.0
        projection_sem = np.sqrt((dp_dx * x_sem) ** 2 + (dp_dy * y_sem) ** 2)
        residual_ortho_sem = np.sqrt((dro_dx * x_sem) ** 2 + (dro_dy * y_sem) ** 2)
        resid_vertical_sem = np.sqrt((drv_dx * x_sem) ** 2 + (drv_dy * y_sem) ** 2)
        return projection_sem, residual_ortho_sem, resid_vertical_sem

    def _get_ols_params_by_date(df_for_fit, through_origin, fit_mode="loo"):
        """Fit OLS within each (animal, date) on df_for_fit rows.

        Returns
        -------
        params : dict (animal, date, bregion) -> (a, b)
        fit_info : dict (animal, date) -> dict with all-points fit diagnostics
            (a, b, r2, n, used for per-session diagnostic plots).
        """
        params = {}
        fit_info = {}
        for (animal, date), gdate in df_for_fit.groupby(["animal", "date"]):
            bregions = gdate["bregion"].unique().tolist()
            if len(bregions) < 3:
                continue
            # all-points fit (for diagnostics + FIT_MODE='all')
            x_all = gdate["x"].values.astype(float)
            y_all = gdate["y"].values.astype(float)
            a_all, b_all = _ols_slope_intercept(x_all, y_all, through_origin)
            if a_all is None:
                continue
            fit_info[(animal, date)] = {
                "animal": animal,
                "date": date,
                "a": a_all,
                "b": b_all,
                "r2": _r2(x_all, y_all, a_all, b_all, through_origin),
                "n": len(gdate),
                "n_bregion": len(bregions),
                "through_origin": through_origin,
                "fit_mode": fit_mode,
                "fit_on": FIT_ON,
            }
            if fit_mode == "all":
                for bregion in bregions:
                    params[(animal, date, bregion)] = (a_all, b_all)
            else:
                for bregion in bregions:
                    others = gdate[gdate["bregion"] != bregion]
                    if others["bregion"].nunique() < 2:
                        continue
                    a, b = _ols_slope_intercept(
                        others["x"].values.astype(float), others["y"].values.astype(float), through_origin)
                    if a is None:
                        continue
                    params[(animal, date, bregion)] = (a, b)
        return params, fit_info

    def _apply_date_ols_to_datapts(df_xy_pts, params, through_origin, fit_mode, datapt_cols=None):
        """Project each row's (x,y) using the session OLS line for its (animal, date, bregion)."""
        datapt_cols = datapt_cols or []
        rows = []
        for _, row in df_xy_pts.iterrows():
            key = (row["animal"], row["date"], row["bregion"])
            if key not in params:
                continue
            a, b = params[key]
            x0, y0 = float(row["x"]), float(row["y"])
            x_sem, y_sem = float(row["x_sem"]), float(row["y_sem"])
            projection, residual_ortho, resid_vertical = _proj_resid(x0, y0, a, b)
            projection_sem, residual_ortho_sem, resid_vertical_sem = _proj_resid_sem(x_sem, y_sem, a, b)
            out = {
                "animal": row["animal"],
                "date": row["date"],
                "bregion": row["bregion"],
                "x": x0,
                "y": y0,
                "x_sem": x_sem,
                "y_sem": y_sem,
                "slope": b,
                "intercept": a,
                "through_origin": through_origin,
                "fit_mode": fit_mode,
                "fit_on": FIT_ON,
                "projection": projection,
                "residual_ortho": residual_ortho,
                "resid_vertical": resid_vertical,
                "projection_sem": projection_sem,
                "residual_ortho_sem": residual_ortho_sem,
                "resid_vertical_sem": resid_vertical_sem,
                # active residual filled by _select_resid_kind
                "residual": residual_ortho,
                "residual_sem": residual_ortho_sem,
                "ratio": y0 / x0 if x0 != 0 else np.nan,
            }
            for c in datapt_cols:
                out[c] = row[c]
            rows.append(out)
        return pd.DataFrame(rows)

    def _select_resid_kind(df, resid_kind):
        """Set residual / residual_sem from orthogonal or vertical deviation."""
        df = df.copy()
        if resid_kind == "vertical":
            df["residual"] = df["resid_vertical"]
            df["residual_sem"] = df["resid_vertical_sem"]
        else:
            df["residual"] = df["residual_ortho"]
            df["residual_sem"] = df["residual_ortho_sem"]
        df["resid_kind"] = resid_kind
        return df

    def _plot_session_fit_diagnostics(df_for_fit, df_means, fit_info, params, through_origin,
                                      max_sessions=None, square=True, share_axes=False,
                                      resid_kind="orthogonal"):
        """Per (animal, date): regression points, fit line, R2, residual drops of bregion means.

        Displayed line / R² / slope / intercept are always the all-points session fit.
        Residual feet use that same line when FIT_MODE='all', or each bregion's LOO
        line when FIT_MODE='loo' (so feet match the residuals used in analysis).
        resid_kind controls whether drops are orthogonal or vertical.
        """
        sessions = sorted(fit_info.keys())
        if max_sessions is not None:
            sessions = sessions[:max_sessions]
        if len(sessions) == 0:
            return
        ncols = 4
        nrows = int(np.ceil(len(sessions) / ncols))
        panel = 3.4 if square else 3.6
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(panel * ncols, panel * nrows),
            squeeze=False,
            sharex=share_axes, sharey=share_axes)
        for ax, (animal, date) in zip(axes.flatten(), sessions):
            info = fit_info[(animal, date)]
            a_all, b_all = info["a"], info["b"]
            gfit = df_for_fit[(df_for_fit["animal"] == animal) & (df_for_fit["date"] == date)]
            gmean = df_means[(df_means["animal"] == animal) & (df_means["date"] == date)]
            # regression datapoints
            for br in gfit["bregion"].unique():
                g = gfit[gfit["bregion"] == br]
                c = map_bregion_to_color.get(br, "gray")
                ax.scatter(g["x"], g["y"], c=c, s=18, alpha=0.45, edgecolors="none", zorder=2)
            # bregion means + residual drop to fit line
            for _, row in gmean.iterrows():
                br = row["bregion"]
                c = map_bregion_to_color.get(br, "gray")
                x0, y0 = float(row["x"]), float(row["y"])
                x_sem = float(row["x_sem"]) if "x_sem" in row and np.isfinite(row["x_sem"]) else 0.0
                y_sem = float(row["y_sem"]) if "y_sem" in row and np.isfinite(row["y_sem"]) else 0.0
                key = (animal, date, br)
                if FIT_MODE == "loo" and key in params:
                    a_use, b_use = params[key]
                else:
                    a_use, b_use = a_all, b_all
                if resid_kind == "vertical":
                    xf, yf = x0, a_use + b_use * x0
                else:
                    xf, yf = _foot_of_perpendicular(x0, y0, a_use, b_use)
                ax.plot([x0, xf], [y0, yf], color=c, lw=1.0, alpha=0.85, zorder=3)
                ax.scatter([xf], [yf], c=c, s=20, marker="x", zorder=4)
                ax.errorbar(
                    [x0], [y0], xerr=[x_sem], yerr=[y_sem],
                    fmt="o", color=c, ecolor=c, elinewidth=1.0, capsize=0,
                    markersize=6, markeredgecolor="k", markeredgewidth=0.6, zorder=5)
            # fit line spanning data (all-points fit)
            xs = np.concatenate([gfit["x"].values, gmean["x"].values])
            ys = np.concatenate([gfit["y"].values, gmean["y"].values])
            x_lo, x_hi = float(np.min(xs)), float(np.max(xs))
            if through_origin and x_lo > 0:
                x_lo = 0.0
            x_line = np.linspace(x_lo, x_hi, 50)
            ax.plot(x_line, a_all + b_all * x_line, "k-", lw=1.5, zorder=1)
            ax.axhline(0, color="k", alpha=0.2, lw=0.8)
            ax.axvline(0, color="k", alpha=0.2, lw=0.8)
            if square:
                # equal data units + square box; pad slightly beyond data
                y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
                if through_origin and y_lo > 0:
                    y_lo = 0.0
                lo = min(x_lo, y_lo)
                hi = max(x_hi, y_hi)
                pad = 0.05 * (hi - lo) if hi > lo else 0.05
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{animal} {date}\na={a_all:.3g}, b={b_all:.3g}, R²={info['r2']:.3f}, n={info['n']}",
                fontsize=8)
            ax.set_xlabel("x (eff1)", fontsize=8)
            ax.set_ylabel("y (eff2)", fontsize=8)
        for ax in axes.flatten()[len(sessions):]:
            ax.axis("off")
        if share_axes and square:
            # one shared square limit across all used axes
            used = list(axes.flatten()[:len(sessions)])
            xlims = [ax.get_xlim() for ax in used]
            ylims = [ax.get_ylim() for ax in used]
            lo = min(min(x[0] for x in xlims), min(y[0] for y in ylims))
            hi = max(max(x[1] for x in xlims), max(y[1] for y in ylims))
            for ax in used:
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", adjustable="box")
        feet_note = "feet=LOO line per bregion" if FIT_MODE == "loo" else "feet=all-points line"
        drop_note = "vertical drops" if resid_kind == "vertical" else "orthogonal drops"
        fig.suptitle(
            f"Per-session OLS | RESID_KIND={resid_kind} ({drop_note}) | fit_on={FIT_ON} | "
            f"{feet_note} | through_origin={through_origin}\n"
            f"small pts=regression data; large dots=bregion means; lines=residual ({resid_kind})",
            y=1.02, fontsize=10)
        plt.tight_layout()
        return fig

    def _rescale_by_max_projection(DF):
        """Within each (animal, date), divide projection and residual by the max
        projection across all bregions for that session (so max projection = 1)."""
        df = DF.copy()
        df["max_proj"] = df.groupby(["animal", "date"])["projection"].transform("max")
        bad = (df["max_proj"] == 0) | ~np.isfinite(df["max_proj"])
        scale_cols = [
            "projection", "residual", "projection_sem", "residual_sem",
            "residual_ortho", "resid_vertical",
            "residual_ortho_sem", "resid_vertical_sem",
        ]
        scale_cols = [c for c in scale_cols if c in df.columns]
        df.loc[bad, scale_cols] = np.nan
        ok = ~bad
        for c in scale_cols:
            df.loc[ok, c] = df.loc[ok, c] / df.loc[ok, "max_proj"]
        return df

    def _plot_resid_heatmap(DF, title, resid_kind):
        """One heatmap: rows=(animal, date), cols=bregion, color=residual deviation."""
        if DF is None or len(DF) == 0:
            return None
        order = [br for br in map_bregion_to_color if br in DF["bregion"].unique()]
        df = DF.copy()
        df["_session"] = df["animal"].astype(str) + " " + df["date"].astype(str)
        # stable session order: animal then date
        sessions = (
            df[["animal", "date", "_session"]]
            .drop_duplicates()
            .sort_values(["animal", "date"])["_session"]
            .tolist()
        )
        mat = (
            df.pivot_table(index="_session", columns="bregion", values="residual", aggfunc="mean")
            .reindex(index=sessions, columns=order)
        )
        vmax = np.nanmax(np.abs(mat.values)) if np.isfinite(mat.values).any() else 1.0
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        n_row, n_col = mat.shape
        fig_w = max(5.5, 0.7 * n_col + 2.2)
        fig_h = max(3.5, 0.32 * n_row + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            mat, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": f"RESID_KIND={resid_kind}", "shrink": 0.8},
            annot=True, fmt=".2f", annot_kws={"size": 7})
        ax.set_xlabel("bregion")
        ax.set_ylabel("animal date")
        ax.set_title(title)
        plt.tight_layout()
        return fig, mat

    def _plot_resid_vs_proj(DF, ax, title, xlab=None, ylab=None):
        from matplotlib.lines import Line2D
        bregions_plot = [br for br in map_bregion_to_color if br in DF["bregion"].unique()]
        order = [br for br in bregions_plot if br != "preSMA"] + (["preSMA"] if "preSMA" in bregions_plot else [])
        animals = sorted(DF["animal"].unique())
        animal_markers = {a: m for a, m in zip(animals, ["o", "s", "^", "D", "v", "P"])}
        for bregion in order:
            g = DF[DF["bregion"] == bregion]
            c = map_bregion_to_color[bregion]
            # per-date point ± SE; marker by animal, color by bregion
            ax.errorbar(
                g["projection"], g["residual"],
                xerr=g["projection_sem"], yerr=g["residual_sem"],
                fmt="none", ecolor=c, alpha=ALPHA_DATE, elinewidth=0.8, capsize=0,
                zorder=2 if bregion != "preSMA" else 3)
            for animal in animals:
                ga = g[g["animal"] == animal]
                if len(ga) == 0:
                    continue
                ax.scatter(
                    ga["projection"], ga["residual"], c=c,
                    marker=animal_markers[animal],
                    label=bregion if animal == animals[0] else None,
                    alpha=ALPHA_DATE, s=40, edgecolors="none",
                    zorder=3 if bregion == "preSMA" else 2)
            # bregion mean ± SE across dates (square marker, thicker errorbars)
            ax.errorbar(g["projection"].mean(), g["residual"].mean(),
                        xerr=g["projection"].sem(), yerr=g["residual"].sem(),
                        fmt="s", color=c, ecolor=c, elinewidth=2.2, capsize=0,
                        markersize=7, markeredgecolor="k", markeredgewidth=0.6,
                        zorder=5 if bregion == "preSMA" else 4)
        ax.axhline(0, color="k", alpha=0.35, lw=1)
        ax.set_xlabel(xlab or "projection (along OLS line)")
        ax.set_ylabel(ylab or f"residual ({RESID_KIND})")
        ax.set_title(title)
        # bregion color legend + animal marker legend
        br_handles = [
            Line2D([0], [0], marker="s", color="none", markerfacecolor=map_bregion_to_color[br],
                   markersize=7, label=br)
            for br in order]
        an_handles = [
            Line2D([0], [0], marker=animal_markers[a], color="k", linestyle="none",
                   markersize=7, label=a)
            for a in animals]
        leg1 = ax.legend(handles=br_handles, fontsize=7, frameon=False, loc="best", title="bregion")
        ax.add_artist(leg1)
        if len(animals) > 1:
            ax.legend(handles=an_handles, fontsize=7, frameon=False, loc="lower right", title="animal")

    def _plot_resid_vs_proj_by_animal(DF, title, xlab=None, ylab=None):
        """resid vs projection, one panel per animal; each point is one date × bregion."""
        from matplotlib.lines import Line2D
        animals = sorted(DF["animal"].unique())
        if len(animals) == 0:
            return None
        bregions_plot = [br for br in map_bregion_to_color if br in DF["bregion"].unique()]
        order = [br for br in bregions_plot if br != "preSMA"] + (["preSMA"] if "preSMA" in bregions_plot else [])
        fig, axes = plt.subplots(
            1, len(animals), figsize=(5.0 * len(animals), 4.5), sharex=False, sharey=False)
        if len(animals) == 1:
            axes = np.array([axes])
        for ax, animal in zip(axes, animals):
            df_a = DF[DF["animal"] == animal]
            for bregion in order:
                g = df_a[df_a["bregion"] == bregion]
                if len(g) == 0:
                    continue
                c = map_bregion_to_color[bregion]
                # one point per date (± SE from within-date x/y SEM)
                ax.errorbar(
                    g["projection"], g["residual"],
                    xerr=g["projection_sem"], yerr=g["residual_sem"],
                    fmt="none", ecolor=c, alpha=ALPHA_DATE, elinewidth=0.8, capsize=0,
                    zorder=2 if bregion != "preSMA" else 3)
                ax.scatter(
                    g["projection"], g["residual"], c=c, marker="o",
                    label=bregion, alpha=ALPHA_DATE, s=42, edgecolors="none",
                    zorder=3 if bregion == "preSMA" else 2)
                # mean across dates for this animal (square, thicker errorbars)
                ax.errorbar(
                    g["projection"].mean(), g["residual"].mean(),
                    xerr=g["projection"].sem(), yerr=g["residual"].sem(),
                    fmt="s", color=c, ecolor=c, elinewidth=2.2, capsize=0,
                    markersize=7, markeredgecolor="k", markeredgewidth=0.6,
                    zorder=5 if bregion == "preSMA" else 4)
            ax.axhline(0, color="k", alpha=0.35, lw=1)
            ax.set_title(animal)
            ax.set_xlabel(xlab or "projection (along OLS line)")
            ax.set_ylabel(ylab or f"residual ({RESID_KIND})")
        br_handles = [
            Line2D([0], [0], marker="s", color="none", markerfacecolor=map_bregion_to_color[br],
                   markersize=7, label=br)
            for br in order]
        axes[-1].legend(handles=br_handles, fontsize=7, frameon=False, loc="best", title="bregion")
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        return fig

    def _compute_presma_residual_diff(DF, group_cols=("animal", "date"), bregion_ref="preSMA"):
        """For each group and non-ref bregion: residual(other) - residual(ref).

        Positive ⇒ other is above preSMA on the residual axis (above the OLS line
        relative to preSMA).

        residual_diff_sem is from error propagation under independence of the two
        region residuals within a session:
          sem(diff) = sqrt(sem(other)^2 + sem(preSMA)^2)
        where each residual_sem already came from propagating x/y SEM through the
        OLS residual formula (a,b treated as fixed).
        """
        rows = []
        for grp, gdate in DF.groupby(list(group_cols)):
            gref = gdate[gdate["bregion"] == bregion_ref]
            if len(gref) != 1:
                continue
            r_ref = float(gref["residual"].iloc[0])
            r_ref_sem = float(gref["residual_sem"].iloc[0]) if "residual_sem" in gref.columns else 0.0
            if not np.isfinite(r_ref_sem):
                r_ref_sem = 0.0
            grp_meta = grp if isinstance(grp, tuple) else (grp,)
            grp_dict = {c: v for c, v in zip(group_cols, grp_meta)}
            for _, row in gdate.iterrows():
                if row["bregion"] == bregion_ref:
                    continue
                r_other = float(row["residual"])
                r_other_sem = float(row["residual_sem"]) if "residual_sem" in row.index and np.isfinite(row["residual_sem"]) else 0.0
                out = {
                    "bregion_other": row["bregion"],
                    "residual_diff": r_other - r_ref,
                    "residual_diff_sem": float(np.sqrt(r_other_sem ** 2 + r_ref_sem ** 2)),
                    "residual_presma": r_ref,
                    "residual_other": r_other,
                    "residual_presma_sem": r_ref_sem,
                    "residual_other_sem": r_other_sem,
                }
                out.update(grp_dict)
                out.setdefault("animal", row["animal"])
                out.setdefault("date", row["date"])
                rows.append(out)
        return pd.DataFrame(rows)

    def _stats_presma_vs_others_wilcoxon(df_diff, bregion_ref="preSMA", alternative="two-sided"):
        """
        Wilcoxon signed-rank on residual_diff = residual(other) - residual(preSMA),
        one value per (animal, date) session (or per group row in df_diff).

        Default alternative is two-sided (diff != 0). Direction of the effect is in
        median_diff / mean_diff: positive means the other region is more above the
        OLS line than preSMA (the expected scientific direction).
        """
        order = [br for br in map_bregion_to_color if br in df_diff["bregion_other"].unique() and br != bregion_ref]
        rows = []
        for br in order:
            vals = df_diff.loc[df_diff["bregion_other"] == br, "residual_diff"].dropna().values
            if len(vals) < 3:
                rows.append({
                    "bregion_other": br,
                    "n": len(vals),
                    "median_diff": np.nan,
                    "mean_diff": np.nan,
                    "wilcoxon_stat": np.nan,
                    "p": np.nan,
                })
                continue
            try:
                stat, p = wilcoxon(vals, alternative=alternative)
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({
                "bregion_other": br,
                "n": len(vals),
                "median_diff": np.median(vals),
                "mean_diff": np.mean(vals),
                "wilcoxon_stat": stat,
                "p": p,
            })
        return pd.DataFrame(rows)

    def _stats_presma_vs_others_lme(df_diff, bregion_ref="preSMA", rand_grp_list=("animal", "date")):
        """
        Separate intercept-only LME per other bregion on residual_diff:
          residual_diff ~ 1, groups = concatenated (animal, date).
        Tests whether residual(other) - residual(preSMA) differs from 0.
        Positive Intercept ⇒ other more above the OLS line than preSMA.
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        import statsmodels.formula.api as smf

        order = [br for br in map_bregion_to_color if br in df_diff["bregion_other"].unique() and br != bregion_ref]
        rows = []
        for br in order:
            dfm = df_diff.loc[df_diff["bregion_other"] == br].dropna(
                subset=["residual_diff"] + list(rand_grp_list)).reset_index(drop=True)
            if len(dfm) < 5:
                rows.append({
                    "bregion_other": br,
                    "n": len(dfm),
                    "coef": np.nan,
                    "se": np.nan,
                    "mean_diff": dfm["residual_diff"].mean() if len(dfm) else np.nan,
                    "p": np.nan,
                })
                continue
            _, dfm = grouping_append_and_return_inner_items(
                dfm, list(rand_grp_list), new_col_name="_lme_grp", return_df=True)
            try:
                md = smf.mixedlm("residual_diff ~ 1", dfm, groups=dfm["_lme_grp"])
                mdf = md.fit(reml=False)
                rows.append({
                    "bregion_other": br,
                    "n": len(dfm),
                    "coef": mdf.params["Intercept"],
                    "se": mdf.bse["Intercept"],
                    "mean_diff": dfm["residual_diff"].mean(),
                    "p": mdf.pvalues["Intercept"],
                })
            except Exception as err:
                print(f"LME failed for bregion_other={br}: {err}")
                rows.append({
                    "bregion_other": br,
                    "n": len(dfm),
                    "coef": np.nan,
                    "se": np.nan,
                    "mean_diff": dfm["residual_diff"].mean(),
                    "p": np.nan,
                })
        return pd.DataFrame(rows)

    def _stats_lme_datapt(df, y, fixed_col, ref_level, rand_grp_list=("animal", "date")):
        """
        Mixed-effects model at datapoint level: random intercepts for (animal, date).
        fixed_col is categorical (e.g. bregion or bregion_other).
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        import statsmodels.formula.api as smf

        cols = [y, fixed_col] + list(rand_grp_list)
        dfm = df.dropna(subset=cols).copy()
        if len(dfm) < 10 or dfm[fixed_col].nunique() < 2:
            return pd.DataFrame(), None

        _, dfm = grouping_append_and_return_inner_items(
            dfm, list(rand_grp_list), new_col_name="_lme_grp", return_df=True)
        dfm[fixed_col] = dfm[fixed_col].astype(str)
        str_treat = f"C({fixed_col}, Treatment('{ref_level}'))"
        formula = f"{y} ~ {str_treat}"
        try:
            md = smf.mixedlm(formula, dfm, groups=dfm["_lme_grp"])
            mdf = md.fit(reml=False)
        except Exception as err:
            print(f"LME failed ({formula}): {err}")
            return pd.DataFrame(), None

        rows = []
        for exog_name in mdf.params.index:
            if exog_name == "Intercept":
                level = ref_level
            elif exog_name.startswith(f"C({fixed_col}, Treatment("):
                level = exog_name.split("[T.")[1].rstrip("]")
            else:
                level = exog_name
            rows.append({
                fixed_col: level,
                "coef": mdf.params[exog_name],
                "se": mdf.bse[exog_name],
                "p": mdf.pvalues[exog_name],
                "n": len(dfm),
            })
        return pd.DataFrame(rows), mdf

    def _df_presma_diff_with_zero(df_diff):
        """Add preSMA column at y=0 (preSMA minus preSMA) for each datapoint group."""
        meta_cols = [c for c in df_diff.columns
                     if c not in ("bregion_other", "residual_diff", "residual_diff_sem",
                                  "residual_presma", "residual_other",
                                  "residual_presma_sem", "residual_other_sem")]
        df_zero = df_diff[meta_cols].drop_duplicates().copy()
        df_zero["bregion_other"] = "preSMA"
        df_zero["residual_diff"] = 0.0
        df_zero["residual_diff_sem"] = 0.0
        df_zero["residual_presma"] = np.nan
        df_zero["residual_other"] = np.nan
        df_zero["residual_presma_sem"] = np.nan
        df_zero["residual_other_sem"] = np.nan
        return pd.concat([df_zero, df_diff], axis=0, ignore_index=True)

    def _presma_diff_x_order(df_plot):
        """Same bregion order as resid-vs-bregion (map_bregion_to_color), including preSMA."""
        return [br for br in map_bregion_to_color if br in df_plot["bregion_other"].unique()]

    def _fmt_pval(p):
        if not np.isfinite(p):
            return "p=nan"
        if p < 0.001:
            return f"p={p:.1e}"
        return f"p={p:.3f}"

    def _annotate_pvalues(ax, order, stats_col, df_stats, y_col, df, x_col, alpha=0.05, skip_levels=None):
        """Annotate each category with its p-value; red if significant."""
        if df_stats is None or len(df_stats) == 0:
            return
        skip_levels = skip_levels or set()
        ymin, ymax = ax.get_ylim()
        y_span = ymax - ymin if ymax > ymin else 1.0
        for i, lev in enumerate(order):
            if lev in skip_levels:
                continue
            if lev not in df_stats[stats_col].values:
                continue
            p = df_stats.loc[df_stats[stats_col] == lev, "p"].values[0]
            if not np.isfinite(p):
                continue
            g = df.loc[df[x_col] == lev, y_col]
            if len(g) == 0:
                y_pos = ymax - 0.04 * y_span
            else:
                y_pos = g.max() + 0.04 * y_span
            color = "red" if p < alpha else "0.35"
            ax.text(i, y_pos, _fmt_pval(p), ha="center", va="bottom", fontsize=7, color=color, clip_on=True)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))

    def _animal_x_offsets(animals, width=0.28):
        """Map each animal to a fixed x offset within a categorical bin."""
        animals = list(animals)
        n = len(animals)
        if n <= 1:
            return {animals[0]: 0.0} if n == 1 else {}
        offs = np.linspace(-width, width, n)
        return {a: float(o) for a, o in zip(animals, offs)}

    def _plot_presma_residual_diff(df_diff, ax, title, df_stats=None, dodge_by_animal=False,
                                   plot_point_yerr=True, point_alpha=None):
        """point_alpha: override for scatter/errorbar of individual points (date or datapt)."""
        if point_alpha is None:
            point_alpha = ALPHA_DATAPT if dodge_by_animal else ALPHA_DATE
        df_plot = _df_presma_diff_with_zero(df_diff)
        order = _presma_diff_x_order(df_plot)
        animals = sorted(df_plot["animal"].unique())
        animal_markers = {a: m for a, m in zip(animals, ["o", "s", "^", "D", "v", "P"])}
        animal_offsets = _animal_x_offsets(animals) if dodge_by_animal else {a: 0.0 for a in animals}
        rng = np.random.default_rng(0)
        x_map = {br: i for i, br in enumerate(order)}
        jitter = 0.06 if dodge_by_animal else 0.18
        for animal in animals:
            df_a = df_plot[df_plot["animal"] == animal]
            off = animal_offsets[animal]
            for br in order:
                g = df_a.loc[df_a["bregion_other"] == br]
                if len(g) == 0:
                    continue
                xs = x_map[br] + off + rng.uniform(-jitter, jitter, size=len(g))
                ys = g["residual_diff"].values
                c = map_bregion_to_color.get(br, "gray")
                if plot_point_yerr and "residual_diff_sem" in g.columns:
                    ax.errorbar(
                        xs, ys, yerr=g["residual_diff_sem"].values,
                        fmt="none", ecolor=c, alpha=point_alpha, elinewidth=0.8, capsize=0, zorder=2)
                ax.scatter(
                    xs, ys,
                    c=c,
                    marker=animal_markers[animal],
                    s=22 if dodge_by_animal else 28, alpha=point_alpha, edgecolors="none",
                    zorder=3)
                if dodge_by_animal:
                    ax.errorbar(
                        x_map[br] + off, ys.mean(),
                        yerr=sem(ys) if len(ys) > 1 else 0.0,
                        fmt=animal_markers[animal], color=c,
                        ecolor="k", elinewidth=1.2, capsize=0,
                        markersize=7, markeredgecolor="k", markeredgewidth=0.7, zorder=5)
        if not dodge_by_animal:
            for i, br in enumerate(order):
                g = df_plot.loc[df_plot["bregion_other"] == br, "residual_diff"]
                if len(g) == 0:
                    continue
                ax.errorbar(i, g.mean(), yerr=g.sem(), fmt="o", color="k", markersize=6, capsize=0, zorder=5)
        _annotate_pvalues(
            ax, order, "bregion_other", df_stats, "residual_diff", df_plot, "bregion_other",
            skip_levels={"preSMA"})
        ax.axhline(0, color="k", alpha=0.35, lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45)
        ax.set_xlabel("bregion")
        ax.set_ylabel("residual(other) - residual(preSMA)")
        ax.set_title(title)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker=animal_markers[a], color="k", linestyle="none",
                   markersize=7, label=a)
            for a in animals]
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="best", title="animal")

    # Fit OLS on df_fit (bregion means or low-level pts); always evaluate session DF on bregion means
    params, fit_info = _get_ols_params_by_date(
        df_fit, through_origin=FIT_THROUGH_ORIGIN, fit_mode=FIT_MODE)
    DF = _select_resid_kind(
        _apply_date_ols_to_datapts(
            df_xy, params, through_origin=FIT_THROUGH_ORIGIN, fit_mode=FIT_MODE, datapt_cols=[]),
        RESID_KIND)
    DF_FIT_INFO = pd.DataFrame(list(fit_info.values())) if fit_info else pd.DataFrame()
    DF_sc = _rescale_by_max_projection(DF)
    DF_plot = DF_sc if RESCALE_BY_MAX_PROJECTION else DF

    _mode_lbl = "LOO" if FIT_MODE == "loo" else "single fit (all regions)"
    _fit_lbl = "through origin (y=bx)" if FIT_THROUGH_ORIGIN else "with intercept"
    _on_lbl = "low-level pts" if FIT_ON == "low_level" else "bregion means"
    _resid_lbl = "orthogonal dist" if RESID_KIND == "orthogonal" else "vertical residual y-(a+bx)"
    _origin_tag = "origin" if FIT_THROUGH_ORIGIN else "intercept"
    _fig_tag = f"fit-{FIT_MODE}_on-{FIT_ON}_resid-{RESID_KIND}_{_origin_tag}"
    print(f"=== fit_mode={FIT_MODE} | fit_on={FIT_ON} ({_on_lbl}) | resid={RESID_KIND} | {_fit_lbl} ===")
    print(DF_FIT_INFO.to_string(index=False) if len(DF_FIT_INFO) else "(no sessions fit)")
    if SAVE_FIGS:
        print(f"Saving figures to: {savedir}")

    def _savefig(fig, name):
        """Save figure to savedir if SAVE_FIGS; name is stem without extension."""
        if not SAVE_FIGS or fig is None:
            return
        from pythonlib.tools.plottools import savefig
        path = os.path.join(savedir, f"{name}.pdf")
        savefig(fig, path)

    def _savecsv(df, name, index=False):
        """Save dataframe as CSV in savedir if SAVE_FIGS; name is stem without extension."""
        if not SAVE_FIGS or df is None:
            return
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return
        path = os.path.join(savedir, f"{name}.csv")
        df.to_csv(path, index=index)
        print(f"Saved stats table: {path}")

    # Per-date diagnostic: regression points, fit line (all-pts), R², residual drops of means
    fig = _plot_session_fit_diagnostics(
        df_fit, df_xy, fit_info, params, through_origin=FIT_THROUGH_ORIGIN,
        square=DIAG_SQUARE, share_axes=DIAG_SHARE_AXES, resid_kind=RESID_KIND)
    _stem_01 = f"01_session_ols_diagnostics_{_fig_tag}"
    _savefig(fig, _stem_01)
    _savecsv(DF_FIT_INFO, f"{_stem_01}_fit_info")

    # Concise summary heatmap across dates
    _hm_out = _plot_resid_heatmap(
        DF,
        f"{_mode_lbl} | {_on_lbl} | RESID_KIND={RESID_KIND} ({_resid_lbl})\n"
        f"residual by bregion × session",
        RESID_KIND)
    DF_HEAT = _hm_out[1] if _hm_out is not None else None
    _stem_02 = f"02_heatmap_resid_by_bregion_x_session_{_fig_tag}"
    if _hm_out is not None:
        _savefig(_hm_out[0], _stem_02)
        _savecsv(DF_HEAT, f"{_stem_02}_values", index=True)

    # (1) residual vs projection — raw scale
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _plot_resid_vs_proj(
        DF, ax,
        f"{_mode_lbl} | RESID_KIND={RESID_KIND} ({_resid_lbl}) | {_fit_lbl}\n"
        f"(neg residual = below line = lower y/x)")
    plt.tight_layout()
    _savefig(fig, f"03_resid_vs_proj_{_fig_tag}")

    fig = _plot_resid_vs_proj_by_animal(
        DF,
        f"{_mode_lbl} | RESID_KIND={RESID_KIND} ({_resid_lbl}) | {_fit_lbl} | by animal\n"
        f"resid vs proj (datapoint = date)")
    _savefig(fig, f"03b_resid_vs_proj_by_animal_{_fig_tag}")
    
    # (1b) residual vs projection — rescaled so max projection per (animal, date) = 1
    if RESCALE_BY_MAX_PROJECTION:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _plot_resid_vs_proj(
            DF_sc, ax,
            f"{_mode_lbl} | RESID_KIND={RESID_KIND} | {_fit_lbl} — rescaled\n"
            f"(proj, resid) / max_proj(animal, date)",
            xlab="projection / max_proj(animal, date)",
            ylab="residual / max_proj(animal, date)")
        plt.tight_layout()
        _savefig(fig, f"04_resid_vs_proj_rescaled_{_fig_tag}")

        fig = _plot_resid_vs_proj_by_animal(
            DF_sc,
            f"{_mode_lbl} | RESID_KIND={RESID_KIND} | {_fit_lbl} | by animal — rescaled\n"
            f"resid vs proj (datapoint = date)",
            xlab="projection / max_proj(animal, date)",
            ylab="residual / max_proj(animal, date)")
        _savefig(fig, f"04b_resid_vs_proj_by_animal_rescaled_{_fig_tag}")

    # (2) residual by bregion — no projection axis; directly compares regions

    def _plot_resid_by_bregion(DF, ax, title, df_stats=None, plot_point_yerr=False,
                              dodge_by_animal=False, point_alpha=None):
        """Residual vs bregion: points colored by bregion, marker by animal.

        dodge_by_animal=False: pooled bar + overlapping jitter (previous style).
        dodge_by_animal=True: x-dodge points by animal within each bregion, plus one
        mean±SE summary marker per animal.
        point_alpha: override for individual points (defaults: ALPHA_DATAPT if dodge else ALPHA_DATE).
        """
        if point_alpha is None:
            point_alpha = ALPHA_DATAPT if dodge_by_animal else ALPHA_DATE
        from matplotlib.lines import Line2D
        order = [br for br in map_bregion_to_color if br in DF["bregion"].unique()]
        palette = {br: map_bregion_to_color[br] for br in order}
        animals = sorted(DF["animal"].unique())
        animal_markers = {a: m for a, m in zip(animals, ["o", "s", "^", "D", "v", "P"])}
        animal_offsets = _animal_x_offsets(animals) if dodge_by_animal else {a: 0.0 for a in animals}
        x_map = {br: i for i, br in enumerate(order)}
        rng = np.random.default_rng(0)

        if not dodge_by_animal:
            sns.barplot(
                data=DF, x="bregion", y="residual", order=order, palette=palette,
                ax=ax, errorbar="se", capsize=0, errcolor="k", errwidth=1, alpha=0.45, zorder=1)

        jitter = 0.06 if dodge_by_animal else 0.18
        for animal in animals:
            df_a = DF[DF["animal"] == animal]
            off = animal_offsets[animal]
            for br in order:
                g = df_a.loc[df_a["bregion"] == br]
                if len(g) == 0:
                    continue
                xs = x_map[br] + off + rng.uniform(-jitter, jitter, size=len(g))
                ys = g["residual"].values
                c = map_bregion_to_color[br]
                if plot_point_yerr and "residual_sem" in g.columns:
                    ax.errorbar(
                        xs, ys, yerr=g["residual_sem"].values,
                        fmt="none", ecolor=c, alpha=point_alpha, elinewidth=0.8, capsize=0, zorder=2)
                ax.scatter(
                    xs, ys, c=c, marker=animal_markers[animal],
                    s=22 if dodge_by_animal else 28, alpha=point_alpha,
                    edgecolors="none", zorder=3)
                if dodge_by_animal:
                    # one summary marker per animal within this bregion
                    ax.errorbar(
                        x_map[br] + off, ys.mean(),
                        yerr=sem(ys) if len(ys) > 1 else 0.0,
                        fmt=animal_markers[animal], color=c, ecolor="k",
                        elinewidth=1.2, capsize=0, markersize=7,
                        markeredgecolor="k", markeredgewidth=0.7, zorder=5)

        _annotate_pvalues(ax, order, "bregion", df_stats, "residual", DF, "bregion")
        ax.axhline(0, color="k", alpha=0.35, lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45)
        ax.set_xlabel("bregion")
        ax.set_ylabel(f"residual ({RESID_KIND})")
        ax.set_title(title)
        handles = [
            Line2D([0], [0], marker=animal_markers[a], color="k", linestyle="none",
                   markersize=7, label=a)
            for a in animals]
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="best", title="animal")

    def _stats_signed_rank_vs_zero(df, y_col, group_col, levels=None, alternative="two-sided", min_n=3):
        """One-sample Wilcoxon signed-rank vs 0 for each level of group_col."""
        if levels is None:
            levels = [br for br in map_bregion_to_color if br in df[group_col].unique()]
        rows = []
        for lev in levels:
            vals = df.loc[df[group_col] == lev, y_col].dropna().astype(float).values
            if len(vals) < min_n:
                rows.append({
                    group_col: lev, "n": len(vals), "median": np.nan, "mean": np.nan,
                    "wilcoxon_stat": np.nan, "p": np.nan,
                })
                continue
            try:
                stat, p = wilcoxon(vals, alternative=alternative)
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({
                group_col: lev,
                "n": len(vals),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "wilcoxon_stat": stat,
                "p": p,
            })
        return pd.DataFrame(rows)

    def _plot_resid_by_bregion_by_animal(DF, title, plot_point_yerr=False, point_alpha=None):
        """Low-level residual by bregion, one panel per animal (not pooled).

        Overlays Wilcoxon signed-rank p-values (residual vs 0) per bregion.
        """
        if point_alpha is None:
            point_alpha = ALPHA_DATAPT
        order = [br for br in map_bregion_to_color if br in DF["bregion"].unique()]
        palette = {br: map_bregion_to_color[br] for br in order}
        animals = sorted(DF["animal"].unique())
        fig, axes = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 4.2), sharex=False, sharey=False)
        if len(animals) == 1:
            axes = np.array([axes])
        rng = np.random.default_rng(0)
        x_map = {br: i for i, br in enumerate(order)}
        list_stats = []
        for ax, animal in zip(axes, animals):
            df_a = DF[DF["animal"] == animal]
            sns.barplot(
                data=df_a, x="bregion", y="residual", order=order, palette=palette,
                ax=ax, errorbar="se", capsize=0, errcolor="k", errwidth=1, alpha=0.45, zorder=1)
            for br in order:
                g = df_a.loc[df_a["bregion"] == br]
                if len(g) == 0:
                    continue
                xs = x_map[br] + rng.uniform(-0.18, 0.18, size=len(g))
                ys = g["residual"].values
                c = map_bregion_to_color[br]
                if plot_point_yerr and "residual_sem" in g.columns:
                    ax.errorbar(
                        xs, ys, yerr=g["residual_sem"].values,
                        fmt="none", ecolor=c, alpha=point_alpha, elinewidth=0.8, capsize=0, zorder=2)
                ax.scatter(xs, ys, c=c, marker="o", s=22, alpha=point_alpha, edgecolors="none", zorder=3)
            stats_a = _stats_signed_rank_vs_zero(df_a, "residual", "bregion", levels=order)
            stats_a.insert(0, "animal", animal)
            list_stats.append(stats_a)
            _annotate_pvalues(ax, order, "bregion", stats_a, "residual", df_a, "bregion")
            ax.axhline(0, color="k", alpha=0.35, lw=1)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45)
            ax.set_title(animal)
            ax.set_xlabel("bregion")
            ax.set_ylabel(f"residual ({RESID_KIND})")
        fig.suptitle(title + "\nWilcoxon signed-rank: residual vs 0", y=1.04)
        plt.tight_layout()
        df_stats = pd.concat(list_stats, ignore_index=True) if list_stats else pd.DataFrame()
        return fig, df_stats

    def _plot_presma_residual_diff_by_animal(df_diff, title, point_alpha=None):
        """other-preSMA residual diff, one panel per animal; includes preSMA at y=0.

        Overlays Wilcoxon signed-rank p-values (residual_diff vs 0) per bregion.
        """
        if point_alpha is None:
            point_alpha = ALPHA_DATAPT
        df_plot = _df_presma_diff_with_zero(df_diff)
        order = _presma_diff_x_order(df_plot)
        palette = {br: map_bregion_to_color.get(br, "gray") for br in order}
        animals = sorted(df_plot["animal"].unique())
        fig, axes = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 4.2), sharex=False, sharey=False)
        if len(animals) == 1:
            axes = np.array([axes])
        list_stats = []
        for ax, animal in zip(axes, animals):
            df_a = df_plot[df_plot["animal"] == animal]
            # stats on real diffs only (exclude the synthetic preSMA zeros)
            df_a_stats = df_diff[df_diff["animal"] == animal]
            sns.stripplot(
                data=df_a, x="bregion_other", y="residual_diff", order=order, palette=palette,
                ax=ax, alpha=point_alpha, size=3, jitter=0.2)
            for i, br in enumerate(order):
                g = df_a.loc[df_a["bregion_other"] == br, "residual_diff"]
                if len(g) == 0:
                    continue
                ax.errorbar(i, g.mean(), yerr=g.sem(), fmt="o", color="k", markersize=5, capsize=0, zorder=5)
            stats_a = _stats_signed_rank_vs_zero(
                df_a_stats, "residual_diff", "bregion_other",
                levels=[br for br in order if br != "preSMA"])
            stats_a.insert(0, "animal", animal)
            list_stats.append(stats_a)
            _annotate_pvalues(
                ax, order, "bregion_other", stats_a, "residual_diff", df_a, "bregion_other",
                skip_levels={"preSMA"})
            ax.axhline(0, color="k", alpha=0.35, lw=1)
            ax.set_title(animal)
            ax.set_xlabel("bregion")
            ax.set_ylabel("residual(other) - residual(preSMA)")
            ax.tick_params(axis="x", rotation=45)
        fig.suptitle(title + "\nWilcoxon signed-rank: residual_diff vs 0", y=1.04)
        plt.tight_layout()
        df_stats = pd.concat(list_stats, ignore_index=True) if list_stats else pd.DataFrame()
        return fig, df_stats

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _plot_resid_by_bregion(
        DF, ax, f"{_mode_lbl} | {_fit_lbl}\nresidual by bregion",
        plot_point_yerr=True, point_alpha=ALPHA_DATE)
    plt.tight_layout()
    _savefig(fig, f"05_resid_by_bregion_{_fig_tag}")

    # (3) preSMA vs each other bregion: residual(other) - residual(preSMA), per (animal, date)
    DF_diff = _compute_presma_residual_diff(DF)
    stats_date = _stats_presma_vs_others_wilcoxon(DF_diff)
    print(f"=== other vs preSMA | Wilcoxon signed-rank | {_fit_lbl} ===")
    print(stats_date.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _plot_presma_residual_diff(
        DF_diff, ax,
        f"{_mode_lbl} | {_fit_lbl}\nresidual(other) - residual(preSMA)",
        df_stats=stats_date, point_alpha=ALPHA_DATE)
    plt.tight_layout()
    _stem_06 = f"06_other_minus_presma_resid_{_fig_tag}"
    _savefig(fig, _stem_06)
    _savecsv(stats_date, f"{_stem_06}_stats")

    # (4) low-level conditions: same session OLS params, residuals per individual datapt
    out_ll = {}
    if var_datapt is not None and df_xy_ll is not None and len(df_xy_ll) > 0:
        group_cols_ll = ("animal", "date", var_datapt)
        DF_ll = _select_resid_kind(
            _apply_date_ols_to_datapts(
                df_xy_ll, params, through_origin=FIT_THROUGH_ORIGIN, fit_mode=FIT_MODE, datapt_cols=[var_datapt]),
            RESID_KIND)

        print(f"=== low-level ({var_datapt}) | n datapts per bregion ===")
        print(DF_ll.groupby("bregion")["residual"].agg(["count", "mean", "sem"]))

        DF_diff_ll = _compute_presma_residual_diff(DF_ll, group_cols=group_cols_ll)

        # LME at datapoint level: RE = (animal, date), FE = bregion
        stats_ll_resid, mdf_ll_resid = _stats_lme_datapt(
            DF_ll, "residual", "bregion", ref_level="preSMA")

        # Separate intercept-only LME per other bregion on residual_diff
        stats_ll_diff = _stats_presma_vs_others_lme(DF_diff_ll)

        print(f"=== LME residual ~ bregion | low-level ({var_datapt}) | RE(animal,date) | {_fit_lbl} ===")
        print(stats_ll_resid.to_string(index=False))
        print(f"=== LME residual_diff ~ 1 per bregion_other | low-level ({var_datapt}) | RE(animal,date) | {_fit_lbl} ===")
        print(stats_ll_diff.to_string(index=False))

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _plot_resid_by_bregion(
            DF_ll, ax,
            f"{_mode_lbl} | {_on_lbl} | {_fit_lbl}\nresidual by bregion ({var_datapt})",
            df_stats=stats_ll_resid,
            dodge_by_animal=LL_DODGE_BY_ANIMAL,
            point_alpha=ALPHA_DATAPT)
        plt.tight_layout()
        _stem_07 = f"07_ll_resid_by_bregion_{var_datapt}_{_fig_tag}"
        _savefig(fig, _stem_07)
        _savecsv(stats_ll_resid, f"{_stem_07}_stats")

        fig, stats_ll_resid_by_animal = _plot_resid_by_bregion_by_animal(
            DF_ll,
            f"{_mode_lbl} | {_on_lbl} | {_fit_lbl} | by animal\nresidual by bregion ({var_datapt})",
            point_alpha=ALPHA_DATAPT)
        _stem_08 = f"08_ll_resid_by_bregion_by_animal_{var_datapt}_{_fig_tag}"
        _savefig(fig, _stem_08)
        _savecsv(stats_ll_resid_by_animal, f"{_stem_08}_stats")
        print(f"=== Wilcoxon residual vs 0 | low-level by animal ({var_datapt}) ===")
        print(stats_ll_resid_by_animal.to_string(index=False))

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _plot_presma_residual_diff(
            DF_diff_ll, ax,
            f"{_mode_lbl} | {_on_lbl} | {_fit_lbl}\nother-preSMA residual ({var_datapt})",
            df_stats=stats_ll_diff,
            dodge_by_animal=LL_DODGE_BY_ANIMAL,
            point_alpha=ALPHA_DATAPT)
        plt.tight_layout()
        _stem_09 = f"09_ll_other_minus_presma_{var_datapt}_{_fig_tag}"
        _savefig(fig, _stem_09)
        _savecsv(stats_ll_diff, f"{_stem_09}_stats")

        fig, stats_ll_diff_by_animal = _plot_presma_residual_diff_by_animal(
            DF_diff_ll,
            f"{_mode_lbl} | {_on_lbl} | {_fit_lbl} | by animal\nother-preSMA ({var_datapt})",
            point_alpha=ALPHA_DATAPT)
        _stem_10 = f"10_ll_other_minus_presma_by_animal_{var_datapt}_{_fig_tag}"
        _savefig(fig, _stem_10)
        _savecsv(stats_ll_diff_by_animal, f"{_stem_10}_stats")
        print(f"=== Wilcoxon residual_diff vs 0 | low-level by animal ({var_datapt}) ===")
        print(stats_ll_diff_by_animal.to_string(index=False))

        out_ll = {
            "DF_ll": DF_ll,
            "DF_diff_ll": DF_diff_ll,
            "stats_ll_resid": stats_ll_resid,
            "stats_ll_diff": stats_ll_diff,
            "stats_ll_resid_by_animal": stats_ll_resid_by_animal,
            "stats_ll_diff_by_animal": stats_ll_diff_by_animal,
            "mdf_ll_resid": mdf_ll_resid,
        }

    return {
        "DF": DF,
        "DF_sc": DF_sc,
        "DF_plot": DF_plot,
        "DF_diff": DF_diff,
        "stats_date": stats_date,
        "DF_FIT_INFO": DF_FIT_INFO,
        "DF_HEAT": DF_HEAT,
        **out_ll,
    }

def final_dfeffect_load_dfgaps(dfeffect, animal, savedir):
    """
    Load dfgaps for analysis that is...
    
    Relating stregnth of encoding of "chunk_within_rank" to speed of gaps
    (positive correlation).
    """

    assert len(dfeffect["animal"].unique()) == 1
    list_date = dfeffect["date"].unique().tolist()

    DFGAPS, map_gapsemantic_to_dfgaps = final_dfeffect_load_dfgaps_inner(list_date, animal, savedir)

    return DFGAPS, map_gapsemantic_to_dfgaps

    # # Load pre-saved gap durations
    # # Collect across dates
    # from glob import glob

    # list_dfgaps = []
    # for _date in list_date:

    #     searchstr = f"/lemur2/lucas/analyses/main/syntax_gap_durations/{animal}_{_date}_*/dfgaps.pkl"
    #     # path = f"/lemur2/lucas/analyses/main/syntax_gap_durations/{animal}_{_date}_dirgrammardiego3b/dfgaps.pkl"
    #     list_path = glob(searchstr)
    #     assert len(list_path)==1
    #     path = list_path[0]
    #     dfgaps = pd.read_pickle(path)
    #     dfgaps["date"] = _date
    #     dfgaps["animal"] = animal

    #     list_dfgaps.append(dfgaps)
    # DFGAPS = pd.concat(list_dfgaps).reset_index(drop=True)    

    # ### Preprocess
    # # - first get cr and shape for prev and next stroke.
    # DFGAPS["shape_prev"] = [x[0] for x in DFGAPS["gap_shape"]]
    # DFGAPS["shape_next"] = [x[1] for x in DFGAPS["gap_shape"]]

    # DFGAPS["chunk_rank_prev"] = [x[0] for x in DFGAPS["gap_chunk_rank"]]
    # DFGAPS["chunk_rank_next"] = [x[1] for x in DFGAPS["gap_chunk_rank"]]

    # # convert cr to ints where -1 means prev stroke was start, and 99 means next stroke is "done"
    # DFGAPS["chunk_rank_prev"] = [int(x) if not np.isnan(x) else int(-1) for x in DFGAPS["chunk_rank_prev"]]
    # DFGAPS["chunk_rank_next"] = [int(x) if not np.isnan(x) else int(99) for x in DFGAPS["chunk_rank_next"]]

    # # Eclude gaps from onset and offset
    # DFGAPS = DFGAPS[(DFGAPS["chunk_rank_prev"]>-1) & (DFGAPS["chunk_rank_next"]<99)].reset_index(drop=True)

    # # Relabel gaps by ("gap_semantic_vs_prev_stroke")
    # def F(diff_chunk_rank_global):
    #     if diff_chunk_rank_global!=0:
    #         gap_semantic_vs_prev_stroke = "new_chk"
    #     else:
    #         assert diff_chunk_rank_global == 0
    #         gap_semantic_vs_prev_stroke = "within_chk"
    #     return gap_semantic_vs_prev_stroke
    # DFGAPS["gap_semantic_vs_prev_stroke"] = DFGAPS["diff_chunk_rank_global"].apply(F)

    # # Sanity check that new variables make sense.
    # from pythonlib.tools.pandastools import grouping_print_n_samples
    # # grouping_print_n_samples(DFGAPS, ["animal", "date", "gap_semantic_vs_prev_stroke", "diff_chunk_rank_global", "gap_shape","gap_chunk_rank", "gap_chunk_within_rank"])
    # savepath = f"{savedir}/counts_gap_semantic-1.txt"
    # grouping_print_n_samples(DFGAPS, 
    #                          ["gap_semantic_vs_prev_stroke", "diff_chunk_rank_global", "gap_shape","gap_chunk_rank", "gap_chunk_within_rank"],
    #                          savepath=savepath)
    # savepath = f"{savedir}/counts_gap_semantic-2.txt"
    # grouping_print_n_samples(DFGAPS, 
    #                          ["date", "gap_semantic_vs_prev_stroke", "diff_chunk_rank_global", "gap_shape","gap_chunk_rank", "gap_chunk_within_rank"],
    #                          savepath=savepath)

    # # Restrict to cases that are "canonical" transitions (ie not skipping a gap)
    # n1 = len(DFGAPS)
    # DFGAPS = DFGAPS[DFGAPS["diff_chunk_rank_global"].isin([0, 1])].reset_index(drop=True)
    # n2 = len(DFGAPS)
    # assert n2/n1 > 0.8, "why throw out so many? is this due to weird labels on 2-shape days?"

    # ### Also get final agged (ie one datapt per chunk)
    # # Aggregate so that each (cr, shape) gets two gap timings: (i) within and (ii) transition to next.
    # from pythonlib.tools.pandastools import aggregGeneral
    # map_gapsemantic_to_dfgaps = {}
    # for which_gap_semantic_higher in ["within_chk", "to_next_chk", "from_prev_chk"]:

    #     if which_gap_semantic_higher == "within_chk":
    #         # Get gaps within the chunk
    #         which_gap_semantic = "within_chk" # gaps within chunk
    #         var_chunk_rank = "chunk_rank_prev"
    #         var_shape = "shape_prev"
    #     elif which_gap_semantic_higher == "to_next_chk":
    #         # Get gaps after this chunk finishes
    #         which_gap_semantic = "new_chk" # gaps within chunk
    #         var_chunk_rank = "chunk_rank_prev"
    #         var_shape = "shape_prev"
    #     elif which_gap_semantic_higher == "from_prev_chk":
    #         # Get gap that led into this chunk
    #         # (agg so that the following chunk is relevant)
    #         which_gap_semantic = "new_chk" # gaps within chunk
    #         var_chunk_rank = "chunk_rank_next"
    #         var_shape = "shape_next"
    #     else:
    #         assert False

    #     DFGAPS_AGG = aggregGeneral(DFGAPS, ["animal", "date", var_chunk_rank, var_shape, "gap_semantic_vs_prev_stroke"], ["gap_dur"])

    #     # Give a label for dfgaps that will be used for aligning to neural data
    #     assert len(DFGAPS_AGG["animal"].unique()) == 1, "assuming I can ignore animal, below"
    #     assert len(dfeffect["animal"].unique()) == 1

    #     if False: # Not anymore, since I ensure is all ints in neural data
    #         DFGAPS_AGG[var_chunk_rank] = DFGAPS_AGG[var_chunk_rank].astype(float)
    #     DFGAPS_AGG = append_col_with_grp_index(DFGAPS_AGG, ["date", var_chunk_rank, var_chunk_rank, var_shape, var_shape], "da_cr_sh_12")

    #     # Finally, merge gaps and neural data.
    #     # - Create a new column in neurel data: gap duration
    #     dfgaps_agg = DFGAPS_AGG[DFGAPS_AGG["gap_semantic_vs_prev_stroke"] == which_gap_semantic].reset_index(drop=True)

    #     map_gapsemantic_to_dfgaps[which_gap_semantic_higher] = dfgaps_agg

    #     savepath = f"{savedir}/counts_gap_semantic-which_gap_semantic_higher={which_gap_semantic_higher}.txt"
    #     grouping_print_n_samples(dfgaps_agg, 
    #                             ["animal", "date", var_chunk_rank, var_shape, "gap_semantic_vs_prev_stroke"],
    #                             savepath=savepath)

    # return DFGAPS, map_gapsemantic_to_dfgaps

def final_dfeffect_load_dfgaps_inner(list_date, animal, savedir):
    """
    Load dfgaps for analysis that is...
    
    Relating stregnth of encoding of "chunk_within_rank" to speed of gaps
    (positive correlation).

    RETURNS:
    - DFGAPS, one datapt per trial (ie gap)
    - map_gapsemantic_to_dfgaps, after agging, one datapt per chunk_rank
    """
    # Load pre-saved gap durations
    # Collect across dates
    from glob import glob

    list_dfgaps = []
    for _date in list_date:

        searchstr = f"/lemur2/lucas/analyses/main/syntax_gap_durations/{animal}_{_date}_*/dfgaps.pkl"
        # path = f"/lemur2/lucas/analyses/main/syntax_gap_durations/{animal}_{_date}_dirgrammardiego3b/dfgaps.pkl"
        list_path = glob(searchstr)
        if len(list_path)!=1:
            print(list_path)
            print(searchstr)
            assert False, "not getting path for this... Need to extract raw gaps data?"
        path = list_path[0]
        dfgaps = pd.read_pickle(path)
        dfgaps["date"] = _date
        dfgaps["animal"] = animal

        list_dfgaps.append(dfgaps)
    DFGAPS = pd.concat(list_dfgaps).reset_index(drop=True)    

    ### Preprocess
    # - first get cr and shape for prev and next stroke.
    DFGAPS["shape_prev"] = [x[0] for x in DFGAPS["gap_shape"]]
    DFGAPS["shape_next"] = [x[1] for x in DFGAPS["gap_shape"]]

    DFGAPS["chunk_rank_prev"] = [x[0] for x in DFGAPS["gap_chunk_rank"]]
    DFGAPS["chunk_rank_next"] = [x[1] for x in DFGAPS["gap_chunk_rank"]]

    # convert cr to ints where -1 means prev stroke was start, and 99 means next stroke is "done"
    DFGAPS["chunk_rank_prev"] = [int(x) if not np.isnan(x) else int(-1) for x in DFGAPS["chunk_rank_prev"]]
    DFGAPS["chunk_rank_next"] = [int(x) if not np.isnan(x) else int(99) for x in DFGAPS["chunk_rank_next"]]

    # Eclude gaps from onset and offset
    DFGAPS = DFGAPS[(DFGAPS["chunk_rank_prev"]>-1) & (DFGAPS["chunk_rank_next"]<99)].reset_index(drop=True)

    # Relabel gaps by ("gap_semantic_vs_prev_stroke")
    def F(diff_chunk_rank_global):
        # if diff_chunk_rank_global!=0: # previously was this, but I changed to >0 as that seems tighter
        if diff_chunk_rank_global>0:
            gap_semantic_vs_prev_stroke = "new_chk"
        else:
            assert diff_chunk_rank_global == 0
            gap_semantic_vs_prev_stroke = "within_chk"
        return gap_semantic_vs_prev_stroke
    DFGAPS["gap_semantic_vs_prev_stroke"] = DFGAPS["diff_chunk_rank_global"].apply(F)

    # Sanity check that new variables make sense.
    from pythonlib.tools.pandastools import grouping_print_n_samples
    # grouping_print_n_samples(DFGAPS, ["animal", "date", "gap_semantic_vs_prev_stroke", "diff_chunk_rank_global", "gap_shape","gap_chunk_rank", "gap_chunk_within_rank"])
    savepath = f"{savedir}/counts_gap_semantic-1.txt"
    grouping_print_n_samples(DFGAPS, 
                             ["gap_semantic_vs_prev_stroke", "diff_chunk_rank_global", "gap_shape","gap_chunk_rank", "gap_chunk_within_rank"],
                             savepath=savepath)
    savepath = f"{savedir}/counts_gap_semantic-2.txt"
    grouping_print_n_samples(DFGAPS, 
                             ["date", "gap_semantic_vs_prev_stroke", "diff_chunk_rank_global", "gap_shape","gap_chunk_rank", "gap_chunk_within_rank"],
                             savepath=savepath)

    # Restrict to cases that are "canonical" transitions (ie not skipping a gap)
    n1 = len(DFGAPS)
    DFGAPS = DFGAPS[DFGAPS["diff_chunk_rank_global"].isin([0, 1])].reset_index(drop=True)
    n2 = len(DFGAPS)
    assert n2/n1 > 0.8, "why throw out so many? is this due to weird labels on 2-shape days?"

    ### Also get final agged (ie one datapt per chunk)
    # Aggregate so that each (cr, shape) gets two gap timings: (i) within and (ii) transition to next.
    from pythonlib.tools.pandastools import aggregGeneral
    map_gapsemantic_to_dfgaps = {}
    for which_gap_semantic_higher in ["within_chk", "to_next_chk", "from_prev_chk"]:

        if which_gap_semantic_higher == "within_chk":
            # Get gaps within the chunk
            which_gap_semantic = "within_chk" # gaps within chunk
            var_chunk_rank = "chunk_rank_prev"
            var_shape = "shape_prev"
        elif which_gap_semantic_higher == "to_next_chk":
            # Get gaps after this chunk finishes
            which_gap_semantic = "new_chk" # gaps within chunk
            var_chunk_rank = "chunk_rank_prev"
            var_shape = "shape_prev"
        elif which_gap_semantic_higher == "from_prev_chk":
            # Get gap that led into this chunk
            # (agg so that the following chunk is relevant)
            which_gap_semantic = "new_chk" # gaps within chunk
            var_chunk_rank = "chunk_rank_next"
            var_shape = "shape_next"
        else:
            assert False

        DFGAPS_AGG = aggregGeneral(DFGAPS, ["animal", "date", var_chunk_rank, var_shape, "gap_semantic_vs_prev_stroke"], ["gap_dur"])

        # Give a label for dfgaps that will be used for aligning to neural data
        assert len(DFGAPS_AGG["animal"].unique()) == 1, "assuming I can ignore animal, below"

        if False: # Not anymore, since I ensure is all ints in neural data
            DFGAPS_AGG[var_chunk_rank] = DFGAPS_AGG[var_chunk_rank].astype(float)
        DFGAPS_AGG = append_col_with_grp_index(DFGAPS_AGG, ["date", var_chunk_rank, var_chunk_rank, var_shape, var_shape], "da_cr_sh_12")

        # Finally, merge gaps and neural data.
        # - Create a new column in neurel data: gap duration
        dfgaps_agg = DFGAPS_AGG[DFGAPS_AGG["gap_semantic_vs_prev_stroke"] == which_gap_semantic].reset_index(drop=True)

        map_gapsemantic_to_dfgaps[which_gap_semantic_higher] = dfgaps_agg

        savepath = f"{savedir}/counts_gap_semantic-which_gap_semantic_higher={which_gap_semantic_higher}.txt"
        grouping_print_n_samples(dfgaps_agg, 
                                ["animal", "date", var_chunk_rank, var_shape, "gap_semantic_vs_prev_stroke"],
                                savepath=savepath)

    return DFGAPS, map_gapsemantic_to_dfgaps 

def _final_dfeffect_postprocess_clean(dfgeneric, analysis, animal):
    """
    """

    ### Dates 
    if animal=="Diego":
        # dates_remove = [250319, 250321]
        dates_remove = [250319, 250321, 250416, 250417] # Generally exclude, recent, with the latter 3 including cross-AB
        if analysis == "pig_vs_sp":
            # Testing, the above makes more sense.
            # dates_remove = []
            dates_remove = [250319, 250321, 250416, 250417] # TESTING
        if analysis=="two_shapes":
            dates_remove = [250319, 250321, 250416, 250417]
            # dates_remove = []
    elif animal=="Pancho":
        # dates_remove = [250322]
        dates_remove = [240830, 250322]
        if analysis=="rank_up_vs_down":
            dates_remove.append(230811) # this is low N tasks and low variability
        # if analysis=="two_shapes":
        #     dates_remove.append(220909) # this is low N tasks and low variability
        # if analysis == "pig_vs_sp":
        #     # Testing, the above makes more sense.
        #     dates_remove = []
    else:
        assert False
    dfgeneric = dfgeneric[~(dfgeneric["date"].isin(dates_remove))].reset_index(drop=True)

    ### Make sure these are ints.
    from pythonlib.tools.pandastools import integerify_values
    for col in ["chunk_rank", "chunk_within_rank"]:
        integerify_values(dfgeneric, f"{col}_1")
        integerify_values(dfgeneric, f"{col}_2")
        dfgeneric = append_col_with_grp_index(dfgeneric, [f"{col}_1", f"{col}_2"], f"{col}_12")                

    return dfgeneric

def final_dfeffect_postprocess(DFEFFECT_ALL, animal, analysis, savedir, n_min_trials_per_label=2, 
                               HACK_dates=None, two_shapes_remove_probe_trials=True):
    """
    Helper to clean up DFEFFECT_ALL, which holds data for all dates

    LT CHECKED (only for sp_vs_pig)
    """
    from neuralmonkey.analyses.euclidian_distance import dfdist_postprocess_condition_prune_to_var_pairs_exist
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good, append_col_with_grp_index
    from neuralmonkey.analyses.euclidian_distance import dfdist_convert_merge_pair_to_get_all_levels
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

    assert not isinstance(savedir, int)

    ### [Optional] if you want to merge questions (e.g, diff dates have diff questions)
    if analysis == "pig_vs_sp":
        if animal=="Pancho":
            if HACK_dates is None:
                # _dates = [230810, 230811, 230829, 240830, 250322]
                _dates = [230810, 230811, 230829, 231116, 240830]
            elif HACK_dates == 0:
                _dates = [230810, 230811, 230829]
            elif HACK_dates == 1:
                _dates = [230810, 230811, 230829, 240830]
            elif HACK_dates == 2:
                _dates = [230810, 230811, 230829, 250322]
            else:
                assert False         

            map_question_to_dates = {
                # "4c":[231114, 231116],
                "4c":[231114],
                "4":_dates,
            }
        elif animal=="Diego":
            map_question_to_dates = {
                "4c":[230816, 230817, 231116, 240822, 250319],
                "4":[230723, 230724, 230726, 230727, 230728, 230730, 230815, 230913, 230914, 230915, 231118, 240827, 250321],
            }
        else:
            assert False
    elif analysis in ["two_shapes", "rank_within", "rank_up_vs_down", "n_in_chunk"]:
        # Don't try to merge questions
        map_question_to_dates = None
    else:
        print(analysis)
        assert False

    ### Optionally, rename effects that are across questions (but whihc you belioeve should be the sanme name)
    if analysis == "pig_vs_sp":
        list_df = []
        for q, list_dates in map_question_to_dates.items():
            eff1 = f"{q}_shapePIG_ss=shape|none|none"
            eff2 = f"{q}_shapeSP_ss=shape|none|none"
            list_effect = [eff1, eff2]
            map_effect_to_neweffect = {
                eff1:"shapePIG",
                eff2:"shapeSP",
            }
            for date in list_dates:
                df = DFEFFECT_ALL[(DFEFFECT_ALL["date"] == date) & (DFEFFECT_ALL["effect"].isin(list_effect))].reset_index(drop=True)
                df["effect"] = [map_effect_to_neweffect[eff] for eff in df["effect"]]    
                # if len(df)==0:
                #     assert False, f"Missing this date: {date} for this animal {animal}"
                list_df.append(df)    
        DFEFFECT = pd.concat(list_df).reset_index(drop=True)
        eff1 = "shapePIG"
        eff2 = "shapeSP"
        DFEFFECT = DFEFFECT[DFEFFECT["effect"].isin([eff1, eff2])].reset_index(drop=True)
    elif analysis == "two_shapes":
        eff1 = "11_2SH_syntax" # x
        eff2 = "11_2SH_epochshape"
        # Keep just those dates with both effects existing
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        DFEFFECT, _ = extract_with_levels_of_conjunction_vars_helper(DFEFFECT_ALL, "effect", ["date", "bregion"], levels_var=[eff1, eff2])
        # Keep only those desired effects
        DFEFFECT = DFEFFECT[DFEFFECT["effect"].isin([eff1, eff2])].reset_index(drop=True)
    elif analysis == "rank_within":
        eff1 = "11_shape_ss=global"
        eff2 = "11_rankwithin_ss=global" 
        # Keep just those dates with both effects existing
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        DFEFFECT, _ = extract_with_levels_of_conjunction_vars_helper(DFEFFECT_ALL, "effect", ["date", "bregion"], levels_var=[eff1, eff2])
        # Keep only those desired effects
        DFEFFECT = DFEFFECT[DFEFFECT["effect"].isin([eff1, eff2])].reset_index(drop=True)
    elif analysis == "rank_up_vs_down":
        eff1 = "14_rankwithin_dn"
        eff2 = "14_rankwithin_up"
        # Keep just those dates with both effects existing
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        DFEFFECT, _ = extract_with_levels_of_conjunction_vars_helper(DFEFFECT_ALL, "effect", ["date", "bregion"], levels_var=[eff1, eff2])
        # Keep only those desired effects
        DFEFFECT = DFEFFECT[DFEFFECT["effect"].isin([eff1, eff2])].reset_index(drop=True)
    elif analysis == "n_in_chunk":
        eff1 = "11_motor_ss=global"
        eff2 = "11_ninchunk_ss=global"
        # Keep just those dates with both effects existing
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        DFEFFECT, _ = extract_with_levels_of_conjunction_vars_helper(DFEFFECT_ALL, "effect", ["date", "bregion"], levels_var=[eff1, eff2])
        # Keep only those desired effects
        DFEFFECT = DFEFFECT[DFEFFECT["effect"].isin([eff1, eff2])].reset_index(drop=True)
    else:
        assert False

    ### Dates 
    DFEFFECT = _final_dfeffect_postprocess_clean(DFEFFECT, analysis, animal)
    # if animal=="Diego":
    #     dates_remove = [250319, 250321]
    # elif animal=="Pancho":
    #     dates_remove = [250322]
    #     if analysis=="rank_up_vs_down":
    #         dates_remove.append(230811) # this is low N tasks and low variability
    # else:
    #     assert False
    # DFEFFECT = DFEFFECT[~(DFEFFECT["date"].isin(dates_remove))].reset_index(drop=True)

    # ### Make sure these are ints.
    # from pythonlib.tools.pandastools import integerify_values
    # for col in ["chunk_rank", "chunk_within_rank"]:
    #     integerify_values(DFEFFECT, f"{col}_1")
    #     integerify_values(DFEFFECT, f"{col}_2")
    #     DFEFFECT = append_col_with_grp_index(DFEFFECT, [f"{col}_1", f"{col}_2"], f"{col}_12")                

    ### For "two shapes" extra thigs to do.
    if analysis == "two_shapes":
        print("Two shapes, before removing cross-AB probes and aligning to chunk_rank_global: ", len(DFEFFECT))
        # (1) First, get the "pure" cr and shape info (not 1, 2)
        list_vars = ["epoch", "chunk_rank", "shape"]
        vars_others = ["date", "bregion"]
        dfmerged = dfdist_convert_merge_pair_to_get_all_levels(DFEFFECT, list_vars, vars_others)
        
        # Print how it looks before realigning.
        savepath = f"{savedir}/two_shapes-counts-before_prune_and_align_cr_global.txt"
        grouping_print_n_samples(dfmerged, ["date", "epoch", "chunk_rank", "shape"], savepath=savepath)
        if False:
            # grouping_plot_n_samples_conjunction_heatmap(dfmerged, "chunk_rank", "shape", ["date", "epoch"])
            grouping_plot_n_samples_conjunction_heatmap(dfmerged, "chunk_rank", "shape", ["date"])

        # (2) Remove probe trials that are cross-AB.
        assert two_shapes_remove_probe_trials in [False, True], "need to give bool"
        if two_shapes_remove_probe_trials:
            # For each date with x, decide whether to keep or throw out x
            if animal == "Diego":
                days_with_x_keep = [240822] # not probes, but epoch name happens to be this.
                days_with_x_remove = [250319] # these are probes
            elif animal == "Pancho":
                days_with_x_keep = []
                days_with_x_remove = [250322] # these are probes
            else:
                assert False
            
            # Confirm that you inputed all dates
            dates_with_x = dfmerged[["|x" in epoch for epoch in dfmerged["epoch"]]]["date"].unique()
            for d in dates_with_x:
                assert (d in days_with_x_keep) or (d in days_with_x_remove), "no biggi -just add this date to the list. figure out if x epochs are probes. Do this by running : grouping_print_n_samples(dfmerged, [date, epoch, chunk_rank, shape]). if few trials then the yare probes"
            
            # Remove the epochs
            epochs_crossed = [ep for ep in dfmerged["epoch"].unique() if "|x" in ep]
            a = (DFEFFECT["date"].isin(days_with_x_remove))
            b = (DFEFFECT["epoch_1"].isin(epochs_crossed))
            c = (DFEFFECT["epoch_2"].isin(epochs_crossed))
            bools_remove = a & (b | c)
            DFEFFECT = DFEFFECT[~bools_remove].reset_index(drop=True)

        # (3) Keep only shapes whose chunk_rank matches their chunk_rank_global
        from pythonlib.dataset.dataset_analy.grammar import _chunk_rank_global_extract
        dfmerged["task_kind"] = "prims_on_grid" # Just a hack, inner code needs this...
        dfchunkrankmap = _chunk_rank_global_extract(dfmerged, check_low_freq_second_shape=False)

        # First, Assign a new column to dfeffect with chunk_rank_global
        map_DaEpSh_to_crglob = {}
        for _, row in dfchunkrankmap.iterrows():
            k = (row["date"], row["epoch"], row["shape"])
            v = row["chunk_rank_global"]
            map_DaEpSh_to_crglob[k] = v
        for i in [1, 2]:
            DFEFFECT[f"chunk_rank_global_{i}"] = [map_DaEpSh_to_crglob[(row["date"], row[f"epoch_{i}"], row[f"shape_{i}"])] for _, row in DFEFFECT.iterrows()]

        # Second, only keep rows in which crg is equal to cr.
        DFEFFECT = DFEFFECT[(DFEFFECT["chunk_rank_global_1"] == DFEFFECT["chunk_rank_1"])].reset_index(drop=True)
        DFEFFECT = DFEFFECT[(DFEFFECT["chunk_rank_global_2"] == DFEFFECT["chunk_rank_2"])].reset_index(drop=True)

        # (4) Finally, sanity checks. Check that each shape exists in only one chunk_rank
        assert sum((DFEFFECT["epoch_same"] == True) & (DFEFFECT["shape_same"] == True) & (DFEFFECT["chunk_rank_same"] == False))==0
        assert sum((DFEFFECT["epoch_same"] == True) & (DFEFFECT["shape_same"] == False) & (DFEFFECT["chunk_rank_same"] == True))==0
        print("... after: ", len(DFEFFECT))

        # (5) Print the final results
        list_vars = ["epoch", "chunk_rank", "shape"]
        vars_others = ["date", "bregion"]
        dfmerged = dfdist_convert_merge_pair_to_get_all_levels(DFEFFECT, list_vars, vars_others)
        savepath = f"{savedir}/two_shapes-counts-after_prune_and_align_cr_global.txt"
        grouping_print_n_samples(dfmerged, ["date", "epoch", "chunk_rank", "shape"], savepath=savepath)

    ### Sanity check that all have same dimensionality within a date
    for date in DFEFFECT["date"].unique():
        assert len(set(DFEFFECT[(DFEFFECT["date"] == date)]["data_dim"]))==1
        assert len(set(DFEFFECT[(DFEFFECT["date"] == date)]["npcs_euclidean"]))==1

    ### Restrict to just those <var_must_exist_across_context> which exist across <var_context>
    if analysis == "pig_vs_sp":
        var_must_exist_across_context = "shapeloc"
        var_context = "task_kind"
        for _i in [1, 2]:
            DFEFFECT = append_col_with_grp_index(DFEFFECT, [f"shape_{_i}", f"gridloc_{_i}"], f"shapeloc_{_i}")
        DFEFFECT = append_col_with_grp_index(DFEFFECT, ["shapeloc_1", "shapeloc_2"], "shapeloc_12")
    elif analysis == "two_shapes":
        var_must_exist_across_context = "cr_and_w"
        var_context = "epoch"
        for _i in [1, 2]:
            DFEFFECT = append_col_with_grp_index(DFEFFECT, [f"chunk_rank_{_i}", f"chunk_within_rank_{_i}"], f"cr_and_w_{_i}")
        DFEFFECT = append_col_with_grp_index(DFEFFECT, ["cr_and_w_1", "cr_and_w_2"], "cr_and_w_12")
    elif analysis in ["rank_within", "rank_up_vs_down", "n_in_chunk"]:
        # Dont do this, not required
        var_must_exist_across_context = None
        var_context = None
    else:
        assert False
    
    if var_must_exist_across_context is not None:
        print(f"Before prune just those {var_must_exist_across_context} which exist across {var_context}:: ", len(DFEFFECT))
        vars_conj = ["date"]
        grpdict = grouping_append_and_return_inner_items_good(DFEFFECT, vars_conj)
        list_df =[]
        for _grp, inds in grpdict.items():
            df = DFEFFECT.iloc[inds]
            savedir_this = f"{savedir}/prune-{_grp}"
            os.makedirs(savedir_this, exist_ok=True)
            _df = dfdist_postprocess_condition_prune_to_var_pairs_exist(df, var_must_exist_across_context, var_context, 
                                                                        plot_counts_savedir=savedir_this)
            plt.close("all")
            if len(df)!=len(_df):
                print("pruned from: ", len(df), " to ", len(_df))

            list_df.append(_df)
        DFEFFECT = pd.concat(list_df).reset_index(drop=True)
        savepath = f"{savedir}/prune_var_must_exist-AFTER.txt"
        print("... after: ", len(DFEFFECT))

    ### Prune to min n
    print("Before prune based on min n: ", len(DFEFFECT))
    DFEFFECT = DFEFFECT[(DFEFFECT["n1"]>=n_min_trials_per_label ) & (DFEFFECT["n2"]>=n_min_trials_per_label)].reset_index(drop=True)
    print("... after: ", len(DFEFFECT))

    # (5) Print the final results
    list_vars = ["epoch", "chunk_rank", "shape"]
    vars_others = ["date", "bregion"]
    dfmerged = dfdist_convert_merge_pair_to_get_all_levels(DFEFFECT, list_vars, vars_others)
    savepath = f"{savedir}/two_shapes-counts-FINAL.txt"
    grouping_print_n_samples(dfmerged, ["date", "epoch", "chunk_rank", "shape"], savepath=savepath)
    # savepath = f"{savedir}/two_shapes-counts-FINAL-2.txt"
    # grouping_print_n_samples(dfmerged, ["date", "epoch", "chunk_rank", "chunk_within_rank_fromlast", "shape"], savepath=savepath)

    return DFEFFECT, eff1, eff2

def final_dfeffect_mean_simple_PIGvsSP(dfeffect, eff1, eff2, doplot=False):
    """
    Get stats related to effects (eff1 and eff2), including the means for each, and
    the difference between those means for each brain region. ie dfeffect["effect"] has values eff1 and eff2

    Simple -- return so that there is one row per ("effect", "date", "bregion")
    - Ie get mean over all data to get one value per ("effect", "date", "bregion")
    """
    ###### GOOD
    from pythonlib.tools.pandastools import aggregGeneral, plot_45scatter_means_flexible_grouping
    from neuralmonkey.analyses.euclidian_distance import dfdist_compute_effects_diff_wideform
    yvar = "dist_yue_diff"

    assert len(dfeffect["animal"].unique())==1
    # assert len(dfeffect["question"].unique())==1 # is fine to be >1, sometimes I combine questions...
    assert len(dfeffect["subspace"].unique())==1

    # (1) Agg so that datapt = date
    dfeffect = aggregGeneral(dfeffect, ["effect", "date", "bregion"], [yvar]) # Datapt = (labels1, labels2)

    # (2) For each bregion, get mean value for each effect (across dates)
    dfeffect_agg = aggregGeneral(dfeffect, ["bregion", "effect"], [yvar])

    # (3) For each bregion, get mean difference between effects
    var_effect = "effect"
    vars_grp = ["bregion"]
    dfsummary, dfpivot, effect_div_name, eff1_name, eff2_name = dfdist_compute_effects_diff_wideform(dfeffect, var_effect, 
                                                                            eff1, eff2, vars_grp, 
                                                                            diff_func="div")

    # (4) Merge means for each effect (stored in dfeffect_agg) with differences between effects (stored in dfsummary)
    dfsummary["effect"] = effect_div_name # Simply rename, for easy comparison
    dfsummary[yvar] = dfsummary[effect_div_name] # Simply rename
    dfmerge_long = pd.concat([dfsummary.loc[:, dfeffect_agg.columns], dfeffect_agg], axis=0)

    # (5) Also get version with one row per bregion (where columns are effects and difference of effects)
    dfmerge_wide = pd.merge(dfpivot, dfsummary, on="bregion")

    if doplot:
        _, fig = plot_45scatter_means_flexible_grouping(dfmerge_long, "effect", eff1, effect_div_name, 
                                                        None, yvar, "bregion", shareaxes=True)
        _, fig = plot_45scatter_means_flexible_grouping(dfmerge_long, "effect", eff1, eff2, 
                                                        None, yvar, "bregion", shareaxes=True)
        
    return dfmerge_long, dfmerge_wide, effect_div_name, eff1_name, eff2_name

def final_dfeffect_mean_simple_PIGvsSP_bootstrap(DFEFFECT, eff1, eff2, vars_conj=None, nboot = 50):
    """
    On each run, gets score, which is one value for each (bregion, effect). 

    On each bootstrap iteration, resamples within each level of vars_conj.

    Returns concated df holding concated bootstrapped data.
    """
    from pythonlib.tools.statstools import bootstramp_resample

    if vars_conj is None:
        vars_conj = ["bregion", "date", "effect"] # Will sample within each level of this.
    
    list_df = []
    for _i in range(nboot):
        if _i%20==0:
            print(_i)
        dfeffect_boot = bootstramp_resample(DFEFFECT, vars_conj)
        dfmerge_long, _, effect_div_name, eff1_name, eff2_name = final_dfeffect_mean_simple_PIGvsSP(dfeffect_boot, eff1, eff2, 
                                                                                                    doplot=False)
        dfmerge_long["i_boot"] = _i
        # print(dfmerge_long)
        # print(effect_div_name)
        # assert False
        list_df.append(dfmerge_long) # columns (bregion, effect[each and their diffs all get a row], dist_yue_diff, i_boot)

    DFSCORE_BOOT = pd.concat(list_df).reset_index(drop=True)
    DFSCORE_BOOT = append_col_with_grp_index(DFSCORE_BOOT, ["bregion", "i_boot"], "br_i")
    DFSCORE_BOOT["index"] = DFSCORE_BOOT.index

    return DFSCORE_BOOT, effect_div_name, eff1_name, eff2_name

def get_params_this_save_suffix(animal, save_suffix):
    """
    Helper to get contrast idx pairs that you wish to plot, for this save_suffix.
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import params_get_contrasts_of_interest

    # if save_suffix in "AnBmCk_general":
    #     # This is everything, except two_shape_sets
    # elif save_suffix in "two_shape_sets":
    #     # This is just two_shape_sets
    # elif save_suffix in ["sh_vs_seqsup", "sh_vs_dir", "sh_vs_col"]:
    #     # this is other question..
    # else:
    #     assert False

    DICT_VVO_TO_LISTIDX = params_get_contrasts_of_interest(return_list_flat=False)

    # Dates
    dates, question, _, _ = load_preprocess_get_dates(animal, save_suffix)

    # Var-var_other indices
    map_savesuffix_to_contrast_idx_pairs = {}
    if save_suffix == "AnBmCk_general":
        # This is everything, except two_shape_sets
        map_savesuffix_to_contrast_idx_pairs = {k:v for k, v in DICT_VVO_TO_LISTIDX.items() if not k=="two_shape_sets"}
    elif save_suffix == "two_shape_sets":
        map_savesuffix_to_contrast_idx_pairs = {k:v for k, v in DICT_VVO_TO_LISTIDX.items() if k=="two_shape_sets"}
    elif save_suffix in ["sh_vs_seqsup", "sh_vs_dir", "sh_vs_col"]:
        map_savesuffix_to_contrast_idx_pairs = "IGNORE"
    else:
        assert False, "these use different system, not referencing the indices in LIST_VAR. See older code."
    
    return question, dates, map_savesuffix_to_contrast_idx_pairs


def mult_plot_all_wrapper(just_return_data=False):
    """
    Wrapper to make all plots that requiring reloading things, etc.
    """

    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    import os
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import params_getter_euclidian_vars_grammar

    HACK = True # to limit to jsut idx 0 and 1 (still extracting/computing those)

    # Params
    subspace_projection_fitting_twind = (-0.8, 0.3)
    # dates_skip_failed = [220908, 220909, 230817, 230829, 230913, 230922, 230920, 230924, 230925]
    # dates_skip_failed = [230817, 230913]
    dates_skip_failed = [
        220831, # Pancho, loading session fails...
        # 220901, # Pancho, dloading session fails...
        # 230728, 
        # 230817, 
        # 240830,
        250325, # Pancho
        250319, # Diego
        ]

    ### Param sets
    # (1) Old version (large LIST_VAR)
    LIST_SAVE_SUFFIX = ["two_shape_sets"]
    # for save_suffix in ["sh_vs_seqsup"]:
    # for save_suffix in ["two_shape_sets", "AnBmCk_general"]:
    # for save_suffix in ["AnBmCk_general"]:
    # for save_suffix in ["sh_vs_seqsup", "two_shape_sets", "AnBmCk_general"]:
    LIST_SUBSPACE = ["epch_sytxrol", "syntax_role", "sytx_all"]
    version_seqsup_good=False
    get_all_twind_scal = True

    # # (2) New version (small LIST_VAR, for seqsup)
    # LIST_SAVE_SUFFIX = ["sh_vs_seqsup"]
    # LIST_SUBSPACE = ["stxsuperv"]
    # version_seqsup_good=True
    # get_all_twind_scal = False

    for save_suffix in LIST_SAVE_SUFFIX:
        for subspace_projection in LIST_SUBSPACE:
            for animal in ["Diego", "Pancho"]:
            # for animal in ["Pancho", "Diego"]:

                ### LOAD DFDIST
                # Params for loading dataset
                question, dates, map_savesuffix_to_contrast_idx_pairs = get_params_this_save_suffix(animal, save_suffix)

                LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT, \
                        use_strings_for_vars_others, list_subspace_projection, is_seqsup_version = \
                            params_getter_euclidian_vars_grammar(question, version_seqsup_good, HACK=HACK)
                
                # from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
                # LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question, 

                # Fixed params
                which_level = "stroke"
                event = "00_stroke"
                combine = False

                from neuralmonkey.classes.session import _REGIONS_IN_ORDER, _REGIONS_IN_ORDER_COMBINED
                list_bregion = _REGIONS_IN_ORDER
                if animal == "Diego":
                    list_bregion = [br for br in list_bregion if not br=="dlPFC_p"]

                # for save_suffix in map_savesuffix_to_dates.keys():
                #     contrast_idx_pairs = map_savesuffix_to_contrast_idx_pairs[save_suffix]

                # Flatten to list of indices
                if map_savesuffix_to_contrast_idx_pairs == "IGNORE":
                    # Then first load all the indices
                    list_contrast_idx = list(range(len(LIST_VAR)))
                else:
                    # Then just load those you will use
                    list_contrast_idx = sorted(set([vvv for v in map_savesuffix_to_contrast_idx_pairs.values() for vv in v for vvv in vv]))


                from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _get_list_twind_by_animal
                _list_twind, _, _ = _get_list_twind_by_animal(animal, event, "traj_to_scalar")
                _twscal = _list_twind[0]
                if get_all_twind_scal:
                    list_twind_scal = [_twscal, (-0.3, -0.1)]
                else:
                    list_twind_scal = [_twscal]

                for twind_scal in list_twind_scal:

                    ### Collect all raw, across all contrast idx.
                    list_dfdist =[]
                    for date in dates:
                        
                        if date in dates_skip_failed:
                            continue
                        
                        for bregion in list_bregion:
                            for contrast_idx in list_contrast_idx:
                                var_effect = LIST_VAR[contrast_idx]

                                if version_seqsup_good:
                                    SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/syntax_good/EUCLIDIAN_SHUFF/{animal}-{date}-comb={combine}-q={question}-seqsupgood"
                                else:
                                    SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/syntax_good/EUCLIDIAN_SHUFF/{animal}-{date}-comb={combine}-q={question}"

                                SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-fit_twind={subspace_projection_fitting_twind}/contrast={contrast_idx}|{var_effect}"

                                path = f"{SAVEDIR}/dfdist-twind_scal={twind_scal}.pkl"
                                
                                print("Loading this path: ", path)
                                if not os.path.exists(path):
                                    # Then check that lost data
                                    assert os.path.exists(f"{SAVEDIR}/preprocess/lost_all_dat_in_preprocess.pdf"), "no explanation for why failed to find saved data"
                                else:
                                    dfdist = pd.read_pickle(path)
                                    print(SAVEDIR)

                                    dfdist["date"] = date
                                    dfdist["animal"] = animal

                                    dfdist["varsame_effect_context"] = dfdist[f"same-{var_effect}|_vars_others"]
                                    dfdist["contrast_effect"] = f"{contrast_idx}|{var_effect}"
                                    dfdist["metaparams"] = f"{subspace_projection}|{subspace_projection_fitting_twind}|{twind_scal}"

                                    list_dfdist.append(dfdist)
                    DFDIST = pd.concat(list_dfdist).reset_index(drop=True)

                    if just_return_data:
                        return DFDIST

                    ### RUN PLOTS
                    if save_suffix == "sh_vs_seqsup" and version_seqsup_good==True:
                        # Latest, good, focused on shape vs. seqsup.
                        # These are pruning LIST_VAR so that just those that matter.
                        # Also keeping var_other as tuple, allowing easy analysis -- including allowing controling for chunk|shape, 
                        # which is not possible for below.
                        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/EUCLIDIAN_SHUFF/MULT/{animal}-savesuff={save_suffix}-subspace={subspace_projection}-twscal={twind_scal}-comb={combine}-seqsupgood={version_seqsup_good}"
                        os.makedirs(SAVEDIR, exist_ok=True)
                        print(SAVEDIR)

                        from neuralmonkey.scripts.analy_syntax_good_eucl_state import mult_plot_grammar_vs_seqsup_new
                        for contrast_version in ["shape_index", "shape_within_chunk"]:
                            mult_plot_grammar_vs_seqsup_new(DFDIST, SAVEDIR, contrast_version)

                    elif save_suffix == "sh_vs_seqsup" and version_seqsup_good==False:
                        ### For sh vs. seqsup (reduced effect of stroke index, comparing same tasks, with and without sequence supervision)
                        from neuralmonkey.scripts.analy_syntax_good_eucl_state import mult_plot_grammar_vs_seqsup

                        ### Summary plots
                        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/EUCLIDIAN_SHUFF/MULT/{animal}-savesuff={save_suffix}-subspace={subspace_projection}-twscal={twind_scal}-comb={combine}"
                        os.makedirs(SAVEDIR, exist_ok=True)
                        print(SAVEDIR)

                        for shape_or_loc_rule in ["shape", "loc"]:
                            mult_plot_grammar_vs_seqsup(DFDIST, SAVEDIR, animal, shape_or_loc_rule)                
                    else:
                        from neuralmonkey.scripts.analy_syntax_good_eucl_state import postprocess_dfdist_collected
                        DFDIST, DFDIST_AGG = postprocess_dfdist_collected(DFDIST)

                        if False:
                            # Given question, return all the contrasts that use this question
                            DICT_VVO_TO_LISTIDX
                            list_dates_get, question, twind_analy, fr_normalization_method = load_preprocess_get_dates(animal, dir_suffix)

                        ### Summary plots
                        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/EUCLIDIAN_SHUFF/MULT/{animal}-savesuff={save_suffix}-subspace={subspace_projection}-twscal={twind_scal}-comb={combine}"
                        os.makedirs(SAVEDIR, exist_ok=True)
                        print(SAVEDIR)

                        from  neuralmonkey.scripts.analy_syntax_good_eucl_state import mult_plot_all
                        assert len(DFDIST_AGG)>0
                        mult_plot_all(DFDIST_AGG, map_savesuffix_to_contrast_idx_pairs, SAVEDIR, question, skip_contrast_idx_pair_if_fail=True)

def _targeted_pca_clean_plots_and_dfdist_MULT_plot_single(DFDIST_THIS, colname_conj_same, question, SAVEDIR, order=None,
                                                          yvar="dist_yue_diff"):
    """
    Helper to plot contrasts in catplot, for this question. 
    PARAMS:
    - order, list of contrasts (each a string, such as "0|1|0") to restrict the plot to (and in this order).
    """

    if len(DFDIST_THIS)>0:
        fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y=yvar, hue_order=order,
                    col="subspace", kind="bar", errorbar="se")
        savefig(fig, f"{SAVEDIR}/q={question}-catplot-1.pdf")

        if False: # not usulaly checked
            fig = sns.catplot(data=DFDIST_THIS, x=colname_conj_same, hue="subspace", y=yvar, order=order,
                        col="bregion", kind="bar", errorbar="se")
            from pythonlib.tools.snstools import rotateLabel
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/q={question}-catplot-2.pdf")

        fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y=yvar, hue_order=order,
                    col="subspace", kind="boxen")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.2)
        savefig(fig, f"{SAVEDIR}/q={question}-catplot-3.pdf")

        plt.close("all")

def targeted_pca_MULT_1_load_and_save(animal, date, run, expt_kind, OVERWRITE=False):
    """
    First, run this to collect all results across bregions, subspaces, questions, etc (for a given animal-date) and
    then save a single DFDIST. Do this becuase it takes time to load and postprocess.

    You can then load the DFDIST and do analyses.
    expt_kind="RULE_ANBMCK_STROKE"

    LT Checked
    """
    ### [MULT] Loading all dfdists and making summary plots
    # run = 12
    # SAVEDIR = f"/tmp/SYNTAX_TARGETED_PCA_run{run}"
    from glob import glob
    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific
    from pythonlib.tools.pandastools import replace_None_with_string
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import append_col_with_grp_index

    OLD_VERSION = False
    expected_n_subspaces = 6
    expected_n_questions = 4
    expected_n_bregions = 8

    if run==1:
        euclidean_label_vars = ["chunk_within_rank", "chunk_rank", "shape"]
        OLD_VERSION = True
    elif run==3:
        # Good
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 3
    elif run==4:
        # Updated projetion, which (i) suntracts variables and (ii) adds conitnuos motor
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 2
    elif run==5:
        # Now only subtracts "first stroke". Also added ordinal logistic regression
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 2
        expected_n_subspaces=4
    elif run==6:
        # Now only subtracts "first stroke". Also added ordinal logistic regression
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 1
        expected_n_subspaces=2
        expected_n_questions = 6
    elif run==7:
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 1
        expected_n_subspaces=1
        expected_n_questions = 7
    elif run==8:
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 1
        expected_n_subspaces=1
        expected_n_questions = 7
    elif run==9:
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 1
        expected_n_subspaces=2
        expected_n_questions = 7
    elif run==10:
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 1
        expected_n_subspaces=1
        expected_n_questions = 5
    elif run==11:
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 4
        expected_n_subspaces= 1
        expected_n_questions = 10
    elif run==12:
        euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        expected_n_iters = 1
        expected_n_subspaces= 2
        expected_n_questions = 10
    elif run > 12:
        pass
        # euclidean_label_vars = ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc"]
        # expected_n_iters = 1
        # expected_n_subspaces= 2
        # expected_n_questions = 10
    else:
        print(run)
        assert False

    SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"
    import os
    os.makedirs(SAVEDIR_MULT, exist_ok=True)

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/{animal}-{date}-q={expt_kind}"
        
    # check if done
    if not OVERWRITE:
        if os.path.exists(f"{SAVEDIR_MULT}/DFDIST-{animal}-{date}.pkl"):
            return None

    try:
        LIST_DFDIST =[]
        for bregion in _REGIONS_IN_ORDER_COMBINED:
            path_search = f"{SAVEDIR}/bregion={bregion}/FITTING_*"
            list_dir = glob(path_search)
            if len(list_dir)==0:
                print("Found no directories matching: ", path_search)

            # Concatenate across iterations
            map_subspaceiter_to_dfdist = {}
            for _, savedir in enumerate(list_dir):                
                # print(f"{bregion} --- [{_i}]/[{len(list_dir)}],  --- {savedir}")
                path = f"{savedir}/dfdist.pkl"
                dfdist = pd.read_pickle(path)
                dfdist["var_subspace"] = [tuple(x)  if isinstance(x, list) else x for x in dfdist["var_subspace"]]  
                if "var_conj" not in dfdist:
                    dfdist["var_conj"] = "none"
                    dfdist["var_conj_lev"] = "none"
                dfdist = append_col_with_grp_index(dfdist, ["var_subspace", "var_conj", "var_conj_lev"], "subspace")
                dfdist["n1"] = [x[0] for x in dfdist["n_1_2"]]
                dfdist["n2"] = [x[1] for x in dfdist["n_1_2"]]
                dfdist["bregion"] = bregion

                tmp = dfdist["i_proj"].unique().tolist()
                assert len(tmp)==1
                i_proj = tmp[0]

                tmp = dfdist["subspace"].unique().tolist()
                assert len(tmp)==1
                subspace = tmp[0]

                map_subspaceiter_to_dfdist[subspace, i_proj] = dfdist
            
            # Concatenate across iterations (ie subsamples)
            list_subspace = set([x[0] for x in map_subspaceiter_to_dfdist.keys()])
            map_subspace_to_dfdist = {}
            for subspace in list_subspace:
                _keys = [x for x in map_subspaceiter_to_dfdist.keys() if x[0]==subspace]
                list_dfdist = [map_subspaceiter_to_dfdist[k] for k in _keys]
                dfdist_concat = pd.concat(list_dfdist, axis=0)
                assert "i_proj" in dfdist_concat, "sanity check..."
                dfdist_concat = aggregGeneral(dfdist_concat, ["bregion", "labels_1", "labels_2", "var_subspace", "var_conj", "var_conj_lev", "question", "subspace"], 
                                            ["dist_mean", "DIST_98", "dist_norm", "dist_yue_diff", "n1", "n2", "data_dim", "npcs_euclidean"], nonnumercols=None)
                assert all(dfdist_concat["subspace"]==subspace)
                map_subspace_to_dfdist[subspace] = dfdist_concat
            del map_subspaceiter_to_dfdist

            # Finally, append the correct columns, for each question.
            for subspace, dfdist in map_subspace_to_dfdist.items():            
            # for _i, savedir in enumerate(list_dir):
                
            #     print(f"{bregion} --- [{_i}]/[{len(list_dir)}],  --- {savedir}")
            #     path = f"{savedir}/dfdist.pkl"
            #     dfdist = pd.read_pickle(path)
            
                # Postprocessing        
                # dfdist["var_subspace"] = [tuple(x)  if isinstance(x, list) else x for x in dfdist["var_subspace"]]            
                # dfdist["bregion"] = bregion
                # if "var_conj" not in dfdist:
                #     dfdist["var_conj"] = "none"
                #     dfdist["var_conj_lev"] = "none"
                # dfdist["n1"] = [x[0] for x in dfdist["n_1_2"]]
                # dfdist["n2"] = [x[1] for x in dfdist["n_1_2"]]

                if False:
                    from pythonlib.tools.pandastools import replace_None_with_string
                    dfdist = replace_None_with_string(dfdist)

                if OLD_VERSION:
                    # Before used map_question_to_euclideanvars, I had just a single list of euclidean_label_vars
                    dfdist, colname_conj_same = dfdist_extract_label_vars_specific(dfdist, euclidean_label_vars, return_var_same=True)
                    # get metaparams
                    dfdist["question"] = "ignore"
                    LIST_DFDIST.append(dfdist)
                else:
                    # Now using map_question_to_euclideanvars
                    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params
                    map_question_to_euclideanvars = targeted_pca_clean_plots_and_dfdist_params()["map_question_to_euclideanvars"]

                    # Split dfdist for preprocessing
                    map_question_to_varsame = {}
                    for question, euclidean_label_vars in map_question_to_euclideanvars.items():
                        dfdist_this = dfdist[dfdist["question"] == question].reset_index(drop=True)
                        if len(dfdist_this)>0:

                            if False: # not needed (?) neucase it sdone below
                                dfdist_this, colname_conj_same = dfdist_extract_label_vars_specific(dfdist_this, euclidean_label_vars, return_var_same=True)
                                map_question_to_varsame[question] = colname_conj_same

                            ### Preprocessing for each dfdistthis. is faster here before combining across questions
                            # for each question's dfdist, agg it and then re-extract useful columns
                            if False: 
                                # This is not true anymore, beucase I agg across iters above, each may heave slightly different datasets
                                assert len(dfdist_this["DIST_98"].unique())==1
                            if False:
                                # Already done above...
                                dfdist_this = aggregGeneral(dfdist_this, ["labels_1", "labels_2", "var_subspace", "var_conj", "var_conj_lev", "bregion", "question"], 
                                                            ["dist_mean", "DIST_98", "dist_norm", "dist_yue_diff"], nonnumercols=["n1", "n2"])                                
                            # Need to repopulate, after agging, but need to do this within each question, beucase thye have their own label vars
                            dfdist_this, colname_conj_same = dfdist_extract_label_vars_specific(dfdist_this, euclidean_label_vars, return_var_same=True) # Repopulate the var columns
                            map_question_to_varsame[question] = colname_conj_same

                            LIST_DFDIST.append(dfdist_this)
            del map_subspace_to_dfdist

        if len(LIST_DFDIST)==0:
            # Then always skip, this is just a missing day
            print("Skipping, as list_dfdist is empty")
            return None

        if False: # No need -- it will be obvious if something is missing
            assert len(LIST_DFDIST)==(expected_n_bregions * expected_n_questions * expected_n_subspaces * expected_n_iters), "this is to make sure that you arent saving partial results"

        DFDIST = pd.concat(LIST_DFDIST).reset_index(drop=True)
        del LIST_DFDIST
        DFDIST = replace_None_with_string(DFDIST)
        # DFDIST = stringify_values(DFDIST)
        # DFDIST = append_col_with_grp_index(DFDIST, ["var_subspace", "var_conj", "var_conj_lev"], "subspace")

        # Save it
        pd.to_pickle(DFDIST, f"{SAVEDIR_MULT}/DFDIST-{animal}-{date}.pkl")
        pd.to_pickle(map_question_to_varsame, f"{SAVEDIR_MULT}/map_question_to_varsame-{animal}-{date}.pkl")

    except Exception as err:
        print("ERROR, SKIPPING: ", err)
        raise err
        # return None

def effect_extract_helper_this(DFDIST, question, subspaces, 
                               contrasts_diff, contrasts_either, 
                               only_within_pig, return_extras=False):
    """
    Get sliced DFDIST, which holds pairwise comparisons for this "effect".

    PARAMS:
    - question, str
    - subspaces, either list of str or "all"
    - contrasts_diff, contrasts_either, each a list of str.

    (See within, in dfdist_variables_effect_extract_helper(), for details).

    RETURNS:
    - pruned dfdist (copy) or None if all rows pruned

    LT CHECKED
    """
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_effect_extract_helper
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params
    
    params = targeted_pca_clean_plots_and_dfdist_params()
    map_question_to_euclideanvars = params["map_question_to_euclideanvars"]
    map_question_to_varsame = params["map_question_to_varsame"]

    colname_conj_same = map_question_to_varsame[question]
    vars_in_order = map_question_to_euclideanvars[question]

    if subspaces == "all":
        subspaces = DFDIST["subspace"].unique().tolist()
    
    if question not in DFDIST["question"].unique().tolist():
        # This is ok, just skip this question
        return None
    
    subspaces_exist = [ss in DFDIST["subspace"].unique().tolist() for ss in subspaces]
    if not any(subspaces_exist):
        print("subspaces exist: ", DFDIST["subspace"].unique().tolist())
        print("subspaces desired: ", subspaces)
        return None

    DFDIST_THIS = DFDIST[
        (DFDIST["question"] == question)
        ]
    assert len(DFDIST_THIS)>0

    if only_within_pig:
        DFDIST_THIS = DFDIST_THIS[(DFDIST_THIS["task_kind_12"] == "prims_on_grid|prims_on_grid")]
    if len(DFDIST_THIS)==0:
        return None

    DFDIST_THIS = DFDIST_THIS[(DFDIST_THIS["subspace"].isin(subspaces))]
    if len(DFDIST_THIS)==0:
        return None

    if contrasts_diff is not None:
        dfdist = dfdist_variables_effect_extract_helper(DFDIST_THIS, colname_conj_same, vars_in_order, contrasts_diff, contrasts_either, PRINT=False)
    else:
        # Skip it, you just came for (DFDIST_THIS, colname_conj_same, vars_in_order)
        dfdist = None

    if return_extras:
        return dfdist, DFDIST_THIS, colname_conj_same, vars_in_order
    else:
        # Just return the effects of interest
        return dfdist

def effect_extract_helper_this_wrapper(DFDIST, question, subspaces, contrasts_diff, contrasts_either, 
                                       only_within_pig, effect_name, list_dfeffect):
    """
    Get sliced DFDIST, which holds pairwise comparisons for this "effect".

    PARAMS:
    - question, str
    - subspaces, either list of str or "all"
    - contrasts_diff, contrasts_either, each a list of str. Any vars not included here will be 
    assumed to be "same".

    RETURNS:
    - pruned dfdist (copy) or None if all rows pruned

    LT CHECKED
    """
    if False:
        assert len(subspaces)==1, "Currently I use dist_yue_diff, which requires keeping within the same subspace to be interpretable"

    if question in DFDIST["question"].unique().tolist():
        try:
            dfeffect = effect_extract_helper_this(DFDIST, question, subspaces, contrasts_diff, contrasts_either, only_within_pig)
            for df in list_dfeffect:
                assert effect_name not in df["effect"].unique().tolist(), "you are overwriting someting..."
        except Exception as err:
            print("Failed to find data for: ", question, subspaces)
            print("Existing questions:", DFDIST["question"].unique())
            print("Existing subspace:", DFDIST["subspace"].unique())
            print("Existing task_kind_12:", DFDIST["task_kind_12"].unique())
            raise err

        if dfeffect is not None:
            dfeffect["effect"] = effect_name
            list_dfeffect.append(dfeffect)
        else:
            print("No data for this question (not sure why): ", question)

    else:
        print("Skipped this question (doesnt exist in dfdist): ", question)

def get_contrasts_single_effect(question):
    """
    Reutrn list of str, each a contrast, suhc as "1|0|1|1".
    This resturns all contrasts that have only one "0".
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params

    params = targeted_pca_clean_plots_and_dfdist_params()
    map_question_to_euclideanvars = params["map_question_to_euclideanvars"]

    vars_this_question = map_question_to_euclideanvars[question]
    this_tuple = [1 for _ in range(len(vars_this_question))]

    order = []
    for i in range(len(vars_this_question)):
        this_tuple_this = this_tuple.copy()
        this_tuple_this[i] = 0
        order.append(this_tuple_this)

    # # Add: Effect of shape ()
    # _vars_diff = ["epoch", "shape"]
    # order.append([0 if _var in _vars_diff else 1 for _var in vars_this_question])

    # # Add: Effect of syntax
    # _vars_diff = ["chunk_within_rank", "chunk_rank", "shape"]
    # order.append([0 if _var in _vars_diff else 1 for _var in vars_this_question])

    order = ["|".join([str(x) for x in this_tuple_this]) for this_tuple_this in order]

    return order

def plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order=None):
    """
    Helper to make plots (catplots, showing multple contrats' effects), for a single
    quusetion

    PARAMS:
    - question = "7_ninchunk_vs_rankwithin"
    - order = [
        '0|1|1|1|1|1|1',
        '1|0|1|1|1|1|1',
        '1|1|0|1|1|1|1',
        '1|1|1|0|1|1|1',
        '1|1|1|1|0|1|1',
        '1|1|1|1|1|0|1',
        '1|1|1|1|0|0|1',
        ]
    """
    # from neuralmonkey.scripts.analy_syntax_good_eucl_state import _targeted_pca_clean_plots_and_dfdist_MULT_plot_single
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params

    params = targeted_pca_clean_plots_and_dfdist_params()
    # map_question_to_euclideanvars = params["map_question_to_euclideanvars"]
    map_question_to_varsame = params["map_question_to_varsame"]

    DFDIST_THIS = None
    colname_conj_same = None
    if question in map_question_to_varsame:

        colname_conj_same = map_question_to_varsame[question]
        # vars_this_question = map_question_to_euclideanvars[question]
        
        DFDIST_THIS = DFDIST[
            (DFDIST["question"] == question)
            ].reset_index(drop=True)
        
        if only_within_pig:
            if "task_kind_12" not in DFDIST_THIS:
                # Then assume these are all PIG
                pass
            else:
                DFDIST_THIS = DFDIST_THIS[(DFDIST_THIS["task_kind_12"] == "prims_on_grid|prims_on_grid")]

        if len(DFDIST_THIS)>0:        

            if order is None:
                order = get_contrasts_single_effect(question)

            _targeted_pca_clean_plots_and_dfdist_MULT_plot_single(DFDIST_THIS, colname_conj_same, question, 
                                                                  SAVEDIR, order, yvar=yvar)
    return DFDIST_THIS, colname_conj_same

def get_list_effects():
    """
    Stores list of effect pairs, which you want to plot in 45deg scatter, as useufl comparisons.
    RETURNS:
    - list of 2-tuples (of 2 strings, where the string is an effect0
    """
    LIST_EFFECT_PAIRS = []
    for idx in [2, 6, 10, 11]:
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_rankwithin"))
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_ninchunk"))
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_chunkrank"))
        LIST_EFFECT_PAIRS.append((f"{idx}_shape", f"{idx}_rankwithin"))
    for idx in [11]:
        for ss in ["rank_conj|none|none", "shape|none|none", "global"]:
            LIST_EFFECT_PAIRS.append((f"{idx}_motor_ss={ss}", f"{idx}_rankwithin_ss={ss}"))
            LIST_EFFECT_PAIRS.append((f"{idx}_motor_ss={ss}", f"{idx}_ninchunk_ss={ss}"))
            LIST_EFFECT_PAIRS.append((f"{idx}_motor_ss={ss}", f"{idx}_chunkrank_ss={ss}"))
            LIST_EFFECT_PAIRS.append((f"{idx}_shape_ss={ss}", f"{idx}_rankwithin_ss={ss}"))
    for idx in [8, 13, 9]:
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_rankwithin"))
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_chunkrank"))
        LIST_EFFECT_PAIRS.append((f"{idx}_shape", f"{idx}_rankwithin"))
    for idx in [3, 14]:
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_rankwithin_up"))
        LIST_EFFECT_PAIRS.append((f"{idx}_motor", f"{idx}_rankwithin_dn"))
        LIST_EFFECT_PAIRS.append((f"{idx}_rankwithin_dn", f"{idx}_rankwithin_up"))
    for idx in [8, 9, 10, 11, 13]:
        LIST_EFFECT_PAIRS.append((f"{idx}_2SH_epochshape", f"{idx}_2SH_rankwithin"))
        LIST_EFFECT_PAIRS.append((f"{idx}_2SH_epochshape", f"{idx}_2SH_syntax"))
    for idx in [4, "4c"]:
        for ss in ["shape|none|none", "global"]:
            LIST_EFFECT_PAIRS.append((f"{idx}_shapeSP_ss={ss}", f"{idx}_shapesyntax_ss={ss}"))
            LIST_EFFECT_PAIRS.append((f"{idx}_shapePIG_ss={ss}", f"{idx}_shapesyntax_ss={ss}"))
            LIST_EFFECT_PAIRS.append((f"{idx}_shapePIG_ss={ss}", f"{idx}_shapeSP_ss={ss}"))
    for idx in [24, 25]:
        for ss in ["shape|none|none", "global"]:
            for keep_only_middle_strokes in [False, True]:
                LIST_EFFECT_PAIRS.append((f"{idx}_shape_stkidx-inner={keep_only_middle_strokes}-ss={ss}", f"{idx}_seqsup_stkidx-inner={keep_only_middle_strokes}-ss={ss}"))
                LIST_EFFECT_PAIRS.append((f"{idx}_dir_stkidx-inner={keep_only_middle_strokes}-ss={ss}", f"{idx}_seqsup_stkidx-inner={keep_only_middle_strokes}-ss={ss}"))
                LIST_EFFECT_PAIRS.append((f"{idx}_shape_stkidx-inner={keep_only_middle_strokes}-ss={ss}", f"{idx}_shape_vs_superv-inner={keep_only_middle_strokes}-ss={ss}"))
                LIST_EFFECT_PAIRS.append((f"{idx}_dir_stkidx-inner={keep_only_middle_strokes}-ss={ss}", f"{idx}_dir_vs_superv-inner={keep_only_middle_strokes}-ss={ss}"))
        
    return LIST_EFFECT_PAIRS

# def effect_extract_helper_this_wrapper(DFDIST, question, subspaces, contrasts_diff, contrasts_either, 
#                                        only_within_pig, effect_name, list_dfeffect):
#     """

#     """
#     if False:
#         assert len(subspaces)==1, "Currently I use dist_yue_diff, which requires keeping within the same subspace to be interpretable"
#     if question in DFDIST["question"].unique().tolist():
#         try:
#             dfeffect = effect_extract_helper_this(DFDIST, question, subspaces, contrasts_diff, contrasts_either, only_within_pig)
#             for df in list_dfeffect:
#                 assert effect_name not in df["effect"].unique().tolist(), "you are overwriting someting..."
#         except Exception as err:
#             print("Failed to find data for: ", question, subspaces)
#             print("Existing questions:", DFDIST["question"].unique())
#             print("Existing subspace:", DFDIST["subspace"].unique())
#             print("Existing task_kind_12:", DFDIST["task_kind_12"].unique())
#             raise err
#         if dfeffect is not None:
#             dfeffect["effect"] = effect_name
#             list_dfeffect.append(dfeffect)
#         else:
#             print("No data for this question (not sure why): ", question)
#     else:
#         print("Skipped this question (doesnt exist in dfdist): ", question)

def targeted_pca_MULT_2_postprocess(DFDIST):
    """
    Another postprocessing step...

    Simple -- change the name of the subspace from (var1, var2, ..) to "global" if this ist he only
    subpsace in this datsaet.
    
    PARAMS:
    - dfdist, holds data for a single animal-date (across regions)

    LT CHECKED
    """

    # Sometimes a global subspace is called something like "('epoch', 'gridloc', 'DIFF_gridloc', 'chunk_rank', 'shape', 'rank_conj')|none|none"
    # Here, check if there is only one subspace with "(" and ")" in the name. If so, then assume this is "global", and rename
    # it as "global". Note that the old name is saved in DFDIST["subspace_orig"]
    map_subspace_to_shorthand = None
    subspaces = DFDIST["subspace"].unique().tolist()
    subspaces_potentially_global = [(s.find("(")>=0) and (s.find(")")>=0) for s in subspaces]
    if sum(subspaces_potentially_global)==1:
        # Then assume this is the global one
        ind = [i for i, x in enumerate(subspaces_potentially_global) if x==True][0]
        map_subspace_to_shorthand = {subspaces[ind]:"global"}
        DFDIST["subspace_orig"] = DFDIST["subspace"]
        DFDIST["subspace"] = [map_subspace_to_shorthand[s] if s in map_subspace_to_shorthand else s for s in DFDIST["subspace"]]
    
    return DFDIST, map_subspace_to_shorthand

def targeted_pca_MULT_2_plot_single_load(animal, date, run, yvar):
    """
    Load results for a single (animal, date) after the initial run -- i.e. the dfdist.

    LT CHECKED
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params

    # SAVEDIR = f"/tmp/SYNTAX_TARGETED_PCA_run{run}"
    SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"

    # yvar = "dist_yue_diff"
            
    SAVEDIR = f"{SAVEDIR_MULT}/summary_each_date-yvar={yvar}/{animal}-{date}"

    # Load it
    try:
        DFDIST = pd.read_pickle(f"{SAVEDIR_MULT}/DFDIST-{animal}-{date}.pkl")
        print(animal, date)
    except Exception as err:
        # OK to skip, as the final loading to make plots will fail if there is lacking some date.
        print(err)
        # raise err
        return None, None, None
        
    ### PREP
    # Get the variables
    map_question_to_euclideanvars = targeted_pca_clean_plots_and_dfdist_params()["map_question_to_euclideanvars"]

    # Split dfdist for preprocessing
    map_question_to_varsame = {}
    for question, euclidean_label_vars in map_question_to_euclideanvars.items():
        colname_conj_same = "same-"
        for v in euclidean_label_vars:
            colname_conj_same+=f"{v}|"
        colname_conj_same = colname_conj_same[:-1] # remove the last |
        map_question_to_varsame[question] = colname_conj_same

    os.makedirs(SAVEDIR, exist_ok=True)
    # SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/{animal}-{date}-q=RULE_ANBMCK_STROKE"
    
    # This gets (across group mean pairwise euclidean distnace) minus (wihtin-group of the same metric), without norming to max pairwise dist
    DFDIST["dist_yue_diff_unnorm"] = DFDIST["dist_yue_diff"] * DFDIST["DIST_98"]
    if "task_kind_12" not in DFDIST:
        # e.g., for shape vs. superv, I don't include this, as the other params do weed out SP.
        DFDIST["task_kind_12"] = "prims_on_grid|prims_on_grid"

    DFDIST, _ = targeted_pca_MULT_2_postprocess(DFDIST)
    # # Sometimes a global subspace is called something like "('epoch', 'gridloc', 'DIFF_gridloc', 'chunk_rank', 'shape', 'rank_conj')|none|none"
    # # Here, check if there is only one subspace with "(" and ")" in the name. If so, then assume this is "global", and rename
    # # it as "global". Note that the old name is saved in DFDIST["subspace_orig"]
    # subspaces = DFDIST["subspace"].unique().tolist()
    # subspaces_potentially_global = [(s.find("(")>=0) and (s.find(")")>=0) for s in subspaces]
    # if sum(subspaces_potentially_global)==1:
    #     # Then assume this is the global one
    #     ind = [i for i, x in enumerate(subspaces_potentially_global) if x==True][0]
    #     map_subspace_to_shorthand = {subspaces[ind]:"global"}
    #     DFDIST["subspace_orig"] = DFDIST["subspace"]
    #     DFDIST["subspace"] = [map_subspace_to_shorthand[s] if s in map_subspace_to_shorthand else s for s in DFDIST["subspace"]]

    return DFDIST, map_question_to_euclideanvars, map_question_to_varsame

def prune_keep_only_middle_strokes(dfdist, question):
    """
    Keep only pairs that do not involve the first or last stroke in the trial's sequence.
    """
    # Only keep pairs that do not include the first or last stroke
    try:
        # Must do this first, or else prune_keep_only_middle_strokes() will fail.
        dfdist = dfdist[dfdist["question"]==question].reset_index(drop=True)
        a = (dfdist["stroke_index_1"] > 0) & (dfdist["stroke_index_1"] < (dfdist["FEAT_num_strokes_beh_1"] - 1))
        b = (dfdist["stroke_index_2"] > 0) & (dfdist["stroke_index_2"] < (dfdist["FEAT_num_strokes_beh_2"] - 1))
    except Exception as err:
        print(dfdist.columns)
        print(dfdist["question"].unique())
        print(dfdist["stroke_index_1"].unique())
        print(dfdist["stroke_index_2"].unique())
        print(dfdist["FEAT_num_strokes_beh_1"].unique())
        print(dfdist["FEAT_num_strokes_beh_2"].unique())
        raise err
    return dfdist[a & b].reset_index(drop=True)


def targeted_pca_MULT_2_plot_single(animal, date, run, SKIP_PLOTS = False, OVERWRITE = True):
    """
    This plots results for a single day, as well as extracting effects for that day and saving.

    LT CHEKCED, the extraction of specific effects for the ones in mansucrit. Skiped other effects.
    Also skipped plots.
    
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params
    # from neuralmonkey.scripts.analy_syntax_good_eucl_state import _targeted_pca_clean_plots_and_dfdist_MULT_plot_single
    from pythonlib.tools.snstools import rotateLabel

    # SAVEDIR = f"/tmp/SYNTAX_TARGETED_PCA_run{run}"
    SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"

    SKIP_Q_7 = True
    do_scatter_datapts = False
    # yvar = "dist_yue_diff_unnorm"
    yvar = "dist_yue_diff"
            
    SAVEDIR = f"{SAVEDIR_MULT}/summary_each_date-yvar={yvar}/{animal}-{date}"
    # Skip if done
    if not OVERWRITE:
        if os.path.exists(f"{SAVEDIR}/DFEFFECT.pkl"):
            return None

    DFDIST, map_question_to_euclideanvars, map_question_to_varsame = targeted_pca_MULT_2_plot_single_load(animal, date, run, yvar)
    
    if DFDIST is None:
        # Then this (animal, date) has no data.
        # print("Has no data!! ", animal, date)
        # assert False
        return None
    
    if not SKIP_PLOTS:
        ##### 1_rankwithin_vs_rank
        question = "1_rankwithin_vs_rank"
        only_within_pig = True
        order = [
            '0|0|1|1|1',
            '0|1|1|1|1',
            '1|0|1|1|1',
            '1|1|0|1|1',
            '1|1|1|0|1']
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ##### 2_ninchunk_vs_rankwithin
        question = "2_ninchunk_vs_rankwithin"
        only_within_pig = True
        order = [
            '0|1|1|1|1|1',
            '1|0|1|1|1|1',
            '1|1|0|1|1|1',
            '1|1|1|0|1|1',
            '1|1|1|1|0|1',
            ]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)


        ##### 3_onset_vs_offset
        question = "3_onset_vs_offset"
        only_within_pig = True
        order = [
            '0|0|0|0|0|1',
            '0|1|1|1|1|1',
            '1|0|1|1|1|1',
            '1|1|0|1|1|1',
            '1|1|1|0|1|1',
            '1|1|1|1|0|1']
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ##### 4_shape_vs_chunk (ie. separating "pure shape" encoding from chunk encoding)
        question = "4_shape_vs_chunk"
        only_within_pig = False
        order = ['0|0|0', '0|0|1', '0|1|1', '1|0|0', '1|1|0']

        DFDIST_THIS, colname_conj_same = plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        if len(DFDIST_THIS)>0:
            _targeted_pca_clean_plots_and_dfdist_MULT_plot_single(DFDIST_THIS, colname_conj_same, question, SAVEDIR, order, yvar=yvar)

            # Cleaner plots...
            dfdist = DFDIST_THIS[DFDIST_THIS["subspace"] == "shape|none|none"].reset_index(drop=True)
            order = sorted(DFDIST_THIS[colname_conj_same].unique())
            if len(dfdist)>0:
                fig = sns.catplot(data=dfdist, x=colname_conj_same, hue="task_kind_12", y=yvar, order=order,
                            col="bregion", col_wrap=6, kind="bar", errorbar="se")
                rotateLabel(fig)
                savefig(fig, f"{SAVEDIR}/q={question}-catplot-clean.pdf")
                plt.close("all")

        ##### 5_rankwithin_vs_rank
        question = "5_rankwithin_vs_rank"
        only_within_pig = True
        order = [
            '0|0|1|1|1|1',
            '0|1|1|1|1|1',
            '1|0|1|1|1|1',
            '1|1|0|1|1|1',
            '1|1|1|0|1|1',
            '1|1|1|1|0|1',
            '1|1|1|0|0|1',
            ]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ##### 6_ninchunk_vs_rankwithin
        question = "6_ninchunk_vs_rankwithin"
        only_within_pig = True
        order = [
            '0|1|1|1|1|1|1',
            '0|0|1|1|1|1|1',
            '1|0|1|1|1|1|1',
            '1|1|0|1|1|1|1',
            '1|1|1|0|1|1|1',
            '1|1|1|1|0|1|1',
            '1|1|1|1|1|0|1',
            '1|1|1|1|0|0|1',
            ]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ##### 7_ninchunk_vs_rankwithin
        question = "7_ninchunk_vs_rankwithin"
        only_within_pig = True
        order = [
            '0|1|1|1|1|1|1',
            '1|0|1|1|1|1|1',
            '1|1|0|1|1|1|1',
            '1|1|1|0|1|1|1',
            '1|1|1|1|0|1|1',
            '1|1|1|1|1|0|1',
            '1|1|1|1|0|0|1',
            ]
        _, colname_conj_same = plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        # This is a case where subtracted confounds before compute euclidean, for 7_ninchunk_vs_rankwithin 
        if not SKIP_Q_7:
            if len(DFDIST["subspace"].unique())==1: # HACK, if multple subspace, should take the one that is global
                DFDIST_THIS = DFDIST[
                    (DFDIST["question"].isin(['6_ninchunk_vs_rankwithin', '7_ninchunk_vs_rankwithin'])) & (DFDIST["task_kind_12"] == "prims_on_grid|prims_on_grid")
                    ].reset_index(drop=True)

                if len(DFDIST_THIS)>0:
                    # Get the dist98 BEFORE subtracting out variables
                    dfdist = DFDIST[
                        (DFDIST["question"].isin(['6_ninchunk_vs_rankwithin'])) & (DFDIST["task_kind_12"] == "prims_on_grid|prims_on_grid")
                        ].reset_index(drop=True)
                    list_bregions = dfdist["bregion"].unique().tolist()
                    assert len(dfdist["DIST_98"].unique()) == len(list_bregions), "This is probably beucase there are multipel subspaces in dfdist. Try dfdist[subspace].unique(). Solve by taking the subspace that is global."
                    map_bregion_to_dist98 = {bregion:dfdist[dfdist["bregion"] == bregion]["DIST_98"].values[0] for bregion in list_bregions}

                    # Apply this to all cases
                    DFDIST_THIS["DIST_98_global"] = [map_bregion_to_dist98[bregion] for bregion in DFDIST_THIS["bregion"]]
                    # Recomputed dist_yue_diff, now normalized to global value
                    DFDIST_THIS["dist_yue_diff_global"] = DFDIST_THIS["dist_yue_diff_unnorm"] / DFDIST_THIS["DIST_98_global"]
                    if False:
                        # Plot showing the DIST_98 differs for the two questoins
                        fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue="question", y="DIST_98",
                                    col=colname_conj_same, col_order=order, col_wrap=6, kind="bar", errorbar="se")
                    order = [
                        '0|1|1|1|1|1|1',
                        '1|0|1|1|1|1|1',
                        '1|1|0|1|1|1|1',
                        '1|1|1|0|1|1|1',
                        '1|1|1|1|0|1|1',
                        '1|1|1|1|1|0|1',
                        '1|1|1|1|0|0|1',
                        ]
                    fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y="dist_yue_diff_unnorm", hue_order=order,
                                col="question", row="subspace", kind="bar", errorbar="se")
                    savefig(fig, f"{SAVEDIR}/q={question}-RENORMED-catplot-1.pdf")
                    fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y="dist_yue_diff_global", hue_order=order,
                                col="question", row="subspace", kind="bar", errorbar="se")
                    savefig(fig, f"{SAVEDIR}/q={question}-RENORMED-catplot-2.pdf")
                    fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y="dist_yue_diff", hue_order=order,
                                col="question", row="subspace", kind="bar", errorbar="se")
                    savefig(fig, f"{SAVEDIR}/q={question}-RENORMED-catplot-3.pdf")


                    ### Combine questions 6 and 7 (7 gives the syntax-related contrasts. 6 gives others, the motor ones)
                    # Contrasts to take from qusetion 7
                    contrasts_from_question_7 = [
                        "0|1|1|1|1|1|1",
                        "1|0|1|1|1|1|1",
                        "1|1|0|1|1|1|1",
                    ]
                    dftmp1 = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(contrasts_from_question_7)) & (DFDIST_THIS["question"] == "7_ninchunk_vs_rankwithin")]
                    dftmp2 = DFDIST_THIS[(~DFDIST_THIS[colname_conj_same].isin(contrasts_from_question_7)) & (DFDIST_THIS["question"] == "6_ninchunk_vs_rankwithin")]
                    DFDIST_THIS_COMBINED = pd.concat([dftmp1, dftmp2]).reset_index(drop=True)
                    DFDIST_THIS_COMBINED["subspace"] = "dummy"
                    _targeted_pca_clean_plots_and_dfdist_MULT_plot_single(DFDIST_THIS_COMBINED, colname_conj_same, "6_7_combined", 
                                                                        SAVEDIR, order, yvar="dist_yue_diff_global")


        ################### TWO SHAPES
        # NOTE: in previous analyses, the effects were
        # var_effect : epoch
        # vars_others : ['syntax_concrete', 'behseq_locs_clust', 'syntax_role']

        # var_effect : syntax_role
        # vars_others : ['syntax_concrete', 'behseq_locs_clust', 'epoch']

        # Below is recapitulating, but more carefully for the syntax effect.
        
        ##### 6_ninchunk_vs_rankwithin
        question = "8_twoshapes"
        only_within_pig = True
        order = [
            '1|0|1|1|1|1|1|1|1',

            '1|1|0|1|1|1|1|1|1',
            '1|1|0|1|1|1|1|0|1',
            '1|1|0|1|1|1|0|0|1',

            '1|1|1|0|1|1|1|1|1',
            '1|1|1|1|0|0|1|1|1',
            
            '0|1|1|0|1|1|1|1|1',
            '0|1|1|0|1|1|1|0|1',
            '0|1|1|0|1|1|0|0|1',
            
            '1|0|0|0|1|1|1|1|1',
            ]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ###
        question = "9_twoshapes"
        only_within_pig = True
        if True: # To make the set, and then prune by hand to the
            vars_this_question = map_question_to_euclideanvars[question]
            this_tuple = [1 for _ in range(len(vars_this_question))]

            order = []
            for i in range(len(vars_this_question)):
                this_tuple_this = this_tuple.copy()
                this_tuple_this[i] = 0
                order.append(this_tuple_this)

            # Add: Effect of shape ()
            _vars_diff = ["epoch", "shape"]
            order.append([0 if _var in _vars_diff else 1 for _var in vars_this_question])

            # Add: Effect of syntax
            _vars_diff = ["chunk_within_rank", "chunk_rank", "shape"]
            order.append([0 if _var in _vars_diff else 1 for _var in vars_this_question])

            order = ["|".join([str(x) for x in this_tuple_this]) for this_tuple_this in order]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)
        
        ## 10
        question = "10_twoshapes"
        only_within_pig = True
        order = [
            '1|0|1|1|1|1|1|1|1',
            '1|1|0|1|1|1|1|1|1',
            '1|1|1|0|1|1|1|1|1',
            '1|1|1|1|0|0|0|1|1',
            '0|1|1|0|1|1|1|1|1',
            '1|0|0|0|1|1|1|1|1']
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ### 11
        question = "11_twoshapes"
        only_within_pig = True
        order = [
            '1|0|1|1|1|1|1|1',
            '1|1|0|1|1|1|1|1',
            '1|1|1|0|1|1|1|1',
            '1|1|1|1|0|0|1|1',
            '1|1|1|1|1|1|0|1',
            '0|1|1|0|1|1|1|1',
            '1|0|0|0|1|1|1|1',
            ]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ###
        question = "12_twoshapes"
        only_within_pig = True
        order = [
            '1|0|1|1|1|1|1|1',
            '1|1|0|1|1|1|1|1',
            '1|1|1|0|1|1|1|1',
            '1|1|1|1|0|0|0|1',
            '0|1|1|0|1|1|1|1',
            '1|0|0|0|1|1|1|1',
            ]
        plot_question_overview(DFDIST, question, only_within_pig, SAVEDIR, yvar, order)

        ### TWO SHAPES -- summarize effects
        list_dfeffect = []

        question = "11_twoshapes"
        only_within_pig = True
        subspaces = "all" 
        
        effect_name = "2sh_rank_within"
        contrasts_diff = ["chunk_within_rank"]
        contrasts_either = ["chunk_rank"]
        # dfeffect = effect_extract_helper_this(DFDIST, question, subspaces, contrasts_diff, contrasts_either, 
        #                             only_within_pig)
        # if dfeffect is not None:
        #     dfeffect["effect"] = effect_name
        #     list_dfeffect.append(dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, contrasts_diff, contrasts_either, 
                                       only_within_pig, effect_name, list_dfeffect)


        effect_name = "2sh_epochshape"
        contrasts_diff = ["epoch", "shape"]
        contrasts_either = []
        # dfeffect = effect_extract_helper_this(DFDIST, question, subspaces, 
        #                             contrasts_diff, contrasts_either, only_within_pig)
        # if dfeffect is not None:
        #     dfeffect["effect"] = effect_name
        #     list_dfeffect.append(dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, contrasts_diff, contrasts_either, 
                                       only_within_pig, effect_name, list_dfeffect)

        
        ### COLLECT ALL EFFECTS into a single dataframe
        if len(list_dfeffect)>0:
            DFEFFECT = pd.concat(list_dfeffect).reset_index(drop=True)

            fig = sns.catplot(DFEFFECT, x="bregion", y=yvar, hue="effect", kind="bar")
            savefig(fig, f"{SAVEDIR}/{question}-effects.pdf")

        ### SHAPE VS. SUPERVISION
        # -- want to highlight to "collapse" in effect in preSMA during supervision
        subspaces = ["global"]
        for question in ["20_sh_vs_superv", "21_sh_vs_superv", "22_sh_vs_superv", "23_sh_vs_superv", "24_sh_vs_superv", "25_sh_vs_superv"]:
            if question in DFDIST["question"].unique().tolist():
                
                # Must do this first, or else prune_keep_only_middle_strokes() will fail.
                DFDIST_THIS = DFDIST[DFDIST["question"]==question].reset_index(drop=True)

                for keep_only_middle_strokes in [False, True]:

                    if keep_only_middle_strokes:
                        DFDIST_THIS = prune_keep_only_middle_strokes(DFDIST_THIS, question)

                    if len(DFDIST_THIS)>0:
                        # try:
                        # _, dfdist, colname_conj_same, _ = effect_extract_helper_this(DFDIST_THIS, question, subspaces, None, None, True, True)
                        tmp = effect_extract_helper_this(DFDIST_THIS, question, subspaces, None, None, True, True)
                        if tmp is None:
                            continue

                        _, dfdist, colname_conj_same, _ = tmp

                        # except Exception as err:
                        #     print("Failed to find data for: ", question, subspaces)
                        #     print("Existing questions:", DFDIST_THIS["question"].unique())
                        #     print("Existing subspace:", DFDIST_THIS["subspace"].unique())
                        #     print("Existing task_kind_12:", DFDIST_THIS["task_kind_12"].unique())
                        #     raise err

                        if len(dfdist)>0:
                            
                            fig = sns.catplot(dfdist, x="bregion", y=yvar, hue=colname_conj_same, col="superv_is_seq_sup_12", kind="bar")
                            savefig(fig, f"{SAVEDIR}/SH_VS_SEQSUP-q={question}-catplot-1-inner={keep_only_middle_strokes}.pdf")
                            
                            fig = sns.catplot(dfdist, x="bregion", y=yvar, hue="superv_is_seq_sup_12", col=colname_conj_same, col_wrap=6, kind="bar")
                            fig.set_titles(size=5) 
                            savefig(fig, f"{SAVEDIR}/SH_VS_SEQSUP-q={question}-catplot-2-inner={keep_only_middle_strokes}.pdf")

                            if False: # too slow
                                fig = sns.catplot(dfdist, x="bregion", y=yvar, hue="superv_is_seq_sup_12", col=colname_conj_same, col_wrap=6, 
                                                jitter=True, alpha=0.25)
                                fig.set_titles(size=5) 
                                savefig(fig, f"{SAVEDIR}/SH_VS_SEQSUP-q={question}-catplot-3-inner={keep_only_middle_strokes}.pdf")

    #######################################
    ##### [NEW EFFECTS SUMMARY]
    # (1) Collect all effects
    list_dfeffect = []

    ################ RANK_WITHIN, CHUNK_RANK, N_IN_CHUNK
    ### From 10_twoshapes
    question = "10_twoshapes"
    only_within_pig = True
    subspaces = ["global"]
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "10_rankwithin", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, "10_chunkrank", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, "10_motor", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_n_in_chunk"], [], only_within_pig, "10_ninchunk", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_n_in_chunk", "chunk_rank"], only_within_pig, "10_shape", list_dfeffect)

    ### From 11_twoshapes
    question = "11_twoshapes"
    only_within_pig = True
    for ss in DFDIST["subspace"].unique().tolist():
        subspaces = [ss]
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, f"11_rankwithin_ss={ss}", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, f"11_chunkrank_ss={ss}", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, f"11_motor_ss={ss}", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_n_in_chunk"], [], only_within_pig, f"11_ninchunk_ss={ss}", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_n_in_chunk", "chunk_rank"], only_within_pig, f"11_shape_ss={ss}", list_dfeffect)

    ### From 6_ninchunk_vs_rankwithin
    if "11_twoshapes" not in DFDIST["question"].unique().tolist():
        # Identical to 11
        question = "6_ninchunk_vs_rankwithin"
        only_within_pig = True
        subspaces = ["global"]
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "6_rankwithin", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, "6_chunkrank", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, "6_motor", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_n_in_chunk"], [], only_within_pig, "6_ninchunk", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_n_in_chunk", "chunk_rank"], only_within_pig, "6_shape", list_dfeffect)

    ### From 6_ninchunk_vs_rankwithin
    if True: # Not controlled enough (Actulaly, is useful)
        question = "2_ninchunk_vs_rankwithin"
        only_within_pig = True
        subspaces = ["global"]
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "2_rankwithin", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, "2_chunkrank", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc"], [], only_within_pig, "2_motor", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_n_in_chunk"], [], only_within_pig, "2_ninchunk", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_n_in_chunk", "chunk_rank"], only_within_pig, "2_shape", list_dfeffect)


    ################ RANK_WITHIN, CHUNK_RANK -- control using syntax_concrete
    ### From 8_twoshapes
    question = "8_twoshapes"
    only_within_pig = True
    subspaces = ["global"]
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "8_rankwithin", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, "8_chunkrank", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, "8_motor", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_rank"], only_within_pig, "8_shape", list_dfeffect)

    question = "13_twoshapes"
    only_within_pig = True
    subspaces = ["global"]
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "13_rankwithin", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, "13_chunkrank", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, "13_motor", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_rank"], only_within_pig, "13_shape", list_dfeffect)

    question = "9_twoshapes"
    only_within_pig = True
    subspaces = ["global"]
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "9_rankwithin", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_rank"], [], only_within_pig, "9_chunkrank", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, "9_motor", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_rank"], only_within_pig, "9_shape", list_dfeffect)

    ############## RANK-UP vs. RANK-DN
    question = "14_onset_vs_offset"
    only_within_pig = True
    subspaces = ["global"]
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "14_rankwithin_up", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank_fromlast"], [], only_within_pig, "14_rankwithin_dn", list_dfeffect)
    effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc", "CTXT_loc_prev"], [], only_within_pig, "14_motor", list_dfeffect)

    if "14_onset_vs_offset" not in DFDIST["question"].unique().tolist():
        # Not controlled enough, compared to 14
        question = "3_onset_vs_offset"
        only_within_pig = True
        subspaces = ["global"]
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "3_rankwithin_up", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank_fromlast"], [], only_within_pig, "3_rankwithin_dn", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["gridloc"], [], only_within_pig, "3_motor", list_dfeffect)

    ############## TWO SHAPES
    # for idx in [9, 11]:
    for idx in [8, 9, 10, 11, 13]:
        question = f"{idx}_twoshapes"
        only_within_pig = True
        subspaces = ["global"]
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_within_rank", "chunk_rank"], only_within_pig, f"{idx}_2SH_syntax", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, f"{idx}_2SH_rankwithin", list_dfeffect)
        effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["epoch", "shape"], [], only_within_pig, f"{idx}_2SH_epochshape", list_dfeffect)

    # question = "11_twoshapes"
    # only_within_pig = True
    # subspaces = ["global"]
    # effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["shape"], ["chunk_within_rank", "chunk_rank"], only_within_pig, "11_2SH_syntax", list_dfeffect)
    # effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["chunk_within_rank"], [], only_within_pig, "11_2SH_rankwithin", list_dfeffect)
    # effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["epoch", "shape"], [], only_within_pig, "11_2SH_epochshape", list_dfeffect)

    ############## SP vs. GRAMMAR
    for idx in ["4", "4c"]:
        question = f"{idx}_shape_vs_chunk"
        only_within_pig = False
        for ss in ["shape|none|none", "global"]:
            subspaces = [ss]
            
            dfdist = DFDIST[(DFDIST["task_kind_12"] == "prims_single|prims_single")]
            effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["shape"], [], only_within_pig, f"{idx}_shapeSP_ss={ss}", list_dfeffect)
            dfdist = DFDIST[(DFDIST["task_kind_12"] == "prims_on_grid|prims_on_grid")]
            effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["shape"], [], only_within_pig, f"{idx}_shapePIG_ss={ss}", list_dfeffect)
            effect_extract_helper_this_wrapper(DFDIST, question, subspaces, ["task_kind"], [], only_within_pig, f"{idx}_shapesyntax_ss={ss}", list_dfeffect)

    # ############## SHAPE VS SEQSUP
    only_within_pig = True
    for ss in ["shape|none|none", "global"]:
        if ss in DFDIST["subspace"].unique().tolist():
            subspaces = [ss]
            for idx in [24, 25]:
                question = f"{idx}_sh_vs_superv"
                if question in DFDIST["question"].unique().tolist():
                    for keep_only_middle_strokes in [False, True]:
                        if keep_only_middle_strokes:
                            DFDIST_THIS = prune_keep_only_middle_strokes(DFDIST, question)
                        else:
                            DFDIST_THIS = DFDIST
                        dfdist = DFDIST_THIS[(DFDIST_THIS["epoch_kind_12"] == "shape|shape")]
                        effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["stroke_index"], ["behseq_shapes"], only_within_pig, f"{idx}_shape_stkidx-inner={keep_only_middle_strokes}-ss={ss}", list_dfeffect)

                        dfdist = DFDIST_THIS[(DFDIST_THIS["epoch_kind_12"] == "dir|dir")]
                        effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["stroke_index"], ["behseq_shapes"], only_within_pig, f"{idx}_dir_stkidx-inner={keep_only_middle_strokes}-ss={ss}", list_dfeffect)

                        dfdist = DFDIST_THIS[(DFDIST_THIS["epoch_kind_12"] == "seqsup|seqsup")]
                        effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["stroke_index"], ["behseq_shapes"], only_within_pig, f"{idx}_seqsup_stkidx-inner={keep_only_middle_strokes}-ss={ss}", list_dfeffect)

                        dfdist = DFDIST_THIS[(DFDIST_THIS["epoch_kind_12"].isin(["shape|seqsup", "seqsup|shape"]))]
                        effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["superv_is_seq_sup", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup"], [], only_within_pig, f"{idx}_shape_vs_superv-inner={keep_only_middle_strokes}-ss={ss}", list_dfeffect)

                        dfdist = DFDIST_THIS[(DFDIST_THIS["epoch_kind_12"].isin(["dir|seqsup", "seqsup|dir"]))]
                        effect_extract_helper_this_wrapper(dfdist, question, subspaces, ["superv_is_seq_sup", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup"], [], only_within_pig, f"{idx}_dir_vs_superv-inner={keep_only_middle_strokes}-ss={ss}", list_dfeffect)

    ### Collect
    DFEFFECT = pd.concat(list_dfeffect).reset_index(drop=True)

    # Save it
    pd.to_pickle(DFEFFECT, f"/{SAVEDIR}/DFEFFECT.pkl")

    ##### Various plots for this set of effects (same subspace)
    if not SKIP_PLOTS:
        order = sorted(DFEFFECT["effect"].unique())
        fig = sns.catplot(data=DFEFFECT, x="effect", y=yvar, hue="bregion", kind="bar", errorbar="se", aspect=2, order=order)
        rotateLabel(fig)
        savefig(fig, f"{SAVEDIR}/effects-ALL-overview-catplot-1.pdf")

        fig = sns.catplot(data=DFEFFECT, x="bregion", y=yvar, hue="effect", kind="bar", errorbar="se", aspect=2)
        rotateLabel(fig)
        savefig(fig, f"{SAVEDIR}/effects-ALL-overview-catplot-2.pdf")

        plt.close("all")

        # Specific plots for each contrast string (i.e., each question)
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        from pythonlib.tools.pandastools import grouping_print_n_samples
        for question in DFEFFECT["question"].unique().tolist():
            dfeffect = DFEFFECT[DFEFFECT["question"] == question].reset_index(drop=True)
                
            fig = sns.catplot(data=dfeffect, x="bregion", y=yvar, hue="contrast_string", col="effect", row="contrast_vars", kind="bar", errorbar="se", aspect=2)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/effects-q={question}-overview-catplot-1.pdf")

            fig = sns.catplot(data=dfeffect, x="bregion", y=yvar, hue="contrast_string", col="effect", row="contrast_vars", aspect=2, alpha=0.25, jitter=True)
            rotateLabel(fig)
            savefig(fig, f"{SAVEDIR}/effects-q={question}-overview-catplot-2.pdf")

            # Print contrast levels
            grouping_print_n_samples(dfeffect, ["bregion", "subspace", "question", "contrast_vars", "effect", "contrast_string"],
                                    savepath=f"{SAVEDIR}/effects-q={question}-counts.txt")

            # Plot heatmaps of counts
            _dfeffect = dfeffect[dfeffect["bregion"]==dfeffect["bregion"].values[0]]
            fig = grouping_plot_n_samples_conjunction_heatmap(_dfeffect, "bregion", colname_conj_same, ["effect"])
            savefig(fig, f"{SAVEDIR}/effects-q={question}-counts-1.pdf")

            fig = grouping_plot_n_samples_conjunction_heatmap(_dfeffect, "effect", "contrast_string", ["contrast_vars"])
            savefig(fig, f"{SAVEDIR}/effects-q={question}-counts-2.pdf")

            plt.close("all")

        #########################################
        ### Plot pairwise effects
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
        # Using effectsimple
        # x, y

        # For each effect pair....
        LIST_EFFECT_PAIRS = get_list_effects()

        for eff1, eff2 in LIST_EFFECT_PAIRS:
            _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effect", eff1, eff2, None, yvar, "bregion");
            
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-1.pdf")

            if False: # Is slow
                _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effect", eff1, eff2, "bregion", yvar, "labels_1", 
                                                    plot_text=False, plot_error_bars=False, shareaxes=True, alpha=0.5);
                if fig is not None:
                    savefig(fig, f"{savedir}/scatter-effect-{eff2}-vs-{eff1}-2.pdf")
            
            plt.close("all")
    
    ############################################
    if False:
        ### COLLECT across dates (each datapt is a single contrast pair of items
        from pythonlib.tools.pandastools import aggregGeneral, stringify_values
        DFEFFECT_STR = stringify_values(DFEFFECT)
        DFEFFECT_STR = aggregGeneral(DFEFFECT_STR, ["labels_1", "labels_2", "var_subspace", "bregion", "question", "subspace", "subspace_orig", 
                                "contrast_vars", "contrast_string", "effect", "task_kind_1", "task_kind_2", "task_kind_12"], 
                                ["dist_yue_diff", "dist_yue_diff_unnorm"])
        DFEFFECT_STR["animal"] = animal
        DFEFFECT_STR["date"] = date
        LIST_DFEFFECT_ALL.append(DFEFFECT_STR)

    if not SKIP_PLOTS:
        ###########################################################
        ### [OLDER] Effects plots.
        ##### Combining all effects into a single set of plots
        try:
            MAP_EFFECT_TO_DATA = {}
            savedir = f"{SAVEDIR}/effect_summary"
            os.makedirs(savedir, exist_ok=True)

            # 1. 
            question = "1_rankwithin_vs_rank"
            colname_conj_same = map_question_to_varsame[question]
            DFDIST_THIS = DFDIST[
                (DFDIST["question"] == question) & (DFDIST["task_kind_12"] == "prims_on_grid|prims_on_grid") & (DFDIST["subspace"].isin(["chunk_rank|none|none", "chunk_within_rank|none|none", "rank_conj|none|none", "global"]))
                ].reset_index(drop=True)

            MAP_EFFECT_TO_DATA["rankwithin_1"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(["0|1|1|1|1"])) & (DFDIST_THIS["subspace"].isin(["chunk_within_rank|none|none", "rank_conj|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["chunkrank_1"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(["1|0|1|1|1"])) & (DFDIST_THIS["subspace"].isin(["chunk_rank|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["gridloc_1"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|1|1|0|1']))]
            MAP_EFFECT_TO_DATA["shape_1"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|1|0|1|1']))]

            # 2.
            question = "2_ninchunk_vs_rankwithin"
            colname_conj_same = map_question_to_varsame[question]
            DFDIST_THIS = DFDIST[
                (DFDIST["question"] == question) & (DFDIST["task_kind_12"] == "prims_on_grid|prims_on_grid") & (DFDIST["subspace"].isin(["chunk_n_in_chunk|none|none", "chunk_within_rank|none|none", "rank_conj|none|none", "global"]))
                ].reset_index(drop=True)

            MAP_EFFECT_TO_DATA["n_in_chunk"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['0|1|1|1|1|1'])) & (DFDIST_THIS["subspace"].isin(["chunk_n_in_chunk|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["n_in_chunk-stroke0"] = DFDIST_THIS[(DFDIST_THIS["chunk_within_rank_12"]=="0|0") & (DFDIST_THIS[colname_conj_same].isin(['0|1|1|1|1|1'])) & (DFDIST_THIS["subspace"].isin(["chunk_n_in_chunk|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["n_in_chunk-stroke1plus"] = DFDIST_THIS[(DFDIST_THIS["chunk_within_rank_12"]!="0|0") & (DFDIST_THIS[colname_conj_same].isin(['0|1|1|1|1|1'])) & (DFDIST_THIS["subspace"].isin(["chunk_n_in_chunk|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["rankwithin-clean"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|0|1|1|1|1'])) & (DFDIST_THIS["subspace"].isin(["chunk_within_rank|none|none", "rank_conj|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["gridloc_2"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|1|1|1|0|1']))]
            MAP_EFFECT_TO_DATA["shape_2"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|1|1|0|1|1']))]

            # 3.
            question = "3_onset_vs_offset"
            colname_conj_same = map_question_to_varsame[question]
            DFDIST_THIS = DFDIST[
                (DFDIST["question"] == question) & (DFDIST["task_kind_12"] == "prims_on_grid|prims_on_grid") & (DFDIST["subspace"].isin(["chunk_within_rank_fromlast|none|none", "chunk_within_rank|none|none", "rank_conj|none|none", "global"]))
                ].reset_index(drop=True)

            MAP_EFFECT_TO_DATA["rankwithin_up"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['0|1|1|1|1|1'])) & (DFDIST_THIS["subspace"].isin(["chunk_within_rank|none|none", "rank_conj|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["rankwithin_down"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|0|1|1|1|1'])) & (DFDIST_THIS["subspace"].isin(["chunk_within_rank_fromlast|none|none", "rank_conj|none|none", "global"]))]
            MAP_EFFECT_TO_DATA["gridloc_3"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|1|1|1|0|1']))]
            MAP_EFFECT_TO_DATA["shape_3"] = DFDIST_THIS[(DFDIST_THIS[colname_conj_same].isin(['1|1|1|0|1|1']))]

            # 4.
            question = "4_shape_vs_chunk"
            colname_conj_same = map_question_to_varsame[question]
            DFDIST_THIS = DFDIST[(DFDIST["question"] == question) & (DFDIST["subspace"].isin(["shape|none|none", "global"]))].reset_index(drop=True)

            MAP_EFFECT_TO_DATA["shape_motor"] = DFDIST_THIS[(DFDIST_THIS["task_kind_12"] == "prims_single|prims_single") & (DFDIST_THIS[colname_conj_same].isin(["0|1|1"]))]
            MAP_EFFECT_TO_DATA["shape_syntax"] = DFDIST_THIS[(DFDIST_THIS["task_kind_same"] == False) & (DFDIST_THIS[colname_conj_same].isin(["1|1|0"]))]

            # Flatten into single dataframe
            _list_df = []
            for effect, _df in MAP_EFFECT_TO_DATA.items():
                _df = _df.copy()
                _df["effect"] = effect
                _list_df.append(_df)
            DFEFFECT = pd.concat(_list_df).reset_index(drop=True)

            # Further aggregation, for final plots
            map_effect_to_effectsimple = {
                'chunkrank_1':'chunk_rank',
                'gridloc_1':'gridloc',
                'gridloc_2':'gridloc',
                'gridloc_3':'gridloc',
                'shape_1':'shape_all',
                'shape_2':'shape_all',
                'shape_3':'shape_all',
                'shape_motor':'shape_motor',
                'shape_syntax':'shape_syntax',
                'n_in_chunk':'n_in_chunk',
                'n_in_chunk-stroke0':'n_in_chunk-stroke0',
                'n_in_chunk-stroke1plus':'n_in_chunk-stroke1plus',
                'rankwithin-clean':'rankwithin',
                'rankwithin_up':'rankwithin_up',
                'rankwithin_down':'rankwithin_down',

                'rankwithin_1':'ignore',
                }

            DFEFFECT["effectsimple"] = [map_effect_to_effectsimple[ef] for ef in DFEFFECT["effect"]]

            if "shape_motor" in DFEFFECT["effect"].unique().tolist():
                map_effect_to_effectsimple_v2 = {
                    'gridloc_1':'motor_shloc',
                    'gridloc_2':'motor_shloc',
                    'gridloc_3':'motor_shloc',
                    'shape_motor':'motor_shloc',
                    'chunkrank_1':'chunk_rank',
                    'shape_syntax':'shape_syntax',
                    'n_in_chunk':'n_in_chunk',
                    'n_in_chunk-stroke0':'n_in_chunk-stroke0',
                    'n_in_chunk-stroke1plus':'n_in_chunk-stroke1plus',
                    'rankwithin-clean':'rankwithin',
                    'rankwithin_up':'rankwithin_up',
                    'rankwithin_down':'rankwithin_down',

                    'shape_1':'ignore',
                    'shape_2':'ignore',
                    'shape_3':'ignore',
                    'rankwithin_1':'ignore',
                    }
            else:
                map_effect_to_effectsimple_v2 = {
                    'gridloc_1':'motor_shloc',
                    'gridloc_2':'motor_shloc',
                    'gridloc_3':'motor_shloc',
                    'shape_1':'motor_shloc',
                    'shape_2':'motor_shloc',
                    'shape_3':'motor_shloc',
                    'chunkrank_1':'chunk_rank',
                    'shape_syntax':'shape_syntax',
                    'n_in_chunk':'n_in_chunk',
                    'n_in_chunk-stroke0':'n_in_chunk-stroke0',
                    'n_in_chunk-stroke1plus':'n_in_chunk-stroke1plus',
                    'rankwithin-clean':'rankwithin',
                    'rankwithin_up':'rankwithin_up',
                    'rankwithin_down':'rankwithin_down',

                    'rankwithin_1':'ignore',
                    }

            DFEFFECT["effectsimple_v2"] = [map_effect_to_effectsimple_v2[ef] for ef in DFEFFECT["effect"]]

            if len(DFEFFECT)>0:
                ### Plots (summary of effects)
                # NOTE: to use this is simple -- get mean effect, take mean over all rows
                # Summary plots of effects
                # Summary plots of effects
                order = sorted(DFEFFECT["effect"].unique())

                if False: # not uusally checked
                    fig = sns.catplot(data=DFEFFECT, x="bregion", y=yvar, hue="effect", kind="bar", errorbar="se", aspect=2, hue_order=order)
                    savefig(fig, f"{savedir}/overview-catplot-1.pdf")

                    fig = sns.catplot(data=DFEFFECT, x="effect", y=yvar, col="bregion", kind="bar", errorbar="se", aspect=1, col_wrap=4, order=order)
                    rotateLabel(fig)
                    savefig(fig, f"{savedir}/overview-catplot-2.pdf")

                fig = sns.catplot(data=DFEFFECT, x="effect", y=yvar, hue="bregion", kind="bar", errorbar="se", aspect=2, order=order)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/overview-catplot-3.pdf")

                fig = sns.catplot(data=DFEFFECT, x="bregion", y=yvar, kind="bar", col="effect", errorbar="se", aspect=0.8, col_order=order, col_wrap=8)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/overview-catplot-4.pdf")

                plt.close("all")

                if False:
                    from neuralmonkey.analyses.euclidian_distance import dfdist_expand_convert_from_triangular_to_full
                    dfdist_expand_convert_from_triangular_to_full(DFEFFECT, euclidean_label_vars, PLOT=True, repopulate_relations=False)
                DFEFFECT = append_col_with_grp_index(DFEFFECT, ["labels_1", "labels_2"], "labels_12")

                #########################################
                ### Plot pairwise effects

                from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
                # Using effectsimple
                # x, y
                for eff1, eff2 in [ 
                    ("shape_all", "rankwithin"),
                    ("gridloc", "rankwithin"),
                    ("shape_motor", "rankwithin"),

                    ("shape_all", "chunk_rank"),
                    ("gridloc", "chunk_rank"),
                    ("shape_motor", "chunk_rank"),

                    ("shape_all", "n_in_chunk"),
                    ("gridloc", "n_in_chunk"),
                    ("shape_motor", "n_in_chunk"),

                    ("shape_motor", "shape_syntax"),
                    ("rankwithin_up", "rankwithin_down"),
                    ("n_in_chunk-stroke1plus", "n_in_chunk-stroke0"),

                    ]:
                    
                    _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effectsimple", eff1, eff2, None, yvar, "bregion");
                    if fig is not None:
                        savefig(fig, f"{savedir}/scatter-effectsimple-{eff2}-vs-{eff1}-1.pdf")

                    if do_scatter_datapts: # Is slow
                        _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effectsimple", eff1, eff2, "bregion", yvar, "labels_1", 
                                                            plot_text=False, plot_error_bars=False, shareaxes=True, alpha=0.5);
                        if fig is not None:
                            savefig(fig, f"{savedir}/scatter-effectsimple-{eff2}-vs-{eff1}-2.pdf")

                    plt.close("all")

                # Using "effectsimple_v2"
                for eff1, eff2 in [
                    ("motor_shloc", "rankwithin"),
                    ("motor_shloc", "chunk_rank"),
                    ("motor_shloc", "n_in_chunk"),
                    ("motor_shloc", "shape_syntax"),
                    ]:
                    
                    _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effectsimple_v2", eff1, eff2, None, yvar, "bregion");
                    if fig is not None:
                        savefig(fig, f"{savedir}/scatter-effectsimple_v2-{eff2}-vs-{eff1}-1.pdf")

                    if do_scatter_datapts: # Is slow
                        _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT, "effectsimple_v2", eff1, eff2, "bregion", yvar, "labels_1", 
                                                            plot_text=False, plot_error_bars=False, shareaxes=True, alpha=0.5);
                        if fig is not None:
                            savefig(fig, f"{savedir}/scatter-effectsimple_v2-{eff2}-vs-{eff1}-2.pdf")

                    plt.close("all")
        
            assert False

        except Exception as err:
            return None

def targeted_pca_MULT_3_combined_plots(animal, run, savesuff, SAVEDIR_MULT=None, return_dfeffect=False):
    """
    This plots results across all days for this animal.

    LT CHECKED
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    
    yvar = "dist_yue_diff"
    if SAVEDIR_MULT is None:
        SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"

    list_dates, _, _, _ = load_preprocess_get_dates(animal, savesuff)
    list_dates = list(set(list_dates))

    ### Load and collect all dates
    LIST_DFEFFECT_ALL = []
    for date in list_dates:
        # Load data
        try:
            SAVEDIR = f"{SAVEDIR_MULT}/summary_each_date-yvar={yvar}/{animal}-{date}"
            dfeffect = pd.read_pickle(f"/{SAVEDIR}/DFEFFECT.pkl")
            dfeffect["animal"] = animal
            dfeffect["date"] = date
            
            # Preprocessing at the individula level
            for i in [1, 2]:
                for col in ["chunk_rank", "chunk_within_rank", "chunk_within_rank_fromlast", "chunk_n_in_chunk"]:
                    colnum = f"{col}_{i}"
                    if colnum in dfeffect.columns:
                        dfeffect[colnum] = [int(x) if not isinstance(x, str) else x for x in dfeffect[colnum]] #dfeffect[colnum].astype(int)          

            LIST_DFEFFECT_ALL.append(dfeffect)
            print("Loaded: ", SAVEDIR)
        except FileNotFoundError as err:
            print("Skipped, did not find file: ", f"/{SAVEDIR}/DFEFFECT.pkl")
            # raise err
            continue

    ############################################
    ### [Combined plot across dates]
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    DFEFFECT_ALL = pd.concat(LIST_DFEFFECT_ALL).reset_index(drop=True)
    DFEFFECT_ALL["index"] = DFEFFECT_ALL.index.tolist()

    # For each effect pair....
    LIST_EFFECT_PAIRS = get_list_effects()

    # Also, agg so that each datapt is a single date.
    from pythonlib.tools.pandastools import aggregGeneral
    DFEFFECT_ALL_AGG = aggregGeneral(DFEFFECT_ALL, ["effect", "animal", "date", "bregion", "question", "subspace"], [yvar])

    if return_dfeffect:
        return DFEFFECT_ALL, LIST_EFFECT_PAIRS

    SAVEDIR = f"{SAVEDIR_MULT}/COMBINED-{animal}"
    os.makedirs(SAVEDIR, exist_ok=True)
    for eff1, eff2 in LIST_EFFECT_PAIRS:
        
        if (eff1 in DFEFFECT_ALL["effect"].unique().tolist()) and (eff2 in DFEFFECT_ALL["effect"].unique().tolist()):

            # Only keep dates that have both effects
            from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
            dfeffect_clean, _ = extract_with_levels_of_conjunction_vars_helper(DFEFFECT_ALL, "effect", ["date", "bregion", "animal", "subspace"], levels_var=[eff1, eff2])
            dfeffect_agg_clean, _ = extract_with_levels_of_conjunction_vars_helper(DFEFFECT_ALL_AGG, "effect", ["date", "bregion", "animal", "subspace"], levels_var=[eff1, eff2])

            # (1) Subplot = date
            print(eff1, eff2)
            print(dfeffect_clean["effect"].unique())
            _, fig = plot_45scatter_means_flexible_grouping(dfeffect_clean, "effect", eff1, eff2, "date", yvar, "bregion", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-1.pdf")

            # (2) Subplot = bregion
            _, fig = plot_45scatter_means_flexible_grouping(dfeffect_clean, "effect", eff1, eff2, "bregion", yvar, "date", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-2.pdf")

            # (3) Single summary plot
            _, fig = plot_45scatter_means_flexible_grouping(dfeffect_clean, "effect", eff1, eff2, None, yvar, "bregion", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-3.pdf")

            # Too slow:
            # _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_ALL, "effect", eff1, eff2, "bregion", yvar, "index", plot_error_bars=False, shareaxes=True, plot_text=False);
            # savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-4.pdf")

            # (3) Single summary plot (dates)
            _, fig = plot_45scatter_means_flexible_grouping(dfeffect_agg_clean, "effect", eff1, eff2, None, yvar, "bregion", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-3-datapt=date.pdf")

            ### Catplot summary
            if False:
                sns.catplot(data=dfeffect_clean, x="bregion", y=yvar, hue="effect", col="date", jitter=True, alpha=0.1)
                sns.catplot(data=dfeffect_clean, x="bregion", y=yvar, hue="effect", jitter=True, alpha=0.1)
            # sns.catplot(data=dfeffect_clean, x="bregion", y=yvar, hue="effect", kind="boxen")
            fig = sns.catplot(data=dfeffect_clean, x="bregion", y=yvar, hue="effect", col="date", kind="bar")
            savefig(fig, f"{SAVEDIR}/effects-catplot-{eff2}-vs-{eff1}-1.pdf")

            ### And a "difference metric"
            from pythonlib.tools.pandastools import summarize_featurediff
            dfsummary, _, _, _, COLNAMES_DIFF = summarize_featurediff(
                    dfeffect_agg_clean, "effect", [eff2, eff1], FEATURE_NAMES=[yvar], 
                    INDEX=["animal", "date", "bregion", "question", "subspace"], return_dfpivot=False) 

            fig = sns.catplot(data=dfsummary, x="bregion", y=COLNAMES_DIFF[0], jitter=True, alpha=0.5)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{SAVEDIR}/effects-diff-{eff2}-vs-{eff1}-1-datapt=date.pdf")

            fig = sns.catplot(data=dfsummary, x="bregion", y=COLNAMES_DIFF[0], kind="bar", errorbar="se")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{SAVEDIR}/effects-diff-{eff2}-vs-{eff1}-2-datapt=date.pdf")

            plt.close("all")

def mult_rankwithin_rsa_up_down_good(plot_version, return_for_debug=False, skip_plots=False, RUN=27):
    """
    Does two possible anlayses 

    LT CHECKED (for plot_version=="rankwithin_up_down_good")
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    import os
    import pandas as pd
    from pythonlib.tools.plottools import savefig
    from neuralmonkey.analyses.euclidian_distance import dfdist_expand_convert_from_triangular_to_full, dfdist_variables_effect_extract_helper, dfdist_variables_generate_var_same
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params

    save_suffix = "AnBmCk_general"
    yvar = "dist_yue_diff"
    # analysis = "n_in_chunk"
    question = "11_twoshapes"
    bregions_only = None
    # bregions_only = ["preSMA"]

    # _list_dates = [240830]
    _list_dates = None

    if plot_version=="rankwithin_up_down_good":
        # Better, maintain careful control of location.
        do_agg_over_location = False 
        prune_enough_data = True
        do_final_agg_over_datapts = False # Not needed, can be False (original). Keep False, or else stuff breaks
    elif plot_version=="rsa_heatmaps_sloppy":
        # First run, making the single pairwise heatmaps across all data.
        do_agg_over_location = True
        prune_enough_data = False
        do_final_agg_over_datapts = False
    else:
        assert False

    for animal in ["Diego", "Pancho"]:
    # for animal in ["Pancho"]:
        list_dates, _, _, _ = load_preprocess_get_dates(animal, save_suffix)
        list_dates = list(set(list_dates))

        # Overwrite
        if _list_dates is not None:
            list_dates = _list_dates

        for allow_diff_loc_pre in [False, True]:
        # for allow_diff_loc_pre in [False, True]:

            for date in list_dates:
                
                ### Load data
                path = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}/MULT/DFDIST-{animal}-{date}.pkl"
                if not os.path.exists(path):
                    continue
                DFDIST = pd.read_pickle(path)
                DFDIST = DFDIST[DFDIST["question"] == question].reset_index(drop=True)

                ### Preprocess, etc.
                from pythonlib.tools.pandastools import append_col_with_grp_index
                for i in [1, 2]:
                    DFDIST = append_col_with_grp_index(DFDIST, [f"chunk_rank_{i}", f"shape_{i}", f"chunk_n_in_chunk_{i}", f"chunk_within_rank_{i}"], f"role_{i}")
                    DFDIST = append_col_with_grp_index(DFDIST, [f"chunk_rank_{i}", f"shape_{i}"], f"chunk_shape_{i}")
                
                # PRune to the relevant data
                label_vars = targeted_pca_clean_plots_and_dfdist_params()["map_question_to_euclideanvars"][question]
                contrasts_diff = ["chunk_n_in_chunk"]
                # contrasts_either = ["chunk_within_rank", "chunk_rank", "shape", "chunk_n_in_chunk"]
                if allow_diff_loc_pre:
                    contrasts_either = ["chunk_within_rank", "CTXT_loc_prev"] # Lenient - allow different "loc pre".
                    n_min_trials = 3 # more data exists, since it's lenient
                else:
                    contrasts_either = ["chunk_within_rank"] # Most strict
                    n_min_trials = 2
                var_same = dfdist_variables_generate_var_same(label_vars)
                dfdist = dfdist_variables_effect_extract_helper(DFDIST, var_same, label_vars, contrasts_diff, contrasts_either)
                del DFDIST
                dfdist = dfdist[dfdist["task_kind_12"] == "prims_on_grid|prims_on_grid"].reset_index(drop=True)
                # aggregGeneral(DFDIST, ["question", "bregion", "subspace"], ["dist_yue_diff"])
                # Expand, so that all pairs are represnted
                
                dfdist_full = dfdist_expand_convert_from_triangular_to_full(dfdist, label_vars, False)
                del dfdist

                if len(dfdist_full)==0:
                    continue

                if do_agg_over_location:
                    # Agg over location etc (datapt = relevant variables)
                    from pythonlib.tools.pandastools import aggregGeneral
                    dfdist_full = aggregGeneral(dfdist_full, ["question", "bregion", "subspace", 
                                        "chunk_within_rank_1", "chunk_within_rank_2", 
                                        "chunk_within_rank_fromlast_1", "chunk_within_rank_fromlast_2",
                                        "chunk_rank_1", "chunk_rank_2",
                                        "chunk_n_in_chunk_1", "chunk_n_in_chunk_2", 
                                        "shape_1", "shape_2",
                                        ], ["dist_yue_diff"])

                # Recommpute chunk withi nrank from last if it doesnt exist
                # import numpy as np
                for i in [1, 2]:
                    tmp = []
                    for _, row in dfdist_full.iterrows():
                        if (f"chunk_within_rank_fromlast_{i}" not in row) or (row[f"chunk_within_rank_fromlast_{i}"] == "none"):
                            # Then recompute
                            assert row[f"chunk_within_rank_{i}"] != "none"
                            assert row[f"chunk_within_rank_{i}"] >= 0
                            this = row[f"chunk_within_rank_{i}"] - row[f"chunk_n_in_chunk_{i}"]
                            # print(row[f"chunk_n_in_chunk_{i}"], row[f"chunk_within_rank_{i}"], this)
                        else:
                            this = row[f"chunk_within_rank_fromlast_{i}"]
                        tmp.append(this)
                        assert -this <= row[f"chunk_n_in_chunk_{i}"]
                        if this == -3:
                            assert row[f"chunk_n_in_chunk_{i}"]>2
                            # print(row[f"chunk_n_in_chunk_{i}"])
                        # assert np.abs(tmp[-1])<=row[f"chunk_n_in_chunk_{i}"]
                    dfdist_full[f"chunk_within_rank_fromlast_{i}"] = tmp
                # Sanity checks:
                # dfdist[dfdist["chunk_n_in_chunk_1"] < -dfdist["chunk_within_rank_fromlast_1"]]
                # dfdist.loc[:, ["chunk_n_in_chunk_1", "chunk_within_rank_1", "chunk_within_rank_fromlast_1"]]
                # dfdist_full[dfdist_full["chunk_n_in_chunk_2"] < -dfdist_full["chunk_within_rank_fromlast_2"]].loc[:, ["labels_1", "chunk_n_in_chunk_1", "chunk_within_rank_1", "chunk_within_rank_fromlast_1"]]

                for i in [1, 2]:
                    dfdist_full[f"chunk_within_rank_{i}"] = dfdist_full[f"chunk_within_rank_{i}"].astype(int)
                    dfdist_full[f"chunk_within_rank_fromlast_{i}"] = dfdist_full[f"chunk_within_rank_fromlast_{i}"].astype(int)

                # A new conjunctive "role" variable.
                for i in [1, 2]:
                    dfdist_full = append_col_with_grp_index(dfdist_full, [f"shape_{i}", f"chunk_n_in_chunk_{i}", f"chunk_within_rank_{i}"], f"role_{i}")
                    # dfdist_full = append_col_with_grp_index(dfdist_full, [f"shape_{i}", f"chunk_n_in_chunk_{i}", f"chunk_within_rank_fromlast_{i}"], f"role_{i}")
                    dfdist_full = append_col_with_grp_index(dfdist_full, [f"chunk_n_in_chunk_{i}", f"chunk_within_rank_{i}"], f"n_rank_{i}")
                    dfdist_full = append_col_with_grp_index(dfdist_full, [f"chunk_n_in_chunk_{i}", f"chunk_within_rank_fromlast_{i}"], f"n_ranklast_{i}")
                    dfdist_full = append_col_with_grp_index(dfdist_full, [f"shape_{i}", f"chunk_n_in_chunk_{i}"], f"shape_n_{i}")
                dfdist_full["chunk_within_rank_1min2"] = dfdist_full["chunk_within_rank_1"] - dfdist_full["chunk_within_rank_2"]
                dfdist_full["chunk_within_rank_fromlast_1min2"] = dfdist_full["chunk_within_rank_fromlast_1"] - dfdist_full["chunk_within_rank_fromlast_2"]

                ### For each pair, determine whether it is (i) same rank_up, (ii) same rank_dn, or (iii) neither
                assert all(dfdist_full["chunk_shape_1"] == dfdist_full["chunk_shape_2"]), "sanity, for latest approach assumes this below."
                dfdist_full["chunk_within_rank_fromlast_same"] = dfdist_full["chunk_within_rank_fromlast_1"] == dfdist_full["chunk_within_rank_fromlast_2"]
                dfdist_full["chunk_within_rank_same"] = dfdist_full["chunk_within_rank_1"] == dfdist_full["chunk_within_rank_2"]
                def f(x):   
                    if x["chunk_within_rank_same"] and x["chunk_within_rank_fromlast_same"]:
                        return "same_both"
                    elif x["chunk_within_rank_same"] and not x["chunk_within_rank_fromlast_same"]:
                        return "same_fromstart"
                    elif not x["chunk_within_rank_same"] and x["chunk_within_rank_fromlast_same"]:
                        return "same_fromlast"
                    else:
                        return "neither"
                dfdist_full["chunk_within_rank_pair_class"] = dfdist_full.apply(f, axis=1)

                if prune_enough_data:
                    # Prune to just those pairs that have enough data
                    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
                    dfdist_full_clean, _ = extract_with_levels_of_conjunction_vars_helper(dfdist_full, "chunk_within_rank_pair_class", 
                                                                ["bregion", "chunk_shape_1", "chunk_n_in_chunk_1", "chunk_within_rank_1", "chunk_shape_2", "chunk_n_in_chunk_2"], 
                                                                n_min_trials, levels_var=["same_fromstart", "same_fromlast"])
                    del dfdist_full
                else:
                    dfdist_full_clean = dfdist_full

                if return_for_debug:
                    return dfdist_full_clean

                if len(dfdist_full_clean)==0:
                    print(f"SKIPPING - not enough data for {animal}, {date}, allow_diff_loc_pre={allow_diff_loc_pre}")
                    continue

                ### PLOTS (v1) - pairwise, combining across locations, and directions. This is sloppy, but gets a holistic picture.
                if plot_version=="rsa_heatmaps_sloppy": 
                    SAVEDIR_PLOTS = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}/MULT/compare_ranks_within/allow_diff_loc_pre={allow_diff_loc_pre}/{animal}-{date}"
                    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

                    from pythonlib.tools.pandastools import plot_subplots_heatmap
                    list_bregion = dfdist_full["bregion"].unique().tolist()
                    for bregion in list_bregion:

                        if (bregions_only is not None) and (bregion not in bregions_only):
                            continue
                        
                        ### Heatmaps
                        df = dfdist_full[
                            (dfdist_full["bregion"] == bregion) &  
                            (dfdist_full["chunk_rank_1"] == dfdist_full["chunk_rank_2"]) & 
                            (dfdist_full["shape_1"] == dfdist_full["shape_2"])
                            ].reset_index(drop=True)

                        # plot_subplots_heatmap(df, "role_1", "role_2", "dist_yue_diff", "shape_12", False, False, W = 8)
                        # plot_subplots_heatmap(df, "role_1", "role_2", "dist_yue_diff", None, False, False, W = 8)
                        fig, _ = plot_subplots_heatmap(df, "role_1", "role_2", "dist_yue_diff", "shape_1", False, False, W = 8)
                        savefig(fig, f"{SAVEDIR_PLOTS}/{bregion}-heatmap-same_shape-1.pdf")
                        plt.close("all")

                        
                        ### Catplots
                        for only_if_diff_n in [False, True]:
                            if only_if_diff_n:
                                df = dfdist_full[
                                    (dfdist_full["bregion"] == bregion) &  
                                    (dfdist_full["chunk_rank_1"] == dfdist_full["chunk_rank_2"]) & 
                                    (dfdist_full["shape_1"] == dfdist_full["shape_2"]) & 
                                    (dfdist_full["chunk_n_in_chunk_1"] != dfdist_full["chunk_n_in_chunk_2"])
                                    ].reset_index(drop=True)
                            else:
                                df = dfdist_full[
                                    (dfdist_full["bregion"] == bregion) &  
                                    (dfdist_full["chunk_rank_1"] == dfdist_full["chunk_rank_2"]) & 
                                    (dfdist_full["shape_1"] == dfdist_full["shape_2"])
                                    ].reset_index(drop=True)

                            savedir = f"{SAVEDIR_PLOTS}/only_if_diff_n={only_if_diff_n}"
                            os.makedirs(savedir, exist_ok=True)

                            # Good plot
                            # Plot aligned from end
                            col_order = sorted(df["n_ranklast_2"].unique())
                            fig = sns.catplot(data=df, x="chunk_within_rank_fromlast_1", y = yvar, hue="chunk_n_in_chunk_1", 
                                row="shape_2", col="n_ranklast_2", kind="point", errorbar="se", col_order=col_order)
                            savefig(fig, f"{savedir}/{bregion}-catplot-1a.pdf")

                            # Plot aligned from onset
                            col_order = sorted(df["n_rank_2"].unique())
                            fig = sns.catplot(data=df, x="chunk_within_rank_1", y = yvar, hue="chunk_n_in_chunk_1", 
                                row="shape_2", col="n_rank_2", kind="point", errorbar="se", col_order=col_order)
                            savefig(fig, f"{savedir}/{bregion}-catplot-1b.pdf")

                            # Good plot
                            # col_order = sorted(df["n_ranklast_2"].unique())
                            # fig = sns.catplot(data=df, x="chunk_within_rank_fromlast_1min2", y = yvar, hue="chunk_n_in_chunk_1", 
                            #     row="shape_2", col="n_ranklast_2", kind="point", errorbar="se", col_order=col_order)
                            # savefig(fig, f"{savedir}/{bregion}-catplot-3.pdf")

                            # Good plot
                            col_order = sorted(df["chunk_within_rank_fromlast_2"].unique())
                            fig = sns.catplot(data=df, x="chunk_within_rank_fromlast_1", y = yvar, hue="chunk_n_in_chunk_1", 
                                row="shape_2", col="chunk_within_rank_fromlast_2", kind="point", errorbar="se", col_order=col_order)
                            savefig(fig, f"{savedir}/{bregion}-catplot-2a.pdf")

                            col_order = sorted(df["chunk_within_rank_2"].unique())
                            fig = sns.catplot(data=df, x="chunk_within_rank_1", y = yvar, hue="chunk_n_in_chunk_1", 
                                row="shape_2", col="chunk_within_rank_2", kind="point", errorbar="se", col_order=col_order)
                            savefig(fig, f"{savedir}/{bregion}-catplot-2b.pdf")

                            # Summary good.
                            # for hue in ["n_ranklast_2", "chunk_within_rank_fromlast_2"]:
                            for hue in ["chunk_within_rank_fromlast_2"]:
                                fig = sns.catplot(data=df, x="chunk_within_rank_fromlast_1", y = yvar, hue=hue, 
                                                col="shape_2", col_wrap=10, kind="point", errorbar="se")
                                savefig(fig, f"{savedir}/{bregion}-catplot-3a-{hue}.pdf")

                            # for hue in ["n_rank_2", "chunk_within_rank_2"]:
                            for hue in ["chunk_within_rank_2"]:
                                fig = sns.catplot(data=df, x="chunk_within_rank_1", y = yvar, hue=hue, 
                                                col="shape_2", col_wrap=10, kind="point", errorbar="se")
                                savefig(fig, f"{savedir}/{bregion}-catplot-3b-{hue}.pdf")

                            # Summary combining shapes
                            for this in ["chunk_within_rank_fromlast", "chunk_within_rank"]:
                                fig = sns.catplot(data=df, x=f"{this}_1", y = yvar, hue=f"{this}_2",
                                                kind="point", errorbar="se")
                                savefig(fig, f"{savedir}/{bregion}-catplot-4-{this}.pdf")

                                # Finaly summary all aligned to 0 (combine shapes and n chunk)
                                fig = sns.catplot(data=df, x=f"{this}_1min2", y = yvar, hue=f"{this}_2",
                                                kind="point", errorbar="se")
                                savefig(fig, f"{savedir}/{bregion}-catplot-5-{this}.pdf")


                            plt.close("all")


                ### PLOTS (v2) - Good, replacing the old analysis of rank encoding (up vs. down), but carefully controlling so that
                # the effect is controlling for differences in length (each datapt is a pair of lengths)
                elif plot_version=="rankwithin_up_down_good":
                    # for min_n_in_chunk in [1, 2]:
                        
                    #     dfdist_full_clean_this = dfdist_full_clean[
                    #         (dfdist_full_clean["chunk_n_in_chunk_1"] >= (min_n_in_chunk-0.001)) & 
                    #         (dfdist_full_clean["chunk_n_in_chunk_2"] >= (min_n_in_chunk-0.001))].reset_index(drop=True)
                        
                    #     SAVEDIR_PLOTS = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}/MULT/compare_ranks_within_v2_clean_up_vs_dn/allow_diff_loc_pre={allow_diff_loc_pre}-min_n_in_chunk={min_n_in_chunk}/{animal}-{date}"
                    #     os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

                    #     from neuralmonkey.scripts.analy_syntax_good_eucl_state import mult_plot_rankwithin_up_vs_down_good
                    #     mult_plot_rankwithin_up_vs_down_good(dfdist_full_clean_this, SAVEDIR_PLOTS, do_final_agg_over_datapts)
                    # Plot shwing the relationshiip between ordinal 1 and ordinal 2
                    SAVEDIR_PLOTS = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}/MULT/compare_ranks_within_v2_clean_up_vs_dn/allow_diff_loc_pre={allow_diff_loc_pre}/{animal}-{date}"
                    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

                    from pythonlib.tools.pandastools import plot_subplots_heatmap
                    if "shape_n_1" in dfdist_full_clean.columns:
                        fig, _ = plot_subplots_heatmap(dfdist_full_clean, "chunk_within_rank_1min2", 
                            "chunk_within_rank_fromlast_1min2", "dist_yue_diff", "shape_n_1")
                        savefig(fig, f"{SAVEDIR_PLOTS}/good_contrasts_heatmap.pdf")

                    from pythonlib.tools.pandastools import grouping_print_n_samples
                    savepath = f'{SAVEDIR_PLOTS}/good_counts_data.txt'
                    grouping_print_n_samples(dfdist_full_clean, ["chunk_shape_1", "chunk_n_in_chunk_1", 
                        "chunk_shape_2", "chunk_n_in_chunk_2", 
                        "chunk_within_rank_pair_class", "chunk_within_rank_1min2", "chunk_within_rank_fromlast_1min2",
                        "chunk_within_rank_1", "chunk_within_rank_2", "chunk_within_rank_fromlast_1", "chunk_within_rank_fromlast_2"],
                        savepath=savepath)

                    plt.close("all")

                    from neuralmonkey.scripts.analy_syntax_good_eucl_state import mult_plot_rankwithin_up_vs_down_good
                    mult_plot_rankwithin_up_vs_down_good(dfdist_full_clean, SAVEDIR_PLOTS, do_final_agg_over_datapts, skip_plots=skip_plots)                    
                else:
                    assert False


def mult_rankwithin_rsa_up_down_good_v2(allow_diff_loc_pre = False, n_min_in_chunk=2, n_min=2, do_agg = True,
        just_return_dfdist=False, RUN=27):
    """

    n_min = 2 # NOTE: if 3, then throws out too much.

    """
    ### Load all the dates
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    save_suffix = "AnBmCk_general"
    SAVEDIR_PLOTS_ALL = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{RUN}/MULT/compare_ranks_within_v2_clean_up_vs_dn/allow_diff_loc_pre={allow_diff_loc_pre}"

    list_dfdist = []
    for animal in ["Diego", "Pancho"]:
        list_dates, _, _, _ = load_preprocess_get_dates(animal, save_suffix)
        list_dates = list(set(list_dates))

        print(f"Desired dates for {animal}: {list_dates}")

        for date in list_dates:
            
            if animal=="Diego" and date in [230730, 250416, 250417]:
                continue
            
            if animal=="Pancho" and date in [220831, 250321]:
                # No stress -- these are just dates that dont even exist
                continue
            
            ### Load data
            path = f"{SAVEDIR_PLOTS_ALL}/{animal}-{date}/dfdist_full_clean.pkl"
            if os.path.exists(path):
                dfdist_full_clean = pd.read_pickle(path)
                dfdist_full_clean["animal"] = animal
                dfdist_full_clean["date"] = date
                list_dfdist.append(dfdist_full_clean)
            else:
                print("Skipping, didn't find: ", path)

    DFDIST = pd.concat(list_dfdist).reset_index(drop=True)
    
    savedir = f"{SAVEDIR_PLOTS_ALL}/dynamic_align_on_off/allow_diff_loc_pre={allow_diff_loc_pre}-n_min_in_chunk={n_min_in_chunk}-n_min={n_min}-do_agg={do_agg}"
    os.makedirs(savedir, exist_ok=True)

    # Further cleaning
    analysis = "rank_within"
    # analysis = "rank_up_vs_down"
    tmp = []
    for animal in DFDIST["animal"].unique().tolist():
        dftmp = DFDIST[DFDIST["animal"] == animal].reset_index(drop=True)
        dftmp = _final_dfeffect_postprocess_clean(dftmp, analysis, animal)
        tmp.append(dftmp)
    DFDIST = pd.concat(tmp).reset_index(drop=True)
    print(len(DFDIST))

    ### ADD NEW VARIABLES
    # A new variable, to allow combining across different length sequences
    def f(x):
        if x["chunk_within_rank_1"]>1:
            rank1 = 2
        else:
            rank1 = x["chunk_within_rank_1"]

        if x["chunk_within_rank_fromlast_1"]<-2:
            rank2 = -3
        else:
            rank2 = x["chunk_within_rank_fromlast_1"]

        return (rank1, rank2)
    DFDIST["chunk_within_rank_1_both"] = DFDIST.apply(f, axis=1)
    order_cwrboth = ['0|-3', '0|-2', '1|-3', '2|-3', '1|-2', '2|-2', '0|-1', '1|-1', '2|-1']

    # Further simplify, based on what looks good by eye.
    # - ie count from end dominates.
    # A new variable, to allow combining across different length sequences
    def f(x):
        if x["chunk_within_rank_fromlast_1"] == -1:
            return 99
        elif (x["chunk_within_rank_1"]==0):
            return 0
        elif x["chunk_within_rank_fromlast_1"] == -2:
            return 98
        elif (x["chunk_within_rank_1"]==1) & (x["chunk_within_rank_fromlast_1"] < -2):
            return 1
        elif (x["chunk_within_rank_1"]>1) & (x["chunk_within_rank_fromlast_1"] < -2):
            return 50
        else:
            print(x["chunk_within_rank_1"])
            print(x["chunk_within_rank_fromlast_1"])
            assert False
    DFDIST["chunk_within_rank_1_both_clean"] = DFDIST.apply(f, axis=1)
    order_cwrbothclean = sorted(DFDIST["chunk_within_rank_1_both_clean"].unique())

    # An even cleaner way to group the conditions. (for final plot)
    def f(x):
        if (x["chunk_within_rank_1"] < 2) & (x["chunk_within_rank_fromlast_1"]<-2):
            return "start"
        elif (x["chunk_within_rank_1"] >= 2) & (x["chunk_within_rank_fromlast_1"]>=-2):
            return "end"
        else:
            return "middle"
    DFDIST["chunk_within_rank_1_both_semantic"] = DFDIST.apply(f, axis=1)
        
    ### PLOTS
    from pythonlib.tools.pandastools import grouping_print_n_samples
    savepath = f"{savedir}/counts.txt"
    # grouping_print_n_samples(DFDIST, ["chunk_within_rank_1_both_semantic", "chunk_within_rank_1_both_clean", "chunk_within_rank_1_both"])
    grouping_print_n_samples(DFDIST, ["chunk_within_rank_1_both_clean", "chunk_within_rank_1", "chunk_within_rank_fromlast_1", "chunk_within_rank_1_both", "chunk_within_rank_1_both_semantic"], savepath=savepath)

    ### CLeanup
    _dfdist = DFDIST.copy()
    print(len(_dfdist))

    # (1) clean up 
    _dfdist = _dfdist[
        (_dfdist["chunk_n_in_chunk_1"]>=(n_min_in_chunk-0.001)) & 
        (_dfdist["chunk_n_in_chunk_2"]>=(n_min_in_chunk-0.001))
        ].reset_index(drop=True)
    print(len(_dfdist))

    _dfdist = _dfdist[
        (_dfdist["n1"]>=(n_min-0.001)) & 
        (_dfdist["n2"]>=(n_min-0.001))
        ].reset_index(drop=True)
    print(len(_dfdist))

    # (2) Agg
    if do_agg: # TRue
        from pythonlib.tools.pandastools import aggregGeneral
        var_datapt_grp = ["animal", "date", "bregion", "chunk_shape_1", "chunk_n_in_chunk_1", "chunk_within_rank_1", "chunk_within_rank_fromlast_1", "chunk_within_rank_1_both", "chunk_within_rank_1_both_clean", "chunk_within_rank_1_both_semantic", "chunk_n_in_chunk_2"]
        # var_datapt_grp = ["animal", "date", "bregion", "chunk_shape_1", "chunk_n_in_chunk_1", "chunk_within_rank_1", "chunk_within_rank_fromlast_1", "chunk_within_rank_1_both"]
        var_grp = var_datapt_grp + ["chunk_within_rank_pair_class", "chunk_rank_1"]
        DFDIST_AGG = aggregGeneral(_dfdist, var_grp, ["dist_yue_diff"])
    else:
        DFDIST_AGG = _dfdist

    from pythonlib.tools.pandastools import append_col_with_grp_index
    DFDIST_AGG = append_col_with_grp_index(DFDIST_AGG, ["chunk_rank_1", "chunk_n_in_chunk_1"], "chunk_rank_n_1")

    if just_return_dfdist:
        return DFDIST_AGG

    ### PLOTS
    from pythonlib.tools.plottools import savefig

    # Plot
    for x in ["chunk_within_rank_1", "chunk_within_rank_fromlast_1"]:
        fig = sns.relplot(data=DFDIST_AGG, x=x, y="dist_yue_diff", 
                hue="chunk_within_rank_pair_class", col="bregion", kind="line",
                row="animal")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)                    
        savefig(fig, f"{savedir}/relplot_overview-1-x={x}.pdf")

        fig = sns.relplot(data=DFDIST_AGG, x=x, y="dist_yue_diff", 
                hue="chunk_within_rank_pair_class", col="bregion", alpha=0.5,
                row="animal")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)                    
        savefig(fig, f"{savedir}/relplot_overview-2-x={x}.pdf")
        
        plt.close("all")

    for animal in DFDIST_AGG["animal"].unique().tolist():
        dfdist = DFDIST_AGG[DFDIST_AGG["animal"] == animal]
        for row in ["chunk_n_in_chunk_1", "chunk_rank_1", "chunk_rank_n_1"]:
            fig = sns.relplot(data=dfdist, x="chunk_within_rank_fromlast_1", y="dist_yue_diff", 
                    errorbar="se", 
                    hue="chunk_within_rank_pair_class", col="bregion", kind="line",
                    row=row)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)              

            savefig(fig, f"{savedir}/relplot_row={row}-ani={animal}.pdf")
            plt.close("all")

    ### As function of chunk_within "both"
    from pythonlib.tools.pandastools import stringify_values
    DFDIST_AGG_STR = stringify_values(DFDIST_AGG)
    list_x_order = [
        ("chunk_within_rank_1_both_semantic", ['start', 'middle', 'end']),
        ("chunk_within_rank_1_both", order_cwrboth),
        ("chunk_within_rank_1_both_clean", order_cwrbothclean),
    ]
    for x, order in list_x_order:

        fig = sns.catplot(data=DFDIST_AGG_STR, x=x, y="dist_yue_diff", 
                hue="chunk_within_rank_pair_class", col="bregion", kind="point",
                errorbar="se", 
                row="animal", order=order)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)              
        savefig(fig, f"{savedir}/catplot_good-x={x}-1.pdf")

        fig = sns.catplot(data=DFDIST_AGG_STR, x=x, y="dist_yue_diff", 
                hue="chunk_within_rank_pair_class", col="bregion", kind="point", order=order, errorbar="se")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)              
        savefig(fig, f"{savedir}/catplot_good-x={x}-2.pdf")

        if True: # Bar plots
            fig = sns.catplot(data=DFDIST_AGG_STR, x=x, y="dist_yue_diff", 
                    hue="chunk_within_rank_pair_class", col="bregion", kind="bar", order=order, errorbar="se")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)              
            savefig(fig, f"{savedir}/catplot_good-x={x}-3.pdf")

        plt.close("all")

    # GOOD PLOT (MS) - excuding cases that are length 2
    x = "chunk_within_rank_1_both"
    order = ['0|-3', '1|-3',    '2|-3', '1|-2',     '2|-2', '2|-1'] # excluding length 2: '0|-2',  '1|-1', '0|-1',
    
    fig = sns.catplot(data=DFDIST_AGG_STR, x=x, y="dist_yue_diff", 
            hue="chunk_within_rank_pair_class", col="bregion", kind="point",
            errorbar="se", 
            row="animal", order=order)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)              
    savefig(fig, f"{savedir}/catplot_good-x={x}-nolength2-1.pdf")

    fig = sns.catplot(data=DFDIST_AGG_STR, x=x, y="dist_yue_diff", 
            hue="chunk_within_rank_pair_class", col="bregion", kind="point",
            errorbar="se", order=order)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)              
    savefig(fig, f"{savedir}/catplot_good-x={x}-nolength2-2.pdf")

    # Compare each experiment (date)
    list_x_order = [
        ("chunk_within_rank_1_both", order_cwrboth),
    ]
    for x, order in list_x_order:
        for animal in DFDIST_AGG_STR["animal"].unique().tolist():
            dftmp_this = DFDIST_AGG_STR[DFDIST_AGG_STR["animal"] == animal].reset_index(drop=True)

            fig = sns.catplot(data=dftmp_this, x=x, y="dist_yue_diff", 
                    errorbar="se", 
                    hue="chunk_within_rank_pair_class", col="bregion", kind="point",
                    row="date", order=order)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)              
            
            savefig(fig, f"{savedir}/catplot_good-x={x}-4-ani={animal}.pdf")
            plt.close("all")

    ## Heatmap
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    for bregion in DFDIST_AGG["bregion"].unique().tolist():
        df = DFDIST_AGG[DFDIST_AGG["bregion"]==bregion].reset_index(drop=True)
        fig, _ = plot_subplots_heatmap(df, "chunk_within_rank_1", "chunk_within_rank_fromlast_1", 
                                          "dist_yue_diff", "chunk_within_rank_pair_class", False, True)
        savefig(fig, f"{savedir}/heatmap-1-{bregion}.pdf")
        plt.close("all")

    ### SCATTER
    from pythonlib.tools.plottools import map_continuous_var_to_color_range

    list_x_order = [
        ("chunk_within_rank_1_both", order_cwrboth),
        ("chunk_within_rank_1_both_clean", order_cwrbothclean),
        ("chunk_within_rank_1_both_semantic", ['start', 'middle', 'end']),
    ]
    for var_datapt, order in list_x_order:
        pcols = map_continuous_var_to_color_range(np.linspace(0, 1, len(order)), 0, 1)
        assert len(pcols)==len(order)

        map_order_to_col = {o:col for o, col in zip(order, pcols)}
        # Plot in 2d dynamics space
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

        _, fig = plot_45scatter_means_flexible_grouping(DFDIST_AGG_STR, "chunk_within_rank_pair_class", 
                                            "same_fromstart", "same_fromlast", "bregion", 
                                            "dist_yue_diff", var_datapt, False, shareaxes=True, 
                                            plot_error_bars=True, alpha=0.5, 
                                            map_dataptlev_to_color=map_order_to_col)
        from pythonlib.tools.plottools import legend_add_manual
        legend_add_manual(fig.axes[-1], map_order_to_col.keys(), map_order_to_col.values())
        savefig(fig, f"{savedir}/scatter_good-datapt={var_datapt}-1.pdf")
        plt.close("all")

        # Also one for each animal
        for animal in DFDIST_AGG_STR["animal"].unique().tolist():
            dftmp_this = DFDIST_AGG_STR[DFDIST_AGG_STR["animal"]==animal].reset_index(drop=True)
            
            _, fig = plot_45scatter_means_flexible_grouping(dftmp_this, "chunk_within_rank_pair_class", 
                                                "same_fromstart", "same_fromlast", "bregion", 
                                                "dist_yue_diff", var_datapt, False, shareaxes=True, 
                                                plot_error_bars=True, alpha=0.5, 
                                                map_dataptlev_to_color=map_order_to_col)
            from pythonlib.tools.plottools import legend_add_manual
            legend_add_manual(fig.axes[-1], map_order_to_col.keys(), map_order_to_col.values())
            savefig(fig, f"{savedir}/scatter_good-datapt={var_datapt}-ani={animal}.pdf")
            plt.close("all")

    if False: # sns.replot my not be ideal...
        # Overlay on the same scatterplot
        from pythonlib.tools.pandastools import pivot_table

        var_datapt_grp = ["bregion", "chunk_shape_1", "chunk_n_in_chunk_1", "chunk_within_rank_1", "chunk_within_rank_fromlast_1", "chunk_n_in_chunk_2", "chunk_within_rank_1_both", "chunk_within_rank_1_both_clean"]
        dfpivot = pivot_table(DFDIST_AGG, index=var_datapt_grp, 
                            columns=["chunk_within_rank_pair_class"], values=["dist_yue_diff"], flatten_col_names=True)

        # for y in ["dist_yue_diff-same_fromstart", "dist_yue_diff-same_fromlast"]:
        #     fig = sns.relplot(data=dfpivot, x="chunk_within_rank_fromlast_1", y=y, hue="chunk_shape_1", kind="line", col="bregion")
        #     for ax in fig.axes.flatten():
        #         ax.axhline(0, color="k", alpha=0.5)                    
        dfpivot
        fig = sns.relplot(data=dfpivot, x="dist_yue_diff-same_fromstart", y="dist_yue_diff-same_fromlast", hue="chunk_within_rank_1_both_clean", col="bregion", kind="scatter")


    ############### STATS
    # For each x value (e./g, 0|1) do sign rank comparing "diff from last" vs. "diff from start"
    from pythonlib.tools.statstools import compute_all_pairwise_signrank_wrapper
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    contrast_var = "chunk_within_rank_pair_class"
    list_bregion = DFDIST_AGG_STR["bregion"].unique()
    list_x_order = [
        ("chunk_within_rank_1_both_semantic", ['start', 'middle', 'end']),
        ("chunk_within_rank_1_both", ['0|-3', '0|-2', '1|-3', '2|-3', '1|-2', '2|-2', '0|-1', '1|-1', '2|-1']),
    ]

    for bregion in list_bregion:
        DFDIST_THIS = DFDIST_AGG_STR[DFDIST_AGG_STR["bregion"] == bregion].reset_index(drop=True)

        for x, order in list_x_order:
            
            savedir_this = f"{savedir}/stats/{bregion}-x={x}"
            os.makedirs(savedir_this, exist_ok=True)

            # Collect results from stats.
            res = []
            _grpvars = ["animal", x]
            grpdict = grouping_append_and_return_inner_items_good(DFDIST_THIS, _grpvars)
            for grp, inds in grpdict.items():
                dfdist = DFDIST_THIS.iloc[inds].reset_index(drop=True)
                dfres = compute_all_pairwise_signrank_wrapper(dfdist, 
                    ["date", "chunk_shape_1", "chunk_n_in_chunk_1", "chunk_n_in_chunk_2", "chunk_within_rank_1", "chunk_within_rank_fromlast_1"], 
                    contrast_var, "dist_yue_diff")
                dfres[_grpvars[0]] = grp[0]
                dfres[_grpvars[1]] = grp[1]
                res.append(dfres)
            dfstats = pd.concat(res).reset_index(drop=True)

            # Plot summary
            dfstats = dfstats[dfstats["grp1"] != dfstats["grp2"]].reset_index(drop=True) # Just care about count up vs. count down.
            dfstats["logp"] = np.log10(dfstats["pval"])

            # Plots
            fig = sns.catplot(dfstats, x=x, y="diff_2min1", hue="animal", order=order)
            for ax in fig.axes.flatten():
                ax.axhline(0)
            savefig(fig, f"{savedir_this}/catplot-valdiffs.pdf")

            fig = sns.catplot(dfstats, x=x, y="logp", hue="animal", order=order)
            for ax in fig.axes.flatten():
                ax.axhline(np.log10(0.05))
            savefig(fig, f"{savedir_this}/catplot-log10pval.pdf")

            dfstats.to_csv(f"{savedir_this}/stats.txt")

            plt.close("all")

if __name__=="__main__":
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates

    # animal = sys.argv[1]
    # date = int(sys.argv[2])
    RUN = int(sys.argv[1])
    plot_each_date=int(sys.argv[2])==1
    # RUN = 15

    PLOTS_DO = [2.0, 2.1, 2.2] # Good
    # PLOTS_DO = [2.1, 2.2] # Good
    # PLOTS_DO = [2.1] # Good
    # expt_kind="RULE_ANBMCK_STROKE"
    # expt_kind="RULESW_ANY_SEQSUP_STROKE"

    if RUN in [13, 25, 26]:
        save_suffix = "sh_vs_seqsup"
    else:
        save_suffix = "AnBmCk_general"
    # dates, question, _, _ = load_preprocess_get_dates("Diego", "sh_vs_seqsup")

    # MS check
    # PLOTS_DO = [2.1] # Good

    for plotdo in PLOTS_DO:
        if plotdo==1.0:
            """ Older plots, before doing the good Targeted PCA (2.0+)"""
            mult_plot_all_wrapper()

        elif plotdo==2.0:
            """ Step 1: Save a single DFDIST, one for each (animal, date)"""
            ### Collect all the animal-date pairs
            from multiprocessing import Pool
            MULTIPROCESS_N_CORES = 8
            if False: # Parallel across all dates/animals
                LIST_ANIMAL = []
                LIST_DATE = []
                for animal in ["Diego", "Pancho"]:
                    list_dates, question, _, _ = load_preprocess_get_dates(animal, save_suffix)
                    list_dates = list(set(list_dates))

                    for date in list_dates:
                        LIST_ANIMAL.append(animal)
                        LIST_DATE.append(date)
                
                ### Run
                if True:
                    MULTIPROCESS_N_CORES = 24
                    list_run = [RUN for _ in range(len(LIST_ANIMAL))]
                    list_expt_kind = [question for _ in range(len(LIST_ANIMAL))]
                    with Pool(MULTIPROCESS_N_CORES) as pool:
                        pool.starmap(targeted_pca_MULT_1_load_and_save, zip(LIST_ANIMAL, LIST_DATE, list_run, list_expt_kind))
                else:
                    for animal, date in zip(LIST_ANIMAL, LIST_DATE):
                        targeted_pca_MULT_1_load_and_save(animal, date, run=RUN, expt_kind=question)
            else:
                # Each animal in turn
                for animal in ["Diego", "Pancho"]:
                    LIST_DATE, question, _, _ = load_preprocess_get_dates(animal, save_suffix)
                    LIST_DATE = list(set(LIST_DATE))
                    LIST_ANIMAL = [animal for _ in range(len(LIST_DATE))]
                    list_run = [RUN for _ in range(len(LIST_ANIMAL))]
                    list_expt_kind = [question for _ in range(len(LIST_ANIMAL))]
                    with Pool(MULTIPROCESS_N_CORES) as pool:
                        pool.starmap(targeted_pca_MULT_1_load_and_save, zip(LIST_ANIMAL, LIST_DATE, list_run, list_expt_kind))

            print("-------------------")
        elif plotdo==2.1:
            """ Step 2: Plot effects, and save a single dfeffects (for each animal, date)"""
            ### Collect all the animal-date pairs

            LIST_ANIMAL = []
            LIST_DATE = []
            for animal in ["Pancho", "Diego"]:
                list_dates, question, _, _ = load_preprocess_get_dates(animal, save_suffix)
                list_dates = list(set(list_dates))

                for date in list_dates:
                    LIST_ANIMAL.append(animal)
                    LIST_DATE.append(date)

            print("Getting these (animal, date) pairs")
            for a, d in zip(LIST_ANIMAL, LIST_DATE):
                print(a, d)
            
            ### Run
            OVERWRITE = False
            if True:
                from multiprocessing import Pool
                MULTIPROCESS_N_CORES = 8
                list_run = [RUN for _ in range(len(LIST_ANIMAL))]
                list_skip = [not plot_each_date for _ in range(len(LIST_ANIMAL))]
                list_overwrite = [OVERWRITE for _ in range(len(LIST_ANIMAL))]
                with Pool(MULTIPROCESS_N_CORES) as pool:
                    # pool.starmap(lambda x, y: targeted_pca_MULT_2_plot_single(x, y, run=RUN, SKIP_PLOTS=not plot_each_date, OVERWRITE=OVERWRITE), zip(LIST_ANIMAL, LIST_DATE))
                    pool.starmap(targeted_pca_MULT_2_plot_single, zip(LIST_ANIMAL, LIST_DATE, list_run, list_skip, list_overwrite))
            else:
                for animal, date in zip(LIST_ANIMAL, LIST_DATE):
                    targeted_pca_MULT_2_plot_single(animal, date, run=RUN, SKIP_PLOTS=not plot_each_date, OVERWRITE=OVERWRITE)

        elif plotdo==2.2:
            """ Step 3: Plot effects, and save a single dfeffects (for each animal, date)"""
            ### Collect all the animal-date pairs

            for animal in ["Diego", "Pancho"]:
                targeted_pca_MULT_3_combined_plots(animal, RUN, save_suffix)

        else:
            assert False
