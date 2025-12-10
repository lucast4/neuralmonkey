"""
Syntax plots, during stroke, each figure is a unit, like raster plots but just 
smooth fr using dfallpa.

NOTEBOOK: notebooks_tutorials/251111_syntax_smfr_raster_using_dfallpa.ipynb
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
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import extract_single_pa, load_handsaved_wrapper
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels
from pythonlib.tools.pandastools import append_col_with_grp_index
from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper
from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, replace_values_with_this
from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
from pythonlib.tools.plottools import savefig


def plotwrap_grammar_vs_sp(DFallpa, SAVEDIR_PLOTS):
    """
    Plots of sm fr related to effect of shape (contrasting grammar vs. SP)
    """

    from pythonlib.tools.pandastools import append_col_with_grp_index

    for bregion in DFallpa["bregion"].unique().tolist():
        pa = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=False)

        var = "shape"
        vars_others = ["task_kind", "chunk_within_rank_semantic_v2"]
        for chan in pa.Chans:
            
            print(bregion, chan)
            fig = pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var, vars_others, add_x_zero_line=True)
            savedir = f"{SAVEDIR_PLOTS}/grammar_vs_sp/color={var}-others={vars_others}"
            os.makedirs(savedir, exist_ok=True)

            savefig(fig, f"{savedir}/{chan}-{bregion}.pdf")

            plt.close("all")

def plotwrap_rank_within_chunk(DFallpa, SAVEDIR_PLOTS):
    """
    Plots of sm fr, related to effect of rank witin chunk
    """                

    for bregion in DFallpa["bregion"].unique().tolist():
        pa = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True)

        # (1) 
        LIST_VARS = [
            ("chunk_within_rank_fromlast", ["crg_shape"]),
            ("chunk_within_rank", ["crg_shape"]),
            ]
        for var, vars_others in LIST_VARS:
            savedir = f"{SAVEDIR_PLOTS}/rank_within/color={var}-others={vars_others}"
            os.makedirs(savedir, exist_ok=True)

            for chan in pa.Chans:
                print("Plotting: ", bregion, var, vars_others, chan)
                fig = pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var, vars_others, add_x_zero_line=True)
                savefig(fig, f"{savedir}/{chan}-{bregion}.pdf")
                plt.close("all")

        # (2) 
        LIST_VARS = [
            ("chunk_n_in_chunk", "crg_shape", "chunk_within_rank_fromlast"),
            ("chunk_n_in_chunk", "crg_shape", "chunk_within_rank"),
            ("chunk_within_rank_fromlast", "crg_shape", "chunk_n_in_chunk"),
            ("chunk_within_rank", "crg_shape", "chunk_n_in_chunk"),
            ]

        for var_color, var_row, var_col in LIST_VARS:
            savedir = f"{SAVEDIR_PLOTS}/rank_within/color={var_color}-row={var_row}-col={var_col}"
            os.makedirs(savedir, exist_ok=True)
            for chan in pa.Chans:
                print("Plotting: ", bregion, var_color, var_row, var_col, chan)
                
                # 1
                fig = pa.plotwrappergrid_smoothed_fr_splot_var_colored(var_row, var_col, var_color,
                                                                    chan, do_sort=True, add_x_zero_line=True);
                savefig(fig, f"{savedir}/{chan}-{bregion}.pdf")
                plt.close("all")

def _preprocess_motor_params(PA):
    """
    Quick preprocessing to get motor params, compted, and added to dflab.
    RETURNS: PA, copy of input
    """

    dflab = PA.Xlabels["trials"]
    
    ### Rmeove cases where gap to next is "none"
    inds_keep = dflab[~(dflab["gap_to_next_angle_binned"] == "none")].index.tolist()
    assert len(inds_keep)> 0.9*len(dflab), "not sure why, i thougt theses wer ejust cases where didnt do done buttong..."
    PA = PA.slice_by_dim_indices_wrapper("trials", inds_keep)

    ### Define new variables to bin the motor parameters to fewer bins
    dflab = PA.Xlabels["trials"]
    from pythonlib.tools.vectools import bin_angle_by_direction
    from math import pi
    dflab["gap_from_prev_angle_binned_v2"] = bin_angle_by_direction(dflab["gap_from_prev_angle"], -pi/4, 4, PLOT=False)
    dflab["gap_to_next_angle_binned_v2"] = bin_angle_by_direction(dflab["gap_to_next_angle"], -pi/4, 4, PLOT=False)
    # A new variable which is loc by left vs right
    dflab["gridloc_right"] = dflab["gridloc_x"] >=0
    # pd.cut(dflab["gridloc_x"], bins=3)

    from pythonlib.tools.pandastools import append_col_with_grp_index
    dflab = append_col_with_grp_index(dflab, ["gridloc_right", "gap_from_prev_angle_binned_v2"], "locright_prevangle")
    PA.Xlabels["trials"] = dflab    
    
    return PA


def plotwrap_rank_within_chunk_trialheatmaps(DFallpa, SAVEDIR_PLOTS):
    """
    Heatmaps, with panels split by one variable, and within each panel trials split by different
    variable. Is like plotwrap_rank_within_chunk but plotting trials and heatmaps.
    """
    from neuralmonkey.neuralplots.population import heatmapwrapper_many_useful_plots
    LIST_VARS = [
        ("crg_rank", "syntax_concrete"),
        ("chunk_within_rank_semantic_v2", "crg_shape"),
        ("chunk_within_rank_fromlast", "crg_n"),
        ]

    for var, var_other in LIST_VARS:
        savedir = f"{SAVEDIR_PLOTS}/rank_within_trialheatmaps/color={var}-others={var_other}"
        os.makedirs(savedir, exist_ok=True)

        for bregion in DFallpa["bregion"].unique().tolist():
            # if bregion == "preSMA":
            pa = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True)

            for chan in pa.Chans:
                pa_this = pa.slice_by_dim_values_wrapper("chans", [chan])
                
                heatmapwrapper_many_useful_plots(pa_this, savedir, var, var_conj=var_other, var_is_blocks=False,
                                                mean_over_trials=False, zlims=None, flip_rowcol=True, plot_fancy=False, n_rand_trials=40,
                                                diverge=False,
                                                plot_group_y_by_neuron=False, plot_group_y_by_var_effect=True, plot_each_neuron_no_grouping_y=False,
                                                overlay_stroke_times=True, save_suff=f"{chan}-{bregion}")  
                plt.close("all")

def plotwrap_two_shapes(DFallpa, SAVEDIR_PLOTS):
    """
    Plots of sm fr, related to effect of rank witin chunk, asking about similarity of effect
    across two different shapes.
    """                

    for bregion in DFallpa["bregion"].unique().tolist():
        pa = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True)
        pa.datamod_append_col_with_grp_index(["chunk_rank_global", "chunk_within_rank_fromlast"], "crg_ranklast")
        pa.datamod_append_col_with_grp_index(["chunk_rank_global", "chunk_n_in_chunk"], "crg_n")     

        # (1) 
        LIST_VARS = [
            ("chunk_rank_global", ["epoch"]),
            ]
        for var, vars_others in LIST_VARS:
            savedir = f"{SAVEDIR_PLOTS}/two_shapes/color={var}-others={vars_others}"
            os.makedirs(savedir, exist_ok=True)

            for chan in pa.Chans:
                print("Plotting: ", bregion, var, vars_others, chan)
                fig = pa.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var, vars_others, add_x_zero_line=True)
                savefig(fig, f"{savedir}/{chan}-{bregion}.pdf")
                plt.close("all")

        # (2) 
        LIST_VARS = [
            ("chunk_rank_global", "epoch", "chunk_within_rank_fromlast"),
            ("chunk_within_rank_fromlast", "epoch", "crg_n"),
            # ("chunk_n_in_chunk", "epoch", "crg_ranklast"),
            # ("chunk_within_rank_semantic_v2", "epoch", "chunk_rank_global"),
            ]

        for var_color, var_row, var_col in LIST_VARS:
            savedir = f"{SAVEDIR_PLOTS}/two_shapes/color={var_color}-row={var_row}-col={var_col}"
            os.makedirs(savedir, exist_ok=True)
            for chan in pa.Chans:
                print("Plotting: ", bregion, var_color, var_row, var_col, chan)
                
                # 1
                fig = pa.plotwrappergrid_smoothed_fr_splot_var_colored(var_row, var_col, var_color,
                                                                    chan, do_sort=True, add_x_zero_line=True);
                savefig(fig, f"{savedir}/{chan}-{bregion}.pdf")
                plt.close("all")



def _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True):
    """
    Extract this bregion's pa from DFallpa and do some quick
    preprocessing.
    """
    tmp = DFallpa[DFallpa["bregion"] == bregion]
    assert len(tmp)==1
    PA = tmp["pa"].values[0]

    if apply_grammar_preprocessing:
        # Prep for just analysis of grammar
        F = {"task_kind":["prims_on_grid"]}
        pa = PA.slice_by_labels_filtdict(F)
        dflab = pa.Xlabels["trials"]

        from pythonlib.dataset.dataset_analy.grammar import chunk_rank_global_extract
        dflab["date"] = "dummy"
        chunk_rank_global_extract(dflab, shape_ratio_max=1)

        # a new column
        pa.datamod_append_col_with_grp_index(["chunk_rank_global", "shape"], "crg_shape")
        pa.datamod_append_col_with_grp_index(["chunk_rank_global", "chunk_within_rank_fromlast"], "crg_ranklast")
        pa.datamod_append_col_with_grp_index(["chunk_rank_global", "chunk_within_rank"], "crg_rank")
        pa.datamod_append_col_with_grp_index(["chunk_rank_global", "chunk_n_in_chunk"], "crg_n")     

        # Prune 
        n_min = 4
        pa = pa.slice_extract_with_levels_of_var_good_prune(["epoch", "chunk_rank_global", "shape", "chunk_n_in_chunk", "chunk_within_rank_fromlast"], n_min)

    else:
        # Use this for SP vs. Grammar
        pa = PA.copy()
    
    # Prune to include just time around stroke onset
    twind = [-0.5, 0.8]
    pa = pa.slice_by_dim_values_wrapper("times", twind)
    # dflab = pa.Xlabels["trials"]
    # dflab["task_kind"].value_counts()    

    return pa

def plotwrap_rank_within_chunk_controlling_motor(DFallpa, SAVEDIR_PLOTS):
    """
    Plots related to effect of rank witin chunk, and splitting to control for motor
    variables: location, reach prev, reach next.
    """                

    LIST_VARS = [
        ("chunk_within_rank_fromlast", "gap_to_next_angle_binned_v2", "chunk_n_in_chunk"),
        ("chunk_within_rank_fromlast", "locright_prevangle", "chunk_n_in_chunk"),
        # ("chunk_within_rank_fromlast", "gridloc_right", "chunk_n_in_chunk"),
        # ("chunk_within_rank_fromlast", "locright_prevangle", "gap_to_next_angle_binned_v2"),
        ]

    savedir = f"{SAVEDIR_PLOTS}/rank_within/controlling_motor"
    os.makedirs(savedir, exist_ok=True)

    for bregion in DFallpa["bregion"].unique().tolist():
        pa = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True)
        pa = _preprocess_motor_params(pa)

        list_pa, list_grp = pa.split_by_label("trials", ["crg_shape"])

        for grp, pathis in zip(list_grp, list_pa):
            for chan in pa.Chans:
                for var_color, var_row, var_col in LIST_VARS:
                    
                    print("Plotting: ", grp, bregion, var_color, var_row, var_col, chan)
                    
                    # 1
                    fig = pathis.plotwrappergrid_smoothed_fr_splot_var_colored(var_row, var_col, var_color,
                                                                        chan, do_sort=True, add_x_zero_line=True);
                    savefig(fig, f"{savedir}/{chan}-{bregion}-data={grp}-color={var_color}-row={var_row}-col={var_col}.pdf")
                    plt.close("all")


################ SCALAR PLOTS
def preprocess_pa_scal(pa, twind, chan):
    """
    Quick helper to extract scalar PA for this channel.
    """
    pascal = pa.slice_by_dim_values_wrapper("times", twind).agg_wrapper("times") # (nchans, ntrials, 1)
    dflab = pascal.Xlabels["trials"]

    # get fr 
    dflab["fr_scal"] = pascal.X[pascal.Chans.index(chan), :, :].squeeze()

    return pascal, dflab

def plotwrap_scalar_rankwithin_controlling_motor(DFallpa, savedir):
    """
    Make plots of scalar firing rate, controlling for various motor params,
    including both catplots and heatmaps.

    """
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    twind = [-0.05, 0.2] # For taking scalar fr, relative stroke onset.

    for bregion in DFallpa["bregion"].unique().tolist():
        pa = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True)
        pa = _preprocess_motor_params(pa)

        pascal = pa.slice_by_dim_values_wrapper("times", twind).agg_wrapper("times") # (nchans, ntrials, 1)
        dflab = pascal.Xlabels["trials"]

        for chan in pascal.Chans:

            # get fr 
            dflab["fr_scal"] = pascal.X[pascal.Chans.index(chan), :, :].squeeze()

            # var_row = "gap_to_next_angle_binned"
            # var_col ="chunk_n_in_chunk"
            # var_color = "chunk_within_rank_fromlast"

            for hue in ["gap_to_next_angle_binned_v2", "locright_prevangle"]:
                fig = sns.catplot(data=dflab, x="chunk_within_rank_fromlast", y="fr_scal", hue=hue, 
                            row="crg_shape", col="chunk_n_in_chunk", kind="point", errorbar="se")

                savefig(fig, f"{savedir}/{chan}-{bregion}-hue={hue}-catplot.pdf")

                # If agging across "n in chunk" then ignore cas with 1, since it is ambiguous.
                dflab_this = dflab[dflab["chunk_n_in_chunk"]>1].reset_index(drop=True)
                fig, _ = plot_subplots_heatmap(dflab_this, hue, "chunk_within_rank_fromlast", "fr_scal", "crg_shape", False, True)
                savefig(fig, f"{savedir}/{chan}-{bregion}-hue={hue}-heatmap.pdf")
                plt.close("all")

                if False:
                    sns.catplot(data=dflab, x="gap_to_next_angle_binned", y="fr_scal", hue="chunk_within_rank_fromlast", 
                                row="crg_shape", col="chunk_n_in_chunk", kind="point", errorbar="se")            
                    
def plotwrap_scalar_rankwithin_regression(DFallpa, SAVEDIR, twind):
    """
    Compute effects, for each unit, of rank within, controlling for motor variables, using 
    scalar firing rate, using OLS regression.

    And then save all the results.
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import preprocess_dfallpa_motor_features    
    y_var = "fr_scal_sqrt"

    # Get continuous motor variables
    preprocess_dfallpa_motor_features(DFallpa, do_zscore=True)

    for bregion in DFallpa["bregion"].unique().tolist():
        PA = _extract_pa(DFallpa, bregion, apply_grammar_preprocessing=True)
        pa = _preprocess_motor_params(PA)

        for chan in PA.Chans:
            _, dflab = preprocess_pa_scal(PA, twind, chan)

            # Further preprocessing of dflab
            # - clean data
            dflab = dflab[(dflab["chunk_n_in_chunk"]>1) & ~(dflab["gap_to_next_angle_binned"] == "none")].reset_index(drop=True)
            dflab["fr_scal_sqrt"] = dflab["fr_scal"]**0.5

            # - gridloc, zscore it.
            # dflab["gridloc_x"] = dflab["gridloc_x"].astype(float)
            # dflab["gridloc_y"] = dflab["gridloc_y"].astype(float)
            # zscore gridloc x and y
            for c in ["gridloc_x", "gridloc_y"]:
                dflab[c] = (dflab[c] - dflab[c].mean())/dflab[c].std()

            dflab = append_col_with_grp_index(dflab, ["crg_shape", "chunk_within_rank_fromlast"], "crg_sh_rnklast")

            ### Regression model
            from neuralmonkey.analyses.regression_good import fit_and_score_regression_with_categorical_predictor, plot_ols_results
            from neuralmonkey.analyses.regression_good import compute_condition_estimates, plot_condition_estimates
            from pythonlib.tools.plottools import savefig


            # Version 1 -- run separately for each crg_shape. This is
            # obsolete, as it is better to pool to get the motor coefficeints
            if False: #
                for crg_shape in dflab["crg_shape"].unique().tolist():
                    print("===============", crg_shape)
                    dflab_this = dflab[(dflab["crg_shape"] == crg_shape) & (dflab["chunk_n_in_chunk"]>1) & ~(dflab["gap_to_next_angle_binned"] == "none")].reset_index(drop=True)
                    y_var = "fr_scal_sqrt"
                    x_vars = ["gap_to_next_angle_binned", "chunk_within_rank_fromlast", "gap_from_prev_angle_binned", "gridloc_right"]
                    x_vars_is_cat = [True for _ in range(len(x_vars))]
                    data_test = dflab_this
                    dict_coeff, model, original_feature_mapping, results = fit_and_score_regression_with_categorical_predictor(dflab_this, y_var, x_vars, x_vars_is_cat, data_test, 
                                                                            PRINT=True, demean_y = True)


                    # plot results of regression
                    plot_ols_results(model)
                    fig = plot_condition_estimates(model, ci=True, alpha=0.05, figsize=(20, 20))


            ###Like above, but a single model with conjunctive values for ("crg_shape", "chunk_within_rank_fromlast")
            for effect_var in ["crg_sh_rnklast", "chunk_within_rank_fromlast"]:
                savedir = f"{SAVEDIR}/{effect_var}"
                os.makedirs(savedir, exist_ok=True)

                if False:
                    # Old version, using categorical variables. It works, but probably not as good
                    x_vars_categ = [effect_var, "gap_to_next_angle_binned", "gap_from_prev_angle_binned", "gridloc_right"]
                    x_vars_cont = []
                else:
                    x_vars_categ = [effect_var]
                    x_vars_cont = ["gap_to_next_x", "gap_to_next_y", "gap_from_prev_x", "gap_from_prev_y", "gridloc_x", "gridloc_y"]
                x_vars = x_vars_categ + x_vars_cont
                x_vars_is_cat = [True for _ in range(len(x_vars_categ))] + [False for _ in range(len(x_vars_cont))]
                dict_coeff, model, original_feature_mapping, results = fit_and_score_regression_with_categorical_predictor(dflab, 
                                                                        y_var, x_vars, x_vars_is_cat, dflab, 
                                                                        PRINT=False, demean_y = True)
                # plot results of regression
                fig = plot_ols_results(model)
                savefig(fig, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-coeffs.pdf")
                fig = plot_condition_estimates(model, ci=True, alpha=0.05, figsize=(8, 10))
                savefig(fig, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-estimates.pdf")

                plt.close("all")

                # Save things
                DF_CONJ, DF_MARG, cat_vars = compute_condition_estimates(model)

                pd.to_pickle(DF_CONJ, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-DF_CONJ.pkl")
                pd.to_pickle(DF_MARG, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-DF_MARG.pkl")
                pd.to_pickle(cat_vars, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-cat_vars.pkl")
                pd.to_pickle(dict_coeff, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-dict_coeff.pkl")
                pd.to_pickle(model, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-model.pkl")
                pd.to_pickle(original_feature_mapping, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-original_feature_mapping.pkl")
                pd.to_pickle(results, f"{savedir}/{chan}-{bregion}-effectvar={effect_var}-results.pkl")

def _expand_crg_sh_rnklast_levels(df, col_level="level", suffix=""):
    """
    Helper to expand from string level values to two new columns:
    chunk_within_rank_fromlast and crg_shape
    """
    from pythonlib.tools.stringtools import decompose_string
    # Get chunk_within_rank_fromlast
    def f(lev):
        tmp = decompose_string(lev, "|")
        assert len(tmp)==3
        chunk_within_rank_fromlast = int(tmp[2])
        return chunk_within_rank_fromlast
    df[f"chunk_within_rank_fromlast{suffix}"] = df[col_level].map(f)

    # Get crg_shape
    def f(lev):
        tmp = decompose_string(lev, "|")
        assert len(tmp)==3
        crg_shape = "|".join(tmp[:2])
        return crg_shape
    df[f"crg_shape{suffix}"] = df[col_level].map(f)    

def plotwrap_scalar_rankwithin_regression_MULT(animal, date, var_effect = "crg_sh_rnklast"):
    """
    Collect and plot results from plotwrap_scalar_rankwithin_controlling_motor().
    """
    import pickle
    import os
    from glob import glob
    from neuralmonkey.analyses.regression_good import all_pairwise_balanced_marginal_tests
    from statsmodels.stats.multitest import multipletests
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping, plot_45scatter_means_flexible_grouping_from_wideform
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

    assert var_effect == "crg_sh_rnklast", "hard coded fro this below. easy to change."

    LOADDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/regression_scalar_control_motor/{animal}-{date}/{var_effect}/1-M1-effectvar={var_effect}-DF_MARG.pdf"
    SAVEDIR_PLOTS = f"/lemur2/lucas/analyses/recordings/main/syntax_good/regression_scalar_control_motor/{animal}-{date}/MULT/{var_effect}"
    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

    # Get list of files
    globstr = f"/lemur2/lucas/analyses/recordings/main/syntax_good/regression_scalar_control_motor/{animal}-{date}/{var_effect}/*-*-effectvar={var_effect}-DF_MARG.pkl"
    list_path = glob(globstr)

    # Load for each chan
    list_pairs = None
    list_df = []
    list_results = []
    for path in list_path:

        ######### Load marginal values.
        print(path)
        df = pd.read_pickle(path)

        from pythonlib.tools.expttools import deconstruct_filename
        info = deconstruct_filename(path)
        chan = int(info["filename_components_hyphened"][0])
        bregion =info["filename_components_hyphened"][1]    

        df["chan"] = chan
        df["bregion"] = bregion

        list_df.append(df)

        ######### Also load model in order to get all pairwise comparisons between marginal levels.
        path_model = info["basedirs"][1] + "/" + "-".join(info["filename_components_hyphened"][:-1]) + "-model.pkl"
        with open(path_model, "rb") as f:
            model = pickle.load(f)  

        # Get all pairwise com
        if list_pairs is not None:
            _expand_crg_sh_rnklast_levels(df)
            list_pairs =[]
            for i, row1 in df.iterrows():
                for j, row2 in df.iterrows():
                    if row1["label"]==row2["label"]:
                        continue
                    if row1["crg_shape"] == row2["crg_shape"]:
                        list_pairs.append((row1["level"], row2["level"]))
            list_pairs

        results = all_pairwise_balanced_marginal_tests(model, "crg_sh_rnklast", do_mult_comparisons=False,
            list_pairs=list_pairs)
        
        # First, expand the levels
        _expand_crg_sh_rnklast_levels(results, "level1", "_1")
        _expand_crg_sh_rnklast_levels(results, "level2", "_2")

        # Only include comparisons with same crg_shape
        results = results[results["crg_shape_1"] == results["crg_shape_2"]].reset_index(drop=True)
        results["crg_shape"] = results["crg_shape_1"]

        # Compute multuple comparisons within each crg_shape
        _list_df = []
        for grp in results.groupby("crg_shape"):
            _level = grp[0]
            _df = grp[1].copy()
            
            reject, p_corr, _, _ = multipletests(
                _df["p_uncorrected"].values)
            _df["p_adjusted"] = p_corr
        
            _list_df.append(_df)
        results = pd.concat(_list_df).reset_index(drop=True)

        # Store the reuslts
        results["chan"] = chan
        results["bregion"] = bregion
        list_results.append(results)

    ##### Analysis 1, using the marginal estimates
    savedir = f"{SAVEDIR_PLOTS}/1_marginal_estimates"
    os.makedirs(savedir, exist_ok=True)
    DF_EST = pd.concat(list_df).reset_index(drop=True)

    # Preprocess
    _expand_crg_sh_rnklast_levels(DF_EST)
    
    # Line plots
    list_bregion = DF_EST["bregion"].unique().tolist()
    list_crg_shape = DF_EST["crg_shape"].unique().tolist()
    SIZE = 5
    nrows = len(list_crg_shape)
    ncols = len(list_bregion)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
    for i, bregion in enumerate(list_bregion):
        for j, crg_shape in enumerate(list_crg_shape):
            
            ax = axes[j][i]

            df = DF_EST[(DF_EST["bregion"] == bregion) & (DF_EST["crg_shape"] == crg_shape)].reset_index(drop=True)
            
            sns.lineplot(df, x="chunk_within_rank_fromlast", y="estimate", hue="chan", legend=None, ax=ax)
            
            ax.set_title(f"{crg_shape}-{bregion}")
    savefig(fig, f"{savedir}/alldata-lineplot-1.pdf")

    # Scatterplots
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
    grpdict = grouping_append_and_return_inner_items(DF_EST, ["chan", "bregion", "crg_shape"])
    res = []

    for grp, inds in grpdict.items():
        df = DF_EST.iloc[inds]

        # get successive differences as you count down chunk_within_rank_fromlast
        list_ranks = sorted(df["chunk_within_rank_fromlast"].unique().tolist())
        # list(range(df["chunk_within_rank_fromlast"].min(), df["chunk_within_rank_fromlast"].max()+1))

        # get mean fr for this unit
        mean_estimate = df["estimate"].mean()

        for rank1, rank2 in zip(list_ranks[:-1], list_ranks[1:]):
            assert len(df[df["chunk_within_rank_fromlast"]==rank2]["estimate"])==1
            estimate_diff = df[df["chunk_within_rank_fromlast"]==rank2]["estimate"].mean() - df[df["chunk_within_rank_fromlast"]==rank1]["estimate"].mean()

            res.append({
                "chan":grp[0], 
                "bregion":grp[1], 
                "crg_shape":grp[2],
                "estimate_diff":estimate_diff,
                "mean_estimate":mean_estimate,
                "estimate_diff_div_mean":estimate_diff/mean_estimate,
                "ranks":(rank1, rank2),
                "rank1":rank1,
                "rank2":rank2,
            })
    DF_EST_DIFF = pd.DataFrame(res)

    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    for crg_shape in DF_EST_DIFF["crg_shape"].unique():
        print(crg_shape)
        df = DF_EST_DIFF[DF_EST_DIFF["crg_shape"]==crg_shape].reset_index(drop=True)
        _, fig = plot_45scatter_means_flexible_grouping(df, "ranks", (-3, -2), (-2, -1), "bregion", "estimate_diff", 
                                            "chan", True, shareaxes=True, plot_error_bars=False, alpha=0.15,
                                            SIZE=5);

        savefig(fig, f"{savedir}/scatter-crg_shape={crg_shape}.pdf")

    ##### Analysis 2, plotting n significant cases
    # Using t-test -- Pull out exmaples where just the endpoint is significantly different from each of the items
    savedir = f"{SAVEDIR_PLOTS}/2_stats_ttest_pvals"
    os.makedirs(savedir, exist_ok=True)

    DF_STATS = pd.concat(list_results).reset_index(drop=True)

    # For each (chan, crg_shape) get two values: 
    res = []
    for x in DF_STATS.groupby(["chan", "bregion", "crg_shape"]):
        df = x[1]

        # -- least sig p value from (-1) to all others
        p_max_last_vs_others = np.max(df[(df["chunk_within_rank_fromlast_1"]==-1) | (df["chunk_within_rank_fromlast_2"]==-1)]["p_adjusted"])

        # -- most significant p value between all pairs of those that are not -1
        p_min_others_vs_others = np.min(df[(df["chunk_within_rank_fromlast_1"]!=-1) & (df["chunk_within_rank_fromlast_2"]!=-1)]["p_adjusted"])

        res.append({
            "p_max_last_vs_others":p_max_last_vs_others,
            "p_min_others_vs_others":p_min_others_vs_others,
            "chan":x[0][0],
            "bregion":x[0][1],
            "crg_shape":x[0][2],
        })
    DF_STATS_COMPARISONS = pd.DataFrame(res)

    # Postprocess
    alpha = 0.05
    DF_STATS_COMPARISONS["sig_last_vs_others"] = DF_STATS_COMPARISONS["p_max_last_vs_others"] < alpha
    DF_STATS_COMPARISONS["sig_others_vs_others"] = DF_STATS_COMPARISONS["p_min_others_vs_others"] < alpha
    DF_STATS_COMPARISONS["logp_max_last_vs_others"] = np.log10(DF_STATS_COMPARISONS["p_max_last_vs_others"])
    DF_STATS_COMPARISONS["logp_min_others_vs_others"] = np.log10(DF_STATS_COMPARISONS["p_min_others_vs_others"])
    DF_STATS_COMPARISONS["sig_final"] = (DF_STATS_COMPARISONS["sig_last_vs_others"]==True) & (DF_STATS_COMPARISONS["sig_others_vs_others"]==False)


    # Plot final counts
    fig = grouping_plot_n_samples_conjunction_heatmap(DF_STATS_COMPARISONS, "sig_final", "bregion", ["crg_shape"], norm_method="div_all")
    savefig(fig, f"{savedir}/counts_heatmap-all.pdf")

    # Plot scatter of p-values.
    for x in DF_STATS_COMPARISONS.groupby("crg_shape"):
        grp = x[0]
        print(grp)
        df = x[1].reset_index(drop=True)
        map_datapt_lev_to_colorlev = {row["chan"]:row["sig_final"] for _, row in df.iterrows()}
        _, fig = plot_45scatter_means_flexible_grouping_from_wideform(df, "p_max_last_vs_others", 
                                                            "p_min_others_vs_others", "bregion", "chan", 
                                                            True, 0.4, 6, True, False,
                                                            map_datapt_lev_to_colorlev=map_datapt_lev_to_colorlev)
        savefig(fig, f"{savedir}/scatter-crg_shape={grp}.pdf")

        for norm_meth in [None, "div_all"]:
            fig = grouping_plot_n_samples_conjunction_heatmap(df, "sig_last_vs_others", "sig_others_vs_others", ["bregion"], norm_method=norm_meth);
            savefig(fig, f"{savedir}/counts_heatmap-crg_shape={grp}-norm={norm_meth}.pdf")
    plt.close("all")    
    

if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
    from pythonlib.tools.pandastools import append_col_with_grp_index
    import seaborn as sns
    import os
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single, euclidian_distance_compute_trajectories
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_concatbregion_preprocess_wrapper, dfpa_concat_bregion_to_combined_bregion
    from neuralmonkey.classes.population_mult import dfpa_concat_merge_pa_along_trials


    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good"

    animal = sys.argv[1]
    date = int(sys.argv[2])
    # question = sys.argv[3]
    # run_number = int(sys.argv[4])
    PLOTS_DO = [2.3, 2.2, 2.1]
    PLOTS_DO = [2.3]

    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_concat_merge_pa_along_trials, dfpa_concatbregion_preprocess_wrapper, dfpa_concat_bregion_to_combined_bregion
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_concat_bregion_to_combined_bregion
    from pythonlib.tools.exceptions import NotEnoughDataException

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good"

    version = "stroke"
    combine = False

    ### (1) load Grammar Dfallpa
    question = "RULE_ANBMCK_STROKE"
    DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                        question=question)
    DFallpa = dfpa_concat_bregion_to_combined_bregion(DFallpa)

    if False:
        try:
            ### (2) Load SP data
            _question = "SP_BASE_stroke"
            _twind = [-0.5, 2.1]
            DFallpaSP = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                                question=_question, twind=_twind)
            DFallpaSP = dfpa_concat_bregion_to_combined_bregion(DFallpaSP)

            # Merge SP and grammar along chan indices
            DFallpa = dfpa_concat_merge_pa_along_trials(DFallpa, DFallpaSP)
            del DFallpaSP
        except NotEnoughDataException as err:
            # pass
            # Then this is because chans dont line up. Not good.
            raise err
        except Exception as err:
            pass

    # Make a copy of all PA before normalization
    fr_mean_subtract_method = "raw_fr"
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date, fr_mean_subtract_method=fr_mean_subtract_method)

    ########### SAVEDIR
    from pythonlib.tools.plottools import savefig
    SAVEDIR_PLOTS = f"{SAVEDIR}/smfr_each_unit/{animal}-{date}"
    import os
    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)
    
    ################################### PLOTS
    for plot_do in PLOTS_DO:

        ################# MAIN SM FR PLOTS
        if plot_do == 1.1:
            ### Two shapes
            # Only run this if the need exists
            from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
            list_dates, _, _, _ = load_preprocess_get_dates(animal, "two_shape_sets")
            if date in list_dates:
                plotwrap_two_shapes(DFallpa, SAVEDIR_PLOTS)   
        elif plot_do == 1.2:
            ### Grammar vs. SP
            plotwrap_grammar_vs_sp(DFallpa, SAVEDIR_PLOTS)
        elif plot_do == 1.3:
            ### Rank within chunk
            plotwrap_rank_within_chunk(DFallpa, SAVEDIR_PLOTS)
        elif plot_do == 1.4:
            ### Trial heatmaps
            plotwrap_rank_within_chunk_trialheatmaps(DFallpa, SAVEDIR_PLOTS)

        ################ BOUNDARY NEURONS
        elif plot_do == 2.1:
            ### Smoothed fr, rank_within, controlling for motor.
            # (1) Define new variables to bin the motor parameters to fewer bins
            SAVEDIR_PLOTS = f"{SAVEDIR}/smfr_each_unit_control_motor/{animal}-{date}"
            plotwrap_rank_within_chunk_controlling_motor(DFallpa, SAVEDIR_PLOTS)       
        elif plot_do == 2.2:
            ### Scalar fr, rank within, controlling for motor.
            SAVEDIR_PLOTS = f"{SAVEDIR}/scalar_each_unit_control_motor/{animal}-{date}"
            os.makedirs(SAVEDIR_PLOTS, exist_ok=True)
            plotwrap_scalar_rankwithin_controlling_motor(DFallpa, SAVEDIR_PLOTS)
        elif plot_do == 2.3:
            ### Regression of scalar fr, rank within, controlling for motor effects.
            # --> For MULT analy, see notebook: notebooks_tutorials/251111_syntax_smfr_raster_using_dfallpa.ipynb
            # Section: ### [MULT] Load regression results and plot
            from neuralmonkey.scripts.analy_syntax_good_smfr_units import plotwrap_scalar_rankwithin_regression
            SAVEDIR_PLOTS = f"{SAVEDIR}/regression_scalar_control_motor/{animal}-{date}"
            os.makedirs(SAVEDIR_PLOTS, exist_ok=True)
            # twind = [-0.05, 0.2]
            twind = [-0.1, 0.1]
            plotwrap_scalar_rankwithin_regression(DFallpa, SAVEDIR_PLOTS, twind)     
        ################ BOUNDARY NEURONS
        else:
            assert False