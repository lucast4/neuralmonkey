"""

Organizing good plots for syntax that is related to errors.
Main unique thing is that uses ALLDATA rather than "correct trials only"

NOTEBOOK: notebooks_tutorials/251111_syntax_errors_smfr_pop.ipynb

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

# from neuralmonkey.classes.session import _REGIONS_IN_ORDER, _REGIONS_IN_ORDER_COMBINED
# ORDER_BREGION = _REGIONS_IN_ORDER_COMBINED

# TWIND_ANALY = (-0.6, 1.0) # This is just for windowing final data, not for fitting pca.
# NPCS_KEEP = 8

# N_MIN_TRIALS = 4 # min trials per level, otherwise throws level out.
# NPCS_KEEP = 8

# # PLOT_STATE_SPACE
# # Just as suffixes for loading euclidian_time_resolved. Hacky. Use the twinds from SUBSPACE_PROJ_FIT_TWIND above.
# LIST_TWIND_POSSIBLE = [
#     TWIND_ANALY,
#     (0.05, 0.9),
#     (0.05, 0.6),
#     (-0.3, 0.3),
#     (0.3, 0.3),
#     (-0.5, -0.05), 
#     (0.05, 0.5)
#     ]

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

def plotwrapper_smfr_units(DFallpa, dffuture, SAVEDIR):
    """
    Smoothed firing rates for each unit, colored by its error class
    """
    ### Plot smoothed firing rates, asking about effect of error label
    from neuralmonkey.scripts.analy_syntax_errors_smfr_pop import preprocess_pa_syntax_errors
    from neuralmonkey.scripts.analy_syntax_good_smfr_units import _extract_pa
    import os

    list_bregion = DFallpa["bregion"].unique().tolist()

    for bregion in list_bregion:
        # bregion = "vlPFC"
        pa = _extract_pa(DFallpa, bregion, True)

        # Add useful columns related to errors and next stroke properties.
        preprocess_pa_syntax_errors(pa, dffuture)

        ### Prune data, to lower the number of plots
        dflab = pa.Xlabels["trials"]
        # Keep only those chunks that are not the last chunk (since I just care about errors that go to the wrong chunk)
        dflab = dflab[dflab["Cr_NextChunk_Corr"] < 99]
        # Remove failed strokes
        dflab = dflab[~(dflab["stroke_error_label"].isin(["stk_gram_gdchk_bdrnk", "stk_gram_bdchk", "stk_qual"]))]
        # Do pruning
        pa = pa.slice_by_dim_indices_wrapper("trials", dflab.index.tolist())

        if False:
            # Slice to just important cases
            dflab = pa.Xlabels["trials"]

            # - remove if this is last chunk
            dflab = dflab[~(dflab["Cr_NextStrk_Corr"]=="none")]

            # - remove if stroke is failure
            dflab = dflab[dflab["Success_ThisStrk_Draw"]==1]
            dflab = dflab[dflab["AbortStrokeQuality_ThisStrk_Draw"]==0]

            idxs = dflab.index.tolist()
            pathis = pa.slice_by_dim_indices_wrapper("trials", idxs)
            dflab = pathis.Xlabels["trials"]

        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        if False:
            vars_group = ["chunk_rank", "shape", "Cr_NextChunk_Corr", "chunk_n_in_chunk"]
            # var2 = "Success_NextStrk_Draw"
            var2 = "stroke_error_class"
            var1 = "chunk_within_rank"
            grouping_plot_n_samples_conjunction_heatmap(dflab, var1, var2, vars_group);
            # Different plot for each vars_grp

        ### Plot
        from pythonlib.tools.plottools import savefig
        # var2 = "Success_NextStrk_Draw"
        # var2 = "stroke_error_class"
        var2 = "stroke_error_label"
        var1 = "chunk_within_rank"

        dflab = pa.Xlabels["trials"]
        for vars_grp in [
            ["chunk_rank", "shape", "chunk_n_in_chunk", "Cr_NextChunk_Corr", "gridloc"],
            ["chunk_rank", "shape", "chunk_n_in_chunk", "Cr_NextChunk_Corr"],
            ]:

            savedir = f"{SAVEDIR}/varsgrp={vars_grp}"
            os.makedirs(savedir, exist_ok=True)
            # os.makedirs("/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS/sm_fr_units/ERRORS3", exist_ok=True)
            # os.makedirs("/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS/sm_fr_units/ERRORS3", exist_ok=True)

            # list_pa, list_grp = pa.split_by_label("trials", vars_grp)
            plot_counts_heatmap_savepath = f"{savedir}/counts.pdf"
            # plot_counts_heatmap_savepath = f"/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS/sm_fr_units/ERRORS3/counts.pdf"
            dict_pa = pa.slice_extract_with_levels_of_conjunction_vars_as_dictpa(var2, vars_grp, 
                                                                                prune_min_n_trials=3, 
                                                                                prune_min_n_levs=2,
                                                                                plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)

            for chan in pa.Chans:
                _savedir = f"{savedir}/chan={chan}-{bregion}"
                # _savedir = f"/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS/sm_fr_units/ERRORS3/chan={chan}-{bregion}"
                os.makedirs(_savedir, exist_ok=True)
                print(_savedir)

                # for pasub, grp in zip(list_pa, list_grp):
                for grp, pasub in dict_pa.items():
                    
                    print(grp)
                    fig = pasub.plotwrapper_smoothed_fr_split_by_label_and_subplots(chan, var2, [var1])
                    savefig(fig, f"{_savedir}/{grp}.pdf")
                    plt.close("all")



def plotwrapper_pop_state_eurlidean(DFallpa, dffuture, SAVEDIR_ALL, DO_PLOT_STATESPACE = True, DO_EUCLIDEAN = True):
    """
    Population plots and analyses.
    - State space, split/colored by error status.
    - Euclidean, asking whether error is different from correct.

    """

    # Population plots/
    # -- For each chunk_rank, shape --> targeted PCA to get axis of progression.
    # -- Then for each (location, make plots)

    from neuralmonkey.scripts.analy_syntax_good_eucl_state import preprocess_pa
    from neuralmonkey.scripts.analy_syntax_good_smfr_units import _extract_pa
    import os
    

    # - For initial global regression and subtraction of confounding variables
    do_remove_global_first_stroke = True
    variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
    vars_remove_global = ["stroke_index_is_first"]
    # - Then for subspace identification
    # (note: Same as above, but remove the variable that has been regressed out)
    variables_cont = []
    variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
    do_vars_remove = False
    vars_remove = None
    # # Subspace params
    # list_var_subspace = [
    #     tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
    #     # "shape", # Only run this for the question related to SP vs. grammar.
    #     ]
    # restrict_questions_based_on_subspace = {
    #     "shape":["4_shape_vs_chunk"],
    # }

    # - Update the time window to match the action sybmols stuff
    twind_scal = [-0.35, 0.2]
    tbin_dur = 0.15
    tbin_slide = 0.05

    npcs_keep_force = 50
    prune_min_n_trials = 3  

    ### Preprocessing
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import preprocess_dfallpa_motor_features
    preprocess_dfallpa_motor_features(DFallpa, plot_motor_values=False, do_zscore=True)

    ### Iterate all data
    list_bregion = DFallpa["bregion"].unique().tolist()
    list_dfdist = []
    for bregion in list_bregion:
        # if bregion not in ["PMv", "preSMA"]:
        #     continue
        PA = _extract_pa(DFallpa, bregion)

        # Add useful columns related to errors and next stroke properties.
        preprocess_pa_syntax_errors(PA, dffuture)
    
        SAVEDIR = f"{SAVEDIR_ALL}/bregion={bregion}"
        os.makedirs(SAVEDIR, exist_ok=True)

        filtdict = {}
        # use these regardless of what subspace, as they are good for pruning
        prune_min_n_levs = None
        _var_effect = "chunk_within_rank"
        _vars_others = ["epoch", "chunk_shape", "syntax_concrete", "task_kind"]
        PA = preprocess_pa(PA, _var_effect, _vars_others, prune_min_n_trials, prune_min_n_levs, filtdict,
                    SAVEDIR, 
                    None, None, None, None, None, 
                    skip_dimredu=True, prune_by_conj_var=False)

        # Do dim reduction of PA up here, to be able to skip it below (quicker)
        savedir_pca = f"{SAVEDIR}/pca"
        os.makedirs(savedir_pca, exist_ok=True)
        _, PAscal = PA.dataextract_dimred_wrapper("scal", "pca", savedir_pca, twind_scal, tbin_dur, tbin_slide,
                                    npcs_keep_force)            

        ### First, any controls you want to do, subtracting out activity after regression. Do it on all trials first.
        if do_remove_global_first_stroke:
            _savedir = f"{SAVEDIR}/remove_effect_first_stroke-coeff"
            os.makedirs(_savedir, exist_ok=True)
            _, _, _, _, _, PAscal, _ = PAscal.dataextract_subspace_targeted_pca_wrapper(
                                                        variables_cont_global, variables_cat_global, vars_remove_global,
                                                        None, None, None,
                                                        True, False,
                                                        savedir_coeff_heatmap=_savedir,
                                                        savedir_pca_subspaces=_savedir)

        from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single_one_var_mult_axes

        if DO_PLOT_STATESPACE: # already done. Just unFalse it to run

            ############## (1) Encoding of rank within
            var_subspace = ["rank_conj"]

            if False:
                # This converted to None later.
                LIST_VAR_VAROTHERS = [
                    ## N in chunk
                    # ('rankwithin_error', ['task_kind', 'epoch', 'chunk_shape', 'chunk_n_in_chunk', 'Cr_NextChunk_Corr']),
                    ('stroke_error_label', ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank'])
                ]
            LIST_DIMS = [(0,1), (2,3), (4,5)]
            # var_color = "stroke_error_label"
            # var_subplots = ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank']

            dflab = PAscal.Xlabels["trials"]

            # Iterate over different training datasets, each a differnet chunk_shape
            for chunk_shape_fit in dflab["chunk_shape"].unique().tolist():
                
                # preprocess_pa_syntax_errors(PAscal, dffuture)
                # Only use correct trials to get subspace
                inds_pa_train_1 = dflab[dflab["trial_sequence_error_string"] =="allgreat"].index.tolist()
                inds_pa_train_2 = dflab[dflab["chunk_shape"] == chunk_shape_fit].index.tolist()
                inds_pa_train = [ind for ind in inds_pa_train_1 if ind in inds_pa_train_2]

                assert all(dflab["task_kind"] == "prims_on_grid"), "simpel solutiom, take PIG only"
                inds_pa_test = dflab.index.tolist() # all
                
                ### Get PA in this subspace
                savedir = f"{SAVEDIR}/SS/FITTING_subspc={var_subspace}-chunk_shape={chunk_shape_fit}"
                os.makedirs(savedir, exist_ok=True)
                LIST_VAR_VAROTHERS = None # Will  make plots later, split into diff figures.
                pa_subspace, _, _, _, _ = state_space_targeted_pca_scalar_single_one_var_mult_axes(
                        PAscal, twind_scal, variables_cont, variables_cat, var_subspace, npcs_keep_force, 
                        LIST_VAR_VAROTHERS, LIST_DIMS, savedir, just_extract_paredu=False,
                        savedir_pca_subspaces=savedir,
                        inds_trials_pa_train=inds_pa_train, inds_trials_pa_test=inds_pa_test,
                        skip_dim_redu=True,
                        do_vars_remove=do_vars_remove, vars_remove=vars_remove)

                if pa_subspace is not None:
                    ### Plot state space
                    vars_grp = ["chunk_rank", "shape"] # Each value is a figure.
                    LIST_VAR = [
                        "stroke_error_label",
                        "stroke_error_label",    
                    ]
                    LIST_VARS_OTHERS = [
                        ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank'],
                        ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank', 'gridloc'],
                    ]
                    list_pa, list_grp = pa_subspace.split_by_label("trials", vars_grp)
                    for grp, pa_this in zip(list_grp, list_pa):
                        _savedir = f"{savedir}/vars_figure={vars_grp}={grp}"
                        os.makedirs(_savedir, exist_ok=True)
                        pa_this.plot_state_space_good_wrapper(_savedir, LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT=None, LIST_PRUNE_MIN_N_LEVS=None,
                                                            nmin_trials_per_lev=None, list_dim_timecourse=None, list_dims=LIST_DIMS,
                                                            also_plot_heatmaps=False)
                
            ############ (2) Encoding of shape
            # Is state more like the next shape?
            var_subspace = ["shape"]

            # Only use correct trials to get subspace
            inds_pa_train = dflab[dflab["trial_sequence_error_string"] =="allgreat"].index.tolist()
            inds_pa_test = dflab.index.tolist() # all

            ### Get PA in this subspace
            savedir = f"{SAVEDIR}/SS/FITTING_subspc={var_subspace}-ALL"
            os.makedirs(savedir, exist_ok=True)
            LIST_VAR_VAROTHERS = None # Will  make plots later, split into diff figures.
            pa_subspace, _, _, _, _ = state_space_targeted_pca_scalar_single_one_var_mult_axes(
                    PAscal, twind_scal, variables_cont, variables_cat, var_subspace, npcs_keep_force, 
                    LIST_VAR_VAROTHERS, LIST_DIMS, savedir, just_extract_paredu=False,
                    savedir_pca_subspaces=savedir,
                    inds_trials_pa_train=inds_pa_train, inds_trials_pa_test=inds_pa_test,
                    skip_dim_redu=True,
                    do_vars_remove=do_vars_remove, vars_remove=vars_remove)

            ### Plot state space
            LIST_VAR = [
                "stroke_error_label",
                "stroke_error_label",
                "stroke_error_label",    
            ]
            LIST_VARS_OTHERS = [
                ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk'],
                ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank'],
                ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank', 'gridloc'],
            ]
            _savedir = f"{savedir}/vars_figure=ALL"
            os.makedirs(_savedir, exist_ok=True)
            pa_subspace.plot_state_space_good_wrapper(_savedir, LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT=None, LIST_PRUNE_MIN_N_LEVS=None,
                                                nmin_trials_per_lev=None, list_dim_timecourse=None, list_dims=LIST_DIMS,
                                                also_plot_heatmaps=False)

        if DO_EUCLIDEAN:
            ### Simple, get euclidean distance between correct and error.
            question = "error"

            ################################################
            var_subspace = tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]) # global
            # var_subspace = "rank_conj"
            dflab = PAscal.Xlabels["trials"]
            
            # preprocess_pa_syntax_errors(PAscal, dffuture)
            # Only use correct trials to get subspace
            inds_pa_train = dflab[dflab["trial_sequence_error_string"] =="allgreat"].index.tolist()
            assert all(dflab["task_kind"] == "prims_on_grid")
            inds_pa_test = dflab.index.tolist() # all
            
            ### Get PA in this subspace
            savedir = f"{SAVEDIR}/EUCLID/FITTING_subspc={var_subspace}"
            os.makedirs(savedir, exist_ok=True)
            LIST_VAR_VAROTHERS = None # Will  make plots later, split into diff figures.
            pa_subspace, _, _, _, _ = state_space_targeted_pca_scalar_single_one_var_mult_axes(
                    PAscal, twind_scal, variables_cont, variables_cat, var_subspace, npcs_keep_force, 
                    None, None, savedir, just_extract_paredu=False,
                    savedir_pca_subspaces=savedir,
                    inds_trials_pa_train=inds_pa_train, inds_trials_pa_test=inds_pa_test,
                    skip_dim_redu=True,
                    do_vars_remove=do_vars_remove, vars_remove=vars_remove)


            ### Compute euclidian distnace
            # Restrict to just the contrasts you care about
            # array(['allsuccess', 'stk_gram_gdchk_bdrnk', 'nxt_gram_chk2Early',
            #        'stk_gram_bdchk', 'nxt_gram_rnk', 'stk_qual',
            #        'nxt_gram_chkSkipChk'], dtype=object)
            error_labels_keep = ["allsuccess", "nxt_gram_chk2Early"]
            vars_grp = ['task_kind', 'epoch', 'chunk_rank', 'shape', 'Cr_NextChunk_Corr', 'chunk_n_in_chunk', 'chunk_within_rank', 'gridloc']
            # Keep just conditions which have all of these labels.
            pa_subspace_this, _, _ = pa_subspace.slice_extract_with_levels_of_conjunction_vars("stroke_error_label", vars_grp, 
                                                                                    3, levels_var=error_labels_keep)

            if pa_subspace_this is not None:
                do_plot_rsa = True
                var_effect = "stroke_error_label"
                euclidean_label_vars = vars_grp + [var_effect]

                from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar, dfdist_summary_plots_wrapper, dfdist_postprocess_wrapper
                # Compute distance
                if do_plot_rsa:
                    rsa_heatmap_savedir = savedir
                else:
                    rsa_heatmap_savedir = None
                dfdist, _ = timevarying_compute_fast_to_scalar(pa_subspace_this, label_vars=euclidean_label_vars, 
                                                        rsa_heatmap_savedir=rsa_heatmap_savedir, plot_conjunctions_savedir=_savedir)

                if len(dfdist)>0:
                    # save it
                    dfdist["var_subspace"] = [var_subspace for _ in range(len(dfdist))]
                    dfdist["question"] = question
                    dfdist["bregion"] = bregion
                    dfdist["euclidean_label_vars"] = [tuple(euclidean_label_vars) for _ in range(len(dfdist))]
                    list_dfdist.append(dfdist)

                dfdist.to_pickle(f"{savedir}/dfdist.pkl")

    
    ### QUick plot of DFDIST
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_effect_extract_helper, dfdist_variables_generate_var_same
    import seaborn as sns
    savedir = f"{SAVEDIR_ALL}/MULT/EUCLID/plots"
    os.makedirs(savedir, exist_ok=True)

    if len(list_dfdist)>0:
        DFDIST = pd.concat(list_dfdist).reset_index(drop=True)        
        assert len(DFDIST["euclidean_label_vars"].unique())==1
        euclidean_label_vars = DFDIST["euclidean_label_vars"].unique()[0]
        colname_conj_same = dfdist_variables_generate_var_same(euclidean_label_vars)

        # Get effects of interest
        contrasts_diff = ["stroke_error_label"]
        contrasts_either = []
        DFDIST["contrast_string"] = DFDIST[colname_conj_same]
        dfeffect = dfdist_variables_effect_extract_helper(DFDIST, colname_conj_same, euclidean_label_vars, contrasts_diff, contrasts_either)

        # Plot
        fig = sns.catplot(data=DFDIST, x="contrast_string", y="dist_yue_diff", hue="bregion", kind="bar", errorbar="se", aspect=1.5)
        from pythonlib.tools.snstools import rotateLabel
        rotateLabel(fig)
        savefig(fig, f"{savedir}/catplot-1.pdf")

        fig = sns.catplot(data=DFDIST, x="bregion", y="dist_yue_diff", hue="contrast_string", kind="bar", errorbar="se")
        savefig(fig, f"{savedir}/catplot-2.pdf")

        fig = sns.catplot(data=dfeffect, x="bregion", y="dist_yue_diff", hue="contrast_string")
        savefig(fig, f"{savedir}/catplot-3.pdf")

        fig = sns.catplot(data=dfeffect, x="bregion", y="dist_yue_diff", hue="contrast_string", kind="bar", errorbar="se")
        savefig(fig, f"{savedir}/catplot-4.pdf")

        plt.close("all")
    else:
        DFDIST = None
        
    return DFDIST

def preprocess_pa_syntax_errors(PA, dffuture):
    """
    Helper to preprocess PA, mainly getting varialbes related to the error status of the current stroke as well
    as of the upcoming stroke.

    PARAMS:
    - dffuture, data extracted from:
        # Get future strokes chunks info.
        from pythonlib.dataset.dataset_analy.grammar import syntaxconcrete_extract_wrapper_chunks_future_errors_info
        dffuture = syntaxconcrete_extract_wrapper_chunks_future_errors_info(D)

    RETURNS:
    - (nothing) modifies PA.Xlabels["trials"]
    """

    dflab = PA.Xlabels["trials"]

    ### Merge dffuture and dflab
    # Define a new index column
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dflab = append_col_with_grp_index(dflab, ["trialcode", "stroke_index"], "tc_si")
    dffuture = append_col_with_grp_index(dffuture, ["trialcode", "ind_stroke"], "tc_si")
    
    # Slice out dffuture to match dflab
    from pythonlib.tools.pandastools import slice_by_row_label
    dffuture_sub = slice_by_row_label(dffuture, "tc_si", dflab["tc_si"].tolist(), assert_exactly_one_each=True)

    # Sanity check the dataframes match
    for col in ["shape", "gridloc", "chunk_rank", "chunk_within_rank", "chunk_within_rank_fromlast"]:
        assert all(dffuture_sub[col] == dflab[col])
    print("Good!! dflab and dffuture are exactly algined")

    # Copy columns to dflab
    cols_take = ["trial_stroke_quality_abort", "trial_first_indstroke_that_fails", "trial_success", "trial_sequence_error_string",
                "Cr_NextStrk_Draw", "RnkWthn_NextStrk_Draw", "RnkWthnLast_NextStrk_Draw",
                "Cr_NextStrk_Corr", "RnkWthn_NextStrk_Corr", "RnkWthnLast_NextStrk_Corr",
                "Cr_NextChunk_Corr", "Success_ThisStrk_Draw", "Success_NextStrk_Draw", "AbortStrokeQuality_ThisStrk_Draw",
                "chunk_rank_correct", "chunk_within_rank_correct"]
    for col in cols_take:
        dflab[col] = dffuture_sub[col]

    # from pythonlib.tools.pandastools import grouping_print_n_samples
    # grouping_print_n_samples(dflab, ["trial_success", "trial_stroke_quality_abort", "trial_sequence_error_string"])

    ### A useful column, classifiying each stroke, a Semantic version of failure label. Takes into account
    # the current and upcoming strokes.
    list_label = []
    for _, row in dflab.iterrows():
        
        if row["Success_ThisStrk_Draw"]==False and row["AbortStrokeQuality_ThisStrk_Draw"]==False and (row["chunk_rank"]==row["chunk_rank_correct"]) and (row["chunk_within_rank"]!=row["chunk_within_rank_correct"]):
            # Failed grammar -- correct chunk, incorrect rank
            label = "stk_gram_gdchk_bdrnk"
        elif row["Success_ThisStrk_Draw"]==False and row["AbortStrokeQuality_ThisStrk_Draw"]==False and (row["chunk_rank"]!=row["chunk_rank_correct"]):
            # Failed grammar -- incorrect chunk
            label = "stk_gram_bdchk"
        elif row["Success_ThisStrk_Draw"]==True and row["AbortStrokeQuality_ThisStrk_Draw"]==True:
            # Failed stroke (correct grammar)
            label = "stk_qual"
        elif row["Success_ThisStrk_Draw"]==True and row["AbortStrokeQuality_ThisStrk_Draw"]==False and row["Success_NextStrk_Draw"]==False and (row["Cr_NextStrk_Draw"]>row["Cr_NextStrk_Corr"]) and (row["chunk_rank"]==row["Cr_NextStrk_Corr"]):
            # Next stroke fails grammar -- moved to next chunk even though current not completed
            label = "nxt_gram_chk2Early"
        elif row["Success_ThisStrk_Draw"]==True and row["AbortStrokeQuality_ThisStrk_Draw"]==False and row["Success_NextStrk_Draw"]==False and (row["Cr_NextStrk_Draw"]>row["Cr_NextStrk_Corr"]) and (row["chunk_rank"]<row["Cr_NextStrk_Corr"]):
            # Next stroke fails grammar -- this chunk completed, but you trasntiioned to the incorect chunk
            label = "nxt_gram_chkSkipChk"
        elif row["Success_ThisStrk_Draw"]==True and row["AbortStrokeQuality_ThisStrk_Draw"]==False and row["Success_NextStrk_Draw"]==False and (row["Cr_NextStrk_Draw"]==row["Cr_NextStrk_Corr"]) and (row["RnkWthn_NextStrk_Draw"]!=row["RnkWthn_NextStrk_Corr"]):
            # Next stroke fails grammar -- correct chunk, incorrect rank
            label = "nxt_gram_rnk"
        elif row["Success_ThisStrk_Draw"]==True and row["AbortStrokeQuality_ThisStrk_Draw"]==False and row["Success_NextStrk_Draw"]==True:
            # Success, this and next stroke
            label = "allsuccess"
        else:
            assert False
        list_label.append(label)
    dflab["stroke_error_label"] = list_label

    # Further conjunctive labels for this stroke
    dflab = append_col_with_grp_index(dflab, ["trial_sequence_error_string", "Success_ThisStrk_Draw", "Success_NextStrk_Draw", "Cr_NextStrk_Corr", "Cr_NextStrk_Draw", "stroke_error_label"], "stroke_error_class")
    dflab = append_col_with_grp_index(dflab, ["chunk_within_rank", "stroke_error_class"], "rankwithin_error")

    ### Store 
    PA.Xlabels["trials"] = dflab
    print("Done, merged into dflab")
    

if __name__=="__main__":

    SAVEDIR_ALL = f"/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS"

    # animal = "Diego"
    # date = 230913
    animal = sys.argv[1]
    date = int(sys.argv[2])
    PLOT_DO = int(sys.argv[3])

    # question = sys.argv[3]
    # run_number = int(sys.argv[4])

    # question = "RULE_ANBMCK_STROKE"
    version = "stroke"
    combine = False

    # PLOTS_DO = [1] # Good

    ### Load dfallpa that inlcudes all data, not just correct trials
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_concat_merge_pa_along_trials, dfpa_concatbregion_preprocess_wrapper, dfpa_concat_bregion_to_combined_bregion
    from pythonlib.tools.exceptions import NotEnoughDataException
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_concat_bregion_to_combined_bregion

    # Main
    version = "stroke"
    combine = False

    ### (1) load Grammar Dfallpa
    question = "RULE_ANBMCK_STROKE_ALLDATA"
    DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                        question=question)
    DFallpa = dfpa_concat_bregion_to_combined_bregion(DFallpa)

    # Normaliztaion
    if PLOT_DO in [1]:
        # Then use raw FR
        fr_mean_subtract_method = "raw_fr"
        dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date, fr_mean_subtract_method=fr_mean_subtract_method)
    elif PLOT_DO in [2]:
        # Then use normalized (population stuff)
        dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)
    else:
        print(PLOT_DO)
        assert False

    # First, load the behavioral dataset
    from pythonlib.dataset.dataset import load_dataset_daily_helper

    # Diego_use_main_21 = False
    D = load_dataset_daily_helper(animal, date)
    # Preprocess D for grammar stuff
    D.preprocessGood(params=['one_to_one_beh_task_strokes_allow_unfinished'])
    D.grammarparses_successbinary_score_wrapper()
    for ind in range(len(D.Dat)):
        D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(ind)
    D.grammarparses_syntax_concrete_append_column()

    # Get future strokes chunks info.
    from pythonlib.dataset.dataset_analy.grammar import syntaxconcrete_extract_wrapper_chunks_future_errors_info
    dffuture = syntaxconcrete_extract_wrapper_chunks_future_errors_info(D)
    
    ################################### PLOTS
    if PLOT_DO==1:
        # Units: Smoothed FR plots
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS/sm_fr_units/{animal}-{date}"        
        os.makedirs(SAVEDIR, exist_ok=True)
        plotwrapper_smfr_units(DFallpa, dffuture, SAVEDIR)
    elif PLOT_DO==2:
        # Population: Sm Fr and Euclidean.
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good/ERRORS/statespace_and_euclid/{animal}-{date}"
        os.makedirs(SAVEDIR, exist_ok=True)

        DO_PLOT_STATESPACE = True
        DO_EUCLIDEAN = True
        DFDIST = plotwrapper_pop_state_eurlidean(DFallpa, dffuture, SAVEDIR, DO_PLOT_STATESPACE, DO_EUCLIDEAN)
    else:
        print(PLOT_DO)
        assert False