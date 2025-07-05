"""
Specifically to make plots of heatmaps of shape vs. location, for single prims,
good, for paper.

NOTEBOOK: 241110_shape_invariance_all_plots_SP.ipynb

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

from neuralmonkey.classes.session import _REGIONS_IN_ORDER, _REGIONS_IN_ORDER_COMBINED
ORDER_BREGION = _REGIONS_IN_ORDER_COMBINED

N_MIN_TRIALS_PER_SHAPE = 4
TWIND_ANALY = (-0.6, 1.0) # This is just for windowing final data, not for fitting pca.
NPCS_KEEP = 8

# if int(date)>220720:
#     twind = (0.05, 0.9)
# else:
#     twind = (0.05, 0.6)

# SUBSPACE_PROJ_FIT_TWIND = {
#     "03_samp":[twind],
#     "04_go_cue":[(-0.3, 0.3)],
#     "05_first_raise":[(-0.3, 0.3)],
#     "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
# }

# Just as suffixes for loading euclidian_time_resolved. Hacky. Use the twinds from SUBSPACE_PROJ_FIT_TWIND above.
LIST_TWIND_POSSIBLE = [
    TWIND_ANALY,
    (0.05, 0.9),
    (0.05, 0.6),
    (-0.3, 0.3),
    (0.3, 0.3),
    (-0.5, -0.05), 
    (0.05, 0.5)
    ]


# LIST_ANIMAL_DATE_COMB = [
#     # ("Diego", 231130, True),
#     ("Diego", 231122, True), # Old [1]
#     ("Diego", 231128, True), 
#     ("Diego", 231129, True),
#     ("Diego", 231201, True),
#     ("Diego", 231205, True),
#     ("Diego", 231120, True), # Added
#     ("Diego", 231121, True),
#     ("Diego", 231206, True),
#     ("Diego", 231218, True),
#     ("Diego", 231220, True),
# ]

def mult_load_euclidian_time_resolved(LIST_ANIMAL_DATE_COMB_VAROTHER):
    """
    Load and collect all dates into a single dataframe,
    results from euclidian_time_resolved()
    LIST_ANIMAL_DATE_COMB, should have just one animal, date
    PARAMS:
    - savedir_method_old, this is for stroke.
    """
    # savedir = f"{SAVEDIR}/EUCLIDIAN/{animal}-{date}-combine={combine}-var_other={var_other}"

    SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN/MULT"
    which_level = "trial"
    events = ["03_samp", "04_go_cue", "05_first_raise", "06_on_strokeidx_0"]

    list_df = []
    paths_loaded = []
    for animal, date, combine, var_other in LIST_ANIMAL_DATE_COMB_VAROTHER:
        SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN/{animal}-{date}-combine={combine}-var_other={var_other}"

        if combine:
            REGIONS = _REGIONS_IN_ORDER_COMBINED
        else:
            REGIONS = _REGIONS_IN_ORDER

        for bregion in REGIONS:
            for ev in events:
                for subspace_projection in [None, "pca", "pca_proj", "shape", "shape_size", "shape_loc"]:
                    for remove_drift in [False, True]:
                        for raw_subtract_mean_each_timepoint in [False, True]:
                            for remove_singleprims_unstable in [False, True]:
                                for remove_trials_with_bad_strokes in [False, True]:
                                    for twind_fit in LIST_TWIND_POSSIBLE:
                                        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{ev}-ss={subspace_projection}-nodrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={twind_fit}"
                                        print(SAVEDIR)
                                        try:
                                            path = f"{SAVEDIR}/DFDIST.pkl"
                                            dfdist = pd.read_pickle(path)
                                            paths_loaded.append(path)
                                        except FileNotFoundError as err:
                                            print("Skipping this region:", bregion, animal)
                                            continue

                                        dfdist["animal"] = animal
                                        dfdist["date"] = date
                                        dfdist["bregion"] = bregion
                                        dfdist["combine_areas"] = combine
                                        dfdist["event"] = ev

                                        # Metaparams
                                        dfdist["subspace_projection"] = subspace_projection
                                        dfdist["remove_drift"] = remove_drift
                                        dfdist["raw_subtract_mean_each_timepoint"] = raw_subtract_mean_each_timepoint
                                        dfdist["remove_singleprims_unstable"] = remove_singleprims_unstable
                                        dfdist["remove_trials_with_bad_strokes"] = remove_trials_with_bad_strokes
                                        dfdist["subspace_twind"] = [twind_fit for _ in range(len(dfdist))]

                                        list_df.append(dfdist)

    print("... Loaded these paths:")
    for p in paths_loaded:
        print(p)

    ### Collect across days
    DFDIST = pd.concat(list_df).reset_index(drop=True)
    DFDIST = append_col_with_grp_index(DFDIST, ["remove_trials_with_bad_strokes", "subspace_projection", "subspace_twind", "remove_drift", "raw_subtract_mean_each_timepoint", 
                                    "remove_singleprims_unstable"], "metaparams")
    replace_values_with_this(DFDIST, "subspace_projection", None, "none")
    
    # Check there are no Nones
    assert DFDIST.isnull().values.any() == False, "replace Nones using replace_values_with_this"

    # Some plots use this...
    if "prune_version" not in DFDIST:
        DFDIST["prune_version"] = "dummy"
    if "remove_trials_with_bad_strokes" not in DFDIST:
        DFDIST["remove_trials_with_bad_strokes"] = "dummy"

    return DFDIST

def _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                 twind_analy, tbin_dur, tbin_slide, savedir, raw_subtract_mean_each_timepoint=False,
                                 inds_pa_fit=None, inds_pa_final=None,
                                 n_min_per_lev_lev_others=1, scalar_or_traj="traj"):
    """
    Helper to do dim reduction on PA, returning copy of PA that is modifed.

    MS: checked
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    if subspace_projection is not None:
        dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

        # (1) First, dim reduction
        superv_dpca_var = superv_dpca_params['superv_dpca_var']
        superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']
        
        print("***:", dim_red_method, " fit twind: ", subspace_projection_fitting_twind, ", data twind: ", twind_analy)
        _, PA = PA.dataextract_dimred_wrapper(scalar_or_traj, dim_red_method, savedir, 
                                        subspace_projection_fitting_twind, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
                                        NPCS_KEEP = NPCS_KEEP,
                                        dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict, 
                                        dpca_proj_twind = twind_analy, 
                                        raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                        umap_n_components=None, umap_n_neighbors=None,
                                        inds_pa_fit=inds_pa_fit, inds_pa_final=inds_pa_final,
                                        n_min_per_lev_lev_others=n_min_per_lev_lev_others)
    else:
        if tbin_dur is not None:
            PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        if twind_analy is not None:
            PA = PA.slice_by_dim_values_wrapper("times", twind_analy)
        if raw_subtract_mean_each_timepoint:
            PA = PA.norm_subtract_trial_mean_each_timepoint()
    plt.close("all")
    return PA

def preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, subspace_projection, subspace_projection_fitting_twind,
                twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False,
                skip_dim_reduction=False, var_context_same=None):
    """
    Holds all prepreocessing.

    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, extract_with_levels_of_conjunction_vars_helper, extract_with_levels_of_var_good
    
    dflab = PA.Xlabels["trials"]
    dflab = append_col_with_grp_index(dflab, ["seqc_0_shape", "seqc_0_loc"], "seqc_0_shapeloc")    
    dflab = append_col_with_grp_index(dflab, ["seqc_0_shape", "gridsize"], "seqc_0_shapesize")    
    PA.Xlabels["trials"] = dflab

    ################################### PRUNE TO GOOD N COUNTS
    ### (0) Plot original tabulation of shape vs task_klind
    if savedir is not None:
        dflab = PA.Xlabels["trials"]
        fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "seqc_0_shape", "seqc_0_loc", ["task_kind", "gridsize"])
        path = f"{savedir}/shape_counts-orig.pdf"
        fig.savefig(path)

    # (2) Keep just shapes taht exist across both SP and CHAR.
    dflab = PA.Xlabels["trials"]
    _dfout,_  = extract_with_levels_of_conjunction_vars_helper(dflab, "seqc_0_shape", [var_other], n_min_per_lev=N_MIN_TRIALS_PER_SHAPE,
                                                plot_counts_heatmap_savepath=None)
    
    # (3) Need at least n trials per group
    if var_context_same is not None:
        _dfout, _ = extract_with_levels_of_var_good(_dfout, ["seqc_0_shape", var_other, var_context_same], 2)
    else:
        _dfout, _ = extract_with_levels_of_var_good(_dfout, ["seqc_0_shape", var_other], 2)

    # if len(_dfout)==0 or _dfout is None:
    #     print("Pruned all data!! returning None")
    #     print("... using these params: ", shape_var, task_kinds)
    #     return None
    #     # assert False, "not enough data"

    index_datapt_list_keep = _dfout["index_datapt"].tolist()
    PA = PA.slice_by_labels_filtdict({"index_datapt":index_datapt_list_keep})

    ### Plot counts one final time
    if savedir is not None:
        dflab = PA.Xlabels["trials"]
        # fig = grouping_plot_n_samples_conjunction_heatmap(dflab, shape_var, "task_kind")
        fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "seqc_0_shape", "seqc_0_loc", ["task_kind", "gridsize"])
        path = f"{savedir}/shape_counts-final.pdf"
        fig.savefig(path)

    # ------------------------------ (done)
    # Optioanlly, remove channels with drift
    if remove_drift:
        from neuralmonkey.classes.population_mult import dfallpa_preprocess_sitesdirty_single_just_drift
        PA = dfallpa_preprocess_sitesdirty_single_just_drift(PA, animal, date, savedir=savedir)

    ############ PROJECTION
    PA = _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                 twind_analy, tbin_dur, tbin_slide, savedir, raw_subtract_mean_each_timepoint=False,
                                 n_min_per_lev_lev_others=2)

    # if subspace_projection is not None:
    #     dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

    #     # (1) First, dim reduction
    #     superv_dpca_var = superv_dpca_params['superv_dpca_var']
    #     superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
    #     superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']
        
    #     print("***:", dim_red_method, " fit twind: ", subspace_projection_fitting_twind, ", data twind: ", twind_analy)
    #     _, PA = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
    #                                     subspace_projection_fitting_twind, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
    #                                     NPCS_KEEP = NPCS_KEEP,
    #                                     dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict, 
    #                                     dpca_proj_twind = twind_analy, 
    #                                     raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
    #                                     umap_n_components=None, umap_n_neighbors=None)
    # else:
    #     if tbin_dur is not None:
    #         PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
    #     if twind_analy is not None:
    #         PA = PA.slice_by_dim_values_wrapper("times", twind_analy)
    #     if raw_subtract_mean_each_timepoint:
    #         PA = PA.norm_subtract_trial_mean_each_timepoint()

    return PA    

def euclidian_time_resolved(DFallpa, animal, date, var_other, SAVEDIR_ANALYSIS):
    """
    Eucldian distance [effect of shape vs. task(context)] as function of time, relative to stroke onset.
    """
    import seaborn as sns

    var_effect = "seqc_0_shape"

    raw_subtract_mean_each_timepoint = False

    twind_analy = TWIND_ANALY
    tbin_dur = 0.15
    tbin_slide = 0.01

    if int(date)>220720:
        twind = (0.05, 0.9)
    else:
        twind = (0.05, 0.6)

    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
        "04_go_cue":[(-0.3, 0.3)],
        "05_first_raise":[(-0.3, 0.3)],
        "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
    }


    if var_other == "seqc_0_loc":
        list_subspace_projection = ["pca_proj", "pca", "shape", "shape_loc", None]
    elif var_other == "gridsize":
        list_subspace_projection = ["pca_proj", "pca", "shape", "shape_size", None]
    else:
        assert False

    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
            if subspace_projection is None:
                # Then also plot raw, without clearning
                list_unstable_badstrokes = [(False, False, False), (True, True, True)]
            else:
                # plot only cleaned up data.
                list_unstable_badstrokes = [(True, True, True)]
                
            # for remove_drift in [False]:
            for remove_drift, remove_singleprims_unstable, remove_trials_with_bad_strokes in list_unstable_badstrokes:

                ############################
                if subspace_projection in [None, "pca"]:
                    list_fit_twind = [twind_analy]
                else:
                    list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]
                
                for subspace_projection_fitting_twind in list_fit_twind:
                    
                    # Final save dir
                    SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-nodrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={subspace_projection_fitting_twind}"
                    os.makedirs(SAVEDIR, exist_ok=True)
                    print("SAVING AT ... ", SAVEDIR)

                    # Preprocess
                    savedir = f"{SAVEDIR}/preprocess"
                    os.makedirs(savedir, exist_ok=True)
                    pa = preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, subspace_projection, subspace_projection_fitting_twind, 
                                    twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False)
                    
                    ###################################### Running euclidian
                    from neuralmonkey.analyses.euclidian_distance import timevarying_compute, timevarying_convert_to_scalar
                    vars_group = [var_effect, var_other]
                    DFDIST = timevarying_compute(pa, vars_group)
                    pd.to_pickle(DFDIST, f"{SAVEDIR}/DFDIST.pkl")
                    
                    ### Plot
                    for y in ["dist_mean", "dist_norm", "dist_yue_diff"]:
                        # sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
                        fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue=f"same-{var_effect}|{var_other}", kind="line", errorbar=("ci", 68))
                        savefig(fig, f"{SAVEDIR}/relplot-{y}-1.pdf")

                    plt.close("all")

def euclidian_time_resolved_fast_shuffled(DFallpa, animal, date, var_other, SAVEDIR_ANALYSIS,
                                          DO_SHUFFLE=False, DO_RSA_HEATMAPS=False, var_context_same=None,
                                          quick_mode=False):
    """
    Compute pariwise euclidean distance, using trajectories.

    Good! Much faster method compared to previosu. And does two other things important for more rigorous result:
    1. Train-test split for dim redu (separate for fitting[smaller] and then final data projected)

    Also gets shuffled data optionally

    PARAMS:
    - DO_SHUFFLE, bool, if True, then gets shuffle by shuffling 3 ways, all working at level of trial.
    To analyze and plot results of shuffle, use Stats method #1
    (See notebookb: /home/lucas/code/neuralmonkey/neuralmonkey/notebooks_tutorials/241110_shape_invariance_all_plots_SP.ipynb)
    --> I stopped doing shuffle, since all I wanted was the dfdists. I then run shuffling on those, using method #3.
    - DO_RSA_HEATMAPS, bool. This may take significantly more time...
    """
    import seaborn as sns

    # assert var_context_same is not None, "to make sure I dont forget. you can just enter a dummy variable"
    assert var_context_same is None, "didnt seem towork, so for paper I just used None. Should fix it if want to use."
    var_effect = "seqc_0_shape"

    # If do train-test split for getting subspace and then computing distance after projecting to that subsapce.
    N_SPLITS_OUTER = 1
    N_SPLITS = 8
    N_SHUFF_PER_SPLIT = int(np.ceil(1000/6))
    DO_SHUFFLE = False

    twind_analy = TWIND_ANALY
    tbin_dur = 0.15
    tbin_slide = 0.02

    if int(date)>220720:
        twind = (0.05, 0.9)
    else:
        twind = (0.05, 0.6)

    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
        "04_go_cue":[(-0.3, 0.3)],
        "05_first_raise":[(-0.3, 0.3)],
        "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
    }

    if var_other == "seqc_0_loc":
        list_subspace_projection = ["shape", "shape_loc"]
    elif var_other == "gridsize":
        list_subspace_projection = ["shape", "shape_size"]
    else:
        assert False

    if quick_mode:
        list_subspace_projection = ["shape"]

    map_event_to_listtwind = {
            "03_samp":[(0.05, 0.3), (0.3, 0.6), (0.05, 0.6), (0.5, 1.0)],
            "05_first_raise":[(-0.5,  -0.1), (-0.1, 0.5)],
            "06_on_strokeidx_0":[(-0.5, -0.1), (-0.4, -0.05), (0, 0.5)],
        }
        
    if quick_mode:
        map_event_to_listtwind = {
                "03_samp":[(0.05, 0.6)],
            }

    list_dfdist =[]
    list_dfdist_shuff =[]
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection:
            # plot only cleaned up data.
            list_unstable_badstrokes = [(True, True, True)]
                
            # for remove_drift in [False]:
            for remove_drift, remove_singleprims_unstable, remove_trials_with_bad_strokes in list_unstable_badstrokes:

                ############################
                if subspace_projection in [None, "pca"]:
                    list_fit_twind = [twind_analy]
                else:
                    list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]
                
                for subspace_projection_fitting_twind in list_fit_twind:
                    
                    # Final save dir
                    SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={subspace_projection_fitting_twind}"
                    os.makedirs(SAVEDIR, exist_ok=True)
                    print("SAVING AT ... ", SAVEDIR)

                    if DO_RSA_HEATMAPS:
                        # Plot pairwise distances (rsa heatmaps).
                        # This is done separatee to below becuase it doesnt use the train-test splits.
                        # It shold but I would have to code way to merge multple Cl, which is doable.
                        from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

                        PAthis = preprocess_pa(PA, animal, date, var_other, None, remove_drift, 
                                               subspace_projection, subspace_projection_fitting_twind, 
                                               twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False,
                                               skip_dim_reduction=False)
                        
                        list_twind_scalar = map_event_to_listtwind[event]
                        for twind_scal in list_twind_scalar:
                            savedir = f"{SAVEDIR}/rsa_heatmap/twindscal={twind_scal}"
                            os.makedirs(savedir, exist_ok=True)

                            # Prune to scalar window
                            pa = PAthis.slice_by_dim_values_wrapper("times", twind_scal)

                            # Make rsa heatmaps.
                            timevarying_compute_fast_to_scalar(pa, [var_effect, var_other], rsa_heatmap_savedir=savedir, 
                                                               var_context_same=var_context_same)

                    # Preprocess
                    savedir = f"{SAVEDIR}/preprocess"
                    os.makedirs(savedir, exist_ok=True)

                    skip_dim_reduction = True # will do so below... THis just do other preprocessing, and widowing
                    # PAthis = preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, subspace_projection, subspace_projection_fitting_twind, 
                    #                 twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False,
                    #                 skip_dim_reduction=skip_dim_reduction)
                    PAthis = preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, None, None, 
                                    twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False,
                                    skip_dim_reduction=skip_dim_reduction)

                    ########### DO TRAIN-TEST SPLITS
                    for i_outer in range(N_SPLITS_OUTER):
                        if var_context_same is not None:
                            _vars_grp = [var_effect, var_other, var_context_same]
                        else:
                            _vars_grp = [var_effect, var_other]
                        
                        if False:
                            # Fails, sometimes throws out too much data
                            folds_dflab = PAthis.split_balanced_stratified_kfold_subsample_level_of_var(_vars_grp, None, None, 
                                                                                                        n_splits=N_SPLITS, 
                                                                                                        do_balancing_of_train_inds=False)
                            print(N_SPLITS)
                            print(len(folds_dflab))
                            for a, b in folds_dflab:
                                print(len(a), len(b))
                            assert False, "Check that n splits matches the n splits you wanted."
                        else:
                            # Better, more careful, ensuring enough data for euclidian distance.
                            fraction_constrained_set=0.75
                            n_constrained=3 # Ideally have more than 1 pair
                            list_labels_need_n=None
                            min_frac_datapts_unconstrained=None
                            min_n_datapts_unconstrained=len(PAthis.Xlabels["trials"][var_effect].unique())
                            plot_train_test_counts=True
                            plot_indices=False
                            folds_dflab, fig_unc, fig_con = PAthis.split_stratified_constrained_grp_var(N_SPLITS, _vars_grp, 
                                                                            fraction_constrained_set, n_constrained, 
                                                                            list_labels_need_n, min_frac_datapts_unconstrained,  
                                                                            min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
                            savefig(fig_con, f"{savedir}/after_split_constrained_fold_0.pdf")
                            savefig(fig_unc, f"{savedir}/after_split_unconstrained_fold_0.pdf")
                            plt.close("all")

                        for _i_dimredu, (inds_fit_pca, inds_final) in enumerate(folds_dflab):
                            print(f"...splits, i_outer={i_outer}, i_inner={_i_dimredu}")

                            # train_inds, more inds than than test_inds
                            inds_fit_pca = [int(i) for i in inds_fit_pca]
                            inds_final = [int(i) for i in inds_final]

                            savedir = f"{SAVEDIR}/preprocess/i_outer={i_outer}-i_dimredu={_i_dimredu}"
                            os.makedirs(savedir, exist_ok=True)
                            
                            PAthisRedu = _preprocess_pa_dim_reduction(PAthis, subspace_projection, subspace_projection_fitting_twind,
                                        twind_analy, tbin_dur, tbin_slide, savedir=savedir, raw_subtract_mean_each_timepoint=False,
                                        inds_pa_fit=inds_fit_pca, inds_pa_final=inds_final)

                            if PAthisRedu is None:
                                continue

                            # Take different windows (for computing scalar score)
                            # Go thru diff averaging windows (to get scalar)
                            list_twind_scalar = map_event_to_listtwind[event]
                            for twind_scal in list_twind_scalar:
                                
                                pa = PAthisRedu.slice_by_dim_values_wrapper("times", twind_scal)

                                ###################################### Running euclidian
                                from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar
                                
                                # (1) Data
                                dfdist, _ = timevarying_compute_fast_to_scalar(pa, [var_effect, var_other], 
                                                                               var_context_same=var_context_same,
                                                                               plot_conjunctions_savedir=savedir)
                                
                                dfdist["bregion"] = bregion
                                dfdist["which_level"] = which_level
                                dfdist["event"] = event
                                dfdist["subspace_projection"] = subspace_projection
                                dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]
                                dfdist["shuffle_ver"] = "ignore"
                                dfdist["shuffled"] = False
                                dfdist["shuffle_iter"] = 0
                                dfdist["dim_redu_fold"] = _i_dimredu
                                dfdist["i_outer"] = i_outer
                                dfdist["twind_scal"] = [twind_scal for _ in range(len(dfdist))]
                                list_dfdist.append(dfdist)

                                # (2) Shuffle
                                if DO_SHUFFLE:
                                    assert False, "how deal with var_context_same in shuffle_dataset_varconj()?"
                                    from pythonlib.tools.pandastools import shuffle_dataset_hierarchical, shuffle_dataset_singlevar, shuffle_dataset_varconj

                                    for shuffle_ver in ["datapts_all", "seqc_0_shape", "var_other"]:
                                        pa_shuff = pa.copy()
                                        dflab = pa_shuff.Xlabels["trials"]

                                        for i_shuff in range(N_SHUFF_PER_SPLIT):
                                            print("RUNNING SHUFFLE", shuffle_ver, ", iter:", i_shuff)

                                            # 0. Create shuffled dataset
                                            if shuffle_ver == "datapts_all":
                                                # Lowest level, shuffle all trails idnepetn
                                                # dflabSHUFF = shuffle_dataset_hierarchical(dflab, [var], var_others)
                                                dflab_shuff = shuffle_dataset_varconj(dflab, vars_group, maintain_block_temporal_structure=False)
                                            elif shuffle_ver=="seqc_0_shape":
                                                dflab_shuff = shuffle_dataset_hierarchical(dflab, ["seqc_0_shape"], [var_other], return_in_input_order=True)
                                            elif shuffle_ver=="var_other":
                                                dflab_shuff = shuffle_dataset_hierarchical(dflab, [var_other], ["seqc_0_shape"], return_in_input_order=True)
                                            else:
                                                assert False

                                            pa_shuff.Xlabels["trials"] = dflab_shuff
                                            dfdist_shuff, _ = timevarying_compute_fast_to_scalar(pa_shuff, [var_effect, var_other], var_context_same=var_context_same)

                                            dfdist_shuff["bregion"] = bregion
                                            dfdist_shuff["which_level"] = which_level
                                            dfdist_shuff["event"] = event
                                            dfdist_shuff["subspace_projection"] = subspace_projection
                                            dfdist_shuff["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist_shuff))]
                                            dfdist_shuff["shuffle_ver"] = shuffle_ver
                                            dfdist_shuff["shuffled"] = True
                                            dfdist_shuff["shuffle_iter"] = i_shuff
                                            dfdist_shuff["dim_redu_fold"] = _i_dimredu
                                            dfdist["i_outer"] = i_outer
                                            dfdist_shuff["twind_scal"] = [twind_scal for _ in range(len(dfdist_shuff))]
                                            list_dfdist_shuff.append(dfdist_shuff)      
                                plt.close("all")

                            # Save, intermediate steps
                            # ONly do this if shuff, otherwise is not needed.
                            # (And try to avoid doing this, since it overwrite old data.)
                            if DO_SHUFFLE:
                                import pickle

                                with open(f"{SAVEDIR_ANALYSIS}/list_dfdist.pkl", "wb") as f:
                                    pickle.dump(list_dfdist, f)

                                with open(f"{SAVEDIR_ANALYSIS}/list_dfdist_shuff.pkl", "wb") as f:
                                    pickle.dump(list_dfdist_shuff, f)

    # FINAL SAVE
    import pickle
    with open(f"{SAVEDIR_ANALYSIS}/list_dfdist.pkl", "wb") as f:
        pickle.dump(list_dfdist, f)
    with open(f"{SAVEDIR_ANALYSIS}/list_dfdist_shuff.pkl", "wb") as f:
        pickle.dump(list_dfdist_shuff, f)


def euclidian_time_resolved_fast_shuffled_manuscript(DFallpa, animal, date, var_other, SAVEDIR_ANALYSIS,
                                          DO_SHUFFLE=False, DO_RSA_HEATMAPS=False, var_context_same=None,
                                          quick_mode=False):
    """
    Compute pariwise euclidean distance, using trajectories.
    
    Also plot RSA matrices

    Good! Much faster method compared to previosu. And does two other things important for more rigorous result:
    1. Train-test split for dim redu (separate for fitting[smaller] and then final data projected)

    Also gets shuffled data optionally

    PARAMS:
    - DO_SHUFFLE, bool, if True, then gets shuffle by shuffling 3 ways, all working at level of trial.
    To analyze and plot results of shuffle, use Stats method #1
    (See notebookb: /home/lucas/code/neuralmonkey/neuralmonkey/notebooks_tutorials/241110_shape_invariance_all_plots_SP.ipynb)
    --> I stopped doing shuffle, since all I wanted was the dfdists. I then run shuffling on those, using method #3.
    - DO_RSA_HEATMAPS, bool. This may take significantly more time...

    MS: checked
    """
    import seaborn as sns
    assert var_context_same is None
    var_effect = "seqc_0_shape"

    # If do train-test split for getting subspace and then computing distance after projecting to that subsapce.
    N_SPLITS_OUTER = 1
    N_SPLITS = 8
    N_SHUFF_PER_SPLIT = int(np.ceil(1000/6))
    DO_SHUFFLE = False

    twind_analy = TWIND_ANALY
    tbin_dur = 0.15
    tbin_slide = 0.02

    if int(date)>220720:
        twind = (0.05, 0.9)
    else:
        twind = (0.05, 0.6)

    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
    }

    list_subspace_projection = ["shape"]

    map_event_to_listtwind = {
            "03_samp":[(0.05, 0.6)],
        }

    list_dfdist =[]
    list_dfdist_shuff =[]
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection:
            list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]
            
            for subspace_projection_fitting_twind in list_fit_twind:
                
                # Final save dir
                SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-fit_twind={subspace_projection_fitting_twind}"
                os.makedirs(SAVEDIR, exist_ok=True)
                print("SAVING AT ... ", SAVEDIR)

                if DO_RSA_HEATMAPS:
                    # Plot pairwise distances (rsa heatmaps).
                    # This is done separatee to below becuase it doesnt use the train-test splits.
                    # It shold but I would have to code way to merge multple Cl, which is doable.
                    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

                    PAthis = _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                                twind_analy, tbin_dur, tbin_slide, None)

                    list_twind_scalar = map_event_to_listtwind[event]
                    for twind_scal in list_twind_scalar:
                        savedir = f"{SAVEDIR}/rsa_heatmap/twindscal={twind_scal}"
                        os.makedirs(savedir, exist_ok=True)

                        # Prune to scalar window
                        pa = PAthis.slice_by_dim_values_wrapper("times", twind_scal)

                        # Make rsa heatmaps.
                        timevarying_compute_fast_to_scalar(pa, [var_effect, var_other], rsa_heatmap_savedir=savedir, 
                                                            var_context_same=var_context_same)

                # Preprocess
                savedir = f"{SAVEDIR}/preprocess"
                os.makedirs(savedir, exist_ok=True)

                PAthis = PA.copy()

                ########### DO TRAIN-TEST SPLITS
                for i_outer in range(N_SPLITS_OUTER):
                    if var_context_same is not None:
                        _vars_grp = [var_effect, var_other, var_context_same]
                    else:
                        _vars_grp = [var_effect, var_other]
                    
                    # Better, more careful, ensuring enough data for euclidian distance.
                    fraction_constrained_set=0.75
                    n_constrained=3 # Ideally have more than 1 pair
                    list_labels_need_n=None
                    min_frac_datapts_unconstrained=None
                    min_n_datapts_unconstrained=len(PAthis.Xlabels["trials"][var_effect].unique())
                    plot_train_test_counts=True
                    plot_indices=False
                    folds_dflab, fig_unc, fig_con = PAthis.split_stratified_constrained_grp_var(N_SPLITS, _vars_grp, 
                                                                    fraction_constrained_set, n_constrained, 
                                                                    list_labels_need_n, min_frac_datapts_unconstrained,  
                                                                    min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
                    savefig(fig_con, f"{savedir}/after_split_constrained_fold_0.pdf")
                    savefig(fig_unc, f"{savedir}/after_split_unconstrained_fold_0.pdf")
                    plt.close("all")

                    for _i_dimredu, (inds_fit_pca, inds_final) in enumerate(folds_dflab):
                        print(f"...splits, i_outer={i_outer}, i_inner={_i_dimredu}")

                        # train_inds, more inds than than test_inds
                        inds_fit_pca = [int(i) for i in inds_fit_pca]
                        inds_final = [int(i) for i in inds_final]

                        savedir = f"{SAVEDIR}/preprocess/i_outer={i_outer}-i_dimredu={_i_dimredu}"
                        os.makedirs(savedir, exist_ok=True)
                        
                        PAthisRedu = _preprocess_pa_dim_reduction(PAthis, subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide, savedir=savedir, raw_subtract_mean_each_timepoint=False,
                                    inds_pa_fit=inds_fit_pca, inds_pa_final=inds_final)

                        if PAthisRedu is None:
                            continue

                        # Take different windows (for computing scalar score)
                        # Go thru diff averaging windows (to get scalar)
                        list_twind_scalar = map_event_to_listtwind[event]
                        for twind_scal in list_twind_scalar:
                            
                            pa = PAthisRedu.slice_by_dim_values_wrapper("times", twind_scal)

                            ###################################### Running euclidian
                            from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar
                            
                            # (1) Data
                            dfdist, _ = timevarying_compute_fast_to_scalar(pa, [var_effect, var_other], 
                                                                            var_context_same=var_context_same,
                                                                            plot_conjunctions_savedir=savedir)
                            
                            dfdist["bregion"] = bregion
                            dfdist["which_level"] = which_level
                            dfdist["event"] = event
                            dfdist["subspace_projection"] = subspace_projection
                            dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]
                            dfdist["shuffle_ver"] = "ignore"
                            dfdist["shuffled"] = False
                            dfdist["shuffle_iter"] = 0
                            dfdist["dim_redu_fold"] = _i_dimredu
                            dfdist["i_outer"] = i_outer
                            dfdist["twind_scal"] = [twind_scal for _ in range(len(dfdist))]
                            list_dfdist.append(dfdist)

                            plt.close("all")

    # FINAL SAVE
    import pickle
    with open(f"{SAVEDIR_ANALYSIS}/list_dfdist.pkl", "wb") as f:
        pickle.dump(list_dfdist, f)
    with open(f"{SAVEDIR_ANALYSIS}/list_dfdist_shuff.pkl", "wb") as f:
        pickle.dump(list_dfdist_shuff, f)

    
def euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other, also_load_shuffled=False,
                                                      rename_event_with_eventtwind=True, events_keep=None,
                                                      analysis_kind="shape_invar", convert_to_df_with_postprocessing=False,
                                                      merge_pancho_ss_twinds=False, savedir=None,
                                                      DO_FURTHER_POSTPROCESSING=False, 
                                                      prune_min_n_trials=None, old_run_number=None,
                                                      SAVEDIR_BASE = None,
                                                      SAVEDIR_BASE_char=None,
                                                      manuscript_version=False):
    """
    Helper to reload a SINGLE DAY, and do relevant postprocessing and agging. 

    Two rounds of agging:
    1. over replicates (chan subsets, and trial subsets)
    2. over label pairs.

    MS: checked
    """
    import pickle
    import os

    # Load an old run that's been saved
    if old_run_number: 
        assert analysis_kind in ["char_sp_00_stroke_revision_regr_seman", "char_sp_00_stroke_revision_noregr_seman"]
    assert convert_to_df_with_postprocessing==True, "I usually do this -- are you sure?"

    combine = True

    if SAVEDIR_BASE is None:
        SAVEDIR_BASE = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF"

    if SAVEDIR_BASE_char is None:
        SAVEDIR_BASE_char = "/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE"

    if analysis_kind in ["shape_invar_clean_loc", "shape_invar_clean_size", "shape_invar"]:
        var_effect = "seqc_0_shape"
        var_other = var_other
        SAVEDIR_ORIG = SAVEDIR_BASE
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-var_other={var_other}"
    elif analysis_kind == "shape_invar_context":
        # Like shape invar, but better, since it uses a context variable (e.g compare shape vs. loc, conditioned on context of size)
        assert False, "isnt working well, since somehow this looks the same (not idnetial) as simply shape_invar. Check that this actually worked."
        var_effect = "seqc_0_shape"
        var_other = var_other
        if var_other=="seqc_0_loc":
            var_context_same = "gridsize"
        elif var_other == "gridsize":
            var_context_same = "seqc_0_loc"
        else:
            print(var_other)
            assert False
        SAVEDIR_ORIG = SAVEDIR_BASE
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-var_other={var_other}-var_context_same={var_context_same}"
    elif analysis_kind == "char_sp":
        # /lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE/Diego-231120-combine=True-wl=stroke
        var_effect = "shape_semantic_grp"
        var_other = "task_kind"
        wl = "stroke"
        SAVEDIR_ORIG = SAVEDIR_BASE_char
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}"
    elif analysis_kind == "char_sp_05_first_raise":
        var_effect = "shape_semantic_grp"
        var_other = "task_kind"
        wl = "trial"
        SAVEDIR_ORIG = SAVEDIR_BASE_char
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}-05_first_raise"
    elif analysis_kind == "char_sp_04_go_cue":
        var_effect = "shape_semantic_grp"
        var_other = "task_kind"
        wl = "trial"
        SAVEDIR_ORIG = SAVEDIR_BASE_char
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}-04_go_cue"
    elif analysis_kind == "char_sp_00_stroke":
        # GOOD
        var_effect = "shape_semantic_grp"
        var_other = "task_kind"
        wl = "stroke"
        SAVEDIR_ORIG = SAVEDIR_BASE_char
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}-00_stroke"
    elif analysis_kind in ["char_sp_00_stroke_revision_regr", "char_sp_00_stroke_revision_noregr"]:
        assert False, "not using shape_semantic_grp anymore"
        var_effect = "shape_semantic_grp"
        var_other = "task_kind"
        wl = "stroke"
        if analysis_kind=="char_sp_00_stroke_revision_regr":
            regrhack=True
        elif analysis_kind=="char_sp_00_stroke_revision_noregr":
            regrhack=False
        else:
            assert False
        # SAVEDIR_ORIG = "/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE_revision"
        SAVEDIR_ORIG = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE_revision-VAR_SHAPE={var_effect}"
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}-00_stroke-regrhack={regrhack}"

    elif analysis_kind in ["char_sp_00_stroke_revision_regr_seman", "char_sp_00_stroke_revision_noregr_seman"]:
        # The final revision
        var_effect = "shape_semantic"
        var_other = "task_kind"
        wl = "stroke"
        if analysis_kind=="char_sp_00_stroke_revision_regr_seman":
            regrhack=True
        elif analysis_kind=="char_sp_00_stroke_revision_noregr_seman":
            regrhack=False
        else:
            assert False
        
        if old_run_number is not None:
            SAVEDIR_ORIG = f"{SAVEDIR_BASE_char}_revision_runs/run{old_run_number}"
        else:
            SAVEDIR_ORIG = f"{SAVEDIR_BASE_char}_revision-VAR_SHAPE={var_effect}"
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}-00_stroke-regrhack={regrhack}"

        if manuscript_version:
            SAVEDIR = f"/lemur2/lucas/analyses/manuscripts/1_action_symbols/REPRODUCED_FIGURES/fig6f/{animal}-{date}-regrhack={regrhack}"
    else:
        print(analysis_kind)
        assert False

    SAVEDIR_PLOTS = f"{SAVEDIR}/RELOADED_PLOTS"
    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

    print("Loading: ", f"{SAVEDIR}/list_dfdist.pkl")
    with open(f"{SAVEDIR}/list_dfdist.pkl", "rb") as f:
        list_dfdist = pickle.load(f)

    if also_load_shuffled:
        with open(f"{SAVEDIR}/list_dfdist_shuff.pkl", "rb") as f:
            list_dfdist_shuff = pickle.load(f)
    else:
        list_dfdist_shuff = None

    if events_keep is not None:
        list_dfdist = [dfdist for dfdist in list_dfdist if dfdist["event"].unique()[0] in events_keep]
        if also_load_shuffled:
            list_dfdist_shuff = [dfdist for dfdist in list_dfdist_shuff if dfdist["event"].unique()[0] in events_keep]
    
    if len(list_dfdist)==0:
        if animal=="Pancho" and date==220614:
            # Then this date doesnt have SP. this could fail if you didnt also analyze PIG
            return None, None, None
        elif animal=="Diego" and date==231120:
            # Then this date doesnt have SP. this could fail if you didnt also analyze PIG
            return None, None, None
        else:
            assert False, "empty data..."
    
    if merge_pancho_ss_twinds and animal=="Pancho":
        # earlier days used shorter window, beucase hold perioud was shorter. Simply change the name,
        # so that can merge across dates.
        for df in list_dfdist:
            tmp = []
            for _, row in df.iterrows():
                if (row["subspace_projection_fitting_twind"] == (0.05, 0.9)) and (row["event"]=="03_samp"):
                    tmp.append(("post", "post"))
                elif (row["subspace_projection_fitting_twind"] == (0.05, 0.6)) and (row["event"]=="03_samp"):
                    tmp.append(("post", "post"))
                else:
                    tmp.append(row["subspace_projection_fitting_twind"])
            df["subspace_projection_fitting_twind"] = tmp

    # Now, events have multiple possible twind_scalar. to allow compatible with code,
    # rename events to events+scal.
    assert rename_event_with_eventtwind==True, "downstream code assumes this"
    if rename_event_with_eventtwind:
        for df in list_dfdist:
            df["event"] = df["event"] + "|"  + (df["twind_scal"].astype(str))

        if list_dfdist_shuff is not None:
            for df in list_dfdist_shuff:
                df["event"] = df["event"] + "|"  + (df["twind_scal"].astype(str))

    for df in list_dfdist:
        from pythonlib.tools.pandastools import replace_values_with_this
        replace_values_with_this(df, "subspace_projection", None, "none")

    if convert_to_df_with_postprocessing:
        ### (2) Prep dataset
        from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
        from pythonlib.tools.pandastools import append_col_with_grp_index, aggregGeneral
        import pandas as pd

        # First, generate all df
        print("concatting...")
        DFDISTS = pd.concat(list_dfdist).reset_index(drop=True)
        DFDISTS = datamod_reorder_by_bregion(DFDISTS)
        DFDISTS["animal"] = animal
        DFDISTS["date"] = date

        if "prune_version" not in DFDISTS:
            # Add a dummy variable
            DFDISTS["prune_version"] = "none"

        print("appending...")
        DFDISTS = append_col_with_grp_index(DFDISTS, ["subspace_projection", "subspace_projection_fitting_twind"], "subspace|twind")
        # DFDISTS = append_col_with_grp_index(DFDISTS, ["shuffled", "shuffle_ver"], "shuffled|ver")

        # Agg over the following variables (which index replicates):
        # - dim redu splits (dim_redu_fold)
        # - subsampled channels (chan_subsamp_i and chan_subsamp_chans)
        print("agging over replicates...")
        var_same_same = f"same-{var_effect}|{var_other}"
        if "n_1_2" in DFDISTS:
            DFDISTS["n1"] = [int(x[0]) for x in DFDISTS["n_1_2"]]
            DFDISTS["n2"] = [int(x[1]) for x in DFDISTS["n_1_2"]]
            values = ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98", "n1", "n2"]
            del DFDISTS["n_1_2"]
        else:
            values = ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"]
        DFDISTS = aggregGeneral(DFDISTS, ["animal", "date", var_same_same, "bregion", "prune_version", "which_level", "event", 
                                          "subspace_projection", "subspace_projection_fitting_twind", "subspace|twind", 
                                          "twind_scal", "labels_1", "labels_2"], values)

        # Agg all the "metaparams"
        from pythonlib.tools.pandastools import append_col_with_grp_index
        DFDISTS = append_col_with_grp_index(DFDISTS, ["prune_version", "subspace|twind", "event"], "metaparams")

        ### Do general postprocessing of DFDISTS
        from neuralmonkey.analyses.euclidian_distance import dfdist_postprocess_wrapper
        DFDISTS, _ = dfdist_postprocess_wrapper(DFDISTS, var_effect, var_other, savedir, 
            do_pruning_cleaning=DO_FURTHER_POSTPROCESSING, prune_min_n_trials=prune_min_n_trials)

        ## FINAL AGGS, over label pairs
        # (2) each datapt is 0|1, 1|0, 1|1, 0|0 (i.e., 4 datapts per bregion/metaparams)
        # Agg over all conditions (e.g. label pairs)
        print("agging over label pairs...")
        DFDISTS_AGG = aggregGeneral(DFDISTS, ["animal", "date", "metaparams", "bregion", "prune_version", 
                                              "which_level", "event", "subspace|twind", var_same_same],
                                    ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
        
        return DFDISTS, DFDISTS_AGG, SAVEDIR_PLOTS
    else:
        return list_dfdist, list_dfdist_shuff, SAVEDIR_PLOTS


def euclidian_time_resolved_fast_shuffled_mult_stats_v3(animal, date, var_other, 
        var_shuffle, var_stable, n_shuff = 10000, shuffle_method=2, events_keep=None,
        PLOT_SHUFFLE_HEATMAP=False, HACK=False):
    """
    Entire pipeline to do shuffle, cmopute p-values, and plot, using method that shuffles labels of the
    dist_norm distance matrix, and then recmputes dist_hyue_diff using the shuffled matreix.

    Uses the saved data from euclidian_time_resolved_fast_shuffled, just the data, not the shuffles.

    Note: This is a sound method.
    - Confirmed that recomputing dist_yue_diff here is ideitncal to computing for each (label1, label1) and
    then averaging.
    - Also, using dist_norm is correct, in that the nomralization will still be correct even if shuffle, since norm is based on 
    pairwise between all trials, which is not affected by shuffle.
    """
    from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_append_and_return_inner_items_good
    from pythonlib.tools.pandastools import aggregGeneral, stringify_values
    import pandas as pd
    import matplotlib.pyplot as plt

    assert shuffle_method==2, "code doesnt work for others now, mainy becuase v3 requires var_shuffle, var_stable"

    var_effect = "seqc_0_shape"
    var_same_same = f"same-{var_effect}|{var_other}"
    
    ### (1) Load all data
    list_dfdist, _, SAVEDIR_PLOTS = euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other, False,
        events_keep=events_keep)

    ### (2) Prep dataset
    # savedir = f"{SAVEDIR_PLOTS}/stats_method_3_using_shuffle_dist_mat-nshuff={n_shuff}"
    savedir = f"{SAVEDIR_PLOTS}/shuff_v3_distmat-meth={shuffle_method}-varshuff={var_shuffle}-varstab={var_stable}-nshuff={n_shuff}"
    os.makedirs(savedir, exist_ok=True)

    # First, generate all df
    print("concatting...")
    DFDISTS = pd.concat(list_dfdist).reset_index(drop=True)
    DFDISTS = datamod_reorder_by_bregion(DFDISTS)

    print("appending...")
    DFDISTS = append_col_with_grp_index(DFDISTS, ["subspace_projection", "subspace_projection_fitting_twind"], "subspace|twind")
    DFDISTS = append_col_with_grp_index(DFDISTS, ["shuffled", "shuffle_ver"], "shuffled|ver")

    print("agging...")

    # Agg over all dim redu splits
    DFDISTS = aggregGeneral(DFDISTS, ["bregion", "which_level", "event", "subspace|twind", "labels_1", "labels_2"], 
                            ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
    # DFDISTS = stringify_values(DFDISTS)
        
    def _recompute_dist_yue_diff(dfthis, list_ss):
        """
        dfthis:
        - Should have columns: same-seqc_0_shape|seqc_0_loc, dist_norm
        - each row is the mean for that level, or is trials (in which case will take mean over trials)
        list_ss = "0|0", "1|0", "0|1"
        """

        dist_11 = dfthis[dfthis[var_same_same] == "1|1"]["dist_norm"].values[0]
        # res = []
        dists =[]
        for ss in list_ss:
            dist_norm = dfthis[dfthis[var_same_same] == ss]["dist_norm"].values[0]
            dist_yue_diff = dist_norm - dist_11
            dists.append(dist_yue_diff)

        return dists

    ### (3) Run shuffle
    # shuffle labels (conjunction of shape and loc)
    assert isinstance(DFDISTS["labels_1"].values[0], tuple), "need to be able to decompose into shape and loc.. -- dont strigify"
    import random


    grp_vars = ["bregion", "which_level", "event", "subspace|twind", "shuffled|ver"]
    grpdict = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)

    from pythonlib.cluster.clustclass import Clusters
    cl = Clusters(None) # For methods

    res = []
    list_df =[]
    for grp, inds in grpdict.items():

        # if events_keep is not None:
        #     if not any([ev in grp[2] for ev in events_keep]): # beucase grp[2] will be events|twind_scal.
        #         print("skippig, as this event ", grp[2], " not in ", events_keep)
        #         continue

        dfthis = DFDISTS.iloc[inds].reset_index(drop=True)

        if shuffle_method==0:
            labels_pool = sorted(set(dfthis["labels_1"].unique().tolist() + dfthis["labels_2"].unique().tolist()))

        if shuffle_method==2:
            # (1) Convert to symmetrix full matrix.
            label_vars = ["seqc_0_shape", var_other]
            dfthis = cl.rsa_distmat_convert_from_triangular_to_full(dfthis, label_vars, PLOT=PLOT_SHUFFLE_HEATMAP)

        # make shuffles
        for i_shuff in range(n_shuff):
            
            # Make copies
            dfthis_shuff = dfthis.copy()

            # Shuffle
            if shuffle_method==0:
                # Label 1 and 2, keep theier correlations. --> leads to high dist_yue_diff
                # NEVER USED THIS
                assert False, "never used this..."
                labels_pool_shuff = [x for x in labels_pool]
                random.shuffle(labels_pool_shuff)
                map_lab_to_newlab = {lab:labnew for lab, labnew in zip(labels_pool, labels_pool_shuff)}
                dfthis_shuff["labels_1"] = [map_lab_to_newlab[lab] for lab in dfthis_shuff["labels_1"]]
                dfthis_shuff["labels_2"] = [map_lab_to_newlab[lab] for lab in dfthis_shuff["labels_2"]]
            elif shuffle_method==1:
                # RUN 1 -- PRoblem, PMv location is not signifiacnt.
                # Shuffle each label independently.
                
                assert False, "has problems, see slides ..."

                dfthis_shuff["labels_1"] = dfthis_shuff["labels_1"].sample(frac=1).reset_index(drop=True)
                dfthis_shuff["labels_2"] = dfthis_shuff["labels_2"].sample(frac=1).reset_index(drop=True)
                
                # reset the other columns

                # shape, loc
                dfthis_shuff["seqc_0_shape_1"] = [x[0] for x in dfthis_shuff["labels_1"]]
                dfthis_shuff["seqc_0_shape_2"] = [x[0] for x in dfthis_shuff["labels_2"]]
                dfthis_shuff[f"{var_other}_1"] = [x[1] for x in dfthis_shuff["labels_1"]]
                dfthis_shuff[f"{var_other}_2"] = [x[1] for x in dfthis_shuff["labels_2"]]

                # shape_same, loc_same
                dfthis_shuff["seqc_0_shape_same"] = dfthis_shuff["seqc_0_shape_1"] == dfthis_shuff["seqc_0_shape_2"]
                dfthis_shuff[f"{var_other}_same"] = dfthis_shuff[f"{var_other}_1"] == dfthis_shuff[f"{var_other}_2"]
                
                # both same
                dfthis_shuff = append_col_with_grp_index(dfthis_shuff, ["seqc_0_shape_same", f"{var_other}_same"], var_same_same)
            elif shuffle_method==2:
                # Run 2 -- shuffle var1 within each level of var2, and do so by remapping unique variables, thereby retainig
                # statistics of freqeuncy. This shoudl be good /final.
                # This requires separate runs for testing location and shape.

                from pythonlib.tools.pandastools import shuffle_dataset_hierarchical_remap
                # Do shuffle for labels 1 and 2.
                # This shuffles, e.g., seqc_0_shape.

                # Save original label
                if HACK:
                    vals_before_shuff_1 = dfthis_shuff[f"{var_shuffle}_1"].values
                    vals_before_shuff_2 = dfthis_shuff[f"{var_shuffle}_2"].values

                for i in [1,2]:
                    grouping_column = f"{var_stable}_{i}"
                    column_to_remap = f"{var_shuffle}_{i}"
                    use_same_mapping_across_groups = True # if False, then shuffle has low variance...
                    dfthis_shuff = shuffle_dataset_hierarchical_remap(dfthis_shuff, column_to_remap, grouping_column, 
                                                                      column_to_remap, use_same_mapping_across_groups=use_same_mapping_across_groups)

                if HACK:
                    # replace values, as sanity check --> expect the shuffle outcome to match the raw data
                    dfthis_shuff[f"{var_shuffle}_1"] = vals_before_shuff_1
                    dfthis_shuff[f"{var_shuffle}_2"] = vals_before_shuff_2

                # Re-aassign relational variables
                # shape_same, loc_same
                dfthis_shuff["seqc_0_shape_same"] = dfthis_shuff["seqc_0_shape_1"] == dfthis_shuff["seqc_0_shape_2"]
                dfthis_shuff[f"{var_other}_same"] = dfthis_shuff[f"{var_other}_1"] == dfthis_shuff[f"{var_other}_2"]

                # both same
                var_same_same = f"same-seqc_0_shape|{var_other}"
                dfthis_shuff = append_col_with_grp_index(dfthis_shuff, ["seqc_0_shape_same", f"{var_other}_same"], var_same_same)
            else:
                print(shuffle_method)
                assert False, "what is it"

            if PLOT_SHUFFLE_HEATMAP:
                # Plots in original locations (ie based on labels 1 and 2, which are not changed by shuffle).
                # shwong how the relations are changed byshuffle.
                grouping_plot_n_samples_conjunction_heatmap(dfthis, "labels_1", "labels_2", ["seqc_0_shape_same"]);
                grouping_plot_n_samples_conjunction_heatmap(dfthis_shuff, "labels_1", "labels_2", ["seqc_0_shape_same"]);
                grouping_plot_n_samples_conjunction_heatmap(dfthis, "labels_1", "labels_2", ["seqc_0_loc_same"]);
                grouping_plot_n_samples_conjunction_heatmap(dfthis_shuff, "labels_1", "labels_2", ["seqc_0_loc_same"]);
                grouping_plot_n_samples_conjunction_heatmap(dfthis, "labels_1", "labels_2", ["same-seqc_0_shape|seqc_0_loc"]);
                grouping_plot_n_samples_conjunction_heatmap(dfthis_shuff, "labels_1", "labels_2", ["same-seqc_0_shape|seqc_0_loc"]);

                # Also plot sample size of loc and shape conjucntiosn. Theses hould not chagne
                grouping_plot_n_samples_conjunction_heatmap(dfthis, "seqc_0_loc_1", "seqc_0_loc_2");
                grouping_plot_n_samples_conjunction_heatmap(dfthis, "seqc_0_shape_1", "seqc_0_shape_2");
                grouping_plot_n_samples_conjunction_heatmap(dfthis_shuff, "seqc_0_loc_1", "seqc_0_loc_2");
                grouping_plot_n_samples_conjunction_heatmap(dfthis_shuff, "seqc_0_shape_1", "seqc_0_shape_2");

                assert False, "cannot keep running, too many figs."

            if len(dfthis_shuff[var_same_same].unique().tolist())<4:
                print("Len < 4, skipping")
                assert False, "dont do this shuffle anymore. new version (run 2) sholdnt do this."

            if False:
                # Recompute dist_yue_diff [use this if want to recompute dyd]
                # Do two steps at once: agg across trials, and compute new dist_yue_diff (dist_norm, subtract 1|1 condition)
                from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping, datamod_normalize_row_after_grouping_return_same_len_df
                _, _, dflong_norm, _, _ = datamod_normalize_row_after_grouping(dfthis_shuff, 
                                                                                    var_same_same, 
                                                                                    grp_vars, "dist_norm", "1|1", False)
                dflong_norm["dist_yue_diff"] = dflong_norm["value"]
                dflong_norm["shuffle_iter"] = i_shuff
                dflong_norm["var_stable"] = var_stable
                dflong_norm["var_shuffle"] = var_shuffle
                display(dflong_norm)
                assert False
                list_df.append(dflong_norm)

            elif True:
                # Better
                # agg --> one value for each ss (01, ...) --> 4 rows
                # dftmp = aggregGeneral(dfthis_shuff, ["bregion", "which_level", "event", "subspace|twind", 
                #                             "seqc_0_loc_same", "subspace_projection", "subspace_projection_fitting_twind", 
                #                             "same-seqc_0_shape|seqc_0_loc"], 
                #                             ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff"])
                dftmp = aggregGeneral(dfthis_shuff, grp_vars + [var_same_same], 
                        ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff"])

                dftmp["dist_yue_diff"] = _recompute_dist_yue_diff(dftmp, list_ss = dftmp["same-seqc_0_shape|seqc_0_loc"].tolist())
                dftmp["shuffle_iter"] = i_shuff
                dftmp["var_stable"] = var_stable
                dftmp["var_shuffle"] = var_shuffle
                list_df.append(dftmp)
            else: # old method, slow
                # Compute scores
                _grpdict = grouping_append_and_return_inner_items_good(dfthis_shuff, ["same-seqc_0_shape|seqc_0_loc"])
                
                for _grp, _inds in _grpdict.items():
                    res.append({
                        "shuffle_iter":i_shuff,
                        "same-seqc_0_shape|seqc_0_loc":_grp[0],
                        "dist_mean":dfthis_shuff.iloc[_inds]["dist_mean"].mean(),
                        "dist_norm":dfthis_shuff.iloc[_inds]["dist_norm"].mean(),
                        "dist_yue_diff":dfthis_shuff.iloc[_inds]["dist_yue_diff"].mean(),
                    })
                    for var, val in zip(grp_vars, grp):
                        res[-1][var] = val
    if True:
        # new, which does recomputation of dist_yue above
        DFSTATS = pd.concat(list_df).reset_index(drop=True)
        # DFSTATS["dist_yue_diff"] = DFSTATS["value"]
    else:
        # Old, which here does recomputation of dist_yue above
        DFSTATS = pd.DataFrame(res)
        DFSTATS


        # Recompte dist_yue_diff for each shuffle rendition, as this forms the null distribution.
        grp_vars = ["bregion", "which_level", "event", "subspace|twind", "shuffled|ver", "shuffle_iter"]
        grpdict = grouping_append_and_return_inner_items_good(DFSTATS, grp_vars)

        res = []
        list_df =[]
        for _i, (grp, inds) in enumerate(grpdict.items()):
            dfthis = DFSTATS.iloc[inds].reset_index(drop=True)
            if len(dfthis)>4:
                print(len(dfthis))
                assert False, "the 4 csaes of 00, 01, ..."
            
            dists = _recompute_dist_yue_diff(dfthis, list_ss = dfthis["same-seqc_0_shape|seqc_0_loc"].tolist())
            
            # Replace
            dfthis["dist_yue_diff"] = dists

            list_df.append(dfthis)

    ### (4) Save and plot
    ## FINAL AGGS
    # (2) each datapt is 0|1, 1|0, 1|1, 0|0 (i.e., 4 datapts per bregion/metaparams)
    # Agg over all conditions (e.g. label pairs)
    DFDISTS_AGG = aggregGeneral(DFDISTS, ["bregion", "which_level", "event", "subspace|twind", var_same_same],
                                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")

    # Save it
    pd.to_pickle(DFSTATS, f"{savedir}/DFSTATS.pkl")
    pd.to_pickle(DFDISTS, f"{savedir}/DFDISTS.pkl")
    pd.to_pickle(DFDISTS_AGG, f"{savedir}/DFDISTS_AGG.pkl")

    savedir_tmp = savedir
    for var_value in ["dist_yue_diff", "dist_norm"]:
        # Compute p-values by comparing shuff vs. dat (empriical pval)

        savedir = f"{savedir_tmp}/var_value={var_value}"
        os.makedirs(savedir, exist_ok=True)

        grp_vars = ["bregion", "which_level", "event", "subspace|twind", "shuffled|ver", var_same_same]
        grpdict_shuff = grouping_append_and_return_inner_items_good(DFSTATS, grp_vars)
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)

        res =[]
        for grp_dat, inds_dat in grpdict_dat.items():
            inds_shuff = grpdict_shuff[grp_dat]

            assert len(inds_shuff)>0.9*n_shuff, " is ok if not 100%, beucase of line that does print(Len < 4, skipping) above"
            
            val_dat = DFDISTS.iloc[inds_dat][var_value].mean()
            vals_shuff = DFSTATS.iloc[inds_shuff][var_value].values

            from pythonlib.tools.statstools import empiricalPval
            p = empiricalPval(val_dat, vals_shuff)

            # Also compute z-score
            shuff_mean = np.mean(vals_shuff)
            shuff_std = np.std(vals_shuff)
            if shuff_std>0:
                zscore = (val_dat - shuff_mean)/shuff_std
            else:
                zscore = 0
            # - Convert from z-score to pvalue
            from scipy.stats import norm
            p_zscore = 1 - norm.cdf(zscore)  # For one-tailed (right-tailed) test

            # print(val_dat, vals_shuff)
            res.append({
                "p":p,
                "zscore":zscore,
                "p_zscore":p_zscore,
                "log_p_zscore":np.log10(p_zscore)
            })
            for _var, _val in zip(grp_vars, grp_dat):
                res[-1][_var]=_val

        #### Quick plot of p-values
        dfpvalues = pd.DataFrame(res)
        dfpvalues["log_p"] = np.log10(dfpvalues["p"])

        fig = sns.catplot(data=dfpvalues, x="bregion", y="log_p", row="event", col="subspace|twind", 
                        hue=var_same_same, kind="bar")
        for ax in fig.axes.flatten():
            ax.axhline(np.log10(0.05))
            ax.axhline(np.log10(0.001))
        savefig(fig, f"{savedir}/catplot-pvalues.pdf")

        fig = sns.catplot(data=dfpvalues, x="bregion", y="zscore", row="event", col="subspace|twind", 
                        hue=var_same_same, kind="bar")
        savefig(fig, f"{savedir}/catplot-zscore.pdf")        

        fig = sns.catplot(data=dfpvalues, x="bregion", y="log_p_zscore", row="event", col="subspace|twind", 
                        hue=var_same_same, kind="bar")
        for ax in fig.axes.flatten():
            ax.axhline(np.log10(0.05))
            ax.axhline(np.log10(0.001))
        savefig(fig, f"{savedir}/catplot-zscore-pvalues.pdf")
        
        plt.close("all")

        if False: # Done better below
            # Plot the shuffle distribtiions
            # sns.catplot(data=DFSTATS, x="bregion", y="dist_norm", hue="same-seqc_0_shape|seqc_0_loc", row="event", col="subspace|twind", alpha=0.5, jitter=True)
            sns.catplot(data=DFSTATS, x="bregion", y=var_value, hue="same-seqc_0_shape|seqc_0_loc", row="event", col="subspace|twind", alpha=0.5, jitter=True)
            # Plot the data distribtuiosn
            # sns.catplot(data=DFDISTS, x="bregion", y="dist_norm", hue="same-seqc_0_shape|seqc_0_loc", row="event", col="subspace|twind", kind="bar")
            sns.catplot(data=DFDISTS, x="bregion", y=var_value, hue="same-seqc_0_shape|seqc_0_loc", row="event", col="subspace|twind", kind="bar")
        from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
        order_bregion = _REGIONS_IN_ORDER_COMBINED
        list_bregion = order_bregion

        # TODO: Plot p value over catplot distribtiosn (see other code).
        from pythonlib.tools.statstools import plotmod_pvalues
        from pythonlib.tools.plottools import rotate_x_labels

        # grp_vars = ["bregion", "which_level", "event", "subspace|twind", "same-seqc_0_shape|seqc_0_loc"]
        if True: # Figure is too large, use violin plot below.
            grp_vars = ["which_level", "event", "subspace|twind", var_same_same]
            grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)
            grpdict_shuff = grouping_append_and_return_inner_items_good(DFSTATS, grp_vars)
            grpdict_pval = grouping_append_and_return_inner_items_good(dfpvalues, grp_vars)

        # Plot distributions, and overlay the datapt.
            # ncols = 6
            # nrows = int(np.ceil(len(grpdict_dat)/ncols))
            W = 4
            H = 3
            # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
            # for ax, (grp, inds_dat) in zip(axes.flatten(), grpdict_dat.items()):
            for grp, inds_dat in grpdict_dat.items():
                fig, ax = plt.subplots(1, 1, figsize=(W, H))

                inds_shuff = grpdict_shuff[grp]
                inds_pval = grpdict_pval[grp]

                dfthis_dat = DFDISTS_AGG.iloc[inds_dat]
                dfthis_shuff = DFSTATS.iloc[inds_shuff]
                dfthis_pval = dfpvalues.iloc[inds_pval]

                sns.stripplot(data=dfthis_shuff, x="bregion", y=var_value, ax=ax, order=order_bregion, alpha=0.1, jitter=True)
                sns.stripplot(data=dfthis_dat, x="bregion", y=var_value, ax=ax, order=order_bregion)
                
                # Overlay p value
                def _get_p(dfthis_pval, bregion):
                    return dfthis_pval[dfthis_pval["bregion"]==bregion]["p"].values[0]
                list_p = [_get_p(dfthis_pval, bregion) for bregion in order_bregion]    
                xs = ax.get_xticks()
                assert len(xs)==len(list_p)
                plotmod_pvalues(ax, xs, list_p)

                # Pretty plot
                ax.axhline(0, color="k", alpha=0.5)
                ax.set_title(grp, fontsize=8)
                rotate_x_labels(ax)
                savefig(fig, f"{savedir}/summary-1-shuff_vs_dat-grp={grp}.pdf")

                # savefig(fig, f"{savedir}/summary-1-shuff_vs_dat.pdf")
                plt.close("all")

        # Final catplot, with 
        # TODO: Plot p value over catplot distribtiosn (see other code).
        from pythonlib.tools.statstools import plotmod_pvalues
        from pythonlib.tools.plottools import rotate_x_labels

        grp_vars = ["which_level", "event", "subspace|twind"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)
        grpdict_shuff = grouping_append_and_return_inner_items_good(DFSTATS, grp_vars)
        grpdict_pval = grouping_append_and_return_inner_items_good(dfpvalues, grp_vars)

        # Plot distributions, and overlay the datapt.
        ncols = 4
        nrows = int(np.ceil(len(grpdict_dat)/ncols))
        W = 8
        H = 5
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
        for ax, (grp, inds_dat) in zip(axes.flatten(), grpdict_dat.items()):
            inds_shuff = grpdict_shuff[grp]
            inds_pval = grpdict_pval[grp]

            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]
            dfthis_shuff = DFSTATS.iloc[inds_shuff]
            dfthis_pval = dfpvalues.iloc[inds_pval]

            sns.violinplot(data=dfthis_shuff, x="bregion", y=var_value, hue=var_same_same,
                        ax=ax, order=order_bregion, alpha=0.1, jitter=True)
            sns.stripplot(data=dfthis_dat, x="bregion", y=var_value,  hue=var_same_same,
                        ax=ax, order=order_bregion, legend=False)
            
            # Overlay p value
            def _get_p(dfthis_pval, bregion, same_shape_loc):
                return dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]==same_shape_loc)]["p"].values[0]
            for y_loc_frac, same_shape_loc in zip([0.6, 0.7, 0.8], ["0|1", "1|0", "0|0"]):
                list_p = [_get_p(dfthis_pval, bregion, same_shape_loc) for bregion in order_bregion]    
                xs = ax.get_xticks()
                assert len(xs)==len(list_p)
                plotmod_pvalues(ax, xs, list_p, y_loc_frac=y_loc_frac, prefix=same_shape_loc)

            # Pretty plot
            ax.axhline(0, color="k", alpha=0.5)
            ax.set_title(grp, fontsize=8)
            rotate_x_labels(ax)
        savefig(fig, f"{savedir}/summary-2-shuff_vs_dat-violin.pdf")
        plt.close("all")

        # Final catplot, clean, showing just a line for shuffle distribtuion

        # TODO: Plot p value over catplot distribtiosn (see other code).
        from pythonlib.tools.statstools import plotmod_pvalues
        from pythonlib.tools.plottools import rotate_x_labels

        grp_vars = ["which_level", "event", "subspace|twind"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)
        grpdict_shuff = grouping_append_and_return_inner_items_good(DFSTATS, grp_vars)
        grpdict_pval = grouping_append_and_return_inner_items_good(dfpvalues, grp_vars)

        # Bonferonni - n brain regions.
        alpha=0.05
        ncomparisons = len(order_bregion)
        pval_bonf = alpha/ncomparisons

        pval_ver = "empirical"
        # var_value_ver = var_value
        for plot_pval_text, pval_thresh in [
            (False, alpha), (True, alpha), (True, 0.005), (True, pval_bonf), (True, 0.0005)]:
            # Plot distributions, and overlay the datapt.
            ncols = 4
            nrows = int(np.ceil(len(grpdict_dat)/ncols))
            W = 8
            H = 5
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
            for ax, (grp, inds_dat) in zip(axes.flatten(), grpdict_dat.items()):
                inds_shuff = grpdict_shuff[grp]
                inds_pval = grpdict_pval[grp]

                dfthis_dat = DFDISTS_AGG.iloc[inds_dat]
                dfthis_shuff = DFSTATS.iloc[inds_shuff]
                dfthis_pval = dfpvalues.iloc[inds_pval]

                sns.barplot(data=dfthis_dat, x="bregion", y=var_value,  hue=var_same_same,
                            ax=ax, order=order_bregion, hue_order=["0|1", "1|0"])
                
                # Overlay p value
                def _get_p(dfthis_pval, bregion, same_shape_loc, pval_ver):
                    if pval_ver=="empirical":
                        p = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]==same_shape_loc)]["p"].values[0]
                    elif pval_ver=="zscore":
                        p = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]==same_shape_loc)]["p_zscore"].values[0]
                    return p
                
                # for y_loc_frac, same_shape_loc in zip([0.6, 0.7, 0.8], ["0|1", "1|0", "0|0"]):
                if plot_pval_text:
                    for y_loc_frac, same_shape_loc in zip([0.6, 0.7], ["0|1", "1|0"]):
                        list_p = [_get_p(dfthis_pval, bregion, same_shape_loc, pval_ver) for bregion in order_bregion]    
                        xs = ax.get_xticks()
                        assert len(xs)==len(list_p)
                        plotmod_pvalues(ax, xs, list_p, y_loc_frac=y_loc_frac, prefix=same_shape_loc, pthresh=pval_thresh)

                # get 97.5 percentile for each shuffle distribution
                for x, bregion in enumerate(order_bregion):
                    for same_shape_loc in ["0|1", "1|0"]:
                        dftmp = dfthis_shuff[(dfthis_shuff["bregion"]==bregion) & (dfthis_shuff[var_same_same]==same_shape_loc)]
                        
                        shuff_thresh = np.percentile(dftmp[var_value], 100*(1-(pval_thresh/2))) # the alpha level

                        ax.plot([x-0.5, x+0.5], [shuff_thresh, shuff_thresh], "--k", alpha=0.5)
                        if plot_pval_text:
                            ax.text(x-0.5, shuff_thresh, same_shape_loc, fontsize=6)

                # Pretty plot
                ax.axhline(0, color="k", alpha=0.5)
                ax.set_title(grp, fontsize=8)
                rotate_x_labels(ax)

            savefig(fig, f"{savedir}/summary-3-dat_bar-plottext={plot_pval_text}-pvalthresh={pval_thresh}.pdf")
        plt.close("all")

        #################### SCATTER, COLORING BY P-VALUE.
        from pythonlib.tools.plottools import color_make_map_discrete_labels
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
        pstatuses = ["shape_loc", "shape", "loc", "none"]
        grp_vars = ["which_level", "event", "subspace|twind"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)
        grpdict_pval = grouping_append_and_return_inner_items_good(dfpvalues, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]
            dfthis_pval = dfpvalues.iloc[inds_pval]

            for pval_thresh in [alpha, 0.005, pval_bonf, 0.0005]:

                ### For each region, determine if it is signif for shape, loc, or shapeandloc
                map_bregion_pval_status = {}
                for bregion in order_bregion:
                    p_shape = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="0|1")]["p"].values[0]
                    p_loc = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="1|0")]["p"].values[0]

                    if (p_shape<pval_thresh) and (p_loc<pval_thresh):
                        p_status = "shape_loc"
                    elif (p_shape<pval_thresh):
                        p_status = "shape"
                    elif (p_loc<pval_thresh):
                        p_status = "loc"
                    else:
                        p_status = "none"
                    map_bregion_pval_status[bregion] = p_status

                ### Plot
                _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "subspace|twind", 
                                                    var_value, "bregion", True, shareaxes=True,
                                                    map_datapt_lev_to_colorlev=map_bregion_pval_status, colorlevs_that_exist=pstatuses)
                
                savefig(fig, f"{savedir}/scatter45-grp={grp}-pvalthresh={pval_thresh}.pdf")
                plt.close("all")

def _euclidian_time_resolved_fast_shuffled_mult_scatter_plots_params(analysis_kind, animal, var_other):
    """
    Repo of dates
    """
    if analysis_kind in ["shape_invar_clean_loc", "shape_invar_clean_size", "shape_invar_clean"]:
        # Then these were either only (shape, loc) or (shape, size), not (shape, loc, size)
        if animal == "Diego" and var_other == "seqc_0_loc":
            list_date = [230614, 230615, 240508]
        elif animal == "Diego" and var_other == "gridsize":
            list_date = [230618, 230619]
            # list_date = [230618, 230619, 240530]
        elif animal == "Pancho" and var_other == "seqc_0_loc":
            list_date = [220715, 220724]                
        elif animal == "Pancho" and var_other == "gridsize":
            # list_date = [220716, 220717]
            list_date = [220716, 220717, 240530]
        else:
            assert False
    elif analysis_kind in ["shape_invar", "shape_invar_context"]:
        if animal == "Diego" and var_other == "seqc_0_loc":
            # list_date = [230614, 230615]
            list_date = [230614, 230615, 240508, 240509, 240510, 240513, 240530]
        elif animal == "Diego" and var_other == "gridsize":
            # list_date = [230618, 230619]
            list_date = [230618, 230619, 240510, 240530]
        elif animal == "Pancho" and var_other == "seqc_0_loc":
            # list_date = [220608, 220715, 220717]
            # list_date = [220606, 220608, 220609, 220610, 220715, 220717, 220724, 220918, 221218, 240508, 240509, 240510, 240515, 240530]                
            # list_date = [220715, 220717, 220724, 220918, 240508, 240509, 240510, 240530]                
            list_date = [220606, 220608, 220715, 220717, 220724, 220918, 221218, 240508, 240509, 240510, 240515, 240530]                
        elif animal == "Pancho" and var_other == "gridsize":
            # list_date = [220716, 220717]
            list_date = [220606, 220608, 220716, 220717, 220918, 240510, 240530]
        else:
            assert False
    elif analysis_kind in ["char_sp", "char_sp_04_go_cue", "char_sp_05_first_raise", "char_sp_00_stroke",
                           "char_sp_00_stroke_revision_noregr", "char_sp_00_stroke_revision_regr", 
                           "char_sp_00_stroke_revision_noregr_seman", "char_sp_00_stroke_revision_regr_seman"]:
        if animal == "Diego":
            # list_date = [231205, 231122, 231128, 231129, 231201, 231120, 231121, 231206, 231218, 231220] # A date fails.
            list_date = [231205, 231122, 231128, 231129, 231201, 231120, 231206, 231218, 231220]
            # list_date = [231218, 231220] # Just for quick testing.
        elif animal == "Pancho":
            if False:
                # All dates
                list_date_1 = [220618, 220626, 220628, 220630, 230119, 230120, 230126, 230127]
                list_date_2 = [220614, 220616, 220621, 220622, 220624, 220627, 230112, 230117, 230118]
                list_date = list_date_1 + list_date_2
            else:
                # 2022 dates, good PMv signal
                HACK = False
                if HACK:
                    # 220614 doesnt have SP, this failed for revision plots beucase dfdist overwrote... just temporary to see plots.
                    list_date = [220616, 220621, 220622, 220624, 220627, 220618, 220626, 220628, 220630]
                else:
                    list_date = [220614, 220616, 220618, 220621, 220622, 220624, 220626, 220627, 220628, 220630]

        else:
            assert False
    # elif analysis_kind in ["char_sp_00_stroke_revision_noregr", "char_sp_00_stroke_revision_regr", 
    #                        "char_sp_00_stroke_revision_noregr_seman", "char_sp_00_stroke_revision_regr_seman"]:
    #     if animal == "Diego":
    #         # list_date = [231205, 231122, 231128, 231129, 231201, 231120, 231121, 231206, 231218, 231220] # A date fails.
    #         list_date = [231205, 231122, 231128, 231129, 231201, 231120, 231206, 231218, 231220]
    #         # list_date = [231129, 231120]
    #     elif animal == "Pancho":
    #         # 220614 doesnt have SP, this failed for revision plots beucase dfdist overwrote... just temporary to see plots.
    #         list_date = [220614, 220616, 220618, 220621, 220622, 220624, 220626, 220627, 220628, 220630]
    #     else:
    #         assert False
    else:
        print(analysis_kind)
        assert False
    return list_date

def euclidian_time_resolved_fast_shuffled_mult_scatter_plots_manuscriptshape(analysis_kind="shape_invar", just_return_df=False, 
                                                             DO_FURTHER_POSTPROCESSING=False, PLOT_EACH_PAIR=False,
                                                             prune_min_n_trials=None,
                                                             list_metaparams_plot_each_pair=None,
                                                             do_catplots=False, SAVEDIR_BASE=None):
    """
    GOOD -- Main wrapper to load and plot across days, without bothering about shuffle stats.
    Loads data saved by euclidian_time_resolved_fast_shuffled, across dates, and concats, and plots.

    PARAMS:
    - analysis_kind, "shape_invar", "char_sp"

    Works for both (i) shape invar and (ii) char_sp. 
    Goal is for this to work for anything. 

    RETURNS:
    - DFDISTS, each datapt is a label pair (agged over trials and replicates (chan subsets and trial subsets)
    - DFDISTS_AGG, each datapt is a single condition pair (e.g., same|diff) agged over labels.

    MS: checked
    """
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import euclidian_time_resolved_fast_shuffled_mult_reload
    import os
    from pythonlib.tools.pandastools import aggregGeneral
    import pandas as pd
    import matplotlib.pyplot as plt
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    
    # if manuscript_version and analysis_kind=="shape_invar_clean_loc":
    #     SAVEDIR_MULT = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF/MULT"
    if SAVEDIR_BASE is None:
        SAVEDIR_BASE = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF"

    if analysis_kind=="shape_invar_clean_loc":
        SAVEDIR_MULT = f"{SAVEDIR_BASE}/MULT"
        var_effect = "seqc_0_shape"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "seqc_0_loc"),
            ("Diego", "seqc_0_loc"),
            ]
    elif analysis_kind=="shape_invar_clean_size":
        SAVEDIR_MULT = f"{SAVEDIR_BASE}/MULT"
        var_effect = "seqc_0_shape"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "gridsize"),
            ("Diego", "gridsize"),
        ]
    else:
        print(analysis_kind)
        assert False

    if just_return_df:
        # Just take one -- this is for debugging.
        LIST_ANIMAL_VAROTHER = LIST_ANIMAL_VAROTHER[:1]

    for animal, var_other in LIST_ANIMAL_VAROTHER:
        if animal == "Diego" and var_other == "seqc_0_loc":
            list_date = [230614, 230615, 240508]
        elif animal == "Diego" and var_other == "gridsize":
            list_date = [230618, 230619]
        elif animal == "Pancho" and var_other == "seqc_0_loc":
            list_date = [220715, 220724]                
        elif animal == "Pancho" and var_other == "gridsize":
            list_date = [220716, 220717, 240530]
        else:
            assert False

        SAVEDIR = f"{SAVEDIR_MULT}/analy={analysis_kind}-{animal}-var={var_effect}-varother={var_other}-{min(list_date)}-{max(list_date)}-post={DO_FURTHER_POSTPROCESSING}-prunen={prune_min_n_trials}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(SAVEDIR)

        # Collect all dates data
        list_df_raw = []
        list_df = []
        for date in list_date:
            print("Trying to load this animal, var_other, date: ", animal, var_other, date)
            counts_savedir = f"{SAVEDIR}/postprocessing"
            dfdists, dfdists_agg, _ = euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other=var_other, 
                                                                                        analysis_kind=analysis_kind,
                                                                                  convert_to_df_with_postprocessing=True, 
                                                                                  savedir=counts_savedir,
                                                                                  DO_FURTHER_POSTPROCESSING=DO_FURTHER_POSTPROCESSING,
                                                                                  prune_min_n_trials=prune_min_n_trials,
                                                                                  SAVEDIR_BASE=SAVEDIR_BASE)
            
            if dfdists is None:
                print("Skipping this animal date (empty data):", animal, date)
            else:
                dfdists["animal"] = animal
                dfdists["date"] = date
                dfdists_agg["animal"] = animal
                dfdists_agg["date"] = date

                list_df_raw.append(dfdists)
                list_df.append(dfdists_agg)

        DFDISTS = pd.concat(list_df_raw).reset_index(drop=True)
        DFDISTS_AGG = pd.concat(list_df).reset_index(drop=True)

        # SAVE COUNTS
        # assert False, "use the plotting function in euclidian_distance.py"
        for var in ["prune_version", "subspace|twind", "event"]:
            fig = grouping_plot_n_samples_conjunction_heatmap(DFDISTS_AGG, "date", var, ["animal"])
            savefig(fig, f"{SAVEDIR}/groupcounts-var={var}.pdf")
            plt.close("all")

        if just_return_df:
            # return DFDISTS, DFDISTS_AGG, SAVEDIR_PLOTS # NO, since SAVEDIR_PLOTS is for one day
            return DFDISTS, DFDISTS_AGG

        ### Plot summaries
        from neuralmonkey.analyses.euclidian_distance import dfdist_summary_plots_wrapper
        dfdist_summary_plots_wrapper(DFDISTS, DFDISTS_AGG, var_effect, var_other, SAVEDIR,
                                 PLOT_EACH_PAIR=PLOT_EACH_PAIR, 
                                 list_metaparams_plot_each_pair=list_metaparams_plot_each_pair,
                                 do_catplots=do_catplots)
                                 
        ### Plot distribution for each shapes
        from pythonlib.tools.pandastools import aggregGeneral
        from neuralmonkey.analyses.euclidian_distance import dfdist_summary_plots_wrapper
        shapes = sorted(set(DFDISTS[f"{var_effect}_1"].tolist() + DFDISTS[f"{var_effect}_2"].tolist()))
        var_same_same = f"same-{var_effect}|{var_other}"
        for sh in shapes:
            dfdists = DFDISTS[(DFDISTS[f"{var_effect}_1"]==sh) | (DFDISTS[f"{var_effect}_2"]==sh)].reset_index(drop=True)
            dfdists_agg = aggregGeneral(dfdists, ["animal", "date", "metaparams", "bregion", "prune_version", 
                                                    "which_level", "event", "subspace|twind", var_same_same],
                                        ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
                
            savedir = f"{SAVEDIR}/EACH_SHAPE/shape={sh}"
            os.makedirs(savedir, exist_ok=True)
            do_catplots = False
            do_quick = True
            dfdist_summary_plots_wrapper(dfdists, dfdists_agg, var_effect, var_other, SAVEDIR=savedir,
                                        PLOT_EACH_PAIR=False, do_catplots=do_catplots, do_quick=do_quick)
            plt.close("all")

def euclidian_time_resolved_fast_shuffled_mult_scatter_plots(analysis_kind="shape_invar", just_return_df=False, 
                                                             DO_FURTHER_POSTPROCESSING=False, PLOT_EACH_PAIR=False,
                                                             prune_min_n_trials=None,
                                                             list_metaparams_plot_each_pair=None,
                                                             do_catplots=False, SAVEDIR_BASE=None):
    """
    GOOD -- Main wrapper to load and plot across days, without bothering about shuffle stats.
    Loads data saved by euclidian_time_resolved_fast_shuffled, across dates, and concats, and plots.

    PARAMS:
    - analysis_kind, "shape_invar", "char_sp"

    Works for both (i) shape invar and (ii) char_sp. 
    Goal is for this to work for anything. 

    RETURNS:
    - DFDISTS, each datapt is a label pair (agged over trials and replicates (chan subsets and trial subsets)
    - DFDISTS_AGG, each datapt is a single condition pair (e.g., same|diff) agged over labels.
    """
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import euclidian_time_resolved_fast_shuffled_mult_reload
    import os
    from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_append_and_return_inner_items_good
    from pythonlib.tools.pandastools import aggregGeneral, stringify_values
    import pandas as pd
    import matplotlib.pyplot as plt
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.pandastools import plot_subplots_heatmap, sort_by_two_columns_separate_keys

    # if manuscript_version and analysis_kind=="shape_invar_clean_loc":
    #     SAVEDIR_MULT = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF/MULT"
    if SAVEDIR_BASE is None:
        SAVEDIR_BASE = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF"

    if analysis_kind=="shape_invar_clean_loc":
        SAVEDIR_MULT = f"{SAVEDIR_BASE}/MULT"
        var_effect = "seqc_0_shape"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "seqc_0_loc"),
            ("Diego", "seqc_0_loc"),
            ]
    elif analysis_kind=="shape_invar_clean_size":
        SAVEDIR_MULT = f"{SAVEDIR_BASE}/MULT"
        var_effect = "seqc_0_shape"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "gridsize"),
            ("Diego", "gridsize"),
            ]
    elif analysis_kind=="shape_invar":
        SAVEDIR_MULT = f"{SAVEDIR_BASE}/MULT"
        var_effect = "seqc_0_shape"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "seqc_0_loc"),
            ("Pancho", "gridsize"),
            ("Diego", "seqc_0_loc"),
            ("Diego", "gridsize"),
            ]
    elif analysis_kind=="shape_invar_context":
        SAVEDIR_MULT = f"{SAVEDIR_BASE}/MULT"
        var_effect = "seqc_0_shape"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "seqc_0_loc"),
            ("Pancho", "gridsize"),
            ("Diego", "seqc_0_loc"),
            ("Diego", "gridsize"),
            ]
    elif analysis_kind == "char_sp":
        SAVEDIR_MULT = "/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE/MULT"
        var_effect = "shape_semantic_grp"
        LIST_ANIMAL_VAROTHER = [
            # ("Pancho", "task_kind"),
            ("Diego", "task_kind"),
            ]
    elif analysis_kind in ["char_sp_05_first_raise", "char_sp_04_go_cue", "char_sp_00_stroke"]:
        SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE/MULT/{analysis_kind}"
        var_effect = "shape_semantic_grp"
        LIST_ANIMAL_VAROTHER = [
            ("Diego", "task_kind"),
            ("Pancho", "task_kind"),
            ]
    elif analysis_kind in ["char_sp_00_stroke_revision_regr", "char_sp_00_stroke_revision_noregr"]:
        assert False, "not using these anymore.."
        var_effect = "shape_semantic_grp"
        SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE_revision-VAR_SHAPE={var_effect}/MULT/{analysis_kind}"
        # SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE_revision/MULT/{analysis_kind}"
        LIST_ANIMAL_VAROTHER = [
            ("Pancho", "task_kind"),
            ("Diego", "task_kind"),
            ]
    elif analysis_kind in ["char_sp_00_stroke_revision_regr_seman", "char_sp_00_stroke_revision_noregr_seman"]:
        var_effect = "shape_semantic"
        SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE_revision-VAR_SHAPE={var_effect}/MULT/{analysis_kind}"
        # SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE_revision/MULT/{analysis_kind}"
        LIST_ANIMAL_VAROTHER = [
            ("Diego", "task_kind"),
            ("Pancho", "task_kind"),
            ]
    else:
        print(analysis_kind)
        assert False

    if just_return_df:
        # Just take one -- this is for debugging.
        LIST_ANIMAL_VAROTHER = LIST_ANIMAL_VAROTHER[:1]
        # assert len(LIST_ANIMAL_VAROTHER)==1, "this will take only the first itme..."

    for animal, var_other in LIST_ANIMAL_VAROTHER:
        list_date = _euclidian_time_resolved_fast_shuffled_mult_scatter_plots_params(analysis_kind, animal, var_other)

        SAVEDIR = f"{SAVEDIR_MULT}/analy={analysis_kind}-{animal}-var={var_effect}-varother={var_other}-{min(list_date)}-{max(list_date)}-post={DO_FURTHER_POSTPROCESSING}-prunen={prune_min_n_trials}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(SAVEDIR)

        list_df_raw = []
        list_df = []
        for date in list_date:
            print("Trying to load this animal, var_other, date: ", animal, var_other, date)
            counts_savedir = f"{SAVEDIR}/postprocessing"
            dfdists, dfdists_agg, _ = euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other=var_other, 
                                                                                        analysis_kind=analysis_kind,
                                                                                  convert_to_df_with_postprocessing=True, 
                                                                                  savedir=counts_savedir,
                                                                                  DO_FURTHER_POSTPROCESSING=DO_FURTHER_POSTPROCESSING,
                                                                                  prune_min_n_trials=prune_min_n_trials,
                                                                                  SAVEDIR_BASE=SAVEDIR_BASE)
            
            if dfdists is None:
                print("Skipping this animal date (empty data):", animal, date)
            else:
                dfdists["animal"] = animal
                dfdists["date"] = date
                dfdists_agg["animal"] = animal
                dfdists_agg["date"] = date

                list_df_raw.append(dfdists)
                list_df.append(dfdists_agg)

        DFDISTS = pd.concat(list_df_raw).reset_index(drop=True)
        DFDISTS_AGG = pd.concat(list_df).reset_index(drop=True)

        # NOw is all done within euclidian_time_resolved_fast_shuffled_mult_reload()
        # # Agg all the "metaparams"
        # from pythonlib.tools.pandastools import append_col_with_grp_index
        # DFDISTS_AGG = append_col_with_grp_index(DFDISTS_AGG, ["prune_version", "subspace|twind", "event"], "metaparams")
        # DFDISTS = append_col_with_grp_index(DFDISTS, ["prune_version", "subspace|twind", "event"], "metaparams")

        # SAVE COUNTS
        # assert False, "use the plotting function in euclidian_distance.py"
        for var in ["prune_version", "subspace|twind", "event"]:
            fig = grouping_plot_n_samples_conjunction_heatmap(DFDISTS_AGG, "date", var, ["animal"])
            savefig(fig, f"{SAVEDIR}/groupcounts-var={var}.pdf")
            plt.close("all")

        if just_return_df:
            # return DFDISTS, DFDISTS_AGG, SAVEDIR_PLOTS # NO, since SAVEDIR_PLOTS is for one day
            return DFDISTS, DFDISTS_AGG

        ### Plot summaries
        from neuralmonkey.analyses.euclidian_distance import dfdist_summary_plots_wrapper
        dfdist_summary_plots_wrapper(DFDISTS, DFDISTS_AGG, var_effect, var_other, SAVEDIR,
                                 PLOT_EACH_PAIR=PLOT_EACH_PAIR, 
                                 list_metaparams_plot_each_pair=list_metaparams_plot_each_pair,
                                 do_catplots=do_catplots)
                                 
        ### Plot distribution for each shapes
        from pythonlib.tools.pandastools import aggregGeneral
        from neuralmonkey.analyses.euclidian_distance import dfdist_summary_plots_wrapper
        shapes = sorted(set(DFDISTS[f"{var_effect}_1"].tolist() + DFDISTS[f"{var_effect}_2"].tolist()))
        var_same_same = f"same-{var_effect}|{var_other}"
        for sh in shapes:
            dfdists = DFDISTS[(DFDISTS[f"{var_effect}_1"]==sh) | (DFDISTS[f"{var_effect}_2"]==sh)].reset_index(drop=True)
            dfdists_agg = aggregGeneral(dfdists, ["animal", "date", "metaparams", "bregion", "prune_version", 
                                                    "which_level", "event", "subspace|twind", var_same_same],
                                        ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
                
            savedir = f"{SAVEDIR}/EACH_SHAPE/shape={sh}"
            os.makedirs(savedir, exist_ok=True)
            do_catplots = False
            do_quick = True
            dfdist_summary_plots_wrapper(dfdists, dfdists_agg, var_effect, var_other, SAVEDIR=savedir,
                                        PLOT_EACH_PAIR=False, do_catplots=do_catplots, do_quick=do_quick)
            plt.close("all")

def _euclidianshuff_stats_linear_load_mult_dates_postprocess(DFDISTS):
    """
    Various conditioning of DFDISTS, to prep for linear model stats, mainly making the relevant variables 
    caterogical, and conjuntive with (Aniaml, date) -- i.e each datapt is (shape_pair, animal, date)

    MS: checked
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import convert_var_to_categorical, convert_var_to_categorical_mult_columns

    DFDISTS["labels_pair_unique"] = [tuple(sorted((row["labels_1"], row["labels_2"]))) for _, row in DFDISTS.iterrows()]
    DFDISTS["shapes_pair_unique"] = [tuple(sorted((row["labels_1"][0], row["labels_2"][0]))) for _, row in DFDISTS.iterrows()]

    DFDISTS["shape1"] = [lab[0] for lab in DFDISTS["labels_1"].tolist()]
    DFDISTS["loc1"] = [lab[1] for lab in DFDISTS["labels_1"].tolist()]
    DFDISTS["shape2"] = [lab[0] for lab in DFDISTS["labels_2"].tolist()]
    DFDISTS["loc2"] = [lab[1] for lab in DFDISTS["labels_2"].tolist()]

    print("Categorical coding....")
    for x in ["shape", "loc"]:
        print("... For variable: ", x)
        # Append animal/date
        for y in [1,2]:
            DFDISTS = append_col_with_grp_index(DFDISTS, [f"{x}{y}", "animal", "date"], f"{x}{y}")

        # Convert to categorical index
        convert_var_to_categorical_mult_columns(DFDISTS, [f"{x}1", f"{x}2"], [f"{x}1", f"{x}2"], PRINT=True)

    # ANimal,date
    convert_var_to_categorical(DFDISTS, "animal", "animal_cat")
    convert_var_to_categorical(DFDISTS, "date", "date_cat")

    for x in ["shape", "loc"]:
        DFDISTS = append_col_with_grp_index(DFDISTS, [f"{x}1", f"{x}2"], f"{x}12")
        DFDISTS[f"{x}same"] = DFDISTS[f"{x}1"] == DFDISTS[f"{x}2"]
    DFDISTS = append_col_with_grp_index(DFDISTS, ["shape12", "loc12"], f"shapeloc12")

    return DFDISTS

def euclidianshuff_stats_linear_load_mult_dates(animal, list_date, var_other, analysis_kind, merge_pancho_ss_twinds=False,
                                                old_run_number=None, SAVEDIR_BASE=None, SAVEDIR_BASE_char=None,
                                                manuscript_version=False):
    """
    Loads multiple days, with preprpicessing.
    This is important in that if FIRST concats dates, appends (animal, date) to shape and loc variables, and then
    converts to category. If convert to cat first, then loses the uniquenss of each (naimal, date) datapt.

    MS: checked
    """
    list_dfdists=[]
    for date in list_date:
        dfdists, _, _ = euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other, 
                                                                        convert_to_df_with_postprocessing=True,
                                                                        merge_pancho_ss_twinds=merge_pancho_ss_twinds,
                                                                        analysis_kind=analysis_kind, old_run_number=old_run_number,
                                                                        SAVEDIR_BASE=SAVEDIR_BASE, SAVEDIR_BASE_char=SAVEDIR_BASE_char,
                                                                        manuscript_version=manuscript_version)
        if dfdists is not None: # Sometimes date doesnt have data...
            dfdists["animal"] = animal
            dfdists["date"] = date
            list_dfdists.append(dfdists)
    DFDISTS = pd.concat(list_dfdists).reset_index(drop=True)

    return DFDISTS


def euclidianshuff_stats_linear_load(animal, date, var_other, merge_pancho_ss_twinds=False):
    """
    Helper to load this day, along with preprpocessing.
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import convert_var_to_categorical

    DFDISTS, _, _ = euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other, 
                                                                    convert_to_df_with_postprocessing=True,
                                                                    merge_pancho_ss_twinds=merge_pancho_ss_twinds)
    DFDISTS["animal"] = animal
    DFDISTS["date"] = date
    SAVEDIR_PLOTS = f"/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF/MULT/stats_linear_model/{animal}-{date}-var_other={var_other}"
    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)

    #################### Prep for linear model stats
    DFDISTS = _euclidianshuff_stats_linear_load_mult_dates_postprocess(DFDISTS)

    # # # Get a region, for testing.
    # # bregion = "M1"
    # # for dfdist in list_dfdist:
    # #     if (dfdist["bregion"].unique()[0]==bregion) and (dfdist["subspace_projection"].unique()[0]=="shape_loc"):
    # #         print("Got it")
    # #         break
    # DFDISTS["shape1"] = [lab[0] for lab in DFDISTS["labels_1"].tolist()]
    # DFDISTS["loc1"] = [lab[1] for lab in DFDISTS["labels_1"].tolist()]
    # DFDISTS["shape2"] = [lab[0] for lab in DFDISTS["labels_2"].tolist()]
    # DFDISTS["loc2"] = [lab[1] for lab in DFDISTS["labels_2"].tolist()]


    # for x in ["shape", "loc"]:
    #     for y in [1,2]:
    #         DFDISTS = append_col_with_grp_index(DFDISTS, [f"{x}{y}", "animal", "date"], f"{x}{y}")
    # convert_var_to_categorical(DFDISTS, "shape1", "shape1")
    # convert_var_to_categorical(DFDISTS, "loc1", "loc1")
    # convert_var_to_categorical(DFDISTS, "shape2", "shape2")
    # convert_var_to_categorical(DFDISTS, "loc2", "loc2")
    # convert_var_to_categorical(DFDISTS, "animal", "animal_cat")
    # convert_var_to_categorical(DFDISTS, "date", "date_cat")

    # for x in ["shape", "loc"]:
    #     DFDISTS = append_col_with_grp_index(DFDISTS, [f"{x}1", f"{x}2"], f"{x}12")
    #     DFDISTS[f"{x}same"] = DFDISTS[f"{x}1"] == DFDISTS[f"{x}2"]
    # DFDISTS = append_col_with_grp_index(DFDISTS, ["shape12", "loc12"], f"shapeloc12")

    return DFDISTS, SAVEDIR_PLOTS

def _remove_shape(DFDISTS, list_sh_remove, var_same_same):
    """
    Hacky -- remove a shape from DFDISTS.
    """
    # list_sh_remove = ["Lcentered-DR-DR", "V-DD-DD"]

    a = [x[0] not in list_sh_remove for x in DFDISTS["labels_1"]]
    b = [x[0] not in list_sh_remove for x in DFDISTS["labels_2"]]
    no_lines = [aa and bb for aa, bb in zip(a, b)]
    DFDISTS_THIS = DFDISTS[no_lines].reset_index(drop=True)
    print("Old length: ", len(DFDISTS))
    print("New length: ",len(DFDISTS_THIS))

    # Agg
    from pythonlib.tools.pandastools import aggregGeneral
    dfdists_agg = aggregGeneral(DFDISTS, ["animal", "date", "metaparams", "bregion", "prune_version", 
                                            "which_level", "event", "subspace|twind", var_same_same],
                                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
    DFDISTS_THIS_AGG = aggregGeneral(DFDISTS_THIS, ["animal", "date", "metaparams", "bregion", "prune_version", 
                                            "which_level", "event", "subspace|twind", var_same_same],
                                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")

    if False:
        # - Original
        _, fig = plot_45scatter_means_flexible_grouping(dfdists_agg, var_same_same, "1|0", "0|1", "metaparams", 
                                            var_value, "bregion", True, shareaxes=True, SIZE=3.5)
        # - Removing lines
        _, fig = plot_45scatter_means_flexible_grouping(DFDISTS_THIS_AGG, var_same_same, "1|0", "0|1", "metaparams", 
                                            var_value, "bregion", True, shareaxes=True, SIZE=3.5)

    return DFDISTS_THIS, DFDISTS_THIS_AGG

def euclidianshuff_stats_linear_plot_wrapper_manuscript(DFDISTS, SAVEDIR_PLOTS, var_other, var_effect="seqc_0_shape", 
                                             subsample_plot_each_iter_stats=False,
                                             do_subsample_method=True):
    """
    GOOD, MAkes all plots and does stats, final plots for manuscript.
    PARAMS:
    - DFDISTS, 
    (can be single date or multiple dates, in the latter case make sure that each shape is shape-date conjucntion)
    
    MS: checked
    """
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_2br_scatter_wrapper
    from pythonlib.tools.pandastools import aggregGeneral

    assert len(DFDISTS["prune_version"].unique())==1, "code assumes only one, becuase was written for shape invar"
    assert len(DFDISTS["which_level"].unique())==1, "code assumes only one, becuase was written for shape invar"
    
    var_same_same = f"same-{var_effect}|{var_other}"

    ########### SCATTER OF EACH CASE
    savedir = f"{SAVEDIR_PLOTS}/scatter_each_label1"
    import os
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    ### FINAL AGGS
    # (1) each datapt is a unique label (i.e., shape-location) (multiple datapts per 0|1)
    # - make a dupliated, so you have symmetric data
    tmp = DFDISTS.copy()
    tmp["labels_2"] = DFDISTS["labels_1"]
    tmp["labels_1"] = DFDISTS["labels_2"]
    tmp = pd.concat([tmp, DFDISTS]).reset_index(drop=True)
    if False: # Sanity check that got diagonals
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        grouping_plot_n_samples_conjunction_heatmap(tmp, "labels_1", "labels_2")

    # Maintain separate dates
    DFDISTS_DATPT_SHAPE1_EACHDATE = aggregGeneral(tmp, ["animal", "metaparams", "bregion", "prune_version", "which_level", "event", 
                                               "subspace|twind", f"{var_effect}_1", var_same_same, "date"],
                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
    # Agg over dates
    DFDISTS_DATPT_LABEL1 = aggregGeneral(tmp, ["animal", "metaparams", "bregion", "prune_version", "which_level", "event", 
                                               "subspace|twind", "labels_1", var_same_same],
                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")

    ### Plots, scatters
    grp_vars = ["event", "subspace|twind"]
    grpdict = grouping_append_and_return_inner_items_good(DFDISTS_DATPT_LABEL1, grp_vars)
    for grp, inds in grpdict.items():
        dfdists_labels = DFDISTS_DATPT_LABEL1.iloc[inds].reset_index(drop=True)

        savepath = f"{savedir}/scatter_each_label1-grp={grp}-counts.txt"
        grouping_print_n_samples(dfdists_labels, [var_same_same, "bregion", "labels_1"], savepath=savepath)

        # Scatter plot, showing distributions of datapts
        _, fig = plot_45scatter_means_flexible_grouping(dfdists_labels, var_same_same, "1|0", "0|1", "bregion", "dist_yue_diff", "labels_1", 
                                                        shareaxes=True, plot_text=False, SIZE=3.8, alpha=0.3);
        savefig(fig, f"{savedir}/scatter_each_label1-grp={grp}.pdf")
        
        plt.close("all")

    ### COMPARE AREAS
    if do_subsample_method:
        ### Also plot with subsampling of (shape, tk) conditions.
        savedir = f"{SAVEDIR_PLOTS}/SUBSAMPLE"
        os.makedirs(savedir, exist_ok=True)
        var_datapt = None
        _euclidianshuff_stats_linear_2br_scatter_wrapper_SUBSAMPLE(DFDISTS, var_effect, var_same_same, var_datapt, savedir,
                                                                plot_each_iter_stats=subsample_plot_each_iter_stats)

    ###############################################################
    ### Compare areas
    # Note, here each datapt is conjunction of variables + (animal, date), where variables could be, for instance, (shape, loc).
    # i.e., (prim1, loc2) x (animal, date) is a single datapt. 
    # And then the plots/analyses are each pair of these. 
    # This means that could have repeated conditions across days, leading to more datapts.
    plot_heatmap_counts = False
    plot_catplots = False
    var_datapt, agg_over_dates, use_symmetric_dfdist = None, False, False
    
    if use_symmetric_dfdist:
        dfdists = DFDISTS_DATPT_SHAPE1_EACHDATE
    else:
        dfdists = DFDISTS

    dfdists_agg = dfdists
        
    savedir = f"{SAVEDIR_PLOTS}/var_datapt={var_datapt}-aggoverdates={agg_over_dates}"
    os.makedirs(savedir, exist_ok=True)

    DFSTATS_2BR = _euclidianshuff_stats_linear_2br_scatter_wrapper(dfdists_agg, var_same_same, var_datapt, savedir,
                                                                    plot_heatmap_counts=plot_heatmap_counts, 
                                                                    plot_catplots=plot_catplots)

    return DFDISTS, DFDISTS_DATPT_LABEL1, DFSTATS_2BR


def euclidianshuff_stats_linear_plot_wrapper(DFDISTS, SAVEDIR_PLOTS, var_other, var_effect="seqc_0_shape", 
                                             do_vs_zero=False,
                                             subsample_plot_each_iter_stats=False,
                                             do_subsample_method=True):
    """
    GOOD, MAkes all plots and does stats, final plots for manuscript.
    PARAMS:
    - DFDISTS, 
    (can be single date or multiple dates, in the latter case make sure that each shape is shape-date conjucntion)
    date
    """
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    assert len(DFDISTS["prune_version"].unique())==1, "code assumes only one, becuase was written for shape invar"
    assert len(DFDISTS["which_level"].unique())==1, "code assumes only one, becuase was written for shape invar"
    
    var_same_same = f"same-{var_effect}|{var_other}"

    ########### SCATTER OF EACH CASE
    from pythonlib.tools.pandastools import aggregGeneral
    savedir = f"{SAVEDIR_PLOTS}/scatter_each_label1"
    import os
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    ## FINAL AGGS
    # (1) each datapt is a unique label (i.e., shape-location) (multiple datapts per 0|1)
    # - make a dupliated, so you have symmetric data
    tmp = DFDISTS.copy()
    tmp["labels_2"] = DFDISTS["labels_1"]
    tmp["labels_1"] = DFDISTS["labels_2"]
    tmp = pd.concat([tmp, DFDISTS]).reset_index(drop=True)
    if False: # Sanity check that got diagonals
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        grouping_plot_n_samples_conjunction_heatmap(tmp, "labels_1", "labels_2")

    # Maintain separate dates
    DFDISTS_DATPT_SHAPE1_EACHDATE = aggregGeneral(tmp, ["animal", "metaparams", "bregion", "prune_version", "which_level", "event", 
                                               "subspace|twind", f"{var_effect}_1", var_same_same, "date"],
                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
    # Agg over dates
    DFDISTS_DATPT_LABEL1 = aggregGeneral(tmp, ["animal", "metaparams", "bregion", "prune_version", "which_level", "event", 
                                               "subspace|twind", "labels_1", var_same_same],
                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
    if False:
        # Check output
        grouping_plot_n_samples_conjunction_heatmap(DFDISTS_DATPT_LABEL1, "labels_1", "same-seqc_0_shape|seqc_0_loc")

    # # (2) each datapt is 0|1, 1|0, 1|1, 0|0 (i.e., 4 datapts per bregion/metaparams). Each shape has (ntask_kinds) datapts.
    # # Agg over all conditions (e.g. label pairs)
    # DFDISTS_LABEL_AGG = aggregGeneral(DFDISTS_DATPT_LABEL1, ["bregion", "which_level", "event", "subspace|twind", var_same_same],
    #                             ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")

    # grouping_plot_n_samples_conjunction_heatmap(DFDISTS_DATPT_LABEL1, "bregion", "same-seqc_0_shape|seqc_0_loc")
    grp_vars = ["event", "subspace|twind"]
    grpdict = grouping_append_and_return_inner_items_good(DFDISTS_DATPT_LABEL1, grp_vars)
    for grp, inds in grpdict.items():
        dfdists_labels = DFDISTS_DATPT_LABEL1.iloc[inds].reset_index(drop=True)

        savepath = f"{savedir}/scatter_each_label1-grp={grp}-counts.txt"
        grouping_print_n_samples(dfdists_labels, [var_same_same, "bregion", "labels_1"], savepath=savepath)

        if False:
            # Should have <n bregions> per cell.
            grouping_plot_n_samples_conjunction_heatmap(dfdists_labels, "labels_1", var_same_same)

        # Scatter plot, showing distributions of datapts
        _, fig = plot_45scatter_means_flexible_grouping(dfdists_labels, var_same_same, "1|0", "0|1", "bregion", "dist_yue_diff", "labels_1", 
                                                        shareaxes=True, plot_text=False, SIZE=3.8, alpha=0.3);
        savefig(fig, f"{savedir}/scatter_each_label1-grp={grp}.pdf")
        
        plt.close("all")

    ################# vs zero
    if do_vs_zero:
        # SAVEDIR_MULT = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF/MULT"
        savedir = f"{SAVEDIR_PLOTS}/linear_model_vs_zero"
        import os
        os.makedirs(savedir, exist_ok=True)
        print(savedir)

        # # Or, load multiple dates from 
        from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_vs0_compute
        DFSTATS_vs0 = _euclidianshuff_stats_linear_vs0_compute(DFDISTS, var_same_same)

        # Plot
        alpha=0.05
        nregions = len(DFDISTS["bregion"].unique())
        ncontrasts = 2
        ncomparisons = nregions * ncontrasts
        alpha_bonf = alpha/ncomparisons
        import seaborn as sns
        fig = sns.catplot(data=DFSTATS_vs0, x="bregion", y="pval_log10", hue="coeffname_simple", row="event", col="subspace|twind", kind="bar")
        for ax in fig.axes.flatten():
            ax.axhline(np.log10(0.05))
            ax.axhline(np.log10(0.005))
            ax.axhline(np.log10(0.0005))
            ax.axhline(np.log10(alpha_bonf), color="r")
            ax.set_ylim(bottom=-8)
        savefig(fig, f"{savedir}/stats-catplot.pdf")

        ### Plot final scatterplot and p-value results
        from pythonlib.tools.plottools import color_make_map_discrete_labels
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
        from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
        order_bregion = _REGIONS_IN_ORDER_COMBINED

        ### Plot
        var_pval = "pval"
        var_value = "dist_yue_diff"

        pstatuses = ["shape_loc", "shape", "loc", "none"]
        grpvar = "subspace|twind"
        grp_vars = [grpvar]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)
        grpdict_pval = grouping_append_and_return_inner_items_good(DFSTATS_vs0, grp_vars)

        for grp, inds in grpdict_dat.items():
            dfthis = DFDISTS.iloc[inds].reset_index(drop=True)

            inds_pval = grpdict_pval[grp]
            dfthis_pval = DFSTATS_vs0.iloc[inds_pval]

            for pval_thresh in [alpha, 0.005, alpha_bonf, 0.0005]:

                ### For each region, determine if it is signif for shape, loc, or shapeandloc
                map_bregion_pval_status = {}
                for bregion in order_bregion:
                    p_shape = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="0|1")][var_pval].values[0]
                    p_loc = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="1|0")][var_pval].values[0]

                    if (p_shape<pval_thresh) and (p_loc<pval_thresh):
                        p_status = "shape_loc"
                    elif (p_shape<pval_thresh):
                        p_status = "shape"
                    elif (p_loc<pval_thresh):
                        p_status = "loc"
                    else:
                        p_status = "none"
                    map_bregion_pval_status[bregion] = p_status

                ### Plot it
                _, fig = plot_45scatter_means_flexible_grouping(dfthis, var_same_same, "1|0", "0|1", "event", 
                                                    var_value, "bregion", True, shareaxes=True,
                                                    map_datapt_lev_to_colorlev=map_bregion_pval_status, colorlevs_that_exist=pstatuses)
                savefig(fig, f"{savedir}/scatter-{grpvar}={grp}-alpha={pval_thresh}.pdf")
                plt.close("all")
    else:
        DFSTATS_vs0 = None

    ### COMPARE AREAS
    if do_subsample_method:
        ### Also plot with subsampling of (shape, tk) conditions.
        savedir = f"{SAVEDIR_PLOTS}/SUBSAMPLE"
        os.makedirs(savedir, exist_ok=True)
        var_datapt = None
        _euclidianshuff_stats_linear_2br_scatter_wrapper_SUBSAMPLE(DFDISTS, var_effect, var_same_same, var_datapt, savedir,
                                                                plot_each_iter_stats=subsample_plot_each_iter_stats)

    ###############################################################
    ### Compare areas
    # Note, here each datapt is conjunction of variables + (animal, date), where variables could be, for instance, (shape, loc).
    # i.e., (prim1, loc2) x (animal, date) is a single datapt. 
    # And then the plots/analyses are each pair of these. 
    # This means that could have repeated conditions across days, leading to more datapts.

    # Perform this using different aggs (i.e, different definitions of datapt)
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_2br_scatter_wrapper
    from pythonlib.tools.pandastools import aggregGeneral
    plot_heatmap_counts = False
    plot_catplots = False
    for var_datapt, agg_over_dates, use_symmetric_dfdist in [
        (f"{var_effect}_1", False, True), # Each unique (sh1, date)
        # ("labels_1", False, True), # Each unique (sh1, date) -- skip, this replaced by var_effect_1
        ("labels_pair_unique", True, False), # Each unique [(sh1, tk1), (sh2, tk2)]
        (None, False, False), # Original, each unique [(sh1, tk1, date1), (sh2, tk2, date2)]
        ("date", False, False), # each unique (date)
        ("shapes_pair_unique", False, False), # Each unique (sh1, sh2, date),
        ]:

        if use_symmetric_dfdist:
            dfdists = DFDISTS_DATPT_SHAPE1_EACHDATE
        else:
            dfdists = DFDISTS

        if var_datapt is not None:
            # agg over this.
            print("agging over label pairs...")
            vars_agg = ["animal", "metaparams", "bregion", "prune_version", 
                                                    "which_level", "event", "subspace|twind", var_same_same, var_datapt]
            if agg_over_dates == False and "date" not in vars_agg:
                # Determines whether you agg over dates.
                vars_agg.append("date")
            dfdists_agg = aggregGeneral(dfdists, vars_agg,
                                        ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"], nonnumercols="all")
        else:
            dfdists_agg = dfdists
            
        savedir = f"{SAVEDIR_PLOTS}/var_datapt={var_datapt}-aggoverdates={agg_over_dates}"
        os.makedirs(savedir, exist_ok=True)

        DFSTATS_2BR = _euclidianshuff_stats_linear_2br_scatter_wrapper(dfdists_agg, var_same_same, var_datapt, savedir,
                                                                       plot_heatmap_counts=plot_heatmap_counts, 
                                                                       plot_catplots=plot_catplots)

    return DFDISTS, DFDISTS_DATPT_LABEL1, DFSTATS_vs0, DFSTATS_2BR

def _euclidianshuff_stats_linear_vs0_compute(DFDISTS, var_same_same, plot_coeff=False):
    """
    Versus zero.
    """
    yvar = "dist_norm"

    # order_bregion = _REGIONS_IN_ORDER_COMBINED
    # list_bregion = order_bregion

    # contrast_var = var_same_same
    yvar = "dist_norm"
    grp_vars = ["event", "subspace|twind", "bregion"]
    grpdict = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)

    contrast_levels = ["0|0", "0|1", "1|0", "1|1"]
    contrast_levels = ["0|1", "1|0", "1|1"]
    res = []
    for grp, inds in grpdict.items():
        dfthis = DFDISTS.iloc[inds].reset_index(drop=True)
        dflm = dfthis[(dfthis[var_same_same].isin(contrast_levels))].reset_index(drop=True)
        import statsmodels.formula.api as smf

        # Test shape_same, along with effects of each loc pair
        for formula, coeffname_simple, same_same, coeffname in [
            (f"{yvar} ~ C(shapesame, Treatment(True)) + C(loc12)", "shape", "0|1", "C(shapesame, Treatment(True))[T.False]"),
            (f"{yvar} ~ C(locsame, Treatment(True)) + C(shape12)", "loc", "1|0", "C(locsame, Treatment(True))[T.False]"),
            ]:

            if True:
                # OLS
                md = smf.ols(formula, dflm)
                mdf = md.fit()
            else:
                # LME
                md = smf.mixedlm(formula, dflm, groups=dflm["date_cat"], re_formula="~0 ")
                mdf = md.fit()

            # mdf.summary()

            if False: # LME
                import statsmodels.formula.api as smf

                # lev_treat_default = 

                # if lev_treat_default is None:
                #     lev_treat_default = df[fixed_treat].unique().tolist()[0]
                    
                # if rand_grp is None:
                #     assert not rand_grp_list is None
                # else:
                #     assert rand_grp_list is None

                # if rand_grp_list is not None:
                #     _, df = grouping_append_and_return_inner_items(df, rand_grp_list, 
                #         new_col_name="dummytmp", return_df=True)
                #     rand_grp = "dummytmp"
                # else:
                #     assert rand_grp in df.columns



                # formula = f"{yvar} ~ C(shapesame, Treatment(True)) + C(locsame, Treatment(True))"
                # formula = f"{yvar} ~ C(shapesame, Treatment(True)) + C(locsame, Treatment(True)) + C(shape12, Treatment((0,0))) + C(loc12, Treatment((0,0)))"

                formula = f"{yvar} ~ C(shape12, Treatment((0,0))) + C(loc12, Treatment((0,0)))"
                # formula = f"{yvar} ~ C(shape12, Treatment((0,0)))"
                # formula = f"{yvar} ~ C(loc12, Treatment((0,0)))"

                # str_treat = f"C({fixed_treat}, Treatment('{lev_treat_default}'))"
                # formula = f"{yvar} ~ {str_treat}"
                md = smf.mixedlm(formula, dflm, groups=dflm["shape12"])


                formula = f"{yvar} ~ C(shapesame, Treatment(True))"
                md = smf.mixedlm(formula, dflm, groups=dflm["loc12"])

                formula = f"{yvar} ~ C(locsame, Treatment(True))"
                md = smf.mixedlm(formula, dflm, groups=dflm["shape12"])


                mdf = md.fit()

                mdf.summary()

            if plot_coeff:
                # Extract the coefficients
                coefficients = mdf.params

                # Plotting the coefficients in a bar plot
                plt.figure(figsize=(8, 6))
                coefficients.plot(kind='bar')
                plt.title('Coefficients of the OLS Model')
                plt.xlabel('Predictor Variables')
                plt.ylabel('Coefficient Value')
                plt.show()

                import seaborn as sns
                sns.catplot(data=dfthis, x="same-seqc_0_shape|seqc_0_loc", y="dist_yue_diff")
                sns.catplot(data=dfthis, x="same-seqc_0_shape|seqc_0_loc", y="dist_yue_diff", kind="bar")

                sns.catplot(data=dfthis, x="same-seqc_0_shape|seqc_0_loc", y=yvar)
                sns.catplot(data=dfthis, x="same-seqc_0_shape|seqc_0_loc", y=yvar, kind="bar")
                assert False, "cannot do to many plots"
                
            coefficients = mdf.params
            dfcoeff = coefficients.reset_index()
            dfpvals = mdf.pvalues.reset_index()

            assert dfcoeff.iloc[1]["index"] == coeffname
            assert dfpvals.iloc[1]["index"] == coeffname

            res.append({
                "results":mdf,
                "coeffname":coeffname,
                "coeffname_simple":coeffname_simple,
                "coeff_val":dfcoeff.iloc[1][0],
                "pval":dfpvals.iloc[1][0],
                var_same_same:same_same,
            })

            for col, val in zip(grp_vars, grp):
                res[-1][col] = val

    DFSTATS = pd.DataFrame(res)
    DFSTATS["pval_log10"] = np.log10(DFSTATS["pval"])

    return DFSTATS


def _euclidianshuff_stats_linear_2br_compute(DFDISTS, var_same_same, var_datapt):
    """
    Compute pairwise stats (each pair is two brain regions) 
    for effect, across each pair of brain regions.
    PARAMS:
    - DFDISTS, one datapt for each (label1, label2), ie already agged over trials, but keeping all conditions present.
    RETURNS:
    - DFSTATS_2BR, each row is a bregion pair
    MS: checked
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    import statsmodels.formula.api as smf

    if var_datapt is None:
        var_datapt = "shapeloc12"

    yvar = "dist_yue_diff"
    list_bregion = ORDER_BREGION
    grp_vars = ["event", "subspace|twind", "metaparams"]
    grpdict = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)
    res = []
    for grp, inds in grpdict.items():
        dfthis = DFDISTS.iloc[inds].reset_index(drop=True)
        for contrast_lev  in ["1|0", "0|1"]: 
            for i in range(len(list_bregion)):
                for j in range(len(list_bregion)):
                    if j>i:
                        bregion1 = list_bregion[i]
                        bregion2 = list_bregion[j]
                        
                        dflm = dfthis[
                            (dfthis[var_same_same]==contrast_lev) & (dfthis["bregion"].isin([bregion1, bregion2]))
                            ].reset_index(drop=True)

                        if False:
                            # to do sanity check of what datapts actually go in.
                            from pythonlib.tools.pandastools import grouping_print_n_samples
                            savepath = f"/tmp/testthis-grp={grp}-contrast_lev={contrast_lev}-{bregion1}-{bregion2}.txt"
                            grouping_print_n_samples(dflm, ["bregion", "event", "subspace|twind", "same-seqc_0_shape|seqc_0_loc"], 
                                                    savepath=savepath)

                        # Linear model, to get effect of contrast
                        # - fixed effect of brain region and each datapt level
                        formula = f"{yvar} ~ C(bregion, Treatment('{bregion1}')) + C({var_datapt})"

                        md = smf.ols(formula, dflm)
                        mdf = md.fit()
                        # display(mdf.summary())
                        # assert False
                        coefficients = mdf.params
                        dfcoeff = coefficients.reset_index()
                        dfpvals = mdf.pvalues.reset_index()

                        coeffname = f"C(bregion, Treatment('{bregion1}'))[T.{bregion2}]"
                        assert dfcoeff.iloc[1]["index"] == coeffname
                        assert dfpvals.iloc[1]["index"] == coeffname

                        res.append({
                            "results":mdf,
                            "coeffname":coeffname,
                            "coeff_val":dfcoeff.iloc[1][0],
                            "pval":dfpvals.iloc[1][0],
                            var_same_same:contrast_lev,
                            "bregion1":bregion1,
                            "bregion2":bregion2,
                            "formula":formula,
                        })
                        for col, val in zip(grp_vars, grp):
                            res[-1][col] = val

    DFSTATS_2BR = pd.DataFrame(res)
    DFSTATS_2BR["pval_log10"] = np.log10(DFSTATS_2BR["pval"])

    # Make a mirror image
    dftmp = DFSTATS_2BR.copy()
    dftmp["bregion1"] = DFSTATS_2BR["bregion2"]
    dftmp["bregion2"] = DFSTATS_2BR["bregion1"]
    dftmp["coeff_val"] = -dftmp["coeff_val"] 
    dftmp["coeffname"] = "ignore"
    dftmp["formula"] = "ignore"
    DFSTATS_2BR = pd.concat([DFSTATS_2BR, dftmp]).reset_index(drop=True)

    return DFSTATS_2BR


def _euclidianshuff_stats_linear_2br_compute_nsigs(DFSTATS_2BR, var_same_same, pval_thresh):
    """
    Given stats, one for each pair of bregions,
    summarize pairwise by counting, for each region, how mnay cases (in comparisong to other regions) 
    it has that are significantly higher effect 

    RETURNS:
    - DFSTATS_2BR_NSIGS, holds, for each bregion and contrast, the number of other bregions is beats

    MS: checked
    """

    #################### Summarize pairwise by counting, for each region, how mnay cases (in comparisong to other regions)
    # it has that are significantly higher effect
    ### Scatter, with x color reflecting the number of other regions this region is more significant than
    # var_pval = "pval"
    # var_value = "dist_yue_diff"

    # Check each anlays
    grp_vars = ["event", "subspace|twind", "metaparams"]
    grpdict_pval = grouping_append_and_return_inner_items_good(DFSTATS_2BR, grp_vars)
    res = []
    for grp, inds in grpdict_pval.items():
        dfthis_pval = DFSTATS_2BR.iloc[inds].reset_index(drop=True)

        if False:
            grouping_plot_n_samples_conjunction_heatmap(dfthis_pval, "bregion1", "bregion2", ["same-seqc_0_shape|seqc_0_loc"])

        # For each region, count how many other regions it is higher than
        _grpdict = grouping_append_and_return_inner_items_good(dfthis_pval, ["bregion2", var_same_same])
        for _grp, _inds in _grpdict.items():
            _dfthis = dfthis_pval.iloc[_inds]
            # how many cases where coeff is positive, and pval is below threshold?
            n = sum((_dfthis["coeff_val"]>0) & (_dfthis["pval"]<pval_thresh))
            res.append({
                "n_sig":n,
                "bregion":_grp[0],
                var_same_same:_grp[1],
                grp_vars[0]:grp[0],
                grp_vars[1]:grp[1],
                grp_vars[2]:grp[2],
                "pval_thresh":pval_thresh
            })
    DFSTATS_2BR_NSIGS = pd.DataFrame(res)

    return DFSTATS_2BR_NSIGS


def _euclidianshuff_stats_linear_2br_scatter_wrapper(DFDISTS, var_same_same, var_datapt, SAVEDIR_PLOTS, 
                                                     plot_heatmap_counts=True,
                                                     plot_catplots=True,
                                                     plot_results_scatter=True):
    """
    Wrapper for computing pairwise stats and plotting as scatterplots.
    PARAMS:
    - DFDISTS, what you have as the rows will define the datapts that go into 
    stats.
    
    ###############################################################
    ### Compare areas
    # Note, here each datapt is conjunction of variables + (animal, date), where variables could be, for instance, (shape, loc).
    # i.e., (prim1, loc2) x (animal, date) is a single datapt. 
    # And then the plots/analyses are each pair of these. 
    # This means that could have repeated conditions across days, leading to more datapts.

    MS: checked
    """
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_2br_compute
    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    from math import comb
    import os
    from pythonlib.tools.pandastools import grouping_print_n_samples, grouping_plot_n_samples_conjunction_heatmap

    if plot_heatmap_counts or plot_catplots or plot_results_scatter:
        savedir = f"{SAVEDIR_PLOTS}/linear_model_region_vs_region"
        os.makedirs(savedir, exist_ok=True)
        print(savedir)

    ### First, save heatmap showing the datapts counts
    if plot_heatmap_counts:

        grp_vars = ["subspace|twind", "event", "bregion"]
        grpdict = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)
        for grp, inds in grpdict.items():
            dfthis = DFDISTS.iloc[inds].reset_index(drop=True)

            if ("date" in dfthis) and ("labels_1" in dfthis) and ("labels_2" in dfthis):
                savepath = f"{savedir}/COUNTS_USED_IN_STATS-grp={grp}-counts.txt"
                grouping_print_n_samples(dfthis, [var_same_same, "bregion", "labels_1", "labels_2", "date"], savepath=savepath)

                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, "labels_1", "labels_2", [var_same_same, "date"])
                savefig(fig, f"{savedir}/COUNTS_USED_IN_STATS-grp={grp}-counts.pdf")

            # only run this once, as it takes a while, and once is enough
            break

    ### Compute stats
    DFSTATS_2BR = _euclidianshuff_stats_linear_2br_compute(DFDISTS, var_same_same, var_datapt)

    ### Plot
    order_bregion = _REGIONS_IN_ORDER_COMBINED
    npairs = comb(len(order_bregion), 2)
    ncomp = 2
    alpha=0.05
    alpha_bonf_easy = alpha/(npairs)
    alpha_bonf_hard = alpha/(npairs * ncomp) # 

    if plot_catplots:
        ### First, plot nsig
        from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values
        for pval_thresh in  [alpha_bonf_hard]:
            DFSTATS_2BR_NSIGS = _euclidianshuff_stats_linear_2br_compute_nsigs(DFSTATS_2BR, var_same_same, pval_thresh)
            DFSTATS_2BR_NSIGS = append_col_with_grp_index(DFSTATS_2BR_NSIGS, [var_same_same, "pval_thresh"], "var_same_same|pvalthresh")
            fig, axes = plot_subplots_heatmap(DFSTATS_2BR_NSIGS, "bregion", "var_same_same|pvalthresh", "n_sig", "metaparams", annotate_heatmap=True)
            savefig(fig, f"{savedir}/nsig_heatmap-pval_thresh={pval_thresh}.pdf")

        ### Second, plot catplots
        z = np.max(np.abs(np.percentile(DFSTATS_2BR["coeff_val"], [0.5, 99.5])))
        ZLIMS = [-z, z]

        import seaborn as sns
        from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values
        
        grp_vars = ["subspace|twind", "event"]
        grpdict = grouping_append_and_return_inner_items_good(DFSTATS_2BR, grp_vars)
        for grp, inds in grpdict.items():
            dfstats = DFSTATS_2BR.iloc[inds].reset_index(drop=True)

            fig, axes = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "coeff_val", var_same_same, 
                                        True, True, None, True, W=6, ZLIMS=ZLIMS, row_values=order_bregion, col_values=order_bregion)
            savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}.pdf")
            
            zlims = [-5, 0]
            fig, _ = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "pval_log10", var_same_same, 
                                        False, True, None, True, W=6, ZLIMS=zlims, row_values=order_bregion, col_values=order_bregion)
            savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}-pvals.pdf")
            
            zlims = [np.log10(alpha_bonf_hard)-3, np.log10(alpha_bonf_hard)]
            fig, _ = plot_subplots_heatmap(dfstats, "bregion1", "bregion2", "pval_log10", var_same_same, 
                                        False, True, None, True, W=6, ZLIMS=zlims, row_values=order_bregion, col_values=order_bregion)
            savefig(fig, f"{savedir}/COMPARE_AREAS-grp={grp}-pvals_bonf.pdf")

            fig = sns.catplot(data=dfstats, x="bregion1", y="coeff_val", hue=var_same_same, col="bregion2", 
                        col_order=order_bregion, order=order_bregion, col_wrap=6, kind="bar")
            savefig(fig, f"{savedir}/COMPARE_AREAS-catplot-grp={grp}.pdf")

            fig = sns.catplot(data=dfstats, x="bregion1", y="pval_log10", hue=var_same_same, col="bregion2", 
                        col_order=order_bregion, order=order_bregion, col_wrap=6, kind="bar")
            for ax in fig.axes:
                ax.axhline(np.log10(0.05))
                ax.axhline(np.log10(0.005))
                ax.axhline(np.log10(0.0005))
                ax.axhline(np.log10(alpha_bonf_easy), color="r")
                ax.axhline(np.log10(alpha_bonf_hard), color="r")
                ax.set_ylim(bottom=-8)
            savefig(fig, f"{savedir}/COMPARE_AREAS-catplot-grp={grp}-pvalues.pdf")

            plt.close("all")

    ################################# PLOTS
    if plot_results_scatter:
        from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_2br_compute, _euclidianshuff_stats_linear_2br_scatter, _euclidianshuff_stats_linear_2br_compute
        _euclidianshuff_stats_linear_2br_scatter(DFDISTS, DFSTATS_2BR, var_same_same, [alpha_bonf_easy, alpha_bonf_hard], savedir,
                                                plot_heatmap_counts=plot_heatmap_counts)
    
    return DFSTATS_2BR

def _euclidianshuff_stats_linear_2br_scatter(DFDISTS, DFSTATS_2BR, var_same_same, list_alpha_bonf, savedir,
                                             plot_heatmap_counts=True):
    """
    Low-level stuff: All things related to plotting results from comparing 2 bregions.
    """
    # Plot scatterplot using this as color.
    from pythonlib.tools.plottools import map_coord_to_color_2dgradient
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.expttools import writeDictToTxt
    from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values

    n_bregions = 8

    ### Plot
    var_value = "dist_yue_diff"

    ### Plot a legend of colors
    _, fig = map_coord_to_color_2dgradient(0, 0, 0, n_bregions-1, 0, n_bregions-1, plot_legend=True)
    savefig(fig, f"{savedir}/scatter-LEGEND.pdf")

    counts_plotted = False
    plot_heatmap_counts = False # Since I don't relaly 
    for pval_thresh in [0.05, 0.005, 0.0005] + list_alpha_bonf:
        DFSTATS_2BR_NSIGS = _euclidianshuff_stats_linear_2br_compute_nsigs(DFSTATS_2BR, var_same_same, pval_thresh)
        _euclidianshuff_stats_linear_2br_scatter_plot(DFDISTS, DFSTATS_2BR_NSIGS, var_same_same, pval_thresh, savedir,
                                                    plot_heatmap_counts=plot_heatmap_counts)

def _euclidianshuff_stats_linear_2br_scatter_wrapper_SUBSAMPLE(DFDISTS, var_effect, var_same_same, var_datapt, 
                                                               SAVEDIR_PLOTS, plot_each_iter_stats=False):
    """
    Each iter, subsample shapes within each day to try to have comparable number of cases across days --> to balance the days.
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    
    assert len(DFDISTS["animal"].unique())==1
    niter = 20

    # First, determine min n shapes across dates
    grpdict = grouping_append_and_return_inner_items_good(DFDISTS, ["date", "prune_version"])
    n_min_shapes = 1000000
    for grp, inds in grpdict.items():
        shapes = set(DFDISTS.iloc[inds][f"{var_effect}_1"].unique().tolist() + DFDISTS.iloc[inds][f"{var_effect}_2"].unique().tolist())
        if len(shapes)<n_min_shapes:
            n_min_shapes = len(shapes)
    print("This many min shapes across dates: ", n_min_shapes)
    nget = n_min_shapes
    if nget<3:
        nget = 3 # hard to do stats if too few.
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({"n_min_shapes":n_min_shapes, "nget":nget}, f"{SAVEDIR_PLOTS}/n_shapes_used.txt")

    ### 
    list_df_2br_nsigs = []
    list_dfdists = []
    for i_sub_iter in range(niter):
        print("subsample, iter :", i_sub_iter)
        ### Get subsampled dfdists
        map_dateprune_to_shapes = {}
        list_df = []
        for grp, inds in grpdict.items():
            # Shapes that exist
            shapes = set(DFDISTS.iloc[inds][f"{var_effect}_1"].unique().tolist() + DFDISTS.iloc[inds][f"{var_effect}_2"].unique().tolist())
            
            # Pick two
            if len(shapes)>nget:
                import random
                shapes_get = random.sample(shapes, nget)
                shapes_remove = [sh for sh in shapes if sh not in shapes_get]
                map_dateprune_to_shapes[grp] = shapes_get
            else:
                shapes_remove = []

            # Prune
            dfdists = DFDISTS.iloc[inds].reset_index(drop=True)
            a = dfdists[f"{var_effect}_1"].isin(shapes_remove)
            b = dfdists[f"{var_effect}_2"].isin(shapes_remove)
            dfdists = dfdists[~(a | b)]
            assert len(dfdists)>0

            list_df.append(dfdists)
        DFDISTS_THIS = pd.concat(list_df).reset_index(drop=True)

        ### Get stats
        from pythonlib.tools.pandastools import aggregGeneral

        var_datapt, agg_over_dates, use_symmetric_dfdist = None, False, False
        if False: # Just to plot
            savedir = f"{SAVEDIR_PLOTS}/var_datapt={var_datapt}-aggoverdates={agg_over_dates}"
            os.makedirs(savedir, exist_ok=True)

            DFSTATS_2BR = _euclidianshuff_stats_linear_2br_scatter_wrapper(DFDISTS_THIS, var_same_same, var_datapt, savedir, 
                                                                        plot_catplots=False, plot_heatmap_counts=False)

        ### Get n sig
        from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _euclidianshuff_stats_linear_2br_compute, _euclidianshuff_stats_linear_2br_compute_nsigs, _euclidianshuff_stats_linear_2br_scatter_wrapper
        if False:
            # Old  method. Works fine, but doesnt have option to plot stats details.
            dfdists_agg_2br = _euclidianshuff_stats_linear_2br_compute(DFDISTS_THIS, var_same_same, var_datapt)
        else:
            # New method, can plot stats for each iter.
            _savedir = f"{SAVEDIR_PLOTS}/i_sub_iter={i_sub_iter}"
            os.makedirs(_savedir, exist_ok=True)
            dfdists_agg_2br = _euclidianshuff_stats_linear_2br_scatter_wrapper(DFDISTS_THIS, var_same_same, var_datapt, _savedir,
                                                                plot_heatmap_counts=False,
                                                                plot_catplots=plot_each_iter_stats,
                                                                plot_results_scatter=False)
        
        n_bregions = 8
        from math import factorial, comb
        npairs = comb(n_bregions, 2)
        ncomp = 2
        alpha=0.05
        alpha_bonf_easy = alpha/(npairs)
        alpha_bonf_hard = alpha/(npairs * ncomp)

        ### COLLECT
        list_alpha_bonf = [alpha_bonf_easy, alpha_bonf_hard]
        for pval_thresh in [0.05, 0.005, 0.0005] + list_alpha_bonf:
            dfdists_agg_2br_nsigs = _euclidianshuff_stats_linear_2br_compute_nsigs(dfdists_agg_2br, var_same_same, pval_thresh)
            dfdists_agg_2br_nsigs["i_sub_iter"] = i_sub_iter
            dfdists_agg_2br_nsigs["var_datapt"] = var_datapt
            dfdists_agg_2br_nsigs["agg_over_dates"] = agg_over_dates
            dfdists_agg_2br_nsigs["pval_thresh"] = pval_thresh
            list_df_2br_nsigs.append(dfdists_agg_2br_nsigs)
        list_dfdists.append(DFDISTS_THIS)

    df_2br_nsigs_all = pd.concat(list_df_2br_nsigs).reset_index(drop=True)
    # print("HERERE1:")
    # print(df_2br_nsigs_all.columns)
    # assert len(df_2br_nsigs_all)>0

    from pythonlib.tools.pandastools import replace_None_with_string
    df_2br_nsigs_all = replace_None_with_string(df_2br_nsigs_all)
    vars_grp = ["agg_over_dates", "bregion", var_same_same,
                                    "event", "subspace|twind", "pval_thresh", "metaparams"]
    # print("HERERE2:")
    # print(df_2br_nsigs_all.columns)
    # assert len(df_2br_nsigs_all)>0
    df_2br_nsigs_all_mode = df_2br_nsigs_all.groupby(vars_grp)["n_sig"].agg(pd.Series.mode).reset_index()
    # if tie, then take median
    df_2br_nsigs_all_mode["n_sig"] = [n_sig if not isinstance(n_sig, np.ndarray) else np.median(n_sig) for n_sig in df_2br_nsigs_all_mode["n_sig"]]
    
    # For plotting, agg so one datapt per day
    dfdists_agg_all = pd.concat(list_dfdists).reset_index(drop=True)
    dfdists_agg_all = aggregGeneral(dfdists_agg_all, ["animal", "date", var_same_same, "bregion", "prune_version", "which_level", "event", 
                                        "subspace_projection", "subspace_projection_fitting_twind", "subspace|twind", "metaparams",
                                        "twind_scal"], ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"])
    
    ### PLOT
    savedir = f"{SAVEDIR_PLOTS}/scatter"
    os.makedirs(savedir, exist_ok=True)
    for pval_thresh in df_2br_nsigs_all_mode["pval_thresh"].unique():
        df_2br_nsigs_all_mode_this = df_2br_nsigs_all_mode[df_2br_nsigs_all_mode["pval_thresh"] == pval_thresh].reset_index(drop=True)
        _euclidianshuff_stats_linear_2br_scatter_plot(dfdists_agg_all, df_2br_nsigs_all_mode_this, var_same_same, pval_thresh, savedir,
                                                    plot_heatmap_counts=False)

def _euclidianshuff_stats_linear_2br_scatter_plot(DFDISTS, DFSTATS_2BR_NSIGS, var_same_same, pval_thresh, savedir,
                                             plot_heatmap_counts=True):
    """
    Low-level code for plotting results, scatterplots of stats results. Makes scatterplot with colors corresponding to
    signficance.

    MS: checked
    """
    from pythonlib.tools.plottools import map_coord_to_color_2dgradient
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.expttools import writeDictToTxt
    from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values

    n_bregions = 8

    # Save text file of the sample sizes
    from pythonlib.tools.pandastools import grouping_print_n_samples

    savepath = f"{savedir}/samplesize-confirmed_this_used_in_stats.txt"
    grouping_print_n_samples(DFDISTS, ["bregion", "event", "subspace|twind", var_same_same], savepath=savepath)

    if "date" in DFDISTS:
        savepath = f"{savedir}/samplesize-confirmed_this_used_in_stats-split_by_date.txt"
        grouping_print_n_samples(DFDISTS, ["bregion", "event", "subspace|twind", var_same_same, "date"], savepath=savepath)

    # NOTE: shapeloc12 actually means shape-(task_kind)12
    if ("date" in DFDISTS) and ("labels_1" in DFDISTS) and ("labels_2" in DFDISTS):
        savepath = f"{savedir}/samplesize-each_datapt.txt"
        grouping_print_n_samples(DFDISTS, ["bregion", "event", "subspace|twind", "labels_1", "labels_2", "date"], savepath=savepath)

    # NOTE: shapeloc12 actually means shape-(task_kind)12 -- to show that.
    if ("loc1" in DFDISTS) and ("date" in DFDISTS) and ("labels_1" in DFDISTS):
        savepath = f"{savedir}/samplesize-if_char_expt_then_remind_that_loc_is_taskkind.txt"
        grouping_print_n_samples(DFDISTS, ["labels_1", "date", "metaparams"], savepath=savepath)

    ### Plot
    var_value = "dist_yue_diff"

    ### Plot summary heatmaps
    DFSTATS_2BR_NSIGS = append_col_with_grp_index(DFSTATS_2BR_NSIGS, [var_same_same, "pval_thresh"], "var_same_same|pvalthresh")
    fig, axes = plot_subplots_heatmap(DFSTATS_2BR_NSIGS, "bregion", "var_same_same|pvalthresh", "n_sig", "metaparams", annotate_heatmap=True)
    savefig(fig, f"{savedir}/nsig_heatmap-pval_thresh={pval_thresh}.pdf")

    grp_vars = ["subspace|twind", "event"]
    grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS, grp_vars)
    grpdict_pval = grouping_append_and_return_inner_items_good(DFSTATS_2BR_NSIGS, grp_vars)

    for grp, inds in grpdict_dat.items():
        dfthis = DFDISTS.iloc[inds].reset_index(drop=True)

        inds_pval = grpdict_pval[grp]
        dfthis_pval = DFSTATS_2BR_NSIGS.iloc[inds_pval]

        ### For each region, determine if it is signif for shape, loc, or shapeandloc
        # first, map from region to color.
        map_bregion_pval_status = {}
        for bregion in ORDER_BREGION:
            
            tmp = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="0|1")]["n_sig"]
            assert len(tmp)==1
            n_shape = tmp.values[0]

            tmp = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="1|0")]["n_sig"]
            assert len(tmp)==1
            n_loc = tmp.values[0]

            # print(tmp)
            # assert False
            # n_shape = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="0|1")]["n_sig"].values[0]
            # n_loc = dfthis_pval[(dfthis_pval["bregion"]==bregion) & (dfthis_pval[var_same_same]=="1|0")]["n_sig"].values[0]

            p_status = (n_shape, n_loc)
            map_bregion_pval_status[bregion] = p_status

        ### Map to color, based on nsig for (loc, shape)
        map_bregion_to_color = {}
        for bregion, coord in map_bregion_pval_status.items():
            n_shape = coord[0]
            n_loc = coord[1]
            col = map_coord_to_color_2dgradient(n_loc, n_shape, 0, n_bregions-1, 0, n_bregions-1)
            map_bregion_to_color[bregion] = col

        ### Plot sample size
        from pythonlib.tools.pandastools import grouping_print_n_samples
        
        savepath = f"{savedir}/COUNTS_USED_IN_SCATTER_MEANS={grp}-counts-1.txt"
        if "labels_1" in dfthis and "labels_2" in dfthis and "date" in dfthis:
            grouping_print_n_samples(dfthis, [var_same_same, "bregion", "labels_1", "labels_2", "date"], savepath=savepath)
        if ("shape1" in dfthis) and ("loc1" in dfthis):
            savepath = f"{savedir}/COUNTS_USED_IN_SCATTER_MEANS={grp}-counts-2.txt"
            grouping_print_n_samples(dfthis, [var_same_same, "bregion", "shape1", "loc1", "shape2", "loc2"], savepath=savepath)
        if plot_heatmap_counts:
                fig = grouping_plot_n_samples_conjunction_heatmap(dfthis, "labels_1", "labels_2", [var_same_same, "date"])
                savefig(fig, f"{savedir}/COUNTS_USED_IN_SCATTER_MEANS-grp={grp}.pdf")

        ### Plot it
        _, fig = plot_45scatter_means_flexible_grouping(dfthis, var_same_same, "1|0", "0|1", "event", 
                                            var_value, "bregion", True, shareaxes=True,
                                            map_dataptlev_to_color=map_bregion_to_color, alpha=1,
                                            edgecolor="k", SIZE=3)
        
        savefig(fig, f"{savedir}/scatter-{grp}-alpha={pval_thresh}.pdf")
        writeDictToTxt(map_bregion_pval_status, f"{savedir}/n_cases_win-{grp}-alpha={pval_thresh}-{var_same_same}.txt")

        plt.close("all")

def statespace_traj_plot_manuscript(DFallpa, animal, date, SAVEDIR_ANALYSIS, var_other):
    """ 
    Plot PCA trajectories, colored byshape and split by <var_other>. 
    Makes "pretty plots"
    """

    var_effect = "seqc_0_shape"
    raw_subtract_mean_each_timepoint = False
    twind_analy = TWIND_ANALY
    tbin_dur = 0.2
    tbin_slide = 0.01

    if int(date)>220720:
        twind = (0.05, 0.9)
    else:
        twind = (0.05, 0.6)

    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
    }

    list_subspace_projection = ["shape"]

    list_subtr = [False]
    list_smth = [True]
    list_dims = [(0,1), (1,3), (1,2)]


    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection: 

            ############################
            list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]
            
            for subspace_projection_fitting_twind in list_fit_twind:
                
                # Final save dir
                SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-fit_twind={subspace_projection_fitting_twind}"
                os.makedirs(SAVEDIR, exist_ok=True)
                print("SAVING AT ... ", SAVEDIR)

                # Preprocess
                savedir = f"{SAVEDIR}/preprocess"
                os.makedirs(savedir, exist_ok=True)
                pa = _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                            twind_analy, tbin_dur, tbin_slide, savedir)
                
                ####### Plot state space
                for subtr_time_mean in list_subtr:
                    for do_another_smooth in list_smth:
                        
                        if subtr_time_mean:
                            pathis = pa.norm_subtract_trial_mean_each_timepoint()
                        else:
                            pathis = pa.copy()
                        
                        if do_another_smooth:    
                            pathis = pathis.agg_by_time_windows_binned(tbin_dur, tbin_slide)

                        savedir = f"{SAVEDIR}/subtrmean={subtr_time_mean}-do_another_smooth={do_another_smooth}"
                        os.makedirs(savedir, exist_ok=True)

                        LIST_VAR = [
                            var_effect,
                        ]
                        LIST_VARS_OTHERS = [
                            (var_other,),
                        ]
                        PLOT_CLEAN_VERSION = True
                        list_dim_timecourse = list(range(NPCS_KEEP))
                        # list_dims = [(0,1), (1,3), (1,2), (2,3), (3,4)]
                        time_bin_size = None # or else will have choppy trajectories
                        pathis.plot_state_space_good_wrapper(savedir, LIST_VAR, LIST_VARS_OTHERS, PLOT_CLEAN_VERSION=PLOT_CLEAN_VERSION,
                                                        list_dim_timecourse=list_dim_timecourse, list_dims=list_dims,
                                                        time_bin_size=time_bin_size)                


def statespace_traj_plot(DFallpa, animal, date, SAVEDIR_ANALYSIS, var_other,
                         manuscript_version=False):
    """ Use the wrapper plotting for state spac,e which includes sclaar, traj, and 
    timecoures.

    Makes "pretty plots"
    """

    var_effect = "seqc_0_shape"

    raw_subtract_mean_each_timepoint = False
    twind_analy = TWIND_ANALY
    tbin_dur = 0.2
    tbin_slide = 0.01

    if int(date)>220720:
        twind = (0.05, 0.9)
    else:
        twind = (0.05, 0.6)

    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
        "04_go_cue":[(-0.3, 0.3)],
        "05_first_raise":[(-0.3, 0.3)],
        "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
    }

    if var_other == "seqc_0_loc":
        list_subspace_projection = ["pca_proj", "pca", "shape", "shape_loc"]
    elif var_other == "gridsize":
        list_subspace_projection = ["pca_proj", "pca", "shape", "shape_size"]
    else:
        assert False

    if manuscript_version:
        list_subspace_projection = ["shape"]
        list_subtr = [False]
        list_smth = [True]
        list_dims = [(0,1), (1,3), (1,2)]
    else:
        list_subtr = [False, True]
        list_smth = [False, True]
        list_dims = [(0,1), (1,3), (1,2), (2,3), (3,4)]

    remove_drift, remove_singleprims_unstable, remove_trials_with_bad_strokes = True, True, True

    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection: 

            ############################
            if subspace_projection in [None, "pca"]:
                list_fit_twind = [twind_analy]
            else:
                list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]
            
            for subspace_projection_fitting_twind in list_fit_twind:
                
                # Final save dir
                SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-nodrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={subspace_projection_fitting_twind}"
                os.makedirs(SAVEDIR, exist_ok=True)
                print("SAVING AT ... ", SAVEDIR)

                # Preprocess
                savedir = f"{SAVEDIR}/preprocess"
                os.makedirs(savedir, exist_ok=True)
                pa = preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, subspace_projection, 
                                   subspace_projection_fitting_twind, twind_analy, tbin_dur, tbin_slide, 
                                   raw_subtract_mean_each_timepoint=False)
                
                ####### Plot state space
                for subtr_time_mean in list_subtr:
                    for do_another_smooth in list_smth:
                        
                        if subtr_time_mean:
                            pathis = pa.norm_subtract_trial_mean_each_timepoint()
                        else:
                            pathis = pa.copy()
                        
                        if do_another_smooth:    
                            # HACKY -- just trying this out to confirm that smoothing worked above (and was not just for binnig for PCA)
                            # Do another round of smoothing (shouldnt realyl matter, but just sanity check)
                            pathis = pathis.agg_by_time_windows_binned(tbin_dur, tbin_slide)

                        savedir = f"{SAVEDIR}/subtrmean={subtr_time_mean}-do_another_smooth={do_another_smooth}"
                        os.makedirs(savedir, exist_ok=True)

                        LIST_VAR = [
                            var_effect,
                        ]
                        LIST_VARS_OTHERS = [
                            (var_other,),
                        ]
                        PLOT_CLEAN_VERSION = True
                        list_dim_timecourse = list(range(NPCS_KEEP))
                        # list_dims = [(0,1), (1,3), (1,2), (2,3), (3,4)]
                        time_bin_size = None # or else will have choppy trajectories
                        pathis.plot_state_space_good_wrapper(savedir, LIST_VAR, LIST_VARS_OTHERS, PLOT_CLEAN_VERSION=PLOT_CLEAN_VERSION,
                                                        list_dim_timecourse=list_dim_timecourse, list_dims=list_dims,
                                                        time_bin_size=time_bin_size)                

def decodercross_plot(DFallpa, SAVEDIR_ANALYSIS):
    """
    Wrapper to performing decoding analys, training one onset subset of data (eg., locations 1) and
    testing on held-out subset (e.g., lcoations2-4), for example, for decoding primitive.

    Decode shpae with across conditios of location or size, then compute scalar summary score
    by averageing over time window.

    MS: checked
    """
    from neuralmonkey.analyses.decode_good import preprocess_factorize_class_labels_ints
    from neuralmonkey.analyses.decode_good import decodewrapouterloop_categorical_timeresolved_within_condition, decodewrapouterloop_categorical_timeresolved_cross_condition
    from pythonlib.tools.pandastools import aggregGeneral
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    MAP_LABELS_TO_INT = preprocess_factorize_class_labels_ints(DFallpa)
    # For each time bin, decode shape genearlizing across location

    SAVEDIR = f"{SAVEDIR_ANALYSIS}/2_cross_condition_decoding"
    os.makedirs(SAVEDIR, exist_ok=True)
    print(SAVEDIR)

    filtdict = None
    separate_by_task_kind = True

    # PARAMS
    # Context
    list_var_decode = [
        "seqc_0_shape",
        "seqc_0_loc",
        "seqc_0_shape",
        "gridsize",
    ]
    list_vars_conj = [
        ["task_kind", "seqc_0_loc"],
        ["task_kind", "seqc_0_shape"],
        ["task_kind", "gridsize"],
        ["task_kind", "seqc_0_shape"],
    ]

    time_bin_size = 0.15
    slide = 0.05
    subtract_mean_vars_conj = False

    # Cross condition
    DFRES_ACROSS = decodewrapouterloop_categorical_timeresolved_cross_condition(DFallpa, list_var_decode,
                                                        list_vars_conj,
                                                        SAVEDIR, time_bin_size=time_bin_size, slide=slide,
                                                        subtract_mean_vars_conj=subtract_mean_vars_conj,
                                                                        filtdict=filtdict,
                                                                        separate_by_task_kind=separate_by_task_kind)
    # Within condition
    DFRES_WITHIN = decodewrapouterloop_categorical_timeresolved_within_condition(DFallpa, list_var_decode,
                                                        list_vars_conj,
                                                        SAVEDIR, time_bin_size=time_bin_size, slide=slide,
                                                                        filtdict=filtdict,
                                                                        separate_by_task_kind=separate_by_task_kind)
    DFRES_ACROSS["decodekind"] = "across"
    DFRES_WITHIN["decodekind"] = "within"

    pd.to_pickle(DFRES_WITHIN, f"{SAVEDIR}/DFRES_WITHIN.pkl")
    pd.to_pickle(DFRES_ACROSS, f"{SAVEDIR}/DFRES_ACROSS.pkl")
    
    #################################
    ### Summaries of comparison of within- vs. across-decoding
    savedir = f"{SAVEDIR}/combine_within_across"
    os.makedirs(savedir, exist_ok=True)

    DFRES_BOTH = pd.concat([DFRES_ACROSS, DFRES_WITHIN]).reset_index(drop=True)

    for var_decode_and_conj in DFRES_BOTH["var_decode_and_conj"].unique():
        dfthis = DFRES_BOTH[DFRES_BOTH["var_decode_and_conj"] == var_decode_and_conj]
        for y in ["score", "score_adjusted"]:
            fig = sns.relplot(data=dfthis, x="time", y=y, hue="decodekind", row="event", col="bregion",
                            kind="line", height=4)
            savefig(fig, f"{savedir}/catplot-var_decode_and_conj={var_decode_and_conj}-y={y}.pdf")
        plt.close("all")
        
    # Get scalar
    for twind in [(0.05, 0.6), (0.05, 0.75), (0.05, 0.9)]:
        dfthis = DFRES_BOTH[(DFRES_BOTH["time"]>=twind[0]) & (DFRES_BOTH["time"]<=twind[1])].reset_index(drop=True)
        dfagg = aggregGeneral(dfthis, ["var_decode", "vars_conj_condition", "var_decode_and_conj", "bregion", "twind", "event",
                            "task_kind", "tk_ev", "decodekind"], ["score", "score_adjusted"])

        # Combine decodekinds into single plot
        fig = sns.catplot(data=dfagg, x="bregion", y="score_adjusted", hue="decodekind", 
                          col="var_decode_and_conj", row="event", kind="bar")
        savefig(fig, f"{savedir}/scalar-twind={twind}-catplot-combined.pdf")

        # One plot for each decodekind (like older plots)
        for decodekind in dfagg["decodekind"].unique().tolist():
            dfaggthis = dfagg[dfagg["decodekind"] == decodekind]
            fig = sns.catplot(data=dfaggthis, x="bregion", y="score_adjusted", col="var_decode_and_conj", row="event", kind="bar")
            savefig(fig, f"{savedir}/scalar-twind={twind}-catplot-decodekind={decodekind}.pdf")

        for var_decode_and_conj in dfagg["var_decode_and_conj"].unique():
            dfthis = dfagg[dfagg["var_decode_and_conj"] == var_decode_and_conj].reset_index(drop=True)

            SIZE = 3.5
            _, fig = plot_45scatter_means_flexible_grouping(dfthis, "decodekind", "within", "across", "event", "score_adjusted", "bregion", shareaxes=True,
                                                SIZE=SIZE, alpha=0.7)    
            if fig is not None:
                savefig(fig, f"{savedir}/scalar-twind={twind}-scatter-var_decode_and_conj={var_decode_and_conj}.pdf")
        plt.close("all")

    # # Plot scalar summaries of decodin
    # savedir = f"{SAVEDIR}/scalar_each"
    # os.makedirs(savedir, exist_ok=True)
    # for twind in [(0.05, 0.6), (0.05, 0.75), (0.05, 0.9)]:
    #     dfthis = DFRES_ACROSS[(DFRES_ACROSS["time"]>twind[0]) & (DFRES_ACROSS["time"]<twind[1])].reset_index(drop=True)
    #     dfagg = aggregGeneral(dfthis, ["var_decode", "vars_conj_condition", "var_decode_and_conj", "bregion", "twind", "event",
    #                         "task_kind", "tk_ev"], ["score", "score_adjusted"])
    #     fig = sns.catplot(data=dfagg, x="bregion", y="score_adjusted", col="var_decode_and_conj", row="event", kind="bar")
    #     savefig(fig, f"{SAVEDIR}/scalar-twind={twind}-across_.pdf")


    #     fig = sns.catplot(data=dfagg, x="bregion", y="score_adjusted", hue="decodekind", col="var_decode_and_conj", row="event", kind="bar")
    #     savefig(fig, f"{savedir}/scalar-twind={twind}-catplot.pdf")


# IGNORE - was trying to incorproate pca_proj, but this fails, beucase "event" is only accessed later, but it is needed for deciding
# On the proj window.. Can solve by rewariting it..
# def statespace_scalar_plot(DFallpa, animal, date, SAVEDIR, var_other):
#     """
#     State space scalar plots, Single figures with subplots each a (bregion, var_other) and within, the dots colored
#     by shape
#     """
#     from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection

#     # First, preprocess all pa
#     # list_pa =[]
#     # for PA in DFallpa["pa"]:
#     #     pa = preprocess_pa(PA, animal, date, var_other, "/tmp", True, None, None, None, None, False)
#     #     plt.close("all")
#     #     list_pa.append(pa)    
#     # DFallpa["pa"] = list_pa

#     ### Collect each PA (doing dim reduction)
#     # (Project onto shape subspace)
#     tbin_dur = "default"
#     tbin_slide = None

#     if int(date)>220720:
#         twind = (0.05, 1.0)
#     else:
#         twind = (0.05, 0.6)

#     raw_subtract_mean_each_timepoint = False

#     SUBSPACE_PROJ_FIT_TWIND = {
#         "03_samp":[twind],
#         "04_go_cue":[(-0.3, 0.3)],
#         "05_first_raise":[(-0.3, 0.3)],
#         "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
#     }

#     if var_other == "seqc_0_loc":
#         list_subspace_projection = ["pca_proj", "pca", "shape", "shape_loc"]
#     elif var_other == "gridsize":
#         list_subspace_projection = ["pca_proj", "pca", "shape", "shape_size"]
#     else:
#         assert False
#     for subspace_projection in list_subspace_projection:

#         if subspace_projection in [None, "pca"]:
#             list_fit_twind = [twind_analy]
#         else:
#             list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]

#         for subspace_projection_fitting_twind in list_fit_twind:
#             savedir = f"{SAVEDIR}/ssproj={subspace_projection}-fit_twind={subspace_projection_fitting_twind}"
#             os.makedirs(savedir, exist_ok=True)

#             for i, row in DFallpa.iterrows():
#                 PA = row["pa"]
#                 PAredu = preprocess_pa(PA, animal, date, var_other, savedir, True, subspace_projection, subspace_projection_fitting_twind,
#                                 twind, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint)

#                 list_pa.append(PAredu)
#             DFallpa["pa_redu"] = list_pa    

#             # dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

#             # # (1) First, dim reduction
#             # superv_dpca_var = superv_dpca_params['superv_dpca_var']
#             # superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
#             # superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

#             # list_pa = []
#             # for i, row in DFallpa.iterrows():
#             #     PA = row["pa"]
        
#             #     _, PAredu = PA.dataextract_dimred_wrapper("scal", dim_red_method, "/tmp", twind, tbin_dur=tbin_dur, tbin_slide = tbin_slide, NPCS_KEEP=10, 
#             #                                 dpca_var=superv_dpca_var, dpca_vars_group=superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict,
#             #                                 raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)    
#             #     list_pa.append(PAredu)
#             #     plt.close("all")
#             # DFallpa["pa_redu"] = list_pa    


#             ### PLOT
#             from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
#             from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots

#             var_col = var_other
#             var_effect = "seqc_0_shape"

#             # Extract event to plot
#             for event in DFallpa["event"].unique().tolist():
#                 # event = "03_samp"
#                 dfallpa = DFallpa[DFallpa["event"]==event].reset_index(drop=True)

#                 nrows = len(dfallpa)

#                 dflab = PA.Xlabels["trials"]
#                 levels_col = dflab[var_col].unique().tolist()
#                 ncols = len(levels_col)
#                 SIZE =5

#                 for dims in [(0,1), (1,2), (2,3), (3,4)]:
#                     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
#                     # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))
#                     for i, row in dfallpa.iterrows():

#                         bregion = row["bregion"]
#                         event = row["event"]
#                         PAredu = row["pa_redu"]     

#                         for j, lev_col in enumerate(levels_col):
#                             try:
#                                 ax = axes[i][j]
#                             except Exception as err:
#                                 print(axes)
#                                 print(i, j)
#                                 raise err

#                             ax.set_title((bregion, lev_col))

#                             pa = PAredu.slice_by_labels_filtdict({var_col:[lev_col]})

#                             if dims[1]<=pa.X.shape[0]-1:
#                                 xs = pa.X[dims[0], :, 0]
#                                 ys = pa.X[dims[1], :, 0]
#                                 # zs = pa.X[2, :, 0]
#                                 dflab = pa.Xlabels["trials"]
#                                 labels = dflab[var_effect].tolist()

#                                 # _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax, plot_3D=False, zs = zs)
#                                 _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax)
#                     if False:
#                         share_axes_row_or_col_of_subplots(axes, "row", "both")   
                    
#                     savefig(fig, f"{savedir}/scatter-event={event}-dims={dims}.pdf")
#                     plt.close("all")


def statespace_scalar_plot(DFallpa, animal, date, SAVEDIR, var_other):
    """
    State space scalar plots, Single figures with subplots each a (bregion, var_other) and within, the dots colored
    by shape
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion

    # First, preprocess all pa
    list_pa =[]
    for PA in DFallpa["pa"]:
        pa = preprocess_pa(PA, animal, date, var_other, "/tmp", True, 
                        None, None, None, None, None, False, skip_dim_reduction=True)    
        plt.close("all")
        list_pa.append(pa)    
    DFallpa["pa"] = list_pa

    ### Collect each PA (doing dim reduction)
    # (Project onto shape subspace)
    tbin_dur = "default"
    tbin_slide = None

    map_event_to_listtwind = {
            "03_samp":[(0.05, 0.6)],
            "05_first_raise":[(-0.1, 0.5)],
            "06_on_strokeidx_0":[(0, 0.5)],
        }
    # only one twind per event
    for k, v in map_event_to_listtwind.items():
        assert len(v)==1

    if False: # old
        if int(date)>220720:
            twind = (0.05, 1.0)
        else:
            twind = (0.05, 0.6)
        
    if var_other == "seqc_0_loc":
        list_subspace_projection = ["umap", "pca", "shape", "shape_loc"]
    elif var_other == "gridsize":
        list_subspace_projection = ["umap", "pca", "shape", "shape_size"]
    else:
        assert False

    for subspace_projection in list_subspace_projection:

        if subspace_projection == "umap":
            # Then do multiple
            n_iters = 3
        else:
            n_iters = 1

        for i_iter in range(n_iters):
            savedir = f"{SAVEDIR}/ssproj={subspace_projection}-iter={i_iter}"
            os.makedirs(savedir, exist_ok=True)

            dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

            # (1) First, dim reduction
            superv_dpca_var = superv_dpca_params['superv_dpca_var']
            superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
            superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

            list_pa = []
            for _, row in DFallpa.iterrows():
                PA = row["pa"]
                event = row["event"]
                bregion = row["bregion"]
                twind = map_event_to_listtwind[event][0]
                savedir_this = f"{savedir}/preprocess-{bregion}-{event}"
                os.makedirs(savedir_this, exist_ok=True)
                _, PAredu = PA.dataextract_dimred_wrapper("scal", dim_red_method, savedir_this, twind, tbin_dur=tbin_dur, tbin_slide = tbin_slide, 
                                            NPCS_KEEP=NPCS_KEEP, 
                                            dpca_var=superv_dpca_var, dpca_vars_group=superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict,
                                            raw_subtract_mean_each_timepoint=False)    
                list_pa.append(PAredu)
                plt.close("all")
            DFallpa["pa_redu"] = list_pa    

            ### PLOT
            from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
            from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots

            # var_col = var_other
            var_effect = "seqc_0_shape"

            # Extract event to plot
            for event in DFallpa["event"].unique().tolist():
                # event = "03_samp"
                dfallpa = DFallpa[DFallpa["event"]==event].reset_index(drop=True)

                trajgood_plot_colorby_splotby_scalar_2dgrid_bregion(dfallpa, var_effect, var_other, savedir, 
                                                                    pa_var = "pa_redu", 
                                                                    prune_min_n_trials=N_MIN_TRIALS_PER_SHAPE, pretty_plot=True, alpha=0.7)

def heatmaps_plot_wrapper_manuscript(DFallpa, animal, date, SAVEDIR_ANALYSIS, var_other="seqc_0_loc", quick_mode=True):
    """
    Plot heatmap of firing rate (after PCA), splittin by shape vs. var_other
    """
    from neuralmonkey.neuralplots.population import heatmapwrapper_many_useful_plots

    raw_subtract_mean_each_timepoint = False
    twind_analy = TWIND_ANALY
    tbin_dur = 0.2
    tbin_slide = 0.01
    if int(date)>220720:
        twind = (0.05, 1.0)
    else:
        twind = (0.05, 0.6)
    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
    }
    list_subspace_projection = ["shape"]

    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
            # plot only cleaned up data.
            list_mean_over_trials = [False, True]
                

            if subspace_projection in [None, "pca"]:
                list_fit_twind = [twind_analy]
            else:
                list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]

            for subspace_projection_fitting_twind in list_fit_twind:
                
                SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-fit_twind={subspace_projection_fitting_twind}"
                os.makedirs(SAVEDIR, exist_ok=True)

                print("SAVING AT ... ", SAVEDIR)

                ############################
                savedir = f"{SAVEDIR}/preprocess"
                os.makedirs(savedir, exist_ok=True)

                pa = _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                            twind_analy, tbin_dur, tbin_slide, savedir)

                list_twind_scal_eucl = None
                var_effect = "seqc_0_shape"
                pa.plot_heatmap_state_euclidean_wrapper(var_effect, var_other, SAVEDIR, 
                                        list_twind_scal_eucl, 
                                        None, None, list_dims=None, twind_base = (-0.6, -0.05),
                                        do_heatmap=True, do_state_space=False, do_euclidean=False,
                                        quick_mode=quick_mode)

def heatmaps_plot_wrapper(DFallpa, animal, date, SAVEDIR_ANALYSIS, var_other="seqc_0_loc",
                          list_subspace_projection=None, quick_mode=False):
    """
    Make all the plots of shape vs. var_other
    """
    from neuralmonkey.neuralplots.population import heatmapwrapper_many_useful_plots

    raw_subtract_mean_each_timepoint = False

    twind_analy = TWIND_ANALY
    tbin_dur = 0.2
    tbin_slide = 0.01

    if int(date)>220720:
        twind = (0.05, 1.0)
    else:
        twind = (0.05, 0.6)

    SUBSPACE_PROJ_FIT_TWIND = {
        "03_samp":[twind],
        "04_go_cue":[(-0.3, 0.3)],
        "05_first_raise":[(-0.3, 0.3)],
        "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
    }

    if list_subspace_projection is None:
        if var_other == "seqc_0_loc":
            list_subspace_projection = ["pca_proj", "pca", "shape", "shape_loc", None]
        elif var_other == "gridsize":
            list_subspace_projection = ["pca_proj", "pca", "shape", "shape_size", None]
        else:
            assert False

    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
            if subspace_projection is None:
                # Then also plot raw, without clearning
                list_unstable_badstrokes = [(False, False, False), (True, True, True)]
                list_mean_over_trials = [True]
            else:
                # plot only cleaned up data.
                list_unstable_badstrokes = [(True, True, True)]
                list_mean_over_trials = [False, True]
                
            for remove_drift, remove_singleprims_unstable, remove_trials_with_bad_strokes in list_unstable_badstrokes:

                if subspace_projection in [None, "pca"]:
                    list_fit_twind = [twind_analy]
                else:
                    list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]

                for subspace_projection_fitting_twind in list_fit_twind:
                    
                    SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-nodrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={subspace_projection_fitting_twind}"
                    os.makedirs(SAVEDIR, exist_ok=True)

                    print("SAVING AT ... ", SAVEDIR)

                    ############################
                    savedir = f"{SAVEDIR}/preprocess"
                    os.makedirs(savedir, exist_ok=True)

                    pa = preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, subspace_projection, 
                                       subspace_projection_fitting_twind, twind_analy, tbin_dur, tbin_slide, 
                                       raw_subtract_mean_each_timepoint=False)

                    if True:
                        # list_mean_zscore_base = [(False, False, False), (False, True, True), (True, False, False), (True, True, False), (False, False, True)]
                        list_twind_scal_eucl = None
                        var_effect = "seqc_0_shape"
                        pa.plot_heatmap_state_euclidean_wrapper(var_effect, var_other, SAVEDIR, 
                                                list_twind_scal_eucl, 
                                                None, None, list_dims=None, twind_base = (-0.6, -0.05),
                                                do_heatmap=True, do_state_space=False, do_euclidean=False,
                                                quick_mode=quick_mode)
                    else:    
                        for subtr_time_mean, zscore, subtr_baseline in [
                            (False, False, False), (False, True, True), (True, False, False), (True, True, False), (False, False, True)
                            ]:

                            pathis = pa.copy()
                            diverge = False

                            if zscore:
                                pathis = pathis.norm_rel_all_timepoints()
                                zlims = None
                                # zlims = [-2, 2]
                                diverge = True
                            else:
                                zlims = None

                            if subtr_time_mean:
                                pathis = pathis.norm_subtract_trial_mean_each_timepoint()
                                diverge = True

                            if subtr_baseline:
                                twind_base = [-0.6, -0.05]
                                pathis = pathis.norm_rel_base_window(twind_base, "subtract")
                                diverge = True

                            for mean_over_trials in list_mean_over_trials:
                                savedirthis = f"{SAVEDIR}/zscore={zscore}-subtr_time_mean={subtr_time_mean}-subtrbase={subtr_baseline}-mean={mean_over_trials}"
                                os.makedirs(savedirthis, exist_ok=True)

                                ######## HEATMAPS
                                heatmapwrapper_many_useful_plots(pathis, savedirthis, "seqc_0_shape", var_other, False, 
                                                                mean_over_trials=mean_over_trials, zlims=zlims, flip_rowcol=True,
                                                                plot_fancy=True, diverge=diverge)  

                                plt.close("all")

                            if False: # This is done in state-space plots now, see the other code in statespace_traj_plot()..
                                ######## SMOOTHED FR
                                savedirthis = f"{SAVEDIR}/zscore={zscore}-subtr_time_mean={subtr_time_mean}/smoothed_fr"
                                os.makedirs(savedirthis, exist_ok=True)
                                chans = pathis.Chans[:8]

                                # make this a grid of (dims, locs)
                                vars_subplots = [var_other]
                                var = "seqc_0_shape"
                                fig = pathis.plotwrappergrid_smoothed_fr_splot_neuron(var, vars_subplots, chans);
                                for ax in fig.axes:
                                    ax.axhline(0, color="k", alpha=0.3)
                                    ax.axvline(0, color="k", alpha=0.3)
                                savefig(fig, f"{savedirthis}/subplot=neuron.pdf")

                                # Grid of (shape, loc)
                                var_col = "seqc_0_shape"
                                var_row = var_other
                                fig = pathis.plotwrappergrid_smoothed_fr_splot_var(var_row, var_col, chans)
                                for ax in fig.axes:
                                    ax.axhline(0, color="k", alpha=0.3)
                                    ax.axvline(0, color="k", alpha=0.3)
                                savefig(fig, f"{savedirthis}/subplot=vars.pdf")

                                plt.close("all")

def timewarped_strokes_wrapper(animal, date, SAVEDIR_ANALYSIS):
    """
    Do decoding, during strokes, using time-warped data, so that all strokes are aligned across warps. 
    This loads previuosly saved trialpop data, which is already warped.

    NOTE: result didnt see much of an effect.
    """
    import pickle
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import preprocess_pa
    import os
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_wrapper

    var_effect = "seqc_0_shape"
    var_other = "gridsize"
    n_min_per_lev = 4

    ### LOAD TRIALPOP
    path = f"/lemur2/lucas/neural_preprocess/PA_trialpop/{animal}-{date}/PA.pkl"
    with open(path, "rb") as f:
        PAwarp = pickle.load(f)

    # Convert to DFallpa
    dflab_chans = PAwarp.Xlabels["chans"]
    res = []
    for br in dflab_chans["bregion_combined"].unique():
        inds = dflab_chans[dflab_chans["bregion_combined"]==br].index.tolist()
        print(br, inds)
        pa = PAwarp.slice_by_dim_indices_wrapper("chans", inds)
        res.append({
            "which_level":"trial",
            "event":"trial_warp",
            "bregion":br,
            "twind":"ignore",
            "pa":pa
        })
    DFallpa = pd.DataFrame(res)

    ### Preprocess
    # So it doesnt fail doesntream stuff
    DFallpa["event"] = "06_on_strokeidx_0"
    do_sitesdirty_extraction = False # This has been failing...
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date, do_sitesdirty_extraction=do_sitesdirty_extraction)

    ### Extract - what is the time window of strokes?
    event_times = PAwarp.Params["event_times_median"]
    event_names = PAwarp.Params["events_all"]
    assert len(event_times)==len(event_names)

    idx1 = event_names.index("on_strokeidx_0")
    idx2 = event_names.index("off_strokeidx_0")

    twind_stroke = [event_times[idx1], event_times[idx2]]
    print("time window, median stroke: ", twind_stroke)

    # subspace_projection = "shape"
    # subspace_projection = "shape_size"

    # subspace_projection = None
    # do_heatmap=True
    # do_state_space=False
    # do_euclidean=True
    LIST_PARAMS = [
        ("shape", True, True, True),
        ("shape_size", True, True, True),
        (None, True, False, True),
    ]

    LIST_DFDIST =[]
    for subspace_projection, do_heatmap, do_state_space, do_euclidean in LIST_PARAMS:
        for i, row in DFallpa.iterrows():
            PA = row["pa"]
            bregion = row["bregion"]
            event = row["event"]
            which_level = row["which_level"]

            SAVEDIR = f"{SAVEDIR_ANALYSIS}/sub={subspace_projection}-bregion={bregion}"
            os.makedirs(SAVEDIR, exist_ok=True)

            PA.Xlabels["trials"]["index_datapt"] = PA.Xlabels["trials"]["trialcode"]

            remove_drift = True
            tbin_dur = 0.15
            tbin_slide = 0.02

            twind_scal = [twind_stroke[0]-0.25, twind_stroke[-1]] 
            LIST_VAR_SS = [
                "seqc_0_shape",
            ]
            LIST_VARS_OTHERS_SS = [
                ["gridsize"],
            ] 
            list_dims = [(0,1), (1,2), (2,3), (3,4)]
            ndims_timecourse=4

            subspace_projection_fitting_twind = twind_scal
            twind_analy = twind_scal

            savedir = f"{SAVEDIR}/preprocess"
            os.makedirs(savedir, exist_ok=True)
            PAthis = preprocess_pa(PA, animal, date, var_other, savedir, remove_drift, 
                                    subspace_projection, subspace_projection_fitting_twind, 
                                    twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False,
                                    skip_dim_reduction=False)

            ### Score only if var_effect has data across all levels of var_other
            from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
            
            dflab = PAthis.Xlabels["trials"]
            levels_var = dflab[var_other].unique().tolist()
            counts_path = f"{savedir}/counts_after_prune.pdf"
            dfout, dict_dfthis = extract_with_levels_of_conjunction_vars_helper(dflab, var_other, [var_effect], 
                                                                                n_min_per_lev, counts_path, lenient_allow_data_if_has_n_levels=len(levels_var),
                                                                                levels_var=levels_var)
            inds = dfout["_index"].tolist()
            PAthis = PAthis.slice_by_dim_indices_wrapper("trials", inds)

            ### RUN PLOTS
            list_dfdist = PAthis.plot_heatmap_state_euclidean_wrapper(var_effect, var_other, SAVEDIR, 
                                                        [twind_scal], 
                                                        LIST_VAR_SS, LIST_VARS_OTHERS_SS, list_dims, ndims_timecourse,
                                                        do_heatmap=do_heatmap, do_state_space=do_state_space, do_euclidean=do_euclidean)

            for dfdist in list_dfdist:
                dfdist["animal"] = animal
                dfdist["date"] = date
                dfdist["bregion"] = bregion
                dfdist["which_level"] = which_level
                dfdist["event"] = event
                dfdist["subspace_projection"] = subspace_projection
                dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]

                LIST_DFDIST.append(dfdist)
    DFDIST = pd.concat(LIST_DFDIST).reset_index(drop=True)
    DFDIST.to_pickle(f"{SAVEDIR_ANALYSIS}/DFDIST.pkl")

    ### PLOTS  
    from neuralmonkey.analyses.euclidian_distance import dfdist_postprocess_wrapper, dfdist_summary_plots_wrapper

    SAVEDIR = f"{SAVEDIR_ANALYSIS}/PLOTS"
    os.makedirs(SAVEDIR, exist_ok=True)

    # Preprocess
    DFDIST, DFDIST_AGG = dfdist_postprocess_wrapper(DFDIST, var_effect, var_other, SAVEDIR, prune_min_n_trials=n_min_per_lev)

    # Plot
    PLOT_EACH_PAIR = False
    dfdist_summary_plots_wrapper(DFDIST, DFDIST_AGG, var_effect, var_other, SAVEDIR,
                                    PLOT_EACH_PAIR=PLOT_EACH_PAIR)

def decode_scalar_confusion(DFallpa, animal, date, SAVEDIR):
    """
    Decode, using scalar (after PCA), and making confusion matrix plots.
    
    In revision, I used this to test if each shape is separated from each other
    shape.

    Also makes UMAP plots.

    NOTE problem: currently pools all locations, instead of testing generalization across locations. It was just 
    easier to code.
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER, dimredgood_nonlinear_embed_data
    from neuralmonkey.analyses.decode_good import decode_categorical, decode_categorical_plot_confusion_score_quick, decode_categorical_cross_condition
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

    ### PARAMS
    # Decode
    var_decode = "seqc_0_shape"
    var_other = "seqc_0_loc"
    NPCS_KEEP = 50 # high, since decoder will find axis in this space.
    # map_event_to_listtwind = {
    #         "03_samp":[(0.05, 0.3), (0.3, 0.6), (0.05, 0.6), (0.5, 1.0)],
    #         "05_first_raise":[(-0.5,  -0.1), (-0.1, 0.5)],
    #         "06_on_strokeidx_0":[(-0.5, -0.1), (0, 0.5)],
    #     }
    map_event_to_listtwind = {
            "03_samp":[(0.05, 0.6)],
            "05_first_raise":[(-0.1, 0.5)],
            "06_on_strokeidx_0":[(0, 0.5)],
        }
    vars_conj_decode = ["seqc_0_loc", "gridsize"]

    # PCA
    subspace_projection = "pca"
    tbin_dur = "default"
    tbin_slide = None

    ### First, preprocess all pa
    list_pa =[]
    for PA in DFallpa["pa"]:
        pa = preprocess_pa(PA, animal, date, var_other, "/tmp", True, 
                        None, None, None, None, None, False, skip_dim_reduction=True)    
        plt.close("all")
        list_pa.append(pa)    
    DFallpa["pa"] = list_pa

    ### For subsampling chans
    # Only cosnider a subset of regions with most shape-related activity.
    bregions_consider = ["preSMA", "SMA", "PMv", "PMd", "vlPFC"] 
    dfallpa_tmp = DFallpa[DFallpa["bregion"].isin(bregions_consider)]
    NCHANS_MIN = min([len(pa.Chans) for pa in dfallpa_tmp["pa"]])
    print("min n chans: ", NCHANS_MIN)

    # Save the n chans per area
    DFallpa["nchans"] = [len(pa.Chans) for pa in DFallpa["pa"]]
    DFallpa.loc[:, ["event", "bregion", "nchans"]].to_csv(f"{SAVEDIR}/num_chans_per_area.csv")

    ### Collect each PA (doing dim reduction)
    # (Project onto shape subspace)

    for _, row in DFallpa.iterrows():
        event = row["event"]
        bregion = row["bregion"]
        PA_orig = row["pa"]
        print(bregion, event)

        ### Subsample num chans randomly
        from pythonlib.tools.statstools import balanced_subsamples
        nchans = len(PA_orig.Chans)
        if nchans<=NCHANS_MIN:
            do_subsample = False
            nsplits = None
        elif nchans > 3 * NCHANS_MIN:
            do_subsample = True
            nsplits = 10
        elif nchans > 2 * NCHANS_MIN:
            do_subsample = True
            nsplits = 8
        elif nchans > 1.5 * NCHANS_MIN:
            do_subsample = True
            nsplits = 6
        else:
            do_subsample = True
            nsplits = 4
        
        if do_subsample:
            # make sure take each ind at least once
            nsplits_at_least = int(np.ceil(nchans / NCHANS_MIN))
            nsplits = np.max([nsplits_at_least, nsplits])

        print("Do subsample? ", do_subsample, "nsplits: ", nsplits, "nchans:", nchans, "NCHANS_MIN: ", NCHANS_MIN)

        if do_subsample:
            subsamples, _ = balanced_subsamples(nchans, nsplits, NCHANS_MIN, PRINT=True)
        else:
            # Just take all the channels
            subsamples = [list(range(len(PA_orig.Chans)))]

        for i_sub, inds_chans in enumerate(subsamples):
            print(f"Currentyl chans subsmaple #{i_sub}: {inds_chans}")
            PA = PA_orig.slice_by_dim_indices_wrapper("chans", inds_chans)

            list_twind = map_event_to_listtwind[event]
            for twind_scal in list_twind:
                savedir = f"{SAVEDIR}/{bregion}-event={event}-twind={twind_scal}-subchaniterv2={i_sub}"
                os.makedirs(savedir, exist_ok=True)

                # note which chans were used
                from pythonlib.tools.expttools import writeStringsToFile, writeDictToTxtFlattened
                chans_subsampled = {
                    "chan_inds":inds_chans,
                    "chans_orig_values":PA_orig.Chans,
                    "chans_sub_values":PA.Chans,
                    "NCHANS_MIN":NCHANS_MIN,
                    "nchans_this":len(PA_orig.Chans)
                }
                writeDictToTxtFlattened(chans_subsampled, f"{savedir}/chans_subsampled.txt")

                ### Dim reductions
                dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)
                superv_dpca_var = superv_dpca_params['superv_dpca_var']
                superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
                superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']
                _, PAredu = PA.dataextract_dimred_wrapper("scal", dim_red_method, savedir, twind_scal, 
                                            tbin_dur=tbin_dur, tbin_slide = tbin_slide, 
                                            NPCS_KEEP=NPCS_KEEP, 
                                            dpca_var=superv_dpca_var, dpca_vars_group=superv_dpca_vars_group, 
                                            dpca_filtdict=superv_dpca_filtdict,
                                            raw_subtract_mean_each_timepoint=False)    

                
                ### Prune data
                dflab = PAredu.Xlabels["trials"]
                balance_no_missed_conjunctions = True
                prune_min_n_trials = N_MIN_TRIALS_PER_SHAPE
                prune_min_n_levs = 2
                plot_counts_heatmap_savepath = f"{savedir}/counts.pdf"
                dfout, _ = extract_with_levels_of_conjunction_vars(dflab, var_decode, vars_conj_decode,
                                                                        n_min_across_all_levs_var=prune_min_n_trials,
                                                                        lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                                                                        prune_levels_with_low_n=True,
                                                                        ignore_values_called_ignore=True,
                                                                        plot_counts_heatmap_savepath=plot_counts_heatmap_savepath,
                                                                        balance_no_missed_conjunctions=balance_no_missed_conjunctions)
                plt.close("all")
                # if len(dfout)==0:
                #     print("all data pruned!!")
                #     return None
                # Only keep the indices in dfout
                PAredu = PAredu.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)

                ###
                X = PAredu.X.squeeze().T # (ntrials, ndims)
                dflab = PAredu.Xlabels["trials"]

                ### UMAP plot
                Xredu, _ = dimredgood_nonlinear_embed_data(X, "umap")
                trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu, dflab, var_decode, savedir)
                trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu, dflab, var_decode, savedir, vars_subplot=vars_conj_decode)

                ### Decoding
                # for decode_across in [False, True]:
                for decode_across in [True]: # True is better
                    # for do_std in [False, True]:
                    for do_std in [False]: # False is better
                        if decode_across:
                            savedir_this = f"{savedir}/decode-across={decode_across}-varsother={vars_conj_decode}-do_std={do_std}"
                            os.makedirs(savedir_this, exist_ok=True)
                            dfres, _ = decode_categorical_cross_condition(X, dflab, var_decode, vars_conj_decode,
                                                                do_center=True, do_std=do_std)
                            decode_categorical_plot_confusion_score_quick(dfres, savedir_this)

                        else:
                            savedir_this = f"{savedir}/decode-across={decode_across}-do_std={do_std}"
                            os.makedirs(savedir_this, exist_ok=True)
                            RES = decode_categorical(X, dflab[var_decode].tolist(), N_MIN_TRIALS_PER_SHAPE, max_nsplits=None,
                                                    do_center=True, do_std=do_std, plot_resampled_data_path_nosuff=savedir_this,
                                                                    return_mean_score_over_splits=False, 
                                                                    return_predictions_all_trials=True)
                            dfres = pd.DataFrame(RES)
                            decode_categorical_plot_confusion_score_quick(RES, savedir_this)

                        # save final data, to sumamrize across regions
                        import pickle
                        path = f"{savedir_this}/dfres.pkl"
                        with open(path, "wb") as f:
                            pickle.dump(dfres, f)
                
                ### DECODING, pairwise -- For each pair of shapes, do decoding
                do_std = False
                vars_conj_condition = ["seqc_0_loc", "gridsize"]
                savedir_this = f"{savedir}/decodepairwise-varsother={vars_conj_decode}-do_std={do_std}"
                os.makedirs(savedir_this, exist_ok=True)

                from neuralmonkey.analyses.decode_good import decode_categorical_pairwise
                do_across_condition = True
                DFRES, DFRES_COUNTS, dfres_scores = decode_categorical_pairwise(X, dflab, var_decode, N_MIN_TRIALS_PER_SHAPE, savedir_this,
                                        do_across_condition, vars_conj_condition=vars_conj_condition,
                                        do_std=do_std)
                
                # save final data, to sumamrize across regions
                import pickle
                path = f"{savedir_this}/DFRES.pkl"
                with open(path, "wb") as f:
                    pickle.dump(DFRES, f)

                path = f"{savedir_this}/DFRES_COUNTS.pkl"
                with open(path, "wb") as f:
                    pickle.dump(DFRES_COUNTS, f)

                path = f"{savedir_this}/dfres_scores.pkl"
                with open(path, "wb") as f:
                    pickle.dump(dfres_scores, f)


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

    animal = sys.argv[1]
    date = int(sys.argv[2])
    var_other = sys.argv[3] # seqc_0_loc or gridsize

    version = "trial"
    combine = True
    question = "SP_BASE_trial"

    # PLOTS_DO = [1, 2]
    # PLOTS_DO = [0, 1, 2]
    # PLOTS_DO = [4]
    # PLOTS_DO = [2, 5, 0]

    # Euclidian shuffle
    # PLOTS_DO = [4, 5]
    # PLOTS_DO = [0]
    
    # --- Good:
    PLOTS_DO = [0, 5, 4, 2, 3] # Good
    PLOTS_DO = [4] # Good
    # PLOTS_DO = [4]
    # PLOTS_DO = [2, 3] # Good

    # # Timewarped during stroke.
    # PLOTS_DO = [6] # Good

    # # Decode, confusion matrix
    # PLOTS_DO = [7] # Good

    # # Scalar state space plots
    # PLOTS_DO = [1] # 

    ######################################
    if any([x!=6 for x in PLOTS_DO]):
        # Load a single DFallPA
        DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                        question=question)
        # if DFallpa is None:
        #     # Then extract
        #     DFallpa = extract_dfallpa_helper(animal, date, question, combine, do_save=True)

        # Make a copy of all PA before normalization
        dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

        ################ PARAMS
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/shape_invariance"
        # os.makedirs(SAVEDIR, exist_ok=True)
        # print(SAVEDIR)

    ################################### PLOTS
    for plotdo in PLOTS_DO:
        if plotdo==0:
            # (1) Heatmaps, split in many ways, to show population data
            savedir = f"{SAVEDIR}/HEATMAPS/{animal}-{date}-combine={combine}-var_other={var_other}"
            os.makedirs(savedir, exist_ok=True)
            heatmaps_plot_wrapper(DFallpa, animal, date, savedir, var_other=var_other)

        elif plotdo==1:
            # (2) Scalar state space.
            savedir = f"{SAVEDIR}/SCALAR_SS/{animal}-{date}-combine={combine}-var_other={var_other}"
            os.makedirs(savedir, exist_ok=True)
            statespace_scalar_plot(DFallpa, animal, date, savedir, var_other)
        
        elif plotdo==2:
            # Decode (cross-condition)
            savedir = f"{SAVEDIR}/DECODE/{animal}-{date}-combine={combine}-var_other={var_other}"
            os.makedirs(savedir, exist_ok=True)
            decodercross_plot(DFallpa, savedir)
        
        elif plotdo==3:
            # Time-varying euclidian distnaces (e.g., same|diff).
            # MULT ANALYSIS: see notebook 241110_shape_invariance_all_plots_SP
            # ... [LOAD MULT DATA] Euclidian (time-varying, here summarize)
            savedir = f"{SAVEDIR}/EUCLIDIAN/{animal}-{date}-combine={combine}-var_other={var_other}"
            os.makedirs(savedir, exist_ok=True)
            euclidian_time_resolved(DFallpa, animal, date, var_other, savedir)

        elif plotdo==4:
            # Euclidian Shuff. This is better than plotdo==3, because:
            # - 
            # MULT-- methods to load the results, see 
            # notebook: /home/lucas/code/neuralmonkey/neuralmonkey/notebooks_tutorials/241110_shape_invariance_all_plots_SP.ipynb
            # The methods are labeled from #1 to #4.
            # --> #3 is the good one I worked with (under the main header:
            # # [#4] Euclidian, doing stats (and faster way to compute)
            # [#4.4] --> just to plot across all days (ignoring shuffles)

            # FINAL methods for MULT:
            # (1) Summary plots     -->     [#4.4]
            # (2) Stats             -->     Stats method # 4 -- linear model, at level of trials
            if False:
                if var_other == "seqc_0_loc":
                    var_context_same = "gridsize"
                elif var_other == "gridsize":
                    var_context_same = "seqc_0_loc"
                else:
                    assert False
            else:
               var_context_same = None

            if var_context_same is None:
                savedir = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-combine={combine}-var_other={var_other}"
            else:
                savedir = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-combine={combine}-var_other={var_other}-var_context_same={var_context_same}"
            os.makedirs(savedir, exist_ok=True)
            DO_RSA_HEATMAPS = True
            DO_SHUFFLE = False
            euclidian_time_resolved_fast_shuffled(DFallpa, animal, date, var_other, savedir, 
                                                  DO_SHUFFLE=DO_SHUFFLE, DO_RSA_HEATMAPS=DO_RSA_HEATMAPS,
                                                  var_context_same=var_context_same)

        elif plotdo==4.3:
            # Load results from Euclidian Shuff, and do shuffle stats and make summary plots.
            # v3 shuffle method.
            # ** Uses saved results from plotdo == 4
            n_shuff = 101
            # n_shuff = 10000

            for var_stable, var_shuffle in [
                ("seqc_0_shape", "seqc_0_loc"), 
                ("seqc_0_loc", "seqc_0_shape")]:
                euclidian_time_resolved_fast_shuffled_mult_stats_v3(animal, date, var_other, 
                    var_shuffle, var_stable, n_shuff = n_shuff, PLOT_SHUFFLE_HEATMAP=False)

        elif plotdo==5:
            # (2) Traj state space.
            savedir = f"{SAVEDIR}/TRAJ_SS/{animal}-{date}-combine={combine}-var_other={var_other}"
            os.makedirs(savedir, exist_ok=True)
            statespace_traj_plot(DFallpa, animal, date, savedir, var_other)
        
        elif plotdo==6:
            """
            Timewarped euclidean distance, during storkes. The goal was to have a better analysis of 
            eucl dist during strokes, controlling for motor effects -- would expect PMv to show even stronger effect.
            """
            # for animal, date in [
            #     ("Diego", 230619),
            #     ("Pancho", 220716),
            #     ("Pancho", 220717),
            #     ("Pancho", 240530),
            #     ("Diego", 230618),
            #     ]:
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/shape_invariance/TIMEWARPED/{animal}-{date}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

            timewarped_strokes_wrapper(animal, date, SAVEDIR_ANALYSIS)
        
        elif plotdo==7:
            """
            Decode, testing confusion matrix across shapes. Does all kinds of decoding, but the one I used is pairwise
            decoding, done separately for each pair of shapes.
            """

            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/shape_invariance/decode_confusion/{animal}-{date}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            decode_scalar_confusion(DFallpa, animal, date, SAVEDIR_ANALYSIS)

        else:
            print(PLOTS_DO)
            assert False