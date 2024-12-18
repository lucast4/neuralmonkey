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
                                 n_min_per_lev_lev_others=2):
    """
    Helper to do dim reductions.
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    if subspace_projection is not None:
        dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

        # (1) First, dim reduction
        superv_dpca_var = superv_dpca_params['superv_dpca_var']
        superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']
        
        print("***:", dim_red_method, " fit twind: ", subspace_projection_fitting_twind, ", data twind: ", twind_analy)
        _, PA = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
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
                skip_dim_reduction=False):
    """
    Holds all prepreocessing.

    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, extract_with_levels_of_conjunction_vars_helper
    
    dflab = PA.Xlabels["trials"]
    dflab = append_col_with_grp_index(dflab, ["seqc_0_shape", "seqc_0_loc"], "seqc_0_shapeloc")    
    dflab = append_col_with_grp_index(dflab, ["seqc_0_shape", "gridsize"], "seqc_0_shapesize")    
    PA.Xlabels["trials"] = dflab

    ################################### PRUNE TO GOOD N COUNTS
    ### (0) Plot original tabulation of shape vs task_klind
    dflab = PA.Xlabels["trials"]
    fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "seqc_0_shape", "seqc_0_loc", ["task_kind", "gridsize"])
    path = f"{savedir}/shape_counts-orig.pdf"
    fig.savefig(path)


    # (2) Keep just shapes taht exist across both SP and CHAR.
    dflab = PA.Xlabels["trials"]
    _dfout,_  = extract_with_levels_of_conjunction_vars_helper(dflab, "seqc_0_shape", [var_other], n_min_per_lev=N_MIN_TRIALS_PER_SHAPE,
                                                plot_counts_heatmap_savepath=None)
    # if len(_dfout)==0 or _dfout is None:
    #     print("Pruned all data!! returning None")
    #     print("... using these params: ", shape_var, task_kinds)
    #     return None
    #     # assert False, "not enough data"

    index_datapt_list_keep = _dfout["index_datapt"].tolist()
    PA = PA.slice_by_labels_filtdict({"index_datapt":index_datapt_list_keep})

    ### Plot counts one final time
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
                                 twind_analy, tbin_dur, tbin_slide, savedir, raw_subtract_mean_each_timepoint=False)

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
                                          DO_SHUFFLE=False, DO_RSA_HEATMAPS=False):
    """
    Good! Much faster method. And does two other things important for more rigorous result:
    1. Train-test split for dim redu (separate for fitting[smaller] and then final data projected)

    Also gets shuffled data.

    PARAMS:
    - DO_SHUFFLE, bool, if True, then gets shuffle by shuffling 3 ways, all working at level of trial.
    To analyze and plot results of shuffle, use Stats method #1
    (See notebookb: /home/lucas/code/neuralmonkey/neuralmonkey/notebooks_tutorials/241110_shape_invariance_all_plots_SP.ipynb)
    --> I stopped doing shuffle, since all I wanted was the dfdists. I then run shuffling on those, using method #3.
    - DO_RSA_HEATMAPS, bool. This may take significantly more time...
    """
    import seaborn as sns
    # assert False, "save split, or else file gets to mult GB -- run seprately for each bregion."
    import seaborn as sns

    var_effect = "seqc_0_shape"
    vars_group = [var_effect, var_other]

    N_SPLITS = 6
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

    map_event_to_listtwind = {
            "03_samp":[(0.05, 0.3), (0.3, 0.6), (0.05, 0.6), (0.5, 1.0)],
            "05_first_raise":[(-0.5,  -0.1), (-0.1, 0.5)],
            "06_on_strokeidx_0":[(-0.5, -0.1), (0, 0.5)],
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
                            timevarying_compute_fast_to_scalar(pa, vars_group, rsa_heatmap_savedir=savedir)

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
                    folds_dflab = PAthis.split_balanced_stratified_kfold_subsample_level_of_var(vars_group, None, None, 
                                                                                                n_splits=N_SPLITS, 
                                                                                                do_balancing_of_train_inds=False)

                    for _i_dimredu, (train_inds, test_inds) in enumerate(folds_dflab):
                        # train_inds, more inds than than test_inds
                        train_inds = [int(i) for i in train_inds]
                        test_inds = [int(i) for i in test_inds]

                        savedir = f"{SAVEDIR}/preprocess/i_dimredu={_i_dimredu}"
                        os.makedirs(savedir, exist_ok=True)
                        PAthisRedu = _preprocess_pa_dim_reduction(PAthis, subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide, savedir=savedir, raw_subtract_mean_each_timepoint=False,
                                    inds_pa_fit=test_inds, inds_pa_final=train_inds)

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
                            dfdist, _ = timevarying_compute_fast_to_scalar(pa, vars_group)
                            
                            dfdist["bregion"] = bregion
                            dfdist["which_level"] = which_level
                            dfdist["event"] = event
                            dfdist["subspace_projection"] = subspace_projection
                            dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]
                            dfdist["shuffle_ver"] = "ignore"
                            dfdist["shuffled"] = False
                            dfdist["shuffle_iter"] = 0
                            dfdist["dim_redu_fold"] = _i_dimredu
                            dfdist["twind_scal"] = [twind_scal for _ in range(len(dfdist))]
                            list_dfdist.append(dfdist)

                            # (2) Shuffle
                            if DO_SHUFFLE:
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
                                        dfdist_shuff, _ = timevarying_compute_fast_to_scalar(pa_shuff, vars_group)

                                        dfdist_shuff["bregion"] = bregion
                                        dfdist_shuff["which_level"] = which_level
                                        dfdist_shuff["event"] = event
                                        dfdist_shuff["subspace_projection"] = subspace_projection
                                        dfdist_shuff["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist_shuff))]
                                        dfdist_shuff["shuffle_ver"] = shuffle_ver
                                        dfdist_shuff["shuffled"] = True
                                        dfdist_shuff["shuffle_iter"] = i_shuff
                                        dfdist_shuff["dim_redu_fold"] = _i_dimredu
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
    
    # Plots are in notebok.
    
    # from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
    # from pythonlib.tools.pandastools import append_col_with_grp_index
    # from pythonlib.tools.pandastools import aggregGeneral

    # DFDISTS = pd.concat(list_dfdist).reset_index(drop=True)
    # DFDISTS_SHUFF = pd.concat(list_dfdist_shuff).reset_index(drop=True)
    # DFDISTS = datamod_reorder_by_bregion(DFDISTS)
    # DFDISTS_SHUFF = datamod_reorder_by_bregion(DFDISTS_SHUFF)

    # DFDISTS = append_col_with_grp_index(DFDISTS, ["subspace_projection", "subspace_projection_fitting_twind"], "subspace|twind")
    # DFDISTS_SHUFF = append_col_with_grp_index(DFDISTS_SHUFF, ["subspace_projection", "subspace_projection_fitting_twind"], "subspace|twind")
    # DFDISTS_SHUFF = append_col_with_grp_index(DFDISTS_SHUFF, ["dim_redu_fold", "shuffle_iter"], "drfold|shuffiter")

    # pd.to_pickle(DFDISTS, f"{SAVEDIR_ANALYSIS}/DFDISTS.pkl")
    # pd.to_pickle(DFDISTS_SHUFF, f"{SAVEDIR_ANALYSIS}/DFDISTS_SHUFF.pkl")

    # ############## PLOTS (ot completed).
    # # Agg over all dim redu splits
    # DFDISTS = aggregGeneral(DFDISTS, ["same-seqc_0_shape|seqc_0_loc", "bregion", "which_level", "event", 
    #                             "subspace_projection", "subspace_projection_fitting_twind", "shuffle_ver", "shuffled", "shuffle_iter",
    #                             "subspace|twind", "labels_1", "labels_2"], ["dist_mean", "dist_norm", "dist_yue_diff", 
    #                                                                         "DIST_50", "DIST_98"], nonnumercols="all")

    # # Agg over all conditions
    # DFDISTS_AGG = aggregGeneral(DFDISTS, ["same-seqc_0_shape|seqc_0_loc", "bregion", "which_level", "event", 
    #                             "subspace_projection", "subspace_projection_fitting_twind", "shuffle_ver", "shuffled", "shuffle_iter",
    #                             "subspace|twind"], ["dist_mean", "dist_norm", "dist_yue_diff", 
    #                                                                         "DIST_50", "DIST_98"], nonnumercols="all")
    
    # # Get distribition over shuffs
    # # - first, agg over conditions, one datapt per shuff
    # DFDISTS_SHUFF_AGG = aggregGeneral(DFDISTS_SHUFF, ["same-seqc_0_shape|seqc_0_loc", "bregion", "which_level", "event", 
    #                             "subspace_projection", "subspace_projection_fitting_twind", "shuffle_ver", "shuffled", "drfold|shuffiter",
    #                             "subspace|twind"], ["dist_mean", "dist_norm", "dist_yue_diff", 
    #                                                                         "DIST_50", "DIST_98"])


    # fig = sns.catplot(data=DFDISTS_AGG, x="bregion", hue="same-seqc_0_shape|seqc_0_loc", y=y, col="event", row="subspace|twind", aspect=2)
    # fig = sns.catplot(data=DFDISTS, x="bregion", hue="same-seqc_0_shape|seqc_0_loc", y=y, kind="bar", col="event", row="subspace|twind", aspect=2)
    # fig = sns.catplot(data=DFDISTS, x="bregion", hue="same-seqc_0_shape|seqc_0_loc", y="dist_mean", kind="bar", col="event", row="subspace|twind", aspect=2)

    # # Compare to shuffle
    # y = "dist_yue_diff"
    # fig = sns.catplot(data=DFDISTS_SHUFF_AGG, x="bregion", hue="same-seqc_0_shape|seqc_0_loc", y=y, 
    #                 col="event", aspect=2, jitter=True, alpha=0.5)
    # fig = sns.catplot(data=DFDISTS_SHUFF_AGG, x="bregion", hue="same-seqc_0_shape|seqc_0_loc", y="dist_mean", 
    #                 col="event", aspect=2, jitter=True, alpha=0.5)

    # # get p-values and plots that overlay shuff with data

    # from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    # grpvars = ["same-seqc_0_shape|seqc_0_loc", "bregion", "which_level", "event", "subspace|twind"]
    # grpdict_data = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grpvars)
    # grpdict_shuff = grouping_append_and_return_inner_items_good(DFDISTS_SHUFF_AGG, grpvars)

    # # Keep only useful groups
    # keeps = ["1|0", "0|1"]
    # # DFDISTS_AGG[DFDISTS_AGG["same-seqc_0_shape|seqc_0_loc"].isin(keeps)]
    # grpdict_data = {grp:inds for grp, inds in grpdict_data.items() if grp[0] in keeps}
    # grpdict_shuff = {grp:inds for grp, inds in grpdict_shuff.items() if grp[0] in keeps}

    # for grp, inds_data in grpdict_data.items():
    #     inds_shuff = grpdict_shuff[grp]
    #     print(inds_data, " -- ", inds_shuff, " -- ", len(inds_shuff))

    # tmp = DFDISTS_AGG.iloc[inds_data]["dist_yue_diff"].values
    # assert(len(tmp))==1
    # val_dat = tmp[0]
    # vals_shuff = DFDISTS_SHUFF_AGG.iloc[inds_shuff]["dist_yue_diff"].values

    # vals_shuff
    # from pythonlib.tools.statstools import empiricalPval
    # p = empiricalPval(val_dat, vals_shuff)

    # DFDISTS_ALL_AGG = pd.concat([DFDISTS_AGG, DFDISTS_SHUFF_AGG]).reset_index(drop=True)
    # DFDISTS_ALL_AGG.columns
    # for event in DFDISTS_ALL_AGG["event"].unique():
    #     dfthis = DFDISTS_ALL_AGG[DFDISTS_ALL_AGG["event"]==event]
    #     sns.catplot(data=dfthis, x="bregion", y ="dist_yue_diff", col="same-seqc_0_shape|seqc_0_loc", row="event", hue="shuffled", alpha=0.4)
        
    #     sns.catplot(data=dfthis, x="bregion", y ="dist_mean", col="same-seqc_0_shape|seqc_0_loc", row="event", hue="shuffled", alpha=0.4)
    #     sns.catplot(data=dfthis, x="bregion", y ="DIST_50", col="same-seqc_0_shape|seqc_0_loc", row="event", hue="shuffled", alpha=0.4)    
        
def euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other, also_load_shuffled=False,
                                                      rename_event_with_eventtwind=True, events_keep=None,
                                                      analysis_kind="shape_invar", convert_to_df_with_postprocessing=False):
    """

    """
    import pickle
    import os

    # animal = "Diego"
    # date = 230615
    combine = True
    # var_other = "seqc_0_loc"

    if analysis_kind == "shape_invar":
        var_effect = "seqc_0_shape"
        var_other = var_other
        SAVEDIR_ORIG = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF"
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-var_other={var_other}"
    elif analysis_kind == "char_sp":
        # /lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE/Diego-231120-combine=True-wl=stroke
        var_effect = "shape_semantic_grp"
        var_other = "task_kind"
        wl = "stroke"
        SAVEDIR_ORIG = "/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE"
        SAVEDIR = f"{SAVEDIR_ORIG}/{animal}-{date}-combine={combine}-wl={wl}"
    else:
        assert False

    SAVEDIR_PLOTS = f"{SAVEDIR}/RELOADED_PLOTS"
    os.makedirs(SAVEDIR_PLOTS, exist_ok=True)
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
        
    # Now, events have multiple possible twind_scalar. to allow compatible with code,
    # rename events to events+scal.
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
        # savedir = f"{SAVEDIR_PLOTS}/stats_method_3_using_shuffle_dist_mat-nshuff={n_shuff}"
        # os.makedirs(savedir, exist_ok=True)
        from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
        from pythonlib.tools.pandastools import append_col_with_grp_index, aggregGeneral
        import pandas as pd

        # First, generate all df
        print("concatting...")
        DFDISTS = pd.concat(list_dfdist).reset_index(drop=True)
        DFDISTS = datamod_reorder_by_bregion(DFDISTS)

        if "prune_version" not in DFDISTS:
            DFDISTS["prune_version"] = "none"

        print("appending...")
        DFDISTS = append_col_with_grp_index(DFDISTS, ["subspace_projection", "subspace_projection_fitting_twind"], "subspace|twind")
        # DFDISTS = append_col_with_grp_index(DFDISTS, ["shuffled", "shuffle_ver"], "shuffled|ver")

        # display(DFDISTS)
        # assert False
        print("agging...")
        # Agg over all dim redu splits
        var_same_same = f"same-{var_effect}|{var_other}"
        DFDISTS = aggregGeneral(DFDISTS, [var_same_same, "bregion", "prune_version", "which_level", "event", 
                                          "subspace_projection", "subspace_projection_fitting_twind", "subspace|twind", "twind_scal", "labels_1", "labels_2"], 
                                ["dist_mean", "dist_norm", "dist_yue_diff", "DIST_50", "DIST_98"])


        ## FINAL AGGS
        # (2) each datapt is 0|1, 1|0, 1|1, 0|0 (i.e., 4 datapts per bregion/metaparams)
        # Agg over all conditions (e.g. label pairs)
        DFDISTS_AGG = aggregGeneral(DFDISTS, ["bregion", "prune_version", "which_level", "event", "subspace|twind", var_same_same],
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

        print(grp)
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
            print(grp)
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
        import numpy as np
        import seaborn as sns

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

def euclidian_time_resolved_fast_shuffled_mult_scatter_plots(analysis_kind="shape_invar", just_return_df=False):
    """
    GOOD -- load and plot across days, without bothering about shuffle stats.
    Loads data saved by euclidian_time_resolved_fast_shuffled, across dates, and concats, and plots.

    PARAMS:
    - analysis_kind, "shape_invar", "char_sp"

    Works for both (i) shape invar and (ii) char_sp. 
    Goal is for this to work for anything. 

    """
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import euclidian_time_resolved_fast_shuffled_mult_reload
    import os
    from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_append_and_return_inner_items_good
    from pythonlib.tools.pandastools import aggregGeneral, stringify_values
    import pandas as pd
    import matplotlib.pyplot as plt

    if analysis_kind=="shape_invar":
        SAVEDIR_MULT = "/lemur2/lucas/analyses/recordings/main/shape_invariance/EUCLIDIAN_SHUFF/MULT"
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
            ("Pancho", "task_kind"),
            ("Diego", "task_kind"),
            ]
    else:
        assert False

    for animal, var_other in LIST_ANIMAL_VAROTHER:

        if analysis_kind == "shape_invar":
            if animal == "Diego" and var_other == "seqc_0_loc":
                # list_date = [230614, 230615]
                list_date = [230614, 230615, 240508, 240509, 240510, 240513, 240530]
                list_date = [230614, 230615, 240508, 240509, 240510, 240530]
            elif animal == "Diego" and var_other == "gridsize":
                # list_date = [230618, 230619]
                list_date = [230618, 230619, 240510, 240530]
            elif animal == "Pancho" and var_other == "seqc_0_loc":
                # list_date = [220608, 220715, 220717]
                # list_date = [220606, 220608, 220609, 220610, 220715, 220717, 220724, 220918, 221218, 240508, 240509, 240510, 240515, 240530]                
                list_date = [220715, 220717, 220724, 220918, 240508, 240509, 240510, 240530]                
            elif animal == "Pancho" and var_other == "gridsize":
                # list_date = [220716, 220717]
                list_date = [220606, 220608, 220716, 220717, 220918, 240510, 240530]
            else:
                assert False
        elif analysis_kind == "char_sp":
            if animal == "Diego":
                # list_date = [231205, 231122, 231128, 231129, 231201, 231120, 231121, 231206, 231218, 231220] # A date fails.
                list_date = [231205, 231122, 231128, 231129, 231201, 231120, 231206, 231218, 231220]
                # list_date = [231220]
            elif animal == "Pancho":
                list_date = [220618, 220626, 220628, 220630, 230119, 230120, 230126, 230127]
            else:
                assert False
        else:
            print(analysis_kind)
            assert False

        SAVEDIR = f"{SAVEDIR_MULT}/analy={analysis_kind}-{animal}-varother={var_other}-{min(list_date)}-{max(list_date)}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(SAVEDIR)

        list_df = []
        for date in list_date:
            _, dfdists_agg, _ = euclidian_time_resolved_fast_shuffled_mult_reload(animal, date, var_other=var_other, analysis_kind=analysis_kind,
                                                                                  convert_to_df_with_postprocessing=True)
            dfdists_agg["animal"] = animal
            dfdists_agg["date"] = date
            list_df.append(dfdists_agg)

        DFDISTS_AGG = pd.concat(list_df).reset_index(drop=True)

        # Agg all the "metaparams"
        from pythonlib.tools.pandastools import append_col_with_grp_index
        DFDISTS_AGG = append_col_with_grp_index(DFDISTS_AGG, ["prune_version", "subspace|twind", "event"], "metaparams")

        # SAVE COUNTS
        for var in ["prune_version", "subspace|twind", "event"]:
            from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            fig = grouping_plot_n_samples_conjunction_heatmap(DFDISTS_AGG, "date", var, ["animal"])
            savefig(fig, f"{SAVEDIR}/groupcounts-var={var}.pdf")
            plt.close("all")

        if just_return_df:
            return DFDISTS_AGG
        
        from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping, grouping_append_and_return_inner_items_good
        var_same_same = f"same-{var_effect}|{var_other}"
        var_value = "dist_yue_diff"

        savedir = f"{SAVEDIR}/scatterplots"
        os.makedirs(savedir, exist_ok=True)

        # Each event
        grp_vars = ["which_level", "prune_version", "event"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "subspace|twind", 
                                                var_value, "bregion", True, shareaxes=True)
            
            savefig(fig, f"{savedir}/EVENTS-scatter45-grp={grp}.pdf")
            plt.close("all")
            
        # Each event
        grp_vars = ["which_level", "prune_version", "subspace|twind"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "event", 
                                                var_value, "bregion", True, shareaxes=True)
            
            savefig(fig, f"{savedir}/SUBSPACE-scatter45-grp={grp}.pdf")
            plt.close("all")

        # Show each date
        grp_vars = ["which_level", "event", "prune_version", "subspace|twind"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "bregion", 
                                                var_value, "date", True, shareaxes=True)
            
            savefig(fig, f"{savedir}/DATES-scatter45-grp={grp}.pdf")
            plt.close("all")

        grp_vars = ["bregion"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "metaparams", 
                                                var_value, "date", True, shareaxes=True, SIZE=3.5)
            
            savefig(fig, f"{savedir}/REGIONS-scatter45-grp={grp}.pdf")
            plt.close("all")


        ### Plot
        dfthis_dat = DFDISTS_AGG
        _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "metaparams", 
                                            var_value, "bregion", True, shareaxes=True, SIZE=3.5)

        savefig(fig, f"{savedir}/ALL.pdf")
        plt.close("all")

        # Each event
        grp_vars = ["which_level", "prune_version", "subspace|twind", "event"]
        grpdict_dat = grouping_append_and_return_inner_items_good(DFDISTS_AGG, grp_vars)

        for grp, inds_dat in grpdict_dat.items():
            # inds_pval = grpdict_pval[grp]
            dfthis_dat = DFDISTS_AGG.iloc[inds_dat]

            ### Plot
            _, fig = plot_45scatter_means_flexible_grouping(dfthis_dat, var_same_same, "1|0", "0|1", "date", 
                                                var_value, "bregion", True, shareaxes=True)
            
            savefig(fig, f"{savedir}/ALL_DATES-scatter45-grp={grp}.pdf")
            plt.close("all")


def statespace_traj_plot(DFallpa, animal, date, SAVEDIR_ANALYSIS, var_other):
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

    remove_drift, remove_singleprims_unstable, remove_trials_with_bad_strokes = True, True, True

    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection: #

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
                for subtr_time_mean in [False, True]:
                    if subtr_time_mean:
                        pathis = pa.norm_subtract_trial_mean_each_timepoint()
                    else:
                        pathis = pa.copy()
                    
                    savedir = f"{SAVEDIR}/subtrmean={subtr_time_mean}"
                    os.makedirs(savedir, exist_ok=True)

                    LIST_VAR = [
                        var_effect,
                    ]
                    LIST_VARS_OTHERS = [
                        (var_other,),
                    ]
                    PLOT_CLEAN_VERSION = True
                    list_dim_timecourse = list(range(NPCS_KEEP))
                    list_dims = [(0,1), (1,2), (2,3), (3,4)]
                    pathis.plot_state_space_good_wrapper(SAVEDIR, LIST_VAR, LIST_VARS_OTHERS, PLOT_CLEAN_VERSION=PLOT_CLEAN_VERSION,
                                                    list_dim_timecourse=list_dim_timecourse, list_dims=list_dims)                

def decodercross_plot(DFallpa, SAVEDIR_ANALYSIS):
    """
    Decode shpae with across conditios of location or size, then compute scalar summary score
    by averageing over time window.
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
        fig = sns.catplot(data=dfagg, x="bregion", y="score_adjusted", hue="decodekind", col="var_decode_and_conj", row="event", kind="bar")
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

    # First, preprocess all pa
    list_pa =[]
    for PA in DFallpa["pa"]:
        pa = preprocess_pa(PA, animal, date, var_other, "/tmp", True, None, None, None, None, False)
        plt.close("all")
        list_pa.append(pa)    
    DFallpa["pa"] = list_pa

    ### Collect each PA (doing dim reduction)
    # (Project onto shape subspace)
    tbin_dur = "default"
    tbin_slide = None

    if int(date)>220720:
        twind = (0.05, 1.0)
    else:
        twind = (0.05, 0.6)

    raw_subtract_mean_each_timepoint = False

    if var_other == "seqc_0_loc":
        list_subspace_projection = ["pca", "shape", "shape_loc"]
    elif var_other == "gridsize":
        list_subspace_projection = ["pca", "shape", "shape_size"]
    else:
        assert False

    for subspace_projection in list_subspace_projection:

        savedir = f"{SAVEDIR}/ssproj={subspace_projection}"
        os.makedirs(savedir, exist_ok=True)

        dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

        # (1) First, dim reduction
        superv_dpca_var = superv_dpca_params['superv_dpca_var']
        superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

        list_pa = []
        for i, row in DFallpa.iterrows():
            PA = row["pa"]
    
            _, PAredu = PA.dataextract_dimred_wrapper("scal", dim_red_method, "/tmp", twind, tbin_dur=tbin_dur, tbin_slide = tbin_slide, NPCS_KEEP=10, 
                                        dpca_var=superv_dpca_var, dpca_vars_group=superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict,
                                        raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)    
            list_pa.append(PAredu)
            plt.close("all")
        DFallpa["pa_redu"] = list_pa    

        ### PLOT
        from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
        from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots

        var_col = var_other
        var_effect = "seqc_0_shape"

        # Extract event to plot
        for event in DFallpa["event"].unique().tolist():
            # event = "03_samp"
            dfallpa = DFallpa[DFallpa["event"]==event].reset_index(drop=True)

            nrows = len(dfallpa)

            dflab = PA.Xlabels["trials"]
            levels_col = dflab[var_col].unique().tolist()
            ncols = len(levels_col)
            SIZE =5

            for dims in [(0,1), (1,2), (2,3), (3,4)]:
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
                # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))
                for i, row in dfallpa.iterrows():

                    bregion = row["bregion"]
                    event = row["event"]
                    PAredu = row["pa_redu"]     

                    for j, lev_col in enumerate(levels_col):
                        try:
                            ax = axes[i][j]
                        except Exception as err:
                            print(axes)
                            print(i, j)
                            raise err

                        ax.set_title((bregion, lev_col))

                        pa = PAredu.slice_by_labels_filtdict({var_col:[lev_col]})

                        if dims[1]<=pa.X.shape[0]-1:
                            xs = pa.X[dims[0], :, 0]
                            ys = pa.X[dims[1], :, 0]
                            # zs = pa.X[2, :, 0]
                            dflab = pa.Xlabels["trials"]
                            labels = dflab[var_effect].tolist()

                            # _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax, plot_3D=False, zs = zs)
                            _trajgood_plot_colorby_scalar_BASE_GOOD(xs, ys, labels, ax)
                if False:
                    share_axes_row_or_col_of_subplots(axes, "row", "both")   
                
                savefig(fig, f"{savedir}/scatter-event={event}-dims={dims}.pdf")
                plt.close("all")


def heatmaps_plot_wrapper(DFallpa, animal, date, SAVEDIR_ANALYSIS, var_other="seqc_0_loc"):
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

    if var_other == "seqc_0_loc":
        list_subspace_projection = ["pca_proj", "pca", "shape", "shape_loc", None]
    elif var_other == "gridsize":
        list_subspace_projection = ["pca_proj", "pca", "shape", "shape_size", None]
    else:
        assert False

    # drawings_done = []
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
                # mean_over_trials = True
            else:
                # plot only cleaned up data.
                list_unstable_badstrokes = [(True, True, True)]
                list_mean_over_trials = [False, True]
                # mean_over_trials = False
                
            # for remove_drift in [False]:
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
    PLOTS_DO = [4, 5]
    # PLOTS_DO = [4.3]
    # PLOTS_DO = [0]

    # if (animal, date) in 
    # var_other = afasfsdaf
    
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
            savedir = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-combine={combine}-var_other={var_other}"
            os.makedirs(savedir, exist_ok=True)
            DO_RSA_HEATMAPS = True
            DO_SHUFFLE = False
            euclidian_time_resolved_fast_shuffled(DFallpa, animal, date, var_other, savedir, 
                                                  DO_SHUFFLE=DO_SHUFFLE, DO_RSA_HEATMAPS=DO_RSA_HEATMAPS)

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
        else:
            print(PLOTS_DO)
            assert False