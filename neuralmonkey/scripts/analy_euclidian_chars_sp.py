"""
Specifically to ask about reuse of prim representation in chars.
This was previously done in analy_euclidian_dist_pop... but deicded that was too generic -- here there are specific
things needed for chars.
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
from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good


# N_MIN_TRIALS_PER_SHAPE = 5 # defualt, but for chars, useful to be lower to 4
N_MIN_TRIALS_PER_SHAPE = 4
TWIND_ANALY = (-0.4, 0.5)
NPCS_KEEP = 6

def preprocess_clean_stable_single_prims_frate(PA, twind=None, savedir=None):
    """
    Clean this PA so that it keeps only chans with similar frate across all contigs (bloques) that are SP.
    i.e., if fr too diff, then this is evience of drift.
    
    If only one SP bloque, then does nothing.

    RETURNS:
    - copy of PA, but pruned chans that fail this.
    """
    import pandas as pd
    from pythonlib.tools.pandastools import group_continuous_blocks_of_rows
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from pythonlib.tools.statstools import compute_d_prime
    from scipy.stats import ranksums
    import seaborn as sns
    # Split into blocks based on task_kind, and check if is different due to time of day --> i.e., same
    # task kind, diff time of day.

    ### PARAMS
    task_kind_keep = "prims_single"
    map_n_bloques_to_dthresh = {
        2:1.5,
        3:1.7,
        4:1.9
    }
    map_n_bloques_to_pthresh = {
        2:-4.5,
        3:-4.75,
        4:-5
    }
    # dthresh = 1.5
    # pthresh = -4.5

    ### RUN
    if twind is None:
        twind = (-0.2, 0.3) # Based on strokes, for SP/Char

    pa = PA.slice_by_dim_values_wrapper("times", twind)
    pa = pa.agg_wrapper("times")
    dflab = PA.Xlabels["trials"]

    # First, get trialcodes into sortable scalrs
    # trialcode_scalars = [trialcode_to_scalar(tc) for tc in dflab["trialcode"]]
    # dflab["trialcode_scal"] = trialcode_scalars
    df_sorted, _, _ = group_continuous_blocks_of_rows(dflab, "task_kind", "trialcode_scal", 
                                                            new_col="task_kind_grp", PRINT=True, do_sort=False,
                                                            savedir=savedir)
    dflab["task_kind_grp"] = df_sorted["task_kind_grp"]

    # (1) For each task_kind contig, get its psth and std.
    grpdict = grouping_append_and_return_inner_items_good(dflab, ["task_kind", "task_kind_grp"])
    grpdict = {grp:inds for grp, inds in grpdict.items() if grp[0]==task_kind_keep}

    if len(grpdict)<2:
        print("Not enough task_kind bloques --- returning copy of input PA")
        return PA

    # Based on how many groups exist, decide on threshold, with higher (conservative) trheshold for more comparisons.
    n = len(grpdict)

    # if n not in the map, then take the hihghest
    n_max = max(list(map_n_bloques_to_dthresh.keys()))
    print("n_max:", n_max)
    if n > n_max:
        n = n_max

    dthresh = map_n_bloques_to_dthresh[n]
    pthresh = map_n_bloques_to_pthresh[n]

    # Append firing rate onto dflab
    res = []
    for indchan in range(len(pa.Chans)):
        dflab["frates"] = pa.X[indchan, :] 

        for i, (grp1, inds1) in enumerate(grpdict.items()):
            for j, (grp2, inds2) in enumerate(grpdict.items()):
                if j>i:
                    print("---------")
                    print(grp1, inds1)
                    print(grp2, inds2)

                    dflab.iloc[inds1]

                    fr1 = dflab.iloc[inds1]["frates"]
                    fr2 = dflab.iloc[inds2]["frates"]

                    # Compute stats
                    d = compute_d_prime(fr1, fr2)
                    p = ranksums(fr1, fr2).pvalue

                    res.append({
                        "indchan":indchan,
                        "grp1":grp1,
                        "grp2":grp2,
                        "d_prime":d,
                        "p_val":p
                    })

    # (2) Detect cases where pairwise diff is very high, including for same task_kind across 
    dfres = pd.DataFrame(res)

    dfres["chan"] = [PA.Chans[i] for i in dfres["indchan"]]
    dftmp = dfres.groupby(["indchan"])["d_prime"].max()
    dfres = pd.merge(dfres, dftmp, on="indchan")

    dftmp = dfres.groupby(["indchan"])["p_val"].min()
    dfres = pd.merge(dfres, dftmp, on="indchan")

    dfres["p_val_log"] = np.log10(dfres["p_val_y"]) # min pval

    ## Pull out neurons 
    chans_bad = dfres[(dfres["d_prime_y"] > dthresh) | (dfres["p_val_log"] < pthresh)]["chan"].unique().tolist()
    chans_good = [c for c in PA.Chans if c not in chans_bad]


    if savedir is not None:

        fig = sns.relplot(data=dfres, x="d_prime_y", y="p_val_log", hue="chan")
        savefig(fig, f"{savedir}/scatter.pdf")
        
        fig = sns.catplot(data=dfres, x="chan", y="d_prime_y", kind="bar", aspect=2)
        for ax in fig.axes.flatten():
            ax.axhline(dthresh)
        savefig(fig, f"{savedir}/bar-d_prime.pdf")
        
        fig = sns.catplot(data=dfres, x="chan", y="p_val_log", kind="bar", aspect=2)
        for ax in fig.axes.flatten():
            ax.axhline(pthresh)
        fig.axes.flatten()[0].set_title(f"chans_bad={chans_bad}")
        savefig(fig, f"{savedir}/bar-p_val.pdf")


    PA = PA.slice_by_dim_values_wrapper("chans", chans_good)
    
    return PA


def params_shapes_remove(animal, date, shape_var):
    """ 
    Hard coded cases with clear differences between SP and CHAR that are not 
    gotten using the auto method in preprocess_pa, beucase that depends on reverse being better match, but that
    method fails if both fwd and reverse arent great matches
    """ 

    assert shape_var=="shape_semantic_grp", "below assuming this."

    shapes_remove = []
    if animal=="Pancho":
        shapes_remove.append("line-UU-UU") # Sometimes direction flipped, and not caught by eucldian diffs
    
    return shapes_remove

def preprocess_pa(animal, date, PA, savedir, prune_version, shape_var = "shape_semantic_grp", n_min_trials_per_shape=N_MIN_TRIALS_PER_SHAPE,
                  plot_counts_heatmap_savepath=None, plot_drawings=True, remove_chans_fr_drift=False,
                  subspace_projection=None, twind_analy=None, tbin_dur=None, tbin_slide=None, NPCS_KEEP=None, 
                  raw_subtract_mean_each_timepoint=False, remove_singleprims_unstable=False):
    """
    Does not modofiy PA, returns copy
    """ 

    if subspace_projection is not None:
        assert twind_analy is not None
        assert tbin_dur is not None
        assert tbin_slide is not None
        assert NPCS_KEEP is not None

    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    PA = PA.copy()

    ### (0) Plot original tabulation of shape vs task_klind
    dflab = PA.Xlabels["trials"]
    fig = grouping_plot_n_samples_conjunction_heatmap(dflab, shape_var, "stroke_index", ["task_kind"])
    path = f"{savedir}/shape_counts-orig.pdf"
    fig.savefig(path)

    #### Append any new columns
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dflab = PA.Xlabels["trials"]
    # dflab = append_col_with_grp_index(dflab, ["task_kind", "shape_semantic_grp", "stroke_index_is_first"], "task_shape_sifirst")
    dflab = append_col_with_grp_index(dflab, ["task_kind", "shape_semantic_grp", "stroke_index"], "task_shape_si")
    PA.Xlabels["trials"] = dflab

    ########################################
    # (1) Keep just SP and (Char 0)
    if prune_version is None:
        # Then don't prune any trials
        pass
    else:
        if prune_version == "sp_char_0":
            task_kinds = ["prims_single", "character"]
            fd = {"task_kind":task_kinds, "stroke_index":[0]}
        elif prune_version == "sp_char":
            task_kinds = ["prims_single", "character"]
            fd = {"task_kind":task_kinds}
        elif prune_version == "sp_pig":
            task_kinds = ["prims_single", "prims_on_grid"]
            fd = {"task_kind":task_kinds}            
        elif prune_version == "pig_char":
            task_kinds = ["prims_on_grid", "character"]
            fd = {"task_kind":task_kinds}
        elif prune_version == "pig_char_0":
            task_kinds = ["prims_on_grid", "character"]
            fd = {"task_kind":task_kinds, "stroke_index":[0]}
        elif prune_version == "pig_char_1plus":
            task_kinds = ["prims_on_grid", "character"]
            fd = {"task_kind":task_kinds, "stroke_index":list(range(1, 10))}
        else:
            assert False
        PA = PA.slice_by_labels_filtdict(fd)

        # (2) Keep just shapes taht exist across both SP and CHAR.
        dflab = PA.Xlabels["trials"]
        _dfout,_  = extract_with_levels_of_conjunction_vars_helper(dflab, "task_kind", [shape_var], n_min_per_lev=n_min_trials_per_shape,
                                                    plot_counts_heatmap_savepath=plot_counts_heatmap_savepath, 
                                                    levels_var=task_kinds)
        if len(_dfout)==0 or _dfout is None:
            print("Pruned all data!! returning None")
            print("... using these params: ", shape_var, task_kinds)
            return None
            # assert False, "not enough data"
    
        index_datapt_list_keep = _dfout["index_datapt"].tolist()
        PA = PA.slice_by_labels_filtdict({"index_datapt":index_datapt_list_keep})

    ########################
    # Hacky - remove cases where the strokes are too different between SP and CHAR
    # Get strokes for each row
    PA.behavior_extract_strokes_to_dflab()

    if plot_drawings:
        #### Make plots of strokes
        from pythonlib.drawmodel.strokePlots import plotDatStrokesWrapper, plot_single_strok
        import numpy as np
        from pythonlib.tools.plottools import savefig
        dflab = PA.Xlabels["trials"]

        grpdict = grouping_append_and_return_inner_items_good(dflab, [shape_var, "task_kind"])

        # Make many iterations, picking random subsets
        n_iter = 1
        for i in range(n_iter):
            ncols = 6
            nrows = int(np.ceil(len(grpdict)/ncols))
            SIZE = 4
            n_rand = 6
            fig, axes = plt.subplots(nrows, ncols, figsize=(SIZE*ncols, SIZE*nrows), sharex=True, sharey=True)
            for ax, (grp, inds) in zip(axes.flatten(), grpdict.items()):
                strokes = dflab.iloc[inds]["strok_beh"].tolist()

                # Plot just subset of strokes
                plotDatStrokesWrapper(strokes, ax, add_stroke_number=False, alpha=0.4, color="random", n_rand = n_rand)
                ax.set_title(grp)
                # assert False

            savefig(fig, f"{savedir}/stroke_drawings-before_remove_unaligned_strokes-iter_{i}.pdf")

    # Detect cases where CHAR and SP have different stroke onsets for the same identified stroke
    dflab = PA.Xlabels["trials"]
    from pythonlib.dataset.dataset_strokes import DatStrokes
    ds = DatStrokes()
    grpdict = grouping_append_and_return_inner_items_good(dflab, [shape_var])
    shapes_keep = []
    for (sh,), inds in grpdict.items():
        strokes = dflab.iloc[inds]["strok_beh"]

        succ =  ds._strokes_check_all_aligned_direction(strokes, plot_failure=True, thresh=0.7)
        if succ:
            shapes_keep.append(sh)
        else:
            print("remove this shape, not consistent across trials: ", sh)

    # Also remove some added by hand
    shapes_remove = params_shapes_remove(animal, date, shape_var)
    print("Also removing tese shapes. by hand: ", shapes_remove)
    shapes_keep = [sh for sh in shapes_keep if sh not in shapes_remove]
    
    print("Keeping these shapes, becuase they are not similar strokes between SP and CHAR:", shapes_keep)
    PA = PA.slice_by_labels_filtdict({shape_var:shapes_keep})

    if plot_drawings:
        ### Plot strokes again, after removing bad stroke
        dflab = PA.Xlabels["trials"]
        grpdict = grouping_append_and_return_inner_items_good(dflab, [shape_var, "task_kind"])

        # Make many iterations, picking random subsets
        n_iter = 3
        for i in range(n_iter):
            ncols = 6
            nrows = int(np.ceil(len(grpdict)/ncols))
            SIZE = 4
            n_rand = 6
            fig, axes = plt.subplots(nrows, ncols, figsize=(SIZE*ncols, SIZE*nrows), sharex=True, sharey=True)
            for ax, (grp, inds) in zip(axes.flatten(), grpdict.items()):
                strokes = dflab.iloc[inds]["strok_beh"].tolist()

                # Plot just subset of strokes
                plotDatStrokesWrapper(strokes, ax, add_stroke_number=False, alpha=0.4, color="random", n_rand = n_rand)
                ax.set_title(grp)

            savefig(fig, f"{savedir}/stroke_drawings-after_remove_unaligned_strokes-iter_{i}.pdf")

    # Plot counts one final time
    dflab = PA.Xlabels["trials"]
    # fig = grouping_plot_n_samples_conjunction_heatmap(dflab, shape_var, "task_kind")
    fig = grouping_plot_n_samples_conjunction_heatmap(dflab, shape_var, "stroke_index", ["task_kind"])
    path = f"{savedir}/shape_counts-final.pdf"
    fig.savefig(path)

    ############################
    # Optioanlly, remove channels with drift
    if remove_chans_fr_drift:
        from neuralmonkey.classes.population_mult import dfallpa_preprocess_sitesdirty_single_just_drift
        PA = dfallpa_preprocess_sitesdirty_single_just_drift(PA, animal, date, savedir=savedir)

    if remove_singleprims_unstable:
        from neuralmonkey.scripts.analy_euclidian_chars_sp import preprocess_clean_stable_single_prims_frate
        PA = preprocess_clean_stable_single_prims_frate(PA, savedir=savedir)    
        
    ############ PROJECTION
    if subspace_projection is not None:
        dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

        # (1) First, dim reduction
        superv_dpca_var = superv_dpca_params['superv_dpca_var']
        superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

        _, PA = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
                                        twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
                                        NPCS_KEEP = NPCS_KEEP,
                                        dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict, 
                                        dpca_proj_twind = twind_analy, 
                                        raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                        umap_n_components=None, umap_n_neighbors=None)
    else:
        if tbin_dur is not None:
            PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        if twind_analy is not None:
            PA = PA.slice_by_dim_values_wrapper("times", twind_analy)
        if raw_subtract_mean_each_timepoint:
            PA = PA.norm_subtract_trial_mean_each_timepoint()

    return PA

def params_subspace_projection(subspace_projection):
    if subspace_projection == "pca":
        dim_red_method = "pca"
        superv_dpca_params={
            "superv_dpca_var":None,
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":None,
        }
    elif subspace_projection == "shape_prims_single":
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"shape_semantic_grp",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":{"task_kind":["prims_single"]}
        }
    elif subspace_projection == "shape_all":
        # shapes, regardless of index or task kind.
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"shape_semantic_grp",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":{}
        }
    elif subspace_projection == "shape_PIG_stroke0":
        # PIG (0)  
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"shape_semantic_grp",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":{"task_kind":["prims_on_grid"], "stroke_index":[0]}
        }
    elif subspace_projection == "shape_char_stroke0":
        # Char (0)
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"shape_semantic_grp",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":{"task_kind":["character"], "stroke_index":[0]}
        }
    elif subspace_projection == "task_shape_si":
        # Char (0)
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"task_shape_si",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":{}
        }
    else:
        print(subspace_projection)
        assert False
        
    return dim_red_method, superv_dpca_params

def euclidian_time_resolved_wrapper(animal, date, DFallpa, SAVEDIR_ANALYSIS):

    n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
    raw_subtract_mean_each_timepoint = False

    twind_analy = TWIND_ANALY
    tbin_dur = 0.1
    tbin_slide = 0.02

    for bregion in DFallpa["bregion"].unique().tolist():
        for prune_version in ["sp_char_0", "pig_char_0", "pig_char_1plus", "sp_char"]:
            if prune_version in ["sp_char_0"]:
                subspace_projection_extra = "shape_prims_single"
            elif prune_version in ["pig_char_0", "pig_char", "pig_char_1plus", "sp_char"]:
                subspace_projection_extra = "shape_all"
            else:
                print(prune_version)
                assert False
                
            # for subspace_projection in [None, "pca", subspace_projection_extra]:
            for subspace_projection in [None, "pca", "task_shape_si"]: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
                # for remove_drift in [False, True]:
                for remove_drift in [False]:
                    # for raw_subtract_mean_each_timepoint in [False, True]:
                    for raw_subtract_mean_each_timepoint in [False]:
                        for remove_singleprims_unstable in [False, True]:
                            SAVEDIR = f"{SAVEDIR_ANALYSIS}/{bregion}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-subtrmean={raw_subtract_mean_each_timepoint}"
                            os.makedirs(SAVEDIR, exist_ok=True)
                            print("SAVING AT ... ", SAVEDIR)
                            euclidian_time_resolved(animal, date, DFallpa, bregion, prune_version, remove_drift, SAVEDIR, twind_analy,
                                                        tbin_dur, tbin_slide, 
                                                        subspace_projection, NPCS_KEEP, 
                                                        n_min_trials_per_shape = n_min_trials_per_shape, raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                                        remove_singleprims_unstable=remove_singleprims_unstable)


def euclidian_time_resolved(animal, date, DFallpa, bregion, prune_version, remove_drift, SAVEDIR, twind_analy,
                            tbin_dur, tbin_slide, 
                            subspace_projection, NPCS_KEEP, 
                            n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE, raw_subtract_mean_each_timepoint=False,
                            hack_prune_to_these_chans = None,
                            remove_singleprims_unstable=False):
    """
    Eucldian distance [effect of shape vs. task(context)] as function of time, relative to stroke onset.
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.scripts.analy_euclidian_chars_sp import preprocess_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import append_col_with_grp_index
    
    # Run
    PA = extract_single_pa(DFallpa, bregion, which_level="stroke", event="00_stroke")

    savedir = f"{SAVEDIR}/preprocess"
    os.makedirs(savedir, exist_ok=True)
    plot_drawings = False
    PA = preprocess_pa(animal, date, PA, savedir, prune_version, 
                        n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings,
                        remove_chans_fr_drift=remove_drift,
                        subspace_projection=subspace_projection, 
                            twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                            raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                            remove_singleprims_unstable=remove_singleprims_unstable)

    if PA is None:
        return


    if hack_prune_to_these_chans is not None:
        # Optionally, keep specific chans
        # chans_keep = [1053, 1054]
        # chans_keep = [1044, 1049,  1053, 1054, 1057, 1059, 1062]
        assert isinstance(hack_prune_to_these_chans, list)
        PA = PA.slice_by_dim_values_wrapper("chans", hack_prune_to_these_chans)

    ### Quick analyses of euclidian distances
    vars_group = ["task_kind", "shape_semantic_grp"]
    version = "traj"
    DFDIST = PA.dataextractwrap_distance_between_groups(vars_group, version)

    # DFDIST = append_col_with_grp_index(DFDIST, ["shape_semantic_grp_1", "shape_semantic_grp_2"], "shape_semantic_grp_same")
    # DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_1", "task_kind_2"], "task_kind_same")
    DFDIST["task_kind_same"] = DFDIST["task_kind_1"] == DFDIST["task_kind_2"]
    DFDIST["shape_semantic_grp_same"] = DFDIST["shape_semantic_grp_1"] == DFDIST["shape_semantic_grp_2"]
    DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_1", "task_kind_2"], "task_kind_12")
    DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_same", "shape_semantic_grp_same"], "same-task|shape")
    DFDIST = append_col_with_grp_index(DFDIST, ["shape_semantic_grp_same", "task_kind_12"], "same_shape|task_kind_12")
    DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_same", "shape_semantic_grp_same"], "same-task|shape")
    DFDIST = append_col_with_grp_index(DFDIST, ["shape_semantic_grp_1", "shape_semantic_grp_2"], "shape_semantic_grp_12")

    pd.to_pickle(DFDIST, f"{SAVEDIR}/DFDIST.pkl")
    
    for y in ["dist_mean", "dist_norm", "dist_yue_diff"]:
        # sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
        fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same-task|shape", kind="line", errorbar=("ci", 68))
        savefig(fig, f"{SAVEDIR}/relplot-{y}-1.pdf")

        fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
        savefig(fig, f"{SAVEDIR}/relplot-{y}-2.pdf")

        fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="task_kind_12", kind="line", col="same-task|shape", errorbar=("ci", 68))
        savefig(fig, f"{SAVEDIR}/relplot-{y}-3.pdf")

        if False: # slow, and I don't use
            fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="shape_semantic_grp_12", kind="line", col="same-task|shape", 
                        errorbar=("ci", 68), legend=False, alpha=0.5)
            savefig(fig, f"{SAVEDIR}/relplot-{y}-4.pdf")

        plt.close("all")


def run(animal, date,  DFallpa, SAVEDIR, subspace_projection, prune_version, NPCS_KEEP, 
        raw_subtract_mean_each_timepoint, n_min_trials_per_shape,
        PLOT_EACH_REGION, PLOT_STATE_SPACE,
        LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS,
        twind_analy, devo_return_data=False,
        tbin_dur = 0.1, tbin_slide = 0.05):
    """
    Entire pipelined, across areas, for a specific set of params.

    """
    # Some params   
    # twind_analy = (-0.1, 0.4)
    # tbin_dur = 0.1
    # tbin_slice = 0.05

    # twind_analy = (0.05, 0.25)
    # twind_analy = (0.05, 0.25)
    tbin_dur = 0.1
    tbin_slide = 0.05

    # Run
    list_res = []
    list_res_clean = []
    for i_br, bregion in enumerate(DFallpa["bregion"].unique().tolist()):
        
        PA = extract_single_pa(DFallpa, bregion, which_level="stroke", event="00_stroke")

        savedir = f"{SAVEDIR}/preprocess"
        os.makedirs(savedir, exist_ok=True)
        plot_drawings = i_br ==0
        PA = preprocess_pa(animal, date, PA, savedir, prune_version, 
                           n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings,
                           subspace_projection=subspace_projection, 
                           twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                           raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)

        assert False, "PAredu is returned from preprocess_pa"

        # ### New, cleaner method, taking all pairwise distances between trials
        # savedir = f"{SAVEDIR}/each_region/{bregion}"
        # os.makedirs(savedir, exist_ok=True)

        # if subspace_projection is not None:
        #     dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

        #     # (1) First, dim reduction
        #     superv_dpca_var = superv_dpca_params['superv_dpca_var']
        #     superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        #     superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

        #     _, PAredu = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
        #                                     twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
        #                                     NPCS_KEEP = NPCS_KEEP,
        #                                     dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict, 
        #                                     dpca_proj_twind = twind_analy, 
        #                                     raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
        #                                     umap_n_components=None, umap_n_neighbors=None)
        # else:
        #     PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        #     PA = PA.slice_by_dim_values_wrapper("times", twind_analy)
        #     PAredu = PA

        # (2) State space
        if PLOT_STATE_SPACE:
            # otherwise is redundant.
            list_dims = [(0,1), (2,3), (4,5)]
            PAredu.plot_state_space_good_wrapper(savedir, LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS, 
                                                 list_dims=list_dims)

        # # (3) Euclidian distances
        Cldist = PAredu.dataextract_as_distance_matrix_clusters_flex(["task_kind", "shape_semantic_grp"], 
                                                                    return_as_single_mean_over_time=True)
        dfres = Cldist.rsa_distmat_score_all_pairs_of_label_groups(False)

        # Add some things
        dfres["task_kind_same"] = dfres["task_kind_1"] == dfres["task_kind_2"]
        dfres["shape_semantic_grp_same"] = dfres["shape_semantic_grp_1"] == dfres["shape_semantic_grp_2"]
        dfres = append_col_with_grp_index(dfres, ["shape_semantic_grp_same", "task_kind_same"], "shape_task_same")
        dfres = append_col_with_grp_index(dfres, ["task_kind_1", "task_kind_2"], "task_kind_pair")
        dfres = append_col_with_grp_index(dfres, ["shape_semantic_grp_1", "shape_semantic_grp_2"], "shape_semantic_grp_pair")

        # For diff shape, same task_kind, do only single prim tasks
        if subspace_projection in [None, "pca"]:
            dfres_clean = dfres
        elif subspace_projection == "shape_prims_single":
            a = dfres["task_kind_1"] == "prims_single" 
            b = dfres["task_kind_2"] == "prims_single"
            c = dfres["shape_task_same"] =="0|1"
            df1 = dfres[a & b & c]

            # For same shape, diff task kind, include both sp and char
            c = dfres["shape_task_same"] =="1|0"
            df2 = dfres[c]

            dfres_clean = pd.concat([df1, df2]).reset_index(drop=True)
        elif subspace_projection == "shape_PIG_stroke0":
            assert False, "code it"
        elif subspace_projection == "shape_char_stroke0":
            assert False, "code it"
        else:
            print(subspace_projection)
            assert False

        if PLOT_EACH_REGION:
            # sns.catplot(data=dfres, x="dist_yue_diff", y="shape_semantic_grp_pair", hue="task_kind_1", col="shape_task_same")
            from pythonlib.tools.pandastools import plot_subplots_heatmap
            fig, _ = plot_subplots_heatmap(dfres, "shape_semantic_grp_1", "shape_semantic_grp_2", "dist_yue_diff", "task_kind_pair", share_zlim=True)
            savefig(fig, f"{savedir}/heatmap_distances.pdf")
            
            fig = sns.catplot(data=dfres_clean, x="shape_task_same", y="dist_yue_diff", kind="bar")
            savefig(fig, f"{savedir}/catplot-clean.pdf")

            fig = sns.catplot(data=dfres, x="shape_task_same", y="dist_yue_diff", kind="bar")
            savefig(fig, f"{savedir}/catplot-1.pdf")

            fig = sns.catplot(data=dfres, x="shape_task_same", y="dist_yue_diff", jitter=True, alpha=0.5)
            savefig(fig, f"{savedir}/catplot-2.pdf")

        if False:
            # Return single metric
            dfres.groupby("shape_task_same")["dist_yue_diff"].mean().reset_index()
            dfres_clean.groupby("shape_task_same")["dist_yue_diff"].mean().reset_index()

        dfres["bregion"] = bregion
        dfres_clean["bregion"] = bregion

        list_res.append(dfres)
        list_res_clean.append(dfres_clean)

        if devo_return_data:
            return PA

    ############## SAVE
    plt.close("all")
    savedir = f"{SAVEDIR}/summary"
    os.makedirs(savedir, exist_ok=True)

    DFRES = pd.concat(list_res).reset_index(drop=True)
    DFRES_CLEAN = pd.concat(list_res_clean).reset_index(drop=True)

    # Save results
    pd.to_pickle(DFRES, f"{savedir}/DFRES.pkl")
    pd.to_pickle(DFRES_CLEAN, f"{savedir}/DFRES_CLEAN.pkl")

    # Summary plots
    fig = sns.catplot(data=DFRES, x = "bregion", hue="shape_task_same", y="dist_yue_diff", kind="bar", aspect=2)
    savefig(fig, f"{savedir}/catplot-1.pdf")

    fig = sns.catplot(data=DFRES_CLEAN, x = "bregion", hue="shape_task_same", y="dist_yue_diff", kind="bar", aspect=2)
    savefig(fig, f"{savedir}/catplot-clean-1.pdf")

    # Also plot scatter
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    _, fig = plot_45scatter_means_flexible_grouping(DFRES, "shape_task_same", "1|0", "0|1", None, "dist_yue_diff", "bregion", True, SIZE=4);
    savefig(fig, f"{savedir}/scatter-1.pdf")

    _, fig = plot_45scatter_means_flexible_grouping(DFRES_CLEAN, "shape_task_same", "1|0", "0|1", None, "dist_yue_diff", "bregion", True, SIZE=4);
    savefig(fig, f"{savedir}/scatter-clean-1.pdf")

    plt.close("all")

def plot_heatmap_firing_rates_all_wrapper(DFallpa, SAVEDIR_ANALYSIS, animal, date):
    """
    Wrapper, to plot all FR heatmaps (trial vs time) with diff variations.
    Gaol is to visualize theraw data as clsoely and broadly as possible, see effect,
    and also see drift if it exists.
    """

    # prune_version = "sp_char_0"
    # prune_version = None
    n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
    raw_subtract_mean_each_timepoint = False

    # LIST_SS_PRUNE = [
    #     (None, False),
    #     ("pca", False),
    #     ("shape_prims_single", False),
    #     ("pca", True),
    #     ("shape_prims_single", True),
    # ]
    LIST_SS_PRUNE = [
        (None, False),
    ]


    drawings_done = []
    # for prune_version in ["sp_char_0", "pig_char_0"]:
    for prune_version in ["sp_char_0"]:

        # Plot drwaings just once
        if prune_version not in drawings_done:
            PA = DFallpa["pa"].values[0]
            savedir = f"{SAVEDIR_ANALYSIS}/drawings-prune={prune_version}"
            os.makedirs(savedir, exist_ok=True)
            plot_drawings = True
            PA = preprocess_pa(animal, date, PA, savedir, prune_version, 
                            n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings)
            drawings_done.append(prune_version)

        for bregion in DFallpa["bregion"].unique().tolist():
            SAVEDIR = f"{SAVEDIR_ANALYSIS}/{bregion}-prune={prune_version}"

            ################# PREPROCESS
            PA = extract_single_pa(DFallpa, bregion, which_level="stroke", event="00_stroke")
            savedir = f"{SAVEDIR}/preprocess"
            os.makedirs(savedir, exist_ok=True)
            plot_drawings = False
            PA = preprocess_pa(animal, date, PA, savedir, prune_version, 
                            n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings)

            if PA is None:
                continue

            for subspace_projection, prune_chans in LIST_SS_PRUNE:

                savedir = f"{SAVEDIR}/ss={subspace_projection}-prunedrift={prune_chans}"
                os.makedirs(savedir, exist_ok=True)

                ########### PRUNE CHANS
                if prune_chans:
                    # Optioanlly, remove channels with drift
                    from neuralmonkey.classes.population_mult import dfallpa_preprocess_sitesdirty_single_just_drift
                    PAthis = dfallpa_preprocess_sitesdirty_single_just_drift(PA, animal, date, savedir=savedir)
                else:
                    PAthis = PA.copy()

                if PAthis is None:
                    continue

                ########### DIM REDUCTIONS
                tbin_dur = 0.1
                tbin_slice = 0.02
                twind_analy = TWIND_ANALY
                if subspace_projection is not None:
                    dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)
                    superv_dpca_var = superv_dpca_params['superv_dpca_var']
                    superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
                    superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

                    # savedirthis = f"{savedir}/dimred-{subspace_projection}"
                    # os.makedirs(savedirthis, exist_ok=True)
                    _, PAredu = PAthis.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
                                                    twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slice, 
                                                    NPCS_KEEP = NPCS_KEEP,
                                                    dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict, 
                                                    dpca_proj_twind = twind_analy, 
                                                    raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                                    umap_n_components=None, umap_n_neighbors=None)
                else:
                    # quicker, smaller 
                    PAthis = PAthis.agg_by_time_windows_binned(tbin_dur, tbin_slice)
                    PAthis = PAthis.slice_by_dim_values_wrapper("times", twind_analy)
                    PAredu = PAthis

                if PAredu is None:
                    continue

                for subtr_time_mean in [False, True]:
                    if subtr_time_mean:
                        pa = PAredu.norm_subtract_trial_mean_each_timepoint()
                    else:
                        pa = PAredu
                    savedirthis = f"{savedir}/subtr_time_mean={subtr_time_mean}"
                    os.makedirs(savedirthis, exist_ok=True)

                    plot_heatmap_firing_rates_all(pa, savedirthis)

                    plt.close("all")


def plot_heatmap_firing_rates_all(PA, savedir):
    """
    To look at "raw" data, plotting heatmaps (trial vs time), a few different kinds

    """
    from pythonlib.tools.snstools import heatmap_mat
    from  neuralmonkey.neuralplots.population import _heatmap_stratified_y_axis
    from neuralmonkey.neuralplots.population import heatmap_stratified_trials_grouped_by_neuron, heatmap_stratified_neuron_grouped_by_var
    import numpy as np
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    print("--- Saving at: ", savedir)
    ######## PREPROCESS
    # # quicker, smaller 
    # dur = 0.1
    # slide = 0.02
    # PA = PA.agg_by_time_windows_binned(dur, slide)

    # Get a global zlim
    zlims = np.percentile(PA.X.flatten(), [1, 99]).tolist()

    #########################################################
    ############################# (1) 
    dflab = PA.Xlabels["trials"]
    grpdict = grouping_append_and_return_inner_items_good(dflab, ["shape_semantic_grp", "task_kind"])

    n_rand_trials = 10

    SIZE = 3
    ncols = 6
    nrows = int(np.ceil(len(grpdict)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))

    for ax, (grp, inds) in zip(axes.flatten(), grpdict.items()):

        heatmap_stratified_trials_grouped_by_neuron(PA, inds, ax, n_rand_trials=n_rand_trials, zlims=None)

        ax.set_title(grp, fontsize=8)
        # assert False
    savefig(fig, f"{savedir}/grouped_by_neuron.png")
    plt.close("all")

    #########################################################
    ### (2)  Same thing, but each plot is a single (PC, task_kind), split by shape
    shapes = sorted(PA.Xlabels["trials"]["shape_semantic_grp"].unique().tolist())
    task_kinds = sorted(PA.Xlabels["trials"]["task_kind"].unique().tolist())

    list_ind_neur = list(range(len(PA.Chans)))
    ncols = len(list_ind_neur)
    nrows = len(task_kinds)
    SIZE = 3.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), squeeze=False)
    for j, tk in enumerate(task_kinds): # row
        pa_this = PA.slice_by_labels_filtdict({"task_kind":[tk]})

        for i, ind_neur in enumerate(list_ind_neur): # columns

            ax = axes[j][i]
            heatmap_stratified_neuron_grouped_by_var(pa_this, ind_neur, ax, n_rand_trials=n_rand_trials, zlims=zlims, 
                                            y_group_var="shape_semantic_grp", y_group_var_levels=shapes)

            ax.set_title(f"neur={i}-{PA.Chans[i]}|tk={tk}")
    savefig(fig, f"{savedir}/grouped_by_var.png")

    # (3)  Loop over all bregions
    from neuralmonkey.neuralplots.population import heatmapwrapper_stratified_each_neuron_alltrials
    fig = heatmapwrapper_stratified_each_neuron_alltrials(PA, "task_kind")
    savefig(fig, f"{savedir}/each_neuron_over_time.png")
    plt.close("all")


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
    from pythonlib.tools.pandastools import append_col_with_grp_index
    import seaborn as sns
    from pythonlib.tools.plottools import savefig
    import os
    from neuralmonkey.scripts.analy_euclidian_chars_sp import preprocess_pa, run
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single, euclidian_distance_compute_trajectories

    animal = sys.argv[1]
    date = int(sys.argv[2])
    combine = int(sys.argv[3])==1

    question = "CHAR_BASE_stroke"

    # PLOTS_DO = [1, 2]
    PLOTS_DO = [2]
    
    # if animal=="Diego":
    #     combine = True
    # elif animal=="Pancho":
    #     combine = False
    # else:
    #     assert False
    
    # Load a single DFallPA
    DFallpa = load_handsaved_wrapper(animal, date, version="stroke", combine_areas=combine, question=question, return_none_if_no_exist=True)
    if DFallpa is None:
        # Then extract
        DFallpa = extract_dfallpa_helper(animal, date, question, combine, do_save=True)

    # Make a copy of all PA before normalization
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

    from neuralmonkey.classes.population_mult import dfallpa_preprocess_sort_by_trialcode
    dfallpa_preprocess_sort_by_trialcode(DFallpa)

    SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/{animal}-{date}-combine={combine}"

    ################ PARAMS
    if 0 in PLOTS_DO:
        """
        Euclidina distance (scalar), effect of shape vs. task. This is like the oriigal plots, but cleaner.
        Not important, as the plots below (#2) get time-varying, which you can just take mean over to get identical
        to this.
        Useful part: State space plots
        """
        ######### PARAMS
        n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
        LIST_NPCS_KEEP = [4,6,2]
        PLOT_EACH_REGION = True

        ### State space plots
        LIST_VAR = [
            "shape_semantic_grp",
            "shape_semantic_grp",
            "shape_semantic_grp",
        ]
        LIST_VARS_OTHERS = [
            ["task_kind", "stroke_index"],
            ["task_kind", "stroke_index"],
            ["task_kind", "stroke_index"],
        ]
        LIST_CONTEXT = [
            {"same":["stroke_index"], "diff":["task_kind"]},
            {"same":["stroke_index"], "diff":["task_kind"]},
            {"same":["stroke_index"], "diff":["task_kind"]},
        ]
        LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
        LIST_FILTDICT = [
            {"task_kind":["prims_single", "character"], "stroke_index":[0]},
            {"task_kind":["prims_single", "character"]},
            {"task_kind":["prims_single", "prims_on_grid"]},
            ]

        for twind_analy in [(0.05, 0.25), (-0.05, 0.35), (0.1, 0.2)]:
            for subspace_projection in ["shape_prims_single", "pca"]:
                for prune_version in ["sp_char_0", "sp_char"]:
                    for NPCS_KEEP in LIST_NPCS_KEEP:
                        for raw_subtract_mean_each_timepoint in [False, True]:
                            SAVEDIR = f"{SAVEDIR_ANALYSIS}/subspc={subspace_projection}-prunedat={prune_version}-npcs={NPCS_KEEP}-subtr={raw_subtract_mean_each_timepoint}-twind={twind_analy}"
                            os.makedirs(SAVEDIR, exist_ok=True)

                            PLOT_STATE_SPACE = NPCS_KEEP == max(LIST_NPCS_KEEP)
                            run(animal, date, DFallpa, SAVEDIR, subspace_projection, prune_version, NPCS_KEEP, 
                                    raw_subtract_mean_each_timepoint, n_min_trials_per_shape,
                                    PLOT_EACH_REGION, PLOT_STATE_SPACE,
                                    LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS, twind_analy)
                            
    if 1 in PLOTS_DO:
        """
        Heatmaps of raw activity, different variations. 
        Useful, to get intuition of what is going on in euclidian (#2)
        Also plots drawings comparing SP vs. CHAR
        """
        from neuralmonkey.scripts.analy_euclidian_chars_sp import plot_heatmap_firing_rates_all
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/HEATMAPS/{animal}-{date}-combine={combine}"
        os.makedirs(SAVEDIR, exist_ok=True)
        print(SAVEDIR)
        plot_heatmap_firing_rates_all_wrapper(DFallpa, SAVEDIR, animal, date)

    if 2 in PLOTS_DO:
        """
        Time-resolved euclidian distances
        
        Agg across multiple bregions and dates:
        # MULT DATA - euclidian_time_resolved" in 241002_char_euclidian...
        """
        SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/{animal}-{date}-combine={combine}"
        os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
        print(SAVEDIR_ANALYSIS)
        euclidian_time_resolved_wrapper(animal, date, DFallpa, SAVEDIR_ANALYSIS)