"""
Specifically to ask about reuse of prim representation in chars.
This was previously done in analy_euclidian_dist_pop... but deicded that was too generic -- here there are specific
things needed for chars.

NOTEBOOK: 241002_char_euclidian_pop
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
# NPCS_KEEP = 6
NPCS_KEEP = 8

SUBSPACE_PROJ_FIT_TWIND = {
    "00_stroke":[(-0.5, -0.05), (0.05, 0.5), (-0.5, 0.5)],
    # "04_go_cue":[(-0.3, 0.3)],
    # "05_first_raise":[(-0.3, 0.3)],
    # "06_on_strokeidx_0":[(-0.5, -0.05), (0.05, 0.5)],
}
LIST_SUBSPACE_PROJECTION = [None, "pca_proj", "task_shape_si"]

LIST_PRUNE_VERSION = ["sp_char_0"] # Just to do wquicly.
# LIST_PRUNE_VERSION = ["sp_char_0", "pig_char_0", "pig_char_1plus", "sp_char"] # GOOD




def load_euclidian_time_resolved_fast_shuffled(animal, date, bregion, morphset, inds_in_morphset_keep, do_prune_ambig,
                                               var_context_diff, event, twind_final, scalar_or_traj, dim_red_method, NPCS_KEEP):
    
    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    from pythonlib.tools.pandastools import aggregGeneral, extract_with_levels_of_conjunction_vars


    def _prune_ambig_at_least_n_trials_per_base(DF, df_version, n_min_per_lev = 3):
        """
        Input DF must be trial level (before agging)
        Returns copy
        """
        
        if df_version=="DFDIST":
            vars_datapt = ["idx_morph_temp", "idxmorph_assigned_2", "time_bin_idx"] # DFDIST
        elif df_version=="DFPROJ_INDEX":
            vars_datapt = ["idx_morph_temp", "time_bin_idx"] # DFPROJ_INDEX
        else:
            print(df_version)
            assert False

        # (1) Split into ambig vs. (unambig + learned)
        DF["assigned_label"].value_counts()
        dfthis = DF[DF["assigned_label"] == "ambig"].reset_index(drop=True)

        # (2) Only keep (idx_morph_temp) that has at least n trials for both base1 and base2
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
        df_ambig, _ = extract_with_levels_of_conjunction_vars_helper(dfthis, "assigned_base_simple", 
                                                                    vars_datapt, 
                                                                    n_min_per_lev, None, 2, levels_var=["base1", "base2"], 
                                                                    remove_extra_columns=True)
        if len(df_ambig)==0:
            df_ambig, _ = extract_with_levels_of_conjunction_vars_helper(dfthis, "assigned_base_simple", 
                                                                        vars_datapt, 
                                                                        n_min_per_lev-1, None, 2, levels_var=["base1", "base2"],
                                                                        remove_extra_columns=True)

            assert len(df_ambig)>0, "you removed all ambig cases. reduce n until you keep all morphsets"

        # (3) Concat with the remaining
        df_others = DF[DF["assigned_label"] != "ambig"].reset_index(drop=True)
        df_combined = pd.concat([df_ambig, df_others]).reset_index(drop=True)

        if False:
            # To see grouping counts that shows what the above is doing:
            dfthis = DF[DF["time_bin_idx"]==0].reset_index(drop=True)
            grouping_plot_n_samples_conjunction_heatmap(dfthis, "idx_morph_temp", "assigned_base_simple", ["idxmorph_assigned_2"]);
            # grouping_plot_n_samples_conjunction_heatmap(dfthis, "idxmorph_assigned_1", "idxmorph_assigned_2", ["seqc_0_loc_1"]);

        return df_combined

    ### Default params
    # event = "03_samp"

    exclude_flank = True
    # dim_red_method = "dpca"
    # proj_twind = (0.1, 1.0)
    # combine = True
    # raw_subtract_mean_each_timepoint = False
    # scalar_or_traj = "traj"
    # NPCS_KEEP = 8
    # twind_final = (-0.3, 1.2)

    ### LOAD
    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PSYCHO_SP/{animal}-{date}-logistic-combine=True/analy_switching_GOOD_euclidian_index/ev={event}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={twind_final}-npcs={NPCS_KEEP}/bregion={bregion}/morphset={morphset}"
    # SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PSYCHO_SP/{animal}-{date}-logistic-combine=True/switching_euclidian_score_and_plot_traj/ev={event}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred=dpca-twind=(-0.1, 1.2)-npcs={NPCS_KEEP}/bregion={bregion}/morphset={morphset}"
    # SAVEDIR = f"{SAVEDIR_BASE_LOAD}/{animal}-{date}-logistic-combine={combine}/euclidian_score_and_plot/ev={EVENT}-subtr={raw_subtract_mean_each_timepoint}-scal={scalar_or_traj}-dimred={dim_red_method}-twind={proj_twind}-npcs={NPCS_KEEP}-flank={exclude_flank}"
    print("Loading from: ", SAVEDIR)

    # Load data
    DFDIST = pd.read_pickle(f"{SAVEDIR}/DFDIST.pkl")
    DFPROJ_INDEX = pd.read_pickle(f"{SAVEDIR}/DFPROJ_INDEX.pkl")

    # Newer code uses location as a context diff var.
    if var_context_diff is None:
        # so downstream code works.
        DFPROJ_INDEX["seqc_0_loc"] = "ignore"
    elif var_context_diff=="seqc_0_loc":
        DFPROJ_INDEX["seqc_0_loc"] = [lab[1] for lab in DFPROJ_INDEX["labels_1_datapt"]]
    else:
        print(var_context_diff)
        assert False

    if do_prune_ambig:
        DFPROJ_INDEX = _prune_ambig_at_least_n_trials_per_base(DFPROJ_INDEX, "DFPROJ_INDEX")
        DFDIST = _prune_ambig_at_least_n_trials_per_base(DFDIST, "DFDIST")

    # Agg across trials
    DFPROJ_INDEX_AGG = aggregGeneral(DFPROJ_INDEX, ["idxmorph_assigned", "time_bin_idx", "seqc_0_loc"], ["dist_index", "dist_index_norm", "time_bin"], nonnumercols=["assigned_base_simple", "assigned_base", "assigned_label", "idx_morph_temp"])
    DFDIST_AGG = aggregGeneral(DFDIST, ["idxmorph_assigned_1", "idxmorph_assigned_2", "time_bin_idx"], ["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff", "time_bin"], nonnumercols=["assigned_base_simple", "assigned_base", "assigned_label", "idx_morph_temp"])

    try:
        # all inds that should be ambig --> make sure they are called that.
        # prolbem is that some cases I did not call ambig if there were only a few cases of base1/base2.  But Ishould not 
        # do this.
        # if False: # skip this check, since sometimes auto detection doesnt call it ambig (too few trials, or is too noisy)
        assert all(DFDIST[DFDIST["idx_morph_temp"].isin(inds_in_morphset_keep)]["assigned_label"] == "ambig"), "why this was not called ambig? Prob it only had a couple trials... Fix the original code that called ambig vs. not-ambig."

        # all other inds --> make sure not called ambig.
        assert not any(DFDIST[~DFDIST["idx_morph_temp"].isin(inds_in_morphset_keep)]["assigned_label"]=="ambig"), "in this case, definitely go with my hand label"
    except AssertionError as err:
        from pythonlib.tools.pandastools import grouping_print_n_samples
        print("--------------")
        print(animal, date, morphset)
        print("Indices I manually said to keep: ", inds_in_morphset_keep)
        print("Indices automatilcaly labeled as ambig: ", grouping_print_n_samples(DFDIST, ["idx_morph_temp", "assigned_label"]))
        # print(err)
        raise err
    
    if not all([x in DFPROJ_INDEX_AGG["assigned_base"].unique().tolist() for x in ['base1', 'ambig_base1', 'ambig_base2', 'base2']]):
        # if not sorted(DFPROJ_INDEX_AGG["assigned_label"].unique()) == ['ambig', 'base', 'not_ambig']:
        print("Skipping", animal, date, morphset, " since doesnt have all 3 trial labels. Just has: ", sorted(DFPROJ_INDEX_AGG["assigned_label"].unique()))     
        return None

    for df in [DFDIST, DFDIST_AGG, DFPROJ_INDEX, DFPROJ_INDEX_AGG]:
        df["animal"] = animal
        df["date"] = date
        df["bregion"] = bregion
        df["morphset"] = morphset

        # Important, numerical precision...
        df["time_bin"] = df["time_bin"].apply(lambda x:np.round(x, 3))
        # df.sort_values("time_bin").reset_index(drop=True)

    return DFDIST, DFDIST_AGG, DFPROJ_INDEX, DFPROJ_INDEX_AGG

def behstrokes_map_clustshape_to_thresh(animal):
    """
    clust sim max thresholds, for each shape, hand entered.
    Based on visualizing results of behstrokes_extract_char_clust_sim
    """

    if animal == "Pancho":
        map_clustshape_to_thresh = {
            "Lcentered-4-2-0":0.6,
            "Lcentered-4-3-0":0.5,
            "Lcentered-4-4-0":0.58,
            "V-2-1-0":0.55,
            "V-2-2-0":0.58,
            "V-2-4-0":0.58,
            "arcdeep-4-1-0":0.57,
            "arcdeep-4-2-0":0.57,
            "arcdeep-4-4-0":0.57,
            "circle-6-1-0":0.50,
            "line-8-1-0":0.54,
            "line-8-2-0":0.6,
            "line-8-3-0":0.65,
            "line-8-4-0":0.65,
            "squiggle3-3-1-0":0.55,
            "squiggle3-3-2-0":0.55,
            "squiggle3-3-2-1":0.59,    
        }
    elif animal=="Diego":

        map_clustshape_to_thresh = {
            "Lcentered-4-1-0":0.6,
            "Lcentered-4-2-0":0.6,
            "Lcentered-4-3-0":0.6,
            "Lcentered-4-4-0":0.6,
            # "V-2-1-0":0,
            "V-2-2-0":0.6,
            "V-2-3-0":0.6,
            "V-2-4-0":0.6,
            # "arcdeep-4-1-0":0,
            "arcdeep-4-2-0":0.6,
            "arcdeep-4-3-0":0.58,
            "arcdeep-4-4-0":0.58,
            "circle-6-1-0":0.52,
            "line-8-1-0":0.68,
            "line-8-2-0":0.68,
            "line-8-3-0":0.68,
            "line-8-4-0":0.68,
            "squiggle3-3-1-0":0.6,
            "squiggle3-3-1-1":0.6,
            "squiggle3-3-2-0":0.57,
            "squiggle3-3-2-1":0.57,    
            "usquare-1-2-0":0.59,    
            "usquare-1-3-0":0.59,    
            "usquare-1-4-0":0.59,    
            "zigzagSq-1-1-0":0.56,    
            "zigzagSq-1-1-1":0.56,    
            "zigzagSq-1-2-0":0.56,    
            "zigzagSq-1-2-1":0.56,    
        }
    else:
        print(animal)
        assert False
    
    return map_clustshape_to_thresh


def behstrokes_extract_char_clust_sim(PA, animal, date, savedir, PLOT=False):
    """
    Load character cluster data for all rows in PA, and plot examples drwaings, sorted by score,
    copy values into PA, and return the loaded scores.
    Main goal -- to apply new threhsolds on clust_sim_max to throw out bad beh strokes.

    Along the way makes bunch of plots.

    RETURNS:
    - Modifies PA to have a new column clust_sim_max_GOOD, whihc is bool for whether this row has
    good beh
    - ds, DatStrokes object holding clust data fro each row of PA.
    """
    from pythonlib.dataset.dataset_strokes import DatStrokes
    from pythonlib.tools.pandastools import append_col_with_grp_index, slice_by_row_label

    PA.behavior_extract_strokes_to_dflab(trial_take_first_stroke=True)

    dflab = PA.Xlabels["trials"]
    
    ### Load saved cluster values
    ds = DatStrokes()
    df, _ = ds.clustergood_load_saved_cluster_shape_classes_input_basis_set(WHICH_BASIS=animal, ANIMAL=animal, DATE=date, merge_with_self=False)

    ### Slice df to match the rows in dflab
    dflab = append_col_with_grp_index(dflab, ["trialcode", "stroke_index"], "tc_stkidx")
    df = append_col_with_grp_index(df, ["trialcode", "stroke_index"], "tc_stkidx")
    idxs = dflab["tc_stkidx"].tolist()
    _df = slice_by_row_label(df, "tc_stkidx", idxs, assert_exactly_one_each=True)

    # Sanity checks that loaded data matches dflab
    from pythonlib.tools.nptools import isnear
    assert np.all(dflab["character"] == _df["character"])
    assert np.all(dflab["gridloc"] == _df["gridloc"])
    assert np.all(dflab["stroke_index_fromlast"] == _df["stroke_index_fromlast"])
    s1 = np.stack([x[0] for x in dflab["strok_beh"]])
    s2 = np.stack([x[0] for x in _df["strok"]])
    assert isnear(s1, s2)

    ### Extract (merge) cluster values into dflab
    for col in ["clust_sim_max", "clust_sim_max_colname"]:
        dflab[col] = _df[col]
    if PLOT:
        fig, ax = plt.subplots()
        dflab["clust_sim_max"].hist(bins=80, ax=ax)
        savefig(fig, f"{savedir}/hist-clust_sim_max.pdf")

    ### Copy values to ds
    ds.Dat = _df
    ds.Dat["shape_semantic_grp"] = dflab["shape_semantic_grp"]

    ### Plot, show each shape, sorted by clust_sim_max
    # ds.plot_multiple_sorted_by_feature([0,1,2], "clust_sim_max", overlay_beh_or_task=None)
    ds.Dat = append_col_with_grp_index(ds.Dat, ["clust_sim_max_colname", "task_kind"], "clustname|tk")
    ds.Dat = append_col_with_grp_index(ds.Dat, ["shape_semantic_grp", "task_kind"], "shapesemgrp|tk")
    if PLOT:
        # ds.plotshape_multshapes_trials_grid_sort_by_feature(col_grp="clust_sim_max_colname", sort_rows_by_this_feature="clust_sim_max", nrows=10, recenter_strokes=True)
        for col_grp in ["clustname|tk", "shapesemgrp|tk"]:
            for i in range(1):  
                fig = ds.plotshape_multshapes_trials_grid_sort_by_feature(col_grp=col_grp, sort_rows_by_this_feature="clust_sim_max", nrows=10, recenter_strokes=True)
                savefig(fig, f"{savedir}/cols={col_grp}-rows=clust_sim_max-iter={i}.png")
                plt.close("all")

    
    ### Prune, based on new threshold
    map_clustshape_to_thresh = behstrokes_map_clustshape_to_thresh(animal)
    def good(x):
        sh = x["clust_sim_max_colname"]
        return x["clust_sim_max"] > map_clustshape_to_thresh[sh]
    dflab["clust_sim_max_GOOD"] = [good(row) for i, row in dflab.iterrows()]

    # Plot scores and whether is good.
    if PLOT:
        import seaborn as sns
        from pythonlib.tools.snstools import rotateLabel
        fig = sns.catplot(data = dflab, x="clust_sim_max_colname", y="clust_sim_max", hue="clust_sim_max_GOOD", row="task_kind", alpha=0.5, jitter=True, aspect=1.5)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/prune-catplot.pdf")


    # Make drawings, splitting into good and bad (diff figures)
    ds.Dat["clust_sim_max_GOOD"] = dflab["clust_sim_max_GOOD"]
    if PLOT:
        dsGood = DatStrokes()
        dsGood.Dat = ds.Dat[ds.Dat["clust_sim_max_GOOD"]==True].reset_index(drop=True)
        dsBad = DatStrokes()
        dsBad.Dat = ds.Dat[ds.Dat["clust_sim_max_GOOD"]==False].reset_index(drop=True)
        for col_grp in ["clustname|tk", "shapesemgrp|tk"]:
            for i in range(1):  
                fig = dsGood.plotshape_multshapes_trials_grid_sort_by_feature(col_grp=col_grp, sort_rows_by_this_feature="clust_sim_max", nrows=10, recenter_strokes=True)
                savefig(fig, f"{savedir}/prune_good-cols={col_grp}-rows=clust_sim_max-iter={i}.png")
                plt.close("all")

                fig = dsBad.plotshape_multshapes_trials_grid_sort_by_feature(col_grp=col_grp, sort_rows_by_this_feature="clust_sim_max", nrows=10, recenter_strokes=True)
                savefig(fig, f"{savedir}/prune_bad-cols={col_grp}-rows=clust_sim_max-iter={i}.png")
                plt.close("all")

    ### Final
    PA.Xlabels["trials"] = dflab

    return ds


# def behstrokes_preprocess_prune_trials_bad_strokes(DFallpa, animal, date):
#     """
#     Replaces pa with new pa that have pruned trials, rmeoving ones with bad strokes.
#     RETURNS:
#     - (modifies DFallpa)
#     """
#     from pythonlib.tools.pandastools import slice_by_row_label

#     # Only works for storkes.
#     assert DFallpa["event"].unique().tolist() == ["00_stroke"], "with trials, there is ambiguity bceuae of mult strokes per trials..."

#     ### First, use one PA to extract beh and label each trial as good bad based on beh strokes.
#     PA = DFallpa["pa"].values[0]
#     ds = behstrokes_extract_char_clust_sim(PA, animal, date, savedir=None, PLOT=False)
    
#     ### Second, throw out rows, for each PA, that are bad.
#     # PA.Xlabels["trials"]["clust_sim_max_GOOD"].value_counts()
#     DFLAB = PA.Xlabels["trials"]
#     list_pa = []
#     for i, row in DFallpa.iterrows():
#         pa = row["pa"]
#         dflab = pa.Xlabels["trials"]
#         dflab = append_col_with_grp_index(dflab, ["trialcode", "stroke_index"], "tc_stkidx")

#         _df = slice_by_row_label(DFLAB, "tc_stkidx", dflab["tc_stkidx"].tolist(), assert_exactly_one_each=True)

#         dflab["clust_sim_max_GOOD"] = _df["clust_sim_max_GOOD"] # Assign back into dflab
#         inds = dflab[dflab["clust_sim_max_GOOD"]==True].index.tolist() # keep only the "GOOD" cases
#         pathis = pa.slice_by_dim_indices_wrapper("trials", inds, reset_trial_indices=True)
        
#         print(pa.X.shape, " --> ", pathis.X.shape)    

#         list_pa.append(pathis)
#     DFallpa["pa"] = list_pa

def behstrokes_preprocess_assign_col_bad_strokes(DFallpa, animal, date):
    """
    For each pa, assigns a new column "clust_sim_max_GOOD", which is bool, for whtehr this is a 
    good or bad stroke by beh. 
    Downstream code can easily prune bad stroke using this.
    RETURNS:
    - (modifies DFallpa)
    """
    from pythonlib.tools.pandastools import slice_by_row_label

    # Only works for storkes.
    assert DFallpa["event"].unique().tolist() == ["00_stroke"], "with trials, there is ambiguity bceuae of mult strokes per trials..."

    ### First, use one PA to extract beh and label each trial as good bad based on beh strokes.
    PA = DFallpa["pa"].values[0]
    ds = behstrokes_extract_char_clust_sim(PA, animal, date, savedir=None, PLOT=False)
    
    ### Second, throw out rows, for each PA, that are bad.
    # PA.Xlabels["trials"]["clust_sim_max_GOOD"].value_counts()
    DFLAB = PA.Xlabels["trials"]
    list_pa = []
    for i, row in DFallpa.iterrows():
        pa = row["pa"]
        dflab = pa.Xlabels["trials"]
        dflab = append_col_with_grp_index(dflab, ["trialcode", "stroke_index"], "tc_stkidx")

        # get the new column
        _df = slice_by_row_label(DFLAB, "tc_stkidx", dflab["tc_stkidx"].tolist(), assert_exactly_one_each=True)

        # Append and put back into pa
        dflab["clust_sim_max_GOOD"] = _df["clust_sim_max_GOOD"] # Assign back into dflab
        pa.Xlabels["trials"] = dflab
        
        # inds = dflab[dflab["clust_sim_max_GOOD"]==True].index.tolist() # keep only the "GOOD" cases
        # pathis = pa.slice_by_dim_indices_wrapper("trials", inds, reset_trial_indices=True)
        
        # print(pa.X.shape, " --> ", pathis.X.shape)    
        # list_pa.append(pathis)
    # DFallpa["pa"] = list_pa

def beh_plot_event_timing_stroke(PA, animal, date, savedir, shape_var = "shape_semantic_grp", 
                                 MS=None):
    """
    NOTE: has to load MS, which takes a while...
    """

    if MS is None:
        from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
        MS = load_mult_session_helper(date, animal)   

    events=["samp", "go", "first_raise", "on_strokeidx_0", "off_strokeidx_0"]
    normalize_to_this_event="on_strokeidx_0"
    for xlims in [None]:
        fig = PA.behavior_extract_events_timing_plot_distributions(MS, shape_var=shape_var, 
                                                            events=events, normalize_to_this_event=normalize_to_this_event,
                                                            xlims=xlims)
        savefig(fig, f"{savedir}/event_timing-each={shape_var}-1-xlims={xlims}.pdf")
        
    events=["go", "first_raise", "on_strokeidx_0", "off_strokeidx_0"]
    normalize_to_this_event="on_strokeidx_0"
    for xlims in [None, [PA.Times[0], PA.Times[-1]]]:
        fig = PA.behavior_extract_events_timing_plot_distributions(MS, shape_var=shape_var, 
                                                            events=events, normalize_to_this_event=normalize_to_this_event,
                                                            xlims=xlims)
        savefig(fig, f"{savedir}/event_timing-each={shape_var}-2-xlims={xlims}.pdf")
    
        


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
    elif len(grpdict)>20:
        # Then this is not bloques, but is trial structure. 
        print("No need to remove any neruons, nbeucase they are interleaved")
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

    if len(chans_good)>0:
        PA = PA.slice_by_dim_values_wrapper("chans", chans_good)
        return PA
    else:
        return None
    
def params_shapes_remove(animal, date, shape_var):
    """ 
    Hard coded cases with clear differences between SP and CHAR that are not 
    gotten using the auto method in preprocess_pa, beucase that depends on reverse being better match, but that
    method fails if both fwd and reverse arent great matches
    """ 

    assert shape_var in ["seqc_0_shapesemgrp", "shape_semantic_grp"], "below assuming this."

    shapes_remove = []
    if animal=="Pancho":
        shapes_remove.append("line-UU-UU") # Sometimes direction flipped, and not caught by eucldian diffs
    
    return shapes_remove

def preprocess_pa(animal, date, PA, savedir, prune_version, shape_var = "shape_semantic_grp", 
                  n_min_trials_per_shape=N_MIN_TRIALS_PER_SHAPE,
                  plot_counts_heatmap_savepath=None, plot_drawings=True, remove_chans_fr_drift=False,
                  subspace_projection=None, twind_analy=None, tbin_dur=None, tbin_slide=None, NPCS_KEEP=None, 
                  raw_subtract_mean_each_timepoint=False, remove_singleprims_unstable=False,
                  remove_trials_with_bad_strokes=True, subspace_projection_fitting_twind=None,
                  skip_dim_reduction=False):
    """
    Does not modofiy PA, returns copy
    """ 

    if subspace_projection not in [None, "pca"]:
        assert twind_analy is not None
        assert tbin_dur is not None
        assert tbin_slide is not None
        assert NPCS_KEEP is not None
        assert subspace_projection_fitting_twind is not None, "you need to restrict pca to window that matters."

    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    PA = PA.copy()

    ### (0) Plot original tabulation of shape vs task_klind
    dflab = PA.Xlabels["trials"]
    if savedir is not None:
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

        TASK_KINDS = task_kinds

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
            if savedir is not None:
                savefig(fig, f"{savedir}/stroke_drawings-before_remove_unaligned_strokes-iter_{i}.pdf")

    ### Detect cases where CHAR and SP have different stroke onsets for the same identified stroke
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

    ### Also remove some added by hand
    shapes_remove = params_shapes_remove(animal, date, shape_var)
    print("Also removing tese shapes. by hand: ", shapes_remove)
    shapes_keep = [sh for sh in shapes_keep if sh not in shapes_remove]
    
    print("Keeping these shapes, becuase they are not similar strokes between SP and CHAR:", shapes_keep)
    PA = PA.slice_by_labels_filtdict({shape_var:shapes_keep})

    ### Remove bad strokes baesd on new hand-entere clust sim thresholds.
    if remove_trials_with_bad_strokes:
        print("Removing bad strokes, using hand-entered clust_sim_max thresholds...")
        dflab = PA.Xlabels["trials"]
        # behstrokes_preprocess_prune_trials_bad_strokes(DFallpa, animal, date)
        inds = dflab[dflab["clust_sim_max_GOOD"]==True].index.tolist() # keep only the "GOOD" cases
        print(PA.X.shape[1], " --> ", len(inds), " (num trials) ")
        PA = PA.slice_by_dim_indices_wrapper("trials", inds, reset_trial_indices=True)

    # -------------------------------------------------------------
    ############ FINALLY, prune to minimum n trials.
    # (1) Keep just SP and (Char 0)
    if prune_version is None:
        # Then don't prune any trials
        pass
    else:
        # (2) Keep just shapes taht exist across both SP and CHAR.
        dflab = PA.Xlabels["trials"]
        _dfout,_  = extract_with_levels_of_conjunction_vars_helper(dflab, "task_kind", [shape_var], n_min_per_lev=n_min_trials_per_shape,
                                                    plot_counts_heatmap_savepath=plot_counts_heatmap_savepath, 
                                                    levels_var=TASK_KINDS)
        if len(_dfout)==0 or _dfout is None:
            print("Pruned all data!! returning None")
            print("... using these params: ", shape_var, task_kinds)
            return None
    
        index_datapt_list_keep = _dfout["index_datapt"].tolist()
        PA = PA.slice_by_labels_filtdict({"index_datapt":index_datapt_list_keep})
    # -------------------------------------------------------------

    ############# PRINT AND PLOT FINAL TRIAL COUNTS
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

    ### Plot counts one final time
    dflab = PA.Xlabels["trials"]
    if savedir is not None:
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
        if PA is None:
            return None
        
    ############ PROJECTION
    if not skip_dim_reduction:
        from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _preprocess_pa_dim_reduction
        _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                 twind_analy, tbin_dur, tbin_slide, savedir)
        
        # if subspace_projection is not None:
        #     dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

        #     # (1) First, dim reduction
        #     superv_dpca_var = superv_dpca_params['superv_dpca_var']
        #     superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        #     superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

        #     _, PA = PA.dataextract_dimred_wrapper("traj", dim_red_method, savedir, 
        #                                     subspace_projection_fitting_twind, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
        #                                     # twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, 
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


def preprocess_pa_trials(animal, date, PA, savedir, prune_version, shape_var = "seqc_0_shapesemgrp", 
                         n_min_trials_per_shape=N_MIN_TRIALS_PER_SHAPE,
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
    fig = grouping_plot_n_samples_conjunction_heatmap(dflab, shape_var, "task_kind")
    path = f"{savedir}/shape_counts-orig.pdf"
    fig.savefig(path)

    #### Append any new columns
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dflab = PA.Xlabels["trials"]
    # dflab = append_col_with_grp_index(dflab, ["task_kind", "shape_semantic_grp", "stroke_index_is_first"], "task_shape_sifirst")
    dflab = append_col_with_grp_index(dflab, ["task_kind", "seqc_0_shapesemgrp"], "task_shape")
    PA.Xlabels["trials"] = dflab

    ########################################
    # (1) Keep just SP and (Char 0)
    if prune_version is None:
        # Then don't prune any trials
        pass
    else:
        if prune_version == "sp_char":
            task_kinds = ["prims_single", "character"]
            fd = {"task_kind":task_kinds}
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
    PA.behavior_extract_strokes_to_dflab(trial_take_first_stroke=True)

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
    fig = grouping_plot_n_samples_conjunction_heatmap(dflab, shape_var, "task_kind")
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
    if subspace_projection in ["pca", "pca_proj"]:
        dim_red_method = subspace_projection
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
    elif subspace_projection == "task_shape":
        # COnjunction of task and shape
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"task_shape",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":{}
        }
    elif subspace_projection == "shape_loc":
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"seqc_0_shapeloc",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":None,
        }
    elif subspace_projection == "shape_size":
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"seqc_0_shapesize",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":None,
        }
    elif subspace_projection == "shape":
        dim_red_method = "superv_dpca"
        superv_dpca_params={
            "superv_dpca_var":"seqc_0_shape",
            "superv_dpca_vars_group":None,
            "superv_dpca_filtdict":None,
        }
    else:
        print(subspace_projection)
        assert False
        
    return dim_red_method, superv_dpca_params


def euclidian_time_resolved_wrapper(animal, date, DFallpa, SAVEDIR_ANALYSIS):
    """
    To get timecourse of euclidina distance.
    Is final code, except:
    - Is not doing held-out train-test for fitting dPCA.
    
    Mainly use this to visualize timecourse, with final scores and stats done using euclidian_time_resolved_fast_shuffled()
    instead.
    """
    n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
    raw_subtract_mean_each_timepoint = False

    twind_analy = TWIND_ANALY
    # tbin_dur = 0.1
    # tbin_slide = 0.02
    tbin_dur = 0.2
    tbin_slide = 0.02

    for i, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for prune_version in LIST_PRUNE_VERSION:
            if prune_version in ["sp_char_0"]:
                subspace_projection_extra = "shape_prims_single"
            elif prune_version in ["pig_char_0", "pig_char", "pig_char_1plus", "sp_char"]:
                subspace_projection_extra = "shape_all"
            else:
                print(prune_version)
                assert False
                
            # for subspace_projection in [None, "pca", subspace_projection_extra]:
            # for subspace_projection in [None, "pca", "task_shape_si"]: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
            for subspace_projection in LIST_SUBSPACE_PROJECTION:
                if subspace_projection in [None, "pca"]:
                    list_fit_twind = [twind_analy]
                else:
                    list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]
                for subspace_projection_fitting_twind in list_fit_twind:
                    for raw_subtract_mean_each_timepoint in [False]:
                        for remove_drift, remove_singleprims_unstable in [(True, True)]:
                        # for remove_drift in [False]:
                        #     # for raw_subtract_mean_each_timepoint in [False, True]:
                        #     for remove_singleprims_unstable in [False, True]:
                            SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-subtrmean={raw_subtract_mean_each_timepoint}-fit_twind={subspace_projection_fitting_twind}"
                            os.makedirs(SAVEDIR, exist_ok=True)
                            print("SAVING AT ... ", SAVEDIR)
                            euclidian_time_resolved(animal, date, PA, which_level, prune_version, remove_drift, SAVEDIR, twind_analy,
                                                        tbin_dur, tbin_slide, 
                                                        subspace_projection, NPCS_KEEP, 
                                                        n_min_trials_per_shape = n_min_trials_per_shape, 
                                                        raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                                        remove_singleprims_unstable=remove_singleprims_unstable,
                                                        subspace_projection_fitting_twind=subspace_projection_fitting_twind)

def euclidian_time_resolved_wrapper_trial(animal, date, DFallpa, SAVEDIR_ANALYSIS):

    assert False, "have not updated with fitted windows, see stroke version."
    n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
    raw_subtract_mean_each_timepoint = False

    twind_analy = (-1., 1.2)
    tbin_dur = 0.1
    tbin_slide = 0.02

    events_keep = ["03_samp", "05_first_raise"]

    HACK = False
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        if HACK and bregion not in ["PMv"]:
            continue

        if event in events_keep:
            for prune_version in ["sp_char"]:
                for subspace_projection in [None, "pca", "task_shape"]: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
                    for raw_subtract_mean_each_timepoint in [False]:
                        for remove_drift, remove_singleprims_unstable in [(False, False), (True, True)]:
                        # for remove_drift in [False]:
                        #     for remove_singleprims_unstable in [False, True]:
                            SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-subtrmean={raw_subtract_mean_each_timepoint}"
                            os.makedirs(SAVEDIR, exist_ok=True)
                            print("SAVING AT ... ", SAVEDIR)
                            euclidian_time_resolved(animal, date, PA, which_level, prune_version, remove_drift, SAVEDIR, twind_analy,
                                                        tbin_dur, tbin_slide, 
                                                        subspace_projection, NPCS_KEEP, 
                                                        n_min_trials_per_shape = n_min_trials_per_shape, raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                                        remove_singleprims_unstable=remove_singleprims_unstable)




def euclidian_time_resolved_fast_shuffled(DFallpa, animal, date, SAVEDIR_ANALYSIS,
                                          DO_RSA_HEATMAPS=False):
    """
    Good! Much faster method. And does one thing important for more rigorous result:
    1. Train-test split for dim redu (separate for fitting[smaller] and then final data projected)

    Derived from euclidian_time_resolved_fast_shuffled() in shape_invar_SP
    """
    # assert False, "save split, or else file gets to mult GB -- run seprately for each bregion."
    var_effect="shape_semantic_grp"
    var_conj = "task_kind"
    vars_group = [var_effect, var_conj]

    SUBSPACE_PROJ_FIT_TWIND = {
        "00_stroke":[(-0.5, -0.05), (0.05, 0.5), (-0.5, 0.5), (-0.4, 0.3)],
    }

    # LIST_SUBSPACE_PROJECTION = [None, "pca_proj", "task_shape_si", "shape_prims_single"]
    LIST_SUBSPACE_PROJECTION = ["task_shape_si", "shape_prims_single"]
    LIST_PRUNE_VERSION = ["sp_char_0", "pig_char_0", "pig_char_1plus", "sp_char"] # GOOD

    N_SPLITS = 6

    twind_analy = TWIND_ANALY
    tbin_dur = 0.2
    tbin_slide = 0.02

    map_event_to_listtwind_scal = {
        # "00_stroke":[(-0.5, -0.05), (0.05, 0.5), (-0.3, 0.1)],
        "00_stroke":[(-0.5, -0.05), (-0.5, 0), (-0.5, 0.1), (-0.3, 0.1), (-0.3, 0.2), (-0.4, 0.3), (-0.2, 0.3), (0.05, 0.5)],
        }

    list_dfdist =[]
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for prune_version in LIST_PRUNE_VERSION:

            for subspace_projection in LIST_SUBSPACE_PROJECTION:
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
                        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}--prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={subspace_projection_fitting_twind}"
                        os.makedirs(SAVEDIR, exist_ok=True)
                        print("SAVING AT ... ", SAVEDIR)

                        if DO_RSA_HEATMAPS:
                            # Plot pairwise distances (rsa heatmaps).
                            # This is done separatee to below becuase it doesnt use the train-test splits.
                            # It shold but I would have to code way to merge multple Cl, which is doable.
                            from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

                            # PAthis = preprocess_pa(PA, animal, date, var_other, None, remove_drift, 
                            #                        subspace_projection, subspace_projection_fitting_twind, 
                            #                        twind_analy, tbin_dur, tbin_slide, raw_subtract_mean_each_timepoint=False,
                            #                        skip_dim_reduction=False)
                            
                            PAthis = preprocess_pa(animal, date, PA, None, prune_version, 
                                                n_min_trials_per_shape=N_MIN_TRIALS_PER_SHAPE, shape_var=var_effect, plot_drawings=False,
                                                remove_chans_fr_drift=remove_drift, subspace_projection=subspace_projection, 
                                                    twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                                                    raw_subtract_mean_each_timepoint=False, remove_singleprims_unstable=remove_singleprims_unstable,
                                                    remove_trials_with_bad_strokes=remove_trials_with_bad_strokes, 
                                                    subspace_projection_fitting_twind=subspace_projection_fitting_twind)
                            
                            if PAthis is not None:
                                list_twind_scalar = map_event_to_listtwind_scal[event]
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
                        PAthis = preprocess_pa(animal, date, PA, savedir, prune_version, 
                                            n_min_trials_per_shape=N_MIN_TRIALS_PER_SHAPE, shape_var=var_effect, plot_drawings=False,
                                            remove_chans_fr_drift=remove_drift, subspace_projection=subspace_projection, 
                                                twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                                                raw_subtract_mean_each_timepoint=False, remove_singleprims_unstable=remove_singleprims_unstable,
                                                remove_trials_with_bad_strokes=remove_trials_with_bad_strokes, 
                                                subspace_projection_fitting_twind=subspace_projection_fitting_twind,
                                                skip_dim_reduction=skip_dim_reduction)

                        if PAthis is None:
                            continue

                        ########### DO TRAIN-TEST SPLITS
                        if False:
                            folds_dflab = PAthis.split_balanced_stratified_kfold_subsample_level_of_var(vars_group, None, None, 
                                                                                                        n_splits=N_SPLITS, 
                                                                                                        do_balancing_of_train_inds=False)
                        else:
                            # Better, more careful, ensuring enough data for euclidian distance.
                            fraction_constrained_set=0.7
                            n_constrained=3 # Ideally have more than 1 pair
                            list_labels_need_n=None
                            min_frac_datapts_unconstrained=None
                            min_n_datapts_unconstrained=len(PAthis.Xlabels["trials"][var_effect].unique())
                            plot_train_test_counts=True
                            plot_indices=False
                            folds_dflab, fig_unc, fig_con = PAthis.split_stratified_constrained_grp_var(N_SPLITS, vars_group, 
                                                                            fraction_constrained_set, n_constrained, 
                                                                            list_labels_need_n, min_frac_datapts_unconstrained,  
                                                                            min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
                            savefig(fig_con, f"{savedir}/after_split_constrained_fold_0.pdf")
                            savefig(fig_unc, f"{savedir}/after_split_unconstrained_fold_0.pdf")
                            plt.close("all")

                        for _i_dimredu, (train_inds, test_inds) in enumerate(folds_dflab):
                            # train_inds, more inds than than test_inds
                            train_inds = [int(i) for i in train_inds]
                            test_inds = [int(i) for i in test_inds]

                            ############# DO DIM REDUCTION
                            savedir = f"{SAVEDIR}/preprocess/i_dimredu={_i_dimredu}"
                            os.makedirs(savedir, exist_ok=True)
                            from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _preprocess_pa_dim_reduction
                            PAthisRedu = _preprocess_pa_dim_reduction(PAthis, subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide, savedir=savedir, raw_subtract_mean_each_timepoint=False,
                                    inds_pa_fit=test_inds, inds_pa_final=train_inds)

                            if PAthisRedu is None:
                                print("SKIPPING, since PAthisRedu is None: ", SAVEDIR)
                            else:
                                # Take different windows (for computing scalar score)
                                # Go thru diff averaging windows (to get scalar)
                                list_twind_scalar = map_event_to_listtwind_scal[event]
                                for twind_scal in list_twind_scalar:
                                    
                                    pa = PAthisRedu.slice_by_dim_values_wrapper("times", twind_scal)

                                    ###################################### Running euclidian
                                    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar
                                    
                                    # (1) Data
                                    dfdist, _ = timevarying_compute_fast_to_scalar(pa, label_vars=vars_group)
                                    
                                    dfdist["bregion"] = bregion
                                    dfdist["prune_version"] = prune_version
                                    dfdist["which_level"] = which_level
                                    dfdist["event"] = event
                                    dfdist["subspace_projection"] = subspace_projection
                                    dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]
                                    dfdist["dim_redu_fold"] = _i_dimredu
                                    dfdist["twind_scal"] = [twind_scal for _ in range(len(dfdist))]
                                    list_dfdist.append(dfdist)

    # Save, intermediate steps
    import pickle
    with open(f"{SAVEDIR_ANALYSIS}/list_dfdist.pkl", "wb") as f:
        pickle.dump(list_dfdist, f)

# def euclidian_time_resolved_strokes(animal, date, DFallpa, bregion, prune_version, remove_drift, SAVEDIR, twind_analy,
#                             tbin_dur, tbin_slide, 
#                             subspace_projection, NPCS_KEEP, 
#                             n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE, raw_subtract_mean_each_timepoint=False,
#                             hack_prune_to_these_chans = None,
#                             remove_singleprims_unstable=False):
#     """
#     Eucldian distance [effect of shape vs. task(context)] as function of time, relative to stroke onset.
#     """
#     from neuralmonkey.classes.population_mult import extract_single_pa
#     from neuralmonkey.scripts.analy_euclidian_chars_sp import preprocess_pa
#     import seaborn as sns
#     from pythonlib.tools.pandastools import append_col_with_grp_index
    
#     # Run
#     PA = extract_single_pa(DFallpa, bregion, which_level="stroke", event="00_stroke")

#     savedir = f"{SAVEDIR}/preprocess"
#     os.makedirs(savedir, exist_ok=True)
#     plot_drawings = False
#     PA = preprocess_pa(animal, date, PA, savedir, prune_version, 
#                         n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings,
#                         remove_chans_fr_drift=remove_drift,
#                         subspace_projection=subspace_projection, 
#                             twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
#                             raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
#                             remove_singleprims_unstable=remove_singleprims_unstable)

#     if PA is None:
#         return


#     if hack_prune_to_these_chans is not None:
#         # Optionally, keep specific chans
#         # chans_keep = [1053, 1054]
#         # chans_keep = [1044, 1049,  1053, 1054, 1057, 1059, 1062]
#         assert isinstance(hack_prune_to_these_chans, list)
#         PA = PA.slice_by_dim_values_wrapper("chans", hack_prune_to_these_chans)

#     ### Quick analyses of euclidian distances
#     vars_group = ["task_kind", "shape_semantic_grp"]
#     version = "traj"
#     DFDIST = PA.dataextractwrap_distance_between_groups(vars_group, version)

#     # DFDIST = append_col_with_grp_index(DFDIST, ["shape_semantic_grp_1", "shape_semantic_grp_2"], "shape_semantic_grp_same")
#     # DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_1", "task_kind_2"], "task_kind_same")
#     DFDIST["task_kind_same"] = DFDIST["task_kind_1"] == DFDIST["task_kind_2"]
#     DFDIST["shape_semantic_grp_same"] = DFDIST["shape_semantic_grp_1"] == DFDIST["shape_semantic_grp_2"]
#     DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_1", "task_kind_2"], "task_kind_12")
#     DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_same", "shape_semantic_grp_same"], "same-task|shape")
#     DFDIST = append_col_with_grp_index(DFDIST, ["shape_semantic_grp_same", "task_kind_12"], "same_shape|task_kind_12")
#     DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_same", "shape_semantic_grp_same"], "same-task|shape")
#     DFDIST = append_col_with_grp_index(DFDIST, ["shape_semantic_grp_1", "shape_semantic_grp_2"], "shape_semantic_grp_12")

#     pd.to_pickle(DFDIST, f"{SAVEDIR}/DFDIST.pkl")
    
#     for y in ["dist_mean", "dist_norm", "dist_yue_diff"]:
#         # sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
#         fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same-task|shape", kind="line", errorbar=("ci", 68))
#         savefig(fig, f"{SAVEDIR}/relplot-{y}-1.pdf")

#         fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
#         savefig(fig, f"{SAVEDIR}/relplot-{y}-2.pdf")

#         fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="task_kind_12", kind="line", col="same-task|shape", errorbar=("ci", 68))
#         savefig(fig, f"{SAVEDIR}/relplot-{y}-3.pdf")

#         if False: # slow, and I don't use
#             fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="shape_semantic_grp_12", kind="line", col="same-task|shape", 
#                         errorbar=("ci", 68), legend=False, alpha=0.5)
#             savefig(fig, f"{SAVEDIR}/relplot-{y}-4.pdf")

#         plt.close("all")


def euclidian_time_resolved(animal, date, PA, which_level, 
                                   prune_version, remove_drift, SAVEDIR, twind_analy,
                                    tbin_dur, tbin_slide, 
                                    subspace_projection, NPCS_KEEP, 
                                    n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE, raw_subtract_mean_each_timepoint=False,
                                    hack_prune_to_these_chans = None,
                                    remove_singleprims_unstable=False,
                                    subspace_projection_fitting_twind=None):
    """
    Eucldian distance [effect of shape vs. task(context)] as function of time, relative to stroke onset.
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.scripts.analy_euclidian_chars_sp import preprocess_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # which_level = "trial"
    # event = "03_samp"
    remove_trials_with_bad_strokes = True
    # Run
    # PA = extract_single_pa(DFallpa, bregion, which_level=which_level, event=event)
    savedir = f"{SAVEDIR}/preprocess"
    os.makedirs(savedir, exist_ok=True)
    plot_drawings = False
    if which_level == "trial":
        shape_var = "seqc_0_shapesemgrp"
        PA = preprocess_pa_trials(animal, date, PA, savedir, prune_version, shape_var=shape_var,
                            n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings,
                            remove_chans_fr_drift=remove_drift,
                            subspace_projection=subspace_projection, 
                                twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                                raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                remove_singleprims_unstable=remove_singleprims_unstable,
                                subspace_projection_fitting_twind=subspace_projection_fitting_twind)
    elif which_level == "stroke":
        shape_var = "shape_semantic_grp"
        PA = preprocess_pa(animal, date, PA, savedir, prune_version, 
                            n_min_trials_per_shape=n_min_trials_per_shape, shape_var=shape_var, plot_drawings=plot_drawings,
                            remove_chans_fr_drift=remove_drift,
                            subspace_projection=subspace_projection, 
                                twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                                raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                remove_singleprims_unstable=remove_singleprims_unstable,
                                remove_trials_with_bad_strokes=remove_trials_with_bad_strokes,
                                subspace_projection_fitting_twind=subspace_projection_fitting_twind)


    if PA is None:
        return

    if hack_prune_to_these_chans is not None:
        # Optionally, keep specific chans
        # chans_keep = [1053, 1054]
        # chans_keep = [1044, 1049,  1053, 1054, 1057, 1059, 1062]
        assert isinstance(hack_prune_to_these_chans, list)
        PA = PA.slice_by_dim_values_wrapper("chans", hack_prune_to_these_chans)

    ######################################
    ### Quick analyses of euclidian distances
    vars_group = ["task_kind", shape_var]
    version = "traj"
    DFDIST = PA.dataextractwrap_distance_between_groups(vars_group, version)

    DFDIST["task_kind_same"] = DFDIST["task_kind_1"] == DFDIST["task_kind_2"]
    DFDIST[f"{shape_var}_same"] = DFDIST[f"{shape_var}_1"] == DFDIST[f"{shape_var}_2"]
    DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_1", "task_kind_2"], "task_kind_12")
    DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_same", f"{shape_var}_same"], "same-task|shape")
    DFDIST = append_col_with_grp_index(DFDIST, [f"{shape_var}_same", "task_kind_12"], "same_shape|task_kind_12")
    DFDIST = append_col_with_grp_index(DFDIST, ["task_kind_same", f"{shape_var}_same"], "same-task|shape")
    DFDIST = append_col_with_grp_index(DFDIST, [f"{shape_var}_1", f"{shape_var}_2"], f"{shape_var}_12")

    pd.to_pickle(DFDIST, f"{SAVEDIR}/DFDIST.pkl")
    
    for y in ["dist_mean", "dist_norm", "dist_yue_diff"]:
        # sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
        fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same-task|shape", kind="line", errorbar=("ci", 68))
        savefig(fig, f"{SAVEDIR}/relplot-{y}-1.pdf")

        if False: # slow, and I don't use
            fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="same_shape|task_kind_12", kind="line", errorbar=("ci", 68))
            savefig(fig, f"{SAVEDIR}/relplot-{y}-2.pdf")

            fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue="task_kind_12", kind="line", col="same-task|shape", errorbar=("ci", 68))
            savefig(fig, f"{SAVEDIR}/relplot-{y}-3.pdf")

            fig = sns.relplot(data=DFDIST, x="time_bin", y=y, hue=f"{shape_var}_12", kind="line", col="same-task|shape", 
                        errorbar=("ci", 68), legend=False, alpha=0.5)
            savefig(fig, f"{SAVEDIR}/relplot-{y}-4.pdf")

        plt.close("all")


def run(animal, date,  DFallpa, SAVEDIR, subspace_projection, prune_version, NPCS_KEEP, 
        raw_subtract_mean_each_timepoint, n_min_trials_per_shape,
        PLOT_EACH_REGION, PLOT_STATE_SPACE,
        LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS,
        twind_analy, devo_return_data=False):
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
    tbin_slide = 0.02
    remove_drift = True
    remove_singleprims_unstable = True
    remove_trials_with_bad_strokes = True

    # Run
    list_res = []
    list_res_clean = []
    for i_br, bregion in enumerate(DFallpa["bregion"].unique().tolist()):
        
        savedir = f"{SAVEDIR}/each_region/{bregion}"
        os.makedirs(savedir, exist_ok=True)
        
        PA = extract_single_pa(DFallpa, bregion, which_level="stroke", event="00_stroke")

        savedir_this = f"{savedir}/preprocess"
        os.makedirs(savedir_this, exist_ok=True)
        plot_drawings = i_br ==0
        PAredu = preprocess_pa(animal, date, PA, savedir_this, prune_version, 
                           n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=plot_drawings,
                           subspace_projection=subspace_projection, 
                           twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                           raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint, 
                           remove_chans_fr_drift=remove_drift,
                           remove_singleprims_unstable = remove_singleprims_unstable, 
                           remove_trials_with_bad_strokes = remove_trials_with_bad_strokes)

        if PAredu is None:
            continue

        ### New, cleaner method, taking all pairwise distances between trials

        # (2) State space
        if PLOT_STATE_SPACE:
            # otherwise is redundant.
            list_dims = [(0,1), (2,3), (4,5)]
            PAredu.plot_state_space_good_wrapper(savedir, LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS, 
                                                 list_dims=list_dims, nmin_trials_per_lev=n_min_trials_per_shape)

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
        if subspace_projection in [None, "pca", "task_shape_si"]:
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
    if len(list_res)>0:
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

def plot_heatmap_firing_rates_all_wrapper(DFallpa, SAVEDIR_ANALYSIS, animal, date, 
                                          DEBUG_skip_drawings=False, DEBUG_bregion=None,
                                          DEBUG_subspace_projection=None):
    """
    Wrapper, to plot all FR heatmaps (trial vs time) with diff variations.
    Gaol is to visualize theraw data as clsoely and broadly as possible, see effect,
    and also see drift if it exists.

    NOTE: GOOD! Is using good time-window-restricted fitting of pca.
    """
    var_is_blocks = False
    n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
    raw_subtract_mean_each_timepoint = False

    twind_analy = TWIND_ANALY
    # tbin_dur = 0.1
    # tbin_slide = 0.01
    # NPCS_KEEP = 10
    tbin_dur = 0.2
    tbin_slide = 0.01
    NPCS_KEEP = 8

    var_effect="shape_semantic_grp"
    var_conj = "task_kind"
    
    drawings_done = []
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        if (DEBUG_bregion is not None) and (DEBUG_bregion!=bregion):
            continue

        # for prune_version in ["sp_char_0", "pig_char_1plus"]:
        for prune_version in LIST_PRUNE_VERSION:

            # Plot drwaings just once
            if prune_version not in drawings_done and (DEBUG_skip_drawings==False):
                PAtmp = DFallpa["pa"].values[0]
                savedir = f"{SAVEDIR_ANALYSIS}/drawings-prune={prune_version}"
                os.makedirs(savedir, exist_ok=True)
                preprocess_pa(animal, date, PAtmp, savedir, prune_version, 
                                n_min_trials_per_shape=n_min_trials_per_shape, plot_drawings=True)
                drawings_done.append(prune_version)

                try:
                    savedir = f"{SAVEDIR_ANALYSIS}/event_timing-prune={prune_version}"
                    os.makedirs(savedir, exist_ok=True)
                    beh_plot_event_timing_stroke(PAtmp, animal, date, savedir)
                except AssertionError as err:
                    pass

            # for subspace_projection in [None, "pca", subspace_projection_extra]:
            for subspace_projection in [None, "pca_proj", "task_shape_si"]: # NOTE: shape_prims_single not great, you lose some part of preSMA context-dependence...
                if (DEBUG_subspace_projection is not None) and (DEBUG_subspace_projection!=subspace_projection):
                    continue
                if subspace_projection is not None:
                    # plot only cleaned up data.
                    list_unstable_badstrokes = [(True, True, True)]
                else:
                    # Then also plot raw, without clearning
                    list_unstable_badstrokes = [(False, False, False), (True, True, True)]
                
                ########## RESTRICT TO FITTING TWIND
                if subspace_projection in [None, "pca"]:
                    list_fit_twind = [twind_analy]
                else:
                    list_fit_twind = SUBSPACE_PROJ_FIT_TWIND[event]

                for subspace_projection_fitting_twind in list_fit_twind:
                
                    # for remove_drift in [False]:
                    for remove_drift, remove_singleprims_unstable, remove_trials_with_bad_strokes in list_unstable_badstrokes:
                        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-SpUnstable={remove_singleprims_unstable}-RmBadStrks={remove_trials_with_bad_strokes}-fit_twind={subspace_projection_fitting_twind}"
                        os.makedirs(SAVEDIR, exist_ok=True)

                        print("SAVING AT ... ", SAVEDIR)

                        # Preprocess
                        savedir = f"{SAVEDIR}/preprocess"
                        os.makedirs(savedir, exist_ok=True)
                        shape_var = "shape_semantic_grp"
                        pa = preprocess_pa(animal, date, PA, savedir, prune_version, 
                                            n_min_trials_per_shape=n_min_trials_per_shape, shape_var=shape_var, plot_drawings=False,
                                            remove_chans_fr_drift=remove_drift, subspace_projection=subspace_projection, 
                                                twind_analy=twind_analy, tbin_dur=tbin_dur, tbin_slide=tbin_slide, NPCS_KEEP=NPCS_KEEP,
                                                raw_subtract_mean_each_timepoint=False, remove_singleprims_unstable=remove_singleprims_unstable,
                                                remove_trials_with_bad_strokes=remove_trials_with_bad_strokes, 
                                                subspace_projection_fitting_twind=subspace_projection_fitting_twind)

                        # # For raw data, pick out 10 most modulated neurons
                        # if False:
                        #     from neuralmonkey.metrics.scalar import _calc_modulation_by_frsm_event_aligned_time
                        #     res = []
                        #     for i, chan in enumerate(PA.Chans):
                        #         frmat = PA.X[i, :, :]
                        #         r2 = _calc_modulation_by_frsm_event_aligned_time(frmat)
                        #         res.append({
                        #             "r2":r2,
                        #             "chan":chan,
                        #             "indchan":i
                        #         })
                        #     animal, date
                        #     df = pd.DataFrame(res)
                        #     import seaborn as sns
                        #     fig = sns.catplot(data=df, x="chan", y="r2", aspect=2.5, kind="bar")
                        #     from pythonlib.tools.snstools import rotateLabel
                        #     rotateLabel(fig)
                        #     nchans = 10
                        #     list_indchan = df.sort_values("r2", ascending=False)["indchan"].tolist()[:nchans]
                        #     PA = PA.slice_by_dim_indices_wrapper("chans", list_indchan)

                        # for subtr_version in [None, "time", "baseline"]:

                        #     diverge = False
                        #     if subtr_version is None:
                        #         pathis = pa.copy()
                        #     elif subtr_version == "time":
                        #         pathis = pa.norm_subtract_trial_mean_each_timepoint()
                        #         diverge = True
                        #     elif subtr_version == "baseline":
                        #         twind_base = [-0.5, -0.05]
                        #         pathis = pa.norm_rel_base_window(twind_base, "subtract")
                        #         diverge = True
                        #     else:
                        #         assert False

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

                            for mean_over_trials in [False, True]:
                                # savedirthis = f"{SAVEDIR}/HEATMAP-subtr_version={subtr_version}-mean={mean_over_trials}"
                                savedirthis = f"{SAVEDIR}/HEATMAP-zscore={zscore}-subtr_time_mean={subtr_time_mean}-subtrbase={subtr_baseline}-mean={mean_over_trials}"
                                
                                os.makedirs(savedirthis, exist_ok=True)

                                from neuralmonkey.neuralplots.population import heatmapwrapper_many_useful_plots
                                heatmapwrapper_many_useful_plots(pathis, savedirthis, var_effect=var_effect, 
                                                                var_conj=var_conj, 
                                                                var_is_blocks=var_is_blocks, mean_over_trials=mean_over_trials,
                                                                flip_rowcol=True, plot_fancy=True, n_rand_trials=5,
                                                                diverge=diverge)

                            ###################################### Running euclidian
                            if subspace_projection is not None:
                                # savedirthis = f"{SAVEDIR}/SS-subtr_version={subtr_version}"
                                savedirthis = f"{SAVEDIR}/SS-zscore={zscore}-subtr_time_mean={subtr_time_mean}-subtrbase={subtr_baseline}-mean={mean_over_trials}"
                                os.makedirs(savedirthis, exist_ok=True)
                                nmin_trials_per_lev = 5
                                # Plot state space
                                LIST_VAR = [
                                    var_effect,
                                ]
                                LIST_VARS_OTHERS = [
                                    (var_conj,),
                                ]
                                PLOT_CLEAN_VERSION = True
                                list_dim_timecourse = list(range(NPCS_KEEP))
                                list_dims = [(0,1), (1,2), (2,3), (3,4)]
                                pathis.plot_state_space_good_wrapper(savedirthis, LIST_VAR, LIST_VARS_OTHERS, PLOT_CLEAN_VERSION=PLOT_CLEAN_VERSION,
                                                                list_dim_timecourse=list_dim_timecourse, list_dims=list_dims,
                                                                nmin_trials_per_lev=nmin_trials_per_lev)                

# def plot_heatmap_firing_rates_all(PA, savedir):
#     """
#     To look at "raw" data, plotting heatmaps (trial vs time), a few different kinds

#     """

    # from pythonlib.tools.snstools import heatmap_mat
    # from  neuralmonkey.neuralplots.population import _heatmap_stratified_y_axis
    # from neuralmonkey.neuralplots.population import heatmap_stratified_trials_grouped_by_neuron, heatmap_stratified_neuron_grouped_by_var
    # import numpy as np
    # from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    # print("--- Saving at: ", savedir)
    # ######## PREPROCESS
    # # # quicker, smaller 
    # # dur = 0.1
    # # slide = 0.02
    # # PA = PA.agg_by_time_windows_binned(dur, slide)

    # # Get a global zlim
    # zlims = np.percentile(PA.X.flatten(), [1, 99]).tolist()

    # #########################################################
    # ############################# (1) 
    # dflab = PA.Xlabels["trials"]
    # if False:
    #     # OK, but does not structure in terms of rows/cols explciitly,
    #     grpdict = grouping_append_and_return_inner_items_good(dflab, ["shape_semantic_grp", "task_kind"])

    #     n_rand_trials = 10

    #     SIZE = 3
    #     ncols = 6
    #     nrows = int(np.ceil(len(grpdict)/ncols))
    #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))

    #     for ax, (grp, inds) in zip(axes.flatten(), grpdict.items()):

    #         heatmap_stratified_trials_grouped_by_neuron(PA, inds, ax, n_rand_trials=n_rand_trials, zlims=None)

    #         ax.set_title(grp, fontsize=8)
    #         # assert False
    #     savefig(fig, f"{savedir}/grouped_by_neuron.pdf")
    #     plt.close("all")
    # else:
    #     # Better, 
    #     var_row = "shape_semantic_grp"
    #     var_col = "task_kind"

    #     row_levels = dflab[var_row].unique().tolist()
    #     col_levels = dflab[var_col].unique().tolist()

    #     grpdict = grouping_append_and_return_inner_items_good(dflab, [var_row, var_col])

    #     n_rand_trials = 10
    #     nticks = 5

    #     SIZE = 3
    #     ncols = len(col_levels)
    #     nrows = len(row_levels)
    #     fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))


    #     for i, row in enumerate(row_levels):
    #         for j, col in enumerate(col_levels):
    #             ax = axes[i][j]
    #             inds = grpdict[(row, col)]

    #             heatmap_stratified_trials_grouped_by_neuron(PA, inds, ax, n_rand_trials=n_rand_trials, zlims=zlims)

    #             ax.set_title((row, col), fontsize=8)
    #             # assert False

    #             # times = pa.Times
    #             # _inds = np.linspace(0, len(times)-1, nticks).astype(int)
    #             # _vals = times[_inds]
    #             # _vals = [f"{v:.3f}" for v in _vals]
    #             # ax.set_xticks(_inds+0.5, _vals, rotation=45, fontsize=6)
    #     savefig(fig, f"{savedir}/grouped_by_neuron.pdf")
    #     plt.close("all")


    # #########################################################
    # ### (2)  Same thing, but each plot is a single (PC, task_kind), split by shape
    # shapes = sorted(PA.Xlabels["trials"]["shape_semantic_grp"].unique().tolist())
    # task_kinds = sorted(PA.Xlabels["trials"]["task_kind"].unique().tolist())

    # list_ind_neur = list(range(len(PA.Chans)))
    # ncols = len(list_ind_neur)
    # nrows = len(task_kinds)
    # SIZE = 3.5
    # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), squeeze=False)
    # for j, tk in enumerate(task_kinds): # row
    #     pa_this = PA.slice_by_labels_filtdict({"task_kind":[tk]})

    #     for i, ind_neur in enumerate(list_ind_neur): # columns

    #         ax = axes[j][i]
    #         heatmap_stratified_neuron_grouped_by_var(pa_this, ind_neur, ax, n_rand_trials=n_rand_trials, zlims=zlims, 
    #                                         y_group_var="shape_semantic_grp", y_group_var_levels=shapes)

    #         ax.set_title(f"neur={i}-{PA.Chans[i]}|tk={tk}")
    # savefig(fig, f"{savedir}/grouped_by_var.pdf")

    # # (3)  Loop over all bregions
    # from neuralmonkey.neuralplots.population import heatmap_stratified_each_neuron_alltrials
    # fig = heatmap_stratified_each_neuron_alltrials(PA, "task_kind")
    # savefig(fig, f"{savedir}/each_neuron_over_time.pdf")
    # plt.close("all")


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
    trial_ver = int(sys.argv[4])==1

    if trial_ver:
        # Trial level (samp, raise)
        question = "CHAR_BASE_trial"
        PLOTS_DO = [3]
        version = "trial"
    else:
        # Stroke level
        question = "CHAR_BASE_stroke"
        # PLOTS_DO = [1, 2]
        # PLOTS_DO = [2, 1]
        PLOTS_DO = [2, 1, 0, 4] # GOOD
        # PLOTS_DO = [0] # STATE SPACE
        # PLOTS_DO = [4] # Just beh drwaings
        version = "stroke"

        # FINAL, good code:
        PLOTS_DO = [1, 5, 2, 4] # GOOD
        PLOTS_DO = [5] #
        # PLOTS_DO = [2] #

    # if animal=="Diego":
    #     combine = True
    # elif animal=="Pancho":
    #     combine = False
    # else:
    #     assert False
    
    # Load a single DFallPA
    DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, question=question, return_none_if_no_exist=True)
    if DFallpa is None:
        # Then extract
        DFallpa = extract_dfallpa_helper(animal, date, question, combine, do_save=True)

    # Make a copy of all PA before normalization
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

    from neuralmonkey.classes.population_mult import dfallpa_preprocess_sort_by_trialcode
    dfallpa_preprocess_sort_by_trialcode(DFallpa)

    # Determine if rows are bad or good beh storkes (dont prune yet)
    if version == "stroke":
        # if trial, then fails...
        behstrokes_preprocess_assign_col_bad_strokes(DFallpa, animal, date)

    ################ PARAMS
    for plotdo in PLOTS_DO:
        if plotdo==0:
            """
            Euclidina distance (scalar), effect of shape vs. task. This is like the oriigal plots, but cleaner.
            Not important, as the plots below (#2) get time-varying, which you can just take mean over to get identical
            to this.
            Useful part: State space plots
            """
            
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/STATE_SPACE/{animal}-{date}-combine={combine}"

            ######### PARAMS
            n_min_trials_per_shape = N_MIN_TRIALS_PER_SHAPE
            # LIST_NPCS_KEEP = [4,6,2]
            LIST_NPCS_KEEP = [6]
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
                {"task_kind":["prims_on_grid", "character"]},
                ]

            # for twind_analy in [(0.05, 0.25), (-0.05, 0.35), (0.1, 0.2)]:
            for twind_analy in [TWIND_ANALY]:
                for subspace_projection in ["shape_prims_single", "pca", "task_shape_si"]:
                    for prune_version in ["sp_char_0", "pig_char_0", "pig_char_1plus", "sp_char"]:
                        for NPCS_KEEP in LIST_NPCS_KEEP:
                            for raw_subtract_mean_each_timepoint in [False, True]:
                                SAVEDIR = f"{SAVEDIR_ANALYSIS}/subspc={subspace_projection}-prunedat={prune_version}-npcs={NPCS_KEEP}-subtr={raw_subtract_mean_each_timepoint}-twind={twind_analy}"
                                os.makedirs(SAVEDIR, exist_ok=True)
                                
                                PLOT_STATE_SPACE = NPCS_KEEP == max(LIST_NPCS_KEEP)
                                run(animal, date, DFallpa, SAVEDIR, subspace_projection, prune_version, NPCS_KEEP, 
                                        raw_subtract_mean_each_timepoint, n_min_trials_per_shape,
                                        PLOT_EACH_REGION, PLOT_STATE_SPACE,
                                        LIST_VAR, LIST_VARS_OTHERS, LIST_FILTDICT, LIST_PRUNE_MIN_N_LEVS, twind_analy)
                                
        elif plotdo==1:
            """
            [GOOD!] Heatmaps, timecourses, and state-space plots.
            GOOD - using latest code with pca_fit window.

            Heatmaps of raw activity, different variations. 
            Useful, to get intuition of what is going on in euclidian (#2)
            Also plots drawings comparing SP vs. CHAR
            """
            SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/HEATMAPS/{animal}-{date}-combine={combine}"
            os.makedirs(SAVEDIR, exist_ok=True)
            print(SAVEDIR)
            plot_heatmap_firing_rates_all_wrapper(DFallpa, SAVEDIR, animal, date)

        elif plotdo==2:
            """
            Time-resolved euclidian distances - STROKES VERSION
            
            Agg across multiple bregions and dates:
            # MULT DATA - euclidian_time_resolved" in 241002_char_euclidian...
            """
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/{animal}-{date}-combine={combine}-wl={version}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            print(SAVEDIR_ANALYSIS)
            euclidian_time_resolved_wrapper(animal, date, DFallpa, SAVEDIR_ANALYSIS)
        
        elif plotdo==3:
            """
            Time-resolved euclidian distances -- TRIAL version
            (i.e., samp, raise)
            """
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/{animal}-{date}-combine={combine}-wl={version}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            print(SAVEDIR_ANALYSIS)
            euclidian_time_resolved_wrapper_trial(animal, date, DFallpa, SAVEDIR_ANALYSIS)     

        elif plotdo==4:
            """ Just behavior, plot strokes, sorted by shape and task_kind (col) and by clust_sim_max (row)
            Useful for evaluating how close each shape is across task_kind
            """
            PA = DFallpa["pa"].values[0] # assume all PA have same behavior.
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/stroke_drawings/{animal}-{date}-combine={combine}-wl={version}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            print(SAVEDIR_ANALYSIS)
            behstrokes_extract_char_clust_sim(PA, animal, date, SAVEDIR_ANALYSIS, PLOT=True)

        elif plotdo==5:
            """
            GOOD - for scoring euclidian distance, doing all the right careful things, like train-test splits.
            This is quick, but does not get tiemcourse.

            Then, do mult plots (summaries across dates and animals) using notebook: 
            /home/lucas/code/neuralmonkey/neuralmonkey/notebooks_tutorials/241002_char_euclidian_pop.ipynb
            "### [MULT DAYS] for: euclidian_time_resolved_fast_shuffled"
            """
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_QUICK_SHUFFLE/{animal}-{date}-combine={combine}-wl={version}"
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            print(SAVEDIR_ANALYSIS)
            DO_RSA_HEATMAPS = True
            euclidian_time_resolved_fast_shuffled(DFallpa, animal, date, SAVEDIR_ANALYSIS, DO_RSA_HEATMAPS=DO_RSA_HEATMAPS)
        else:
            print(plotdo)
            assert False   