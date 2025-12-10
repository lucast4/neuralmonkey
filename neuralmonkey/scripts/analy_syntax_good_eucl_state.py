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

TWIND_ANALY = (-0.6, 1.0) # This is just for windowing final data, not for fitting pca.
NPCS_KEEP = 8

N_MIN_TRIALS = 4 # min trials per level, otherwise throws level out.
NPCS_KEEP = 8

# PLOT_STATE_SPACE
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

def params_seqsup_extract_and_process(DFDIST, animal, date, shape_or_loc_rule="shape", 
                                      print_existing_variables=False,
                                      remove_first_stroke = False):
    """
    PARAMS:
    - DFDIST, dfdist across all days.
    - date, int, to pick out and process this specific date.
        # date = 250320 
    - shape_or_loc_rule, "shape" or "loc", whether to analyze epochs using shape or location rule
    -- OR --
    returns None if this date doesnt have data for this rule.
    """

        
    ### Prep params
    if shape_or_loc_rule == "shape":
        contrast_idx = 18
    elif shape_or_loc_rule == "loc":
        contrast_idx = 17
    else:
        print(shape_or_loc_rule)
        assert False
    
    ### pull out a single day, and just this contrast
    dfdist = DFDIST[(DFDIST["date"] == date) & (DFDIST["contrast_idx"] == contrast_idx)].reset_index(drop=True)

    def _error_print():
        """
        print the list of all unique pairs of lables that exist.
        """
        from pythonlib.tools.pandastools import grouping_print_n_samples
        print(animal, date)
        grouping_print_n_samples(dfdist, ["labels_1", "labels_2"])    

    ### Get hand-coded epoch params
    if contrast_idx==18:
        # shape rule
        if animal == "Pancho":
            if date in [230920, 230921, 230921, 230923]:
                epoch_name = "AnBmCk2"
                epoch_colorsup = "AnBmCk2"
            elif date in [231019]:
                epoch_name = "AnBmCk2" # epoch
                epoch_colorsup = "AnBmCk2|0" # epoch|colorsup
            elif date in [240828, 240829]:
                epoch_name = "gramP2b" # epoch
                epoch_colorsup = "gramP2b" # epoch|colorsup
            elif date in [250324]:
                epoch_name = "gramP3b" # epoch
                epoch_colorsup = "gramP3b" # epoch|colorsup
            else:
                _error_print()
                assert False
        elif animal == "Diego":
            if date in [230920, 230921, 230922]:
                epoch_name = "llCV3" 
                epoch_colorsup = "llCV3" 
            elif date in [230924, 230925]:
                epoch_name = "llCV3" 
                epoch_colorsup = "llCV3|0" 
            elif date in [250320]:
                epoch_name = "gramD5" 
                epoch_colorsup = "gramD5" 
            else:
                _error_print()
                assert False
        else:
            _error_print()
            assert False

    elif contrast_idx == 17:
        # Direction rule
        if animal == "Pancho":
            if date in [230920, 230921, 230923]:
                epoch_name = "L" 
                epoch_colorsup = "L" 
            elif date in [231019, 240828, 240829, 250324]:
                # Confirmed that these dates not expected to have (using gsheets)
                epoch_name = None
                epoch_colorsup = None
            else:
                _error_print()
                assert False
        elif animal == "Diego":
            if date in [230920, 230921, 230922]:
                epoch_name = "UL" 
                epoch_colorsup = "UL" 
            elif date in [230924, 230925, 250320]:
                # Confirmed that these dates not expected to have (using gsheets)
                epoch_name = None
                epoch_colorsup = None
            else:
                _error_print()
                assert False
        else:
            _error_print()
            assert False
    else:
        _error_print()
        assert False

    if epoch_name is None:
        # Then this date does not have this rule.
        return None, None, None
    
    ### Convert to strings
    epochset_name = f"('{epoch_name}',)" # epochset_name = "('AnBmCk2',)"
    epoch_no_superv = f"{epoch_colorsup}|0"
    if (animal, date) in [("Pancho", 250324)]:
        # for pancho, AnBmCk has no direction, so instead use hand-coded preset directions for seqsup
        epoch_superv = f"presetrand|1"
    else:
        epoch_superv = f"{epoch_colorsup}|S|1"
    
    ###
    # vars_others_keep = [
    #     "('gramP2b',)|gramP2b|0",
    #     "('gramP2b',)|gramP2b|S|1"
    # ]
    vars_others_keep = [
        f"{epochset_name}|{epoch_no_superv}",
        f"{epochset_name}|{epoch_superv}",
    ]
    print("vars_others_keep:", vars_others_keep)

    ### Do two things:
    # - Keep only tasks that are in correct epochset
    # - Keep only pairs (comparisons) with same _vars_others
    dfdistthis = dfdist[(dfdist["_vars_others_1"].isin(vars_others_keep)) & (dfdist["_vars_others_same"] == True) & (dfdist["same-stroke_index|_vars_others"] == "0|1" )].reset_index(drop=True)

    if remove_first_stroke:
        dfdistthis = dfdistthis[(dfdistthis["stroke_index_1"]>0) & (dfdistthis["stroke_index_2"]>0)].reset_index(drop=True)

    if len(dfdistthis) == 0:
        _error_print()
        assert False

    ### Get the var_effect.
    tmp = dfdistthis["var_effect"].unique().tolist()
    assert len(tmp)==1
    var_effect = tmp[0]

    if print_existing_variables:
        _error_print()

    ##### Give generic variable values, useful for combining across dates.
    ### (1) var_other, make it [False, True], i.e., whether is sequence superv.
    map_varother_to_superv = {varother:superv for varother, superv in zip(vars_others_keep, [False, True])}
    print("Mapping var_other to generic booleans (is_superv): ", map_varother_to_superv)

    dfdistthis["var_other_is_superv"] = [map_varother_to_superv[x] for x in dfdistthis["_vars_others_1"]]
    # Sanity check: confirm that can just use _vars_others_1 (it is identical to _vars_others_2)
    assert np.all(dfdistthis["_vars_others_1"] == dfdistthis["_vars_others_2"])    


    return dfdistthis, var_effect, vars_others_keep

def params_seqsup_extract_and_process_new(DFDIST, date, contrast_version="shape_within_chunk"):
    """
    For new(good) seqsup mult analysis, extract a single day and do relevant presprocessing.
    PARAMS:
    - DFDIST, dfdist across all days.
    - date, int, to pick out and process this specific date.
        # date = 250320 
    - contrast_version, See within, the rule you want to analyze.
    """

    if contrast_version=="shape_within_chunk":
        # Control for chunk, test effect of index within chunk
        contrast_idx = 0
    elif contrast_version=="shape_index":
        # Not controlling for chunk/shape, tests effect of stroke index
        contrast_idx = 1
    else:
        assert False

    ### Pull out a single day, and just this contrast
    dfdist = DFDIST[(DFDIST["date"] == date) & (DFDIST["contrast_idx"] == contrast_idx)].reset_index(drop=True)

    ### Pull out variables
    tmp = dfdist["vars_others"].unique()
    if not len(tmp)==1:
        print(tmp)
        assert False
    vars_others = tmp[0]

    tmp = dfdist["var_effect"].unique()
    if not len(tmp)==1:
        print(tmp)
        assert False
    var_effect = tmp[0]

    ### Extract individual variables into new columns
    assert all(dfdist["_vars_others_1"] == dfdist["_vars_others_2"]), "can consolidate into one column"

    for i, var in enumerate(vars_others):
        assert f"{var}" not in dfdist
        assert f"{var}_1" not in dfdist
        assert f"{var}_2" not in dfdist
        dfdist[f"{var}"] = [x[i] for x in dfdist["_vars_others_1"]]
        # dfdist[f"{var}"] = [x[i] for x in dfdist["_vars_others_2"]]

    ### Only keep "diff effect"
    dfdist = dfdist[dfdist[f"{var_effect}_same"] == False].reset_index(drop=True)

    ### Determine a conjunctive variable that is a datapt for plotting in scatterplot
    from pythonlib.tools.pandastools import append_col_with_grp_index
    
    if "chunk_rank" in vars_others and "shape" in vars_others:
        vars_datapt = ["chunk_rank", "shape"]
        dfdist = append_col_with_grp_index(dfdist, vars_datapt, "chrnk_shp")
        
        vars_datapt = ["chunk_rank", "shape", "superv_is_seq_sup"]
        dfdist = append_col_with_grp_index(dfdist, vars_datapt, "chrnk_shp_superv")

        var_datapt = "chrnk_shp"
    else:
        # var_datapt = "epochset_shape"
        var_datapt = None
    
    return dfdist, vars_others, var_effect, var_datapt

def params_get_contrasts_of_interest(return_list_flat=True):
    """
    Automaticlaly getting list of idxs for all contrasts (i.e, var-var_other pair) of interest.
    These indices index into LIST_VAR, etc.
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import params_pairwise_variables_for_plotting
    LIST_LIST_VVO_XY, LIST_dir_suffix= params_pairwise_variables_for_plotting()
    DICT_VVO = {}
    for LIST_VVO_XY, dir_suffix in zip(LIST_LIST_VVO_XY, LIST_dir_suffix):
        DICT_VVO[dir_suffix] = LIST_VVO_XY

    # Collect the contrasts that I actually use in analysis
    if return_list_flat:
        list_contrast_idx = []
        for k, list_contrast_pair in DICT_VVO.items():
            for contrast_pair in list_contrast_pair:
                for contrast in contrast_pair:
                    _ind = contrast.find("|")
                    idx_this = int(contrast[:_ind])
                    list_contrast_idx.append(idx_this)
        list_contrast_idx = sorted(set(list_contrast_idx))
        print("Getting these contrasts: ", list_contrast_idx)    
        return list_contrast_idx
    else:
        # return as dict
        DICT_VVO_TO_LISTIDX = {} # e.g., {two_shape_sets': [[14, 16], [13, 15]]}
        for dir_suffix, list_contrast_pair in DICT_VVO.items():
            DICT_VVO_TO_LISTIDX[dir_suffix] = []
            for contrast_pair in list_contrast_pair:
                
                list_idx = []
                for contrast in contrast_pair:
                    _ind = contrast.find("|")
                    idx_this = int(contrast[:_ind])
                    list_idx.append(idx_this)

                DICT_VVO_TO_LISTIDX[dir_suffix].append(list_idx)
                # print(dir_suffix, list_idx)
        return DICT_VVO_TO_LISTIDX

def preprocess_dfallpa_motor_features(DFallpa, tmax=0.2, plot_motor_values=False, do_zscore=False):
    """
    Collect stroke-related features for each pa, doing it once (for the first pa in DFallpa) and then copying
    to the other PA. THis is quicker than doing it for each PA (as they have identical behaviroal data). 
    Thuis is useful for acting as motor controls later.

    PARAMS:
    - tmax, to define time window (rel stroke onset)
    """

    # assert len(DFallpa["bregion"].unique())==len(DFallpa), "assuming they are all the same trials, etc"
    variables_cont = ("motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "gap_to_next_x", "gap_to_next_y", "velmean_x", "velmean_y")

    # Extract all motor stuff
    pa = DFallpa["pa"].values[0]
    pa.behavior_strokes_kinematics_stats(twind=[0., tmax])
    dflab = pa.Xlabels["trials"]

    if do_zscore:
        # Normalize all continuous variables.
        dflab.loc[:, variables_cont] = (dflab.loc[:, variables_cont] - dflab.loc[:, variables_cont].mean()) / dflab.loc[:, variables_cont].std()

    # Add this to every other pa
    for pa in DFallpa["pa"].values:
        dflab_this = pa.Xlabels["trials"]

        # check they are identical rows
        assert all(dflab.loc[:, ["trialcode", "stroke_index"]] == dflab_this.loc[:, ["trialcode", "stroke_index"]])

        # cols
        dflab_this.loc[:, variables_cont] = dflab.loc[:, variables_cont]

    if plot_motor_values:
        # Plots showing the continuous motor variables
        from pythonlib.tools.pandastools import stringify_values
        import seaborn as sns

        dflab_str = stringify_values(dflab)
        
        if "DIFF_gridloc" not in dflab_str:
            dflab_str["DIFF_gridloc"] = "dummy"

        sns.catplot(data=dflab_str, y="DIFF_gridloc", x="gap_from_prev_x", hue="gap_from_prev_y", alpha=0.15)
        sns.catplot(data=dflab_str, y="DIFF_gridloc", x="gap_from_prev_y", hue="gap_from_prev_x", alpha=0.15)

        S = 6
        for x, y in [
            ("motor_onsetx", "motor_onsety"),
            ("gap_from_prev_x", "gap_from_prev_y"),
            ("velmean_x", "velmean_y")]:
            fig, axes = plt.subplots(1, 3, figsize=(3*S, S), sharex=True, sharey=True)
            for ax, hue in zip(axes.flatten(), ["gridloc", "DIFF_gridloc", "shape"]):
                sns.scatterplot(data=dflab_str, x=x, y=y, hue=hue, marker="x", alpha=0.3, ax=ax)    

def preprocess_pa_syntax(PA):
    """
    Preprocessing, extract variables related to syntax stuff.
    
    Should always run for syntax data, even if it trial or stroke

    RETURNS:
    - Modifies PA (nothing returned)
    """

    dflab = PA.Xlabels["trials"]

    if False: # Actually, is fine
        if any(dflab["epoch"].isin(["base", "baseline"])):
            print(dflab["epoch"].unique())
            assert False, "will fail below, so remove base"

    # dflab = append_col_with_grp_index(dflab, vars_others, "_var_other")
    dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_concrete"], "epch_sytxcncr")
    # dflab = append_col_with_grp_index(dflab, ["epoch", "seqc_0_loc", "seqc_0_shape", "syntax_concrete"], "var_all_conditions")

    # Add n items in each shape slot
    nslots = len(dflab["syntax_concrete"].values[0])
    list_slots = []
    for i in range(3):
        key = f"syntax_slot_{i}"
        if i > nslots-1:
            # Then this slot doesnt exist
            dflab[key] = 0
        else:
            # Assign how many in that slot 
            tmp = []
            for x in dflab["syntax_concrete"]:
                if x==('IGNORE',):
                    tmp.append(-1)
                elif isinstance(x, str):
                    # e.e.g, "none" if this is baseline data
                    tmp.append(-1)
                else:
                    if i+1>len(x):
                        # Then this slot doesnt exist. This happens for two-shape days, sometimes. 
                        tmp.append(0)
                    else:
                        if isinstance(x[i], str):
                            print(x)
                            assert False
                        tmp.append(x[i])
            # print(set(tmp))
            dflab[key] = tmp
        list_slots.append(key)
        # print(dflab[key])
    print("Added these columns to dflab: ", list_slots)

    # Add ratio between slot 0 and 1
    if ("syntax_slot_0" in dflab) & ("syntax_slot_1" in dflab):
        # Add 0.01 so that doesnt devide by 0.
        # print(dflab["syntax_slot_0"].unique())
        # print(dflab["syntax_slot_1"].unique())
        # print(dflab["syntax_slot_2"].unique())

        dflab["syntax_slot_ratio"] = (dflab["syntax_slot_1"]+0.01)/(dflab["syntax_slot_0"]+0.01 + dflab["syntax_slot_1"]+0.01)

        if np.any(np.isnan(dflab["syntax_slot_ratio"])):
            print(dflab["syntax_slot_ratio"])
            assert False

    # count up how many unique shapes are shown
    def _n_shapes(syntax_concrete):
        # e.g., (1,3,0) --> 2
        if syntax_concrete==('IGNORE',):
            tmp.append(-1)
        elif isinstance(syntax_concrete, str):
            return -1
        else:
            return sum([x>0 for x in syntax_concrete])
    dflab["shapes_n_unique"] = [_n_shapes(sc) for sc in dflab["syntax_concrete"]]

    ### For SHAPE vs. SUPERV
    # Better variables that don't require using the specific name (e.g, llCV3)
    # epoch_rand_exclsv: (llCV3, UL, seqsup, colorrank)
    # epoch_kind: (shape, dir, seqsup, colorrank)
    tmp = []
    tmp2 = []
    for i, row in dflab.iterrows():
        if row["superv_is_seq_sup"]:
            # Regardless of the "epoch" these are all random. The color cue will indicate random, etc.
            tmp.append("seqsup")
            tmp2.append("seqsup")
            assert not row["superv_COLOR_METHOD"] == "rank"
        elif row["superv_COLOR_METHOD"] == "rank":
            tmp.append("colorrank")
            tmp2.append("colorrank")
            assert not row["superv_is_seq_sup"]
        else:
            # This this is nether colorrank nor seqsup
            tmp.append(row["epoch"]) 
            if row["epoch_is_AnBmCk"] and (not row["epoch_is_DIR"]):
                tmp2.append("shape")
            elif (not row["epoch_is_AnBmCk"]) and row["epoch_is_DIR"]:
                tmp2.append("dir")
            else:
                assert False
    dflab["epoch_rand_exclsv"] = tmp
    dflab["epoch_kind"] = tmp2

    # Store
    PA.Xlabels["trials"] = dflab


def preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials, prune_min_n_levs, filtdict,
                  savedir, 
                subspace_projection, subspace_projection_fitting_twind,
                twind_analy, tbin_dur, tbin_slide, scalar_or_traj="traj",
                use_strings_for_vars_others=True, is_seqsup_version=False,
                skip_dimredu=False, prune_by_conj_var = True):
    """
    Preprocess, for strokes-level data
    RETURNS:
    - Modifies PA, but also returns...
    
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_print_n_samples
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _preprocess_pa_dim_reduction
    from pythonlib.tools.plottools import savefig
    PA = PA.copy()
    dflab = PA.Xlabels["trials"]

    #### Append any new columns
    dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_role"], "epch_sytxrol")
    dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_role", "shape", "gridloc"], "sytx_all")
    dflab = append_col_with_grp_index(dflab, ["epoch_rand", "shape", "syntax_role", "superv_is_seq_sup"], "stxsuperv")
    dflab = append_col_with_grp_index(dflab, ["chunk_rank", "shape"], "chunk_shape")

    # Also add a conjunctive rank within
    # This is the interaction between cr and chuk_within
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dflab = append_col_with_grp_index(dflab, ["chunk_rank", "shape", "chunk_n_in_chunk", "chunk_within_rank"], "rank_conj")

    # Also reach direction
    tmp = []
    for _, row in dflab.iterrows():
        loc = row["gridloc"]
        loc_prev = row["CTXT_loc_prev"]

        if loc_prev[1] == "START":
            loc_diff = (0, "START")
        else:
            loc_diff = (loc[0] - loc_prev[0], loc[1] - loc_prev[1])

        tmp.append(loc_diff)
    dflab["DIFF_gridloc"] = tmp

    # Also shape change
    tmp = []
    for _, row in dflab.iterrows():
        shape = row["shape"]
        shape_prev = row["CTXT_shape_prev"]

        if shape_prev == "START":
            shape_diff = "START"
        else:
            shape_diff = (shape_prev, shape)

        tmp.append(shape_diff)
    dflab["DIFF_shape"] = tmp

    # Consolidate vars_others into a single variable
    # save text file holding the params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({"var_effect":var_effect, "vars_others":vars_others}, f"{savedir}/vars.txt")
    dflab = append_col_with_grp_index(dflab, vars_others, "_vars_others", use_strings=use_strings_for_vars_others)

    ### (0) Plot original tabulation of shape vs task_klind
    if savedir is not None:
        fig = grouping_plot_n_samples_conjunction_heatmap(dflab, var_effect, "_vars_others")
        path = f"{savedir}/counts-orig.pdf"
        savefig(fig, path)

        path = f"{savedir}/counts-orig.txt"
        grouping_print_n_samples(dflab, ["_vars_others", var_effect], savepath=path)

    ### Do things to PA
    # Put back into PA
    PA.Xlabels["trials"] = dflab

    # Prune n datapts -- var_conj
    if prune_by_conj_var:
        plot_counts_heatmap_savepath = f"{savedir}/counts_conj.pdf"
        PA, _, _= PA.slice_extract_with_levels_of_conjunction_vars(var_effect, vars_others, prune_min_n_trials, prune_min_n_levs,
                                                            plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
    
    # Prune n datapts -- each level of flat grp vars
    _grp_vars = [var_effect] + vars_others
    PA = PA.slice_extract_with_levels_of_var_good_prune(_grp_vars, prune_min_n_trials)
    
    if PA is None:
        print("Pruned all data!!")
        return None

    if filtdict is not None:
        for _var, _levs in filtdict.items():
            PA = PA.slice_by_labels("trials", _var, _levs, verbose=True)

    # Plot final conjunctions
    dflab = PA.Xlabels["trials"]
    if len(dflab) == 0:
        print("Pruned all data!!")
        return None
    
    if savedir is not None:
        fig = grouping_plot_n_samples_conjunction_heatmap(dflab, var_effect, "_vars_others")
        path = f"{savedir}/counts-final.pdf"
        savefig(fig, path)

        path = f"{savedir}/counts-final.txt"
        grouping_print_n_samples(dflab, ["_vars_others", var_effect], savepath=path)

    #### Prune to cases with decent n
    if not skip_dimredu:
        PA = _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide, savedir, scalar_or_traj=scalar_or_traj)
        
    if is_seqsup_version:
        # Then is "good" seqsup version (newer code)
        # e.g, Shape vs. seqsup

        # Prune to just cases that are same epochset -- these are the onlye ones I will analyze anyway. Makes
        # This go  much quicker.
        dflab = PA.Xlabels["trials"]
        
        if "epochset_shape" in vars_others:
            assert "epochset_dir" not in vars_others
            inds = dflab[dflab["epochset_shape"]!=("LEFTOVER",)].index.tolist()
        elif "epochset_dir" in vars_others:    
            inds = dflab[dflab["epochset_dir"]!=("LEFTOVER",)].index.tolist()
        else:
            print(vars_others)
            assert False, "for clean analyses here, I expect this..."
        PA = PA.slice_by_dim_indices_wrapper("trials", inds)
    plt.close("all")

    # Get syntax-related columns
    preprocess_pa_syntax(PA)

    # Save (print) useful summaries of the syntaxes for this day
    dflab = PA.Xlabels["trials"]
    
    if "behseq_shapes" not in dflab:
        dflab["behseq_shapes"] = "ignore" # Just so below printing is possible.
    if "behseq_locs_clust" not in dflab:
        dflab["behseq_locs_clust"] = "ignore" # Just so below printing is possible.

    savepath = f"{savedir}/syntax_counts-1.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_concrete", "behseq_shapes"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-2.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_concrete", "behseq_shapes", "behseq_locs_clust"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-3.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_concrete", "syntax_slot_0", 
                                    "syntax_slot_1", "syntax_slot_2"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-4.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "chunk_rank", "shape"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-5.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_slot_0", "syntax_concrete"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-6.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_slot_1", "syntax_concrete"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-7.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_slot_2", "syntax_concrete"], savepath=savepath)
    
    ### Print counts for shape vs. superv
    # All the epoch-related params
    savepath = f"{savedir}/syntax_counts-sh_vs_superv-epoch_related_vars.txt"
    grouping_print_n_samples(dflab, ["superv_COLOR_METHOD", "epoch_rand_exclsv", "epoch_kind", "epoch_rand", "epoch", "epoch_is_DIR", "epoch_is_AnBmCk", "superv_is_seq_sup", "supervision_stage_concise"], savepath=savepath)
    # using epochset_shape
    savepath = f"{savedir}/syntax_counts-sh_vs_superv-1.txt"
    grouping_print_n_samples(dflab, ["epochset_shape", "behseq_shapes", "behseq_locs_clust", "epoch_rand", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup"], savepath=savepath)
    savepath = f"{savedir}/syntax_counts-sh_vs_superv-2.txt"
    grouping_print_n_samples(dflab, ["epochset_dir", "behseq_shapes", "behseq_locs_clust", "epoch_rand", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup"], savepath=savepath)
    # Ignoring epochset_shape, and just going by the (shape, loc) itself [better]
    savepath = f"{savedir}/syntax_counts-sh_vs_superv-2.txt"
    grouping_print_n_samples(dflab, ["behseq_shapes", "behseq_locs_clust", "epochset_shape", "epoch_rand", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup"], savepath=savepath)

    ### Print counts for SP vs. PIG (syntax_shape)
    savepath = f"{savedir}/syntax_counts-SP_vs_PIG-1.txt"
    grouping_print_n_samples(dflab, ["epoch", "chunk_within_rank", "syntax_concrete", "chunk_rank", "shape", "gridloc", "task_kind"], savepath=savepath)
    savepath = f"{savedir}/syntax_counts-SP_vs_PIG-2.txt"
    grouping_print_n_samples(dflab, ["task_kind", "shape", "gridloc"], savepath=savepath)
    savepath = f"{savedir}/syntax_counts-SP_vs_PIG-3.txt"
    grouping_print_n_samples(dflab, ["shape", "gridloc", "task_kind", "stroke_index"], savepath=savepath)

    return PA

def params_getter_euclidian_vars_grammar(question, version_seqsup_good, HACK=False):
    """
    Wrapper to get LIST_VAR and other variables, 
    allowing for getting good seqsup analysis (if version_seqsup_good==True), 
    where focus just on the variables needed.
    """
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars

    ### Load params
    if version_seqsup_good:
        # Good testing of effect of seqsup on syntax coding.

        # Compute eucl distance within each shape
        LIST_VAR = [
            "chunk_within_rank_semantic_v2", 
            "stroke_index",
            "chunk_within_rank_semantic_v2", 
            "stroke_index",
            "syntax_role", # ------------- Using syntax_role instead of stroke_index
            "syntax_role",
            ]
        LIST_VARS_OTHERS = [
            ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"], 
            ["epochset_shape", "epoch_rand", "superv_is_seq_sup"],
            ["epochset_dir", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"],
            ["epochset_dir", "epoch_rand", "superv_is_seq_sup"],
            ["epochset_shape", "epoch_rand", "superv_is_seq_sup"], # ------------- Using syntax_role instead of stroke_index
            ["epochset_dir", "epoch_rand", "superv_is_seq_sup"],
            ]
        LIST_CONTEXT = [
            {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"], "diff":None},
            {"same":["epochset_shape", "epoch_rand", "superv_is_seq_sup"], "diff":None},
            {"same":["epochset_dir", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"], "diff":None},
            {"same":["epochset_dir", "epoch_rand", "superv_is_seq_sup"], "diff":None},
            {"same":["epochset_shape", "epoch_rand", "superv_is_seq_sup"], "diff":None}, # ------------- Using syntax_role instead of stroke_index
            {"same":["epochset_dir", "epoch_rand", "superv_is_seq_sup"], "diff":None},
            ]
        
        if HACK:
            # very hacky, just keep specific vars that are already extracted
            LIST_VAR = LIST_VAR[:2]
            LIST_VARS_OTHERS = LIST_VARS_OTHERS[:2]
            LIST_CONTEXT = LIST_CONTEXT[:2]

        LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
        # filtdict = {"stroke_index": list(range(1, 10, 1))}
        # filtdict = {"epochset_shape":[("llCV3",)]}
        LIST_FILTDICT = [None for _ in range(len(LIST_VAR))]
        use_strings_for_vars_others = False
        list_subspace_projection = ["stxsuperv"]
        is_seqsup_version = True
    else:
        # Older version
        from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
        LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question, 
                                                                                                                    context_version="new")
        use_strings_for_vars_others = True
        list_subspace_projection = ["sytx_all", "epch_sytxrol", "syntax_role"]
        is_seqsup_version = False

    return LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT, use_strings_for_vars_others, list_subspace_projection, is_seqsup_version

def euclidian_time_resolved_fast_shuffled(DFallpa, animal, SAVEDIR_ANALYSIS, question,
                                          version_seqsup_good=False):
    """
    All code for computing and saving euclidean distnaces.
    """
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar


    twind_analy = (-1, 0.6)
    tbin_dur = 0.15 # Matching params in other analyses
    tbin_slide = 0.02
    prune_min_n_trials = N_MIN_TRIALS

    list_fit_twind = [(-0.8, 0.3)]

    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _get_list_twind_by_animal
    _list_twind, _, _ = _get_list_twind_by_animal(animal, "00_stroke", "traj_to_scalar")
    assert len(_list_twind)==1, "why mutliple?"
    twind_ideal = _list_twind[0]

    # twind_scal = (-0.5, -0.05) # char_sp
    # list_twind_scal = [(-0.1, 0.3)] # syntax, previously
    if False:
        list_twind_scal = [twind_ideal, (-0.3, -0.1)]
    else:
        list_twind_scal = [twind_ideal]

    # ### Load params
    LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT, \
        use_strings_for_vars_others, list_subspace_projection, is_seqsup_version = \
            params_getter_euclidian_vars_grammar(question, version_seqsup_good)

    # if version_seqsup_good:
    #     # Compute eucl distance within each shape
    #     LIST_VAR = [
    #         "chunk_within_rank_semantic_v2", 
    #         "stroke_index"]
    #     LIST_VARS_OTHERS = [
    #         ["epochset_shape", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"],
    #         ["epochset_shape", "epoch_rand", "superv_is_seq_sup"]]
    #     LIST_CONTEXT = [
    #         {"same":["epochset_shape", "epoch_rand", "chunk_rank", "shape", "superv_is_seq_sup"], "diff":None},
    #         {"same":["epochset_shape", "epoch_rand", "superv_is_seq_sup"], "diff":None}]
    #     LIST_PRUNE_MIN_N_LEVS = [2, 2]
    #     # filtdict = {"stroke_index": list(range(1, 10, 1))}
    #     # filtdict = {"epochset_shape":[("llCV3",)]}
    #     LIST_FILTDICT = [None, None]
    #     use_strings_for_vars_others = False
    #     list_subspace_projection = ["stxsuperv"]
    #     is_seqsup_version = True
    # else:
    #     from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    #     LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question, 
    #                                                                                                                 context_version="new",
    #                                                                                                                 version_seqsup_good=version_seqsup_good)
    #     use_strings_for_vars_others = True
    #     list_subspace_projection = ["sytx_all", "epch_sytxrol", "syntax_role"]
    #     is_seqsup_version = False

    ### Automaticlaly getting the contrast of interest
    if question == "RULE_ANBMCK_STROKE":
        # Then prune out the contrast indices. THis is possible beucas ethe indices are documented here
        list_contrast_idx = params_get_contrasts_of_interest()
    elif question == "RULESW_ANY_SEQSUP_STROKE":
        # The inidices are not documneted. Therefore get all of them.
        list_contrast_idx = list(range(len(LIST_VAR)))
    else:
        print(question)
        assert False
    
    # Map from index to variables and other params.
    contrasts_dict = {}
    for idx in sorted(list_contrast_idx):
        contrasts_dict[idx] = [LIST_VAR[idx], LIST_VARS_OTHERS[idx], LIST_CONTEXT[idx], LIST_PRUNE_MIN_N_LEVS[idx], LIST_FILTDICT[idx]]

    # - for method
    from pythonlib.cluster.clustclass import Clusters
    cl = Clusters(None)

    # Save some general params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({
        "list_subspace_projection":list_subspace_projection,
        "twind_analy":twind_analy,
        "tbin_dur":tbin_dur,
        "tbin_slide":tbin_slide,
        "prune_min_n_trials":prune_min_n_trials,
        "list_fit_twind":list_fit_twind,
        "list_twind_scal":list_twind_scal,
        "LIST_VAR":LIST_VAR,
        "LIST_VARS_OTHERS":LIST_VARS_OTHERS,
        "LIST_CONTEXT":LIST_CONTEXT,
        "LIST_PRUNE_MIN_N_LEVS":LIST_PRUNE_MIN_N_LEVS,
        "LIST_FILTDICT":LIST_FILTDICT,
        "list_contrast_idx":list_contrast_idx}, path=f"{SAVEDIR_ANALYSIS}/params.txt")

    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened(contrasts_dict, path=f"{SAVEDIR_ANALYSIS}/contrasts_dict.txt", 
                            header = "contrast_idx: var, vars_others, context, prune_min_n_levs, filtdict")
    
    ### RUN
    list_dfdist = []
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection:
            for subspace_projection_fitting_twind in list_fit_twind:

                # for contrast, contrast_idx in map_contrast_to_idx.items():
                for contrast_idx in list_contrast_idx:

                    # Get variables for this contrast  
                    var_effect = LIST_VAR[contrast_idx]
                    vars_others = LIST_VARS_OTHERS[contrast_idx]
                    context_dict = LIST_CONTEXT[contrast_idx]
                    prune_min_n_levs = LIST_PRUNE_MIN_N_LEVS[contrast_idx]
                    filtdict = LIST_FILTDICT[contrast_idx]
                    vars_group = [var_effect, "_vars_others"]
                
                    SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-fit_twind={subspace_projection_fitting_twind}/contrast={contrast_idx}|{var_effect}"
                    os.makedirs(SAVEDIR, exist_ok=True)
                    print("SAVING AT ... ", SAVEDIR)

                    # Make sure there is no "diff" in context
                    if context_dict is not None:
                        assert (context_dict["diff"] is None) or (len(context_dict["diff"])==0), "need to run the step above, removing diffs, or else will fail to get diff var_others eucliian"

                    print("These params: ", var_effect, vars_others, context_dict, filtdict)
                    
                    # Preprocess
                    savedir = f"{SAVEDIR}/preprocess"
                    os.makedirs(savedir, exist_ok=True)
                    
                    PAthisRedu = preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials, prune_min_n_levs, filtdict,
                                savedir, 
                                subspace_projection, subspace_projection_fitting_twind,
                                twind_analy, tbin_dur, tbin_slide, use_strings_for_vars_others=use_strings_for_vars_others,
                                is_seqsup_version=is_seqsup_version)
                    if PAthisRedu is None:
                        # Try again, using lower min n trials.
                        PAthisRedu = preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials-1, prune_min_n_levs, filtdict,
                                    savedir, 
                                    subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide, use_strings_for_vars_others=use_strings_for_vars_others,
                                    is_seqsup_version=is_seqsup_version)
                        if PAthisRedu is None:
                            # Skip this contrast.
                            fig, _ = plt.subplots()
                            fig.savefig(f"{savedir}/lost_all_dat_in_preprocess.pdf")
                            plt.close("all")
                            continue 

                    ###################################### Running euclidian
                    for twind_scal in list_twind_scal:

                        # Prune to scalar window
                        pathis = PAthisRedu.slice_by_dim_values_wrapper("times", twind_scal)

                        # 
                        rsa_savedir = f"{SAVEDIR}/rsa-twind_scal={twind_scal}"
                        os.makedirs(rsa_savedir, exist_ok=True)

                        # Run
                        dfdist, _ = timevarying_compute_fast_to_scalar(pathis, label_vars=vars_group, rsa_heatmap_savedir=rsa_savedir,
                                                                        prune_levs_min_n_trials=prune_min_n_trials, 
                                                                        context_dict=context_dict)
                        if dfdist is None:
                            # Skip this contrast.
                            fig, _ = plt.subplots()
                            fig.savefig(f"{savedir}/lost_all_dat_in_preprocess.pdf")
                            plt.close("all")
                            continue 
                    
                        # Make sure all pairs are gotten (including both ways). This so later can take mean for each row
                        dfdist = cl.rsa_distmat_convert_from_triangular_to_full(dfdist, label_vars=vars_group, PLOT=False, repopulate_relations=True)

                        # Save
                        dfdist["var_effect"] = var_effect
                        dfdist["vars_others"] = [tuple(vars_others) for _ in range(len(dfdist))]
                        dfdist["context_dict"] = [context_dict for _ in range(len(dfdist))]
                        dfdist["prune_min_n_levs"] = [prune_min_n_levs for _ in range(len(dfdist))]
                        dfdist["filtdict"] = [filtdict for _ in range(len(dfdist))]
                        
                        # dfdist["contrast"] = contrast
                        dfdist["contrast_idx"] = contrast_idx
                        dfdist["bregion"] = bregion
                        # dfdist["prune_version"] = prune_version
                        dfdist["which_level"] = which_level
                        dfdist["event"] = event
                        dfdist["subspace_projection"] = subspace_projection
                        dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]
                        # dfdist["dim_redu_fold"] = _i_dimredu
                        dfdist["twind_scal"] = [twind_scal for _ in range(len(dfdist))]
                        
                        ### Save this df in this folder
                        dfdist.to_pickle(f"{SAVEDIR}/dfdist-twind_scal={twind_scal}.pkl")

                        # Optioanlly, collect all data
                        if False:
                            list_dfdist.append(dfdist)
                        
                        # Older code...
                        # PLOT_HEATMAPS = False
                        # dfres = euclidian_distance_compute_trajectories(PA, LIST_VAR, LIST_VARS_OTHERS, twind_analy, tbin_dur,
                        #                         tbin_slice, savedir, PLOT_TRAJS=PLOT_STATE_SPACE, PLOT_HEATMAPS=PLOT_HEATMAPS,
                        #                         nmin_trials_per_lev=nmin_trials_per_lev,
                        #                         LIST_CONTEXT=LIST_CONTEXT, LIST_FILTDICT=LIST_FILTDICT,
                        #                         LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS,
                        #                         NPCS_KEEP=NPCS_KEEP,
                        #                         dim_red_method = dim_red_method, superv_dpca_params=superv_dpca_params,
                        #                         COMPUTE_EUCLIDIAN = COMPUTE_EUCLIDIAN, PLOT_CLEAN_VERSION = PLOT_CLEAN_VERSION)



def euclidian_time_resolved_fast_shuffled_quick_rsa(DFallpa, animal, SAVEDIR_ANALYSIS):
    """
    All code for computing and saving euclidean distnaces.
    """
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar


    twind_analy = (-1, 0.6)
    tbin_dur = 0.15 # Matching params in other analyses
    tbin_slide = 0.02
    prune_min_n_trials = N_MIN_TRIALS

    list_fit_twind = [(-0.8, 0.3)]

    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _get_list_twind_by_animal
    _list_twind, _, _ = _get_list_twind_by_animal(animal, "00_stroke", "traj_to_scalar")
    assert len(_list_twind)==1, "why mutliple?"
    twind_ideal = _list_twind[0]

    # twind_scal = (-0.5, -0.05) # char_sp
    # list_twind_scal = [(-0.1, 0.3)] # syntax, previously
    if False:
        list_twind_scal = [twind_ideal, (-0.3, -0.1)]
    else:
        list_twind_scal = [twind_ideal]

    # ### Load params
    LIST_VAR = [
        "chunk_within_rank_semantic_v2", 
        # "chunk_within_rank", 
        "chunk_within_rank_fromlast", 
        "syntax_role", # ------------- Using syntax_role instead of stroke_index
        "stroke_index",
        "syntax_role", # ------------- Using syntax_role instead of stroke_index
        ]
    LIST_VARS_OTHERS = [
        ["epoch", "chunk_rank", "shape"], 
        # ["epoch", "chunk_rank", "shape"], 
        ["epoch", "chunk_rank", "shape"], 
        ["epoch", "chunk_rank", "shape"], # ------------- Using syntax_role instead of stroke_index
        ["epoch", "FEAT_num_strokes_beh"],
        ["epoch"], # ------------- Using syntax_role instead of stroke_index
        ]
    LIST_CONTEXT = [
        None,
        # None,
        None,
        None,
        None,
        None,
        ]
    
    LIST_PRUNE_MIN_N_LEVS = [2 for _ in range(len(LIST_VAR))]
    # filtdict = {"stroke_index": list(range(1, 10, 1))}
    # filtdict = {"epochset_shape":[("llCV3",)]}
    LIST_FILTDICT = [None for _ in range(len(LIST_VAR))]
    use_strings_for_vars_others = False
    # list_subspace_projection = ["sytx_all", "epch_sytxrol", "syntax_role"]
    list_subspace_projection = ["sytx_all"]
    is_seqsup_version = False

    ### Automaticlaly getting the contrast of interest
    # The inidices are not documneted. Therefore get all of them.
    list_contrast_idx = list(range(len(LIST_VAR)))
    
    # Map from index to variables and other params.
    contrasts_dict = {}
    for idx in sorted(list_contrast_idx):
        contrasts_dict[idx] = [LIST_VAR[idx], LIST_VARS_OTHERS[idx], LIST_CONTEXT[idx], LIST_PRUNE_MIN_N_LEVS[idx], LIST_FILTDICT[idx]]

    # - for method
    from pythonlib.cluster.clustclass import Clusters
    cl = Clusters(None)

    # Save some general params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({
        "list_subspace_projection":list_subspace_projection,
        "twind_analy":twind_analy,
        "tbin_dur":tbin_dur,
        "tbin_slide":tbin_slide,
        "prune_min_n_trials":prune_min_n_trials,
        "list_fit_twind":list_fit_twind,
        "list_twind_scal":list_twind_scal,
        "LIST_VAR":LIST_VAR,
        "LIST_VARS_OTHERS":LIST_VARS_OTHERS,
        "LIST_CONTEXT":LIST_CONTEXT,
        "LIST_PRUNE_MIN_N_LEVS":LIST_PRUNE_MIN_N_LEVS,
        "LIST_FILTDICT":LIST_FILTDICT,
        "list_contrast_idx":list_contrast_idx}, path=f"{SAVEDIR_ANALYSIS}/params.txt")

    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened(contrasts_dict, path=f"{SAVEDIR_ANALYSIS}/contrasts_dict.txt", 
                            header = "contrast_idx: var, vars_others, context, prune_min_n_levs, filtdict")
    
    ### RUN
    list_dfdist = []
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        which_level = row["which_level"]
        event = row["event"]
        PA = row["pa"]

        for subspace_projection in list_subspace_projection:
            for subspace_projection_fitting_twind in list_fit_twind:

                # for contrast, contrast_idx in map_contrast_to_idx.items():
                for contrast_idx in list_contrast_idx:

                    # Get variables for this contrast  
                    var_effect = LIST_VAR[contrast_idx]
                    vars_others = LIST_VARS_OTHERS[contrast_idx]
                    context_dict = LIST_CONTEXT[contrast_idx]
                    prune_min_n_levs = LIST_PRUNE_MIN_N_LEVS[contrast_idx]
                    filtdict = LIST_FILTDICT[contrast_idx]
                    vars_group = [var_effect, "_vars_others"]
                
                    SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{event}-ss={subspace_projection}-fit_twind={subspace_projection_fitting_twind}/contrast={contrast_idx}|{var_effect}"
                    os.makedirs(SAVEDIR, exist_ok=True)
                    print("SAVING AT ... ", SAVEDIR)

                    # Make sure there is no "diff" in context
                    if context_dict is not None:
                        assert (context_dict["diff"] is None) or (len(context_dict["diff"])==0), "need to run the step above, removing diffs, or else will fail to get diff var_others eucliian"

                    print("These params: ", var_effect, vars_others, context_dict, filtdict)
                    
                    # Preprocess
                    savedir = f"{SAVEDIR}/preprocess"
                    os.makedirs(savedir, exist_ok=True)
                    
                    PAthisRedu = preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials, prune_min_n_levs, filtdict,
                                savedir, 
                                subspace_projection, subspace_projection_fitting_twind,
                                twind_analy, tbin_dur, tbin_slide, use_strings_for_vars_others=use_strings_for_vars_others,
                                is_seqsup_version=is_seqsup_version)
                    if PAthisRedu is None:
                        # Try again, using lower min n trials.
                        PAthisRedu = preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials-1, prune_min_n_levs, filtdict,
                                    savedir, 
                                    subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide, use_strings_for_vars_others=use_strings_for_vars_others,
                                    is_seqsup_version=is_seqsup_version)
                        if PAthisRedu is None:
                            # Skip this contrast.
                            fig, _ = plt.subplots()
                            fig.savefig(f"{savedir}/lost_all_dat_in_preprocess.pdf")
                            plt.close("all")
                            continue 

                    ###################################### Running euclidian
                    for twind_scal in list_twind_scal:

                        # Prune to scalar window
                        pathis = PAthisRedu.slice_by_dim_values_wrapper("times", twind_scal)

                        # 
                        rsa_savedir = f"{SAVEDIR}/rsa-twind_scal={twind_scal}"
                        os.makedirs(rsa_savedir, exist_ok=True)

                        # Run
                        dfdist, _ = timevarying_compute_fast_to_scalar(pathis, label_vars=vars_group, rsa_heatmap_savedir=rsa_savedir,
                                                                        prune_levs_min_n_trials=prune_min_n_trials, 
                                                                        context_dict=context_dict)
                        if dfdist is None:
                            # Skip this contrast.
                            fig, _ = plt.subplots()
                            fig.savefig(f"{savedir}/lost_all_dat_in_preprocess.pdf")
                            plt.close("all")
                            continue 
                    
                        # # Make sure all pairs are gotten (including both ways). This so later can take mean for each row
                        # dfdist = cl.rsa_distmat_convert_from_triangular_to_full(dfdist, label_vars=vars_group, PLOT=False, repopulate_relations=True)

                        # # Save
                        # dfdist["var_effect"] = var_effect
                        # dfdist["vars_others"] = [tuple(vars_others) for _ in range(len(dfdist))]
                        # dfdist["context_dict"] = [context_dict for _ in range(len(dfdist))]
                        # dfdist["prune_min_n_levs"] = [prune_min_n_levs for _ in range(len(dfdist))]
                        # dfdist["filtdict"] = [filtdict for _ in range(len(dfdist))]
                        
                        # # dfdist["contrast"] = contrast
                        # dfdist["contrast_idx"] = contrast_idx
                        # dfdist["bregion"] = bregion
                        # # dfdist["prune_version"] = prune_version
                        # dfdist["which_level"] = which_level
                        # dfdist["event"] = event
                        # dfdist["subspace_projection"] = subspace_projection
                        # dfdist["subspace_projection_fitting_twind"] = [subspace_projection_fitting_twind for _ in range(len(dfdist))]
                        # # dfdist["dim_redu_fold"] = _i_dimredu
                        # dfdist["twind_scal"] = [twind_scal for _ in range(len(dfdist))]
                        
                        # ### Save this df in this folder
                        # dfdist.to_pickle(f"{SAVEDIR}/dfdist-twind_scal={twind_scal}.pkl")

                        # # Optioanlly, collect all data
                        # if False:
                        #     list_dfdist.append(dfdist)
                        
                        # # Older code...
                        # # PLOT_HEATMAPS = False
                        # # dfres = euclidian_distance_compute_trajectories(PA, LIST_VAR, LIST_VARS_OTHERS, twind_analy, tbin_dur,
                        # #                         tbin_slice, savedir, PLOT_TRAJS=PLOT_STATE_SPACE, PLOT_HEATMAPS=PLOT_HEATMAPS,
                        # #                         nmin_trials_per_lev=nmin_trials_per_lev,
                        # #                         LIST_CONTEXT=LIST_CONTEXT, LIST_FILTDICT=LIST_FILTDICT,
                        # #                         LIST_PRUNE_MIN_N_LEVS=LIST_PRUNE_MIN_N_LEVS,
                        # #                         NPCS_KEEP=NPCS_KEEP,
                        # #                         dim_red_method = dim_red_method, superv_dpca_params=superv_dpca_params,
                        # #                         COMPUTE_EUCLIDIAN = COMPUTE_EUCLIDIAN, PLOT_CLEAN_VERSION = PLOT_CLEAN_VERSION)



def postprocess_dfdist_collected(DFDIST):
    """
    General preprocessing - for general plots.
    """
    from pythonlib.tools.pandastools import aggregGeneral, stringify_values
    from pythonlib.tools.pandastools import grouping_get_inner_items

    ### Assign a few useful columns
    if False: # Doing during loading, is faster
        print("1... appending columns")
        DFDIST = append_col_with_grp_index(DFDIST, ["contrast_idx", "var_effect"], "contrast_effect")
        DFDIST = append_col_with_grp_index(DFDIST, ["subspace_projection", "subspace_projection_fitting_twind", "twind_scal"], "metaparams")

    ### make a general param for same var vs. other var, that applies across all contrats (var_effect - var_other pairs)
    if False: # Doing during loading, is faster
        print("2... making varsame_effect_context")
        tmp = []
        for _, row in DFDIST.iterrows():
            var_effect = row["var_effect"]
            var_same_same = f"same-{var_effect}|_vars_others"
            tmp.append(row[var_same_same])
        DFDIST["varsame_effect_context"] = tmp
        assert ~np.any(DFDIST["varsame_effect_context"].isna())

    ### Aggregate, so each label1 has one datapt for (0|1, 0|0, 1|0, 1|1)
    print("3... stringifying")
    DFDIST_STR = stringify_values(DFDIST)
    print("4... agging")
    DFDIST_AGG = aggregGeneral(DFDIST_STR, ["animal", "date", "metaparams", "bregion", "contrast_effect", "labels_1", "varsame_effect_context"], 
                values=["dist_mean", "DIST_50", "DIST_98", "dist_norm", "dist_yue_diff"], 
                nonnumercols=["var_effect", "vars_others", "which_level", "event"])

    # Sanity check -- what contrast is missing a condition?
    # - note that it is possible for "diff context" to be lacking (i.e, 0|0 and 1|0) beucase there is not extraction of 
    # conjuctions that ensure diff context before runnign extraction above. but that is ok, as I only care about diff effect.]
    if False: # Instead, run this outside, since the above takes time.
        print("5... sanity checking")
        grpdict = grouping_get_inner_items(DFDIST_AGG, "labels_1", "varsame_effect_context")
        missing_diff_context = 0
        for grp, vals in grpdict.items():
            assert "0|1" in vals
            assert "1|1" in vals

            if len(vals)<4:
                print(grp, " === ", vals)
                missing_diff_context += 1

        frac_missing = missing_diff_context/len(grpdict)
        assert frac_missing < 0.25, "why so  many label1's are lacking diff var_other? only OK reason is simply the conjuinctions dont exist, but this should be unocmmon."

    # [OPTIONAL] Finally, keep only those labels1 that have all 3 conditions
    # from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    # extract_with_levels_of_conjunction_vars_helper(DFDIST_AGG, "varsame_effect_context", ["labels_1", "varsame_effect_context", "contrast_effect", "metaparams", "bregion"])

    return DFDIST, DFDIST_AGG

def mult_plot_all(DFDIST_AGG, map_savesuffix_to_contrast_idx_pairs, SAVEDIR_MULT, question, skip_contrast_idx_pair_if_fail=False):
    """
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import params_get_contrasts_of_interest
    from pythonlib.tools.snstools import rotateLabel
    import seaborn as sns
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_print_n_samples

    # DICT_VVO_TO_LISTIDX = params_get_contrasts_of_interest(return_list_flat=False)
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question, 
                                                                                                                    context_version="new")

    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    # list_dir_suffix = list(DICT_VVO_TO_LISTIDX.keys())

    # For each pair of contrasts, plot them.
    # for dir_suffix in list_dir_suffix:
    for save_suffix, list_contrast_idx_pairs in map_savesuffix_to_contrast_idx_pairs.items():
        # Given a contrast, figure out which pairs and dates to get
        # list_contrast_idx_pairs = DICT_VVO_TO_LISTIDX[dir_suffix]
        # list_dates_get = load_preprocess_get_dates(animal, dir_suffix)[0]

        # Get all dates that exist
        list_dates_get = sorted(DFDIST_AGG["date"].unique().tolist())

        for contrast_idx_pair in list_contrast_idx_pairs:

            contrast_effect_x, contrast_effect_y = [f"{contrast_idx}|{LIST_VAR[contrast_idx]}" for contrast_idx in contrast_idx_pair]
            print("x: ", contrast_effect_x, "... y:", contrast_effect_y)
            
            savedir = f"{SAVEDIR_MULT}/{save_suffix}/contrastpair-x={contrast_effect_x}-y={contrast_effect_y}"
            os.makedirs(savedir, exist_ok=True)

            ### Get processed datasets
            dfdist_agg = DFDIST_AGG[(DFDIST_AGG["contrast_idx"].isin(contrast_idx_pair)) & (DFDIST_AGG["date"].isin(list_dates_get))].reset_index(drop=True)

            if contrast_effect_x not in dfdist_agg["contrast_effect"].unique() or contrast_effect_y not in dfdist_agg["contrast_effect"].unique():
                if skip_contrast_idx_pair_if_fail == False:
                    grouping_print_n_samples(DFDIST_AGG, ["animal", "date", "contrast_effect"], savepath="/tmp/debug.txt")
                    print("Looking for this:", save_suffix, contrast_idx_pair, list_dates_get)
                    print("These contrast_effect exist in dataset: ", dfdist_agg["contrast_effect"].value_counts())
                    print("These dates exist in dataset: ", DFDIST_AGG["date"].value_counts())
                    assert False, "missing dates?"
                else:
                    continue

            # Pull out just the 0|1 effects (i.e., the desired contrast effects)
            dfdist_agg_01 = dfdist_agg[dfdist_agg["varsame_effect_context"]=="0|1"].reset_index(drop=True)

            fig = grouping_plot_n_samples_conjunction_heatmap(dfdist_agg, "contrast_effect", "varsame_effect_context", ["metaparams", "animal", "date"])
            savefig(fig, f"{savedir}/conjunctions_dfdist_agg-datapt=label1.pdf")
            savepath = f"{savedir}/conjunctions_dfdist_agg-datapt=label1.txt"
            grouping_print_n_samples(dfdist_agg, ["metaparams", "animal", "date", "contrast_effect", "varsame_effect_context"], savepath=savepath) 

            savepath = f"{savedir}/conjunctions_dfdist_agg_01-datapt=labels_1.txt"
            grouping_print_n_samples(dfdist_agg_01, ["animal", "date", "metaparams", "bregion", "contrast_effect", "var_effect", "vars_others"], savepath=savepath)

            ### Plots
            # (1) All bar plots, one for each contrast
            y = "dist_yue_diff"
            fig = sns.catplot(data=dfdist_agg, x="bregion", y = y, hue="varsame_effect_context", kind="bar", col="contrast_effect", aspect=1.25)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-datapt=label_1.pdf")
            plt.close("all")

            # (2) Scatter plots
            for metaparams in dfdist_agg_01["metaparams"].unique():
                dfdist_agg_01_this = dfdist_agg_01[dfdist_agg_01["metaparams"] == metaparams].reset_index(drop=True)

                # Subplots = None
                _, fig = plot_45scatter_means_flexible_grouping(dfdist_agg_01_this, "contrast_effect", contrast_effect_x, contrast_effect_y, 
                                                    None, y, "bregion", True, shareaxes=True, SIZE=3.5);
                savefig(fig, f"{savedir}/scatter45-meta={metaparams}-subplots=None.pdf")

                # Subplots = dates
                _, fig = plot_45scatter_means_flexible_grouping(dfdist_agg_01_this, "contrast_effect", contrast_effect_x, contrast_effect_y, 
                                                    "date", y, "bregion", True, shareaxes=True, SIZE=3.5);
                savefig(fig, f"{savedir}/scatter45-meta={metaparams}-subplots=date.pdf")

                # subplots = 
                _, fig = plot_45scatter_means_flexible_grouping(dfdist_agg_01_this, "contrast_effect", contrast_effect_x, contrast_effect_y, 
                                                    "bregion", y, "date", True, shareaxes=True, SIZE=3.5);
                savefig(fig, f"{savedir}/scatter45-meta={metaparams}-subplots=bregion.pdf")

                plt.close("all")

            # (3) Heatmap, showing scores for each label that exists.
            from pythonlib.tools.pandastools import plot_subplots_heatmap, grouping_append_and_return_inner_items_good
            grpdict = grouping_append_and_return_inner_items_good(dfdist_agg_01, ["contrast_effect", "metaparams", "date"])
            for grp, inds in grpdict.items():
                dfdist_agg_01_this = dfdist_agg_01.iloc[inds].reset_index(drop=True)
                
                fig, axes = plot_subplots_heatmap(dfdist_agg_01_this, "labels_1", "bregion", "dist_yue_diff", None, annotate_heatmap=False);

                savefig(fig, f"{savedir}/heatmap_each_labels1-grp={grp}.pdf")
                
                plt.close("all")

            # assert False

def mult_plot_grammar_vs_seqsup(DFDIST, SAVEDIR, animal, shape_or_loc_rule="shape"):
    """
    Comparing effect of sequence (here, using stroke index as proxy) with seqsup False vs. True.
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import params_seqsup_extract_and_process
    from pythonlib.tools.pandastools import plot_subplots_heatmap, plot_45scatter_means_flexible_grouping, append_col_with_grp_index
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel

    if False:
        from pythonlib.tools.pandastools import grouping_print_n_samples
        grouping_print_n_samples(DFDIST, ["contrast_idx", "var_effect", "vars_others"])
        dfdist["var_others"] = dfdist["vars_others"]
        dfdist["levo"] = dfdist["_vars_others_2"]
        from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _plot_pairwise_btw_levels_for_seqsup
        _plot_pairwise_btw_levels_for_seqsup(dfdist, "/tmp")

    ### Go thru all dates, process each, and collect into a new concated DFDIST
    list_date = DFDIST["date"].unique()
    tmp = []
    var_effect = None
    for date in list_date:
        dfdistthis, _var_effect, _ = params_seqsup_extract_and_process(DFDIST, animal, date, shape_or_loc_rule=shape_or_loc_rule, 
                                            print_existing_variables=False,
                                            remove_first_stroke = False)
        if dfdistthis is not None:
            tmp.append(dfdistthis)
            if var_effect is None:
                var_effect = _var_effect
            else:
                assert var_effect == _var_effect

    DFDIST_THIS = pd.concat(tmp).reset_index(drop=True)

    ### Plots
    y = "dist_yue_diff"
    savedir = f"{SAVEDIR}/grammar_vs_seqsup-rule={shape_or_loc_rule}"
    os.makedirs(savedir, exist_ok=True)

    # (2) Plot 45 scatter
    _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "var_other_is_superv", False, True, 
                                        "same-stroke_index|_vars_others", y, "bregion", True, SIZE=4, shareaxes=True);
    savefig(fig, f"{savedir}/scatter45_combined.pdf")

    _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "var_other_is_superv", False, True, 
                                        "date", y, "bregion", True, SIZE=4, shareaxes=True);
    savefig(fig, f"{savedir}/scatter45_splot=date.pdf")

    _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "var_other_is_superv", False, True, 
                                        "bregion", y, "date", True, SIZE=4, shareaxes=True);
    savefig(fig, f"{savedir}/scatter45_splot=bregion.pdf")

    plt.close("all")

    # (3) Plot catplot bars
    fig = sns.catplot(data=DFDIST_THIS, x="bregion", y=y, hue="var_other_is_superv", col="date", col_wrap=4, aspect=1.5, kind="bar")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_1.pdf")
    
    plt.close("all")

    # (1) plot heatmap summary for each date
    for date in list_date:
        dfdistthis = DFDIST_THIS[DFDIST_THIS["date"] == date].reset_index(drop=True)
        if len(dfdistthis)>0:
            dfdistthis = append_col_with_grp_index(dfdistthis, ["bregion", "var_other_is_superv"], "bregion_varsothers")

            fig, axes = plot_subplots_heatmap(dfdistthis, f"{var_effect}_1", f"{var_effect}_2", y, 
                                            "bregion_varsothers", False, True, annotate_heatmap=True)

            savefig(fig, f"{savedir}/heatmaps_each_day-{date}.pdf")

            plt.close("all")

def mult_plot_grammar_vs_seqsup_new(DFDIST, SAVEDIR, contrast_version="shape_index"):
    """
    [See calling function]
    Comparing effect of sequence (here, using stroke index as proxy) with seqsup False vs. True.
    PARAMS:
    - contrast_version, string, which analysis to do
    --- "shape_index"
    --- "shape_within_chunk"
    """
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import params_seqsup_extract_and_process
    from pythonlib.tools.pandastools import plot_subplots_heatmap, plot_45scatter_means_flexible_grouping, append_col_with_grp_index
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.plottools import savefig
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import params_seqsup_extract_and_process_new

    ### Go thru all dates, process each, and collect into a new concated DFDIST
    list_date = DFDIST["date"].unique()
    tmp = []
    var_effect = None
    vars_others = None
    var_datapt = None
    for date in list_date:
        dfdist, _vars_others, _var_effect, _var_datapt = params_seqsup_extract_and_process_new(DFDIST, date, contrast_version=contrast_version)
        if dfdist is not None:
            tmp.append(dfdist)

            if var_effect is None:
                var_effect = _var_effect
            else:
                assert var_effect == _var_effect

            if vars_others is None:
                vars_others = _vars_others
            else:
                assert vars_others == _vars_others

            if var_datapt is None:
                var_datapt = _var_datapt
            else:
                assert var_datapt == _var_datapt        
    DFDIST_THIS = pd.concat(tmp).reset_index(drop=True)

    ### Plots
    y = "dist_yue_diff"
    savedir = f"{SAVEDIR}/grammar_vs_seqsupgood-rule={contrast_version}"
    os.makedirs(savedir, exist_ok=True)

    # (2) Plot 45 scatter
    _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "superv_is_seq_sup", False, True, 
                                        None, y, "bregion", True, SIZE=4, shareaxes=True);
    savefig(fig, f"{savedir}/scatter45_combined.pdf")

    _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "superv_is_seq_sup", False, True, 
                                        "date", y, "bregion", True, SIZE=4, shareaxes=True);
    savefig(fig, f"{savedir}/scatter45_splot=date.pdf")

    _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "superv_is_seq_sup", False, True, 
                                        "bregion", y, "date", True, SIZE=4, shareaxes=True);
    savefig(fig, f"{savedir}/scatter45_splot=bregion.pdf")

    if var_datapt is not None:
        _, fig = plot_45scatter_means_flexible_grouping(DFDIST_THIS, "superv_is_seq_sup", False, True, 
                                            var_datapt, y, "bregion", True, SIZE=4, shareaxes=True);
        savefig(fig, f"{savedir}/scatter45_splot=datapt.pdf")

    plt.close("all")

    # (3) Plot catplot bars
    if var_datapt is not None:
        fig = sns.catplot(data=DFDIST_THIS, x="bregion", y=y, hue="superv_is_seq_sup", row=var_datapt, col="date", aspect=1, kind="bar")
        rotateLabel(fig)
        savefig(fig, f"{savedir}/catplot_1.pdf")

        plt.close("all")

    fig = sns.catplot(data=DFDIST_THIS, x="bregion", y=y, hue="superv_is_seq_sup", col="date", col_wrap=4, aspect=1, kind="bar")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot_2.pdf")

    plt.close("all")

    # (1) plot heatmap summary for each date
    for date in list_date:
        dfdistthis = DFDIST_THIS[DFDIST_THIS["date"] == date].reset_index(drop=True)
        if len(dfdistthis)>0:
            dfdistthis = append_col_with_grp_index(dfdistthis, ["bregion", "superv_is_seq_sup"], "br_sprv")
                
            # Plot, one plot for each var_datapt
            if var_datapt is not None:
                list_var_datapt = dfdistthis[var_datapt].unique().tolist()            
                for val_datapt in list_var_datapt:
                    _dfdistthis = dfdistthis[dfdistthis[var_datapt] == val_datapt].reset_index(drop=True)
                    fig, axes = plot_subplots_heatmap(_dfdistthis, f"{var_effect}_1", f"{var_effect}_2", y, 
                                                    "br_sprv", False, True, annotate_heatmap=True)
                    savefig(fig, f"{savedir}/heatmaps_each_day-{date}-{var_datapt}={val_datapt}.pdf")

            # Plot, agging across var_datapt
            fig, axes = plot_subplots_heatmap(dfdistthis, f"{var_effect}_1", f"{var_effect}_2", y, 
                                            "br_sprv", False, True, annotate_heatmap=True)
            savefig(fig, f"{savedir}/heatmaps_each_day-{date}.pdf")

            plt.close("all")

    from pythonlib.tools.pandastools import grouping_print_n_samples
    savepath = f"{savedir}/grouping-superv_is_seq_sup-vareffect12.txt"
    grouping_print_n_samples(DFDIST_THIS, ["date", "superv_is_seq_sup", f"{var_effect}_12"], savepath=savepath)

    if var_datapt is not None:
        savepath = f"{savedir}/grouping-superv_is_seq_sup-{var_datapt}-vareffect12.txt"
        grouping_print_n_samples(DFDIST_THIS, ["date", var_datapt, "superv_is_seq_sup", f"{var_effect}_12"], savepath=savepath)

# def _targeted_pca_clean_plots_and_dfdist_MULT_plot_single(DFDIST_THIS, colname_conj_same, question, SAVEDIR, order=None,
#                                                           yvar="dist_yue_diff"):
#     """
#     """

#     if len(DFDIST_THIS)>0:
#         fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y=yvar, hue_order=order,
#                     col="subspace", kind="bar", errorbar="se")
#         savefig(fig, f"{SAVEDIR}/q={question}-catplot-1.pdf")

#         if False: # not usulaly checked
#             fig = sns.catplot(data=DFDIST_THIS, x=colname_conj_same, hue="subspace", y=yvar, order=order,
#                         col="bregion", kind="bar", errorbar="se")
#             from pythonlib.tools.snstools import rotateLabel
#             rotateLabel(fig)
#             savefig(fig, f"{SAVEDIR}/q={question}-catplot-2.pdf")

#         fig = sns.catplot(data=DFDIST_THIS, x="bregion", hue=colname_conj_same, y=yvar, hue_order=order,
#                     col="subspace", kind="boxen")
#         for ax in fig.axes.flatten():
#             ax.axhline(0, color="k", alpha=0.2)
#         savefig(fig, f"{SAVEDIR}/q={question}-catplot-3.pdf")

#         plt.close("all")


def targeted_pca_clean_plots_and_dfdist_params():
    """
    Stores "questions", which are specific sets of variables for computing euclidean dsitance between each conunctive level.
    """ 

    ### Older ones, which are fine, but unnecesary, slows things down.
    # map_question_to_euclideanvars = {
    #     # - To test chunk rank: allow chunk_within_rank_semantic_v2 free (ie 0011)
    #     # - To test chunk_within_rank_semantic_v2: fix (chunk_rank", "shape", "gridloc) [0111]
    #     # - Motor control: (shape", "gridloc) [1100]
    #     # "1_rankwithin_vs_rank": ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc", "task_kind"],

    #     # Does two things: (1) n in chunk, (2) chunk_within_rank (more stringently than above)
    #     # - To test chunk_n_in_chunk: 01111
    #     # chunk_within_rank: 10111 (note, this is stronger test than above, since above does nto control chunk_n_in_chunk, so could be due to #3 only occuring for caes with 3 strokes, for exmaple)
    #     # - note: no reason to use chunk_within_rank_semantic_v2,  because once you condition on chunk_n_in_chunk, then chunk_within_rank is ideal
    #     "2_ninchunk_vs_rankwithin":["chunk_n_in_chunk", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "task_kind"],

    #     # counting from onset or offset.
    #     # 01111 or 10111
    #     "3_onset_vs_offset":["chunk_within_rank", "chunk_within_rank_fromlast", "chunk_rank", "shape", "gridloc", "task_kind"],
    #     "14_onset_vs_offset":["chunk_within_rank", "chunk_within_rank_fromlast", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],

    #     "4_shape_vs_chunk":["shape", "gridloc", "task_kind"],
    #     # "4b_shape_vs_chunk":["shape", "gridloc", "CTXT_loc_prev", "task_kind"], # This doesnt make sense, since SP are always first stroke...
    #     "4c_shape_vs_chunk":["shape", "task_kind"],

    #     # Like 1 and 2, but very strong control for motor
    #     # "5_rankwithin_vs_rank": ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],
    #     # -- Exclude, as it is identical to 11
    #     # "6_ninchunk_vs_rankwithin":["chunk_n_in_chunk", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],
        
    #     # # for triggering "subraction of confounds".
    #     # "7_ninchunk_vs_rankwithin":["chunk_n_in_chunk", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],

    #     # For "two shapes" analy (different levels of control)
    #     "8_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "behseq_locs_clust", "task_kind"],
    #     "13_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "CTXT_loc_next", "task_kind"],
    #     "9_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "task_kind"],

    #     "10_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "CTXT_loc_next", "chunk_n_in_chunk", "task_kind"],
    #     "11_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk", "task_kind"],

    #     # "12_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "CTXT_loc_next", "task_kind"], # Doesnt exists, no data.

    #     ### SH VS SEQSUP
    #     # IGNORE THIS -- it should be obsolete with 24 and 25
    #     # # - Original method, using epochset_shape
    #     # "20_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup", "behseq_locs_clust"],
    #     # "21_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup"],
    #     # "22_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup", "behseq_locs_clust"],
    #     # "23_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup"],

    #     # - New method, ignoring epochset-shape, and using the "ground-truth" based on defining each task by its behseq_shapes
    #     # epoch_rand_exclsv: (llCV3, UL, rand)
    #     # epoch_kind: (shape, dir, rand)
    #     "24_sh_vs_superv":["stroke_index", "behseq_shapes", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup", "behseq_locs_clust", "FEAT_num_strokes_beh"],
    #     "25_sh_vs_superv":["stroke_index", "behseq_shapes", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup", "FEAT_num_strokes_beh"],
    # }

    ### The essential ones -- for quicker running
    map_question_to_euclideanvars = {
        # - To test chunk rank: allow chunk_within_rank_semantic_v2 free (ie 0011)
        # - To test chunk_within_rank_semantic_v2: fix (chunk_rank", "shape", "gridloc) [0111]
        # - Motor control: (shape", "gridloc) [1100]
        # "1_rankwithin_vs_rank": ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc", "task_kind"],

        # Does two things: (1) n in chunk, (2) chunk_within_rank (more stringently than above)
        # - To test chunk_n_in_chunk: 01111
        # chunk_within_rank: 10111 (note, this is stronger test than above, since above does nto control chunk_n_in_chunk, so could be due to #3 only occuring for caes with 3 strokes, for exmaple)
        # - note: no reason to use chunk_within_rank_semantic_v2,  because once you condition on chunk_n_in_chunk, then chunk_within_rank is ideal
        # "2_ninchunk_vs_rankwithin":["chunk_n_in_chunk", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "task_kind"],

        # counting from onset or offset.
        # 01111 or 10111
        "3_onset_vs_offset":["chunk_within_rank", "chunk_within_rank_fromlast", "chunk_rank", "shape", "gridloc", "task_kind"],
        "14_onset_vs_offset":["chunk_within_rank", "chunk_within_rank_fromlast", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],

        "4_shape_vs_chunk":["shape", "gridloc", "task_kind"],
        # "4b_shape_vs_chunk":["shape", "gridloc", "CTXT_loc_prev", "task_kind"], # This doesnt make sense, since SP are always first stroke...
        "4c_shape_vs_chunk":["shape", "task_kind"],

        # Like 1 and 2, but very strong control for motor
        # "5_rankwithin_vs_rank": ["chunk_within_rank_semantic_v2", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],
        # -- Exclude, as it is identical to 11
        # "6_ninchunk_vs_rankwithin":["chunk_n_in_chunk", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],
        
        # # for triggering "subraction of confounds".
        # "7_ninchunk_vs_rankwithin":["chunk_n_in_chunk", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "task_kind"],

        # For "two shapes" analy (different levels of control)
        "8_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "behseq_locs_clust", "task_kind"],
        "13_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "CTXT_loc_next", "task_kind"],
        "9_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "task_kind"],

        "10_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "CTXT_loc_next", "chunk_n_in_chunk", "task_kind"],
        "11_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk", "task_kind"],

        # "12_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "CTXT_loc_next", "task_kind"], # Doesnt exists, no data.

        ### SH VS SEQSUP
        # IGNORE THIS -- it should be obsolete with 24 and 25
        # # - Original method, using epochset_shape
        # "20_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup", "behseq_locs_clust"],
        # "21_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup"],
        # "22_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup", "behseq_locs_clust"],
        # "23_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup"],

        # - New method, ignoring epochset-shape, and using the "ground-truth" based on defining each task by its behseq_shapes
        # epoch_rand_exclsv: (llCV3, UL, rand)
        # epoch_kind: (shape, dir, rand)
        # "24_sh_vs_superv":["stroke_index", "behseq_shapes", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup", "behseq_locs_clust", "FEAT_num_strokes_beh"],
        # "25_sh_vs_superv":["stroke_index", "behseq_shapes", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup", "FEAT_num_strokes_beh"],
    }

    # -- Decreasing levels of control (using n_in_chunk)
    # "10_twoshapes":               ["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "CTXT_loc_next", "chunk_n_in_chunk", "task_kind"],
    # "11_twoshapes":               ["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk", "task_kind"],
    # "6_ninchunk_vs_rankwithin":   ["chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk", "task_kind"],
    # "2_ninchunk_vs_rankwithin":   ["chunk_within_rank", "chunk_rank", "shape", "gridloc", "chunk_n_in_chunk", "task_kind"],
    # (NOTE: 11 and 6 are identical)

    # -- Decreasing levels of control (using syntax_concrete)
    # "8_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "behseq_locs_clust", "task_kind"],
    # "13_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "CTXT_loc_next", "task_kind"],
    # "9_twoshapes":["epoch", "chunk_within_rank", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "syntax_concrete", "task_kind"],

    params = {
        "map_question_to_euclideanvars":map_question_to_euclideanvars,
    }
    
    # Split dfdist for preprocessing
    map_question_to_varsame = {}
    for question, euclidean_label_vars in map_question_to_euclideanvars.items():
        colname_conj_same = "same-"
        for v in euclidean_label_vars:
            colname_conj_same+=f"{v}|"
        colname_conj_same = colname_conj_same[:-1] # remove the last |
        map_question_to_varsame[question] = colname_conj_same
    params["map_question_to_varsame"] = map_question_to_varsame
    
    return params

def targeted_pca_clean_plots_and_dfdist(DFallpa, animal, date, SAVEDIR_ALL, DEBUG=False,
                                        HACK_ONLY_PREPROCESS=False, run_number=None,
                                        DO_PLOT_STATE_SPACE = True, DO_EUCLIDEAN = True, DO_ORDINAL_REGRESSION = False):
    """
    Main code for doing everything, including targeted PCA, and then state sapce and euclidean dsitance plots.
    """
    from pythonlib.tools.plottools import savefig
    import matplotlib.pyplot as plt
    import pandas as pd
    from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single_one_var_mult_axes
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

    assert run_number is not None

    ### DEBUG -- quick testing
    if DEBUG:
        bregions_get = ["preSMA", "M1"]
        n_splits = 1
        do_subspaces_within_chunk = False
    else:
        # # GOOD - but slow
        # bregions_get = None
        # n_splits = 4 # make this high, since with splits you might lose certain low-n labels tuples.
        # do_subspaces_within_chunk = False
        # do_plot_rsa = False
        # DO_ORDINAL_REGRESSION = False
        # DO_PLOT_STATE_SPACE = True
        euclid_prune_min_n_trials = 2 # 2 is better, get data then can always remove them later...

        # QUICKER, for testing
        bregions_get = None
        n_splits = 8 # make this high, since with splits you might lose certain low-n labels tuples.
        do_subspaces_within_chunk = False
        do_plot_rsa = False
        # DO_PLOT_STATE_SPACE = False
        # DO_EUCLIDEAN = False
        # DO_ORDINAL_REGRESSION = True

    # Stratified splits params
    # Constrained = test (i.e, computing euclidean dist). Make this higher so can get enough data for eucldiean.
    # THis is more important than having data for fitting subspace (since I generalyl use global subspace anyway)
    fraction_constrained_set=0.7 # Was 0.4, but decided better to have more data for computing distance, or else you lose data.
    n_constrained=2 
    list_labels_need_n=None
    min_frac_datapts_unconstrained=None
    # min_n_datapts_unconstrained=len(PAscal.Xlabels["trials"][_var_effect].unique())
    min_n_datapts_unconstrained=None
    plot_train_test_counts=True
    plot_indices=False

    # Scalar preprocessing
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _get_list_twind_by_animal
    _list_twind, _, _ = _get_list_twind_by_animal(animal, "00_stroke", "traj_to_scalar")
    twind_scal = _list_twind[0]
    npcs_keep_force = 50
    
    # tbin_dur = 0.2
    # tbin_slide = 0.1
    tbin_dur = 0.15
    tbin_slide = 0.075

    euclidean_npcs_keep = 8
    restrict_questions_based_on_subspace = None
    single_prims_exclude_from_training=True
    HAS_SUPERVISION = False
    force_dont_split_train_test = False
    DO_REGRESS_FIRST_STROKE = False
    prune_min_n_trials = 3

    # DONT MODIFIY
    force_allow_split_train_test = False

    # Regression variables (and also, variables that are candidates for subspaces)
    # Run 1 (7/31/25)
    # variables = ['epoch', 'chunk_rank', 'shape', 'gridloc', 'CTXT_loc_prev', 'CTXT_shape_prev', 'chunk_within_rank', 'stroke_index_is_first']

    # # Run 3 (8/1/25)
    # # variables = ['epoch', 'gridloc', 'DIFF_gridloc', 'shape', 'chunk_rank', 'chunk_within_rank']
    # # variables = ['epoch', 'gridloc', 'DIFF_gridloc', 'shape', 'DIFF_shape', 'chunk_rank', 'chunk_within_rank', 'chunk_within_rank_fromlast', 'chunk_n_in_chunk']
    # variables_cont = []
    # variables_cat = ['epoch', 'gridloc', 'DIFF_gridloc', 'shape', 'chunk_rank', 'chunk_within_rank', 'chunk_within_rank_fromlast', 'chunk_n_in_chunk'] # Removing diff shape, it is too correlated with chunk rank?
    # do_vars_remove = False
    # vars_remove = []
    # # variables_is_cat = [True for _ in range(len(variables))]
    # # Subspace params
    # list_var_subspace = ["chunk_within_rank", "chunk_rank", "chunk_within_rank_fromlast", "chunk_n_in_chunk", "shape", "gridloc"]

    # # Run 4 - 8/21/25 (First time doing both (i) subtract variables and (ii) include cont motor)
    # variables_cont = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj", "chunk_n_in_chunk"]
    # do_vars_remove = True
    # vars_remove = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", 
    #             "velmean_x", "velmean_y", "gridloc", "DIFF_gridloc", "stroke_index_is_first"]
    # # Subspace params
    # list_var_subspace = [
    #     ("motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"),
    #     "chunk_rank", 
    #     "rank_conj", 
    #     "chunk_n_in_chunk", 
    #     "shape", 
    #     "gridloc"]

    # # Run 5 - 8/22/25
    # # - For initial global regression and subtraction of confounding variables
    # do_remove_global_first_stroke = True
    # variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj", "chunk_n_in_chunk"]
    # vars_remove_global = ["stroke_index_is_first"]
    # # - Then for subspace identification
    # # (note: Same as above, but remove the variable that has been regressed out)
    # variables_cont = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj", "chunk_n_in_chunk"]
    # do_vars_remove = False
    # vars_remove = None
    # # Subspace params
    # list_var_subspace = [
    #     "rank_conj", 
    #     "chunk_n_in_chunk", 
    #     "shape",
    #     # "chunk_rank", 
    #     ("gridloc", "DIFF_gridloc", "motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y")]
    
    # # Run 6 - 8/23/25
    # # - For initial global regression and subtraction of confounding variables
    # do_remove_global_first_stroke = True
    # variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
    # vars_remove_global = ["stroke_index_is_first"]
    # # - Then for subspace identification
    # # (note: Same as above, but remove the variable that has been regressed out)
    # variables_cont = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]
    # do_vars_remove = False
    # vars_remove = None
    # # Subspace params
    # list_var_subspace = [
    #     "rank_conj", 
    #     ("gridloc", "DIFF_gridloc", "motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y")]

    # # Run 7 - 8/23/25
    # # - For initial global regression and subtraction of confounding variables
    # do_remove_global_first_stroke = True
    # variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
    # vars_remove_global = ["stroke_index_is_first"]
    # # - Then for subspace identification
    # # (note: Same as above, but remove the variable that has been regressed out)
    # variables_cont = []
    # variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]
    # do_vars_remove = False
    # vars_remove = None
    # # Subspace params
    # list_var_subspace = [
    #     tuple(variables_cat),
    #     ]

    # # Run 8 - 
    # # - For initial global regression and subtraction of confounding variables
    # do_remove_global_first_stroke = True
    # variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
    # variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
    # vars_remove_global = ["stroke_index_is_first"]
    # # - Then for subspace identification
    # variables_cont = []
    # variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]
    # do_vars_remove = False
    # vars_remove = []
    # # Subspace params
    # list_var_subspace = ["rank_conj"]

    if run_number == 9:
        # Run 9 - 8/24/25
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
        variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
        variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "rank_conj", 
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            ]

    elif run_number == 10:
        # Run 10 - 8/24/25
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
        variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            ]

    elif run_number in [11, 12]:
        # Run 11 and 12 - 8/24/25
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
    elif run_number==13:
        # Run 13 - 8/26/25 -- SHAPE VS. SUPERV
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
        variables_cat_global = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj", "superv_is_seq_sup"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "chunk_rank", "shape", "rank_conj", "superv_is_seq_sup"]), # global
            ]
        HAS_SUPERVISION = True
        restrict_questions_based_on_subspace = {
            tuple(["epoch", "chunk_rank", "shape", "rank_conj", "superv_is_seq_sup"]):[
                "20_sh_vs_superv", "21_sh_vs_superv", "22_sh_vs_superv", "23_sh_vs_superv", "24_sh_vs_superv", "25_sh_vs_superv"
                ]}
    elif run_number==14:
        # Run 14 - 8/27/25
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
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 
    
    elif run_number==15:
        # Run 15 - 8/27/25 -- SP vs. PIG (using params based on the char-sp analysis in action symbols paper)
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["shape", "task_kind"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            tuple(["shape", "task_kind"]),
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
            tuple(["shape", "task_kind"]):["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, -0.05]
        tbin_dur = 0.15
        tbin_slide = 0.05

    elif run_number==16:
        # Run 16 - 8/27/25
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
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

        # - Update the time window to match the action sybmols stuff
        _list_twind, _, _ = _get_list_twind_by_animal(animal, "00_stroke", "traj_to_scalar")
        twind_scal = [-0.35, _list_twind[0][1]]
        tbin_dur = 0.15
        tbin_slide = 0.05
    
    elif run_number==17:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["gridloc", "stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first", "task_kind"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj", "task_kind"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 
    # elif run_number==asfasfsafsdf:
    #     # Run 15 - 8/27/25 -- SP vs. PIG (using params based on the char-sp analysis in action symbols paper)
    #     # - For initial global regression and subtraction of confounding variables
    #     do_remove_global_first_stroke = True
    #     variables_cont_global = []
    #     variables_cat_global = ["stroke_index_is_first", "shape", "task_kind"]
    #     vars_remove_global = ["stroke_index_is_first"]
    #     # - Then for subspace identification
    #     # (note: Same as above, but remove the variable that has been regressed out)
    #     variables_cont = []
    #     variables_cat = ["shape", "task_kind"]
    #     do_vars_remove = False
    #     vars_remove = None
    #     # Subspace params
    #     list_var_subspace = [
    #         "shape", # Only run this for the question related to SP vs. grammar.
    #         tuple(["shape", "task_kind"]),
    #         ]
    #     restrict_questions_based_on_subspace = {
    #         "shape":["4_shape_vs_chunk", "11_twoshapes"],
    #         tuple(["shape", "task_kind"]):["4_shape_vs_chunk", "11_twoshapes"],
    #     }
    #     single_prims_exclude_from_training=False
    #     force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

    #     # - Update the time window to match the action sybmols stuff
    #     twind_scal = [-0.35, -0.05]
    #     tbin_dur = 0.15
    #     tbin_slide = 0.05

    elif run_number==18:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj", "task_kind"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            tuple(["shape", "task_kind"]),
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
            tuple(["shape", "task_kind"]):["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, -0.05]
        tbin_dur = 0.15
        tbin_slide = 0.05
        
        # Using char-sp params.
        DO_REGRESS_FIRST_STROKE = True

    elif run_number==19:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["shape", "task_kind"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            # tuple(["shape", "task_kind"]),
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
            # tuple(["shape", "task_kind"]):["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05

    elif run_number == 20:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            # "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        
        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 
    elif run_number == 21:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            # "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

    elif run_number == 22:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            ]
        restrict_questions_based_on_subspace = {
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]):["9_twoshapes", "11_twoshapes"],
        }
        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

    elif run_number == 23:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            # "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]):["11_twoshapes"],
        }
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

    elif run_number == 24:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            # "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]):["11_twoshapes"],
        }
        force_dont_split_train_test = False # If single_prims_exclude_from_training is False, then you shouldn't do splits 

    elif run_number==25:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["epoch_rand_exclsv", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        # variables_cat = ["epoch_rand_exclsv", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "shape"]
        variables_cat = ["epoch_rand_exclsv", "shape"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape",
            tuple(["epoch_rand_exclsv", "shape"]), # global
            ]
        HAS_SUPERVISION = True
        restrict_questions_based_on_subspace = {
            "shape":["20_sh_vs_superv", "21_sh_vs_superv", "22_sh_vs_superv", "23_sh_vs_superv", "24_sh_vs_superv", "25_sh_vs_superv"],
            tuple(["epoch_rand_exclsv", "shape"]):["20_sh_vs_superv", "21_sh_vs_superv", "22_sh_vs_superv", "23_sh_vs_superv", "24_sh_vs_superv", "25_sh_vs_superv"]
            }
        
        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05
        
    elif run_number==26:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
        variables_cat_global = ["epoch_rand_exclsv", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["epoch_rand_exclsv", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "shape", "rank_conj"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape",
            tuple(["epoch_rand_exclsv", "gridloc", "DIFF_gridloc", "shape", "rank_conj"]), # global
            ]
        HAS_SUPERVISION = True
        restrict_questions_based_on_subspace = {
            "shape":["20_sh_vs_superv", "21_sh_vs_superv", "22_sh_vs_superv", "23_sh_vs_superv", "24_sh_vs_superv", "25_sh_vs_superv"],
            tuple(["epoch_rand_exclsv", "gridloc", "DIFF_gridloc", "shape", "rank_conj"]):["20_sh_vs_superv", "21_sh_vs_superv", "22_sh_vs_superv", "23_sh_vs_superv", "24_sh_vs_superv", "25_sh_vs_superv"]
            }
        
        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05

    elif run_number == 27:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            # "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        
        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05

    elif run_number == 28:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["shape", "task_kind"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            # tuple(["shape", "task_kind"]),
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
            # tuple(["shape", "task_kind"]):["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
        }
        single_prims_exclude_from_training=False
        force_dont_split_train_test = True # If single_prims_exclude_from_training is False, then you shouldn't do splits 

        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05

    elif run_number == 29:
        # - For initial global regression and subtraction of confounding variables
        do_remove_global_first_stroke = True
        variables_cont_global = []
        variables_cat_global = ["stroke_index_is_first", "shape", "task_kind"]
        vars_remove_global = ["stroke_index_is_first"]
        # - Then for subspace identification
        # (note: Same as above, but remove the variable that has been regressed out)
        variables_cont = []
        variables_cat = ["shape", "task_kind"]
        do_vars_remove = False
        vars_remove = None
        # Subspace params
        list_var_subspace = [
            "shape", # Only run this for the question related to SP vs. grammar.
            # tuple(["shape", "task_kind"]),
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
            # tuple(["shape", "task_kind"]):["4_shape_vs_chunk", "4c_shape_vs_chunk", "11_twoshapes"],
        }
        single_prims_exclude_from_training=False
        force_allow_split_train_test = True
        fraction_constrained_set=0.75 # make this higher, to include more data in euclidean (for SP).

        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05

    elif run_number == 30:
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
        # Subspace params
        list_var_subspace = [
            tuple(["epoch", "gridloc", "DIFF_gridloc", "chunk_rank", "shape", "rank_conj"]), # global
            # "shape", # Only run this for the question related to SP vs. grammar.
            ]
        restrict_questions_based_on_subspace = {
            "shape":["4_shape_vs_chunk"],
        }
        
        # - Update the time window to match the action sybmols stuff
        twind_scal = [-0.35, 0.2]
        tbin_dur = 0.15
        tbin_slide = 0.05

        # run 30 means this:
        DO_PLOT_STATE_SPACE = False
        DO_EUCLIDEAN = False
        DO_ORDINAL_REGRESSION = True
        euclidean_npcs_keep = 6
        n_splits = 4
    else:
        assert False

    if single_prims_exclude_from_training==False and force_allow_split_train_test==False:
        assert force_dont_split_train_test == True, "If single_prims_exclude_from_training is False, then you shouldn't do splits. The reason is lack of much SP data. Comment this out if you have enough SP data that you actually want to do this"

    if force_dont_split_train_test==True:
        n_splits = 1 # since it uses all data
        
    ### State space plots params
    # LIST_VAR_VAROTHERS = [
    #     # ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
    #     # ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
    #     # ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
    #     # ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),

    #     ## N in chunk
    #     ("chunk_n_in_chunk", ['task_kind', 'epoch', 'chunk_shape', 'chunk_within_rank']),
    #     ("chunk_n_in_chunk", ['task_kind', 'epoch', 'chunk_shape']),
    #     ("chunk_n_in_chunk", ['task_kind', 'epoch']),

    #     ## Rank within
    #     ("chunk_within_rank", ['task_kind', 'epoch', 'chunk_shape', 'chunk_within_rank_fromlast']),
    #     ("chunk_within_rank", ['task_kind', 'epoch', 'chunk_shape', 'chunk_n_in_chunk']),
    #     ("chunk_within_rank", ['task_kind', 'epoch', 'chunk_shape']),
    #     ("chunk_within_rank", ['task_kind', 'epoch']),

    #     ("chunk_within_rank_fromlast", ['task_kind', 'epoch', 'chunk_shape', 'chunk_within_rank']),
    #     ("chunk_within_rank_fromlast", ['task_kind', 'epoch', 'chunk_shape', 'chunk_n_in_chunk']),
    #     ("chunk_within_rank_fromlast", ['task_kind', 'epoch', 'chunk_shape']),
    #     ("chunk_within_rank_fromlast", ['task_kind', 'epoch']),

    #     ## Stroke index (generic)
    #     ("stroke_index", ['task_kind', 'epoch', 'syntax_concrete']),
    #     # ("stroke_index", ['epoch', 'chunk_shape']),
    #     ("stroke_index", ['task_kind', 'epoch']),

    #     ## Location
    #     ("gridloc_x", ['task_kind', 'epoch', 'chunk_shape', 'chunk_within_rank_v2']),
    #     # ("gridloc_x", ['epoch', 'syntax_concrete']),
    #     # ("gridloc_x", ['epoch', 'chunk_shape']),
    #     ("gridloc_x", ['task_kind', 'epoch']),

    #     ## Chunk rank and shape
    #     # ("chunk_shape", ['epoch', 'chunk_within_rank', 'syntax_concrete']),
    #     # ("chunk_shape", ['epoch', 'chunk_within_rank']),
    #     # ("chunk_shape", ['epoch']),

    #     # ("chunk_rank", ['epoch', 'shape', 'chunk_within_rank']),
    #     ("chunk_rank", ['task_kind', 'epoch', 'shape']),
    #     ("chunk_rank", ['task_kind', 'epoch']),

    #     # ("shape", ['epoch', 'chunk_rank', 'chunk_within_rank']),
    #     ("shape", ['epoch', 'task_kind', 'chunk_rank']),
    #     ("shape", ['epoch', 'task_kind']),

    #     ("task_kind", ["epoch", "shape"]),
    # ]

    if HAS_SUPERVISION:
        LIST_VAR_VAROTHERS = [
            # ("superv_is_seq_sup", ['task_kind', 'epoch', 'syntax_concrete', 'chunk_shape', 'chunk_within_rank']),
            # ("chunk_within_rank", ['task_kind', 'epochset_shape', 'epoch_rand_exclsv", "epoch_kind', 'superv_is_seq_sup', 'behseq_shapes', 'chunk_shape']),
            # ("stroke_index", ['task_kind', 'epoch_rand_exclsv", "epoch_kind', 'superv_is_seq_sup', 'behseq_shapes']),

            # === Supervision stuff
            # Original approach
            ("stroke_index", ['task_kind', 'epochset_shape', 'syntax_concrete', 'epoch_rand', 'superv_is_seq_sup']),
            ("stroke_index", ['task_kind', 'epochset_dir', 'syntax_concrete', 'epoch_rand', 'superv_is_seq_sup']),

            # New approach, don't use epochset
            ("stroke_index", ['task_kind', 'behseq_shapes', 'epoch_rand_exclsv', 'epoch_kind', 'superv_is_seq_sup']),
            ("chunk_within_rank", ['task_kind', 'behseq_shapes', 'epoch_rand_exclsv', 'epoch_kind', 'chunk_shape', 'superv_is_seq_sup']),

            # === Other stuff
            ## Rank within
            ("chunk_within_rank", ['task_kind', 'epochset_shape', 'chunk_shape', 'syntax_concrete']),
            ("chunk_within_rank", ['task_kind', 'epochset_shape', 'chunk_shape', 'chunk_n_in_chunk']),
            ("chunk_within_rank", ['task_kind', 'epochset_shape', 'chunk_shape']),
            ("chunk_within_rank", ['task_kind', 'epochset_shape']),

            ## Chunk rank and shape
            ("chunk_rank", ['task_kind', 'epochset_shape', 'shape']),

            ("shape", ['epochset_shape', 'task_kind']),
        ]
    else:
        # Pruned, to be quicker
        LIST_VAR_VAROTHERS = [
            ## N in chunk
            ("chunk_n_in_chunk", ['task_kind', 'epoch', 'chunk_shape', 'chunk_within_rank']),
            ("chunk_n_in_chunk", ['task_kind', 'epoch', 'chunk_shape']),
            ("chunk_n_in_chunk", ['task_kind', 'epoch']),

            ## Rank within
            ("chunk_within_rank", ['task_kind', 'epoch', 'chunk_shape', 'chunk_n_in_chunk']),
            ("chunk_within_rank", ['task_kind', 'epoch', 'chunk_shape']),
            ("chunk_within_rank", ['task_kind', 'epoch']),

            ("chunk_within_rank_fromlast", ['task_kind', 'epoch', 'chunk_shape', 'chunk_n_in_chunk']),
            ("chunk_within_rank_fromlast", ['task_kind', 'epoch', 'chunk_shape']),
            ("chunk_within_rank_fromlast", ['task_kind', 'epoch']),

            ## Location
            ("gridloc_x", ['task_kind', 'epoch']),

            ## Chunk rank and shape
            ("chunk_rank", ['task_kind', 'epoch', 'shape']),

            ("shape", ['epoch', 'task_kind']),
        ]
    LIST_DIMS = [(0,1), (2,3)]

    ### Euclidean dist params (ie one set for each "question")
    # 1: Two things: (1) rank within an dsimple within vs. chunk vs. motor (shape, gridloc)
    # TODO: possibly also exclude gridloc. Note that this leads to larger effects of grammar stuff even in M1. Is cleaner this way.
    # TODO: possibly also include DIFF_gridloc. But problem is that then there is not much data..
    map_question_to_euclideanvars = targeted_pca_clean_plots_and_dfdist_params()["map_question_to_euclideanvars"]
    # if HAS_SUPERVISION:
    #     map_question_to_euclideanvars = 
    

    #     20_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup", "behseq_locs_clust"],
    #     # "21_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup"],
    #     # "22_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup", "behseq_locs_clust"],
    #     # "23_sh_vs_superv":["stroke_index", "epochset_shape", "behseq_shapes", "epoch_rand", "superv_is_seq_sup"],

    #     # - New method, ignoring epochset-shape, and using the "ground-truth" based on defining each task by its behseq_shapes
    #     # epoch_rand_exclsv: (llCV3, UL, rand)
    #     # epoch_kind: (shape, dir, rand)
    #     "24_sh_vs_superv":["stroke_index", "behseq_shapes", "epoch_rand_exclsv", "epoch_kind", "superv_is_seq_sup", "behseq_locs_clust"],
    #     "25_sh_vs_superv

    # - First, sanity check that all regions (PAs) have same rows
    _trials = None
    for pa in DFallpa["pa"].values:
        if _trials is not None:
            assert _trials == pa.Xlabels["trials"].loc[:, ["trialcode", "stroke_index"]].values.tolist(), "cannot use the same folds_dflab across all pa."
        else:
            _trials = pa.Xlabels["trials"].loc[:, ["trialcode", "stroke_index"]].values.tolist()       

    # Use the same split folds for each bregion
    folds_dflab = None
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        PA = row["pa"]

        if (bregions_get is not None) and (bregion not in bregions_get):
            continue
        
        # SAVEDIR_ALL = "/tmp"
        SAVEDIR = f"{SAVEDIR_ALL}/bregion={bregion}"
        os.makedirs(SAVEDIR, exist_ok=True)

        filtdict = {}
        # use these regardless of what subspace, as they are good for pruning
        if False:
            # Before run 11
            prune_min_n_levs = None
            _var_effect = "chunk_within_rank_semantic_v2"
        else:
            # Run 11
            prune_min_n_levs = None
            _var_effect = "chunk_within_rank"
        _vars_others = ["epoch", "chunk_shape", "syntax_concrete", "task_kind"]
        PA = preprocess_pa(PA, _var_effect, _vars_others, prune_min_n_trials, prune_min_n_levs, filtdict,
                    SAVEDIR, 
                    None, None, None, None, None, 
                    skip_dimredu=True, prune_by_conj_var=False)

        if HACK_ONLY_PREPROCESS:
            # Just run this once
            break

        if DO_REGRESS_FIRST_STROKE:
            PA = PA.regress_neuron_task_variables_subtract_from_activity(0.1, 0.02, twind_scal, "shape")

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
            
        ### Method 1: Use entire data for fitting and projecting (with cross-validation)
        # Now split into train (fitting targeted PCA) and testing (projection).
        ### Get subsamples
        if folds_dflab is None:
            if False:
                vars_stratification = ["epoch", "chunk_within_rank_semantic_v2", "chunk_shape", "syntax_concrete", "task_kind"]
            else:
                vars_stratification = ["epoch", "chunk_within_rank", "chunk_rank", "shape", "syntax_concrete", "task_kind", "gridloc"]
            folds_dflab, fig_unc, fig_con = PAscal.split_stratified_constrained_grp_var(n_splits, vars_stratification, 
                                                            fraction_constrained_set, n_constrained, 
                                                            list_labels_need_n, min_frac_datapts_unconstrained,  
                                                            min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
            # folds_dflab --> (unc, cons) (train_inds, test_inds)

            if force_dont_split_train_test:
                folds_dflab = [(train_inds+test_inds, train_inds+test_inds) for train_inds, test_inds in folds_dflab]
                folds_dflab = [folds_dflab[0]] # just take one fold...

            savefig(fig_con, f"{SAVEDIR}/after_split_constrained_test_fold_0.pdf") # TEST
            savefig(fig_unc, f"{SAVEDIR}/after_split_unconstrained_train_fold_0.pdf") # TRIAN
            plt.close("all")
        else:
            # Then use this folds_dflab...
            pass
        

        # Save some params
        from pythonlib.tools.expttools import writeDictToYaml, writeDictToTxtFlattened
        writeDictToYaml({
            "vars_stratification":vars_stratification,
            "map_question_to_euclideanvars":map_question_to_euclideanvars,
            "euclidean_npcs_keep":euclidean_npcs_keep,
            "LIST_VAR_VAROTHERS":LIST_VAR_VAROTHERS,
            "list_var_subspace":list_var_subspace,
            "variables_cont":variables_cont,
            "variables_cat":variables_cat,
            "twind_scal":twind_scal,
        }, f"{SAVEDIR}/params.yaml")
        writeDictToTxtFlattened({
            "vars_stratification":vars_stratification,
            "map_question_to_euclideanvars":map_question_to_euclideanvars,
            "euclidean_npcs_keep":euclidean_npcs_keep,
            "LIST_VAR_VAROTHERS":LIST_VAR_VAROTHERS,
            "list_var_subspace":list_var_subspace,
            "variables_cont":variables_cont,
            "variables_cat":variables_cat,
            "twind_scal":twind_scal,
        }, f"{SAVEDIR}/params.txt")

        ### RUN!
        did_state_space = {} # only do this once out of all splits (projections).
        for var_subspace in list_var_subspace:
            did_state_space[var_subspace] = False
                
        for i_proj, (train_inds, test_inds) in enumerate(folds_dflab):
            
            train_inds = [int(i) for i in train_inds]
            test_inds = [int(i) for i in test_inds]
            if len(test_inds)<len(train_inds):
                print(len(train_inds), len(test_inds))
                assert False, "sanity chcek that I understand my code.."
            print("n_train, n_test:", len(train_inds), len(test_inds))

            if single_prims_exclude_from_training:
                # HACK: Single prims trials should always be in test inds, not train inds.
                dflab = PAscal.Xlabels["trials"]
                inds_sp = dflab[dflab["task_kind"]=="prims_single"].index.tolist()
                test_inds = test_inds + [i for i in train_inds if i in inds_sp]
                train_inds = [i for i in train_inds if i not in inds_sp]

            for var_subspace in list_var_subspace:

                savedir = f"{SAVEDIR}/FITTING_subspc={var_subspace}-iter={i_proj}"
                os.makedirs(savedir, exist_ok=True)

                if DO_PLOT_STATE_SPACE and not did_state_space[var_subspace]:
                    _LIST_VAR_VAROTHERS = LIST_VAR_VAROTHERS
                    did_state_space[var_subspace] = True
                else:
                    _LIST_VAR_VAROTHERS = None

                ### Get PA in this subspace
                pa_subspace, _, _, dfcoeff, _ = state_space_targeted_pca_scalar_single_one_var_mult_axes(
                        PAscal, twind_scal, variables_cont, variables_cat, var_subspace, npcs_keep_force, 
                        _LIST_VAR_VAROTHERS, LIST_DIMS, savedir, just_extract_paredu=False,
                        savedir_pca_subspaces=savedir,
                        inds_trials_pa_train=train_inds, inds_trials_pa_test=test_inds,
                        skip_dim_redu=True,
                        do_vars_remove=do_vars_remove, vars_remove=vars_remove)

                ### Save neural data
                import pickle
                path = f"{savedir}/pa_subspace.pkl"
                with open(path, "wb") as f:
                    pickle.dump(pa_subspace, f)

                # Also save coefficients
                pd.to_pickle(dfcoeff, f"{savedir}/dfcoeff.pkl")

                # Prune to desired n dimensions.
                _npcs_keep_euclidean = min([euclidean_npcs_keep, pa_subspace.X.shape[0]])
                pa_subspace_this = pa_subspace.slice_by_dim_indices_wrapper("chans", list(range(_npcs_keep_euclidean)))

                ################################################
                ### Compute euclidian distnace
                if DO_EUCLIDEAN:
                    list_dfdist = []
                    for question, euclidean_label_vars in map_question_to_euclideanvars.items():

                        if restrict_questions_based_on_subspace is not None:
                            if var_subspace in restrict_questions_based_on_subspace:
                                questions_allowed = restrict_questions_based_on_subspace[var_subspace]
                                if question not in questions_allowed:
                                    print(f"Skipping question ({question}) for subspace ({var_subspace}).")
                                    continue

                        ### [Optional] Things to do for each question
                        if question == "7_ninchunk_vs_rankwithin":
                            # Regress out motor covariates before computing eucldiean distance
                            variables_cont = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y"]
                            variables_cat = ["epoch", "gridloc", "DIFF_gridloc", "stroke_index_is_first", "chunk_rank", "shape", "rank_conj"]
                            vars_remove = ["motor_onsetx", "motor_onsety", "gap_from_prev_x", "gap_from_prev_y", "velmean_x", "velmean_y", "gridloc", "DIFF_gridloc", "chunk_rank", "shape"]
                            _, _, _, _, _, pa_subspace_this_this, _ = pa_subspace_this.dataextract_subspace_targeted_pca_wrapper(variables_cont, variables_cat, vars_remove,
                                                                                        None, None, 
                                                                                        PLOT_COEFF_HEATMAP=False, 
                                                                                        savedir_coeff_heatmap=None, demean=False)
                        elif question == "4_shape_vs_chunk":
                            # Prune to just those (shapes/loc) that exist in both task_kind.
                            pa_subspace_this_this, _, _ = pa_subspace_this.slice_extract_with_levels_of_conjunction_vars("task_kind", ["shape", "gridloc"], 1,
                                                        levels_var=["prims_single", "prims_on_grid"])
                        elif question == "4c_shape_vs_chunk":
                            # Prune to just those (shapes) that exist in both task_kind.
                            pa_subspace_this_this, _, _ = pa_subspace_this.slice_extract_with_levels_of_conjunction_vars("task_kind", ["shape"], 1,
                                                        levels_var=["prims_single", "prims_on_grid"])                    
                        else:
                            pa_subspace_this_this = pa_subspace_this

                        if pa_subspace_this_this is not None:
                            # Compute distance
                            _savedir = f"{savedir}/eucldist-{question}"
                            os.makedirs(_savedir, exist_ok=True)
                            if do_plot_rsa:
                                rsa_heatmap_savedir = _savedir
                            else:
                                rsa_heatmap_savedir = None
                            dfdist, _ = timevarying_compute_fast_to_scalar(pa_subspace_this_this, label_vars=euclidean_label_vars, 
                                                                    rsa_heatmap_savedir=rsa_heatmap_savedir, plot_conjunctions_savedir=_savedir,
                                                                    prune_levs_min_n_trials=euclid_prune_min_n_trials)

                            if len(dfdist)>0:
                                # save it
                                dfdist["i_proj"] = i_proj
                                dfdist["var_subspace"] = [var_subspace for _ in range(len(dfdist))]
                                dfdist["question"] = question
                                dfdist["npcs_euclidean"] = _npcs_keep_euclidean
                                list_dfdist.append(dfdist)

                    dfdist = pd.concat(list_dfdist).reset_index(drop=True)    

                    # SAVE                    
                    pd.to_pickle(dfdist, f"{savedir}/dfdist.pkl")

                ################################################
                if DO_ORDINAL_REGRESSION:
                    from neuralmonkey.scripts.analy_syntax_good_eucl_state import kernel_ordinal_logistic_regression_wrapper

                    LIST_VAR_VAROTHERS_REGR = [
                        ("chunk_within_rank_fromlast", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk"]),
                        ("chunk_within_rank", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev", "chunk_n_in_chunk"]),
                        ("chunk_within_rank_fromlast", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev"]),
                        ("chunk_within_rank", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "CTXT_loc_prev"]),
                        ("chunk_within_rank_fromlast", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "stroke_index_is_first"]),
                        ("chunk_within_rank", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "stroke_index_is_first"]),
                        ("chunk_within_rank_fromlast", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "stroke_index_is_first", "chunk_n_in_chunk"]),
                        ("chunk_within_rank", ["task_kind", "epoch", "chunk_rank", "shape", "gridloc", "stroke_index_is_first", "chunk_n_in_chunk"]),
                    ]
                    # Exclude single prims
                    pa_subspace_this_PIG = pa_subspace_this.slice_by_labels_filtdict({"task_kind":["prims_on_grid"]})
                    nsplits_ord_regr = 12

                    if pa_subspace_this_PIG.X.shape[1]>8: # n trials.
                        
                        if True:
                            from  neuralmonkey.scripts.analy_syntax_good_eucl_trial import ordinalregress_1_compute
                            dfcross, dfwithin = ordinalregress_1_compute(pa_subspace_this_PIG, LIST_VAR_VAROTHERS_REGR, savedir, 
                                                                         nsplits=nsplits_ord_regr, apply_kernel = False, 
                                                                         plot_indiv=False, plot_summary=True)     

                            for _df in [dfcross, dfwithin]:
                                _df["var_subspace"] = [var_subspace for _ in range(len(_df))]
                                _df["i_proj"] = i_proj

                            pd.to_pickle(dfcross, f"{savedir}/dfcross.pkl")
                            pd.to_pickle(dfwithin, f"{savedir}/dfwithin.pkl")
                        else:
                            for yvar, vars_grp in LIST_VAR_VAROTHERS_REGR:
                                # yvar = "chunk_within_rank_fromlast"
                                # vars_grp = ["task_kind", "epoch", "chunk_shape", "chunk_n_in_chunk"]

                                savedir_this = f"{savedir}/kernel_ordinal_regress-yvar={yvar}-vars_grp={vars_grp}"
                                os.makedirs(savedir_this, exist_ok=True)

                                try:

                                    dfcross, dfwithin = kernel_ordinal_logistic_regression_wrapper(pa_subspace_this_PIG, yvar, vars_grp, 
                                                                                                savedir_this, plot_test_data_projected=False,
                                                                                                nsplits=nsplits_ord_regr)
                                    # Save                    
                                    for _df in [dfcross, dfwithin]:
                                        _df["var_effect"] = yvar
                                        _df["vars_others"] = [tuple(vars_grp) for _ in range(len(_df))]
                                        _df["i_proj"] = i_proj
                                        _df["var_subspace"] = [var_subspace for _ in range(len(_df))]

                                    pd.to_pickle(dfcross, f"{savedir_this}/dfcross.pkl")
                                    pd.to_pickle(dfwithin, f"{savedir_this}/dfwithin.pkl")

                                    # Plot
                                    kernel_ordinal_logistic_regression_plot(dfcross, dfwithin, vars_grp, savedir_this)                            
                                except Exception as err:
                                    pass
                                    # print("Probably not enough data ... ")
                                    # print(pa_subspace_this_PIG.X.shape)
                                
                                    # from pythonlib.tools.pandastools import filter_by_min_n, _check_index_reseted
                                    # dflab = pa_subspace_this_PIG.Xlabels["trials"]
                                    # _check_index_reseted(dflab)
                                    # dflab_pruned = filter_by_min_n(dflab, yvar, n_min_per_level=3)
                                    # inds_keep = dflab_pruned["_index"].tolist()
                                    # pa_subspace_this_PIG = pa_subspace_this_PIG.slice_by_dim_indices_wrapper("trials", inds_keep)
                                    
                                    # print(1, len(dflab), len(dflab_pruned))
                                    # print(2, dflab)
                                    # print(3, dflab_pruned)
                                    # print(4, dflab[yvar].value_counts())
                                    # print(5, dflab_pruned[yvar].value_counts())
                                    # print(6, pa_subspace_this_PIG.Xlabels["trials"][yvar].value_counts())
                                    # raise err
                                            
        if do_subspaces_within_chunk:
            ### Method 2: Use one level of a vars group to fit subspace, and then project all data onto it.
            # Imagine data like this, where var_grp is A, and var_subspace is 1,2,3,..
            # A1 A1 A2 A2 A3 A3 B1 B1 B2 B2 B3 C1 C1 C2 C2 C3 C3

            # Each sample, is something like this. Here is illustrating when 
            # fitting using level-A, but this does all levels. 
            # [A1 A2 A3] (A1 A2 A3 B1 B1 B2 B2 B3 C1 C1 C2 C2 C3 C3)
            # where [] is training and () is testing.

            assert False, "this is obsolete. Is not using interatction variable chunk_rank:rank_within"
            ### Params
            # Subspace fitting
            var_subspace = "chunk_within_rank" # Inner var
            tpca_splits_vars_other = ["chunk_shape"] # Outer var

            # Store original indices
            DFLAB = PAscal.Xlabels["trials"]
            DFLAB["_index_orig_pa"] = list(range(len(DFLAB)))
            PAscal.Xlabels["trials"] = DFLAB

            list_grp_pa, list_grp_name = PAscal.split_by_label("trials", tpca_splits_vars_other)
            for pa_grp, name_grp in zip(list_grp_pa, list_grp_name):

                # Split pa to two sets, stratified by var_subspace
                folds_dflab, fig_unc, fig_con = pa_grp.split_stratified_constrained_grp_var(n_splits, [var_subspace], 
                                                                fraction_constrained_set, n_constrained, 
                                                                list_labels_need_n, min_frac_datapts_unconstrained,  
                                                                min_n_datapts_unconstrained, plot_train_test_counts, plot_indices)
                

                for i_proj, (inds_grp_train, inds_grp_test) in enumerate(folds_dflab): # Indices into pa_grp

                    # Converting to original pa inds 
                    dflab = pa_grp.Xlabels["trials"]
                    # For the grp, split into two
                    inds_grp_train_orig = dflab.iloc[inds_grp_train]["_index_orig_pa"].to_list() # To fit subspace
                    inds_grp_test_orig = dflab.iloc[inds_grp_test]["_index_orig_pa"].to_list() # To plot
                    # The remaining inds
                    inds_notgrp_orig = [i for i in range(len(DFLAB)) if i not in inds_grp_train_orig+inds_grp_test_orig]

                    savedir = f"{SAVEDIR}/FITTING_subspc={var_subspace}-conjvar={tpca_splits_vars_other}-conjlev={name_grp}-iter={i_proj}"
                    os.makedirs(savedir, exist_ok=True)

                    ### Compute and plot subspace
                    _inds_train = inds_grp_train_orig
                    _inds_test = inds_grp_test_orig + inds_notgrp_orig
                    # outer var is not allowed in variables (it never varies within this training inds)
                    _variables_cat = [v for v in variables if v not in tpca_splits_vars_other]
                    _variables_cont = []
                    pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, _ = state_space_targeted_pca_scalar_single_one_var_mult_axes(
                            PAscal, twind_scal, _variables_cont, _variables_cat, var_subspace, npcs_keep_force, 
                            LIST_VAR_VAROTHERS, LIST_DIMS, savedir, just_extract_paredu=False,
                            savedir_pca_subspaces=savedir, tbin_dur=tbin_dur, tbin_slide=tbin_slide,
                            inds_trials_pa_train=_inds_train, inds_trials_pa_test=_inds_test,
                            skip_dim_redu=True)

                    ################################################
                    ### Compute euclidian distnace
                    list_dfdist = []
                    for question, euclidean_label_vars in map_question_to_euclideanvars.items():
                            
                        ### Compute euclidian distnace
                        # MAke RSA Plot
                        _npcs_keep_euclidean = min([euclidean_npcs_keep, pa_subspace.X.shape[0]])
                        pa_subspace_this = pa_subspace.slice_by_dim_indices_wrapper("chans", list(range(_npcs_keep_euclidean)))

                        dfdist, _ = timevarying_compute_fast_to_scalar(pa_subspace_this, label_vars=euclidean_label_vars, 
                                                                rsa_heatmap_savedir=savedir, plot_conjunctions_savedir=savedir)

                        # save it
                        dfdist["i_proj"] = i_proj
                        dfdist["var_subspace"] = var_subspace
                        dfdist["var_conj"] = [tuple(tpca_splits_vars_other) for _ in range(len(dfdist))]
                        dfdist["var_conj_lev"] = [name_grp for _ in range(len(dfdist))]
                        dfdist["question"] = question
                        dfdist["npcs_euclidean"] = _npcs_keep_euclidean
                        list_dfdist.append(dfdist)

                    dfdist = pd.concat(list_dfdist).reset_index(drop=True)    

                    # SAVE
                    pd.to_pickle(dfdist, f"{savedir}/dfdist.pkl")

                    # Also save coefficients
                    pd.to_pickle(dfcoeff, f"{savedir}/dfcoeff.pkl")

def kernel_ordinal_logistic_regression_wrapper(PA, yvar, vars_grp, savedir, plot_test_data_projected = False, 
                                               nsplits = 4, do_rezero=True, PRINT=False, PLOT=False, 
                                               do_upsample=True, apply_kernel=True, list_do_grid_search=None,
                                               n_min_per_level = 3):
    """
    Perform logistc regression (e.g., predicting rank as a function of neural activity). 
    Uses the kernel trick to account for nonlinearity (e.g., curved axes)

    PARAMS:
    - yvar, the predicted variable. Must be integer.
        yvar = "chunk_within_rank_fromlast"
    - vars_grp, regression performed wtihin each class of this list of var (conjunctive).
        vars_grp = ["task_kind", "epoch", "chunk_shape", "chunk_n_in_chunk"]
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from neuralmonkey.analyses.regression_good import kernel_ordinal_logistic_regression, _kernel_ordinal_logistic_regression_example
    from pythonlib.tools.pandastools import filter_by_min_n, _check_index_reseted

    # n_min_per_level = 3
    if False:
        assert nsplits >= n_min_per_level

    # This seems most consistently the best.
    list_rescale_std = [False]
    if list_do_grid_search is None:
        do_grid_search = [True]
    else:
        do_grid_search = list_do_grid_search

    ### Preprocessing
    # Exit, if PA has only one level, or is not enough data
    if len(PA.Xlabels["trials"][yvar].unique())==1:
        return None, None
    
    ### Make sure the data are ordinal, from 0, 1, ...
    dflab = PA.Xlabels["trials"]
    Y = dflab[yvar].values
    Y = Y.astype(np.int64)
    if do_rezero:
        Y = Y - min(Y) 
    dflab[f"{yvar}-ord"] = Y
    yvar = f"{yvar}-ord"
    if not isinstance(dflab[yvar].values[0], (int, np.integer)):
        print(type(dflab[yvar].values[0]))
        assert False, "must make this int. Usef dflab[yvar].astype(int)"
    PA.Xlabels["trials"] = dflab

    # Keep only classes with at least some min n data
    dflab = PA.Xlabels["trials"]
    _check_index_reseted(dflab)
    dflab_pruned = filter_by_min_n(dflab, yvar, n_min_per_level=n_min_per_level, must_not_fail=True)
    if PRINT:
        print("len dflab, before and after pruned to min n trials and leveld: ", len(dflab), len(dflab_pruned))
    inds_keep = dflab_pruned["_index"].tolist()
    PA = PA.slice_by_dim_indices_wrapper("trials", inds_keep)

    if do_rezero:
        # Must do again, beucase filter_by_min_n may have removed the 0.
        dflab = PA.Xlabels["trials"]
        dflab[yvar] = dflab[yvar] - min(dflab[yvar])
        PA.Xlabels["trials"] = dflab

    ### 
    assert PA.X.shape[2]==1 # must not be time-varying
    dflab = PA.Xlabels["trials"]
    Xall = PA.X.squeeze().T
    Yall = dflab[yvar].values
    try:
        # assert min(Yall)==0
        assert isinstance(Yall[0], np.integer)
        assert len(set(Yall))>1
    except Exception as err:
        print(Yall)
        print(set(Yall))
        raise err
    grpdict = grouping_append_and_return_inner_items_good(dflab, vars_grp)

    from pythonlib.tools.pandastools import grouping_print_n_samples
    savepath = f"{savedir}/counts.txt"
    grouping_print_n_samples(dflab, vars_grp+[yvar], savepath=savepath)

    def clean_preprocess_data(x, y, savepath_upsample=None):
        """
        Keeping only data with at least minimun n trials. And upsampling if high trial imbalance.
        """    
        from pythonlib.tools.nptools import filter_array_to_include_minimum_n_items
        from pythonlib.tools.statstools import decode_resample_balance_dataset
        _, _inds_keep = filter_array_to_include_minimum_n_items(y, n_min_per_level)
        x = x[_inds_keep, :]
        y = y[_inds_keep]
        if len(y)==0:
            # No data
            return None, None
        elif len(set(y))==1:
            # Only one class. Can't use this.
            return None, None
        else:
            # Good.
            if do_upsample:
                # Optionally, upsample dataset
                x, y = decode_resample_balance_dataset(x, y, "upsample", savepath_upsample)
            return x, y

    def _score(model, x_test, y_test, y_train):
        """
        Helper to score these data using this model
        """
        from sklearn.metrics import balanced_accuracy_score, accuracy_score

        # Only keep test labels that fall within training distribution
        bools_test_keep = np.isin(y_test, y_train)
        if sum(bools_test_keep)==0:
            return None, None, None, None, None
        
        x_test = x_test[bools_test_keep, :]
        y_test = y_test[bools_test_keep]

        # Do prediction
        y_test_pred = model.predict(x_test)

        # It is possible for predictions to include classes not present in the y_test. If so,
        # then you can't use balanced_accuracy
        # Note that y_test_pred could hold values not in y_test
        if PRINT:
            print("here: ", np.isin(y_test_pred, y_test))
            print("y_test_pred: ", y_test_pred)
            print("y__test: ", y_test)
        if np.any(np.isin(y_test_pred, y_test)==False):
            balanced_accuracy = np.nan
            balanced_accuracy_adjusted = np.nan
        else:
            balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
            if len(np.unique(y_test)) == 1:
                # Then canot do this
                balanced_accuracy_adjusted = np.nan
            else:
                balanced_accuracy_adjusted = balanced_accuracy_score(y_test, y_test_pred, adjusted=True)
        accuracy = accuracy_score(y_test, y_test_pred)
        if PRINT:
            print("balanced_accuracy, balanced_accuracy_adjusted, accuracy --> ", balanced_accuracy, balanced_accuracy_adjusted, accuracy)
        return balanced_accuracy, balanced_accuracy_adjusted, accuracy, x_test, y_test

    ### RUN
    RES_CROSS = []
    RES_WITHIN = []
    n_skips = 0
    n_tot = 0
    for rescale_std in list_rescale_std:
        for do_grid_search in do_grid_search:
            # list_res = []
            for grp, inds in grpdict.items():

                Xthis = Xall[inds, :]
                Ythis = Yall[inds]
                # dflab_this = dflab.iloc[inds]

                if len(set(Ythis))>1:

                    ### 1. Generalization to different grps
                    x_train = Xthis
                    y_train = Ythis
                    x_train, y_train = clean_preprocess_data(x_train, y_train)
                    if x_train is None:
                        continue
                    if not np.all(np.diff(sorted(set(y_train)))==1):
                        print("[SKIPPING], since have gap in y values: ", sorted(set(y_train)))
                        n_skips+=1
                        continue
                    else:
                        n_tot+=1

                    res, fig = kernel_ordinal_logistic_regression(x_train, y_train, rescale_std=rescale_std, 
                                                                    PLOT=True, do_grid_search=do_grid_search, apply_kernel=apply_kernel)
                    savefig(fig, f"{savedir}/ord_regr_scatter-grp={grp}-rescale_std={rescale_std}-grid={do_grid_search}.pdf")
                    plt.close("all")

                    if False: # I dont use this.
                        if do_grid_search:
                            res["best_n_components"] = res["cv_best_params"]["ker__n_components"]

                    model = res["model"]
                    assert np.all(model.predict(x_train) == res["y_pred"]), "sanity check"

                    # Try predicting all other cases. 
                    for grp_test, inds_test in grpdict.items():
                        x_test = Xall[inds_test]
                        y_test = Yall[inds_test]
                        assert len(inds_test)>0
                        print("Test shape: ", x_test.shape, y_test.shape, inds_test)

                        print("Scoring ...:", len(y_test))
                        balanced_accuracy, balanced_accuracy_adjusted, accuracy, x_test, y_test = _score(model, x_test, y_test, y_train)

                        if accuracy is not None:
                            # Could be None if all test data were exlcuded as they are not part of training.
                                
                            # Plot projection of held-out data onto this axis.
                            if plot_test_data_projected:
                                from neuralmonkey.analyses.regression_good import kernel_ordinal_logistic_regression_plot
                                fig = kernel_ordinal_logistic_regression_plot(x_test, y_test, res)
                                savefig(fig, f"{savedir}/ord_regr_scatter-grp={grp}-rescale_std={rescale_std}-grid={do_grid_search}-ACROSS-test_grp={grp_test}.pdf")
                                # assert False
                                plt.close("all")

                            RES_CROSS.append({
                                "grp_train":grp,
                                "grp_test":grp_test,
                                "y_train_unique":set(y_train),
                                "y_test_unique":set(y_test),
                                "y_train":y_train,
                                "y_test":y_test,
                                "balanced_accuracy":balanced_accuracy,
                                "balanced_accuracy_adjusted":balanced_accuracy_adjusted,
                                "accuracy":accuracy,
                                "score_train":res["score"],
                                "res":res,
                                "yvar":yvar, 
                                "vars_grp":tuple(vars_grp),
                            })
                    
                    ### 2. Generaliztaion to held-out trials
                    from pythonlib.tools.statstools import split_stratified_constrained_multiple
                    fraction_constrained_set = 0.7
                    n_constrained = 2

                    # each fold (unconstrainted, constrainted), ie (test, train)
                    folds = split_stratified_constrained_multiple(Ythis, nsplits, fraction_constrained_set, n_constrained, PLOT=False)

                    for i_split, (inds_test, inds_train) in enumerate(folds):
                        print("Running split #:", i_split)
                        x_train = Xthis[inds_train, :]
                        y_train = Ythis[inds_train]

                        x_test = Xthis[inds_test, :]
                        y_test = Ythis[inds_test]

                        x_train, y_train = clean_preprocess_data(x_train, y_train)
                        if x_train is None:
                            continue

                        # Train model
                        if not np.all(np.diff(sorted(set(y_train)))==1):
                            print("[SKIPPING], since have gap in y values: ", sorted(set(y_train)))
                            continue
                        res = kernel_ordinal_logistic_regression(x_train, y_train, rescale_std=rescale_std, 
                                                                        PLOT=False, do_grid_search=do_grid_search,
                                                                        apply_kernel=apply_kernel)
                        model = res["model"]
                        
                        # Test on held-out trials.
                        balanced_accuracy, balanced_accuracy_adjusted, accuracy, x_test, y_test = _score(model, x_test, y_test, y_train)

                        if accuracy is not None:
                            # Could be None if all test data were exlcuded as they are not part of training.

                            RES_WITHIN.append({
                                "grp":grp,
                                "i_split":i_split,
                                "y_train_unique":set(y_train),
                                "y_test_unique":set(y_test),
                                "y_train":y_train,
                                "y_test":y_test,
                                "balanced_accuracy":balanced_accuracy,
                                "balanced_accuracy_adjusted":balanced_accuracy_adjusted,
                                "accuracy":accuracy,
                                "score_train":res["score"],
                                "res":res,
                                "yvar":yvar, 
                                "vars_grp":tuple(vars_grp),
                            })

                            # Plot projection of held-out data onto this axis.
                            if plot_test_data_projected:
                                from neuralmonkey.analyses.regression_good import kernel_ordinal_logistic_regression_plot
                                fig = kernel_ordinal_logistic_regression_plot(x_test, y_test, res)
                                if i_split==0:
                                    savefig(fig, f"{savedir}/ord_regr_scatter-grp={grp}-rescale_std={rescale_std}-grid={do_grid_search}-WITHIN-test_data.pdf")
                                    plt.close("all")

    from pythonlib.tools.pandastools import stringify_values
    dfcross = pd.DataFrame(RES_CROSS)
    dfwithin = pd.DataFrame(RES_WITHIN)

    if len(dfwithin)==0 and len(dfcross)==0:
        return None, None
    
    if False:
        if (n_tot > 10) and (n_skips/n_tot) > 0.4:
            print(n_skips)
            print(n_tot)
            assert False, "wjhy skiped so many?"

    if len(dfcross)==0 and len(dfwithin)==0:
        return None, None
    
    if len(dfcross)>0:
        dfcross["n_labels_train"] = [len(x) for x in dfcross["y_train_unique"]]
    dfwithin["n_labels_train"] = [len(x) for x in dfwithin["y_train_unique"]]


    ### Some postprocessing
    # To be able to compare the angles of the regression axes.
    list_coeff = []
    for _, row in dfwithin.iterrows():
        coeff = row["res"]["coeff"]
        list_coeff.append(coeff)
    dfwithin["coeff"] = list_coeff

    if False: # not actually needed
        # This is important to match tuple legnths across yvar. otherwise downstream agg will fail.
        from pythonlib.tools.pandastools import pad_tuple_values_to_same_length
        for col in ["grp", "vars_grp"]:
            pad_tuple_values_to_same_length(DFWITHIN, col, col)

    if False:
        # Older code, for plotting
        coeff_mat = np.stack(DFWITHIN["coeff"])
        labels_row = DFWITHIN["grp"].tolist()
        labels_row = [tuple(x) for x in DFWITHIN.loc[:, ["yvar", "grp"]].values.tolist()]

    # Agg across all splits (for within). Do this AFTER getting coefficients above
    from pythonlib.tools.pandastools import aggregGeneral
    def F(x):
        X = np.stack(x)
        return np.mean(X, axis=0)
    aggdict = {
        "coeff":[F],
        "balanced_accuracy": ["mean"],
        "balanced_accuracy_adjusted": ["mean"],
        "accuracy": ["mean"],
        "score_train": ["mean"],
        "n_labels_train": ["mean"]
    }
    dfwithin = aggregGeneral(dfwithin, ["grp", "yvar", "vars_grp"], list(aggdict.keys()), aggmethod=aggdict)


    if PLOT:
        kernel_ordinal_logistic_regression_wrapper_plot(dfcross, dfwithin, vars_grp, savedir)

        # # Remove cases with bad data, before stringifying
        # if len(dfcross)>0:
        #     if False:
        #         # Instead, you should use accuracy
        #         # print(dfcross["balanced_accuracy_adjusted"].value_counts())
        #         dfcross = dfcross[dfcross["balanced_accuracy_adjusted"]!=-np.inf].reset_index(drop=True)
        #         dfcross = dfcross[~dfcross["balanced_accuracy_adjusted"].isna()].reset_index(drop=True)
        #         # print(dfcross["balanced_accuracy_adjusted"].value_counts())
        #     else:
        #         dfcross_clean = dfcross[dfcross["balanced_accuracy_adjusted"]!=-np.inf].reset_index(drop=True)
        #         dfcross_clean = dfcross_clean[~dfcross_clean["balanced_accuracy_adjusted"].isna()].reset_index(drop=True)

        # ### Plots
        # import seaborn as sns
        # from pythonlib.tools.snstools import rotateLabel
        # from pythonlib.tools.pandastools import plot_subplots_heatmap
        # dfwithin_str = stringify_values(dfwithin)
        # if len(dfcross)>0:
        #     dfcross_clean_str = stringify_values(dfcross_clean)
        #     dfcross_str = stringify_values(dfcross)
                
        # if False:
        #     fig = sns.catplot(dfcross_str, x="grp_train", y="accuracy", hue="grp_test", col="n_labels_train", alpha=0.5)
        #     rotateLabel(fig)

        # for y in ["accuracy", "accuracy_adjusted"]:
        #     if len(dfcross)>0:
        #         fig, _ = plot_subplots_heatmap(dfcross_str, "grp_test", "grp_train", y, None, share_zlim=True, 
        #                             diverge=False, annotate_heatmap=True, W=11, ncols=5)
        #         savefig(fig, f"{savedir}/CROSS-heatmap-yvar={y}.pdf")

        #     # Summary catplot        
        #     fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10, jitter=True, alpha=0.5)
        #     for ax in fig.axes.flatten():
        #         ax.set_ylim([-0.1, 1.1])
        #     rotateLabel(fig)
        #     savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-1.pdf")

        #     # Summary catplot        
        #     fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10)
        #     for ax in fig.axes.flatten():
        #         ax.set_ylim([-0.1, 1.1])
        #     rotateLabel(fig)
        #     savefig(fig, f"{savedir}/WITHIN-catplot-yvar={y}.pdf")

        #     plt.close("all")

        # # These are pruned
        # for y in ["balanced_accuracy", "balanced_accuracy_adjusted"]:
        #     if len(dfcross)>0:
        #         fig, _ = plot_subplots_heatmap(dfcross_clean_str, "grp_test", "grp_train", y, None, share_zlim=True, 
        #                             diverge=False, annotate_heatmap=True, W=11, ncols=5)
        #         savefig(fig, f"{savedir}/CROSS-heatmap-yvar={y}.pdf")

        #     # Summary catplot        
        #     fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10, jitter=True, alpha=0.5)
        #     for ax in fig.axes.flatten():
        #         ax.set_ylim([-0.1, 1.1])
        #     rotateLabel(fig)
        #     savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-1.pdf")

        #     # Summary catplot        
        #     fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10)
        #     for ax in fig.axes.flatten():
        #         ax.set_ylim([-0.1, 1.1])
        #     rotateLabel(fig)
        #     savefig(fig, f"{savedir}/WITHIN-catplot-yvar={y}.pdf")

        #     plt.close("all")

    return dfcross, dfwithin

def kernel_ordinal_logistic_regression_wrapper_postprocess_mult_varsgrp(DFCROSS, DFWITHIN):
    """
    Helper to run kernel_ordinal_logistic_regression_wrapper_postprocess, but for cases where 
    If dfwithin/dfcross is concatenation of multiple (yvar, vars_grp) pairs, then this preproesses
    each separately, then returns them concatnated.
    """
    
    from pythonlib.tools.pandastools import append_col_with_grp_index
    if "regr_yvar_grp" not in DFWITHIN:
        DFWITHIN = append_col_with_grp_index(DFWITHIN, ["yvar", "vars_grp"], "regr_yvar_grp")
        DFCROSS = append_col_with_grp_index(DFCROSS, ["yvar", "vars_grp"], "regr_yvar_grp")

    assert sorted(DFWITHIN["regr_yvar_grp"].unique()) == sorted(DFCROSS["regr_yvar_grp"].unique()), "or else this will throw out data or fail"

    list_dfcross = []
    list_dfwithin = []
    for regr_yvar_grp in DFWITHIN["regr_yvar_grp"].unique():
        
        dfwithin = DFWITHIN[DFWITHIN["regr_yvar_grp"] == regr_yvar_grp].reset_index(drop=True)
        dfcross = DFCROSS[DFCROSS["regr_yvar_grp"] == regr_yvar_grp].reset_index(drop=True)

        vars_grp = dfwithin["vars_grp"].unique().tolist()[0]
        dfcross, dfwithin, _, _, _, _ = kernel_ordinal_logistic_regression_wrapper_postprocess(
            dfcross, dfwithin, vars_grp)
        
        list_dfcross.append(dfcross)
        list_dfwithin.append(dfwithin)
    DFCROSS = pd.concat(list_dfcross).reset_index(drop=True)
    DFWITHIN = pd.concat(list_dfwithin).reset_index(drop=True)

    return DFCROSS, DFWITHIN

def kernel_ordinal_logistic_regression_wrapper_postprocess(dfcross, dfwithin, vars_grp):
    """
    Postprocessing for a singel (var, varsother), many things.
    """

    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values, aggregGeneral
    import seaborn as sns
    from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific, dfdist_extract_label_vars_specific_single
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_generate_constrast_strings, dfdist_variables_effect_extract_helper

    # print(dfwithin["vars_grp"].unique().tolist())
    # print(vars_grp)
    try:
        assert [tuple(vars_grp)] == dfwithin["vars_grp"].unique().tolist(), "must be only one level of vars_grp, and that should be this."
        assert len(dfwithin["yvar"].unique())==1
    except Exception as err:
        print([vars_grp])
        print(dfwithin["vars_grp"].unique().tolist())
        raise err

    from pythonlib.tools.pandastools import append_col_with_grp_index
    if "regr_yvar_grp" not in dfwithin:
        dfwithin = append_col_with_grp_index(dfwithin, ["yvar", "vars_grp"], "regr_yvar_grp")
        dfcross = append_col_with_grp_index(dfcross, ["yvar", "vars_grp"], "regr_yvar_grp")

    ### Preprocessing
    # Get adjusted accuracy
    dfcross["accuracy_chance"] = 1/dfcross["n_labels_train"]
    dfcross["accuracy_adjusted"] = (dfcross["accuracy"] - dfcross["accuracy_chance"])/(1-dfcross["accuracy_chance"])

    dfwithin["accuracy_chance"] = 1/dfwithin["n_labels_train"]
    dfwithin["accuracy_adjusted"] = (dfwithin["accuracy"] - dfwithin["accuracy_chance"])/(1-dfwithin["accuracy_chance"])

    # agg over splits
    dfwithin = aggregGeneral(dfwithin, ["grp", "yvar", "vars_grp", "regr_yvar_grp"], 
                             values=["balanced_accuracy", "balanced_accuracy_adjusted", "accuracy", "accuracy_adjusted", "score_train", "n_labels_train"], 
                             nonnumercols="all")

    # Get the individual variable classes as new columns in dfcross
    dfcross, varsame = dfdist_extract_label_vars_specific(dfcross, vars_grp, return_var_same=True, var1="grp_train", var2="grp_test")
    dfwithin = dfdist_extract_label_vars_specific_single(dfwithin, vars_grp, var1="grp")

    # Additional varialbes of interest
    if "chunk_n_in_chunk_1" in dfcross:
        dfcross = append_col_with_grp_index(dfcross, ["shape_1", "chunk_n_in_chunk_1"], "shape_n_1")
        dfcross = append_col_with_grp_index(dfcross, ["shape_2", "chunk_n_in_chunk_2"], "shape_n_2")

    if False:
        # Instead, you should use accuracy
        # print(dfcross["balanced_accuracy_adjusted"].value_counts())
        dfcross = dfcross[dfcross["balanced_accuracy_adjusted"]!=-np.inf].reset_index(drop=True)
        dfcross = dfcross[~dfcross["balanced_accuracy_adjusted"].isna()].reset_index(drop=True)
        # print(dfcross["balanced_accuracy_adjusted"].value_counts())
    else:
        dfcross_clean = dfcross[dfcross["balanced_accuracy_adjusted"]!=-np.inf].reset_index(drop=True)
        dfcross_clean = dfcross_clean[~dfcross_clean["balanced_accuracy_adjusted"].isna()].reset_index(drop=True)

    # Stringify before plotting
    dfcross_clean_str = stringify_values(dfcross_clean)
    dfcross_str = stringify_values(dfcross)
    dfwithin_str = stringify_values(dfwithin)

    return dfcross, dfwithin, dfcross_str, dfwithin_str, dfcross_clean_str, varsame


def kernel_ordinal_logistic_regression_wrapper_plot(dfcross, dfwithin, vars_grp, savedir):
    """
    Quick plots of results from kernel_ordinal_logistic_regression_wrapper (low-level, used during each run)
    """
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import plot_subplots_heatmap, stringify_values
    import seaborn as sns
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_effect_extract_helper

    dfcross, dfwithin, dfcross_str, dfwithin_str, dfcross_clean_str, varsame = kernel_ordinal_logistic_regression_wrapper_postprocess(
        dfcross, dfwithin, vars_grp)
    
    # ### Preprocessing
    # # Get adjusted accuracy
    # dfcross["accuracy_chance"] = 1/dfcross["n_labels_train"]
    # dfcross["accuracy_adjusted"] = (dfcross["accuracy"] - dfcross["accuracy_chance"])/(1-dfcross["accuracy_chance"])

    # dfwithin["accuracy_chance"] = 1/dfwithin["n_labels_train"]
    # dfwithin["accuracy_adjusted"] = (dfwithin["accuracy"] - dfwithin["accuracy_chance"])/(1-dfwithin["accuracy_chance"])

    # # agg over splits
    # dfwithin = aggregGeneral(dfwithin, ["grp"], values=["balanced_accuracy", "balanced_accuracy_adjusted", "accuracy", "accuracy_adjusted"], nonnumercols="all")

    # # Get the individual variable classes as new columns in dfcross
    # dfcross, varsame = dfdist_extract_label_vars_specific(dfcross, vars_grp, return_var_same=True, var1="grp_train", var2="grp_test")
    # dfwithin = dfdist_extract_label_vars_specific_single(dfwithin, vars_grp, var1="grp")
    
    # # Additional varialbes of interest
    # if "chunk_n_in_chunk_1" in dfcross:
    #     dfcross = append_col_with_grp_index(dfcross, ["shape_1", "chunk_n_in_chunk_1"], "shape_n_1")
    #     dfcross = append_col_with_grp_index(dfcross, ["shape_2", "chunk_n_in_chunk_2"], "shape_n_2")

    # if False:
    #     # Instead, you should use accuracy
    #     # print(dfcross["balanced_accuracy_adjusted"].value_counts())
    #     dfcross = dfcross[dfcross["balanced_accuracy_adjusted"]!=-np.inf].reset_index(drop=True)
    #     dfcross = dfcross[~dfcross["balanced_accuracy_adjusted"].isna()].reset_index(drop=True)
    #     # print(dfcross["balanced_accuracy_adjusted"].value_counts())
    # else:
    #     dfcross_clean = dfcross[dfcross["balanced_accuracy_adjusted"]!=-np.inf].reset_index(drop=True)
    #     dfcross_clean = dfcross_clean[~dfcross_clean["balanced_accuracy_adjusted"].isna()].reset_index(drop=True)

    # # Stringify before plotting
    # dfcross_clean_str = stringify_values(dfcross_clean)
    # dfcross_str = stringify_values(dfcross)
    # dfwithin_str = stringify_values(dfwithin)

    ######################
    ### Plots

    ### Plot cross groups
    if False: # This is done above, in kernel_ordinal_logistic_regression_wrapper (and better, as it uses dfcross_clean_str. The one here will fail becuase it doesnt.)
        for y in ["accuracy", "accuracy_adjusted", "balanced_accuracy", "balanced_accuracy_adjusted"]:
            fig, _ = plot_subplots_heatmap(dfcross, "grp_test", "grp_train", y, None, share_zlim=True, 
                                diverge=False, annotate_heatmap=False, W=12, ncols=5)
            savefig(fig, f"{savedir}/CROSS-heatmap-yvar={y}.pdf")

            # Summary catplot        
            fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10, jitter=True, alpha=0.5)
            for ax in fig.axes.flatten():
                ax.set_ylim([-0.1, 1.1])
            rotateLabel(fig)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-1.pdf")

            # Summary catplot        
            fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10, kind="bar")
            for ax in fig.axes.flatten():
                ax.set_ylim([-0.1, 1.1])
            rotateLabel(fig)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-2.pdf")
        plt.close("all")
    else:
        for y in ["accuracy", "accuracy_adjusted"]:
            if len(dfcross)>0:
                fig, _ = plot_subplots_heatmap(dfcross_str, "grp_test", "grp_train", y, None, share_zlim=True, 
                                    diverge=False, annotate_heatmap=False, W=11, ncols=5)
                savefig(fig, f"{savedir}/CROSS-heatmap-yvar={y}.pdf")

            # Summary catplot        
            fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10, jitter=True, alpha=0.5)
            for ax in fig.axes.flatten():
                ax.set_ylim([-0.1, 1.1])
            rotateLabel(fig)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-1.pdf")

            # Summary catplot        
            fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10)
            for ax in fig.axes.flatten():
                ax.set_ylim([-0.1, 1.1])
            rotateLabel(fig)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-2.pdf")

            plt.close("all")

        # These are pruned
        for y in ["balanced_accuracy", "balanced_accuracy_adjusted"]:
            if len(dfcross)>0:
                fig, _ = plot_subplots_heatmap(dfcross_clean_str, "grp_test", "grp_train", y, None, share_zlim=True, 
                                    diverge=False, annotate_heatmap=False, W=11, ncols=5)
                savefig(fig, f"{savedir}/CROSS-heatmap-yvar={y}.pdf")

            # Summary catplot        
            fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10, jitter=True, alpha=0.5)
            for ax in fig.axes.flatten():
                ax.set_ylim([-0.1, 1.1])
            rotateLabel(fig)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-1.pdf")

            # Summary catplot        
            fig = sns.catplot(data=dfwithin_str, x="grp", y=y, hue="n_labels_train", height=10)
            for ax in fig.axes.flatten():
                ax.set_ylim([-0.1, 1.1])
            rotateLabel(fig)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=grp-yvar={y}-2.pdf")

            plt.close("all")

    ### Plot overviews
    if "shape_1" in dfcross.columns:
        for y, diverge, ZLIMS in [
            ("accuracy", False, [-0.1, 1.1]),
            ("accuracy_adjusted", True, None),
            ]:

            fig, _ = plot_subplots_heatmap(dfcross_clean_str, "shape_1", "shape_2", y, None, diverge, True, annotate_heatmap=True, ZLIMS=ZLIMS)
            savefig(fig, f"{savedir}/CROSS-heatmap-shape-yvar={y}-1.pdf")

            if "shape_n_1" in dfcross_clean_str.columns:
                fig, _ = plot_subplots_heatmap(dfcross_clean_str, "shape_n_1", "shape_n_2", y, None, diverge, True, annotate_heatmap=True, ZLIMS=ZLIMS, W=10)
                savefig(fig, f"{savedir}/CROSS-heatmap-shape-yvar={y}-2.pdf")
            
            fig = sns.catplot(data=dfwithin_str, x="shape", y=y, hue="n_labels_train", jitter=True, alpha=0.5)
            savefig(fig, f"{savedir}/WITHIN-catplot-x=shape-yvar={y}-1.pdf")

            fig = sns.catplot(data=dfwithin_str, x="shape", y=y, hue="n_labels_train", kind="bar")
            savefig(fig, f"{savedir}/WITHIN-catplot-x=shape-yvar={y}-2.pdf")
        plt.close("all")

        ### Get effects to plot
        list_dfeffect = []

        # Genrealize across sahpe sets?
        contrasts_diff = ["shape"]
        if "chunk_n_in_chunk_1" in dfcross_str:
            contrasts_either = ["chunk_rank", "chunk_n_in_chunk"]
        else:
            contrasts_either = ["chunk_rank"]
        df = dfdist_variables_effect_extract_helper(dfcross_str, varsame, vars_grp, contrasts_diff, contrasts_either)
        df["effect"] = "Xshape"
        list_dfeffect.append(df)

        # Genrealize across n (within shape set)?
        if "chunk_n_in_chunk_1" in dfcross_str:
            contrasts_diff = ["chunk_n_in_chunk"]
            contrasts_either = []
            df = dfdist_variables_effect_extract_helper(dfcross_str, varsame, vars_grp, contrasts_diff, contrasts_either)
            df["effect"] = "Xn_Wshape"
            list_dfeffect.append(df)

        # Control: score within condition (cross-validated)
        df = dfwithin_str.copy()
        df["effect"] = "Wall"
        df["shape_1"] = df["shape"]
        list_dfeffect.append(df)

        # from pythonlib.tools.pandastools import replace_None_with_string
        DFEFFECT = pd.concat(list_dfeffect)
        DFEFFECT = stringify_values(DFEFFECT)
        assert sum(DFEFFECT["accuracy_adjusted"]=="none")==0

        for y in ["accuracy", "accuracy_adjusted", "balanced_accuracy", "balanced_accuracy_adjusted"]:
            dfeffect = DFEFFECT[~(DFEFFECT[y] == "none")]
            # fig = sns.catplot(data=dfeffect, x="effect", y=y, hue=varsame, col="shape_1", alpha=0.5, jitter=True)
            fig = sns.catplot(data=dfeffect, x="effect", y=y, hue=varsame, col="shape_1", alpha=0.5)
            savefig(fig, f"{savedir}/EFFECT-catplot-yvar={y}-1.pdf")

            # fig = sns.catplot(data=dfeffect, x="effect", y=y, col="shape_1", alpha=0.5, jitter=True)
            fig = sns.catplot(data=dfeffect, x="effect", y=y, col="shape_1", alpha=0.5)
            savefig(fig, f"{savedir}/EFFECT-catplot-yvar={y}-2.pdf")
            
            fig = sns.catplot(data=dfeffect, x="effect", y=y, col="shape_1", kind="bar", errorbar="se")
            savefig(fig, f"{savedir}/EFFECT-catplot-yvar={y}-3.pdf")
        plt.close("all")
                
def kernel_ordinal_logistic_regression_wrapper_CONCATED_postprocess(DFCROSS, DFWITHIN, vars_datapt):
    """
    [MULT ANALY], another round of posprocessing...

    PARAMS:
    - DFCROSS, DFWITHIN, these are after concated across dates
    - vars_datapt, list of str, defines unique daapts (will additionaly conjunct with dates)
        # vars_datapt = ["epoch", "FEAT_num_strokes_beh", "syntax_slot_0", "syntax_slot_1", "seqc_0_loc", "seqc_0_shape"]
        # vars_datapt = ["epoch", "chunk_rank", "shape"]
    """
    ### Get agged data
    from pythonlib.tools.pandastools import aggregGeneral, replace_None_with_string
    from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
    from neuralmonkey.scripts.analy_syntax_good_eucl_state_MULT import targeted_pca_MULT_2_postprocess
    from pythonlib.tools.pandastools import append_col_with_grp_index
    
    DFWITHIN, _ = targeted_pca_MULT_2_postprocess(DFWITHIN)
    DFCROSS, _ = targeted_pca_MULT_2_postprocess(DFCROSS)
    assert not any(DFWITHIN["balanced_accuracy"].isna())

    DFWITHIN = append_col_with_grp_index(DFWITHIN, ["yvar", "vars_grp"], "regr_yvar_grp")
    DFCROSS = append_col_with_grp_index(DFCROSS, ["yvar", "vars_grp"], "regr_yvar_grp")

    if "y_test_unique" in DFCROSS:
        DFCROSS["n_labels_test"] = [len(x) for x in DFCROSS["y_test_unique"]]
    DFWITHIN["n_labels_test"] = DFWITHIN["n_labels_train"]

    # Finally, round up the value for "n_train" and "n_test" (cant have fractions)
    for col in ["n_labels_train", "n_labels_test"]:
        DFCROSS[col] = np.ceil(DFCROSS[col]).astype(int)
        DFWITHIN[col] = np.ceil(DFWITHIN[col]).astype(int)

        # And replace >3 with one number
        DFCROSS.loc[DFCROSS[col]>2, col] = 99
        DFWITHIN.loc[DFWITHIN[col]>2, col] = 99
        
    DFWITHIN = replace_None_with_string(DFWITHIN)
    DFCROSS = replace_None_with_string(DFCROSS)

    # Add index.
    DFWITHIN = append_col_with_grp_index(DFWITHIN, ["date", "grp"], "date_grp")
    DFCROSS = append_col_with_grp_index(DFCROSS, ["date", "grp_train"], "date_grp_train")

    # This defines a "datapoint" unit

    # Datapoints
    DFWITHIN = append_col_with_grp_index(DFWITHIN, ["date"] + vars_datapt, "ep_cr_sh")
    DFCROSS = append_col_with_grp_index(DFCROSS, ["date"] + [f"{v}_1" for v in vars_datapt], "ep_cr_sh_1")

    # One datapt per (date, shape)
    DFWITHIN_AGG_SHP = aggregGeneral(DFWITHIN, ["bregion", "subspace", "date", "yvar", "vars_grp", "regr_yvar_grp", "n_labels_train"] + vars_datapt, values=["balanced_accuracy", "balanced_accuracy_adjusted", "accuracy", "accuracy_adjusted", "score_train"])
    # One datapt per (date)
    DFWITHIN_AGG_DATE = aggregGeneral(DFWITHIN, ["bregion", "subspace", "date", "yvar", "vars_grp", "regr_yvar_grp", "n_labels_train"], values=["balanced_accuracy", "balanced_accuracy_adjusted", "accuracy", "accuracy_adjusted", "score_train"])
    # - Reorder bregions.
    DFWITHIN_AGG_SHP = datamod_reorder_by_bregion(DFWITHIN_AGG_SHP)
    DFWITHIN_AGG_DATE = datamod_reorder_by_bregion(DFWITHIN_AGG_DATE)

    return DFCROSS, DFWITHIN, DFWITHIN_AGG_SHP, DFWITHIN_AGG_DATE

def kernel_ordinal_logistic_regression_wrapper_CONCATED_plot_all(DFCROSS, DFWITHIN, DFWITHIN_AGG_SHP, DFWITHIN_AGG_DATE, savedir,
                                                                 only_essential=False):
    """
    
    [MULT] FInal set of plots. Generic enough that it should work across different expreiments.

    PARAMS:
    - DFCROSS, DFWITHIN, these are after concated across dates
    - vars_datapt, list of str, defines unique daapts (will additionaly conjunct with dates)
        # vars_datapt = ["epoch", "FEAT_num_strokes_beh", "syntax_slot_0", "syntax_slot_1", "seqc_0_loc", "seqc_0_shape"]
        # vars_datapt = ["epoch", "chunk_rank", "shape"]
    - only_essential, if False, then also plots things like with individual datapts, which can take time.
    """
    ##############################################
    ### PLOTS
    from pythonlib.tools.pandastools import aggregGeneral, replace_None_with_string, stringify_values
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good, plot_45scatter_means_flexible_grouping

    ### Questions
    from pythonlib.tools.plottools import savefig
    import seaborn as sns
    assert len(DFWITHIN["subspace"].unique())==1, "assumes this, otherwise will have to split by subspace in plots below"

    # (1) Main effect
    for yvar in ["accuracy_adjusted"]:
        if not only_essential:
            fig = sns.catplot(data=DFWITHIN, x="bregion", y=yvar, hue="n_labels_train", row="regr_yvar_grp", col="date", jitter=True, alpha=0.5, height=10)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/1_maineffect-yvar={yvar}-1.pdf")

        fig = sns.catplot(data=DFWITHIN, x="bregion", y=yvar, hue="n_labels_train", row="regr_yvar_grp", col="date", kind="point", errorbar="se", height=10)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        savefig(fig, f"{savedir}/1_maineffect-yvar={yvar}-2.pdf")

        # Agg across dates
        for df, suff in [
            (DFWITHIN_AGG_SHP, "data=shp"), 
            (DFWITHIN_AGG_DATE, "data=date")]:

            if not only_essential:
                fig = sns.catplot(data=df, x="bregion", y=yvar, row="n_labels_train", col="regr_yvar_grp", jitter=True, alpha=0.5, height=10)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/1_maineffect-yvar={yvar}-{suff}-1.pdf")

            fig = sns.catplot(data=df, x="bregion", y=yvar, col="n_labels_train", hue="regr_yvar_grp", kind="bar", errorbar="se",  height=10)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/1_maineffect-yvar={yvar}-{suff}-2.pdf")

        plt.close("all")    

    # (1b) Main effect, split by shapes
    if "shape" in DFWITHIN:
        list_date = DFWITHIN["date"].unique().tolist()
        for date in list_date:
            dfwithin = DFWITHIN[DFWITHIN["date"] == date].reset_index(drop=True)
            for yvar in ["accuracy_adjusted"]:
                if not only_essential:
                    fig = sns.catplot(data=dfwithin, x="shape", y=yvar, hue="regr_yvar_grp", row="n_labels_train", col="bregion", 
                                        jitter=True, alpha=0.5, errorbar="se", height=10)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/1b_maineffect_shape-date={date}-yvar={yvar}-1.pdf")

                fig = sns.catplot(data=dfwithin, x="shape", y=yvar, hue="regr_yvar_grp", row="n_labels_train", col="bregion", 
                                kind="bar", errorbar="se", height=10)
                savefig(fig, f"{savedir}/1b_maineffect_shape-date={date}-yvar={yvar}-2.pdf")

                if False: # save time
                    fig = sns.catplot(data=dfwithin, x="bregion", y=yvar, hue="shape", row="regr_yvar_grp", col="n_labels_train", 
                                    kind="bar", errorbar="se", height=10)
                    savefig(fig, f"{savedir}/1b_maineffect_shape-date={date}-yvar={yvar}-3.pdf")
                
                plt.close("all")

    if "chunk_within_rank_fromlast" in DFWITHIN["yvar"].unique().tolist():
        # (2) Compare fwd vs. backward (for each shape).
        # This is done in above (1b).
        # Also: scatter
        prune_to_only_cases_with_mult_n_in_chunk = True
        grpdict = grouping_append_and_return_inner_items_good(DFWITHIN, ["vars_grp", "n_labels_train"])
        for yvar in ["accuracy_adjusted"]:
        # for yvar in ["accuracy_adjusted", "balanced_accuracy_adjusted"]:
            for grp, inds in grpdict.items():
                print(" ==== ", grp)
                dfwithin = DFWITHIN.iloc[inds].reset_index(drop=True)

                # This only makes sense if n_in_chunk varies. Otherwise all points will be along the diagonal
                if prune_to_only_cases_with_mult_n_in_chunk:
                    dfwithin = dfwithin[dfwithin["n_in_chunk_existing_n"]>1].reset_index(drop=True)

                if len(dfwithin)>0:
                    for var_datapt in ["date_grp", "ep_cr_sh"]:
                        # Color each pt by chunk_rank
                        map_datapt_lev_to_colorlev = {}
                        for _, row in dfwithin.iterrows():
                            if row[var_datapt] not in map_datapt_lev_to_colorlev:
                                map_datapt_lev_to_colorlev[row[var_datapt]] = row["chunk_rank"]
                            else:
                                assert map_datapt_lev_to_colorlev[row[var_datapt]] == row["chunk_rank"]
                        colorlevs_that_exist = sorted(dfwithin["chunk_rank"].unique())
                        _, fig = plot_45scatter_means_flexible_grouping(dfwithin, "yvar", "chunk_within_rank_fromlast", "chunk_within_rank", 
                                                            "bregion", yvar, var_datapt, False, shareaxes=True, 
                                                            map_datapt_lev_to_colorlev=map_datapt_lev_to_colorlev,
                                                            colorlevs_that_exist=colorlevs_that_exist)

                        if fig is not None:
                            savefig(fig, f"{savedir}/1b_scatter_up_vs_dn-{grp}-datapt={var_datapt}-y={yvar}-variable_n={prune_to_only_cases_with_mult_n_in_chunk}.pdf")
                            plt.close("all")

    # (3) Generalization across shapes
    # (3b) Generalization to other "n in chunk"
    from neuralmonkey.analyses.euclidian_distance import dfdist_variables_effect_extract_helper, dfdist_variables_generate_var_same

    list_yvar_grp = DFCROSS["regr_yvar_grp"].unique().tolist()
    list_n_labels_train = DFCROSS["n_labels_train"].unique().tolist()
    # list_n_labels_train = [99]
    # list_yvar_grp = ["chunk_within_rank_fromlast|('task_kind', 'epoch', 'chunk_rank', 'shape', 'gridloc', 'stroke_index_is_first', 'chunk_n_in_chunk')"]    
    for n_labels_train in list_n_labels_train:
        for regr_yvar_grp in list_yvar_grp:
            dfcross = DFCROSS[(DFCROSS["n_labels_train"]==n_labels_train) & (DFCROSS["regr_yvar_grp"] == regr_yvar_grp)].reset_index(drop=True)
            dfwithin = DFWITHIN[(DFWITHIN["n_labels_train"]==n_labels_train) & (DFWITHIN["regr_yvar_grp"] == regr_yvar_grp)].reset_index(drop=True)

            if len(dfcross)>0:
                vars_grp = dfcross["vars_grp"].unique()[0]
                varsame = dfdist_variables_generate_var_same(vars_grp)

                ### Collect effects
                list_dfeffect = []

                if "shape_1" in dfcross:
                    ### THen this is strokes data 

                    # Genrealize across sahpe sets?
                    contrasts_diff = ["shape"]
                    if "chunk_n_in_chunk" in vars_grp:
                        contrasts_either = ["chunk_rank", "chunk_n_in_chunk"]
                    else:
                        contrasts_either = ["chunk_rank"]
                    df = dfdist_variables_effect_extract_helper(dfcross, varsame, vars_grp, contrasts_diff, contrasts_either)
                    df["effect"] = "Xshape"
                    list_dfeffect.append(df)

                    # Genrealize across sahpe sets?
                    contrasts_diff = ["shape"]
                    contrasts_either = ["chunk_rank", "chunk_n_in_chunk", "gridloc", "CTXT_loc_prev"]
                    df = dfdist_variables_effect_extract_helper(dfcross, varsame, vars_grp, contrasts_diff, contrasts_either)
                    df["effect"] = "Xshape_lenient"
                    list_dfeffect.append(df)

                    # Genrealize across n (within shape set)?
                    if "chunk_n_in_chunk" in vars_grp:
                        contrasts_diff = ["chunk_n_in_chunk"]
                        contrasts_either = []
                        df = dfdist_variables_effect_extract_helper(dfcross, varsame, vars_grp, contrasts_diff, contrasts_either)
                        df["effect"] = "Xn_Wshape"
                        list_dfeffect.append(df)

                        contrasts_diff = ["chunk_n_in_chunk"]
                        contrasts_either = ["gridloc", "CTXT_loc_prev"]
                        df = dfdist_variables_effect_extract_helper(dfcross, varsame, vars_grp, contrasts_diff, contrasts_either)
                        df["effect"] = "Xn_Wshape_lenient"
                        list_dfeffect.append(df)

                    # Control: score within condition (cross-validated)
                    df = dfwithin.copy()
                    df["effect"] = "Wall"
                    df["shape_1"] = df["shape"]
                    df["date_grp_train"] = df["date_grp"]
                    df["ep_cr_sh_1"] = df["ep_cr_sh"]
                    df["n_labels_test"] = df["n_labels_train"]
                elif "seqc_0_shape_1" in dfcross:
                    ### THen this is trials data 

                    if "syntax_slot_1" in vars_grp:
                        contrasts_diff = ["syntax_slot_1"]
                        contrasts_either = ["FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape"]
                        df = dfdist_variables_effect_extract_helper(dfcross, varsame, vars_grp, contrasts_diff, contrasts_either)
                        df["effect"] = "Xslot1"
                        list_dfeffect.append(df)

                    if "syntax_slot_0" in vars_grp:
                        contrasts_diff = ["syntax_slot_0"]
                        contrasts_either = ["FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape"]
                        df = dfdist_variables_effect_extract_helper(dfcross, varsame, vars_grp, contrasts_diff, contrasts_either)
                        df["effect"] = "Xslot0"
                        list_dfeffect.append(df)

                    # Control: score within condition (cross-validated)
                    df = dfwithin.copy()
                    df["effect"] = "Wall"
                    # df["shape_1"] = df["shape"]
                    df["date_grp_train"] = df["date_grp"]
                    df["ep_cr_sh_1"] = df["ep_cr_sh"]
                    df["n_labels_test"] = df["n_labels_train"]

                list_dfeffect.append(df)

                # from pythonlib.tools.pandastools import replace_None_with_string
                DFEFFECT = pd.concat(list_dfeffect)
                DFEFFECT = stringify_values(DFEFFECT)
                assert sum(DFEFFECT["accuracy_adjusted"]=="none")==0

                if len(DFEFFECT)>0:
                    # for y in ["accuracy", "accuracy_adjusted", "balanced_accuracy", "balanced_accuracy_adjusted"]:
                    for y in ["accuracy_adjusted"]:
                        dfeffect = DFEFFECT[~(DFEFFECT[y] == "none")]

                        # Combining shapes
                        fig = sns.catplot(data=dfeffect, x="bregion", y=y, hue="effect", col="date", kind="bar", errorbar="se")
                        savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-1.pdf")

                        fig = sns.catplot(data=dfeffect, x="bregion", y=y, hue="effect", col="date", row="n_labels_test", kind="bar", errorbar="se")
                        savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-1b.pdf")

                        if not only_essential:
                            fig = sns.catplot(data=dfeffect, x="bregion", y=y, hue="effect", col="date", jitter=True, alpha=0.5)
                            savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-2.pdf")
                            
                            fig = sns.catplot(data=dfeffect, x="bregion", y=y, hue="effect", col="date", row="n_labels_test", jitter=True, alpha=0.5)
                            savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-2b.pdf")


                        if "shape_1" in dfeffect:
                            # Split by shapes   
                            fig = sns.catplot(data=dfeffect, x="bregion", y=y, hue="effect", col="shape_1", row="date", kind="bar", errorbar="se")
                            savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-3.pdf")

                            # Also split by n_lab_test
                            for n_labels_test in dfeffect["n_labels_test"].unique():
                                dfeffect_this = dfeffect[dfeffect["n_labels_test"] == n_labels_test]
                                fig = sns.catplot(data=dfeffect_this, x="bregion", y=y, hue="effect", col="shape_1", row="date", kind="bar", errorbar="se")
                                savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-3-nlabtest={n_labels_test}.pdf")

                            if False: # save time
                                fig = sns.catplot(data=dfeffect, x="bregion", y=y, hue="effect", col="shape_1", row="date", jitter=True, alpha=0.5)
                                savefig(fig, f"{savedir}/EFFECT-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-yvar={y}-4.pdf")

                        plt.close("all")

                        # Scatterplot (x = Wall)
                        for var_datapt in ["date_grp_train", "ep_cr_sh_1", "date"]:
                            for _y_lev_manip in ["Xshape", "Xn_Wshape"]:
                                for lenient_suff in ["", "_lenient"]:
                                    y_lev_manip = f"{_y_lev_manip}{lenient_suff}"
                                    _, fig = plot_45scatter_means_flexible_grouping(dfeffect, "effect", "Wall", y_lev_manip, 
                                                                                    "bregion", y, var_datapt, False, 
                                                                                    shareaxes=True, alpha=0.5)
                                    if fig is not None:
                                        savefig(fig, f"{savedir}/EFFECT_SCATTER-nlab={n_labels_train}-regr_yvar_grp={regr_yvar_grp}-ylev={y_lev_manip}-datapt={var_datapt}-value={y}.pdf")
                                        plt.close("all")

                        # Scatterplot (comapre generalization scores)
                        for lenient_suff in ["", "_lenient"]:
                            y_lev_manip = f"Xshape{lenient_suff}"
                            x_lev_manip = f"Xn_Wshape{lenient_suff}"
                            for var_datapt in ["date_grp_train", "ep_cr_sh_1", "date"]:                            
                                _, fig = plot_45scatter_means_flexible_grouping(dfeffect, "effect", x_lev_manip, y_lev_manip, 
                                                                                "bregion", y, var_datapt, False, 
                                                                                shareaxes=True, alpha=0.5)
                                if fig is not None:
                                    savefig(fig, f"{savedir}/EFFSCATGEN-n={n_labels_train}-yvar={regr_yvar_grp}-x={x_lev_manip}-y={y_lev_manip}-dat={var_datapt}-val={y}.pdf")
                                    plt.close("all")

                        # Scatterplot, overlaying generlatiion to (diff shape) and (same shape, diff n) on same plot.
                        from pythonlib.tools.pandastools import pivot_table
                        from pythonlib.tools.plottools import set_axis_lims_square_bounding_data_45line
                        if "Xn_Wshape" in dfeffect["effect"].unique().tolist():
                            for var_datapt in ["date_grp_train", "ep_cr_sh_1", "date"]:                            
                                    
                                dfeffect_wide = pivot_table(dfeffect, ["bregion", var_datapt], ["effect"], [y], flatten_col_names=True)
                                list_bregion = dfeffect_wide["bregion"].unique().tolist()

                                SIZE = 3
                                alpha = 0.5
                                ncols = 3
                                nrows = int(np.ceil(len(list_bregion)/ncols))
                                xvar = f"{y}-Wall"
                                for do_lenient in [False, True]: 
                                    
                                    list_yvar = [f"{y}-Xshape", f"{y}-Xn_Wshape"]
                                    if do_lenient:
                                        list_yvar = [f"{y}_lenient" for y in list_yvar]

                                    for do_paired_pts in [False, True]: # if True, then keeps only datapts with both x and y

                                        dfeffect_wide_this = dfeffect_wide.copy()
                                        if do_paired_pts:
                                            # then only keep cases with both kinds of generalization    
                                            # for _y in list_yvar:
                                            #     dfeffect_wide_this = dfeffect_wide_this[~(dfeffect_wide_this[_y].isna())]
                                            dfeffect_wide_this = dfeffect_wide_this[~(dfeffect_wide_this.loc[:, list_yvar].isna().any(axis=1))] # only keep rows without any nan

                                        ### (1) Scatterplots
                                        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
                                        xs = []
                                        ys = []
                                        for bregion, ax in zip(list_bregion, axes.flatten()):
                                            dfeffect_wide_bregion = dfeffect_wide_this[dfeffect_wide_this["bregion"] == bregion]
                                            for yvar in list_yvar:

                                                dfeffect_wide_this_this = dfeffect_wide_bregion[~(dfeffect_wide_bregion[yvar].isna())]

                                                if len(dfeffect_wide_this_this)>0:
                                                    sns.scatterplot(dfeffect_wide_this_this, x=xvar, y=yvar, ax=ax, alpha=alpha)
                                                    xs.extend(dfeffect_wide_this_this[xvar].tolist())
                                                    ys.extend(dfeffect_wide_this_this[yvar].tolist())
                                            ax.set_title(bregion)

                                        # Square axes.
                                        for ax in axes.flatten():
                                            set_axis_lims_square_bounding_data_45line(ax, xs, ys, dotted_lines="unity")
                                            set_axis_lims_square_bounding_data_45line(ax, xs, ys, dotted_lines="plus")        

                                        savefig(fig, f"{savedir}/EFFSCATMERG-n={n_labels_train}-yvar={regr_yvar_grp}-lent={do_lenient}-paired={do_paired_pts}-dat={var_datapt}-val={y}.pdf")
                                        
                                        plt.close("all")


                                        ### Plot barplots
                                        from pythonlib.tools.pandastools import convert_wide_to_long
                                        from pythonlib.tools.snstools import rotateLabel
                                        xvars = ["accuracy_adjusted-Wall", "accuracy_adjusted-Xn_Wshape", "accuracy_adjusted-Xshape"]
                                        dfeffect_wide_this_long = convert_wide_to_long(dfeffect_wide_this, xvars, ["bregion", var_datapt])

                                        fig = sns.catplot(data=dfeffect_wide_this_long, x="col_from_wide", y="col_from_wide_value", col="bregion", col_wrap=6,
                                                        order=xvars)
                                        rotateLabel(fig)
                                        for ax in fig.axes.flatten():
                                            ax.axhline(0, color="k", alpha=0.5)
                                        savefig(fig, f"{savedir}/EFFBARS-n={n_labels_train}-yvar={regr_yvar_grp}-paired={do_paired_pts}-dat={var_datapt}-val={y}-1.pdf")

                                        fig = sns.catplot(data=dfeffect_wide_this_long, x="col_from_wide", y="col_from_wide_value", col="bregion", col_wrap=6,
                                                        kind="bar", errorbar="se", order=xvars)
                                        rotateLabel(fig)                                        
                                        savefig(fig, f"{savedir}/EFFBARS-n={n_labels_train}-yvar={regr_yvar_grp}-paired={do_paired_pts}-dat={var_datapt}-val={y}-2.pdf")

                                        # - lines connecting each. Actually is not needed.
                                        # fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
                                        # for bregion, ax in zip(list_bregion, axes.flatten()):    
                                        #     dfeffect_wide_bregion = dfeffect_wide_this[dfeffect_wide_this["bregion"] == bregion]
                                        #     dfeffect_wide_this_this = dfeffect_wide_bregion[~(dfeffect_wide_bregion.loc[:, xvars].isna().any(axis=1))] # only keep rows without any nan

                                        #     xs = np.arange(len(xvars))
                                        #     for _, ys in enumerate(dfeffect_wide_this_this.loc[:, xvars].values):
                                        #         ax.plot(xs, ys, "-ok", alpha=0.5)

                                        #### Plot stats (comapre each effect to each other efect)
                                        if do_paired_pts:
                                            from pythonlib.tools.statstools import compute_all_pairwise_signrank_wrapper
                                            from pythonlib.tools.pandastools import convert_wide_to_long
                                            for bregion, ax in zip(list_bregion, axes.flatten()):    
                                                dfeffect_wide_bregion = dfeffect_wide_this[dfeffect_wide_this["bregion"] == bregion]
                                                # dfeffect_wide_this_this = dfeffect_wide_bregion[~(dfeffect_wide_bregion.loc[:, xvars].isna().any(axis=1))] # only keep rows without any nan

                                                savedir_this = f"{savedir}/EFFSTATS-n={n_labels_train}-yvar={regr_yvar_grp}-dat={var_datapt}-val={y}-br={bregion}"
                                                os.makedirs(savedir_this, exist_ok=True)
                                                dfeffect_wide_bregion_long = convert_wide_to_long(dfeffect_wide_bregion, xvars, ["bregion", var_datapt])
                                                compute_all_pairwise_signrank_wrapper(dfeffect_wide_bregion_long, [var_datapt], 
                                                                                    "col_from_wide", "col_from_wide_value", True, savedir=savedir_this, 
                                                                                    plot_contrast_vars=xvars)



def targeted_pca_state_space_split_over(DFallpa, SAVEDIR_ANALYSIS, 
                                       variables, variables_is_cat, LIST_VAR_VAROTHERS_SS, # For dim reduction and plotting state space
                                       twind_scal_force):
    """
    Quick, extract axes for each split, concatenate those axes and then orthogonalize to form a single
    subspace...
    
    And then plot state space over that.

    Here, "data_splits" means that can compute regression axes using different splits of dataset, each a level of vars_others, and
    then concatenate the results across all splits, to result in a single subspace

    Here, is hard codede for the "chunk_shape" and "chunk_within_rank" variables.
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_heatmap_var_vs_grpvar
    from pythonlib.tools.plottools import savefig
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER

    for PA in DFallpa["pa"].values:      
        from pythonlib.tools.pandastools import append_col_with_grp_index
        #### Append any new columns
        dflab = PA.Xlabels["trials"]
        dflab = append_col_with_grp_index(dflab, ["chunk_rank", "shape"], "chunk_shape")
        PA.Xlabels["trials"] = dflab

    # min_levs_per_levother = 2
    # prune_levs_min_n_trials = 4
    tbin_dur = 0.2
    tbin_slide = 0.1
    npcs_keep_force = 50
    normalization = "orthonormal"

    var_effect_1 = "chunk_shape"
    var_effect_within_split = "chunk_within_rank"
    var_other_for_split = "chunk_shape"

    # list_dfangle = []
    for _, row in DFallpa.iterrows():
        PA = row["pa"].copy()
        bregion = row["bregion"]
        event = row["event"]
        
        # if twind_scal_force is None:
        #     twind_scal = map_event_to_twind[event]
        # else:
        twind_scal = twind_scal_force

        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{event}-{bregion}"
        os.makedirs(SAVEDIR, exist_ok=True)

        ### Collect each axis
        # PLOT_COEFF_HEATMAP = False
        # demean = False # Must be false, as we dont want function to modify PA
        # get_axis_for_categorical_vars = True

        ### Convert to scalat
        # Expand channels to (chans X time bins)
        pca_reduce = True
        _, PAscal, _, _, _= PA.dataextract_state_space_decode_flex(twind_scal, tbin_dur, tbin_slide, "trials_x_chanstimes",
                                                            pca_reduce=pca_reduce, npcs_keep_force=npcs_keep_force)

        ### Get axes for each split
        DFCOEFF, columns_each_split = PAscal.regress_neuron_task_variables_all_chans_data_splits(variables, variables_is_cat, 
                                                                    var_effect_within_split, var_other_for_split)

        # Plot final
        PAscal.regress_neuron_task_variables_all_chans_plot_coeffs(DFCOEFF, savedir_coeff_heatmap=SAVEDIR)

        ### Using DFCOEFF, project to this subspace
        list_subspace_tuples = [(var_effect_1, col) for col in columns_each_split]
        dict_subspace_pa, _, _ = PAscal.dataextract_subspace_targeted_pca_project(DFCOEFF, 
                                                                                    list_subspace_tuples, normalization)

        ### Plot state space, for each pair of axes (i.e,, each time using one of the chunk ranks).
        LIST_DIMS = [(0,1), (1,2)]
        for subspace_tuple in list_subspace_tuples:

            savedir = f"/{SAVEDIR}/subspace={subspace_tuple}"
            os.makedirs(savedir, exist_ok=True)

            # ### Get VAF between each axis of subspace
            # # - first, get meaned data.
            # # data_trials = x.T # (chans, trials)
            # subspace_axes = dict_subspace_axes_orig[subspace]
            # assert len(subspace)==subspace_axes.shape[1]
            # min_n_trials_in_lev = 3
            # pa_mean = PA.slice_and_agg_wrapper("trials", variables, min_n_trials_in_lev=min_n_trials_in_lev)
            # data_mean = pa_mean.X.squeeze() # (nchans, nconditions)

            # naxes = subspace_axes.shape[1]
            # for i in range(naxes):
            #     for j in range(naxes):
            #         if j>i:
            #             basis_vectors_1 = subspace_axes[:,i][:, None]
            #             basis_vectors_2 = subspace_axes[:,j][:, None]
            #             out = dimredgood_subspace_variance_accounted_for(data_mean, basis_vectors_1, basis_vectors_2)
            #             from pythonlib.tools.expttools import writeDictToTxtFlattened
            #             writeDictToTxtFlattened(out, f"{savedir}/VAF-subspace={subspace}.txt")

            ### PLOT -- plot state space
            PAredu = dict_subspace_pa[subspace_tuple]
            Xredu = PAredu.X # (chans, trials, 1)
            dflab = PAredu.Xlabels["trials"]
            x = Xredu.squeeze().T # (trials, chans)

            for i, (var_effect, vars_others) in enumerate(LIST_VAR_VAROTHERS_SS):
                if (var_effect in dflab.columns) and all([_var in dflab.columns for _var in vars_others]):

                    ### Plot scalars
                    # First, save counts
                    fig = grouping_plot_n_samples_heatmap_var_vs_grpvar(dflab, var_effect, vars_others)
                    savefig(fig, f"{savedir}/counts-{i}-var={var_effect}-varother={'|'.join(vars_others)}")

                    # Second, plot scalars
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(x, dflab, var_effect, savedir,
                                                                    vars_subplot=vars_others, list_dims=LIST_DIMS,
                                                                    overlay_mean_orig=True)
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
    question = sys.argv[3]
    run_number = int(sys.argv[4])

    # question = "RULE_ANBMCK_STROKE"
    version = "stroke"
    combine = False

    # PLOTS_DO = [4] # Good
    # PLOTS_DO = [4.1] # Good
    # PLOTS_DO = [6.1] # Good
    # PLOTS_DO = [6.2] # Good
    PLOTS_DO = [4.2] # Good
    PLOTS_DO = [7.2] # Good
    PLOTS_DO = [7.1] # Good

    if any([(x>=7) and (x<8) for x in PLOTS_DO]) and (question == "RULE_ANBMCK_STROKE"):
    # if 7.1 in PLOTS_DO:
        # Then you want to also load SP
        ### (1) load Grammar Dfallpa
        DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                            question=question)
        DFallpa = dfpa_concat_bregion_to_combined_bregion(DFallpa)

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

        except Exception as err:
            print(err)

    else:
        # Load a single DFallPA
        DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                        question=question)
        if combine == False:
            DFallpa = dfpa_concat_bregion_to_combined_bregion(DFallpa)

    # Make a copy of all PA before normalization
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

    ################ PARAMS
    
    ################################### PLOTS
    for plotdo in PLOTS_DO:
        # if plotdo==0:
        #     # (1) Heatmaps, split in many ways, to show population data
        #     savedir = f"{SAVEDIR}/HEATMAPS/{animal}-{date}-combine={combine}-var_other={var_other}"
        #     os.makedirs(savedir, exist_ok=True)
        #     heatmaps_plot_wrapper(DFallpa, animal, date, savedir, var_other=var_other)

        # elif plotdo==1:
        #     # (2) Scalar state space.
        #     savedir = f"{SAVEDIR}/SCALAR_SS/{animal}-{date}-combine={combine}-var_other={var_other}"
        #     os.makedirs(savedir, exist_ok=True)
        #     statespace_scalar_plot(DFallpa, animal, date, savedir, var_other)
        
        # elif plotdo==2:
        #     # Decode (cross-condition)
        #     savedir = f"{SAVEDIR}/DECODE/{animal}-{date}-combine={combine}-var_other={var_other}"
        #     os.makedirs(savedir, exist_ok=True)
        #     decodercross_plot(DFallpa, savedir)
        
        # elif plotdo==3:
        #     # Time-varying euclidian distnaces (e.g., same|diff).
        #     # MULT ANALYSIS: see notebook 241110_shape_invariance_all_plots_SP
        #     # ... [LOAD MULT DATA] Euclidian (time-varying, here summarize)
        #     savedir = f"{SAVEDIR}/EUCLIDIAN/{animal}-{date}-combine={combine}-var_other={var_other}"
        #     os.makedirs(savedir, exist_ok=True)
        #     euclidian_time_resolved(DFallpa, animal, date, var_other, savedir)

        if plotdo==4:
            # Euclidian Shuff. Replicating what previuosly did in the generic euclidan distance code.
            # This is better than plotdo==3, because:

            # savedir = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-combine={combine}-var_other={var_other}"
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-comb={combine}-q={question}"
        
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            # DO_RSA_HEATMAPS = True
            # DO_SHUFFLE = False
            euclidian_time_resolved_fast_shuffled(DFallpa, animal, SAVEDIR_ANALYSIS, question)

        elif plotdo==4.1:
            # Euclidian shuff, for shape vs. seqsup. New code that is better and much faster, as prunes to focus on the
            # important contrasts in dataset.

            SAVEDIR_ANALYSIS = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-comb={combine}-q={question}-seqsupgood"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            euclidian_time_resolved_fast_shuffled(DFallpa, animal, SAVEDIR_ANALYSIS, question, version_seqsup_good=True)

        elif plotdo==4.2:
            # Just doing RSA plots without caring much about context, etc.
            # Useful for quick inspection of representational geometry
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/RSA_QUICK/{animal}-{date}-comb={combine}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            euclidian_time_resolved_fast_shuffled_quick_rsa(DFallpa, animal, SAVEDIR_ANALYSIS)

        # elif plotdo==5:
        #     # (2) Traj state space.
        #     savedir = f"{SAVEDIR}/TRAJ_SS/{animal}-{date}-combine={combine}-var_other={var_other}"
        #     os.makedirs(savedir, exist_ok=True)
        #     statespace_traj_plot(DFallpa, animal, date, savedir, var_other)

        elif plotdo==6.1:
            ### Targeted dim reduction, where single axis for "chunk_within" is identified combining data across all chunks.
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_dim_redu_angles/{animal}-{date}-comb={combine}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

            # DFallpa = DFallpa[:2]

            for PA in DFallpa["pa"].values:      
                from pythonlib.tools.pandastools import append_col_with_grp_index
                #### Append any new columns
                dflab = PA.Xlabels["trials"]
                dflab = append_col_with_grp_index(dflab, ["chunk_rank", "shape"], "chunk_shape")
                PA.Xlabels["trials"] = dflab

            from pythonlib.tools.vectools import average_vectors_wrapper, get_vector_from_angle
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single, targeted_pca_euclidian_dist_angles

            variables = ['epoch', 'chunk_shape', 'gridloc', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 
                        'CTXT_shape_prev', 'chunk_within_rank']
            variables_is_cat = [True, True, True, True, True, True, True, False]
            assert len(variables)==len(variables_is_cat)

            list_subspaces = [
                ("chunk_shape", "chunk_within_rank"),
            ]

            LIST_VAR_VAROTHERS_SS = [
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
                ("chunk_within_rank", ['epoch', 'chunk_shape']),
                ("chunk_within_rank", ['epoch']),
                ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
                ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
                ("chunk_shape", ['epoch', 'chunk_within_rank']),
                ("chunk_shape", ['epoch']),
            ]

            LIST_DIMS = [(0,1), (1,2)]

            LIST_VAR_VAROTHERS_REGR = [
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
                ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
            ]

            subspace_tuple = ("chunk_shape", "chunk_within_rank")

            min_levs_per_levother = 2
            prune_levs_min_n_trials = 4

            from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _get_list_twind_by_animal
            _list_twind, _, _ = _get_list_twind_by_animal(animal, "00_stroke", "traj_to_scalar")
            twind_scal = _list_twind[0]

            DFANGLE = targeted_pca_euclidian_dist_angles(DFallpa, SAVEDIR_ANALYSIS, 
                                                        variables, variables_is_cat, list_subspaces, LIST_VAR_VAROTHERS_SS, # For dim reduction and plotting state space
                                                        subspace_tuple, LIST_VAR_VAROTHERS_REGR, twind_scal_force=twind_scal)

            ### (2) Make all plots
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import targeted_pca_euclidian_dist_angles_plots
            for var_vector_length in ["dist_yue_diff", "dist_norm"]:
                for length_method in ["sum", "dot"]:
                    for min_levs_exist in [3, 2]:
                        savedir = f"{SAVEDIR_ANALYSIS}/PLOTS/varlength={var_vector_length}-lengthmeth={length_method}-minlevs={min_levs_exist}"
                        os.makedirs(savedir, exist_ok=True)
                        targeted_pca_euclidian_dist_angles_plots(DFANGLE, var_vector_length, length_method, min_levs_exist, savedir)

        elif plotdo==6.2:
            ### [Devo] New method to get different subspaces each using subbset of data
            # i.e., get separate axes for chunk_within_shape for each shape
            from pythonlib.tools.vectools import average_vectors_wrapper, get_vector_from_angle
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single, targeted_pca_euclidian_dist_angles
            from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_state_space_split_over

            SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_dim_redu_angles_split/{animal}-{date}-comb={combine}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

            min_levs_per_levother = 2
            prune_levs_min_n_trials = 4

            from neuralmonkey.scripts.analy_euclidian_dist_pop_script import _get_list_twind_by_animal
            _list_twind, _, _ = _get_list_twind_by_animal(animal, "00_stroke", "traj_to_scalar")
            twind_scal = _list_twind[0]

            variables = ['epoch', 'chunk_shape', 'gridloc', 'loc_on_clust', 'CTXT_locoffclust_prev', 'loc_off_clust', 
                        'CTXT_shape_prev', 'chunk_within_rank']
            variables_is_cat = [True, True, True, True, True, True, True, False]

            LIST_DIMS = [(0,1), (1,2)]
            LIST_VAR_VAROTHERS = [
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
                ("chunk_within_rank", ['epoch', 'chunk_shape', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
                ("chunk_within_rank", ['epoch', 'chunk_shape']),
                ("chunk_within_rank", ['epoch']),
                ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev', 'loc_off_clust']),
                ("chunk_shape", ['epoch', 'chunk_within_rank', 'loc_on_clust', 'CTXT_locoffclust_prev', 'CTXT_shape_prev']),
                ("chunk_shape", ['epoch', 'chunk_within_rank']),
                ("chunk_shape", ['epoch']),
            ]

            targeted_pca_state_space_split_over(DFallpa, SAVEDIR_ANALYSIS, 
                                                variables, variables_is_cat, LIST_VAR_VAROTHERS, # For dim reduction and plotting state space
                                                twind_scal)
        elif plotdo in [7.1, 7.2]:
            ### [Good] Targeted PCA, doing very carefully. This is similar to above, but many updates:
            # Major:
            # - using multiple axes for each subspace, instead of one axis per subspace.
            # Minor:
            # - Careful train-test splitting of data

            # MULT ANALY: notebooks_tutorials/250510_syntax_good_state.ipynb
            # --> [MULT, Euclidean, ALL ANALYSES] Load DFEFFECT and plot. Final figures for paper.

            # This plots state space, and also computes euclidean
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_dim_redu_v2/run{run_number}/{animal}-{date}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            
            preprocess_dfallpa_motor_features(DFallpa, plot_motor_values=False, do_zscore=True)

            # If is 7.2, then does nothing, except run the preprpcessing in order to extract and print useful information about syntax
            HACK_ONLY_PREPROCESS = plotdo==7.2

            # Run
            # DO_PLOT_STATE_SPACE = False
            # DO_EUCLIDEAN = False
            # DO_ORDINAL_REGRESSION = True

            DO_PLOT_STATE_SPACE = True
            DO_EUCLIDEAN = True
            DO_ORDINAL_REGRESSION = False
            targeted_pca_clean_plots_and_dfdist(DFallpa, animal, date, 
                                                SAVEDIR_ANALYSIS, 
                                                DEBUG=False, HACK_ONLY_PREPROCESS=HACK_ONLY_PREPROCESS,
                                                run_number=run_number, 
                                                DO_PLOT_STATE_SPACE = DO_PLOT_STATE_SPACE, 
                                                DO_EUCLIDEAN = DO_EUCLIDEAN, 
                                                DO_ORDINAL_REGRESSION = DO_ORDINAL_REGRESSION)
        else:
            print(PLOTS_DO)
            assert False