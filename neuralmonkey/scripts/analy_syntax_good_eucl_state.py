"""
Organizing good plots for syntax, espeically:
- euclidian dist
- state space

NOTEBOOK: 230510_syntax_good.ipynb

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
        dfdistthis = dfdistthis[(dfdistthis["stroke_index_1"]>1) & (dfdistthis["stroke_index_2"]>1)].reset_index(drop=True)

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


def preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials, prune_min_n_levs, filtdict,
                  savedir, 
                subspace_projection, subspace_projection_fitting_twind,
                twind_analy, tbin_dur, tbin_slide, scalar_or_traj="traj"):

    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper, grouping_plot_n_samples_conjunction_heatmap
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_print_n_samples
    from neuralmonkey.scripts.analy_shape_invariance_all_plots_SP import _preprocess_pa_dim_reduction
    from pythonlib.tools.plottools import savefig
    
    PA = PA.copy()
    dflab = PA.Xlabels["trials"]

    #### Append any new columns
    dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_role"], "epch_sytxrol")
    dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_role", "shape", "gridloc"], "sytx_all")

    # Consolidate vars_others into a single variable
    # save text file holding the params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({"var_effect":var_effect, "vars_others":vars_others}, f"{savedir}/vars.txt")
    dflab = append_col_with_grp_index(dflab, vars_others, "_vars_others")

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

    # Prune n datapts
    plot_counts_heatmap_savepath = f"{savedir}/counts_conj.pdf"
    PA, _, _= PA.slice_extract_with_levels_of_conjunction_vars(var_effect, vars_others, prune_min_n_trials, prune_min_n_levs,
                                                        plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)
    
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
    PA = _preprocess_pa_dim_reduction(PA, subspace_projection, subspace_projection_fitting_twind,
                                twind_analy, tbin_dur, tbin_slide, savedir, scalar_or_traj=scalar_or_traj)
    
    plt.close("all")
    
    return PA

def euclidian_time_resolved_fast_shuffled(DFallpa, animal, SAVEDIR_ANALYSIS, question):
    """

    """
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

    list_subspace_projection = ["sytx_all", "epch_sytxrol", "syntax_role"]

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
    list_twind_scal = [twind_ideal, (-0.3, -0.1)]

    ### Load params
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    LIST_VAR, LIST_VARS_OTHERS, LIST_CONTEXT, LIST_PRUNE_MIN_N_LEVS, LIST_FILTDICT = params_getter_euclidian_vars(question, 
                                                                                                                  context_version="new")

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
                                twind_analy, tbin_dur, tbin_slide)
                    if PAthisRedu is None:
                        # Try again, using lower min n trials.
                        PAthisRedu = preprocess_pa(PA, var_effect, vars_others, prune_min_n_trials-1, prune_min_n_levs, filtdict,
                                    savedir, 
                                    subspace_projection, subspace_projection_fitting_twind,
                                    twind_analy, tbin_dur, tbin_slide)
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

def mult_plot_grammar_vs_seqsup(DFDIST, SAVEDIR, animal, date, shape_or_loc_rule="shape"):
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

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good"

    animal = sys.argv[1]
    date = int(sys.argv[2])
    question = sys.argv[3]

    # question = "RULE_ANBMCK_STROKE"

    version = "stroke"
    combine = False

    PLOTS_DO = [4] # Good

    # Load a single DFallPA
    DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                     question=question)

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
            # Euclidian Shuff. This is better than plotdo==3, because:

            # savedir = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-combine={combine}-var_other={var_other}"
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/EUCLIDIAN_SHUFF/{animal}-{date}-comb={combine}-q={question}"
        
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
            # DO_RSA_HEATMAPS = True
            # DO_SHUFFLE = False
            euclidian_time_resolved_fast_shuffled(DFallpa, animal, SAVEDIR_ANALYSIS, question)

        # elif plotdo==5:
        #     # (2) Traj state space.
        #     savedir = f"{SAVEDIR}/TRAJ_SS/{animal}-{date}-combine={combine}-var_other={var_other}"
        #     os.makedirs(savedir, exist_ok=True)
        #     statespace_traj_plot(DFallpa, animal, date, savedir, var_other)
        else:
            print(PLOTS_DO)
            assert False