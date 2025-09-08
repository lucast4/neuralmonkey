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
    """
    ### [MULT] Loading all dfdists and making summary plots
    # run = 12
    # SAVEDIR = f"/tmp/SYNTAX_TARGETED_PCA_run{run}"

    OLD_VERSION = False
    expected_n_subspaces = 6
    expected_n_questions = 4

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
    from glob import glob
    from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
    from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific
    from pythonlib.tools.pandastools import replace_None_with_string, stringify_values
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import append_col_with_grp_index

    expected_n_bregions = 8

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
                dfdist_concat = aggregGeneral(dfdist_concat, ["bregion", "labels_1", "labels_2", "var_subspace", "var_conj", "var_conj_lev", "question", "subspace"], 
                                            ["dist_mean", "DIST_98", "dist_norm", "dist_yue_diff", "n1", "n2", "data_dim", "npcs_euclidean"], nonnumercols=None)
                assert all(dfdist_concat["subspace"]==subspace)
                map_subspace_to_dfdist[subspace] = dfdist_concat

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

        if len(LIST_DFDIST)==0:
            # Then always skip, this is just a missing day
            print("Skipping, as list_dfdist is empty")
            return None

        if False: # No need -- it will be obvious if something is missing
            assert len(LIST_DFDIST)==(expected_n_bregions * expected_n_questions * expected_n_subspaces * expected_n_iters), "this is to make sure that you arent saving partial results"

        DFDIST = pd.concat(LIST_DFDIST).reset_index(drop=True)
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
    # assert len(DFDIST_THIS)>0

    DFDIST_THIS = DFDIST_THIS[(DFDIST_THIS["subspace"].isin(subspaces))]
    if len(DFDIST_THIS)==0:
        return None
    # assert len(DFDIST_THIS)>0

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
    - contrasts_diff, contrasts_either, each a list of str.

    RETURNS:
    - pruned dfdist (copy) or None if all rows pruned
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
    map_question_to_euclideanvars = params["map_question_to_euclideanvars"]
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
    
    PARAMS:
    - dfdist, holds data for a single animal-date (across regions)
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

def targeted_pca_MULT_2_plot_single_load(animal, date, run):
    """
    Load results for a single (animal, date) after the initial run -- i.e. the dfdist.
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params

    SAVEDIR = f"/tmp/SYNTAX_TARGETED_PCA_run{run}"
    SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"

    yvar = "dist_yue_diff"
            
    SAVEDIR = f"{SAVEDIR_MULT}/summary_each_date-yvar={yvar}/{animal}-{date}"

    # Load it
    try:
        DFDIST = pd.read_pickle(f"{SAVEDIR_MULT}/DFDIST-{animal}-{date}.pkl")
        print(animal, date)
    except Exception as err:
        print(err)
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

    DFDIST, map_subspace_to_shorthand = targeted_pca_MULT_2_postprocess(DFDIST)
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
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import targeted_pca_clean_plots_and_dfdist_params
    import seaborn as sns
    # from neuralmonkey.scripts.analy_syntax_good_eucl_state import _targeted_pca_clean_plots_and_dfdist_MULT_plot_single
    from pythonlib.tools.snstools import rotateLabel

    SAVEDIR = f"/tmp/SYNTAX_TARGETED_PCA_run{run}"
    SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"

    SKIP_Q_7 = True
    do_scatter_datapts = False
    # yvar = "dist_yue_diff_unnorm"
    yvar = "dist_yue_diff"
            
    SAVEDIR = f"{SAVEDIR_MULT}/summary_each_date-yvar={yvar}/{animal}-{date}"

    DFDIST, map_question_to_euclideanvars, map_question_to_varsame = targeted_pca_MULT_2_plot_single_load(animal, date, run)
    
    if DFDIST is None:
        # Then this (animal, date) has no data.
        return None
    
    # Skip if done
    if not OVERWRITE:
        if os.path.exists(SAVEDIR):
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
    """
    
    yvar = "dist_yue_diff"
    if SAVEDIR_MULT is None:
        SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/syntax_good/targeted_dim_redu_v2/run{run}/MULT"

    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
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
            LIST_DFEFFECT_ALL.append(dfeffect)
            print("Loaded: ", SAVEDIR)
        except FileNotFoundError as err:
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
    DFEFFECT_ALL_AGG = aggregGeneral(DFEFFECT_ALL, ["effect", "animal", "date", "bregion", "question", "subspace"], ["dist_yue_diff"])


    if return_dfeffect:
        return DFEFFECT_ALL, LIST_EFFECT_PAIRS

    SAVEDIR = f"{SAVEDIR_MULT}/COMBINED-{animal}"
    os.makedirs(SAVEDIR, exist_ok=True)
    
    for eff1, eff2 in LIST_EFFECT_PAIRS:
        
        if (eff1 in DFEFFECT_ALL["effect"].unique().tolist()) and (eff2 in DFEFFECT_ALL["effect"].unique().tolist()):

            # (1) Subplot = date
            print(eff1, eff2)
            print(DFEFFECT_ALL["effect"].unique())
            _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_ALL, "effect", eff1, eff2, "date", yvar, "bregion", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-1.pdf")

            # (2) Subplot = bregion
            _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_ALL, "effect", eff1, eff2, "bregion", yvar, "date", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-2.pdf")

            # (3) Single summary plot
            _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_ALL, "effect", eff1, eff2, None, yvar, "bregion", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-3.pdf")

            # Too slow:
            # _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_ALL, "effect", eff1, eff2, "bregion", yvar, "index", plot_error_bars=False, shareaxes=True, plot_text=False);
            # savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-4.pdf")

            # (3) Single summary plot (dates)
            _, fig = plot_45scatter_means_flexible_grouping(DFEFFECT_ALL_AGG, "effect", eff1, eff2, None, yvar, "bregion", shareaxes=True);
            if fig is not None:
                savefig(fig, f"{SAVEDIR}/effects-scatter-{eff2}-vs-{eff1}-3-datapt=date.pdf")
        
            # And a "difference metric"
            from pythonlib.tools.pandastools import summarize_featurediff
            dfsummary, _, _, _, COLNAMES_DIFF = summarize_featurediff(
                    DFEFFECT_ALL_AGG, "effect", [eff2, eff1], FEATURE_NAMES=[yvar], 
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
    
    # # Free up space
    # del LIST_DFEFFECT_ALL
    # del DFEFFECT_ALL

if __name__=="__main__":
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates

    # animal = sys.argv[1]
    # date = int(sys.argv[2])
    RUN = int(sys.argv[1])
    plot_each_date=int(sys.argv[2])==1
    # RUN = 15

    PLOTS_DO = [2.0, 2.1, 2.2] # Good
    # PLOTS_DO = [2.1, 2.2] # Good
    # PLOTS_DO = [2.2] # Good
    # expt_kind="RULE_ANBMCK_STROKE"
    # expt_kind="RULESW_ANY_SEQSUP_STROKE"

    if RUN in [13, 25, 26]:
        save_suffix = "sh_vs_seqsup"
    else:
        save_suffix = "AnBmCk_general"
    # dates, question, _, _ = load_preprocess_get_dates("Diego", "sh_vs_seqsup")

    for plotdo in PLOTS_DO:
        if plotdo==1.0:
            """ Older plots, before doing the good Targeted PCA (2.0+)"""
            mult_plot_all_wrapper()

        elif plotdo==2.0:
            """ Step 1: Save a single DFDIST"""
            ### Collect all the animal-date pairs
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
                from multiprocessing import Pool
                MULTIPROCESS_N_CORES = 24
                list_run = [RUN for _ in range(len(LIST_ANIMAL))]
                list_expt_kind = [question for _ in range(len(LIST_ANIMAL))]
                with Pool(MULTIPROCESS_N_CORES) as pool:
                    pool.starmap(targeted_pca_MULT_1_load_and_save, zip(LIST_ANIMAL, LIST_DATE, list_run, list_expt_kind))
            else:
                for animal, date in zip(LIST_ANIMAL, LIST_DATE):
                    targeted_pca_MULT_1_load_and_save(animal, date, run=RUN, expt_kind=question)

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
            if False:
                from multiprocessing import Pool
                MULTIPROCESS_N_CORES = 24
                with Pool(MULTIPROCESS_N_CORES) as pool:
                    pool.starmap(lambda x, y: targeted_pca_MULT_2_plot_single(x, y, run=RUN), zip(LIST_ANIMAL, LIST_DATE))
            else:
                for animal, date in zip(LIST_ANIMAL, LIST_DATE):
                    targeted_pca_MULT_2_plot_single(animal, date, run=RUN, SKIP_PLOTS=not plot_each_date)

        elif plotdo==2.2:
            """ Step 3: Plot effects, and save a single dfeffects (for each animal, date)"""
            ### Collect all the animal-date pairs

            for animal in ["Diego", "Pancho"]:
                targeted_pca_MULT_3_combined_plots(animal, RUN, save_suffix)

        else:
            assert False
