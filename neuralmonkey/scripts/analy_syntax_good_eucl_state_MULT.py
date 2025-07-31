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

if __name__=="__main__":

    mult_plot_all_wrapper()