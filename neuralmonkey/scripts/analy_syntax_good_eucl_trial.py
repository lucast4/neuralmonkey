"""
To analyze trial-data, i.e, usually during planning, before movement onset.
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

from pythonlib.tools.snstools import rotateLabel

NPCS_KEEP = 8
N_MIN_TRIALS = 4

map_event_to_twind = {
    "03_samp":[0.2, 1.2],
    "05_first_raise":[-0.5, -0.05],
    "06_on_strokeidx_0":[-0.5, -0.05],
    "00_stroke":[-0.3, 0.3]
}

def preprocess_dfallpa_basic_quick(DFallpa):
    """
    Quickly process the data, to add some columns that are useful for plotting.
    """
    
    for PA in DFallpa["pa"]:
        
        from neuralmonkey.scripts.analy_syntax_good_eucl_state import preprocess_pa_syntax
        preprocess_pa_syntax(PA)
        
        dflab = PA.Xlabels["trials"]

        # dflab = append_col_with_grp_index(dflab, vars_others, "_var_other")
        # dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_concrete"], "epch_sytxcncr")
        dflab = append_col_with_grp_index(dflab, ["epoch", "seqc_0_loc", "seqc_0_shape", "syntax_concrete"], "var_all_conditions")

        # # Add n items in each shape slot
        # nslots = len(dflab["syntax_concrete"].values[0])
        # list_slots = []
        # for i in range(3):
        #     key = f"syntax_slot_{i}"
        #     if i > nslots-1:
        #         dflab[key] = 0
        #     else:
        #         dflab[key] = [x[i] for x in dflab["syntax_concrete"]]
        #     list_slots.append(key)
        # print("Added these columns to dflab: ", list_slots)

        # # Add ratio between slot 0 and 1
        # if ("syntax_slot_0" in dflab) & ("syntax_slot_1" in dflab):
        #     # Add 0.01 so that doesnt devide by 0.
        #     dflab["syntax_slot_ratio"] = (dflab["syntax_slot_1"]+0.01)/(dflab["syntax_slot_0"]+0.01 + dflab["syntax_slot_1"]+0.01)

        #     if np.any(np.isnan(dflab["syntax_slot_ratio"])):
        #         print(dflab["syntax_slot_ratio"])
        #         assert False

        # # count up how many unique shapes are shown
        # def _n_shapes(syntax_concrete):
        #     # e.g., (1,3,0) --> 2
        #     return sum([x>0 for x in syntax_concrete])
        # dflab["shapes_n_unique"] = [_n_shapes(sc) for sc in dflab["syntax_concrete"]]

        PA.Xlabels["trials"] = dflab

def preprocess_dfallpa(DFallpa, subspace_projection, tbin_slide, tbin_dur, savedir, HACK=False):
    """
    Preprocess the data, to add some columns that are useful for plotting.
    Also, do the dimensionality reduction.
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    
    raw_subtract_mean_each_timepoint = False

    # prune to just 03_samp, for now.
    # map_event_to_twind = {
    #     "03_samp":[0.2, 1.0],
    #     "05_first_raise":[-0.5, -0.05],
    #     "06_on_strokeidx_0":[-0.5, -0.05],
    # }

    ### Preprocessing
    dim_red_method, superv_dpca_params = params_subspace_projection(subspace_projection)

    if "umap" in subspace_projection:
        LIST_DIMS = [(0,1)]
    else:
        LIST_DIMS = [(0,1), (2,3), (3,4)]

    # (1) First, dim reduction
    if HACK:
        superv_dpca_var = "syntax_slot_0"
        superv_dpca_vars_group = None
        superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']
    else:
        superv_dpca_var = superv_dpca_params['superv_dpca_var']
        superv_dpca_vars_group = superv_dpca_params['superv_dpca_vars_group']
        superv_dpca_filtdict = superv_dpca_params['superv_dpca_filtdict']

    # Go thru each PA
    list_pa = []
    # _savedir = "/tmp"
    for _, row in DFallpa.iterrows():
        PA = row["pa"]
        event = row["event"]
        bregion = row["bregion"]
        twind_scal = map_event_to_twind[event]

        _savedir = f"{savedir}/preprocess/{event}-{bregion}"
        os.makedirs(_savedir, exist_ok=True)

        _, PAredu = PA.dataextract_dimred_wrapper("scal", dim_red_method, _savedir, twind_scal, tbin_dur=tbin_dur, 
                                                tbin_slide = tbin_slide, NPCS_KEEP=NPCS_KEEP, dpca_var=superv_dpca_var, 
                                                dpca_vars_group=superv_dpca_vars_group, dpca_filtdict=superv_dpca_filtdict, 
                                                raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)    
        list_pa.append(PAredu)
        plt.close("all")
    
    # NOTE: This _could_ be None, if dimreduc variables dont exist.
    DFallpa["pa_redu"] = list_pa    

    return LIST_DIMS
    
def preprocess_pa(PA, grp_vars_prune_enough_trials, prune_min_n_trials, savedir):
    """
    Preprocess PA before doing analysis, for trial-level data
    RETURNS:
    - Returns a copy of PA
    """
    from pythonlib.tools.pandastools import grouping_print_n_samples
    
    PA = PA.copy()
    
    # Prune n datapts -- each level of flat grp vars
    PA = PA.slice_extract_with_levels_of_var_good_prune(grp_vars_prune_enough_trials, prune_min_n_trials)
    
    if PA is None:
        print("Pruned all data!!")
        return None
    
    # Save (print) useful summaries of the syntaxes for this day
    dflab = PA.Xlabels["trials"]
    path = f"{savedir}/counts-final.txt"
    grouping_print_n_samples(dflab, grp_vars_prune_enough_trials, savepath=path)

    savepath = f"{savedir}/syntax_counts-1.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_concrete", "behseq_shapes"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-2.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_concrete", "behseq_shapes", "behseq_locs_clust"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-3.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_concrete", "syntax_slot_0", 
                                    "syntax_slot_1", "syntax_slot_2"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-5.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_slot_0", "syntax_concrete"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-6.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_slot_1", "syntax_concrete"], savepath=savepath)

    savepath = f"{savedir}/syntax_counts-7.txt"
    grouping_print_n_samples(dflab, ["epoch", "FEAT_num_strokes_beh", "syntax_slot_2", "syntax_concrete"], savepath=savepath)
    
    return PA

def state_space_targeted_pca_do_projection(DFallpa, variables, variables_is_cat, list_subspaces, SAVEDIR,
                                           tbin_dur = 0.2, tbin_slide = 0.1):
    """
    Run dim reduction in bulk and extract dim reduced PA objects, and return in a dict.
    RETURNS:
    - map_BrEvSs_to_Paredu: dict, mapping (bregion, event, subspace) to PAredu
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_heatmap_var_vs_grpvar
    from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
    from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER
    from neuralmonkey.analyses.state_space_good import dimredgood_subspace_variance_accounted_for

    # tbin_dur = 0.2
    # tbin_slide = 0.1
    npcs_keep_force = 50
    normalization = "orthonormal"
    PLOT_COEFF_HEATMAP = False

    map_BrEvSs_to_Paredu = {}
    for _, row in DFallpa.iterrows():
        PA = row["pa"]
        bregion = row["bregion"]
        event = row["event"]        

        twind_scal = map_event_to_twind[event]

        # Expand channels to (chans X time bins)
        pca_reduce = True
        _, PA, _, _, _= PA.dataextract_state_space_decode_flex(twind_scal, tbin_dur, tbin_slide, "trials_x_chanstimes",
                                                            pca_reduce=pca_reduce, npcs_keep_force=npcs_keep_force)

        ### Compute regression -- get coefficients
        savedir_coeff_heatmap = f"{SAVEDIR}"
        dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed, dfcoeff, PA = PA.dataextract_subspace_targeted_pca(variables, 
                                                variables_is_cat, list_subspaces, demean=True, 
                                                normalization=normalization, plot_orthonormalization=False, 
                                                PLOT_COEFF_HEATMAP=PLOT_COEFF_HEATMAP, PRINT=False,
                                                savedir_coeff_heatmap=savedir_coeff_heatmap)
        
        for subspace in list_subspaces:

            print("Running ... ", bregion, event, subspace)

            ### PLOT -- plot state space
            PAredu = dict_subspace_pa[subspace]
            key = (bregion, event, subspace)
            map_BrEvSs_to_Paredu[key] = PAredu

    return map_BrEvSs_to_Paredu

def state_space_targeted_pca_scalar_single_one_axis_per_var(PA, twind_scal, variables, variables_is_cat, list_subspaces, 
                                           LIST_VAR_VAROTHERS, LIST_DIMS, SAVEDIR, just_extract_paredu=False, 
                                           tbin_dur = 0.2, tbin_slide = 0.1):
    """
    [GOOD], do targeted PCA for a single PA, along with state space plots.
    PARAMS:
    - variables: list of variables to regress out
    - variables_is_cat: list of bools, whether each variable is categorical. This doesnt affect the regression, so your data
    needs to be either cat or ordinal.
    list_subspaces: list of tuples, each tuple is a subspace to project onto. usually tuples are of length 2, holding strings 
    referring to the variable names.
    - LIST_VAR_VAROTHERS, LIST_DIMS, SAVEDIR, for plotting state space.
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_heatmap_var_vs_grpvar
    from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
    from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER
    from neuralmonkey.analyses.state_space_good import dimredgood_subspace_variance_accounted_for

    # tbin_dur = 0.2
    # tbin_slide = 0.1
    npcs_keep_force = 50
    normalization = "orthonormal"
    PLOT_COEFF_HEATMAP = True

    assert list_subspaces is not None

    ### Convert to scalat
    if False:
        # Get scalars
        PA = PA.slice_by_dim_values_wrapper("times", twind_scal).agg_wrapper("times")
    else:
        # Expand channels to (chans X time bins)
        pca_reduce = True
        _, PA, _, _, _= PA.dataextract_state_space_decode_flex(twind_scal, tbin_dur, tbin_slide, "trials_x_chanstimes",
                                                            pca_reduce=pca_reduce, npcs_keep_force=npcs_keep_force)

    ### Compute regression -- get coefficients
    savedir_coeff_heatmap = f"{SAVEDIR}"
    dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed, dfcoeff, PA = PA.dataextract_subspace_targeted_pca_one_axis_per_var(
                                            variables, variables_is_cat, list_subspaces, demean=True, 
                                            normalization=normalization, plot_orthonormalization=False, 
                                            PLOT_COEFF_HEATMAP=PLOT_COEFF_HEATMAP, PRINT=False,
                                            savedir_coeff_heatmap=savedir_coeff_heatmap)

    # Skip plots, just get data
    if just_extract_paredu:
        return dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed, dfcoeff, PA
    
    ### Compute subspace angles
    for subspace in list_subspaces:

        savedir = f"/{SAVEDIR}/subspace={subspace}"
        os.makedirs(savedir, exist_ok=True)

        ### Get VAF between each axis of subspace
        # - first, get meaned data.
        # data_trials = x.T # (chans, trials)
        subspace_axes = dict_subspace_axes_orig[subspace]
        assert len(subspace)==subspace_axes.shape[1]
        min_n_trials_in_lev = 3
        pa_mean = PA.slice_and_agg_wrapper("trials", variables, min_n_trials_in_lev=min_n_trials_in_lev)
        data_mean = pa_mean.X.squeeze() # (nchans, nconditions)

        naxes = subspace_axes.shape[1]
        for i in range(naxes):
            for j in range(naxes):
                if j>i:
                    basis_vectors_1 = subspace_axes[:,i][:, None]
                    basis_vectors_2 = subspace_axes[:,j][:, None]
                    out = dimredgood_subspace_variance_accounted_for(data_mean, basis_vectors_1, basis_vectors_2)
                    from pythonlib.tools.expttools import writeDictToTxtFlattened
                    writeDictToTxtFlattened(out, f"{savedir}/VAF-subspace={subspace}.txt")
    
    ### Plot all subspaces
    for subspace in list_subspaces:

        savedir = f"/{SAVEDIR}/subspace={subspace}"
        os.makedirs(savedir, exist_ok=True)

        ### PLOT -- plot state space
        PAredu = dict_subspace_pa[subspace]
        state_space_targeted_pca_scalar_single_plot_(PAredu, LIST_VAR_VAROTHERS, LIST_DIMS, savedir)
    # for subspace in list_subspaces
    #     ### PLOT -- plot state space
    #     PAredu = dict_subspace_pa[subspace]
    #     Xredu = PAredu.X # (chans, trials, 1)
    #     dflab = PAredu.Xlabels["trials"]
    #     x = Xredu.squeeze().T # (trials, chans)

    #     for i, (var_effect, vars_others) in enumerate(LIST_VAR_VAROTHERS):
    #         if (var_effect in dflab.columns) and all([_var in dflab.columns for _var in vars_others]):

    #             ### Plot scalars
    #             # First, save counts
    #             fig = grouping_plot_n_samples_heatmap_var_vs_grpvar(dflab, var_effect, vars_others)
    #             savefig(fig, f"{savedir}/counts-{i}-var={var_effect}-varother={'|'.join(vars_others)}")

    #             # Second, plot scalars
    #             trajgood_plot_colorby_splotby_scalar_WRAPPER(x, dflab, var_effect, savedir,
    #                                                             vars_subplot=vars_others, list_dims=LIST_DIMS,
    #                                                             overlay_mean_orig=True)
    #             plt.close("all")
        
    return dict_subspace_pa, dict_subspace_axes_orig, dict_subspace_axes_normed, dfcoeff, PA

def state_space_targeted_pca_scalar_single_one_var_mult_axes(PA, twind_scal, variables_cont, variables_cat, 
                                                             var_subspace, npcs_keep, 
                                                             LIST_VAR_VAROTHERS, LIST_DIMS, SAVEDIR, 
                                                             just_extract_paredu=False,
                                                             savedir_pca_subspaces=None, 
                                                             tbin_dur = 0.2, tbin_slide = 0.1,
                                                             inds_trials_pa_train=None, inds_trials_pa_test=None,
                                                             skip_dim_redu=False,
                                                             do_vars_remove=False, vars_remove=None,
                                                             npcs_keep_force=50):
    """
    [GOOD], do targeted PCA for a single PA, along with state space plots.
    PARAMS:
    - variables: list of variables to regress out
    - variables_is_cat: list of bools, whether each variable is categorical. Note: if False, then will be treated as continuous variable
    list_subspaces: list of tuples, each tuple is a subspace to project onto. usually tuples are of length 2, holding strings 
    referring to the variable names.
    - var_subspace, either str or list of str
    - LIST_VAR_VAROTHERS, LIST_DIMS, SAVEDIR, for plotting state space.

    RETURNS:
    - pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, PAscalTest
    (Note, could be None if there isn't enough data or variaiotn in data to get targeted PCs)
    """
    # from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    # from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_heatmap_var_vs_grpvar
    # from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
    # from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
    # from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion
    # from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER
    from neuralmonkey.analyses.state_space_good import dimredgood_subspace_variance_accounted_for

    variables = variables_cont + variables_cat
    variables_is_cat = [False for _ in range(len(variables_cont))] + [True for _ in range(len(variables_cat))]

    # tbin_dur = 0.2
    # tbin_slide = 0.1
    # npcs_keep_force = 50
    normalization = "orthonormal"
    PLOT_COEFF_HEATMAP = SAVEDIR is not None
    
    ### Convert to scalat
    if skip_dim_redu:
        # Assume you inputed it
        PAscal = PA.copy()
        assert PAscal.X.shape[2]==1
    else:
        
        if False:
            # Get scalars
            PA = PA.slice_by_dim_values_wrapper("times", twind_scal).agg_wrapper("times")
        else:
            # Expand channels to (chans X time bins)
            if False: # old
                pca_reduce = True
                _, PAscal, _, _, _= PA.dataextract_state_space_decode_flex(twind_scal, tbin_dur, tbin_slide, "trials_x_chanstimes",
                                                                    pca_reduce=pca_reduce, npcs_keep_force=npcs_keep_force)
            else:
                # Identical, and is cleaner code
                savedir_pca = f"{SAVEDIR}/pca"
                os.makedirs(savedir_pca, exist_ok=True)
                _, PAscal = PA.dataextract_dimred_wrapper("scal", "pca", savedir_pca, twind_scal, tbin_dur, tbin_slide,
                                            npcs_keep_force)            

    ### Compute regression -- get coefficients
    savedir_coeff_heatmap = f"{SAVEDIR}"
    if True:
        # This is the new version that works regardless of whether you want to subtract a variable
        # then subtract a variable's coefficients
        pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, PAscalTest, _, _ = PAscal.dataextract_subspace_targeted_pca_wrapper(
                                                    variables_cont, variables_cat, vars_remove,
                                                    var_subspace, npcs_keep, normalization,
                                                    PLOT_COEFF_HEATMAP, PRINT=False,
                                                    savedir_coeff_heatmap=savedir_coeff_heatmap,
                                                    savedir_pca_subspaces=savedir_pca_subspaces,
                                                    inds_trials_pa_train=inds_trials_pa_train, inds_trials_pa_test=inds_trials_pa_test,
                                                    demean=True)
    else:
        # Old version -- works, but does not regress out variables.
        pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, PAscalTest = PAscal.dataextract_subspace_targeted_pca_one_var_mult_axes(
                                                    variables, variables_is_cat, var_subspace, npcs_keep, 
                                                    demean=True, 
                                                    normalization=normalization, plot_orthonormalization=False, 
                                                    PLOT_COEFF_HEATMAP=PLOT_COEFF_HEATMAP, PRINT=False,
                                                    savedir_coeff_heatmap=savedir_coeff_heatmap,
                                                    savedir_pca_subspaces=savedir_pca_subspaces,
                                                    inds_trials_pa_train=inds_trials_pa_train, inds_trials_pa_test=inds_trials_pa_test)

    # Skip plots, just get data
    if just_extract_paredu:
        return pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, PAscalTest
    
    ### Compute subspace angles
    if False: # Might work, need to check that it works.
        for subspace in list_subspaces:

            savedir = f"/{SAVEDIR}/subspace={subspace}"
            os.makedirs(savedir, exist_ok=True)

            ### Get VAF between each axis of subspace
            # - first, get meaned data.
            # data_trials = x.T # (chans, trials)
            subspace_axes = dict_subspace_axes_orig[subspace]
            assert len(subspace)==subspace_axes.shape[1]
            min_n_trials_in_lev = 3
            pa_mean = PA.slice_and_agg_wrapper("trials", variables, min_n_trials_in_lev=min_n_trials_in_lev)
            data_mean = pa_mean.X.squeeze() # (nchans, nconditions)

            naxes = subspace_axes.shape[1]
            for i in range(naxes):
                for j in range(naxes):
                    if j>i:
                        basis_vectors_1 = subspace_axes[:,i][:, None]
                        basis_vectors_2 = subspace_axes[:,j][:, None]
                        out = dimredgood_subspace_variance_accounted_for(data_mean, basis_vectors_1, basis_vectors_2)
                        from pythonlib.tools.expttools import writeDictToTxtFlattened
                        writeDictToTxtFlattened(out, f"{savedir}/VAF-subspace={subspace}.txt")
        
    ### Plot all subspaces
    if LIST_VAR_VAROTHERS is not None and len(LIST_VAR_VAROTHERS)>0 and pa_subspace is not None:
        if SAVEDIR is not None:
            savedir = f"/{SAVEDIR}/subspace={var_subspace}"
            os.makedirs(savedir, exist_ok=True)

            if len(pa_subspace.X)>0:
                ### PLOT -- plot state space
                state_space_targeted_pca_scalar_single_plot_(pa_subspace, LIST_VAR_VAROTHERS, LIST_DIMS, savedir)
            else:
                fig, ax = plt.subplots()
                ax.set_title("pa_subspace.X is empty!!")
                savefig(fig, f"{savedir}/no_data.pdf")
        
    return pa_subspace, subspace_axes_orig, subspace_axes_normed, dfcoeff, PAscalTest


def state_space_targeted_pca_scalar_single_plot_(pa_subspace, LIST_VAR_VAROTHERS, LIST_DIMS, savedir):
    """
    [GOOD], do targeted PCA for a single PA, along with state space plots.
    PARAMS:
    - variables: list of variables to regress out
    - variables_is_cat: list of bools, whether each variable is categorical. This doesnt affect the regression, so your data
    needs to be either cat or ordinal.
    list_subspaces: list of tuples, each tuple is a subspace to project onto. usually tuples are of length 2, holding strings 
    referring to the variable names.
    - LIST_VAR_VAROTHERS, LIST_DIMS, SAVEDIR, for plotting state space.
    """
    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_heatmap_var_vs_grpvar
    from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
    from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER
    from neuralmonkey.analyses.state_space_good import dimredgood_subspace_variance_accounted_for

    ### Plot all subspaces
    Xredu = pa_subspace.X # (chans, trials, 1)
    dflab = pa_subspace.Xlabels["trials"]
    x = Xredu.squeeze().T # (trials, chans)

    for i, (var_effect, vars_others) in enumerate(LIST_VAR_VAROTHERS):
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

def targeted_pca_combined_v2_good(DFallpa, SAVEDIR_ANALYSIS, LIST_VAR_VAROTHERS_SS, twind_scal_force=None,
                                  DEBUG=False): 
    """
    [GOOD wrapper] Pipeline to do this:
    - project to targeted PCA
    - Make state space (scalar) plots
    - Good computation of angles and effects, using euclidian distnace and averaged vectors, not regression.
    PARAMS:
    - var_effect, MUST be ordinal
    """
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar

    if DEBUG:
        only_bregions = ["M1", "vlPFC"]
        nsplits_ord_regr = 2
        LIST_DIMS = [(0,1)]
        # plot_regr = False
    else:
        only_bregions = None
        nsplits_ord_regr = 10
        LIST_DIMS = [(0,1), (2,3)]
        # plot_regr = False

    min_levs_per_levother = 2
    prune_levs_min_n_trials = 3
    npcs_keep_force = 60

    # TODO: Determine this for each expt (based on whether skips any shape, and whether varies in num strokes.)
    LIST_VAR_VAROTHERS_REGR = [
        ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1", "epoch", "FEAT_num_strokes_beh"]),
        ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "epoch", "FEAT_num_strokes_beh"]),
        # ("FEAT_num_strokes_beh", ["epoch", "seqc_0_loc", "seqc_0_shape"]),
    ]

    for _, row in DFallpa.iterrows():
        PA = row["pa"].copy()
        bregion = row["bregion"]
        event = row["event"]
        
        if (only_bregions is not None) and (bregion not in only_bregions):
            continue

        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{event}-{bregion}"
        os.makedirs(SAVEDIR, exist_ok=True)

        if twind_scal_force is None:
            twind_scal = map_event_to_twind[event]
        else:
            twind_scal = twind_scal_force

        ### General preprocessing
        grp_vars_prune_enough_trials = ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1", "syntax_slot_2", 
                                        "epoch", "FEAT_num_strokes_beh", "task_kind"]
        savedir = f"{SAVEDIR}/preprocess"
        os.makedirs(savedir, exist_ok=True)
        PA = preprocess_pa(PA, grp_vars_prune_enough_trials, prune_levs_min_n_trials, savedir)

        ### Dim reductions, AND scalar state space plots
        variables_cont = []
        variables_cat = ["epoch", "FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1", "syntax_slot_2"]
        # var_subspace = "syntax_slot_0"
        var_subspace = variables_cat
        # variables_cat = ["epoch", "FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape"]
        # var_subspace = "seqc_0_loc"
        npcs_keep = 6
        tbin_dur = 0.2
        tbin_slide = 0.1
        pa_subspace, _, _, _, _ = state_space_targeted_pca_scalar_single_one_var_mult_axes(
                                PA, twind_scal, variables_cont, variables_cat, 
                                var_subspace, npcs_keep, 
                                LIST_VAR_VAROTHERS_SS, LIST_DIMS, SAVEDIR, 
                                savedir_pca_subspaces=SAVEDIR, 
                                tbin_dur = tbin_dur, tbin_slide = tbin_slide,
                                npcs_keep_force=npcs_keep_force)
        ### Save neural data
        import pickle
        path = f"{SAVEDIR}/pa_subspace.pkl"
        with open(path, "wb") as f:
            pickle.dump(pa_subspace, f)

        ### Ordinal regression (including generalization)
        savedir = f"{SAVEDIR}/ordinal_regression"
        os.makedirs(savedir, exist_ok=True)
        from neuralmonkey.scripts.analy_syntax_good_eucl_trial import ordinalregress_1_compute
        from neuralmonkey.scripts.analy_syntax_good_eucl_trial import ordinalregress_2_regr_coeff_pairs
        # Extract
        DFCROSS, DFWITHIN = ordinalregress_1_compute(pa_subspace, LIST_VAR_VAROTHERS_REGR, savedir, nsplits=nsplits_ord_regr,
                                apply_kernel = False,  plot_indiv=False, plot_summary=True)
        for _df in [DFCROSS, DFWITHIN]:
            _df["bregion"] = bregion
            _df["event"] = event
            _df["var_subspace"] = [var_subspace for _ in range(len(_df))]

        ### Save
        pd.to_pickle(DFCROSS, f"{SAVEDIR}/DFCROSS.pkl")
        pd.to_pickle(DFWITHIN, f"{SAVEDIR}/DFWITHIN.pkl")

        # Compare angles of regression coefficients across all conditions.
        ordinalregress_2_regr_coeff_pairs(DFWITHIN, savedir=savedir)

        ### EUCLIDEAN + ANGLES
        list_dfangle = []
        list_dfdist = []
        for i_var, (var_effect, vars_others) in enumerate(LIST_VAR_VAROTHERS_REGR):
            # var_effect = LIST_VAR_VAROTHERS_REGR[1][0]
            # vars_others = LIST_VAR_VAROTHERS_REGR[1][1]

            ### (1) Prune data and vars_others
            # from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
            # extract_with_levels_of_conjunction_vars_helper(dflab, var=var_effect, vars_others=vars_others, 
            #                                        n_min_per_lev=prune_levs_min_n_trials, lenient_allow_data_if_has_n_levels=3)        
            pa_subspace_this, vars_others = pa_subspace.slice_prune_dflab_and_vars_others(var_effect, vars_others, 
                                                        prune_levs_min_n_trials, min_levs_per_levother)


            ### (2) Compute euclidian distances
            from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar
            savedir = f"{SAVEDIR}/euclid-{i_var}-var={var_effect}-varothers={vars_others}"
            os.makedirs(savedir, exist_ok=True)
            dfdist, _ = timevarying_compute_fast_to_scalar(pa_subspace_this, [var_effect] + vars_others, 
                                                            plot_conjunctions_savedir=savedir,
                                                            prune_levs_min_n_trials=2,
                                                            get_only_one_direction=False)

            ### (3) Get angles between all conditions
            from neuralmonkey.analyses.euclidian_distance import compute_angle_between_conditions
            dfangle = compute_angle_between_conditions(pa_subspace_this, dfdist, var_effect, vars_others)
            # Merge with dfdist
            assert np.all(dfangle["labels_1"] == dfdist["labels_1"])
            assert np.all(dfangle["labels_2"] == dfdist["labels_2"])
            dfdist["theta"] = dfangle["theta"]
            dfdist["norm"] = dfangle["norm"]
            # dfdist["vector"] = dfangle["vector"]  

            from neuralmonkey.analyses.euclidian_distance import dfdist_variables_generate_constrast_strings, dfdist_variables_generate_var_same
            # var_same = dfdist_variables_generate_var_same([var_effect] + vars_others)
            # var_same_val = dfdist_variables_generate_constrast_strings([var_effect] + vars_others, contrasts_diff=[var_effect], contrasts_either=[])[0]
            from neuralmonkey.analyses.euclidian_distance import compute_average_angle_between_pairs_of_levels_of_vareffect
            dfanglemean = compute_average_angle_between_pairs_of_levels_of_vareffect(dfdist, var_effect)

            ### COLLECT
            for _df in [dfdist, dfanglemean]:
                _df["bregion"] = bregion
                _df["event"] = event
                _df["var_subspace"] = [var_subspace for _ in range(len(_df))]
                _df["var_idx"] = i_var
                _df["var_effect"] = var_effect
                _df["vars_others"] = [tuple(vars_others) for _ in range(len(_df))]
            list_dfdist.append(dfdist)
            list_dfangle.append(dfanglemean)    

        DFANGLE = pd.concat(list_dfangle).reset_index(drop=True)
        DFDIST = pd.concat(list_dfdist).reset_index(drop=True)
        
        # save
        DFDIST.to_pickle(f"{SAVEDIR}/DFDIST.pkl")
        DFANGLE.to_pickle(f"{SAVEDIR}/DFANGLE.pkl")

def targeted_pca_euclidian_dist_angles(DFallpa, SAVEDIR_ANALYSIS, 
                                       variables, variables_is_cat, list_subspaces, LIST_VAR_VAROTHERS_SS, # For dim reduction and plotting state space
                                       subspace_tuple, LIST_VAR_VAROTHERS_ANGLES,
                                       subspace_filtdict =None, twind_scal_force=None): # For computing angles.
    """
    [GOOD wrapper] Pipeline to do this:
    - project to targeted PCA
    - Make state space (scalar) plots
    - Good computation of angles and effects, using euclidian distnace and averaged vectors, not regression.
    PARAMS:
    - var_effect, MUST be ordinal
    """
    from neuralmonkey.analyses.euclidian_distance import timevarying_compute_fast_to_scalar
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from pythonlib.tools.vectools import cart_to_polar
    from pythonlib.tools.vectools import average_angle, get_vector_from_angle, average_vectors_wrapper

    assert subspace_filtdict is None, "not yet coded"

    min_levs_per_levother = 2
    prune_levs_min_n_trials = 4

    list_dfangle = []
    for _, row in DFallpa.iterrows():
        PA = row["pa"].copy()
        bregion = row["bregion"]
        event = row["event"]
        
        if twind_scal_force is None:
            twind_scal = map_event_to_twind[event]
        else:
            twind_scal = twind_scal_force

        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{event}-{bregion}"
        os.makedirs(SAVEDIR, exist_ok=True)

        ### Dim reductions, AND scalar state space plots
        # if subspace_filtdict:
        #     # Then take subset of data for computing the state space 
        just_extract_paredu = False
        LIST_DIMS = [(0,1)]
        # dict_subspace_pa, _, _, _, _ = state_space_targeted_pca_scalar_single(
        #                         PA, twind_scal, variables, variables_is_cat, list_subspaces, 
        #                         LIST_VAR_VAROTHERS_SS, LIST_DIMS, SAVEDIR, just_extract_paredu=just_extract_paredu)
        dict_subspace_pa, _, _, _, _ = state_space_targeted_pca_scalar_single_one_axis_per_var(
                                PA, twind_scal, variables, variables_is_cat, list_subspaces, 
                                LIST_VAR_VAROTHERS_SS, LIST_DIMS, SAVEDIR, just_extract_paredu=just_extract_paredu)

        ### Esing euclidian distance to score axes -- angles, etc
        # subspace = ('syntax_slot_0', 'syntax_slot_1')
        PAredu = dict_subspace_pa[subspace_tuple]

        for i_var, (var_effect, vars_others) in enumerate(LIST_VAR_VAROTHERS_ANGLES):
            # var_effect = LIST_VAR_VAROTHERS_REGR[1][0]
            # vars_others = LIST_VAR_VAROTHERS_REGR[1][1]

            ### (1) Prune data and vars_others
            # from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
            # extract_with_levels_of_conjunction_vars_helper(dflab, var=var_effect, vars_others=vars_others, 
            #                                        n_min_per_lev=prune_levs_min_n_trials, lenient_allow_data_if_has_n_levels=3)        
            pa, vars_others = PAredu.slice_prune_dflab_and_vars_others(var_effect, vars_others, 
                                                        prune_levs_min_n_trials, min_levs_per_levother)



            ### (2) Compute euclidian distances
            #  make a conjunctive variable.
            pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], vars_others, "_var_other")
            savedir = f"{SAVEDIR}/{i_var}-var={var_effect}-varothers={vars_others}-eucldist"
            os.makedirs(savedir, exist_ok=True)
            dfdist, _ = timevarying_compute_fast_to_scalar(pa, [var_effect, '_var_other'], 
                                                            plot_conjunctions_savedir=savedir,
                                                            prune_levs_min_n_trials=prune_levs_min_n_trials,
                                                            get_only_one_direction=False)

            ### (3) Get angles between all conditions
            from neuralmonkey.analyses.euclidian_distance import compute_angle_between_conditions
            dfangle = compute_angle_between_conditions(pa, dfdist, var_effect, vars_others)
            assert np.all(dfangle["labels_1"] == dfdist["labels_1"])
            assert np.all(dfangle["labels_2"] == dfdist["labels_2"])
            dfdist["theta"] = dfangle["theta"]
            dfdist["norm"] = dfangle["norm"]

            # dflab = pa.Xlabels["trials"]
            # res = []
            # for _, row in dfdist.iterrows():
            #     if row["labels_1"] == row["labels_2"]:
            #         theta = 0.
            #         norm = 0.
            #     else:
            #         inds1 = dflab[(dflab[var_effect] == row["labels_1"][0]) & (dflab["_var_other"] == row["labels_1"][1])].index.tolist()
            #         inds2 = dflab[(dflab[var_effect] == row["labels_2"][0]) & (dflab["_var_other"] == row["labels_2"][1])].index.tolist()

            #         x1 = pa.X[:, inds1].squeeze()
            #         x2 = pa.X[:, inds2].squeeze()

            #         x1_mean = np.mean(x1, axis=1)            
            #         x2_mean = np.mean(x2, axis=1)            

            #         vec = x2_mean - x1_mean
            #         theta, norm = cart_to_polar(vec[0], vec[1])

            #     # Append
            #     res.append({
            #         "labels_1":row["labels_1"],
            #         "labels_2":row["labels_2"],
            #         "theta":theta,
            #         "norm":norm,
            #         # "vector":vec,
            #     })
            # # Merge this with dfdist
            # dfangle = pd.DataFrame(res)
            # assert np.all(dfangle["labels_1"] == dfdist["labels_1"])
            # assert np.all(dfangle["labels_2"] == dfdist["labels_2"])
            # dfdist["theta"] = dfangle["theta"]
            # dfdist["norm"] = dfangle["norm"]
            # # dfdist["vector"] = dfangle["vector"]

            ### (4) For each var_other, get vector average
            from neuralmonkey.analyses.euclidian_distance import compute_average_angle_between_pairs_of_levels_of_vareffect
            dfangle = compute_average_angle_between_pairs_of_levels_of_vareffect(dfdist, var_effect)

            # # # first, only consider "same var other" (i/e. get angles within a var_iother)
            # dfdist_same = dfdist[dfdist["_var_other_same"]].reset_index(drop=True)
            # grpdict = grouping_append_and_return_inner_items_good(dfdist_same, ["_var_other_1"])
            # res = []
            # for grp, inds in grpdict.items():
            #     print(grp, inds)

            #     # Collect all the angles between adjacent values of var_effect
            #     dftmp = dfdist_same.iloc[inds]
            #     levs_exist = dftmp[f"{var_effect}_1"].unique()

            #     if len(levs_exist)>=min_levs_per_levother:
            #         print("levs_exist:", levs_exist)
            #         assert len(levs_exist)>1

            #         # Get adjacent values of var_effect
            #         tmp = []
            #         for lev1, lev2 in zip(levs_exist[:-1], levs_exist[1:]):
            #             this = dftmp[(dftmp[f"{var_effect}_1"] == lev1) & (dftmp[f"{var_effect}_2"] == lev2)]
            #             assert len(this)==1
            #             tmp.append(this)
            #         dftmp = pd.concat(tmp)
                    
            #         # Compute average vector, different possible methods
            #         for var_vector_length in ["dist_norm", "dist_yue_diff"]:
            #             angles = dftmp["theta"].values
            #             weights = dftmp[var_vector_length].values
            #             vectors_arr = np.stack([w * get_vector_from_angle(a) for a, w in zip(angles, weights)])
                        
            #             for length_method in ["sum", "dot"]:
            #                 # More general
            #                 angle_mean, norm_mean = average_vectors_wrapper(vectors_arr, length_method=length_method)

            #                 res.append({
            #                     "var_other":grp[0],
            #                     "levs_exist":levs_exist,
            #                     "angles":angles,
            #                     "weights":weights,
            #                     "angle_mean":angle_mean,
            #                     "norm_mean":norm_mean,
            #                     "var_vector_length":var_vector_length,
            #                     "length_method":length_method
            #                 })
            # dfangle = pd.DataFrame(res)

            ### COLLECT
            dfangle["var_idx"] = i_var
            dfangle["var_effect"] = var_effect
            dfangle["vars_others"] = [tuple(vars_others) for _ in range(len(dfangle))]
            dfangle["subspace"] = [subspace_tuple for _ in range(len(dfangle))]
            dfangle["bregion"] = bregion
            dfangle["event"] = event

            list_dfangle.append(dfangle)    

    DFANGLE = pd.concat(list_dfangle).reset_index(drop=True)

    # save
    DFANGLE.to_pickle(f"{SAVEDIR_ANALYSIS}/DFANGLE.pkl")
    
    return DFANGLE

def targeted_pca_euclidian_dist_angles_compute_dfdot(DFANGLE_ALL, var_vector_length, length_method, min_levs_exist, 
                                                     SAVEDIR=None):
    """
    A metric to summarize extent to which angle encoding var_effect is aligned across vars_others. 

    Get consistency of vector across vars_others, where consistency is using pairwise dot product and where the 
    vector encodes the average efect of <var_effect>, where the "average" is defined by (var_vector_length, length_method). 
    
    # var_vector_length = "dist_norm"
    var_vector_length = "dist_yue_diff"
    # length_method = "dot"
    length_method = "sum"
    """
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    if "date" in DFANGLE_ALL:
        assert len(DFANGLE_ALL["date"].unique())==1, "this is meant for a single (animal, date)"
    
    # Prune to what will plot here
    DFANGLE_ALL["levs_exist_n"] = [len(levs_exist) for levs_exist in DFANGLE_ALL["levs_exist"]]

    DFANGLE = DFANGLE_ALL[
        (DFANGLE_ALL["var_vector_length"] == var_vector_length) & 
        (DFANGLE_ALL["length_method"] == length_method) &
        (DFANGLE_ALL["levs_exist_n"] >= min_levs_exist)].reset_index(drop=True)

    if len(DFANGLE)==0:
        # DFANGLE["levs_exist_n"].hist()
        return None
    
    assert len(DFANGLE)>0

    ### (4) Summarize consistency of angle across levels of ovar (dot product of vectors)
    from pythonlib.tools.vectools import get_vector_from_angle

    # Compute dot project between all (othervars), including both within and across var_effect
    # Get dot product between all pairs of vectors
    assert not np.any(DFANGLE["norm_mean"].isna())
    DFANGLE["vector_mean"] = [l * get_vector_from_angle(a) for a, l in zip(DFANGLE["angle_mean"], DFANGLE["norm_mean"])]
    res_dot = []
    grpdict = grouping_append_and_return_inner_items_good(DFANGLE, ["bregion", "event", "var_effect", "var_other"])
    for i, (grp1, inds1) in enumerate(grpdict.items()):
        for j, (grp2, inds2) in enumerate(grpdict.items()):
            if grp1[0]==grp2[0]: # same bregion
                bregion =grp1[0]
                if grp1[1]==grp2[1]: # same event
                    event = grp1[1]
                    if j>i:
                        # The vector average vector (x,y) encoding <var_effect> for this <var_other>
                        tmp1 = DFANGLE.iloc[inds1]
                        tmp2 = DFANGLE.iloc[inds2]
                        assert len(tmp1)==1
                        assert len(tmp2)==1
                        vec1 = tmp1["vector_mean"].values[0]
                        vec2 = tmp2["vector_mean"].values[0]

                        # Get dot product
                        res_dot.append({
                            # "dot_product_mean":np.mean(dot_products),
                            "var_effect_1":grp1[2],
                            "var_effect_2":grp2[2],
                            "var_other_1":grp1[3],
                            "var_other_2":grp2[3],
                            "dot_product":np.dot(vec1, vec2),
                            "bregion":bregion,
                            "event":event,
                            "var_vector_length":var_vector_length,
                            "length_method":length_method,
                            "min_levs_exist":min_levs_exist})
    if len(res_dot)>0:
        DF_DOT = pd.DataFrame(res_dot)
        DF_DOT = append_col_with_grp_index(DF_DOT, ["var_effect_1", "var_effect_2"], "var_effect_12")

        if SAVEDIR is not None:
            DF_DOT.to_csv(f"{SAVEDIR}/DF_DOT.csv")

            # Plots
            fig = sns.catplot(data=DF_DOT, x="bregion", y="dot_product", hue="var_effect_12", jitter=True, alpha=0.25, aspect=1.5)
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{SAVEDIR}/dot_across_ovar-catplot-1.pdf")

            fig = sns.catplot(data=DF_DOT, x="bregion", y="dot_product", hue="var_effect_12", kind="bar", aspect=1.5)
            rotateLabel(fig)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{SAVEDIR}/dot_across_ovar-catplot-2.pdf")

            ### 
            for var_effect_12 in DF_DOT["var_effect_12"].unique():
                dfdot = DF_DOT[DF_DOT["var_effect_12"] == var_effect_12].reset_index(drop=True)
                fig, _ = plot_subplots_heatmap(dfdot, "var_other_1", "var_other_2", "dot_product", "bregion", True, 
                                            share_zlim=True, annotate_heatmap=False)
                savefig(fig, f"{SAVEDIR}/dot_across_ovar-heatmap-vareffect12={var_effect_12}.pdf")
    else:
        DF_DOT = None
        
    plt.close("all")

    return DF_DOT

def targeted_pca_euclidian_dist_angles_plots(DFANGLE_ALL, var_vector_length, length_method, min_levs_exist, SAVEDIR):
    """
    Plot results from targeted_pca_euclidian_dist_angles, for a single (animal, date)
    # var_vector_length = "dist_norm"
    var_vector_length = "dist_yue_diff"
    # length_method = "dot"
    length_method = "sum"
    """
    from pythonlib.tools.snstools import rotateLabel

    if "date" in DFANGLE_ALL:
        assert len(DFANGLE_ALL["date"].unique())==1, "this is meant for a single (animal, date)"
    
    # Prune to what will plot here
    DFANGLE_ALL["levs_exist_n"] = [len(levs_exist) for levs_exist in DFANGLE_ALL["levs_exist"]]

    DFANGLE = DFANGLE_ALL[
        (DFANGLE_ALL["var_vector_length"] == var_vector_length) & 
        (DFANGLE_ALL["length_method"] == length_method) &
        (DFANGLE_ALL["levs_exist_n"] >= min_levs_exist)].reset_index(drop=True)

    if len(DFANGLE)==0:
        # DFANGLE["levs_exist_n"].hist()
        return 
    assert len(DFANGLE)>0

    ####################### PLOTS of angles across all levels of var_other (i.e., the rawest possible)
    ### (1) Catplot
    try:
        fig = sns.catplot(data=DFANGLE, x="bregion", y="norm_mean", hue="var_effect", jitter=True, alpha=0.7)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        savefig(fig, f"{SAVEDIR}/allvectors-catplot-1.pdf")
    except Exception as err:
        print(DFANGLE)
        print(DFANGLE.columns)
        raise err

    fig = sns.catplot(data=DFANGLE, x="bregion", y="norm_mean", hue="var_effect", kind="bar")
    rotateLabel(fig)
    savefig(fig, f"{SAVEDIR}/allvectors-catplot-2.pdf")
    plt.close("all")

    ### (2) Polar plots
    grpdict = grouping_append_and_return_inner_items_good(DFANGLE, ["bregion", "event"])
    ncols = 4
    nrows = int((np.ceil(len(grpdict)+1)/ncols))

    list_var_effect = sorted(DFANGLE["var_effect"].unique())

    from pythonlib.tools.plottools import color_make_map_discrete_labels
    map_vareffect_to_color = color_make_map_discrete_labels(list_var_effect)[0]

    SIZE = 3
    # fig, axes = plt.subplots(nrows, ncols, subplot_kw={'polar': True}, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)
    fig, axes = plt.subplots(nrows, ncols, subplot_kw={'polar': True}, figsize=(ncols*SIZE, nrows*SIZE), squeeze=False)
    for i, (grp, inds) in enumerate(grpdict.items()):
        ax = axes.flatten()[i]
        ax.set_title(grp, fontsize=8)

        for var_effect in list_var_effect:
                
            dftmp = DFANGLE.iloc[inds]
            dfangle = dftmp[(dftmp["var_effect"]==var_effect)].reset_index(drop=True)

            # Rose plot of all the angles
            angles = dfangle["angle_mean"]
            lengths = dfangle["norm_mean"]

            alpha = 0.7
            from pythonlib.tools.plottools import rose_plot
            # fig, ax = plt.subplots(subplot_kw={'polar': True})
            color = map_vareffect_to_color[var_effect]
            for a, l in zip(angles, lengths):
                ax.plot([0, a], [0, l], "-", color=color, alpha=alpha)
            ax.plot(angles, lengths, "o", color=color, alpha=alpha)
            # ax.stem(angles, lengths)
            # rose_plot(ax, angles)
    from pythonlib.tools.plottools import legend_add_manual
    ax = axes.flatten()[i]
    legend_add_manual(ax, map_vareffect_to_color.keys(), map_vareffect_to_color.values())
    savefig(fig, f"{SAVEDIR}/allvectors-polarplots.pdf")

    ### (3) Print all data
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    DFANGLE_STR = stringify_values(DFANGLE)
    DFANGLE_STR["levs_exist"] = [tuple(x) for x in DFANGLE_STR["levs_exist"]]
    DFANGLE_STR["angles"] = [tuple(x) for x in DFANGLE_STR["angles"]]
    DFANGLE_STR["weights"] = [tuple(x) for x in DFANGLE_STR["weights"]]

    DFANGLE_STR.to_csv(f"{SAVEDIR}/DFANGLE.csv")
    # grouping_print_n_samples(DFANGLE_STR, ["event", "bregion", "var_vector_length", "length_method", "var_effect", "var_other", "levs_exist", "angles", "weights", "angle_mean", "norm_mean"])
    
    fig, _ = plot_subplots_heatmap(DFANGLE, "var_other", "var_effect", "norm_mean", "bregion", share_zlim=True, diverge=True)
    savefig(fig, f"{SAVEDIR}/allvectors-heatmap-vector_length.pdf")

    fig, _ = plot_subplots_heatmap(DFANGLE, "var_other", "var_effect", "angle_mean", "bregion", share_zlim=True, diverge=True)
    savefig(fig, f"{SAVEDIR}/allvectors-heatmap-vector_angle.pdf")

    plt.close("all")

    ### (4) Summarize consistency of angle across levels of ovar (dot product of vectors)
    from pythonlib.tools.vectools import get_vector_from_angle

    # Compute dot project between all (othervars), including both within and across var_effect
    # Get dot product between all pairs of vectors
    DFANGLE["vector_mean"] = [l * get_vector_from_angle(a) for a, l in zip(DFANGLE["angle_mean"], DFANGLE["norm_mean"])]
    res_dot = []
    grpdict = grouping_append_and_return_inner_items_good(DFANGLE, ["bregion", "event", "var_effect", "var_other"])
    for i, (grp1, inds1) in enumerate(grpdict.items()):
        for j, (grp2, inds2) in enumerate(grpdict.items()):
            if grp1[0]==grp2[0]: # same bregion
                bregion =grp1[0]
                if grp1[1]==grp2[1]: # same event
                    event = grp1[1]
                    if j>i:
                        tmp1 = DFANGLE.iloc[inds1]
                        tmp2 = DFANGLE.iloc[inds2]
                        assert len(tmp1)==1
                        assert len(tmp2)==1

                        vec1 = tmp1["vector_mean"].values[0]
                        vec2 = tmp2["vector_mean"].values[0]
                        
                        # Get dot product
                        res_dot.append({
                            # "dot_product_mean":np.mean(dot_products),
                            "var_effect_1":grp1[2],
                            "var_effect_2":grp2[2],
                            "var_other_1":grp1[3],
                            "var_other_2":grp2[3],
                            "dot_product":np.dot(vec1, vec2),
                            "bregion":bregion,
                            "event":event,
                        })
    if len(res_dot)>0:
        DF_DOT = pd.DataFrame(res_dot)
        DF_DOT = append_col_with_grp_index(DF_DOT, ["var_effect_1", "var_effect_2"], "var_effect_12")
        DF_DOT.to_csv(f"{SAVEDIR}/DF_DOT.csv")

        # Plots
        fig = sns.catplot(data=DF_DOT, x="bregion", y="dot_product", hue="var_effect_12", jitter=True, alpha=0.25, aspect=1.5)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        savefig(fig, f"{SAVEDIR}/dot_across_ovar-catplot-1.pdf")

        fig = sns.catplot(data=DF_DOT, x="bregion", y="dot_product", hue="var_effect_12", kind="bar", aspect=1.5)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)
        savefig(fig, f"{SAVEDIR}/dot_across_ovar-catplot-2.pdf")

        ### 
        for var_effect_12 in DF_DOT["var_effect_12"].unique():
            dfdot = DF_DOT[DF_DOT["var_effect_12"] == var_effect_12].reset_index(drop=True)
            fig, _ = plot_subplots_heatmap(dfdot, "var_other_1", "var_other_2", "dot_product", "bregion", True, 
                                        share_zlim=True, annotate_heatmap=False)
            savefig(fig, f"{SAVEDIR}/dot_across_ovar-heatmap-vareffect12={var_effect_12}.pdf")

    plt.close("all")

def ordinalregress_1_compute(pa_subspace, LIST_VAR_VAROTHERS_REGR, SAVEDIR, nsplits=10,
                             apply_kernel = False, plot_indiv=False, plot_summary=True):
    """
    Helper to do regression for multiple varialbes (one by one).
    This is helper that does multiple (var, vars_others) and concatenates the results.

    Also, does both within- and across- methods for train-test split.
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_state import kernel_ordinal_logistic_regression_wrapper

    n_min_per_level = 3
    list_do_grid_search = [True]

    list_dfcross = []
    list_dfwithin = []
    for i_var, (yvar, vars_grp) in enumerate(LIST_VAR_VAROTHERS_REGR):

        ### Prune the variables, based on what exists for this expt
        pa_subspace_this, vars_grp = pa_subspace.slice_prune_dflab_and_vars_others(yvar, vars_grp, 
                                            n_min_per_level, 2)

        savedir = f"{SAVEDIR}/ordinal_regress-yvar={yvar}=varsgrp={vars_grp}"
        os.makedirs(savedir, exist_ok=True)
        dfcross, dfwithin = kernel_ordinal_logistic_regression_wrapper(pa_subspace_this, yvar, vars_grp, savedir, 
                                                plot_test_data_projected=plot_indiv, nsplits=nsplits, PLOT=plot_summary, 
                                                do_rezero=False, apply_kernel=apply_kernel, list_do_grid_search=list_do_grid_search,
                                                n_min_per_level=n_min_per_level)

        if dfcross is None:
            assert dfwithin is None
            continue
            
        for _df in [dfcross, dfwithin]:
            _df["var_idx"] = i_var
            _df["yvar"] = yvar
            _df["var_effect"] = yvar
            _df["vars_grp"] = [tuple(vars_grp) for _ in range(len(_df))]
            _df["vars_others"] = [tuple(vars_grp) for _ in range(len(_df))]

        list_dfcross.append(dfcross)
        list_dfwithin.append(dfwithin)

        # Also plot?
        # if plot_summary:
        #     kernel_ordinal_logistic_regression_wrapper_plot(dfcross, dfwithin, vars_grp, savedir)

    DFCROSS = pd.concat(list_dfcross).reset_index(drop=True)
    DFWITHIN = pd.concat(list_dfwithin).reset_index(drop=True)

    if False: # This is now done within kernel_ordinal_logistic_regression_wrapper (becuase it can fail here)
        ### Some postprocessing
        # Compare the angles of the regression axes.
        list_coeff = []
        for _, row in DFWITHIN.iterrows():
            coeff = row["res"]["coeff"]
            list_coeff.append(coeff)
        DFWITHIN["coeff"] = list_coeff

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
        DFWITHIN = aggregGeneral(DFWITHIN, ["grp", "yvar", "vars_grp"], list(aggdict.keys()), aggmethod=aggdict)

    return DFCROSS, DFWITHIN

def ordinalregress_2_regr_coeff_pairs(DFWITHIN, savedir):
    """
    Compute pairwise comparison of regression coefficients, across all conditions
    """
    from pythonlib.tools.vectools import cosine_similarity

    ### (1) Get all pairwise correlations of regression coefficients
    res = []
    for i1, row1 in DFWITHIN.iterrows():
        for i2, row2 in DFWITHIN.iterrows():
            if i2>i1:
                coeff1 = row1["coeff"]
                coeff2 = row2["coeff"]
                
                # Cosine similarity
                sim_cos = cosine_similarity(coeff1, coeff2)

                # Dot product
                dot_prod = coeff1 @ coeff2

                # Correlation coefficient
                corr = np.corrcoef(coeff1, coeff2)[0, 1]

                res.append({
                    "labels_1":row1["grp"],
                    "yvar_1":row1["yvar"],
                    "vars_grp_1":row1["vars_grp"],

                    "labels_2":row2["grp"],
                    "yvar_2":row2["yvar"],
                    "vars_grp_2":row2["vars_grp"],
                    
                    "cosine_sim":sim_cos,
                    "dot_prod":dot_prod,
                    "corr_coeff":corr,
                })
    DF_REGR_PAIRS = pd.DataFrame(res)

    ### Postprocessing
    # Create a new label that is a higher-level label (label, yvar, grp), where label itself is a tuple
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from neuralmonkey.analyses.euclidian_distance import dfdist_extract_label_vars_specific, dfdist_summary_plots_wrapper
    vars_grp_meta = ["labels", "yvar", "vars_grp"]
    DF_REGR_PAIRS = append_col_with_grp_index(DF_REGR_PAIRS, [f"{v}_1" for v in vars_grp_meta], "metalabels_1", use_strings=False)
    DF_REGR_PAIRS = append_col_with_grp_index(DF_REGR_PAIRS, [f"{v}_2" for v in vars_grp_meta], "metalabels_2", use_strings=False)
    # Add the columns
    DF_REGR_PAIRS, colname_conj_same = dfdist_extract_label_vars_specific(DF_REGR_PAIRS, vars_grp_meta, True, 
                                                                        var1="metalabels_1", var2="metalabels_2")
    
    ### Plots
    for y in ["cosine_sim", "dot_prod", "corr_coeff"]:

        # Catplots
        fig = sns.catplot(data=DF_REGR_PAIRS, x=colname_conj_same, y=y, jitter=True, alpha=0.5)
        for ax in fig.axes.flatten():
            ax.axhline(0)
        savefig(fig, f"{savedir}/catplot-y={y}-1.pdf")

        fig = sns.catplot(data=DF_REGR_PAIRS, x=colname_conj_same, y=y, kind="boxen")
        for ax in fig.axes.flatten():
            ax.axhline(0)    
        savefig(fig, f"{savedir}/catplot-y={y}-2.pdf")

        fig = sns.catplot(data=DF_REGR_PAIRS, x=colname_conj_same, y=y, kind="bar")
        for ax in fig.axes.flatten():
            ax.axhline(0)        
        savefig(fig, f"{savedir}/catplot-y={y}-3.pdf")

        # Heatmap
        from neuralmonkey.analyses.euclidian_distance import dfdist_expand_convert_from_triangular_to_full
        from pythonlib.cluster.clustclass import generate_clustclass_from_flat_df
        DF_REGR_PAIRS_FULL = dfdist_expand_convert_from_triangular_to_full(DF_REGR_PAIRS, vars_grp_meta, 
                                                    var1="metalabels_1", var2="metalabels_2", PLOT=False, remove_diagonal=False)
        
        # Make heatmap plot
        Cl = generate_clustclass_from_flat_df(DF_REGR_PAIRS_FULL, "metalabels_1", "metalabels_2", 
                                            var_value=y, var_labels=vars_grp_meta,
                                            fake_the_diagonal=0)
        fig, ax = Cl.rsa_plot_heatmap(diverge=True, sort_order=(2, 1, 0))
        savefig(fig, f"{savedir}/heatmap-y={y}.pdf")

        if False: # Replaced by above
            ##### Older approach to get heatmap
            dat = np.stack(DFWITHIN["coeff"])
            corr_mat = np.corrcoef(dat)

            dat.shape
            dot_mat = dat@dat.T
            labels = [tuple(x) for x in DFWITHIN.loc[:, ["yvar", "grp"]].values.tolist()]

            from pythonlib.tools.snstools import heatmap_mat

            fig, ax = plt.subplots(figsize=(10, 10))
            heatmap_mat(dot_mat, ax, False, zlims=None, diverge=True, labels_row=labels, labels_col=labels)
        
        plt.close("all")

def state_space_plot_scalar_wrapper(DFallpa, SAVEDIR, LIST_VAR_VAROTHERS, LIST_DIMS):
    """
    Simply plot scalar state space, for each variable.
    """
    # # First, preprocess all pa
    # list_pa =[]
    # for PA in DFallpa["pa"]:
    #     pa = preprocess_pa(PA, animal, date, var_other, "/tmp", True, None, None, None, None, False)
    #     plt.close("all")
    #     list_pa.append(pa)    
    # DFallpa["pa"] = list_pa

    from neuralmonkey.scripts.analy_euclidian_chars_sp import params_subspace_projection
    from pythonlib.tools.pandastools import append_col_with_grp_index, grouping_plot_n_samples_heatmap_var_vs_grpvar
    from neuralmonkey.analyses.state_space_good import _trajgood_plot_colorby_scalar_BASE_GOOD
    from pythonlib.tools.plottools import share_axes_row_or_col_of_subplots
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_2dgrid_bregion

    ### PARAMS

    # (Project onto shape subspace)
    HACK = False
    # twind = (0.05, 1.0)

    # list_subspace_projection = ["umap"]
        
    ### Collect each PA (doing dim reduction)

    ### Plot, for each variable
    # savedir = f"{SAVEDIR}/var={var_effect}-varothers={'|'.join(vars_others)}"
    savedir = SAVEDIR

    for i, (var_effect, vars_others) in enumerate(LIST_VAR_VAROTHERS):
        
        ### PLOT

        if False:
            ### METHOD 1 -- 2d grid of (bregion vs var_other)
            # Problem: figures are too large.

            var_other = "_var_other"
            # Extract event to plot
            for event in DFallpa["event"].unique().tolist():
                # event = "03_samp"
                dfallpa = DFallpa[DFallpa["event"]==event].reset_index(drop=True)
                
                trajgood_plot_colorby_splotby_scalar_2dgrid_bregion(dfallpa, var_effect, var_other, savedir, pa_var = "pa_redu")
        else:
            ### METHOD 2 -- one figure per bregion. 
            from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER

            for _, row in DFallpa.iterrows():
                
                PAredu = row["pa_redu"]
                if PAredu is None:
                    print("Skipping, since PAredu is None")
                    continue

                bregion = row["bregion"]
                event = row["event"]

                Xredu = PAredu.X
                dflab = PAredu.Xlabels["trials"]
                x = Xredu.squeeze().T

                if (var_effect in dflab.columns) and all([_var in dflab.columns for _var in vars_others]):
                    ### Plot scalars
                    savedir_this = f"{savedir}/{event}-{bregion}"
                    os.makedirs(savedir_this, exist_ok=True)

                    # First, save counts
                    fig = grouping_plot_n_samples_heatmap_var_vs_grpvar(dflab, var_effect, vars_others)
                    savefig(fig, f"{savedir_this}/counts-{i}-var={var_effect}-varother={'|'.join(vars_others)}")

                    # Second, plot scalars
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(x, dflab, var_effect, savedir_this,
                                                                    vars_subplot=vars_others, list_dims=LIST_DIMS,
                                                                    overlay_mean_orig=True)
        
        plt.close("all")

def regression_wrapper_wrapper(DFallpa, LIST_VAR_VAROTHERS_REGR, SAVEDIR_ANALYSIS, ndims):
    """
    Compute rgression for each variable, and plot.
    This is wrapper of wrapper, beucase it calls the regression_wrapper, which it operates on multiple var_effect and vars_others.
    PARAMS:
    - LIST_VAR_VAROTHERS_REGR, list of tuples, each tuple is (var_effect, vars_others), where 
    vars_others is a list of variables to condition on. Will do regression (X is neural data, y is var_effect).
    """
    ### Regression plots
    for var_effect, vars_others in LIST_VAR_VAROTHERS_REGR:

        # first, skip this if any variable does not exist in data
        pa = DFallpa["pa"].values[0]
        dflab = pa.Xlabels["trials"]
        if var_effect not in dflab.columns:
            print("Skipping, becuase lacks this var_effect: ", var_effect)
            continue
        if not all([varo in dflab.columns for varo in vars_others]):
            print("Skipping, becuase lacks this var_other: ", [varo in dflab.columns for varo in vars_others])
            print("...vars:", vars_others)
            continue
        
        # (1) Perform and collect all regresion stuff.
        DFRES_ALL, DFDISTCOS_ALL, savedir = regression_wrapper(DFallpa, var_effect, vars_others, f"{SAVEDIR_ANALYSIS}/regression", 
                                                    ndims = ndims, PLOT_SCATTER_PRED_VS_ACTUAL = False)
        
        print("... Currently running: ", savedir)

        if DFRES_ALL is not None:
            # (2) Regression, within each grp
            # savedir = f"{SAVEDIR}/MULT_PLOTS"
            # os.makedirs(savedir, exist_ok=True)
            
            # Prune, based on data quality (not on r2)
            min_levs = 3
            min_r2_test = None
            min_ntot = 20
            min_nmin = 5
            DFRES_ALL_PRUNED, DFDISTCOS_ALL_PRUNED = regression_wrapper_prune(DFRES_ALL, DFDISTCOS_ALL, min_levs, min_r2_test, 
                                                                            min_ntot, min_nmin)
            if len(DFRES_ALL_PRUNED)>0:
                for savesuff, dfthis in zip(
                    ["all", "pruned"],
                    [DFRES_ALL, DFRES_ALL_PRUNED]):

                    savedirthis = f"{savedir}/plot_each_grp-{savesuff}"
                    os.makedirs(savedirthis, exist_ok=True)
                    regression_wrapper_plot_each_grp(dfthis, savedirthis)
                    plt.close("all")

        if (DFRES_ALL is not None) and not all(DFRES_ALL["same_grp"]==True):
            # (NOTE: if is all, then skip, since this means there is only one level of var_effect (for each var_other), and this means cannot)
            # (4) Cosine similarity plots
            # Prune, based on good regression scores.
            min_r2_test = 0.1
            DFRES_ALL_PRUNED, DFDISTCOS_ALL_PRUNED = regression_wrapper_prune(DFRES_ALL, DFDISTCOS_ALL, min_levs, min_r2_test, 
                                                                            min_ntot, min_nmin)

            if len(DFRES_ALL_PRUNED)>0:
                for savesuff, dfthis in zip(
                    ["all", "pruned"],
                    [DFDISTCOS_ALL, DFDISTCOS_ALL_PRUNED]):

                    savedirthis = f"{savedir}/plot_cosine_sim-{savesuff}"
                    os.makedirs(savedirthis, exist_ok=True)
                    regression_wrapper_plot_cosine_sim(dfthis, savedirthis)
                    plt.close("all")


def regression_wrapper(DFallpa, var_effect, vars_others, SAVEDIR_BASE, ndims = 4, PLOT_SCATTER_PRED_VS_ACTUAL = False):
    """
    Do regression for a single var_effect and vars_others. See regression_wrapper_wrapper for more details.
    PARAMS:
        var_effect = "syntax_slot_0"
        vars_others = ["FEAT_num_strokes_beh", "epoch", "seqc_0_shape", "seqc_0_loc", "syntax_slot_1"]
        # vars_others = ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "syntax_slot_1"]
        # vars_others = ["FEAT_num_strokes_beh", "epoch", "syntax_slot_1"]
    """
    from neuralmonkey.scripts.analy_syntax_good_eucl_trial import regression_fit_and_score_single

    SAVEDIR = f"{SAVEDIR_BASE}/var={var_effect}-varother={'|'.join(vars_others)}-ndims={ndims}"
    os.makedirs(SAVEDIR, exist_ok=True)

    ### 
    LIST_DFRES = []
    LIST_DISTCOS = []
    for _, row in DFallpa.iterrows():
        bregion = row["bregion"]
        event = row["event"]

        ### (1) Extract regression fits
        plot_savedir = f"{SAVEDIR}/{event}-{bregion}/regression_fits"
        os.makedirs(plot_savedir, exist_ok=True)
        DFRES, PAredu, X, Y, Cl = regression_fit_and_score_single(DFallpa, bregion, event, var_effect, vars_others, ndims, 
                                                                  plot_savedir=plot_savedir)

        if DFRES is not None:
            ### (2) Plot scatter
            if PLOT_SCATTER_PRED_VS_ACTUAL:
                from neuralmonkey.scripts.analy_syntax_good_eucl_trial import regression_plot_scatter_ypred_vs_yactual
                savedir = f"{SAVEDIR}/{event}-{bregion}/scatter_pred_vs_actual"
                os.makedirs(savedir, exist_ok=True)
                regression_plot_scatter_ypred_vs_yactual(DFRES, X, Y, savedir)

            ### (3) Cosine distances of effect-axes between conditions
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import regression_cosine_distance_compute
            savedir = f"{SAVEDIR}/{event}-{bregion}/cosine_similarity-ALL"
            os.makedirs(savedir, exist_ok=True)
            DFDISTCOS = regression_cosine_distance_compute(DFRES, X, Y, savedir)

            for min_r2 in [0.05, 0.1, 0.2]:

                savedir = f"{SAVEDIR}/{event}-{bregion}/cosine_similarity-min_r2={min_r2}"
                os.makedirs(savedir, exist_ok=True)

                # Determine which groups have good regression fits. 
                list_grp_good_r2 = DFRES[(DFRES["same_grp"] == True) & (DFRES["r2_test"] >= min_r2)]["grp_train"].tolist()
                # filter to just this dfres
                dfres = DFRES[(DFRES["grp_train"].isin(list_grp_good_r2)) & (DFRES["grp_test"].isin(list_grp_good_r2))].reset_index(drop=True)

                ### (3) Cosine distances of effect-axes between conditions
                if len(dfres)>0:
                    from neuralmonkey.scripts.analy_syntax_good_eucl_trial import regression_cosine_distance_compute
                    _ = regression_cosine_distance_compute(dfres, X, Y, savedir)

            DFRES["bregion"] = bregion
            DFRES["event"] = event
            DFDISTCOS["bregion"] = bregion
            DFDISTCOS["event"] = event

            LIST_DFRES.append(DFRES)
            LIST_DISTCOS.append(DFDISTCOS)

    if len(LIST_DFRES)>0:
        DFRES_ALL = pd.concat(LIST_DFRES).reset_index(drop=True)
        DFDISTCOS_ALL = pd.concat(LIST_DISTCOS).reset_index(drop=True)

        # Append some things
        # def _sum(train_labels_counts):
        #     return sum(train_labels_counts.values())
        # def _min(train_labels_counts):
        #     return min(train_labels_counts.values())
        DFRES_ALL["n_tot"] = [sum(train_labels_counts.values()) for train_labels_counts in DFRES_ALL["train_labels_counts"]]
        DFRES_ALL["n_min_across_levs"] = [min(train_labels_counts.values()) for train_labels_counts in DFRES_ALL["train_labels_counts"]]

        ## Save data
        path = f"{SAVEDIR}/DFRES_ALL.pkl"
        DFRES_ALL.to_pickle(path)

        path = f"{SAVEDIR}/DFDISTCOS_ALL.pkl"
        DFDISTCOS_ALL.to_pickle(path)
    else:
        DFRES_ALL, DFDISTCOS_ALL = None, None

    return DFRES_ALL, DFDISTCOS_ALL, SAVEDIR

def regression_wrapper_prune(DFRES_ALL, DFDISTCOS_ALL, min_levs = 3, min_r2_test = None, min_ntot = 15, min_nmin = 4):
    """
    To return a pruned version of DFRES_ALL, DFDISTCOS_ALL, using filter criteria. Usualyl, is to ensure keep only
    those grps where there is (i) enough data to do regression and (ii) good enough regression to do cosine distance

    PARAMS:
    min_levs = 3
    min_r2_test = 0.1
    # min_r2_train = 0.05
    min_r2_test = None
    min_ntot = 15
    min_nmin = 4

    RETURNS:
    - DFRES_ALL_PRUNED, DFDISTCOS_ALL_PRUNED, each copied and pruned from original.
    """
    from pythonlib.tools.pandastools import aggregGeneral

    # First, get a dfres where each grp contributes just one row.
    # - one row for each (grp, bregion, event)
    dfthis = DFRES_ALL[DFRES_ALL["same_grp"]==True].reset_index(drop=True)
    DFRES_EACH_GRP = aggregGeneral(dfthis, ["grp_train", "bregion", "event"], values=["r2_train", "r2_test"])

    # Second, determine, for each grp, its r2_test to itself
    map_row_to_r2test = {}  
    for i, row in DFRES_EACH_GRP.iterrows(): 
        # Store results.
        key = (row["grp_train"], row["bregion"], row["event"])
        assert key not in map_row_to_r2test
        map_row_to_r2test[key] = row["r2_test"]

    ### Prune to just cases with good regression 
    map_row_to_keep = {}
    for i, row in DFRES_ALL.iterrows():

        keep = True

        # Check tjhat dataset has at least <min_levs> many levels 
        if min_levs is not None:
            n_levs = len(row["train_labels_counts"])
            if n_levs < min_levs:
                keep = False

        # Check that r2 was higher than this.
        if min_r2_test is not None:
            # For this grp_train, get its r2_test vs itself.
            _key = (row["grp_train"], row["bregion"], row["event"])
            grptrain_r2test = map_row_to_r2test[_key] # r2_test for this grp_train
            if grptrain_r2test <  min_r2_test:
                keep = False
            
        # if min_r2_train is not None:
        #     if row["r2_train"] < min_r2_train:
        #         keep = False

        if min_ntot is not None:
            if row["n_tot"] < min_ntot:
                keep = False
            
        if min_nmin is not None:
            if row["n_min_across_levs"] < min_nmin:
                keep = False
            
        # Store results.
        key = (row["grp_train"], row["grp_test"], row["bregion"], row["event"])
        assert key not in map_row_to_keep
        map_row_to_keep[key] = keep

    nkeep = sum(map_row_to_keep.values())
    ntot = len(map_row_to_keep.values())
    print("keep/tot:", nkeep, ntot)

    ### Do pruning
    # (1) DFRES
    keeps = []
    for i, row in DFRES_ALL.iterrows():
        key = (row["grp_train"], row["grp_test"], row["bregion"], row["event"])
        keeps.append(map_row_to_keep[key])
    DFRES_ALL_PRUNED = DFRES_ALL[keeps].reset_index(drop=True)

    # (2) COS
    keeps = []
    for i, row in DFDISTCOS_ALL.iterrows():
        key = (row["grp_1"], row["grp_2"], row["bregion"], row["event"])
        keeps.append(map_row_to_keep[key])
    DFDISTCOS_ALL_PRUNED = DFDISTCOS_ALL[keeps].reset_index(drop=True)

    return DFRES_ALL_PRUNED, DFDISTCOS_ALL_PRUNED

def regression_wrapper_plot_each_grp(DFRES_ALL, savedir, yvar = "r2_test"):
    """
    Helper function for plotting regression results.
    """

    # dfthis = DFRES_ALL[DFRES_ALL["same_grp"] == True].reset_index(drop=True)
    dfthis = DFRES_ALL[DFRES_ALL["same_grp"] == True].reset_index(drop=True)

    fig = sns.catplot(data=dfthis, x="bregion", y=yvar, col="event", jitter=True, alpha=0.5)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)
    savefig(fig, f"{savedir}/within_each_grp-catplot-1.pdf")

    try:
        fig = sns.catplot(data=dfthis, x="bregion", y=yvar, col="event", kind="bar", errorbar="se")
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.5)    
        savefig(fig, f"{savedir}/within_each_grp-catplot-2.pdf")
    except Exception as err:
        fig, ax = plt.subplots()
        savefig(fig, f"{savedir}/within_each_grp-catplot-2-FAILED.pdf")

def regression_wrapper_plot_cosine_sim(DFDISTCOS_ALL, savedir):
    """
    Helper function for plotting cosine similarity results.
    """

    dfthis = DFDISTCOS_ALL[DFDISTCOS_ALL["same_grp"] == False].reset_index(drop=True)

    fig = sns.catplot(data=dfthis, x="bregion", y="sim_cos", col="event", jitter=True, alpha=0.25, aspect=1.5)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)    
    savefig(fig, f"{savedir}/cosine_sim-catplot-1.pdf")

    fig = sns.catplot(data=dfthis, x="bregion", y="sim_cos", col="event", kind="bar", errorbar="se", aspect=1.5)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0, color="k", alpha=0.5)        
    savefig(fig, f"{savedir}/cosine_sim-catplot-2.pdf")

def regression_fit_and_score_single_ordinal(DFallpa, bregion, event, var_effect, vars_others, ndims, n_min_trials=N_MIN_TRIALS, 
                                    plot_savedir=None):
    """

    For trials version -- do regression for a single var_effect and vars_others. 
    See regression_wrapper_wrapper for more details.

    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

    tmp = DFallpa[(DFallpa["event"] == event) & (DFallpa["bregion"] == bregion)]["pa_redu"].values[0]
    if tmp is None:
        # This is usually because dim reduction failed, variables desired for targeted PCA did not exist.
        return None, None, None, None, None
    PAredu = tmp.copy()

    # First, prune to cases with enough data 
    dflab = PAredu.Xlabels["trials"]
    savepath_pre = "/tmp/counts_pre.pdf"
    savepath_post = "/tmp/counts_post.pdf"
    dfout, dict_dfthis = extract_with_levels_of_conjunction_vars_helper(dflab, var_effect, vars_others, n_min_trials, savepath_post, 2,
                                                                        plot_counts_also_before_prune_path=savepath_pre)
    if len(dfout)==0:
        # No data after pruning -- skip
        return None, None, None, None, None
    # assert len(dfout)>0, "all data pruned!!"
    
    # Only keep the indices in dfout
    PAredu = PAredu.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)

    ### Extract final data
    dflab = PAredu.Xlabels["trials"]
    Xredu = PAredu.X
    X = Xredu.squeeze().T[:, :ndims]
    Y = dflab[var_effect].values
    
    ##### Get pairwise socres between all groups
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
    from neuralmonkey.analyses.regression_good import fit_and_score_regression, ordinal_fit_and_score_train_test_splits
    from pythonlib.tools.listtools import tabulate_list
    # do_upsample=True
    # Iterate over all levels of conditioned grp vars
    groupdict = grouping_append_and_return_inner_items(dflab, vars_others)
    RES = []
    for grp_train, inds_train in groupdict.items():
        # Test decoder on all other levels of grouping var
        for grp_test, inds_test in groupdict.items():
            
            # Gather test data
            X_train = X[inds_train, :] # (ntrials, nchans)
            labels_train = [Y[i] for i in inds_train]

            X_test = X[inds_test, :] # (ntrials, nchans)
            labels_test = [Y[i] for i in inds_test]

            train_labels_counts = tabulate_list(labels_train)
            test_labels_counts = tabulate_list(labels_test)

            if grp_train == grp_test:
                # Then these are same group -- compute using train-test split of held-out trials.
                assert inds_train == inds_test
                dfres, r2_test = ordinal_fit_and_score_train_test_splits(X_train, labels_train, max_nsplits=None)

                # also save training on itself
                reg, r2_train, _ = fit_and_score_regression(X_train, labels_train)
            else:
                if False:
                    # Only keep test inds that are labels which exist in training
                    # - find inds in labels_all that are labels that exist in train.
                    indstmp = np.argwhere(np.isin(labels_all, labels_train)).squeeze().tolist() # (n,) array of int indices
                    inds_test = [i for i in inds_test if i in indstmp]

                if False:
                    # check the distribution of var_decode labels.
                    test_labels_counts = tabulate_list(labels_test)

                    # Each test label  must exist in training data
                    for lab in test_labels_counts.keys():
                        if lab not in train_labels_counts.keys():
                            print(train_labels_counts)
                            print(test_labels_counts)
                            assert False, "add clause to skip test cases that have labels that dont exist in training data?"

                # score it
                reg, r2_train, r2_test = fit_and_score_regression(X_train, labels_train, X_test, labels_test)

            # print(r2_train, " -- ", r2_test)

            ### Save results
            RES.append({
                "var_effect":var_effect,
                "vars_others":tuple(vars_others),
                "grp_train":grp_train,
                "grp_test":grp_test,
                "r2_train":r2_train,
                "r2_test":r2_test,
                "train_labels_counts":train_labels_counts,
                "test_labels_counts":test_labels_counts,
                "inds_train":inds_train, # Store things for each of making plots later 
                "inds_test":inds_test,
                "reg":reg,
            })

    if len(RES)==0:
        return None, None, None, None, None
    
    DFRES = pd.DataFrame(RES)
    DFRES["same_grp"] = DFRES["grp_train"] == DFRES["grp_test"]

    ### Plot?
    if plot_savedir is not None:
        from pythonlib.cluster.clustclass import generate_clustclass_from_flat_df

        Cl = generate_clustclass_from_flat_df(DFRES, "grp_train", "grp_test", var_value="r2_test", var_labels=vars_others)
        fig, _ = Cl.rsa_plot_heatmap()
        savefig(fig, f"{plot_savedir}/heatmap-r2_test.pdf")

        Cl2 = Cl.copy_with_slicing()
        Cl2._Xinput[Cl._Xinput<0] = 0.
        fig, _  = Cl2.rsa_plot_heatmap()
        savefig(fig, f"{plot_savedir}/heatmap-r2_test-abovezero.pdf")

        # Plot regression r2 within each grp
        dftmp = DFRES[DFRES["same_grp"]]

        fig = sns.catplot(data=dftmp, x="r2_train", y="grp_train", kind="bar")
        savefig(fig, f"{plot_savedir}/catplot-r2_train.pdf")

        fig = sns.catplot(data=dftmp, x="r2_test", y="grp_train", kind="bar")
        savefig(fig, f"{plot_savedir}/catplot-r2_test.pdf")

        plt.close("all")

    return DFRES, PAredu, X, Y, Cl

def regression_fit_and_score_single(DFallpa, bregion, event, var_effect, vars_others, ndims, n_min_trials=N_MIN_TRIALS, 
                                    plot_savedir=None):
    """
    Do regression for a single var_effect and vars_others. See regression_wrapper_wrapper for more details.
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

    tmp = DFallpa[(DFallpa["event"] == event) & (DFallpa["bregion"] == bregion)]["pa_redu"].values[0]
    if tmp is None:
        # This is usually because dim reduction failed, variables desired for targeted PCA did not exist.
        return None, None, None, None, None
    PAredu = tmp.copy()

    # First, prune to cases with enough data 
    dflab = PAredu.Xlabels["trials"]
    savepath_pre = "/tmp/counts_pre.pdf"
    savepath_post = "/tmp/counts_post.pdf"
    dfout, dict_dfthis = extract_with_levels_of_conjunction_vars_helper(dflab, var_effect, vars_others, n_min_trials, savepath_post, 2,
                                                                        plot_counts_also_before_prune_path=savepath_pre)
    if len(dfout)==0:
        # No data after pruning -- skip
        return None, None, None, None, None
    # assert len(dfout)>0, "all data pruned!!"
    
    # Only keep the indices in dfout
    PAredu = PAredu.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)

    ### Extract final data
    dflab = PAredu.Xlabels["trials"]
    Xredu = PAredu.X
    X = Xredu.squeeze().T[:, :ndims]
    Y = dflab[var_effect].values
    
    ##### Get pairwise socres between all groups
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
    from neuralmonkey.analyses.regression_good import fit_and_score_regression, ordinal_fit_and_score_train_test_splits
    from pythonlib.tools.listtools import tabulate_list
    # do_upsample=True
    # Iterate over all levels of conditioned grp vars
    groupdict = grouping_append_and_return_inner_items(dflab, vars_others)
    RES = []
    for grp_train, inds_train in groupdict.items():
        # Test decoder on all other levels of grouping var
        for grp_test, inds_test in groupdict.items():
            
            # Gather test data
            X_train = X[inds_train, :] # (ntrials, nchans)
            labels_train = [Y[i] for i in inds_train]

            X_test = X[inds_test, :] # (ntrials, nchans)
            labels_test = [Y[i] for i in inds_test]

            train_labels_counts = tabulate_list(labels_train)
            test_labels_counts = tabulate_list(labels_test)


            if grp_train == grp_test:
                # Then these are same group -- compute using train-test split of held-out trials.
                assert inds_train == inds_test
                dfres, r2_test = ordinal_fit_and_score_train_test_splits(X_train, labels_train, max_nsplits=None)

                # also save training on itself
                reg, r2_train, _ = fit_and_score_regression(X_train, labels_train)
            else:
                if False:
                    # Only keep test inds that are labels which exist in training
                    # - find inds in labels_all that are labels that exist in train.
                    indstmp = np.argwhere(np.isin(labels_all, labels_train)).squeeze().tolist() # (n,) array of int indices
                    inds_test = [i for i in inds_test if i in indstmp]

                if False:
                    # check the distribution of var_decode labels.
                    test_labels_counts = tabulate_list(labels_test)

                    # Each test label  must exist in training data
                    for lab in test_labels_counts.keys():
                        if lab not in train_labels_counts.keys():
                            print(train_labels_counts)
                            print(test_labels_counts)
                            assert False, "add clause to skip test cases that have labels that dont exist in training data?"

                # score it
                reg, r2_train, r2_test = fit_and_score_regression(X_train, labels_train, X_test, labels_test)

            # print(r2_train, " -- ", r2_test)

            ### Save results
            RES.append({
                "var_effect":var_effect,
                "vars_others":tuple(vars_others),
                "grp_train":grp_train,
                "grp_test":grp_test,
                "r2_train":r2_train,
                "r2_test":r2_test,
                "train_labels_counts":train_labels_counts,
                "test_labels_counts":test_labels_counts,
                "inds_train":inds_train, # Store things for each of making plots later 
                "inds_test":inds_test,
                "reg":reg,
            })

    if len(RES)==0:
        return None, None, None, None, None
    
    DFRES = pd.DataFrame(RES)
    DFRES["same_grp"] = DFRES["grp_train"] == DFRES["grp_test"]

    ### Plot?
    if plot_savedir is not None:
        from pythonlib.cluster.clustclass import generate_clustclass_from_flat_df

        Cl = generate_clustclass_from_flat_df(DFRES, "grp_train", "grp_test", var_value="r2_test", var_labels=vars_others)
        fig, _ = Cl.rsa_plot_heatmap()
        savefig(fig, f"{plot_savedir}/heatmap-r2_test.pdf")

        Cl2 = Cl.copy_with_slicing()
        Cl2._Xinput[Cl._Xinput<0] = 0.
        fig, _  = Cl2.rsa_plot_heatmap()
        savefig(fig, f"{plot_savedir}/heatmap-r2_test-abovezero.pdf")

        # Plot regression r2 within each grp
        dftmp = DFRES[DFRES["same_grp"]]

        fig = sns.catplot(data=dftmp, x="r2_train", y="grp_train", kind="bar")
        savefig(fig, f"{plot_savedir}/catplot-r2_train.pdf")

        fig = sns.catplot(data=dftmp, x="r2_test", y="grp_train", kind="bar")
        savefig(fig, f"{plot_savedir}/catplot-r2_test.pdf")

        plt.close("all")

    return DFRES, PAredu, X, Y, Cl

def regression_extract_data(DFRES, X, Y, grp_train, grp_test):
    """
    Extract data for a given regression fit, using the grp_train and grp_test.
    """
    tmp = DFRES[(DFRES["grp_train"]==grp_train) & (DFRES["grp_test"]==grp_test)]
    assert len(tmp)==1
    inds_train = tmp["inds_train"].values[0]  
    inds_test = tmp["inds_test"].values[0] 
    reg = tmp["reg"].values[0]
    r2_test = tmp["r2_test"].values[0]
    r2_train = tmp["r2_train"].values[0]
    
    X_train = X[inds_train, :] # (ntrials, nchans)
    labels_train = [Y[i] for i in inds_train]

    X_test = X[inds_test, :] # (ntrials, nchans)
    labels_test = [Y[i] for i in inds_test]

    return X_train, labels_train, X_test, labels_test, reg, r2_train, r2_test

def regression_plot_scatter_ypred_vs_yactual(DFRES, X, Y, savedir):
    """
    Plot y-predicted vs. y-actual, for cross between all training and testing sets.
    NOTE: for trainig vs. training, uses the global model (not the train-test splits)
    """

    ### Plot scatter (ypred vs. yactual) for all regression models
    import seaborn as sns
    from pythonlib.tools.plottools import plot_y_at_each_x_mean_sem

    list_grp = sorted(DFRES["grp_train"].unique())

    for i, grp_fit in enumerate(list_grp):

        ncols = 6
        nrows = int(np.ceil(len(list_grp)/ncols))
        SIZE = 3
        # fig, axes = plt.subplots(nrows, ncols, figsize=(nrows*SIZE, ncols*SIZE))
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*SIZE, nrows*SIZE))

        vals_all = np.array([])
        for j, grp_dat in enumerate(list_grp):
            
            _, _, X_test, y_test, reg, r2_train, r2_test = regression_extract_data(DFRES, X, Y, grp_fit, grp_dat)
            # ax = axes[i][j]
            ax = axes.flatten()[j]
            # ax.set_xlabel(f"{var_effect}-actual")
            # ax.set_ylabel(f"{var_effect}-pred")
            ax.set_ylabel(f"train={grp_fit}", fontsize=8)
            ax.set_xlabel(f"test={grp_dat}", fontsize=8)
            ax.set_title(f"pred_vs_actual r2={r2_test:.2f}", fontsize=8)

            # Plot
            y_pred = reg.predict(X_test)

            # add jitter
            y_test_jitter = y_test + (np.random.rand(len(y_test))-0.5)/6

            # Plot of predicted vs. actual
            sns.scatterplot(x=y_test_jitter, y=y_pred, ax=ax, alpha=0.4)
            # sns.pointplot(x=np.array(y_test)+0.001, y=y_pred, ax=ax, color="k")
            plot_y_at_each_x_mean_sem(y_test, y_pred, ax)

            # from pythonlib.tools.plottools import set_axis_lims_square_bounding_data_45line
            # set_axis_lims_square_bounding_data_45line(ax, y_test, y_pred, dotted_lines="unity")

            vals_all = np.concatenate([vals_all, y_test])
            vals_all = np.concatenate([vals_all, y_pred])

        if True:
            from pythonlib.tools.plottools import set_axis_lims_square_bounding_data_45line
            for ax in axes.flatten():
                set_axis_lims_square_bounding_data_45line(ax, vals_all, vals_all, dotted_lines="unity")
        else:
            ylim = ax.get_ylim()
            for ax in axes.flatten():
                ax.plot(ylim, ylim, '--k', alpha=0.5)
        
        savefig(fig, f"{savedir}/scatter-grp_fit={grp_fit}.pdf")
        plt.close("all")

def regression_cosine_distance_compute(DFRES, X, Y, savedir):
    """
    Get cosine distance pairwise between all regression axes
    """
    from pythonlib.tools.vectools import cosine_similarity
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.snstools import rotateLabel

    tmp = DFRES["vars_others"].unique().tolist()
    assert len(tmp)==1
    vars_others = tmp[0]

    # Get cosine between each axis
    list_grp = sorted(DFRES["grp_train"].unique())
    res = []
    for i, grp_1 in enumerate(list_grp):
        for j, grp_2 in enumerate(list_grp):

            _, _, _, _, reg_1, _, _ = regression_extract_data(DFRES, X, Y, grp_1, grp_1)
            _, _, _, _, reg_2, _, _ = regression_extract_data(DFRES, X, Y, grp_2, grp_2)
            
            # print(grp_1, " -- ", grp_2, " -- ", reg_1.coef_, reg_2.coef_,  " -- ", r2_1, r2_2)
            
            # get cosine distance
            sim_cos = cosine_similarity(reg_1.coef_, reg_2.coef_)

            res.append({
                "grp_1":grp_1,
                "grp_2":grp_2,
                "sim_cos":sim_cos
            })
    dfdistcos = pd.DataFrame(res)
    dfdistcos["same_grp"] = dfdistcos["grp_1"] == dfdistcos["grp_2"]

    ### PLOTS
    # Convert to heatmap and plot.
    from pythonlib.cluster.clustclass import generate_clustclass_from_flat_df
    Cl = generate_clustclass_from_flat_df(dfdistcos, "grp_1", "grp_2", var_value="sim_cos", var_labels=vars_others)
    fig, _ = Cl.rsa_plot_heatmap(zlims=[-1, 1], diverge=True)
    savefig(fig, f"{savedir}/cosinedist-heatmap.pdf")

    # Plot distributions of distances
    dfdistcos_str = stringify_values(dfdistcos)

    fig = sns.catplot(data=dfdistcos_str, x="grp_1", y="sim_cos", hue="same_grp")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/cosinedist-catplot-1.pdf")

    fig = sns.catplot(data=dfdistcos_str, x="grp_1", y="sim_cos", hue="same_grp", kind="bar")
    rotateLabel(fig)
    savefig(fig, f"{savedir}/cosinedist-catplot-2.pdf")

    fig = sns.catplot(data=dfdistcos_str, x="same_grp", y="sim_cos")
    savefig(fig, f"{savedir}/cosinedist-catplot-3.pdf")

    fig = sns.catplot(data=dfdistcos_str, x="same_grp", y="sim_cos", kind="bar")
    savefig(fig, f"{savedir}/cosinedist-catplot-4.pdf")

    fig = sns.displot(data=dfdistcos_str, x="sim_cos", hue="same_grp", element="step")
    savefig(fig, f"{savedir}/cosinedist-hist-1.pdf")

    plt.close("all")

    return dfdistcos


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion, dfpa_concat_bregion_to_combined_bregion
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
    from pythonlib.tools.pandastools import append_col_with_grp_index
    import seaborn as sns
    import os
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single, euclidian_distance_compute_trajectories

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/syntax_good_trial"

    animal = sys.argv[1]
    date = int(sys.argv[2])
    # question = sys.argv[3]
    question = "RULE_BASE_trial"

    version = "trial"
    combine = False

    # events_keep = ["03_samp", "06_on_strokeidx_0"]
    events_keep = ["03_samp"]

    # PLOTS_DO = [1.1, 1.2] 
    # PLOTS_DO = [1.1, 1.2] 
    # PLOTS_DO = [1.1] 
    # PLOTS_DO = [2.2] 
    # PLOTS_DO = [3.1] 
    PLOTS_DO = [4.0] 

    # Load a single DFallPA
    DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, 
                                     question=question)
    
    ### Preprocessing
    DFallpa = DFallpa[DFallpa["event"].isin(events_keep)].reset_index(drop=True)

    if combine == False and 4.0 in PLOTS_DO:
        # Then combine it
        DFallpa = dfpa_concat_bregion_to_combined_bregion(DFallpa)

    # Make a copy of all PA before normalization
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

    # Some preprocessing
    preprocess_dfallpa_basic_quick(DFallpa)
    # for PA in DFallpa["pa"]:
    #     dflab = PA.Xlabels["trials"]

    #     # dflab = append_col_with_grp_index(dflab, vars_others, "_var_other")
    #     dflab = append_col_with_grp_index(dflab, ["epoch", "syntax_concrete"], "epch_sytxcncr")

    #     # Add n items in each shape slot
    #     nslots = len(dflab["syntax_concrete"].values[0])-1
    #     list_slots = []
    #     for i in range(nslots):
    #         key = f"syntax_slot_{i}"
    #         dflab[key] = [x[i] for x in dflab["syntax_concrete"]]
    #         list_slots.append(key)
    #     print("Added these columns to dflab: ", list_slots)

    #     # Add ratio between slot 0 and 1
    #     if ("syntax_slot_0" in dflab) & ("syntax_slot_1" in dflab):
    #         # Add 0.01 so that doesnt devide by 0.
    #         dflab["syntax_slot_ratio"] = (dflab["syntax_slot_1"]+0.01)/(dflab["syntax_slot_0"]+0.01 + dflab["syntax_slot_1"]+0.01)

    #         if np.any(np.isnan(dflab["syntax_slot_ratio"])):
    #             print(dflab["syntax_slot_ratio"])
    #             assert False

    #     # count up how many unique shapes are shown
    #     def _n_shapes(syntax_concrete):
    #         # e.g., (1,3,0) --> 2
    #         return sum([x>0 for x in syntax_concrete])
    #     dflab["shapes_n_unique"] = [_n_shapes(sc) for sc in dflab["syntax_concrete"]]

    #     PA.Xlabels["trials"] = dflab


    LIST_VAR_VAROTHERS_SS = [
        ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
        ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
        ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch"]),

        ("shapes_n_unique", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),

        ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
        ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "syntax_slot_1"]),
        ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
        ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
        ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch"]),

        ("syntax_slot_1", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
        ("syntax_slot_1", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "syntax_slot_0"]),

        ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1"]),
        ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
        ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
        ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch"]),

        ("syntax_concrete", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
        ("syntax_concrete", ["FEAT_num_strokes_beh", "epoch"]),

        ("FEAT_num_strokes_beh", ["epoch", "seqc_0_loc", "seqc_0_shape"]),
        ("epoch", ["FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape"]),
        ("seqc_0_shape", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
    ]

    LIST_VAR_VAROTHERS_REGR = [
        ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
        # ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "syntax_slot_1"]),
        # ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
        # ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
        # ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch"]),
        # ("syntax_slot_1", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "syntax_slot_0"]),
        ("syntax_slot_1", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
        ("FEAT_num_strokes_beh", ["epoch", "seqc_0_loc", "seqc_0_shape"]),
    ]


    ### QUICK - print grouping labels.
    from pythonlib.tools.pandastools import grouping_print_n_samples
    savedir = f"{SAVEDIR}/conjunctions/{animal}-{date}-comb={combine}-q={question}"            
    os.makedirs(savedir, exist_ok=True)
    PA = DFallpa["pa"].values[0]
    dflab = PA.Xlabels["trials"]

    savepath = f"{savedir}/counts-1.txt"
    grouping_print_n_samples(dflab, ["FEAT_num_strokes_beh", "epoch", "syntax_concrete"], savepath=savepath)

    savepath = f"{savedir}/counts-2.txt"
    grouping_print_n_samples(dflab, ["FEAT_num_strokes_beh", "epoch", "syntax_concrete", "behseq_shapes", "behseq_locs"], savepath=savepath)

    ################################### PLOTS
    for plotdo in PLOTS_DO:
        if plotdo in [1.1, 1.2]:
            list_subspace_projection = ["syntax_slot_0", "epch_sytxcncr", "syntax_slot_1", "pca_umap"]
            tbin_dur = "default"
            tbin_slide = None

            for subspace_projection in list_subspace_projection:
                
                # dflab = DFallpa["pa"].values[0].Xlabels["trials"]
                # if su

                SAVEDIR_ANALYSIS = f"{SAVEDIR}/statespace_and_regression/{animal}-{date}-comb={combine}-q={question}-ssproj={subspace_projection}"            
                os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

                ### Preprocessing
                LIST_DIMS = preprocess_dfallpa(DFallpa, subspace_projection, tbin_slide, tbin_dur, SAVEDIR_ANALYSIS, HACK=False)

                if 1.1 in PLOTS_DO:
                    ### State space scalar plots.
                    savedir = f"{SAVEDIR_ANALYSIS}/state_space_scalar"
                    os.makedirs(savedir, exist_ok=True)
                    state_space_plot_scalar_wrapper(DFallpa, savedir, LIST_VAR_VAROTHERS_SS, LIST_DIMS)

                if 1.2 in PLOTS_DO:
                    
                    ### Regression plots
                    ndims = 4
                    regression_wrapper_wrapper(DFallpa, LIST_VAR_VAROTHERS_REGR, SAVEDIR_ANALYSIS, ndims)

        elif plotdo == 2.1:
            # [MAYBE this:] state space plots, using targeted PCA
            variables = ["epoch", "FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1", "syntax_slot_2"]
            variables_is_cat = [True, True, True, True, False, False, False]

            # Which subspaces to plot and compare between
            list_subspaces = [
                ("syntax_slot_0", "syntax_slot_1"),
                ("syntax_slot_1", "syntax_slot_2"),   
                ("syntax_slot_0", "syntax_slot_2"),   
            ]

            #################### STATE SPACE PLOTS
            LIST_DIMS = [(0,1)]
            # prune to just 03_samp, for now.
            # map_event_to_twind = {
            #     "03_samp":[0.2, 1.0],
            #     "05_first_raise":[-0.5, -0.05],
            #     "06_on_strokeidx_0":[-0.5, -0.05],
            # }

            for _, row in DFallpa.iterrows():
                PA = row["pa"]
                bregion = row["bregion"]
                event = row["event"]        

                twind_scal = map_event_to_twind[event]

                SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_dim_reduction/{animal}-{date}-comb={combine}-q={question}/{event}-{bregion}"            
                os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
                state_space_targeted_pca_scalar_single(PA, twind_scal, variables, variables_is_cat, list_subspaces, 
                                                        LIST_VAR_VAROTHERS_SS, LIST_DIMS, SAVEDIR_ANALYSIS)

            #################### REGRESSION (using targeted PCs)
            list_subspaces = [
                ("syntax_slot_0", "syntax_slot_1"),
                # ("syntax_slot_1", "syntax_slot_2"),   
                # ("syntax_slot_0", "syntax_slot_2"),   
            ]

            # Do regression to score strength of encoding in this space
            map_BrEvSs_to_Paredu = state_space_targeted_pca_do_projection(DFallpa, variables, variables_is_cat, 
                                                                        list_subspaces, None)
            
            ### Regression plots
            for subspace in list_subspaces:

                SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_dim_reduction_regr/{animal}-{date}-comb={combine}-q={question}/subspace={subspace}"            
                os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
                
                ndims = len(subspace)

                list_paredu = []
                for _, row in DFallpa.iterrows():
                    key = (row["bregion"], row["event"], subspace)
                    paredu = map_BrEvSs_to_Paredu[key]
                    list_paredu.append(paredu)
                DFallpa["pa_redu"] = list_paredu

                regression_wrapper_wrapper(DFallpa, LIST_VAR_VAROTHERS_REGR, SAVEDIR_ANALYSIS, ndims)

        elif plotdo == 2.2:
            """ Good, projec to targeted PCA space, and then ask about consistency of angle for trajectory along slot 0 and slot 1.
            Uses method based on dot products of euclidean distance, which is less noisy than regression. 
            Goals:
            (1) Clear effect of slot num
            (2) Factorized of slot 0 vs. slot 1.
            """
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_dim_redu_angles/{animal}-{date}-comb={combine}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

            # During samp
            from neuralmonkey.scripts.analy_syntax_good_eucl_trial import state_space_targeted_pca_scalar_single, targeted_pca_euclidian_dist_angles

            # variables = ["epoch", "seqc_0_loc", "seqc_0_shape", "FEAT_num_strokes_beh", "syntax_slot_0", "syntax_slot_1"]
            variables = ["epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1"] # exclude FEAT_num_strokes_beh, because it is always correlated with (slot0, slot1) in my dataset
            variables_is_cat = [True, True, True, False, False, False]

            list_subspaces = [
                ("syntax_slot_0", "syntax_slot_1"),
                # ("syntax_slot_1", "syntax_slot_2"),   
                # ("syntax_slot_0", "syntax_slot_2"),   
            ]

            LIST_VAR_VAROTHERS_SS = [
                # ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
                # ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
                # ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch"]),

                # ("shapes_n_unique", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),

                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1", "epoch", "FEAT_num_strokes_beh"]),
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1", "epoch"]),
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape"]),
                ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch"]),

                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "epoch", "FEAT_num_strokes_beh"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "epoch"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape"]),
                ("syntax_slot_1", ["FEAT_num_strokes_beh", "epoch"]),

                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1"]),
                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch"]),

                # ("syntax_concrete", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
                ("syntax_concrete", ["FEAT_num_strokes_beh", "epoch"]),

                ("FEAT_num_strokes_beh", ["epoch", "seqc_0_loc", "seqc_0_shape"]),
                # ("epoch", ["FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape"]),
                ("seqc_0_shape", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
            ]

            LIST_DIMS = [(0,1), (1,2)]

            LIST_VAR_VAROTHERS_REGR = [
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1", "epoch", "FEAT_num_strokes_beh"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "epoch", "FEAT_num_strokes_beh"]),
                # ("FEAT_num_strokes_beh", ["epoch", "seqc_0_loc", "seqc_0_shape"]),
            ]

            ### (1) Compute all angles, etc (and make state space plots)
            subspace_tuple = ('syntax_slot_0', 'syntax_slot_1')
            DFANGLE = targeted_pca_euclidian_dist_angles(DFallpa, SAVEDIR_ANALYSIS, 
                                                variables, variables_is_cat, list_subspaces, LIST_VAR_VAROTHERS_SS, # For dim reduction and plotting state space
                                                subspace_tuple, LIST_VAR_VAROTHERS_REGR)

            ### (2) Make all plots
            for var_vector_length in ["dist_yue_diff", "dist_norm"]:
                for length_method in ["sum", "dot"]:
                    for min_levs_exist in [3, 2]:
                        savedir = f"{SAVEDIR_ANALYSIS}/PLOTS/varlength={var_vector_length}-lengthmeth={length_method}-minlevs={min_levs_exist}"
                        os.makedirs(savedir, exist_ok=True)
                        targeted_pca_euclidian_dist_angles_plots(DFANGLE, var_vector_length, length_method, min_levs_exist, savedir)
        elif plotdo==3.1:
            # Just to try loading DFallpa
            SAVEDIR_ANALYSIS = f"{SAVEDIR}/dates_with_success_loading_dfallpa/{animal}-{date}-comb={combine}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

        elif plotdo == 4.0:
            """ 
            New, came back to this after syntax state. Does many things, including euclidean, angles, regression, and comapring
            alginmetn of regression axes.

            MULT: see notebook.
            """

            SAVEDIR_ANALYSIS = f"{SAVEDIR}/targeted_pca_v2/{animal}-{date}-q={question}"            
            os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)

            LIST_VAR_VAROTHERS_SS = [
                # ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
                # ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
                # ("syntax_slot_ratio", ["FEAT_num_strokes_beh", "epoch"]),

                # ("shapes_n_unique", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),

                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1", "epoch", "FEAT_num_strokes_beh"]),
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1", "epoch"]),
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
                ("syntax_slot_0", ["seqc_0_loc", "seqc_0_shape"]),
                ("syntax_slot_0", ["FEAT_num_strokes_beh", "epoch"]),

                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "epoch", "FEAT_num_strokes_beh"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "epoch"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
                ("syntax_slot_1", ["seqc_0_loc", "seqc_0_shape"]),
                ("syntax_slot_1", ["FEAT_num_strokes_beh", "epoch"]),

                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0", "syntax_slot_1"]),
                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_0"]),
                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape", "syntax_slot_1"]),
                # ("syntax_slot_2", ["FEAT_num_strokes_beh", "epoch"]),

                # ("syntax_concrete", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc", "seqc_0_shape"]),
                ("syntax_concrete", ["FEAT_num_strokes_beh", "epoch"]),

                ("FEAT_num_strokes_beh", ["epoch", "seqc_0_loc", "seqc_0_shape"]),
                # ("epoch", ["FEAT_num_strokes_beh", "seqc_0_loc", "seqc_0_shape"]),
                ("seqc_0_shape", ["FEAT_num_strokes_beh", "epoch", "seqc_0_loc"]),
            ]

            targeted_pca_combined_v2_good(DFallpa, SAVEDIR_ANALYSIS, LIST_VAR_VAROTHERS_SS, 
                                          twind_scal_force=None)

            # TODO: summary plot here: 
            # ### (2) Make all plots
            # for var_vector_length in ["dist_yue_diff", "dist_norm"]:
            #     for length_method in ["sum", "dot"]:
            #         for min_levs_exist in [3, 2]:
            #             savedir = f"{SAVEDIR_ANALYSIS}/PLOTS/varlength={var_vector_length}-lengthmeth={length_method}-minlevs={min_levs_exist}"
            #             os.makedirs(savedir, exist_ok=True)
            #             targeted_pca_euclidian_dist_angles_plots(DFANGLE, var_vector_length, length_method, min_levs_exist, savedir)

        else:
            print(plotdo)
            assert False