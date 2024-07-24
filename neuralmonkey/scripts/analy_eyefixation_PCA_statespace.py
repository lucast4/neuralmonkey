"""
PCA plots aligned to fixation onsets -- Kedar plots
Focusing on contrasting shape-fixation vs. seqc_0_shape.
e.g., See effect of one, controlling for other? And see similar state space?


"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys

from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa

from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper


def plot_all(DFallpa, SAVEDIR_ANALYSIS):
    from neuralmonkey.analyses.state_space_good import trajgood_construct_df_from_raw, trajgood_plot_colorby_splotby, trajgood_plot_colorby_splotby_scalar
    from pythonlib.tools.plottools import savefig
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
    import matplotlib.pyplot as plt
    import os
    from neuralmonkey.analyses.state_space_good import dimredgood_nonlinear_embed_data

    # Add variables
    def f(smi):
        if smi == -1:
            return "withinfixation"
        elif smi in [0,1]:
            return "early"
        elif smi>1:
            return "late"
        else:
            assert False
    for pa in DFallpa["pa"]:
        dflab = pa.Xlabels["trials"]
        dflab["early_late_by_smi"] = [f(smi) for smi in dflab["shape-macrosaccade-index"]]

    # Keep just high N shapes
    n_min = 10
    dflabthis = dflab[dflab["FEAT_num_strokes_task"]==1]
    # Need at least n_min unique trialcodes
    shapes_keep = [sh for sh in dflabthis["seqc_0_shape"].unique() if len(dflabthis[dflabthis["seqc_0_shape"]==sh]["trialcode"].unique())>n_min]

    ### DPCA params
    var_pca = "seqc_0_shape"
    vars_grouping = None
    pca_twind = (-0.3, 0.3)
    n_pcs_subspace_max = 10
    filtdict_train = {
        "FEAT_num_strokes_beh":[1],
        "FEAT_num_strokes_task":[1],
        "seqc_0_shape":shapes_keep,
    }

    # For pancho 240618, or else too many low N shapes
    filtdict_test = {
        "seqc_0_shape":shapes_keep,
    }

    ### USER PARAMS
    tbin_dur = 0.1
    tbin_slide = 0.1
    umap_n_neighbors = 45
    pca_frac_var_keep = 0.8
    n_min_per_levo= 5
    list_twind_overall = [
        [-0.3, 0.0],
        [0.0, 0.3]
    ]
    # METHOD = "umap"
    METHOD = "pca"
    list_var_color_var_subplot = [
        ["seqc_0_shape", ("task_kind",)],
        ["seqc_0_shape", ("seqc_0_loc", "task_kind")],
        ["seqc_0_shape", ("shape-fixation", "task_kind")],
        ["seqc_0_shape", ("seqc_0_loc", "shape-fixation", "task_kind")],
        # ["seqc_0_shape", ("shape-macrosaccade-index", "seqc_0_loc", "shape-fixation", "task_kind")],
        ["seqc_0_shape", ("shape-macrosaccade-index", "shape-fixation", "task_kind")],
        ["seqc_0_shape", ("is-first-macrosaccade", "shape-fixation", "task_kind")],
        ["seqc_0_shape", ("early-or-late-planning-period", "shape-fixation", "task_kind")],
        ["seqc_0_shape", ("early_late_by_smi", "shape-fixation", "task_kind")],
        ["shape-fixation", ("task_kind",)],
        ["shape-fixation", ("seqc_0_loc", "task_kind")],
        ["shape-fixation", ("seqc_0_shape", "task_kind")],
        ["shape-fixation", ("seqc_0_loc", "seqc_0_shape", "task_kind")],
        # ["shape-fixation", ("shape-macrosaccade-index", "seqc_0_loc", "seqc_0_shape", "task_kind")],
        ["shape-fixation", ("shape-macrosaccade-index", "seqc_0_shape", "task_kind")],
        ["shape-fixation", ("is-first-macrosaccade", "seqc_0_shape", "task_kind")],
        ["shape-fixation", ("early-or-late-planning-period", "seqc_0_shape", "task_kind")],
        ["shape-fixation", ("early_late_by_smi", "seqc_0_shape", "task_kind")],
    ]

    #LIST_DIMS = [(0,1)]
    LIST_DIMS = [(0,1), (2,3)]

    ### HARD PARAMS
    reshape_method = "trials_x_chanstimes"

    for i, row in DFallpa.iterrows():
        pa = row["pa"]
        br = row["bregion"]
        # wl = row["which_level"]
        ev = row["event"]
        # tw = row["twind"]
        for twind_overall in list_twind_overall:
            for METHOD in ["umap", "pca", "dpca"]:
                
                ###################### SCALAR DATA
                # Extract data
                if METHOD == "pca":
                    _, pathis, _, _, _ = pa.dataextract_state_space_decode_flex(twind_overall, tbin_dur, tbin_slide, reshape_method,
                                                                    pca_reduce=True, pca_frac_var_keep=pca_frac_var_keep)
                    pathis = pathis.slice_by_labels_filtdict(filtdict_test)
                    dflab = pathis.Xlabels["trials"]
                    Xredu = pathis.X.squeeze(axis=2).T
                    LIST_DIMS = [(0,1), (2,3)]
                elif METHOD == "dpca":
                    _, pathis, _, _, _ = pa.dataextract_pca_demixed_subspace(var_pca, vars_grouping,
                                                                    pca_twind, tbin_dur, # -- PCA params start
                                                                    filtdict=filtdict_train,
                                                                    raw_subtract_mean_each_timepoint=False,
                                                                    pca_subtract_mean_each_level_grouping=True,
                                                                    n_min_per_lev_lev_others=5, prune_min_n_levs = 2,
                                                                    n_pcs_subspace_max = n_pcs_subspace_max, 
                                                                    do_pca_after_project_on_subspace=False,
                                                                    PLOT_STEPS=False, SANITY=False,
                                                                    reshape_method=reshape_method,
                                                                    pca_tbin_slice=tbin_slide, return_raw_data=False,
                                                                    proj_twind=None, proj_tbindur=None, proj_tbin_slice=None)      
                    pathis = pathis.slice_by_labels_filtdict(filtdict_test)
                    dflab = pathis.Xlabels["trials"]
                    Xredu = pathis.X.squeeze(axis=2).T
                    LIST_DIMS = [(0,1), (2,3)]
                elif METHOD == "umap":
                    # Embed data
                    _, pathis, _, _, _ = pa.dataextract_state_space_decode_flex(twind_overall, tbin_dur, tbin_slide, reshape_method,
                                                                    pca_reduce=True, pca_frac_var_keep=pca_frac_var_keep,
                                                                    extra_dimred_method=METHOD,
                                                                    umap_n_neighbors=umap_n_neighbors)
                    pathis = pathis.slice_by_labels_filtdict(filtdict_test)
                    dflab = pathis.Xlabels["trials"]
                    Xredu = pathis.X.squeeze(axis=2).T
                    LIST_DIMS = [(0,1)]
                else:
                    print(METHOD)
                    assert False
                print("Shape of Xredu: ", Xredu.shape)

                # if METHOD=="umap":
                #     # Embed data
                #     Xredu, _ = dimredgood_nonlinear_embed_data(X, METHOD=METHOD, n_components=2, umap_n_neighbors=umap_n_neighbors)
                # elif METHOD=="pca":
                #     # Xredu = pathis.X
                #     Xredu = pathis.X.squeeze(axis=2).T
                # else:
                #     print(METHOD)
                #     assert False

                ##### Plot scalars
                savedir = f"{SAVEDIR_ANALYSIS}/raw_pca_plots/{br}-{ev}-twind={'_'.join([str(t) for t in twind_overall])}-METHOD={METHOD}"
                print(savedir)
                os.makedirs(savedir, exist_ok=True)

                for var_color, var_subplot in list_var_color_var_subplot:
                    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_scalar_WRAPPER
                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu, dflab, var_color, savedir,
                                            vars_subplot=var_subplot, list_dims=LIST_DIMS, n_min_per_levo=n_min_per_levo)
                plt.close("all")    


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    combine_areas = int(sys.argv[3])==1

    ### PARAMS
    # if animal == "Pancho":
    #     combine_areas = False
    # else:
    #     combine_areas = True

    which_level = "saccade_fix_on"
    # fr_normalization_method = "each_time_bin"
    for fr_normalization_method in ["across_time_bins", "each_time_bin"]:

        ### LOAD
        DFallpa = load_handsaved_wrapper(animal=animal, date=date, version=which_level, combine_areas=combine_areas, use_time=True)

        ### PREPROCESS
        from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels
        dfpa_concatbregion_preprocess_clean_bad_channels(DFallpa, PLOT=False)

        for pa in DFallpa["pa"]:
            pa.X = pa.X**0.5

        from neuralmonkey.classes.population_mult import dfallpa_preprocess_fr_normalization
        plot_savedir = "/tmp"
        dfallpa_preprocess_fr_normalization(DFallpa, fr_normalization_method, plot_savedir)    

        ### PLOTS
        SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/eyetracking_analyses/SUBSPACES/{animal}-{date}-combine={combine_areas}-wl={which_level}-norm={fr_normalization_method}"
        os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
        plot_all(DFallpa, SAVEDIR_ANALYSIS)


