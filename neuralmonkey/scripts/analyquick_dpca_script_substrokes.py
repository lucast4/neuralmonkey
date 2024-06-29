"""
Quick plotting of state space plots of population neural data for substrokes,
coloring by (motor variables, shapes) and subsplots
splitting by (motor, shapes).
Goal: inspection by eye whether shape categories are different even if control for motor varibles.

This led to decoding script (analyquick_decode_substrokes.py).

More general state-space plots (focusing on DPCA): analy_dpca_script_quick

COpied from notebook:
240128_snippets_demixed_PCA

"""

from neuralmonkey.scripts.analy_dpca_script_quick import preprocess_pa_to_frtensor
from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
from neuralmonkey.classes.session import load_mult_session_helper
import os
import pandas as pd
from neuralmonkey.classes.population_mult import snippets_extract_popanals_split_bregion_twind

from pythonlib.tools.plottools import savefig
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from neuralmonkey.scripts.analy_dpca_script_quick import plot_statespace_2d_overlaying_all_othervar

# Get results for a single pa
from neuralmonkey.scripts.analy_dpca_script_quick import plothelper_get_variables
from neuralmonkey.analyses.state_space_good import  trajgood_plot_colorby_splotby_scalar

from pythonlib.tools.plottools import savefig
from pythonlib.tools.pandastools import append_col_with_grp_index
from neuralmonkey.analyses.rsa import preprocess_rsa_prepare_popanal_wrapper

from pythonlib.tools.listtools import stringify_list

# Load q_params
from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
# Load data
from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper



##### Single trial analysis
def transform_from_pa(dpca, pa, marginalization):
    """

    :param dpca:
    :param pa: (nchans, ntrials, ntimes)
    :param marginalization:
    :return:
    - trialX_proj, (ndims_reduced, ntrials, ntimes)
    """

    if False:
        # NOT WORKING!!
        # Vectorized
        trialX = np.transpose(pa.X, [1, 0, 2]) # (trials, chans, times)
        trialX_proj = transform_trial(dpca, trialX, "s")
    else:
        # Debuggin, Redo it without using trialR
        D = dpca.D[marginalization] # (nchan, ndim)

        outs = []
        for trial in range(pa.X.shape[1]):
            x = pa.X[:,trial,:] # (nchans, ntimes)
            x_proj = np.dot(D.T, x) # (ndim, ntimes)

            outs.append(x_proj)

        trialX_proj = np.stack(outs, axis=0) # (ntrials, ndims, ntimes)
        trialX_proj = np.transpose(trialX_proj, (1, 0, 2)) # (ndims, ntrials, ntimes)

        assert trialX_proj.shape[1]==pa.X.shape[1]
        assert trialX_proj.shape[0]==D.shape[1]
        assert trialX_proj.shape[2]==pa.X.shape[2]

    assert not np.any(np.isnan(trialX_proj))

    return trialX_proj

def transform_trial(dpca, X, marginalization):
    """

    :param dpca:
    :param X: shape (ntrials, nchans, ...)
    :param marginalization:
    :return:
    """

    assert not np.any(np.isnan(X))
    # print(X.shape) # (trials, olddim(i.e., chans), features)

    axis_chans = 1
    D, Xr         = dpca.D[marginalization], X.reshape((X.shape[axis_chans],-1))

    print(D.shape)
    print(Xr.shape)
    newshape = tuple((X.shape[0], D.shape[1]) + X.shape[2:])# (trials, newdim, features)
    print(newshape)
    X_transformed = np.dot(D.T, Xr).reshape(newshape)

    # print(X_transformed.shape)
    # print(X.shape)
    assert len(X_transformed.shape)==len(X.shape)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[2:] == X.shape[2:]

    return X_transformed

def dpca_compute_pa_to_space(pa, effect_vars, keep_all_margs=True):
    """ [GOOD]
    Given PA, compute dPCA space, and project data to this space
    PARAMS:
    - pa, Raw. Here does all required preprocessing of pa.
    - effect_vars, the vars which will be used in dPCA to find demixed subspaces.
    NOTE: will also be used for preprocessing...
    """
    from neuralmonkey.scripts.analy_dpca_script_quick import preprocess_pa_to_frtensor
    from dPCA import dPCA

    assert isinstance(effect_vars, (list, tuple))

    # Clean up PA
    # Prune to remove low N data.
    pa, res_check_tasksets, res_check_effectvars = preprocess_rsa_prepare_popanal_wrapper(pa,
                                                                      effect_vars,
                                                                      False,
                                                                      False, None, None, False, None)


    # Preprocess
    R, trialR, map_var_to_lev, map_grp_to_idx, params_dpca, PAnorm = preprocess_pa_to_frtensor(pa,
                                                                                               effect_vars,
                                                                                               keep_all_margs=keep_all_margs)

    # print("---- OUTPUT")
    # print("R.shape:", R.shape)
    # print("trialR:", trialR.shape)
    # print("map_var_to_lev:", map_var_to_lev)
    # print("map_grp_to_idx:", map_grp_to_idx)
    # print("params_dpca:", params_dpca)
    #
    # assert not np.any(np.isnan(R))
    #
    # print(np.mean(trialR, axis=3))
    # assert False
    #
    # tmp = np.mean(trialR, axis=0)
    # print(np.isnan(tmp[0, :, 0]))
    # print(np.where(np.isnan(tmp[0, :, 0])))
    # print(trialR[0, 0, 15, :])
    # print(tmp[0, 15, 0])

    #### Fit model
    # We then instantiate a dPCA model where the two parameter axis are labeled by 's' (stimulus) and 't' (time) respectively. We set regularizer to 'auto' to optimize the regularization parameter when we fit the data.
    labels = params_dpca["labels"]
    join = params_dpca["join"]
    n_components = params_dpca["n_components"]
    dpca = dPCA.dPCA(labels=labels, regularizer='auto', join=join, n_components=n_components)
    dpca.protect = ['t']

    # Now fit the data (R) using the model we just instatiated. Note that we only need trial-to-trial data when we want to optimize over the regularization parameter.
    Z = dpca.fit_transform(R, trialR)

    plothelper_get_variables(Z, effect_vars, params_dpca) # Add variables to params

    return dpca, Z, R, trialR, map_var_to_lev, map_grp_to_idx, params_dpca, PAnorm
#
# effect_vars = ["shape"]
# dpca_compute_pa_to_space(pa, effect_vars)


if __name__=="__main__":

    assert False, "update this with nmoteobok 240128_snips..."

    from pythonlib.tools.plottools import savefig
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
    import os
    import sys

    SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/dPCA"

    ############### PARAMS
    exclude_bad_areas = True
    SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
    bin_by_time_dur = 0.05
    bin_by_time_slide = 0.025
    which_level = "substroke"
    combine_into_larger_areas = True
    HACK_RENAME_SHAPES = False

    if False:
        # METHOD 2 - Merging across time windows into single PA bofre doing dPCA
        question = "SP_shape_loc_TIME"
        slice_agg_slices = [
            ("trial", "03_samp", (-0.3, 0.5)),
            ("trial", "04_go_cue", (-0.45, 0.25)),
            ("trial", "06_on_strokeidx_0", (-0.25, 0.7))
        ]
        slice_agg_vars_to_split = ["bregion"]
        slice_agg_concat_dim = "times"

        list_time_windows = [sl[2] for sl in slice_agg_slices]
        events_keep = list(set([sl[1] for sl in slice_agg_slices]))
        print(list_time_windows)
    else:
        # METHOD 1 - Standard, running separately for each PA
        question = "SS_shape"
        slice_agg_slices = None
        slice_agg_vars_to_split = None
        slice_agg_concat_dim = None

        # list_time_windows = [(-0.3, 0.)]
        events_keep = ["00_substrk"]
        # list_time_windows = [(-0.3, 0.)]
        # events_keep = ["06_on_strokeidx_0"]


    ##### Single trial analysis

    # Load q_params
    # Load data
    from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper



    # Get results for a single pa
    from neuralmonkey.scripts.analy_dpca_script_quick import plothelper_get_variables


    SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/dPCA"
    keep_all_margs = False

    ############### PARAMS
    list_failures =[]
    list_errs = []

    for animal in ["Pancho", "Diego"]:
        if animal=="Diego":
            dates = [230615, 230616, 230618, 230619]
        elif animal=="Pancho":
            dates = [220716, 220715, 220718, 220719, 220918, 221217]
        else:
            assert False

        for date in dates:
            from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
            q_params = rsagood_questions_dict(animal, date, question)[question]
            try:
                exclude_bad_areas = True
                SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
                bin_by_time_dur = 0.05
                bin_by_time_slide = 0.025
                which_level = "substroke"

                if False:
                    # METHOD 2 - Merging across time windows into single PA bofre doing dPCA
                    question = "SP_shape_loc_TIME"
                    slice_agg_slices = [
                        ("trial", "03_samp", (-0.3, 0.5)),
                        ("trial", "04_go_cue", (-0.45, 0.25)),
                        ("trial", "06_on_strokeidx_0", (-0.25, 0.7))
                    ]
                    slice_agg_vars_to_split = ["bregion"]
                    slice_agg_concat_dim = "times"

                    list_time_windows = [sl[2] for sl in slice_agg_slices]
                    events_keep = list(set([sl[1] for sl in slice_agg_slices]))
                    print(list_time_windows)
                else:
                    # METHOD 1 - Standard, running separately for each PA
                    question = "SS_shape"
                    slice_agg_slices = None
                    slice_agg_vars_to_split = None
                    slice_agg_concat_dim = None

                    list_time_windows = [(-0.3, 0.)]
                    events_keep = ["00_substrk"]
                    # list_time_windows = [(-0.3, 0.)]
                    # events_keep = ["06_on_strokeidx_0"]
                    print(list_time_windows)

                for list_time_windows in [
                    [(-0.3, 0.)],
                    [(0., 0.3)]
                    ]:
                    q_params = rsagood_questions_dict(animal, date, question)[question]

                    DFallpa = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows,
                                                              which_level=which_level,
                                                              combine_into_larger_areas=combine_into_larger_areas,
                                                              bin_by_time_dur=bin_by_time_dur,
                                                              bin_by_time_slide=bin_by_time_slide,
                                                              HACK_RENAME_SHAPES=HACK_RENAME_SHAPES)



                    a = stringify_list(list_time_windows, return_as_str=True)
                    b = stringify_list(events_keep, return_as_str=True)

                    SAVEDIR = f"{SAVEDIR_ANALYSES}/{animal}-{date}/{question}/{a}--{b}--{keep_all_margs}"

                    list_br = DFallpa["bregion"].unique().tolist()
                    # effect_vars = ["seqc_0_shape", "seqc_0_loc"]
                    # br = "PMv"
                    # ev = "06_on_strokeidx_0"
                    # br = "PMv"
                    ev = "00_substrk"

                    for br in list_br:
                        tmp = DFallpa[(DFallpa["bregion"]==br) & (DFallpa["event"]==ev)]
                        assert len(tmp)==1
                        pa = tmp["pa"].values[0]

                        pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], ["shape", "index_within_stroke"], "shape_idx", False)
                        pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"], ["distcum_binned", "angle_binned", "circularity_binned"], "di_an_ci_binned", False)

                        # Clean up PA
                        pa, res_check_tasksets, res_check_effectvars = preprocess_rsa_prepare_popanal_wrapper(pa, **q_params)

                        # Restrict analysis to just first substroke
                        if True:
                            pa = pa.slice_by_labels("trials", "index_within_stroke", [0])

                        for effect_vars in [
                            ["shape"],
                            ["dist_angle"]
                        ]:

                            if effect_vars == ["shape"]:
                                marginalization="s"
                            elif effect_vars == ["dist_angle"]:
                                marginalization="m"
                            else:
                                assert False

                            savedir = f"{SAVEDIR}/scatter_color_diff_ways/effect_vars-{'|'.join(effect_vars)}"
                            os.makedirs(savedir, exist_ok=True)

                            # Compute dPCs, and project data.
                            dpca, Z, R, trialR, map_var_to_lev, map_grp_to_idx, params_dpca, panorm = dpca_compute_pa_to_space(pa, effect_vars, keep_all_margs=keep_all_margs)
                            plothelper_get_variables(Z, effect_vars, params_dpca) # Add variables to params

                            ##### Single trial analysis

                            # First, convert to final data using PA (e.g., scalar)
                            pathis = panorm.agg_wrapper("times")

                            dflab = pathis.Xlabels["trials"]

                            trialX_proj = transform_from_pa(dpca, pathis, marginalization)
                            dim1 = 0
                            dim2 = 1
                            xs = trialX_proj[:, dim1]
                            ys = trialX_proj[:, dim2]

                            for color_var in ["angle", "distcum", "circ_signed", "velocity", "shape", "di_an_ci_ve_bin"]:
                                for subplot_var in ["bregion", "shape", "di_an_ci_ve_bin"]:
                                    # color_var = "seqc_0_shape"
                                    # subplot_var = "seqc_0_loc"
                                    # color_var = "shape"
                                    # subplot_var = "bregion"

                                    # color_var = "di_an_ci_binned"
                                    # subplot_var = "bregion"

                                    labels = dflab[color_var]

                                    fig, axes = trajgood_plot_colorby_splotby_scalar(xs, ys, labels, dflab[subplot_var],
                                                                                color_var, subplot_var, overlay_mean=False, SIZE=6)
                                    savefig(fig, f"{savedir}/{br}-color_{color_var}-splot_{subplot_var}.pdf")

                                    plt.close("all")
            except Exception as err:
                list_failures.append((animal, date))
                list_errs.append(err)
                pass