"""
DEcoding -- Quickly test decoding of both shape and motor-related params for each substroke.
Mainly, condition on motor bin and decode substroke shape, or vice version.

Also implements dim reduction (PCA and dPCA) before decoding.

This is first pass at a good set of decoding code.

Updated: Main decode stuff in analy_decode_script.py
(This is all very scratch, but can move stuff from here to there).

Notebook: 240213_snippets_decode_substrokes

LOG:
2/27/24 - Should consoldate all stuff here to analy_decode_script.py (almost done),
especially he subspace dim reduction stuff (see above).
"""

import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt

DEBUG = False

def check_var_conti_discr(var_decode):
    """

    """

    if "binned" in var_decode or "_bin" in var_decode:
        return "discr"
    elif isinstance(dflab[var_decode].values[0], (str, tuple)):
        return "discr"
    elif var_decode in ["di_an_ci_ve_bin", "di_an_ci_binned", "dist_angle", "shape", "gridloc", "angle_binned", "distcum_binned", "shape_oriented", "gridsize"]:
        return "discr"
    elif var_decode in ["velocity", "angle", "distcum", "circ_signed"]:
        return "conti"
    else:
        print(var_decode)
        assert False


def load_saved_data(animal, date, prune_to_first_substroke, twind, question="SS_shape"):
    """ Load previously saved decoding data (using script below).
    Quick and dirtly.
    """
    SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/DECODE/{animal}-{date}/{question}/firstsubstrk_{prune_to_first_substroke}-twind_{twind}"

    path = f"{SAVEDIR_ANALYSES}/dfres.pkl"
    dfres = pd.read_pickle(path)

    path = f"{SAVEDIR_ANALYSES}/dfres_single.pkl"
    dfres_single = pd.read_pickle(path)

    return dfres, dfres_single, SAVEDIR_ANALYSES

def plot_summary_decode_results(dfres, dfres_single, SAVEDIR):
    """ Plot summary plots for dcoding across all analyses.
    PARAMS:
    - SAVEDIR, the based dir (i.e.w ill append summary_plots)
    """

    from neuralmonkey.scripts.analyquick_decode_substrokes import load_saved_data
    import os
    import matplotlib.pyplot as plt
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.snstools import rotateLabel
    # import pandas as pd

    # Plots
    ##### Make all plots (decode)m
    savedir = f"{SAVEDIR}/summary_plots"
    os.makedirs(savedir, exist_ok=True)

    from pythonlib.tools.snstools import rotateLabel
    import seaborn as sns
    fig = sns.catplot(data=dfres_single, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others")
    plt.axhline(0)
    plt.axhline(1)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/single_decoder_across_all_othervar-1.pdf")

    fig = sns.catplot(data=dfres_single, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others", kind="bar", ci=68)
    plt.axhline(0)
    plt.axhline(1)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/single_decoder_across_all_othervar-2.pdf")

    fig = sns.catplot(data=dfres_single, x="bregion", y="score_adjusted_mean", row="var_decode", col="vars_others", kind="bar", ci=68)
    plt.axhline(0)
    plt.axhline(1)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/single_decoder_across_all_othervar-score_adjusted-1.pdf")

    ################################
    import seaborn as sns
    fig = sns.catplot(data=dfres, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others")
    plt.axhline(0)
    plt.axhline(1)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/separate_decoder_each_othervar-1.pdf")

    fig = sns.catplot(data=dfres, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others", kind="bar", ci=68)
    plt.axhline(0)
    plt.axhline(1)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/separate_decoder_each_othervar-2.pdf")

    fig = sns.catplot(data=dfres, x="bregion", y="score_adjusted_mean", row="var_decode", col="vars_others", kind="bar", ci=68)
    plt.axhline(0)
    plt.axhline(1)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/separate_decoder_each_othervar-score_adjusted-1.pdf")

    plt.close("all")


if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])

    ############### PARAMS
    # animal = "Diego"
    # date = 230616
    exclude_bad_areas = True
    SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
    bin_by_time_dur = 0.05
    bin_by_time_slide = 0.025
    which_level = "substroke"

    # METHOD 1 - Standard, running separately for each PA
    question = "SS_shape"
    slice_agg_slices = None
    slice_agg_vars_to_split = None
    slice_agg_concat_dim = None

    # list_time_windows = [(-0.3, 0.)]
    # list_time_windows = [(-0.3, 0), (0., 0.3)]
    list_time_windows = [(-0.3, 0)]
    events_keep = ["00_substrk"]
    # list_time_windows = [(-0.3, 0.)]
    # events_keep = ["06_on_strokeidx_0"]
    print(list_time_windows)

    ev = "00_substrk"
    thresh_frac_var = 0.85 # PCA, cumvar
    n_min_across_all_levs_var = 5

    # Demixed PCA
    list_dpca_marginalization = ["shape", None]

    ########################################## RUN
    # Load q_params
    from neuralmonkey.analyses.rsa import rsagood_questions_dict, rsagood_questions_params
    q_params = rsagood_questions_dict(animal, date, question)[question]
    # Load data
    from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper

    combine_into_larger_areas = True
    HACK_RENAME_SHAPES = False
    DFallpa = dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows, which_level=which_level,
                                              combine_into_larger_areas=combine_into_larger_areas,
                                              exclude_bad_areas=exclude_bad_areas, bin_by_time_dur=bin_by_time_dur,
                                              bin_by_time_slide=bin_by_time_slide, SPIKES_VERSION=SPIKES_VERSION,
                                              HACK_RENAME_SHAPES=HACK_RENAME_SHAPES)



    #%%
    # Load DPallPA as in above, then run from here
    #%%
    SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/DECODE"

    #%%
    list_br = sorted(DFallpa["bregion"].unique().tolist())
    if DEBUG:
        list_br = ["M1"]


    for prune_to_first_substroke in [True, False]:


        if True:
            # Continuous
            list_var_decode = ["velocity", "angle", "distcum", "circ_signed"]
            list_vars_others = [["shape"], ["shape"], ["shape"], ["shape"]]
            # list_vars_others = [["dummy"], ["dummy"], ["dummy"], ["dummy"]]
        else:
            # Discrete
            list_var_decode = ["shape", "di_an_ci_ve_bin"]
            list_vars_others = [["di_an_ci_ve_bin"], ["shape"]]

        # Combine all
        if DEBUG:
            list_var_decode = ["shape"] + ["shape"] + ["velocity"]
            list_vars_others = [["ss_this_ctxt"], ["di_an_ci_ve_bin"]] + [["shape"]]
        else:
            # list_var_decode = ["shape", "di_an_ci_ve_bin"] + ["velocity", "angle", "distcum", "circ_signed"]
            # list_vars_others = [["di_an_ci_ve_bin"], ["shape"]] + [["shape"], ["shape"], ["shape"], ["shape"]]
            list_var_decode = ["shape", "shape", "di_an_ci_ve_bin"] + ["angle"]
            list_vars_others = [["ss_this_ctxt"], ["di_an_ci_ve_bin"], ["shape"]] + [["shape"]]

        if prune_to_first_substroke==False:
            # Then append index within stroke
            tmp =[]
            for x in list_vars_others:
                tmp.append(list(x) + ["index_within_stroke"])
            list_vars_others = tmp

        from pythonlib.tools.listtools import tabulate_list
        # Count the lowest n data across classes.
        from neuralmonkey.analyses.decode_good import decode_categorical

        if DEBUG:
            list_time_windows = [list_time_windows[0]]

        for twind in list_time_windows:

            for dpca_marginalization in list_dpca_marginalization:
                # a = stringify_list(list_time_windows, return_as_str=True)
                # b = stringify_list(events_keep, return_as_str=True)
                SAVEDIR = f"{SAVEDIR_ANALYSES}/{animal}-{date}/{question}/firstsubstrk_{prune_to_first_substroke}-twind_{twind}-dpca_marg_{dpca_marginalization}"
                os.makedirs(SAVEDIR, exist_ok=True)

                params_this = {
                    "n_min_across_all_levs_var":n_min_across_all_levs_var,
                    "thresh_frac_var":thresh_frac_var,
                    "prune_to_first_substroke":prune_to_first_substroke,
                }
                writeDictToTxt(params_this, f"{SAVEDIR}/params.txt")

                ##### Condition on one variable and test decoding (within that variable)

                RES_SINGLE = [] # Single decoder across all data (after pruning to have conjunctions)
                RES = [] # Separate decoder for each lev of conj
                for br in list_br:
                    # br = "PMv"

                    savedir_preprocess = f"{SAVEDIR}/preprocess_plots/{br}"
                    os.makedirs(savedir_preprocess, exist_ok=True)

                    from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization
                    from pythonlib.tools.pandastools import append_col_with_grp_index
                    from neuralmonkey.analyses.rsa import preprocess_rsa_prepare_popanal_wrapper
                    from neuralmonkey.scripts.analy_dpca_script_quick import plothelper_get_variables

                    # Extract PA
                    tmp = DFallpa[(DFallpa["bregion"]==br) & (DFallpa["event"]==ev) & (DFallpa["twind"]==twind)]
                    assert len(tmp)==1
                    pa = tmp["pa"].values[0]

                    # Restrict analysis to just first substroke
                    if prune_to_first_substroke:
                        pa = pa.slice_by_labels("trials", "index_within_stroke", [0])

                    ###### USE PCS OR DEMIXED PCS
                    if dpca_marginalization is not None:
                        ##### Use dPCA projections for decoding
                        from neuralmonkey.scripts.analy_dpca_script_substrokes import dpca_compute_pa_to_space, transform_from_pa, transform_trial

                        dpca, Z, R, trialR, map_var_to_lev, map_grp_to_idx, params_dpca, panorm = dpca_compute_pa_to_space(
                            pa, [dpca_marginalization], keep_all_margs=False)
                        if dpca_marginalization == "shape":
                            marginalization = "s"
                        elif dpca_marginalization == "dist_angle":
                            marginalization = "m" # motor
                        else:
                            print(dpca_marginalization)
                            assert False

                        # First, convert to final data using PA (e.g., scalar)
                        panorm_scal = panorm.agg_wrapper("times")
                        # dflab = panorm_scal.Xlabels["trials"]
                        trialX_proj = transform_from_pa(dpca, panorm_scal, marginalization) # (ndims, ntrials, 1)

                        # Prune to dims with >0.1% explained variance, or 3, whichever larger
                        tmp = np.argwhere(np.array(dpca.explained_variance_ratio_[marginalization])>0.001)
                        if len(tmp)==0:
                            ndims = 4
                        else:
                            ndims = max(tmp)+1
                            ndims = max([4, ndims]) # take at least 3 dimensions
                            ndims = int(ndims)
                        trialX_proj = trialX_proj[:ndims, :, :]

                        fig, ax = plt.subplots()
                        ax.plot(dpca.explained_variance_ratio_[marginalization])
                        ax.axvline(ndims, color="r")
                        ax.set_title("var exp, dpca, vline=ndims taken")
                        savefig(fig, f"{savedir_preprocess}/demixed_pca_explainedvar.pdf")

                        assert trialX_proj.shape[2]==1
                        trialX_proj = trialX_proj.squeeze()
                        X_orig = trialX_proj.T # (ntrials, nchans)

                    else:
                        # PCA.
                        # Standard preprocessing
                        if False:
                            # Dont do this; will do later in context of decoding.
                            pa, res_check_tasksets, res_check_effectvars = preprocess_rsa_prepare_popanal_wrapper(pa, **q_params)
                        _, panorm_scal, _, _, _, _ = popanal_preprocess_scalar_normalization(pa, None,
                                                                                             DO_AGG_TRIALS=False)

                        # PCA
                        trialX = panorm_scal.X.copy()
                        assert trialX.shape[2]==1
                        trialX = trialX.squeeze()
                        X_orig = trialX.T # (ntrials, nchans)

                        # Prune with PCA
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=None)
                        Xpca = pca.fit_transform(X_orig) # (ntrials, ndims)
                        cumvar = np.cumsum(pca.explained_variance_ratio_)
                        npcs_keep = np.argmin(np.abs(cumvar - thresh_frac_var))

                        fig, ax = plt.subplots()
                        ax.plot(pca.explained_variance_ratio_)
                        ax.axvline(npcs_keep, color="r")
                        X_orig = Xpca[:, :npcs_keep]
                        savefig(fig, f"{savedir_preprocess}/pca_explainedvar.pdf")


                    # Get labels
                    dflab = panorm_scal.Xlabels["trials"].copy()

                    for var_decode, vars_others in zip(
                            list_var_decode,
                            list_vars_others):

                        map_levother_to_labels = {}

                        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
                        # var_decode = "shape"
                        # vars_others = ["di_an_ci_ve_bin"]

                        conti_or_discr = check_var_conti_discr(var_decode)

                        if conti_or_discr=="discr":
                            # var_decode = "di_an_ci_ve_bin"
                            # vars_others = ["shape"]

                            Labels_orig = panorm_scal.Xlabels["trials"][var_decode]
                            assert len(Labels_orig)==X_orig.shape[0]
                            Labels_orig_int, map_int_to_lab = Labels_orig.factorize()
                            map_int_to_lab = {i:lab for i, lab in enumerate(map_int_to_lab)}

                            writeDictToTxt({"map_int_to_lab":map_int_to_lab}, f"{savedir_preprocess}/map_int_to_lab-var_{var_decode}-var_others_{'|'.join(vars_others)}.txt")

                            # var_decode = "di_an_ci_ve_bin"
                            # var_decode = "velocity"
                            path_conj = f"{savedir_preprocess}/conjunctions-var_{var_decode}-var_others_{'|'.join(vars_others)}.png"
                            dflab_pruned, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var_decode,
                                                                                    vars_others,
                                                                                    n_min_across_all_levs_var=n_min_across_all_levs_var,
                                                                                    lenient_allow_data_if_has_n_levels=2,
                                                                                    prune_levels_with_low_n=True,
                                                                                    plot_counts_heatmap_savepath=path_conj)

                            if len(dflab_pruned)>0:

                                ### FIRST, a single decoder over entire data (after pruning to levels of othervar that have some data across var)
                                if len(dflab_pruned)==0:
                                    print(len(dflab))
                                    print(var_decode)
                                    print(vars_others)
                                    print("see conjunctions which failed at:", path_conj)
                                    assert False

                                inds = dflab_pruned["_index"].tolist()
                                X = X_orig[inds, :]
                                labels = Labels_orig_int[inds]
                                plot_resampled_data_path_nosuff = f"{savedir_preprocess}/var-{var_decode}-ALL_DATA-{br}"
                                res = decode_categorical(X, labels, n_min_across_all_levs_var, plot_resampled_data_path_nosuff)
                                plt.close("all")

                                # avearge over folds
                                score_mean = np.mean([r["score_xval"] for r in res])
                                score_std = np.std([r["score_xval"] for r in res])
                                score_adjusted_mean = np.mean([r["score_xval_adjusted"] for r in res])

                                RES_SINGLE.append({
                                    "score_xval_mean":score_mean,
                                    "score_adjusted_mean":score_adjusted_mean,
                                    "score_xval_std":score_std,
                                    "nclasses":len(set(labels)),
                                    "n_dat":len(labels),
                                    "n_splits":res[0]["n_splits"],
                                    "n_min_across_labs":res[0]["n_min_across_labs"],
                                    "n_max_across_labs":res[0]["n_max_across_labs"],
                                    "bregion":br,
                                    "var_decode":var_decode,
                                    "vars_others":tuple(vars_others)
                                })
                                PLOT_RESAMPLED_DATA = False
                                # For each level of "others" do decoding

                                for levo, dfthis in dict_dfthis.items():

                                    # Do decode/test
                                    inds = dfthis["_index"].tolist()
                                    X = X_orig[inds, :]
                                    labels = Labels_orig_int[inds]

                                    # Sanity check, same data across all bregions.
                                    from pythonlib.tools.checktools import check_objects_identical
                                    if levo in map_levother_to_labels.items():
                                        if not check_objects_identical(map_levother_to_labels[levo], labels):
                                            print(map_levother_to_labels[levo])
                                            print(labels)
                                            assert False
                                    else:
                                        map_levother_to_labels[levo]=labels

                                    plot_resampled_data_path_nosuff = f"{savedir_preprocess}/var-{var_decode}-lev_other_{levo}-{br}"
                                    res = decode_categorical(X, labels, n_min_across_all_levs_var, plot_resampled_data_path_nosuff)
                                    plt.close("all")

                                    # avearge over folds
                                    score_mean = np.mean([r["score_xval"] for r in res])
                                    score_std = np.std([r["score_xval"] for r in res])
                                    score_adjusted_mean = np.mean([r["score_xval_adjusted"] for r in res])

                                    RES.append({
                                        "lev_other":levo,
                                        "score_xval_mean":score_mean,
                                        "score_adjusted_mean":score_adjusted_mean,
                                        "score_xval_std":score_std,
                                        "nclasses":len(set(labels)),
                                        "n_dat":len(labels),
                                        "n_splits":res[0]["n_splits"],
                                        "n_min_across_labs":res[0]["n_min_across_labs"],
                                        "n_max_across_labs":res[0]["n_max_across_labs"],
                                        "bregion":br,
                                        "var_decode":var_decode,
                                        "vars_others":tuple(vars_others)
                                    })
                        elif conti_or_discr=="conti":
                            # Continuous -- re-bin within each level of othervar...
                            PLOT_RESAMPLED_DATA = False
                            # For each level of "others" do decoding
                            n_min_across_all_levs_var = 8 # min n in each class

                            # For motor decoding, use bins local to the sahpe (i.e., recalcualte teh bins),
                            # and do separately for each feature.

                            Labels_orig = dflab[var_decode]
                            assert len(Labels_orig)==X_orig.shape[0]

                            # SKIP THis, since it resets the indices...
                            # from pythonlib.tools.pandastools import extract_with_levels_of_var_good
                            # dfthis, inds_keep = extract_with_levels_of_var_good(dflab, vars_others, n_min_across_all_levs_var*2)


                            ### FIRST, a single decoder over entire data (after pruning to levels of othervar that have some data across var)
                            X = X_orig
                            labels = Labels_orig

                            # Bin data.
                            from pythonlib.tools.nptools import bin_values, bin_values_by_rank
                            nbins = int(np.floor(len(labels)/n_min_across_all_levs_var))
                            if nbins>6:
                                nbins = 6
                            labels = bin_values_by_rank(labels, nbins=nbins)

                            plot_resampled_data_path_nosuff = f"{savedir_preprocess}/var-{var_decode}-ALL_DATA-{br}"
                            res = decode_categorical(X, labels, n_min_across_all_levs_var, plot_resampled_data_path_nosuff)
                            plt.close("all")

                            # avearge over folds
                            score_mean = np.mean([r["score_xval"] for r in res])
                            score_std = np.std([r["score_xval"] for r in res])
                            score_adjusted_mean = np.mean([r["score_xval_adjusted"] for r in res])

                            RES_SINGLE.append({
                                "score_xval_mean":score_mean,
                                "score_adjusted_mean":score_adjusted_mean,
                                "score_xval_std":score_std,
                                "nclasses":len(set(labels)),
                                "n_dat":len(labels),
                                "n_splits":res[0]["n_splits"],
                                "n_min_across_labs":res[0]["n_min_across_labs"],
                                "n_max_across_labs":res[0]["n_max_across_labs"],
                                "bregion":br,
                                "var_decode":var_decode,
                                "vars_others":tuple(vars_others)
                            })

                            from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
                            groupdict = grouping_append_and_return_inner_items(dflab, vars_others)


                            for levo, inds in groupdict.items():

                                if len(inds)>=n_min_across_all_levs_var*2:
                                    # Otherwise cant get >1 class for decoding

                                    # Do decode/test
                                    X = X_orig[inds, :]
                                    labels = Labels_orig[inds]

                                    # Bin data.
                                    from pythonlib.tools.nptools import bin_values, bin_values_by_rank
                                    nbins = int(np.floor(len(labels)/n_min_across_all_levs_var))
                                    if nbins>6:
                                        nbins = 6
                                    labels = bin_values_by_rank(labels, nbins=nbins)

                                    # Sanity check, same data across all bregions.
                                    from pythonlib.tools.checktools import check_objects_identical
                                    if levo in map_levother_to_labels.items():
                                        if not check_objects_identical(map_levother_to_labels[levo], labels):
                                            print(map_levother_to_labels[levo])
                                            print(labels)
                                            assert False
                                    else:
                                        map_levother_to_labels[levo]=labels

                                    from neuralmonkey.analyses.decode_good import decode_categorical
                                    plot_resampled_data_path_nosuff = f"{savedir_preprocess}/var-{var_decode}-lev_other_{levo}-{br}"
                                    res = decode_categorical(X, labels, n_min_across_all_levs_var, plot_resampled_data_path_nosuff)
                                    plt.close("all")

                                    # avearge over folds
                                    score_mean = np.mean([r["score_xval"] for r in res])
                                    score_std = np.std([r["score_xval"] for r in res])
                                    score_adjusted_mean = np.mean([r["score_xval_adjusted"] for r in res])

                                    RES.append({
                                        "lev_other":levo,
                                        "score_xval_mean":score_mean,
                                        "score_adjusted_mean":score_adjusted_mean,
                                        "score_xval_std":score_std,
                                        "nclasses":len(set(labels)),
                                        "n_dat":len(labels),
                                        "n_splits":res[0]["n_splits"],
                                        "n_min_across_labs":res[0]["n_min_across_labs"],
                                        "n_max_across_labs":res[0]["n_max_across_labs"],
                                        "bregion":br,
                                        "var_decode":var_decode,
                                        "vars_others":tuple(vars_others)
                                    })
                        else:
                            print(conti_or_discr)
                            assert False


                ##### SAVE DATA
                savedir = f"{SAVEDIR}/summary_plots"
                os.makedirs(savedir, exist_ok=True)
                dfres_single = pd.DataFrame(RES_SINGLE)
                dfres = pd.DataFrame(RES)

                dfres_single.to_csv(f"{SAVEDIR}/dfres_single.csv")
                dfres.to_csv(f"{SAVEDIR}/dfres.csv")

                pd.to_pickle(dfres_single, f"{SAVEDIR}/dfres_single.pkl")
                pd.to_pickle(dfres, f"{SAVEDIR}/dfres.pkl")

                ####### PLOTS
                plot_summary_decode_results(dfres, dfres_single, SAVEDIR)

                #
                # from pythonlib.tools.snstools import rotateLabel
                # import seaborn as sns
                # fig = sns.catplot(data=dfres_single, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others")
                # plt.axhline(0)
                # plt.axhline(1)
                # rotateLabel(fig)
                # savefig(fig, f"{savedir}/single_decoder_across_all_othervar-1.pdf")
                #
                # fig = sns.catplot(data=dfres_single, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others", kind="bar", ci=68)
                # plt.axhline(0)
                # plt.axhline(1)
                # rotateLabel(fig)
                # savefig(fig, f"{savedir}/single_decoder_across_all_othervar-2.pdf")
                #
                # fig = sns.catplot(data=dfres_single, x="bregion", y="score_adjusted_mean", row="var_decode", col="vars_others", kind="bar", ci=68)
                # plt.axhline(0)
                # plt.axhline(1)
                # rotateLabel(fig)
                # savefig(fig, f"{savedir}/single_decoder_across_all_othervar-score_adjusted-1.pdf")
                #
                # ################################
                # import seaborn as sns
                # fig = sns.catplot(data=dfres, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others")
                # plt.axhline(0)
                # plt.axhline(1)
                # rotateLabel(fig)
                # savefig(fig, f"{savedir}/separate_decoder_each_othervar-1.pdf")
                #
                # fig = sns.catplot(data=dfres, x="bregion", y="score_xval_mean", hue="nclasses", row="var_decode", col="vars_others", kind="bar", ci=68)
                # plt.axhline(0)
                # plt.axhline(1)
                # rotateLabel(fig)
                # savefig(fig, f"{savedir}/separate_decoder_each_othervar-2.pdf")
                #
                # fig = sns.catplot(data=dfres, x="bregion", y="score_adjusted_mean", row="var_decode", col="vars_others", kind="bar", ci=68)
                # plt.axhline(0)
                # plt.axhline(1)
                # rotateLabel(fig)
                # savefig(fig, f"{savedir}/separate_decoder_each_othervar-score_adjusted-1.pdf")
                #
                # plt.close("all")
