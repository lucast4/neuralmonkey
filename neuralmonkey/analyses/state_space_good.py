""" Mixture of things, related to population-level analyses.
- DECODING
- STATE SPACE PLOTS (trajectories)
- RSA
"""


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.expttools import load_yaml_config
from pythonlib.tools.plottools import makeColors, legend_add_manual, savefig, rotate_x_labels, rotate_y_labels
from pythonlib.tools.expttools import writeDictToYaml
from pythonlib.tools.pandastools import append_col_with_grp_index, convert_to_2d_dataframe
from pythonlib.tools.snstools import rotateLabel

# version_distance = "pearson"

def popanal_preprocess_scalar(PA, grouping_vars, subtract_mean_each_level_of_var ="IGNORE",
                              plot_example_chan=None,
                              plot_example_split_var=None):
    """ Preprocess PA, with different options for normalization ,etc
    PARAMS:
    - subtract_mean_each_level_of_var, eitehr None (ignore) or str, in which case will
    find mean fr for each level of this var, then subtract this mean from all datpts that
    have this level for this var. e.g., if subtract_mean_each_level_of_var=="gridloc", then
    de-means for each location.
    """

    if plot_example_chan==False:
        plot_example_chan = None
    # assert isinstance(plot_example_chan, bool) or isinstance(plot_example_chan, int)

    if subtract_mean_each_level_of_var is None:
        subtract_mean_each_level_of_var = "IGNORE"

    # 1) First, rescale all FR (as in Churchland stuff), but isntead of using
    # range, use std (like z-score)
    if False:
        # FR
        normvec = np.mean(np.mean(PA.X, axis=1, keepdims=True), axis=2, keepdims=True) # (chans, 1, 1)
    else:
        # STD (across all trials and times)
        normvec = np.std(np.reshape(PA.X, (PA.X.shape[0], -1)), axis=1) # (chans, 1, 1)
        assert len(normvec.shape)==1
        normvec = np.reshape(normvec, [normvec.shape[0], 1,1])
    normmin = np.percentile(normvec, [2.5]) # get the min, add this on to still have
    # higher FR neurons more important
    PAnorm = PA.copy()
    PAnorm.X = PAnorm.X/(normvec+normmin)

    # 1) subtract global mean (i.e., at each time bin)
    PAnorm = PAnorm.norm_subtract_trial_mean_each_timepoint()

    if False: # Replaced with rescale step above
        # 2) rescale (for each time bin, across all trials
        Xstd = np.std(PAnorm.X, axis=1, keepdims=True) # (chans, 1, times)
        if False:
            # Just z-score.
            PAnorm.X = PAnorm.X/Xstd
        else:
            # Greater penalty for low FR neurons. I.e., This solves problem
            # that z-score places low FR neruons at same scale as high FR.
            # instead, this penalizes low FR.
            fr_min_1 = np.percentile(np.mean(Xstd, axis=2), [2.5]).squeeze() # Lowest fr across channels.
            fr_min_2 = np.percentile(np.mean(PAnorm.X, axis=2), [2.5]).squeeze() # Lowest fr across channels.
            print(np.mean(Xstd, axis=2))
            print(np.mean(PAnorm.X, axis=2))
            assert False
            PAnorm.X = PAnorm.X/(Xstd + fr_min)

    # 3) Subtract mean based on grouping by other variable
    if not subtract_mean_each_level_of_var=="IGNORE":
        from neuralmonkey.classes.population import concatenate_popanals
        list_pa = PAnorm.split_by_label("trials", subtract_mean_each_level_of_var)[0]
        list_pa_norm = []
        for pa in list_pa:
            # subtract global mean
            list_pa_norm.append(pa.norm_subtract_trial_mean_each_timepoint())
        PAnorm = concatenate_popanals(list_pa_norm, "trials")

    ### GET SCALARS (by averaging over time and grouping by variables of interest)
    # - get single "pseudotrial" for each conjunctive level
    # vars = ["shape_oriented", "gridloc", "FEAT_num_strokes_task", "stroke_index"]
    # vars = ["shape_oriented", "gridloc"]
    PAagg, groupdict = PAnorm.slice_and_agg_wrapper("trials", grouping_vars, return_group_dict=True)
    # print("Sample sizes for each level of grouping vars")
    # for k,v in groupdict.items():
    #     print(k, " -- ", len(v))
    # assert False

    PAagg = PAagg.agg_wrapper("times") # mean over time --> (chans, trials)

    ######## PLOTS
    if plot_example_chan is not None:
        add_legend = True
        fig, axes = plt.subplots(2,2, figsize=(8,8))

        for pathis, ax in zip([PA, PAnorm, PAagg], axes.flatten()):
            pathis.plotwrapper_smoothed_fr_split_by_label("trials", plot_example_split_var,
                ax=ax, add_legend=add_legend, chan=plot_example_chan)
            ax.axhline(0)
    else:
        fig, axes = None, None

    return PAnorm, PAagg, fig, axes, groupdict

def rsa_convert_PA_to_Cl(PAagg, grouping_vars, version_distance="pearson"):
    """ Convert from scalar popanal to raw Cl and distance matrix Cl objects
    """
    from pythonlib.cluster.clustclass import Clusters

    for var in grouping_vars:
        assert var in PAagg.Xlabels["trials"].columns

    # Pull out data in correct format, and return as clusters.
    assert PAagg.X.shape[2]==1, "take mean over time first"
    X = PAagg.X.squeeze().T # (ndat, nchans)
    labels_rows = PAagg.Xlabels["trials"].loc[:, grouping_vars].values.tolist()
    labels_rows = [tuple(x) for x in labels_rows] # list of tuples
    labels_cols = PAagg.Chans # list of ints
    params = {
        "label_vars":grouping_vars,
    }
    Clraw = Clusters(X, labels_rows, labels_cols, ver="rsa", params=params)

    # Distnace matrix
    Clsim = Clraw.distsimmat_convert(version_distance=version_distance)
    #
    # if version_distance=="euclidian":
    #     from scipy.spatial import distance_matrix
    #     D = distance_matrix(X, X)
    # elif version_distance=="pearson":
    #     # correlation matrix
    #     D = np.corrcoef(X)
    # else:
    #     assert False
    # params = {
    #
    # }
    # Clsim = Clusters(D, labels_rows, labels_rows, ver="dist", params=params)
    # if doplot:
    #     figsim, X, labels_col, labels_row, ax = Clsim._plot_heatmap_data(D, labels_sorted,
    #                                                                labels_sorted, diverge=diverge,
    #                                                                zlims=zlims,
    #                                                                   rotation=90, rotation_y=0)
    # else:
    #     figsim = None

    return Clraw, Clsim



def rsa_plot_raw_matrix(PAagg, grouping_vars, sort_order, version_distance="pearson",
                        doplot=True):
    """ Plot the input data (i.e. not sim matrix), which must be
    already converted to scalar represntation, with options for sorting labels
    to allow visualziation of interesting patterns
    INPUT:

    """
    from pythonlib.tools.listtools import argsort_list_of_tuples
    from pythonlib.cluster.clustclass import Clusters

    # Pull out data in correct format, and return as clusters.
    X = PAagg.X.squeeze().T # (ndat, nchans)
    labels_rows = PAagg.Xlabels["trials"].loc[:, grouping_vars].values.tolist()
    labels_rows = [tuple(x) for x in labels_rows] # list of tuples
    labels_cols = PAagg.Chans # list of ints

    # Sort labels if needed
    # key =lambda x:(x[1], x[2], x[0])
    if sort_order is not None:
        key =lambda x:tuple([x[i] for i in sort_order])
        inds_sort = argsort_list_of_tuples(labels_rows, key)
    else:
        inds_sort = list(range(len(labels_rows)))

    labels_sorted = [labels_rows[i] for i in inds_sort]
    Xsort = X[inds_sort] # sort the X also

    Clraw = Clusters(Xsort, labels_sorted, labels_cols)

    # Plot
    if doplot:
        figraw, X, labels_col, labels_row, ax = Clraw._plot_heatmap_data(Xsort, labels_sorted,
                                                                   labels_cols, diverge=True,
                                                                      rotation=90, rotation_y=0)
    else:
        figraw = None

    #### Also plot the correlation matrix
    if version_distance=="euclidian":
        from scipy.spatial import distance_matrix
        D = distance_matrix(Xsort, Xsort)
        diverge = False

        inds = np.where(~np.eye(D.shape[0],dtype=bool))
        zmin = np.min(D[inds])
        zlims = (zmin, None)
    elif version_distance=="pearson":
        # correlation matrix
        D = np.corrcoef(Xsort)
        diverge=True
        zlims = (None, None)
    else:
        assert False

    # plot heatmap
    Clsim = Clusters(D, labels_sorted, labels_sorted)
    if doplot:
        figsim, X, labels_col, labels_row, ax = Clsim._plot_heatmap_data(D, labels_sorted,
                                                                   labels_sorted, diverge=diverge,
                                                                   zlims=zlims,
                                                                      rotation=90, rotation_y=0)
    else:
        figsim = None

    return Clraw, Clsim, figraw, figsim

# def rsa_distmat_quantify_same_diff_variables(Clsim, ind_var, ignore_diagonal=True):
#     """
#     PARAMS:
#     - ind_var = 0 # e..g, if each row is labeled with a tuple like (shape, loc), then if
#     ind_var==0, then this means "same" is defined as having same shape
#     """
#
#     # Collect mapping
#     map_pair_labels_to_indices = {} # (lab1, lab2) --> (col, row)
#     for i, lr in enumerate(Clsim.Labels):
#         for j, lc in enumerate(Clsim.LabelsCols):
#             # only take off diagonal
#             if ignore_diagonal:
#                 if i>=j:
#                     continue
#             else:
#                 if i>j:
#                     continue
#             map_pair_labels_to_indices[(lr, lc)] = (i, j)
#
#     # Find the coordinates of "same" and "diff" pairs.
#     # given a var dimension, get all indices that are "same" along that dimension
#     list_inds_same = []
#     list_inds_diff = []
#     for lab_pair, inds in map_pair_labels_to_indices.items():
#         a = lab_pair[0][ind_var]
#         b = lab_pair[1][ind_var]
#         if a==b:
#             # then is "same"
#             list_inds_same.append(inds)
#         else:
#             list_inds_diff.append(inds)
#
#     # Collect data
#     list_i = [x[0] for x in list_inds_same]
#     list_j = [x[1] for x in list_inds_same]
#     vals_same = Clsim.Xinput[(list_i, list_j)]
#
#     list_i = [x[0] for x in list_inds_diff]
#     list_j = [x[1] for x in list_inds_diff]
#     vals_diff = Clsim.Xinput[(list_i, list_j)]
#
#     return vals_same, vals_diff


def load_mult_data_helper(animal, DATE, version_distance, list_which_level=None, question=None):
    """ Load all data across (i) timw widopws (ii) which levels
    RETURNS:
        - RES, list of dicts, each a single specific dataset, across bregions and
        time windows, for a specific Snippets (e.g., which_level), with the following
        keys:
        dict_keys(['version_distance', 'which_level', 'DFRES_SAMEDIFF', 'DFRES_THEOR', 'DictBregionTwindPA', 'DictBregionTwindClraw', 'DictBregionTwindClsim', 'EFFECT_VARS', 'list_time_windows', 'SAVEDIR', 'subtract_mean_each_level_of_var'])
    """

    if list_which_level is None:
        list_which_level = ["stroke", "stroke_off"]

    SAVEDIR_LOAD = "/gorilla1/analyses/recordings/main/RSA"


    params = {
        "DATE":DATE,
        "animal":animal,
        "list_which_level":list_which_level,
        "version_distance":version_distance,
        "SAVEDIR":SAVEDIR_LOAD,
        "question":question
    }

    ####### LOAD
    RES = []
    for which_level in list_which_level:
        print("Getting: ", which_level)
        if question is None:
            # Then is not split by question
            savedir = f"{SAVEDIR_LOAD}/{animal}/{DATE}/{which_level}/{version_distance}"
            SAVEDIR_MULT = f"{SAVEDIR_LOAD}/{animal}/MULT/{DATE}/{version_distance}"
        else:
            # split
            savedir = f"{SAVEDIR_LOAD}/{animal}/SPLIT_BY_QUESTIONS/{DATE}/{which_level}/{version_distance}/{question}"
            SAVEDIR_MULT = f"{SAVEDIR_LOAD}/{animal}/SPLIT_BY_QUESTIONS/MULT/{DATE}/{version_distance}/{question}"

        path = f"{savedir}/resthis.pkl"
        print("Loading res from: ", path)
        try:
            with open(path, "rb") as f:
                res = pickle.load(f)
            RES.append(res)
        except Exception as err:
            print(path)
            print("Couldnt load this data! *******************", version_distance, animal, DATE, which_level)
            assert False

    import os
    os.makedirs(SAVEDIR_MULT, exist_ok=True)
    path = f"{SAVEDIR_MULT}/params.yaml"
    writeDictToYaml(params, path)
    # bregions
    from neuralmonkey.neuralplots.brainschematic import REGIONS_IN_ORDER

    return RES, SAVEDIR_MULT, params, REGIONS_IN_ORDER


def load_single_data(RES, bregion, twind, which_level):
    """ Helper to load...
    """

    tmp = [res for res in RES if res["which_level"]==which_level]
    if len(tmp)!=1:
        print(tmp)
        assert False
    res = tmp[0]

    # Extract specifics
    key = (bregion, twind)
    PA = res["DictBregionTwindPA"][key]
    Clraw = res["DictBregionTwindClraw"][key]
    Clsim = res["DictBregionTwindClsim"][key]

    return res, PA, Clraw, Clsim


def pipeline_rsa_scalar_population_MULT_PLOTS(DFMULT_THEOR, SAVEDIR_MULT, yvar):

    list_which_level = sorted(DFMULT_THEOR["which_level"].unique().tolist())

    ##### Preprocess
    DFMULT_THEOR["twind_str"] = ["_to_".join([str(tt) for tt in t]) for t in DFMULT_THEOR["twind"].tolist()]
    DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["which_level", "twind_str"], "wl_tw", strings_compact=True)
    EFFECT_VARS = DFMULT_THEOR["var"].unique().tolist()
    list_bregion = DFMULT_THEOR["bregion"].unique().tolist()

    ##### Plots
    savedir = f"{SAVEDIR_MULT}/overview"
    os.makedirs(savedir, exist_ok=True)

    if False: # too busy, I never look at it
        # (1) Pointplot, showing all results, but hard to read.
        fig = sns.catplot(data=DFMULT_THEOR, x="twind", y=yvar, col="bregion", hue="var", kind="point", row="which_level")
        rotateLabel(fig)
        savefig(fig, f"{savedir}/pointplot-{yvar}-bregions.pdf")
        plt.close("all")

    # IN PROGRESS - subtracting global mean within each level of (effect var). Decided it wasnt needed
    # conjucntion of twind and which_level
    if False:
        from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping
        # datamod_normalize_row_after_grouping(DFMULT_THEOR, "bregion",
        grp = ["bregion"]
        DFMULT_THEOR.groupby(grp).transform(lambda x: (x - x.mean()) / x.std())

        # Normalize by subtracting mean effect within each bregion, to allow comparison of each bregion's "signature"
        from pythonlib.tools.pandastools import aggregGeneral
        aggregGeneral(DFMULT_THEOR, ["var" , ""])

    if False:
        # IN PROGRESS - ONE VECTOR FOR EACH BREGION (across var, which_level, and time window).
        # new variable, conjunction of var and time window
        DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["var", "wl_tw"], "var_wl_tw", strings_compact=True)

        # Heatmap
        ncols = 3
        W = 4
        H = 4
        nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
        dfthis = DFMULT_THEOR
        for norm_method in [None, "row_sub", "col_sub"]:
            _, fig, _, _ = convert_to_2d_dataframe(dfthis, "bregion", "var_wl_tw", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                    norm_method=norm_method)
            savefig(fig, f"{savedir}/heatmap-")


    ########################################### HEATMAPS
    # (2) Heatmaps, easier to parse (Concatting all which_levels)
    W = 4
    H = 4
    ncols = 3
    for norm_method in [None, "all_sub", "row_sub", "col_sub"]:
        # Heatmap - one subplot for each var
        nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
        for var, ax in zip(EFFECT_VARS, axes.flatten()):
            print(var)
            dfthis = DFMULT_THEOR[DFMULT_THEOR["var"]==var]
            convert_to_2d_dataframe(dfthis, "bregion", "wl_tw", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                    norm_method=norm_method, ax=ax)
            ax.set_title(var)
        savefig(fig, f"{savedir}/heatmap-subplot_by_var-norm_{norm_method}-ccvar_{yvar}.pdf")

        # Heatmap - one subplot for each bregion
        nrows = int(np.ceil(len(list_bregion)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
        for bregion, ax in zip(list_bregion, axes.flatten()):
            print(norm_method, bregion)
            dfthis = DFMULT_THEOR[DFMULT_THEOR["bregion"]==bregion]
            convert_to_2d_dataframe(dfthis, "var", "wl_tw", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                    norm_method=norm_method, ax=ax)
            ax.set_title(bregion)
        savefig(fig, f"{savedir}/heatmap_concat-subplot_by_bregion-norm_{norm_method}-ccvar_{yvar}.pdf")

        plt.close("all")

    # (3) Heatmaps, easier to parse (Separate plots for each which_levels)
    if False: # Skip, not really needed, and takes a while
        W = 4
        H = 4
        ncols = 3
        for which_level in list_which_level:
            for norm_method in [None, "all_sub", "row_sub", "col_sub"]:
                # Heatmap - one subplot for each var
                nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
                for var, ax in zip(EFFECT_VARS, axes.flatten()):
                    print(var)
                    dfthis = DFMULT_THEOR[(DFMULT_THEOR["var"]==var) & (DFMULT_THEOR["which_level"]==which_level)]
                    convert_to_2d_dataframe(dfthis, "bregion", "twind", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                            norm_method=norm_method, ax=ax)
                    ax.set_title(var)
                savefig(fig, f"{savedir}/heatmap_whichlevel_{which_level}-subplot_by_var-norm_{norm_method}-ccvar_{yvar}.pdf")


                # Heatmap - one subplot for each bregion
                nrows = int(np.ceil(len(list_bregion)/ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
                for bregion, ax in zip(list_bregion, axes.flatten()):
                    print(norm_method, bregion)
                    dfthis = DFMULT_THEOR[(DFMULT_THEOR["bregion"]==bregion) & (DFMULT_THEOR["which_level"]==which_level)]
                    convert_to_2d_dataframe(dfthis, "var", "twind", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                            norm_method=norm_method, ax=ax)
                    ax.set_title(bregion)
                savefig(fig, f"{savedir}/heatmap_whichlevel_{which_level}-subplot_by_bregion-norm_{norm_method}-ccvar_{yvar}.pdf")

                plt.close("all")

    ################################################
    ##################### Kernels - each bregions "signature" by concatting across all times and var
    savedir = f"{SAVEDIR_MULT}/kernels"
    os.makedirs(savedir, exist_ok=True)

    def kernel_convert_from_combined_to_split(kernel_combined):
        """ Convert from list of tuple (event string, weight) to
        two lists; events and weights.
        """
        kernel = [k[0] for k in kernel_combined]
        weights = [k[1] for k in kernel_combined]

        return kernel, weights

    def kernel_extract_which_var(kernel_name):

        map_name_to_var = {
            "reach_prev_durstk":"gap_from_prev_angle_binned",
            "reach_next_durstk":"gap_to_next_angle_binned",
            "reach_prev":"gap_from_prev_angle_binned",
            "reach_next":"gap_to_next_angle_binned",
            "stkidx_stroke":"stroke_index_fromlast_tskstks",
            "shape":"shape_oriented",
            "loc_future":"CTXT_loc_next",
            "shape_future":"CTXT_shape_next",
            "angle_future":"gap_to_next_angle_binned",
            "stkidx_entire":"stroke_index_fromlast_tskstks"}

        if kernel_name not in map_name_to_var.keys():
            # The name is the var
            return kernel_name
        else:
            return map_name_to_var[kernel_name]

    def _check_twind_center_within(twind, tmin, tmax):
        """ Return True if the center of the window (twind)
        is greater than tmin and les sthan tmax"""
        cen = np.mean(twind)
        return cen>tmin and cen<tmax

    # Construct kernels for each (i.e., events in order and weights in order)
    list_twind = DFMULT_THEOR["twind"].unique().tolist()

    kernel_reach_prev = []
    kernel_reach_next = []
    kernel_reach_prev_durstk = []
    kernel_reach_next_durstk = []
    kernel_gridloc = []
    kernel_shape = []
    kernel_future_loc = []
    kernel_future_angle = []
    kernel_future_shape = []
    kernel_stkidx_during = []
    kernel_taskkind = []
    # kernel_stkidx_entire = []
    for wl in list_which_level:
        for twind in list_twind:
            ev = f"{wl}|{twind[0]}_to_{twind[1]}"

            ########### DURING GAPS (related to the gap -- i.e., motor)
            # Previous gap (it's delayed back, since that is causal to the gap)
            if wl=="stroke" and _check_twind_center_within(twind, -0.5, -0.25):
                # then window ends before align time
                k = 1
            else:
                k = 0
            kernel_reach_prev.append((ev, k))

            # Next gap (it's delayed back, since that is causal to the gap)
            if wl=="stroke_off" and _check_twind_center_within(twind, 0.1, 0.3):
                # then window ends before align time
                k = 1
            else:
                k = 0
            kernel_reach_next.append((ev, k))

            ############# CURRENT STROKE
            # vars related to current stroke, during current stroke.
            if wl=="stroke" and _check_twind_center_within(twind, -0.1, 0.3):
                # then window ends before align time
                k = 1
            else:
                k = 0
            kernel_gridloc.append((ev, k))
            kernel_shape.append((ev, k))
            kernel_stkidx_during.append((ev, k))
            kernel_taskkind.append((ev, k))
            kernel_reach_prev_durstk.append((ev, k))
            kernel_reach_next_durstk.append((ev, k))

            ############ FUTURE (during current stroke)
            # predicting future, aligned to current stroke.
            if wl=="stroke_off" and _check_twind_center_within(twind, -0.5, -0.1):
                # then window ends before align time
                k = 1
            else:
                k = 0
            kernel_future_loc.append((ev, k))
            kernel_future_shape.append((ev, k))
            kernel_future_angle.append((ev, k))

            # Stroke index (entire)
            k=1
            # kernel_stkidx_entire.append((ev, k))

    KERNELS = {
        "reach_prev":kernel_reach_prev,
        "reach_next":kernel_reach_next,
        "reach_prev_durstk":kernel_reach_prev_durstk,
        "reach_next_durstk":kernel_reach_next_durstk,
        "gridloc":kernel_gridloc,
        "stkidx_stroke":kernel_stkidx_during,
        "shape":kernel_shape,
        "task_kind":kernel_taskkind,
        "loc_future":kernel_future_loc,
        "shape_future":kernel_future_shape,
        "angle_future":kernel_future_angle,
        # "stkidx_entire":kernel_stkidx_entire,
    }

    ############### Plot kernels
    # Plot kernel templates
    fig, ax = plt.subplots()

    for i, (name, kernel) in enumerate(KERNELS.items()):
        k, w= kernel_convert_from_combined_to_split(kernel)

        var = kernel_extract_which_var(name)

        ax.scatter(k, np.ones(len(k))*i, c=[1-x for x in w], alpha=0.5, label=name)
        ax.text(0, i, var, color="r", alpha=0.5)

    ax.set_yticks(list(range(len(KERNELS.keys()))), labels=KERNELS.keys())
    ax.set_ylabel("kernel name")
    ax.set_xlabel("time window")
    rotate_x_labels(ax, 90)
    rotate_y_labels(ax, 0)
    ax.set_title("Kernels (dark dot = 1); red: var it operates on")
    savefig(fig, f"{savedir}/kernels_weights.pdf")

    # PREPROCESS - # for each variable, get 2d df (bregion x twinds)
    from neuralmonkey.analyses.event_temporal_modulation import kernel_compute, _kernel_compute_scores
    DictBregionToDf2d = {}
    for bregion in list_bregion:
        dfthis = DFMULT_THEOR[DFMULT_THEOR["bregion"]==bregion]
        dftmp, fig, ax, rgba_values = convert_to_2d_dataframe(dfthis, "var", "wl_tw", False, "mean", yvar, annotate_heatmap=False, dosort_colnames=False)
        DictBregionToDf2d[bregion] = dftmp
    DictVarToDf2d = {}
    for var in EFFECT_VARS:
        dfthis = DFMULT_THEOR[DFMULT_THEOR["var"]==var]
        dftmp, fig, ax, rgba_values = convert_to_2d_dataframe(dfthis, "bregion", "wl_tw", False, "mean", yvar, annotate_heatmap=False, dosort_colnames=False)
        DictVarToDf2d[var] = dftmp

    ##### Score data using kernesl
    res = []

    for i, (name, kernel) in enumerate(KERNELS.items()):
        k, w= kernel_convert_from_combined_to_split(kernel)
        var = kernel_extract_which_var(name)

        if var in EFFECT_VARS:
            # Ie if not, then this kernel is not defined, this dataset doesnt include this effect

            dfthis = DictVarToDf2d[var]

            # apply kernel
            scores = _kernel_compute_scores(dfthis, k, w)

            # Distribute scores across bregions
            assert list_bregion==dfthis.index.tolist()

            # Collect
            for s, br in zip(scores, list_bregion):

                res.append({
                    "bregion":br,
                    "score":s,
                    "kernel_name":name,
                    "kernel_events":k,
                    "kernel_weights":w,
                    "var":var
                })
    dfres_kernels = pd.DataFrame(res)

    if len(dfres_kernels)>0:
        # Second-order kernels, whose dimensions are the first-order kernels
        # NOTE: First-order kernels operate over time windows (for a specific var)
        # --> Therefore, second-order kernels are 2d-kernels, operating first over time, then over variables.
        # NOTE: many first order kernels are perfectly fine to keep as second-order..

        # first, get reshaped df (bregion, first order kernel name)
        # dfres_kernels_2d = convert_to_2d_dataframe(dfres_kernels, "bregion", "kernel_name", False, "mean", "score", dosort_colnames=False, list_cat_1=REGIONS_IN_ORDER)[0]
        dfres_kernels_2d = convert_to_2d_dataframe(dfres_kernels, "bregion", "kernel_name", False, "mean", "score", dosort_colnames=False)[0]

        k = ("reach_next", "reach_prev")
        w = (1,1)
        name = "reachdir"
        if all([_k in dfres_kernels_2d.columns for _k in k]):
            scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
            dfres_kernels_2d[name] = scores

        k = ("reach_next_durstk", "reach_prev_durstk")
        w = (1,1)
        name = "reachdir_durstk"
        if all([_k in dfres_kernels_2d.columns for _k in k]):
            scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
            dfres_kernels_2d[name] = scores

        k = ("gridloc", "reachdir_durstk")
        w = (1,-1)
        name = "gridloc_abstract"
        if all([_k in dfres_kernels_2d.columns for _k in k]):
            scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
            dfres_kernels_2d[name] = scores

        # k = ("gridloc", "reachdir")
        # w = (1,-1)
        # name = "gridloc_abstract"
        # if all([_k in dfres_kernels_2d.columns for _k in k]):
        #     scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
        #     dfres_kernels_2d[name] = scores

        from pythonlib.tools.snstools import heatmap
        fig, axes = plt.subplots(2,2, figsize=(10,10))
        for ax, norm_method in zip(axes.flatten(), [None, "all_sub", "col_sub", "row_sub"]):
            heatmap(dfres_kernels_2d, ax, False, (None, None), norm_method=norm_method)
            # convert_to_2d_dataframe(dfres_kernels, "bregion", "kernel_name", True, "mean", "score", annotate_heatmap=False, norm_method="col_sub", dosort_colnames=False, list_cat_1=REGIONS_IN_ORDER)
            ax.set_title(f"norm_{norm_method}")
        savefig(fig, f"{savedir}/heatmap-kernel_scores-ccvar_{yvar}.pdf")

        plt.close("all")
    else:
        dfres_kernels_2d = None

    PARAMS = {
        "EFFECT_VARS":EFFECT_VARS,
        "list_bregion":list_bregion,
        "KERNELS":KERNELS,
    }

    return DFMULT_THEOR, DictBregionToDf2d, DictVarToDf2d, dfres_kernels_2d, PARAMS


def pipeline_rsa_scalar_population_MULT_PLOT_DETAILED(animal, DATE, version_distance,
                                                      list_which_level, question=None):
    """ Plot distance scores for each level of each var. Plots "same" i.e, dist between the
    same level across different levels of othervar, and diff...
    """
    from pythonlib.tools.pandastools import grouping_count_n_samples
    from pythonlib.tools.listtools import stringify_list
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    # Is similarity dominated by a specific level of var?
    # -- e.g, "stroke_index_semantic" is high... is this just becuase "first_stroke" is similar?
    RES, SAVEDIR_MULT, params, list_bregion = load_mult_data_helper(animal, DATE, version_distance,
                                                                            list_which_level=list_which_level,
                                                                            question=question)
    list_twind = RES[0]["list_time_windows"]
    savedir = f"{SAVEDIR_MULT}/details_each_level"
    os.makedirs(savedir, exist_ok=True)

    ###### COLLECT DATA
    resthis = []
    for which_level in list_which_level:
        for bregion in list_bregion:
            for twind in list_twind:
                res, PA, Clraw, Clsim = load_single_data(RES, bregion, twind, which_level)
                EFFECT_VARS = res["EFFECT_VARS"]

                for var in EFFECT_VARS:
                    var_levs = Clsim.rsa_labels_extract_var_levels()[var]

                    for lev in var_levs:

                        ######## COUNT N SAMPLES
                        dfthis = PA.Xlabels["trials"][PA.Xlabels["trials"][var]==lev] # trial, not agged.

                        # Count n trials for this level
                        n = len(dfthis)

                        # Count n trials across each conjunction of othervars
                        n_each_other_var = tuple(grouping_count_n_samples(dfthis, [v for v in EFFECT_VARS if not v==var]))
                        # groupdict = grouping_append_and_return_inner_items(dfthis, [v for v in EFFECT_VARS if not v==var])
                        # # - collect distribution of n across othervars
                        # n_each_other_var = []
                        # for lev_other, inds in groupdict.items():
                        #     n_each_other_var.append(len(inds))
                        #

                        if n>1:
                            # get indices that are same, and diff
                            ma_same, ma_diff = Clsim.rsa_matindex_same_diff_this_level(var, lev)

                            # upper triangular
                            ma_ut = Clsim._rsa_matindex_generate_upper_triangular()

                            assert sum(sum(ma_same & ma_ut))>0
                            assert sum(sum(ma_diff & ma_ut))>0

                            # get values
                            d_same = Clsim.Xinput[ma_same & ma_ut].mean()
                            d_diff = Clsim.Xinput[ma_diff & ma_ut].mean()

                            assert not np.isnan(d_same)
                            assert not np.isnan(d_diff)
                        else:
                            d_same = np.nan
                            d_diff = np.nan

                        ### SAVE
                        resthis.append({
                            "which_level": which_level,
                            "bregion": bregion,
                            "twind":twind,
                            "var":var,
                            "lev":lev,
                            "lev_str":stringify_list(lev, return_as_str=True),
                            "n":n,
                            "n_each_other_var":n_each_other_var,
                            "dist_same":d_same,
                            "dist_diff":d_diff
                        })
    # Save
    dfres = pd.DataFrame(resthis)

    # how many had only n of 1?
    if False:
        dfres["n"].hist(bins=20)
        dfres["n"]==1

    #### Distribution of counts for each level of each var (nrows, i.e,, trials)

    # Pull out just a single bregion for counts analysis, they are identical
    dfresthis = dfres[dfres["bregion"]==list_bregion[0]]

    ncols = 3
    nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    list_lev = dfresthis["lev"].unique().tolist()
    twind = dfresthis["twind"].values[0]
    which_level = dfresthis["which_level"].values[0]
    for ax, var in zip(axes.flatten(), EFFECT_VARS):
        ax.set_title(var)
        ax.set_xlabel("n counts, across all conjunctions of other var")
        for i, lev in enumerate(list_lev):
            dftmp = dfresthis[(dfresthis["which_level"]==which_level) & (dfresthis["twind"]==twind) & (dfresthis["var"]==var) & (dfresthis["lev"]==lev)]
            if len(dftmp)>0:
                assert len(dftmp)==1
                n_each_other_var = dftmp.iloc[0]["n_each_other_var"]
                ax.plot(n_each_other_var, np.ones(len(n_each_other_var))*i, "o", label=lev, alpha=0.5)
                ax.text(min(n_each_other_var), i, f"n={n_each_other_var}", fontsize=5)
            else:
                # This is expected, since im looping through levs which are colelcted acorss all
                # var, not just this var
                pass

        ax.legend()
    path = f"{savedir}/n_rows_per_level.pdf"
    savefig(fig, path)
    plt.close("all")

    #### Plot distances between datapts that are "same" and "diff" for each level
    for twind in list_twind:
        for which_level in list_which_level:
            dfresthis = dfres[(dfres["which_level"]==which_level) & (dfres["twind"]==twind)].dropna().reset_index(drop=True)
            if len(dfresthis)==0:
                print(which_level)
                print(twind)
                print(dfres["which_level"].unique())
                print(dfres["twind"].unique())
                assert False

            # Between pairs of datapts with same values for each level
            for val in ["dist_same", "dist_diff"]:
                fig, axes = plot_subplots_heatmap(dfresthis, "bregion", "lev_str", val,
                                                  "var", annotate_heatmap=False, share_zlim=True)
                savefig(fig, f"{savedir}/{val}-wl_{which_level}-twind_{twind}.pdf")
            plt.close("all")

def pipeline_rsa_scalar_population_MULT(animal, DATE, version_distance, yvar, list_which_level,
                                        question=None):
    """ Load data saved in pipeline_rsa_scalar_population acrooss
    different (time windows, bregions, effects, which_levels[from SP extraction],
    version_distance, ...) and make single plots combining all of them
    """

    ########## LOAD ALL DATA
    RES, SAVEDIR_MULT, params, REGIONS_IN_ORDER = load_mult_data_helper(animal, DATE, version_distance,
                                                                        list_which_level=list_which_level,
                                                                        question=question)

    ## COLLECT
    # Compare to theoetical simmat
    list_df = []
    # list_which_level = []
    for res in RES:
        which_level = res["which_level"]
        version_distance = res["version_distance"]
        df = res["DFRES_THEOR"]
        df["which_level"] = which_level
        df["version_distance"] = version_distance
        list_df.append(df)
    #     list_which_level.append(which_level)
    # list_which_level = sorted(set(list_which_level))
    DFMULT_THEOR = pd.concat(list_df).reset_index(drop=True)

    return pipeline_rsa_scalar_population_MULT_PLOTS(DFMULT_THEOR, SAVEDIR_MULT, yvar)


def pipeline_rsa_scalar_population(SP, EFFECT_VARS, list_time_windows,
                                   SAVEDIR, version_distance = "pearson",
                                   subtract_mean_each_level_of_var = "IGNORE",
                                   PLOT_INDIV=True, SKIP_ANALY_PLOTTING=True):

    from pythonlib.tools.pandastools import summarize_featurediff
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # Features that should always extract (Strokes dat)
    list_features_extraction = ["trialcode", "aborted", "trial_neural", "event_time", "task_kind",
                                "stroke_index", "stroke_index_fromlast", "stroke_index_fromlast_tskstks",
                                "stroke_index_semantic", "stroke_index_semantic_tskstks",
                                "shape_oriented", "gridloc", "gridsize",
                                "FEAT_num_strokes_task", "FEAT_num_strokes_beh",
                                "CTXT_loc_next", "CTXT_shape_next",
                                "CTXT_loc_prev", "CTXT_shape_prev",
                                "gap_from_prev_angle_binned", "gap_to_next_angle_binned",
                                ]

    ######################## PREPPING
    # get back all the outliers, since they just a single removed outlier (chan x trial) will throw out the entire trial.
    SP.datamod_append_outliers()
    SP.datamod_append_unique_indexdatapt()

    # Append variables by hand
    D = SP.datasetbeh_extract_dataset()
    D.extract_beh_features()
    SP.datasetbeh_append_column("FEAT_num_strokes_task", D)
    SP.datasetbeh_append_column("aborted", D)

    # For the rest, try to get automatically.
    vars_to_extract = EFFECT_VARS + list_features_extraction
    assert SP.datasetbeh_append_column_helper(vars_to_extract, D)==True # Extract all the vars here

    # Conjunction of stroke index and num strokes in task.
    SP.DfScalar = append_col_with_grp_index(SP.DfScalar, ["FEAT_num_strokes_task", "stroke_index"], "nstk_stkidx", False)

    savedir = f"{SAVEDIR}/{SP.SN.Animal}/{SP.SN.Date}/{SP.Params['which_level']}/{version_distance}"
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    ####################### EXTRACT DATA
    # 1) Extract population data
    assert len(SP.Params["list_events_uniqnames"])==1, "assuming is strokes, just a single event... otherwise iterate"
    event = SP.Params["list_events_uniqnames"][0]
    list_features_extraction = list(set(list_features_extraction + EFFECT_VARS))
    chans_needed = SP.SN.sitegetter_all()
    PA, _ = SP.dataextract_as_popanal_statespace(chans_needed, event,
                                                 list_features_extraction=list_features_extraction,
                                              which_fr_sm = "fr_sm", max_frac_trials_lose=0.02)

    # Split PA based on chans (e.g., bregions), times (e.g., different time slices) BEFORE doing downstream analyses
    list_bregion = SP.SN.sitegetter_get_brainregion_list()
    DictBregionTwindPA = {}
    for bregion in list_bregion:
        for twind in list_time_windows:
            # Bregion
            chans_needed = SP.SN.sitegetter_all([bregion])
            pa = PA.slice_by_dim_values_wrapper("chans", chans_needed)
            # Times
            pa = pa.slice_by_dim_values_wrapper("times", twind)

            DictBregionTwindPA[(bregion, twind)] = pa
            print(bregion, " -- ", twind, " -- (data shape:)", pa.X.shape)

    if SKIP_ANALY_PLOTTING:
        return None, None, DictBregionTwindPA, \
            None, None, savedir
    else:
        return _pipeline_score_all_pa(DictBregionTwindPA, EFFECT_VARS, savedir,
                                      version_distance, subtract_mean_each_level_of_var,
                                    PLOT_INDIV)

    # #################### COMPUTE DISTANCE MATRICES AND SCORE RELATIVE TO THEORETICAL MATRICES.
    # PLOT = PLOT_INDIV
    # list_dfres = []
    # list_dfres_theor = []
    # DictBregionTwindClraw = {}
    # DictBregionTwindClsim = {}
    # ct = 0
    # for (bregion, twind), pa in DictBregionTwindPA.items():
    #     print("Scoring, for: ", bregion, twind)
    #
    #     sdir = f"{savedir}/preprocess/{bregion}-{twind}"
    #     os.makedirs(sdir, exist_ok=True)
    #     print("Saving to: ", sdir)
    #
    #     PLOT_THEORETICAL_SIMMATS = ct==0 and PLOT==True # Only do once. this same across bregions and twinds.
    #     dfres_same_diff, dfres_theor, Clraw, Clsim, PAagg = _preprocess_rsa_scalar_population(pa, EFFECT_VARS, version_distance,
    #                                                                                           PLOT=PLOT,
    #                                                                                           sdir=sdir,
    #                                                                                           subtract_mean_each_level_of_var=subtract_mean_each_level_of_var,
    #                                                                                            PLOT_THEORETICAL_SIMMATS=PLOT_THEORETICAL_SIMMATS)
    #
    #     # Collect results
    #     dfres_same_diff["bregion"] = bregion
    #     dfres_same_diff["twind"] = [twind for _ in range(len(dfres_same_diff))]
    #     dfres_theor["bregion"] = bregion
    #     dfres_theor["twind"] = [twind for _ in range(len(dfres_theor))]
    #
    #     list_dfres.append(dfres_same_diff)
    #     list_dfres_theor.append(dfres_theor)
    #
    #     DictBregionTwindClraw[(bregion, twind)] = Clraw
    #     DictBregionTwindClsim[(bregion, twind)] = Clsim
    #
    #     plt.close("all")
    #
    # DFRES_SAMEDIFF = pd.concat(list_dfres).reset_index(drop=True)
    # DFRES_THEOR = pd.concat(list_dfres_theor).reset_index(drop=True)
    #
    # ###############################
    # # SUMMARY PLOTS
    # sdir = f"{savedir}/summary"
    # os.makedirs(sdir, exist_ok=True)
    #
    # ########################### 1. Same (level within var) vs. diff.
    # if version_distance=="pearson":
    #     GROUPING_LEVELS = ["diff", "same"]
    # elif version_distance=="euclidian":
    #     GROUPING_LEVELS = ["same", "diff"]
    # else:
    #     assert False
    # dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = summarize_featurediff(DFRES_SAMEDIFF,
    #                                                                                               "same_or_diff", GROUPING_LEVELS, ["mean"], ["ind_var", "ind_var_str", "bregion", "sort_order", "grouping_vars", "subtract_mean_each_level_of_var"])
    # dfsummaryflat = SP.SN.datamod_sitegetter_reorder_by_bregion(dfsummaryflat)
    #
    # # Summarize all
    # var_subplot = ["grouping_vars", "subtract_mean_each_level_of_var"]
    # _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "shape_oriented", "gridloc", var_subplot, "value", "bregion");
    # savefig(fig, f"{sdir}/plot45_same_vs_diff-1.pdf")
    # if "stroke_index" in EFFECT_VARS:
    #     _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "shape_oriented", "stroke_index", var_subplot, "value", "bregion");
    #     savefig(fig, f"{sdir}/plot45_same_vs_diff-2.pdf")
    #     _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "gridloc", "stroke_index", var_subplot, "value", "bregion");
    #     savefig(fig, f"{sdir}/plot45_same_vs_diff-3.pdf")
    # plt.close("all")
    #
    # if False:
    #     # Not working, problem is dfthis comes up empty. easy fix.
    #     # Is redundant given the following "theoretical comparison" plots.
    #     # Plot for each grouping_vars
    #
    #     grouping_vars = ("shape_oriented", "gridloc", "stroke_index")
    #     # grouping_vars = ["shape_oriented", "gridloc"]
    #     a = DFRES_SAMEDIFF["grouping_vars"].isin([grouping_vars])
    #     b = DFRES_SAMEDIFF["subtract_mean_each_level_of_var"].isin(["IGNORE"])
    #     dfthis = DFRES_SAMEDIFF[(a) & (b)]
    #
    #     a = dfsummaryflat["grouping_vars"].isin([grouping_vars])
    #     b = dfsummaryflat["subtract_mean_each_level_of_var"].isin(["IGNORE"])
    #
    #     dfthis_diff = dfsummaryflat[(a) & (b)]
    #     dfthis_diff
    #     from pythonlib.tools.pandastools import plot_pointplot_errorbars
    #     from pythonlib.tools.plottools import rotate_x_labels
    #
    #     # pull out the non-hierarchical dataframes
    #     fig, axes = plt.subplots(2,2, figsize=(10,10), sharey=True)
    #     list_ind_var = sorted(dfthis["ind_var"].unique().tolist())
    #     list_same_diff = ["same", "diff"]
    #     ct = 0
    #     for ind_var, ax in zip(list_ind_var, axes.flatten()):
    #         ind_var_str = grouping_vars[ind_var]
    #         ax.set_title(ind_var_str)
    #         ct+=1
    #         for same_diff in list_same_diff:
    #             dfthisthis = dfthis[dfthis["ind_var"]==ind_var]
    #             plot_pointplot_errorbars(dfthisthis, "bregion", "mean", ax=ax, yvar_err=f"sem", hue="same_or_diff")
    #             # assert False
    #             #
    #             # x=dfthisthis["bregion"]
    #             # y=dfthisthis[f"{same_diff}_mean"]
    #             # yerr = dfthisthis[f"{same_diff}_sem"]
    #             # lab = f"{same_diff}-{ind_var_str}"
    #             # ax.errorbar(x=x, y=y, yerr=yerr, label=lab)
    #             # # ax.bar(x, y, yerr=yerr, label=lab, alpha=0.4)
    #             # # sns.barplot(data=dfthisthis, x="bregion", y="same_mean", yerr=dfthisthis["same_sem"])
    #
    #         ax.axhline(0, color="k", alpha=0.25)
    #         rotate_x_labels(ax, 75)
    #         ax.legend()
    #
    #     ax = axes.flatten()[ct]
    #     list_ind_var = sorted(dfthis["ind_var"].unique().tolist())
    #     list_same_diff = ["same", "diff"]
    #     for ind_var in list_ind_var:
    #         ind_var_str = grouping_vars[ind_var]
    #         dfthisthis = dfthis_diff[dfthis_diff["ind_var"]==ind_var]
    #
    #         x=dfthisthis["bregion"]
    #         y=dfthisthis["value"]
    #         lab = f"{ind_var_str}"
    #         ax.plot(x, y, label=lab)
    #
    #     ax.axhline(0, color="k", alpha=0.25)
    #     rotate_x_labels(ax, 75)
    #     ax.legend()
    #
    # #################### 2. Comparing data simmat to theoretical simmats.
    # for yvar in ["cc", "mr_coeff"]:
    #     fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", kind="point", aspect=1.5,
    #                     row="twind")
    #     rotateLabel(fig)
    #     savefig(fig, f"{sdir}/vs_theor_simmat-pointplot-{yvar}.pdf")
    #
    #     fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", alpha=0.5, aspect=1.5, row="twind")
    #     rotateLabel(fig)
    #     savefig(fig, f"{sdir}/vs_theor_simmat-scatterplot-{yvar}.pdf")
    #
    #     # Summarize results in a heatmap (region x effect)
    #     for norm in ["col_sub", "row_sub", None]:
    #         for twind in list_time_windows:
    #             dfthis = DFRES_THEOR[DFRES_THEOR["twind"] == twind]
    #             _, fig, _, _ = convert_to_2d_dataframe(dfthis, "bregion", "var",
    #                                                    True, "mean",
    #                                                    yvar, annotate_heatmap=False, dosort_colnames=False,
    #                                                    norm_method=norm)
    #             savefig(fig, f"{sdir}/vs_theor_simmat-heatmap-{yvar}-norm_{norm}-twind_{twind}.pdf")
    #
    #             plt.close("all")
    #
    # return DFRES_SAMEDIFF, DFRES_THEOR, DictBregionTwindPA, \
    #     DictBregionTwindClraw, DictBregionTwindClsim, savedir



def _pipeline_score_all_pa(DictBregionTwindPA, EFFECT_VARS, savedir,
                           version_distance = "pearson",
                           subtract_mean_each_level_of_var = "IGNORE",
                           PLOT_INDIV=True):

    from pythonlib.tools.pandastools import summarize_featurediff
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # Extract all
    list_time_windows = sorted(set([k[1] for k in DictBregionTwindPA.keys()]))

    #################### COMPUTE DISTANCE MATRICES AND SCORE RELATIVE TO THEORETICAL MATRICES.
    list_dfres = []
    list_dfres_theor = []
    DictBregionTwindClraw = {}
    DictBregionTwindClsim = {}
    ct = 0
    for (bregion, twind), pa in DictBregionTwindPA.items():
        print("Scoring, for: ", bregion, twind)

        sdir = f"{savedir}/preprocess/{bregion}-{twind}"
        os.makedirs(sdir, exist_ok=True)
        # print("Saving to: ", sdir)

        PLOT_THEORETICAL_SIMMATS = ct==0 and PLOT_INDIV==True # Only do once. this same across bregions and twinds.
        dfres_same_diff, dfres_theor, Clraw, Clsim, PAagg = _preprocess_rsa_scalar_population(pa, EFFECT_VARS, version_distance,
                                                                                              PLOT=PLOT_INDIV,
                                                                                              sdir=sdir,
                                                                                              subtract_mean_each_level_of_var=subtract_mean_each_level_of_var,
                                                                                               PLOT_THEORETICAL_SIMMATS=PLOT_THEORETICAL_SIMMATS)

        if PAagg is None:
            # Then lost all data, due to pruning
            # list_dfres.append(None)
            # list_dfres_theor.append(None)
            DictBregionTwindClraw[(bregion, twind)] = None
            DictBregionTwindClsim[(bregion, twind)] = None
        else:
            # Collect results
            if dfres_same_diff is not None:
                dfres_same_diff["bregion"] = bregion
                dfres_same_diff["twind"] = [twind for _ in range(len(dfres_same_diff))]
            dfres_theor["bregion"] = bregion
            dfres_theor["twind"] = [twind for _ in range(len(dfres_theor))]

            list_dfres.append(dfres_same_diff)
            list_dfres_theor.append(dfres_theor)
            DictBregionTwindClraw[(bregion, twind)] = Clraw
            DictBregionTwindClsim[(bregion, twind)] = Clsim

        plt.close("all")

    ###############################
    # SUMMARY PLOTS
    sdir = f"{savedir}/summary"
    os.makedirs(sdir, exist_ok=True)

    if len(list_dfres_theor)==0:
        DFRES_THEOR = None
        DFRES_SAMEDIFF = None
    else:
        DFRES_THEOR = pd.concat(list_dfres_theor).reset_index(drop=True)

        if False:
            # Just stop doing this here.
            DFRES_SAMEDIFF = pd.concat(list_dfres).reset_index(drop=True)

            ########################### 1. Same (level within var) vs. diff.
            if version_distance=="pearson":
                GROUPING_LEVELS = ["diff", "same"]
            elif version_distance=="euclidian":
                GROUPING_LEVELS = ["same", "diff"]
            else:
                assert False
            dfsummary, dfsummaryflat, COLNAMES_NOABS, COLNAMES_ABS, COLNAMES_DIFF = summarize_featurediff(DFRES_SAMEDIFF,
                                                                                                          "same_or_diff", GROUPING_LEVELS, ["mean"], ["ind_var", "ind_var_str", "bregion", "sort_order", "grouping_vars", "subtract_mean_each_level_of_var"])
            if False:
                dfsummaryflat = SP.SN.datamod_sitegetter_reorder_by_bregion(dfsummaryflat)

            # Summarize all
            var_subplot = ["grouping_vars", "subtract_mean_each_level_of_var"]
            _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "shape_oriented", "gridloc", var_subplot, "value", "bregion");
            savefig(fig, f"{sdir}/plot45_same_vs_diff-1.pdf")
            if "stroke_index" in EFFECT_VARS:
                _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "shape_oriented", "stroke_index", var_subplot, "value", "bregion");
                savefig(fig, f"{sdir}/plot45_same_vs_diff-2.pdf")
                _, fig = plot_45scatter_means_flexible_grouping(dfsummaryflat, "ind_var_str", "gridloc", "stroke_index", var_subplot, "value", "bregion");
                savefig(fig, f"{sdir}/plot45_same_vs_diff-3.pdf")
            plt.close("all")

            if False:
                # Not working, problem is dfthis comes up empty. easy fix.
                # Is redundant given the following "theoretical comparison" plots.
                # Plot for each grouping_vars

                grouping_vars = ("shape_oriented", "gridloc", "stroke_index")
                # grouping_vars = ["shape_oriented", "gridloc"]
                a = DFRES_SAMEDIFF["grouping_vars"].isin([grouping_vars])
                b = DFRES_SAMEDIFF["subtract_mean_each_level_of_var"].isin(["IGNORE"])
                dfthis = DFRES_SAMEDIFF[(a) & (b)]

                a = dfsummaryflat["grouping_vars"].isin([grouping_vars])
                b = dfsummaryflat["subtract_mean_each_level_of_var"].isin(["IGNORE"])

                dfthis_diff = dfsummaryflat[(a) & (b)]
                dfthis_diff
                from pythonlib.tools.pandastools import plot_pointplot_errorbars

                # pull out the non-hierarchical dataframes
                fig, axes = plt.subplots(2,2, figsize=(10,10), sharey=True)
                list_ind_var = sorted(dfthis["ind_var"].unique().tolist())
                list_same_diff = ["same", "diff"]
                ct = 0
                for ind_var, ax in zip(list_ind_var, axes.flatten()):
                    ind_var_str = grouping_vars[ind_var]
                    ax.set_title(ind_var_str)
                    ct+=1
                    for same_diff in list_same_diff:
                        dfthisthis = dfthis[dfthis["ind_var"]==ind_var]
                        plot_pointplot_errorbars(dfthisthis, "bregion", "mean", ax=ax, yvar_err=f"sem", hue="same_or_diff")
                        # assert False
                        #
                        # x=dfthisthis["bregion"]
                        # y=dfthisthis[f"{same_diff}_mean"]
                        # yerr = dfthisthis[f"{same_diff}_sem"]
                        # lab = f"{same_diff}-{ind_var_str}"
                        # ax.errorbar(x=x, y=y, yerr=yerr, label=lab)
                        # # ax.bar(x, y, yerr=yerr, label=lab, alpha=0.4)
                        # # sns.barplot(data=dfthisthis, x="bregion", y="same_mean", yerr=dfthisthis["same_sem"])

                    ax.axhline(0, color="k", alpha=0.25)
                    rotate_x_labels(ax, 75)
                    ax.legend()

                ax = axes.flatten()[ct]
                list_ind_var = sorted(dfthis["ind_var"].unique().tolist())
                list_same_diff = ["same", "diff"]
                for ind_var in list_ind_var:
                    ind_var_str = grouping_vars[ind_var]
                    dfthisthis = dfthis_diff[dfthis_diff["ind_var"]==ind_var]

                    x=dfthisthis["bregion"]
                    y=dfthisthis["value"]
                    lab = f"{ind_var_str}"
                    ax.plot(x, y, label=lab)

                ax.axhline(0, color="k", alpha=0.25)
                rotate_x_labels(ax, 75)
                ax.legend()
        else:
            DFRES_SAMEDIFF = None


        #################### 2. Comparing data simmat to theoretical simmats.
        for yvar in ["cc", "mr_coeff"]:
            fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", kind="point", aspect=1.5,
                            row="twind")
            rotateLabel(fig)
            savefig(fig, f"{sdir}/vs_theor_simmat-pointplot-{yvar}.pdf")

            fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", alpha=0.5, aspect=1.5, row="twind")
            rotateLabel(fig)
            savefig(fig, f"{sdir}/vs_theor_simmat-scatterplot-{yvar}.pdf")

            # Summarize results in a heatmap (region x effect)
            for norm in ["col_sub", "row_sub", None]:
                for twind in list_time_windows:
                    dfthis = DFRES_THEOR[DFRES_THEOR["twind"] == twind]
                    _, fig, _, _ = convert_to_2d_dataframe(dfthis, "bregion", "var",
                                                           True, "mean",
                                                           yvar, annotate_heatmap=False, dosort_colnames=False,
                                                           norm_method=norm)
                    savefig(fig, f"{sdir}/vs_theor_simmat-heatmap-{yvar}-norm_{norm}-twind_{twind}.pdf")

                    plt.close("all")

        # Save results
        # resthis = {
        #     "version_distance":version_distance,
        #     "which_level":which_level,
        #     "DFRES_SAMEDIFF":DFRES_SAMEDIFF,
        #     "DFRES_THEOR":DFRES_THEOR,
        #     "DictBregionTwindPA":DictBregionTwindPA,
        #     "DictBregionTwindClraw":DictBregionTwindClraw,
        #     "DictBregionTwindClsim":DictBregionTwindClsim,
        #     "EFFECT_VARS":EFFECT_VARS,
        #     "list_time_windows":list_time_windows,
        #     "SAVEDIR":SAVEDIR,
        #     "subtract_mean_each_level_of_var":subtract_mean_each_level_of_var,
        # }
        #
        # import pickle
        # path = f"{savedir}/resthis.pkl"
        # with open(path, "wb") as f:
        #     pickle.dump(resthis, f)
        #     print("saved to: ", path)


    return DFRES_SAMEDIFF, DFRES_THEOR, DictBregionTwindPA, \
        DictBregionTwindClraw, DictBregionTwindClsim, savedir

def preprocess_rsa_prepare_popanal_wrapper(PA, effect_vars, exclude_last_stroke, exclude_first_stroke, min_taskstrokes,
                                           max_taskstrokes, keep_only_first_stroke=False):
    """ This (trial-level) PA, prune it to be ready for input into RSA anaysi,
    including pruning lelvels with few trials, and ekeeping only speicifc tasksets
    """

    if keep_only_first_stroke:
        assert exclude_first_stroke==False
        assert exclude_last_stroke==False

    if keep_only_first_stroke:
        vals = [0]
        PA = PA.slice_by_labels("trials", "stroke_index", vals) # list(range(-10, -1)) --> [-10, ... , -2]
        # assert len(PA.Xlabels["trials"])>0
    if exclude_last_stroke:
        vals = PA.Xlabels["trials"]["stroke_index_fromlast"].unique().tolist()
        vals = [v for v in vals if v <-1]
        PA = PA.slice_by_labels("trials", "stroke_index_fromlast", vals) # list(range(-10, -1)) --> [-10, ... , -2]
        # assert len(PA.Xlabels["trials"])>0
    if exclude_first_stroke:
        vals = PA.Xlabels["trials"]["stroke_index"].unique().tolist()
        vals = [v for v in vals if v >0]
        PA = PA.slice_by_labels("trials", "stroke_index", vals) # list(range(-10, -1)) --> [-10, ... , -2]
        # assert len(PA.Xlabels["trials"])>0
    PA = PA.slice_by_labels("trials", "FEAT_num_strokes_task", list(range(min_taskstrokes, max_taskstrokes+1))) # list(range(-10, -1)) --> [-10, ... , -2]

    assert len(PA.Xlabels["trials"])>0

    res_check_effectvars_before = _preprocess_pa_check_how_much_data(PA, effect_vars)
    PA, res_check_before, res_check_after, vars_remove, reason_vars_remove = preprocess_prune_pa_enough_data(PA, effect_vars)
    if PA is None:
        # show what went in
        for k, v in res_check_effectvars_before.items():
            print(" ----- ", k)
            print(v)
        assert False, "why threw out all the data?"

    # summarize variables used above for pruning taskset
    res_check_tasksets = _preprocess_pa_check_how_much_data(PA, effect_vars)
    res_check_effectvars = res_check_after

    return PA, res_check_tasksets, res_check_effectvars

def preprocess_prune_pa_enough_data(PA, EFFECT_VARS):
    """ REpeatedly run this until there is no more change inoutput. This isbeaucase some
    steps can lead to change in other steps (e.g, remove level, then you don have enough
    levels, etc"""

    # res_check_before=pd.DataFrame([0])
    # res_check_after=pd.DataFrame([1])
    ct = 1
    did_change = True
    while did_change:
        # Then previuos run made change. try again
        PA, res_check_before, res_check_after, vars_remove, reason_vars_remove, did_change \
            = _preprocess_prune_pa_enough_data(PA, EFFECT_VARS)
        if PA is None:
            # Then no data at all
            break
        # if did_change:
        #     print(PA)
        #     print(reason_vars_remove)
        ct+=1

        assert ct<20, "bug, recursion??"
    return PA, res_check_before, res_check_after, vars_remove, reason_vars_remove




def _preprocess_prune_pa_enough_data(PA, EFFECT_VARS,
                                     n_min_lev_per_var = 2,
                                     n_min_rows_per_lev = 5,
                                     n_min_rows_per_conjunction_of_var_othervar = 2,
                                     DEBUG=False):
    """
    Prunes PA. For each var in EFFECT_VARS, checks each level, and removes all trials of that level
    if it fails any of the cheks (see PARAMS for the checks).
    PARAMS:
    - n_min_lev_per_var, int, each var must have at least this many levels. Prunes ENTIRE
    PA if any var fails (i.e, returns None)
    - n_min_rows_per_lev, int, each level (of each var) must have at least this many rows
    (where whether rows are trials or mean_over_trials depends on what PA passed in)
    - n_min_rows_per_conjunction_of_var_othervar, int, for each level, check how many
    rows it has for each level of (all other vars conjunction), and must be more than this.
    RETURNS:
        - PA, pruned. None if entire PA is empty
        - res_check_before, df
        - res_check_after, df
    """

    # assert False, "doesnt make sense to run this here becuase whether have enough data depends on var. if try all var, then will throw out data if one var isnt' good..."
    from pythonlib.tools.pandastools import grouping_count_n_samples

    res_check_before = _preprocess_pa_check_how_much_data(PA, EFFECT_VARS)
    did_change = False
    vars_remove = []
    vars_keep = []
    reason_vars_remove = {}
    for var in EFFECT_VARS:
        dfres_check = res_check_before[var]
        var_levs = dfres_check["lev"].unique().tolist()

        # N levels too low?
        tmp = dfres_check["n_levs_this_var"].unique().tolist()
        assert len(tmp)==1
        n_levs_this_var = tmp[0]
        if n_levs_this_var<n_min_lev_per_var:
            vars_remove.append(var)
            reason_vars_remove[var] = "not_enough_levels"
            continue

        #################### CHECK EACH LEVEL WITHIN THIS VAR
        # For each level,
        levs_remove = []
        levs_keep = []
        for lev in var_levs:

            dfres_check_thislev = dfres_check[dfres_check["lev"]==lev]
            assert len(dfres_check_thislev)==1

            # N rows for this level
            if dfres_check_thislev["n_rows_this_lev"].values[0]<n_min_rows_per_lev:
                levs_remove.append(lev)
                continue

            # N othervars that this level spans
            n_each_other_var = dfres_check_thislev["n_rows_for_each_othervar_lev"].values[0]
            n_other_var_levs_with_enough_rows = sum([n >= n_min_rows_per_conjunction_of_var_othervar for n in n_each_other_var])
            if n_other_var_levs_with_enough_rows<2:
                levs_remove.append(lev)
                continue

            # got this far. keep it
            levs_keep.append(lev)
        if DEBUG:
            print(f"Keeping/removing these levs for {var}:", levs_keep, levs_remove)
        assert sort_mixed_type(levs_keep+levs_remove) == sort_mixed_type(var_levs)

        #################### MODIFY PA given results of check of levels.
        # Prune PA (levels, this var)
        if DEBUG:
            print(var, var_levs, levs_keep, levs_remove)
        if len(levs_keep)==0:
            # Then remove this entire dataset...
            vars_remove.append(var)
            reason_vars_remove[var] = "all_levels_fail_somehow"
            continue
        elif len(levs_keep)<len(var_levs):
            # Then just remove the levels...
            PA = PA.slice_by_labels("trials", var, levs_keep)
            did_change = True
        else:
            # keeping all var levs...
            pass

        # got this far, keep it
        vars_keep.append(var)

    assert sort_mixed_type(vars_keep+vars_remove) == sort_mixed_type(EFFECT_VARS)

    # If any var is to be removed, then this means you would remove all rows. This means
    # you can't study these effects with this PA.
    if len(vars_remove)>0:
        did_change = True
        return None, res_check_before, None, vars_remove, reason_vars_remove, did_change
    else:
        # Assess how much data is available.
        res_check_after = _preprocess_pa_check_how_much_data(PA, EFFECT_VARS)

        # how many levs must be thrown out for each var?
        return PA, res_check_before, res_check_after, vars_remove, reason_vars_remove, did_change

def _preprocess_pa_check_how_much_data(PA, EFFECT_VARS):
    """ Ideally pass in the PA _before_ normalization, so that the
    rows here are trials.
    """
    from pythonlib.tools.pandastools import grouping_count_n_samples

    for var in EFFECT_VARS:
        if var not in PA.Xlabels["trials"].columns:
            print(var)
            print(PA.Xlabels["trials"].columns)
            assert False

    # - n levels for each var
    res = {}
    for var in EFFECT_VARS:
        var_levs = PA.Xlabels["trials"][var].unique().tolist()

        # for each level, how many levels of other var
        resthis = []
        for lev in var_levs:
            ######## COUNT N SAMPLES
            dfthis = PA.Xlabels["trials"][PA.Xlabels["trials"][var]==lev]

            assert len(dfthis)>0

            # Count n trials for this level
            # True if greater than <n_min_cases_per_lev> in this level.
            # n = len(dfthis) # num rows that has this level

            # Count n trials across each conjunction of othervars
            n_each_other_var = tuple(grouping_count_n_samples(dfthis, [v for v in EFFECT_VARS if not v==var]))

            # min, median, and max
            # resthis[lev] = (
            #     len(dfthis), # num rows that has this level
            #     len(n_each_other_var), # num othervar conjunctive levels spanned
            #     min(n_each_other_var), # min n rows across other-levels
            #     int(np.median(n_each_other_var)),
            #     max(n_each_other_var)
            # )

            resthis.append({
                "var":var,
                "n_levs_this_var":len(var_levs),
                "lev":lev,
                "n_rows_this_lev":len(dfthis), # num rows that has this level
                "n_othervar_levs_spanned":len(n_each_other_var), # num othervar conjunctive levels spanned
                "min_n_rows_across_othervar_levs":min(n_each_other_var), # min n rows across other-levels
                "median_n_rows_across_othervar_levs":int(np.median(n_each_other_var)),
                "max_n_rows_across_othervar_levs":max(n_each_other_var),
                "n_othervar_levs_with_only_one_row":sum([n==1 for n in n_each_other_var]),
                "n_rows_for_each_othervar_lev":n_each_other_var,
                })

        res[var] = pd.DataFrame(resthis)

    return res

def _preprocess_rsa_scalar_population(PA, grouping_vars, version_distance = "pearson",
                                      subtract_mean_each_level_of_var=None, PLOT = False,
                                      sdir=None, PLOT_THEORETICAL_SIMMATS=False,
                                      COMPUTE_SAME_DIFF_DIST = False,
                                      COMPUTE_VS_THEOR_MAT = True
                                      ):
    """
    Operates for a single PA, usually a single subpopulation (e.g., bregion) and time slice.
    """
    from pythonlib.tools.plottools import savefig
    from itertools import permutations
    from scipy.stats import sem

    if PLOT:
        assert sdir is not None

    HACK_SKIP_SORTING = False
    RES = []
    assert isinstance(grouping_vars, list)

    if PLOT:
        plot_example_chan = PA.Chans[0]
    else:
        plot_example_chan = None
    plot_example_split_var="shape_oriented"
    if plot_example_split_var not in grouping_vars:
        plot_example_split_var=grouping_vars[0]

    PAnorm, PAagg, fig, axes, groupdict = popanal_preprocess_scalar(PA, grouping_vars,
                                                                    subtract_mean_each_level_of_var,
                                                                    plot_example_chan=plot_example_chan,
                                                                    plot_example_split_var=plot_example_split_var)
    if PLOT:
        path = f"{sdir}/preprocess_example_chan_{plot_example_chan}.pdf"
        savefig(fig, path)

    # Prune data in PA, to have as balanced as possible, and remove noise.
    if False: # NOt ready
        dict_keeps, dict_levs_keep_each_var, PAagg = _preprocess_prune_pa_enough_data(PAagg, grouping_vars)

    ######################## EXIT, IF NO DATA
    if PAagg is None:
        # Then pruning led to loss of all data
        return None, None, None, None, None

    ####################### CONTINUE
    # Extract Data in Cl format
    Clraw, Clsim = rsa_convert_PA_to_Cl(PAagg, grouping_vars, version_distance)

    DictVarToClsimtheor = {}
    for var in grouping_vars:
        Cltheor, _ = Clsim.rsa_distmat_construct_theoretical(var, PLOT=False)
        DictVarToClsimtheor[var] = Cltheor

    ################## STUFF REALTED TO SIMILARITY STRUCTURE (BETWEEN VARIABLES).
    # Plot heatmaps (raw and sim mats)
    if PLOT:
        if len(grouping_vars)<4:
            list_sort_order = permutations(range(len(grouping_vars)))
        else:
            list_sort_order = [list(range(len(grouping_vars)))]
        for sort_order in list_sort_order:
            figraw, ax = Clraw.rsa_plot_heatmap(sort_order, diverge=False)
            figsim, ax = Clsim.rsa_plot_heatmap(sort_order, diverge=False)
            # - name this sort order
            main_var = grouping_vars[sort_order[0]]
            s = "_".join([str(i) for i in sort_order])
            s+=f"_{main_var}"

            path = f"{sdir}/heat_raw-sort_order_{s}.pdf"
            savefig(figraw, path)

            path = f"{sdir}/heat_sim-sort_order_{s}.pdf"
            savefig(figsim, path)

            # Make the same plots for all theoretical sim mats
            if PLOT_THEORETICAL_SIMMATS:
                for var in grouping_vars:
                    Cltheor = DictVarToClsimtheor[var]
                    figsim, ax = Cltheor.rsa_plot_heatmap(sort_order, diverge=False)
                    # - name this sort order
                    main_var = grouping_vars[sort_order[0]]
                    s = "_".join([str(i) for i in sort_order])
                    s+=f"_{main_var}"
                    # - save
                    path = f"{sdir}/heat_sim-THEOR_{var}-sort_order_{s}.pdf"
                    savefig(figsim, path)


    #### For each var, compare beh PA to ground-truth under each possible variable
    RES_VS_THEOR = []
    list_vec = []
    for var in grouping_vars:
        Cltheor = DictVarToClsimtheor[var]
        # Cltheor, fig = Clsim.rsa_distmat_construct_theoretical(var, PLOT=False)

        # plot
        # if PLOT_THEORETICAL_SIMMATS:
        #     # Plot heatmaps (raw and sim mats)
        #     sort_order = (0,) # each tuple is only len 1...
        #     figsim, ax = Cltheor.rsa_plot_heatmap(sort_order, diverge=False)
        #     s = "_".join([str(i) for i in sort_order])
        #     s+=f"_{var}"
        #     path = f"{sdir}/heat_sim-THEOR-sort_order_{s}.pdf"
        #     savefig(figsim, path)

        if COMPUTE_VS_THEOR_MAT:
            # Correlation matrix between data and theoreitcal sim mats
            # - get upper triangular
            vec_data = Clsim.dataextract_upper_triangular_flattened()
            vec_theor = Cltheor.dataextract_upper_triangular_flattened()
            c = np.corrcoef(vec_data, vec_theor)[0,1]
            if np.isnan(c):
                print(vec_data, vec_theor)
                print(c)
                Cltheor.rsa_plot_heatmap()
                assert False, "proibably becuase only one datapt for each level of var..."

            # Collect
            RES_VS_THEOR.append({
                "var":var,
                "cc":c,
                "Clsim_theor":Cltheor
            })
            list_vec.append(vec_theor)

    if COMPUTE_VS_THEOR_MAT:
        ## MULTIPLE REGRESSION
        from sklearn.linear_model import LinearRegression
        X = np.stack(list_vec) # (nvar, ndat)
        y = Clsim.dataextract_upper_triangular_flattened()
        reg = LinearRegression().fit(X.T,y)

        # collect results
        RES_VS_THEOR_MULT_REGR = []
        for i, var in enumerate(grouping_vars):
            _found = False
            for res in RES_VS_THEOR:
                if res["var"]==var:
                    res["mr_coeff"] = reg.coef_[i]
                    _found=True
            assert _found==True

        dfres_theor = pd.DataFrame(RES_VS_THEOR)
    else:
        dfres_theor = None

    ##### Get mean within-and across context distances
    if COMPUTE_SAME_DIFF_DIST:
        for ind_var in range(len(grouping_vars)):
            vals_same, vals_diff = Clsim.rsa_distmat_quantify_same_diff_variables(ind_var)
            # vals_same, vals_diff = rsa_distmat_quantify_same_diff_variables(Clsim, ind_var)

            # RES.append({
            #     "same_mean": np.mean(vals_same),
            #     "diff_mean": np.mean(vals_diff),
            #     "same_sem": sem(vals_same),
            #     "diff_sem": sem(vals_diff),
            #     "sort_order":sort_order,
            #     "ind_var":ind_var,
            #     "bregion":bregion,
            #     "grouping_vars":grouping_vars,
            #     "subtract_mean_each_level_of_var":subtract_mean_each_level_of_var
            # })

            for same_or_diff, vals in zip(
                    ["same", "diff"],
                    [vals_same, vals_diff]):
                RES.append({
                    "same_or_diff":same_or_diff,
                    "vals":vals,
                    "mean": np.mean(vals),
                    "sem": sem(vals),
                    "sort_order":tuple(sort_order) if sort_order is not None else "IGNORE",
                    "ind_var":ind_var,
                    "ind_var_str":grouping_vars[ind_var],
                    # "bregion":bregion,
                    "grouping_vars":tuple(grouping_vars),
                    "subtract_mean_each_level_of_var":subtract_mean_each_level_of_var,
                    "Clsim":Clsim,
                    "version_distance":version_distance
                })
            plt.close("all")
        dfres_same_diff = pd.DataFrame(RES)
    else:
        dfres_same_diff = None

    return dfres_same_diff, dfres_theor, Clraw, Clsim, PAagg

def preprocess_pca(SP, pca_trial_agg_grouping, pca_time_agg_method=None,
                   list_event_window=None,
                   pca_norm_subtract_condition_invariant = True,
                   list_bregion = None,
                   list_vars_others=None,
                   do_balance=True):
    # pca_trial_agg_grouping = "epoch"
    # pca_trial_agg_grouping = "seqc_0_shape"
    # list_bregion = ["PMv_m"]

    assert False, "Quicker: first extract PA using PA, _ = SP.dataextract_as_popanal_statespace, then split into bregions and timw windows, then do below. see rsa for how donw."
    import pandas as pd

    if list_bregion is None:
        from neuralmonkey.classes.session import REGIONS_IN_ORDER as list_bregion

    RES = []
    for i, (event, pre_dur, post_dur) in enumerate(list_event_window):
        # Get PCA space using all data in window
        for bregion in list_bregion:

            print(f"{event} - {pre_dur} - {post_dur} - {bregion}")

            # Get sites for this regions
            sites = SP.sitegetter_map_region_to_sites(bregion)

            # Do PCA
            PApca, fig, PA, sample_meta = SP.dataextract_as_popanal_statespace_balanced_pca(sites, event, pre_dur, post_dur,
                                                                                            pca_trial_agg_grouping=pca_trial_agg_grouping,
                                                                                            pca_time_agg_method=pca_time_agg_method,
                                                                                            pca_norm_subtract_condition_invariant=pca_norm_subtract_condition_invariant,
                                                                                            pca_plot=False,
                                                                                            list_vars_others=list_vars_others,
                                                                                            do_balance=do_balance)

            RES.append({
                "event":event,
                "pre_dur":pre_dur,
                "post_dur":post_dur,
                "event_wind":f"{event}_{pre_dur}_{post_dur}",
                "bregion":bregion,
                "PApca":PApca,
                "PA":PA,
                "sample_meta":sample_meta
            })

    DF_PA_SPACES = pd.DataFrame(RES)

    return DF_PA_SPACES

# Which dataset to use to construct PCA?
# def pca_make_space(PA, DF, trial_agg_method, trial_agg_grouping,
#     time_agg_method=None,
#     norm_subtract_condition_invariant=False,
#     ploton=True):
#     """ Prperocess data (e.g,, grouping by trial and time) and then
#     Make a PopAnal object holding (i) data for PCA and (ii) the results of
#     PCA.
#     PARAMS:
#     - PA, popanal object, holds all data.
#     - DF, dataframe, with one column for each categorical variable you care about (in DATAPLOT_GROUPING_VARS).
#     The len(DF) must equal num trials in PA (asserts this)
#     - trial_agg_grouping, list of str defining how to group trials, e.g,
#     ["shape_oriented", "gridloc"]
#     - norm_subtract_condition_invariant, bool, if True, then at each timepoint subtracts
#     mean FR across trials
#     RETURNS:
#     - PApca, a popanal holding the data that went into PCA, and the results of PCA,
#     and methods to project any new data to this space.
#     """
#
#     assert DF==None, "instead, put this in PA.Xlabels"
#
#     # First, decide whether to take mean over some way of grouping trials
#     if trial_agg_method==None:
#         # Then dont aggregate by trials
#         PApca = PA.copy()
#     elif trial_agg_method=="grouptrials":
#         # Then take mean over trials, after grouping, so shape
#         # output is (nchans, ngrps, time), where ngrps < ntrials
#         if DF is None:
#             DF = PA.Xlabels["trials"]
#         if False:
#             groupdict = grouping_append_and_return_inner_items(DF, trial_agg_grouping)
#             # groupdict = DS.grouping_append_and_return_inner_items(trial_agg_grouping)
#             PApca = PA.agg_by_trialgrouping(groupdict)
#         else:
#             # Better, since it retains Xlabels
#             PApca = PA.slice_and_agg_wrapper("trials", trial_agg_grouping)
#     else:
#         print(trial_agg_method)
#         assert False
#
#     # First, whether to subtract mean FR at each timepoint
#     if norm_subtract_condition_invariant:
#         PApca = PApca.norm_subtract_mean_each_timepoint()
#
#     # second, whether to agg by time (optional). e..g, take mean over time
#     if time_agg_method=="mean":
#         PApca = PApca.agg_wrapper("times")
#         # PApca = PApca.mean_over_time(return_as_popanal=True)
#     else:
#         assert time_agg_method==None
#
#     print("Shape of data going into PCA (chans, trials, times):", PApca.X.shape)
#     fig = PApca.pca("svd", ploton=ploton)
#
#     return PApca, fig

def _preprocess_extract_PApca(DF_PA_SPACES, event_wind, bregion):
    tmp = DF_PA_SPACES[(DF_PA_SPACES["event_wind"]==event_wind) & (DF_PA_SPACES["bregion"]==bregion)]
    assert len(tmp)==1
    PApca = tmp.iloc[0]["PApca"]
    return PApca


def _preprocess_extract_plot_params(SP, PApca):
    list_event = SP.DfScalar["event"].unique().tolist()
    sites = PApca.Chans

    return list_event, sites

def plot_statespace_grpbyevent_overlay_othervars(PApca, SP, var, vars_others, PLOT_TRIALS=False,
    dims_pc = (0,1), alpha_mean=0.5, alpha_trial=0.2, n_trials_rand=10, time_windows_mean=None,
    list_event_data=None):

    if list_event_data is None:
        # get all events
        list_event_data, _ = _preprocess_extract_plot_params(SP, PApca)

    # 2) One subplot for each event.
    ncols = 3
    nrows = int(np.ceil(len(list_event_data)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))
    for event_dat, ax in zip(list_event_data, axes.flatten()):
        print(event_dat)
        ax.set_title(event_dat)

        # 1) get all data, so that can get what levels of othervar exist
        _, levels_var, levels_othervar = SP.statespate_pca_extract_data(PApca, event_dat, var, vars_others,
                                                                                  levels_var=None,
                                                                                  levels_othervar=None)
        # 2) then iterate over all levles of othervar
        for levother in levels_othervar:
            DICT_DF_DAT, levels_var, _ = SP.statespate_pca_extract_data(PApca, event_dat, var, vars_others,
                                                                                      levels_var=None,
                                                                                      levels_othervar=[levother])
            # Plot
            _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_mean,
                ax, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial, n_trials_rand)
    return fig


def plot_statespace_grpbyevent(PApca, SP, var, vars_others, PLOT_TRIALS=False,
    dims_pc = (0,1), time_windows_mean=None, alpha_mean=0.5, alpha_trial=0.2,
    n_trials_rand=10):
    """ 
    Plot results for this PApca (specifying sites, such as for a bregion), 
    trajectories for each event.
    One subplot for each event, overlaying all levels of (var, othervar)
    """
    list_event, sites = _preprocess_extract_plot_params(SP, PApca)
    ncols = 3
    nrows = int(np.ceil(len(list_event)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))

    for event, ax in zip(list_event, axes.flatten()):
        ax.set_title(event)

        DICT_DF_DAT, _, _ = SP._statespace_pca_extract_data(sites, event, var, vars_others)

        _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_mean,
            ax, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial, n_trials_rand)
    return fig


def plot_statespace_grpbyvarsothers(event, PApca, SP, var, vars_others, PLOT_TRIALS=False,
    dims_pc = (0,1), alpha_trial=0.2, alpha_mean=0.5,
    time_windows_mean=None, n_trials_rand=10):
    """
    One figure for this event
    Each subplot is a level of varsothers
    PARAMS:
    - time_windows_mean, list of 2-tuples, each a (pre_dur, post_dur), where negative
    pre_dur means before. Converts fr from (nchans, times) to (nchans, len(times_windows_mean))
    """
    ## One figure for each event (each subplot a level of othervar)

    # import numpy as np
    # from pythonlib.tools.plottools import makeColors
    # from pythonlib.tools.plottools import legend_add_manual
    # dims_pc = (0,1)
    # sites = PApca.Chans
    # PLOT_TRIALS = False


    # Extract data for this event
    list_event, sites = _preprocess_extract_plot_params(SP, PApca)
    DICT_DF_DAT, levels_var, levels_othervar = SP._statespace_pca_extract_data(sites, event, var, vars_others)

    list_cols = makeColors(len(levels_var))

    # print("levels_var:", levels_var)
    # print("levels_othervar:", levels_othervar)

    # one subplot for each level of othervar
    ncols = 3
    nrows = int(np.ceil(len(levels_othervar)/ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))

    for levother, ax in zip(levels_othervar, axes.flatten()):
        ax.set_title(levother)
        for lev, col in zip(levels_var, list_cols):
            key = (lev, levother)
            if key in DICT_DF_DAT.keys():
                df = DICT_DF_DAT[key]

                if len(df)==0:
                    print(f"No data, skipping: {event}, {key}")
                    continue

                _plot_statespace_df_on_ax(PApca, SP, df, time_windows_mean,
                    ax, col, PLOT_TRIALS, dims_pc, alpha_mean, n_trials_rand=n_trials_rand)

        # overlay legend
        legend_add_manual(ax, levels_var, list_cols, 0.2)

    return fig

def _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_mean,
    ax, PLOT_TRIALS=False, dims_pc=(0,1), alpha_mean=0.5,
    alpha_trial=0.2, n_trials_rand=10):

    list_cols = makeColors(len(DICT_DF_DAT))

    for (key, df), col in zip(DICT_DF_DAT.items(), list_cols):

        if len(df)==0:
            print(f"No data, skipping: {key}")
            continue

        _plot_statespace_df_on_ax(PApca, SP, df, time_windows_mean,
            ax, col, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial=alpha_trial,
            n_trials_rand=n_trials_rand)

    # overlay legend
    legend_add_manual(ax, list(DICT_DF_DAT.keys()), list_cols, 0.1)


def _plot_statespace_df_on_ax(PApca, SP, df, time_windows_mean,
    ax, col = "k", PLOT_TRIALS=False, dims_pc=(0,1), alpha_mean=0.5,
    alpha_trial=0.2, n_trials_rand=10):
    """ Very low-level, plot a single trajectory on a single axis.
    PARAMS:
    - PApca, Popanal, holding the pca results.
    - SP, Snippets, holding data.
    - df, the specific slice of SP.DfScalar which you want to plot.
    - time_windows_mean, list of tuples, each (predur, postdur) if
    you want to bin fr into this trajectory.
    """

    _, sites = _preprocess_extract_plot_params(SP, PApca)

    times_to_mark = [0]
    # times_to_mark_markers = [f"{t}" for t in times_to_mark]
    times_to_mark_markers = ["d"]

    # convert to pa
    # get frmat from data
    PAdata = SP.dataextract_as_popanal_good(df, chans_needed=sites)

    if time_windows_mean is not None:
        PAdata.X, PAdata.Times = PAdata.agg_by_time_windows(time_windows=time_windows_mean)

    # plot
    # For each level of a var, plot it in different color, and overlay many trials
    if False: # Do this outside, using SP
        list_pa, list_levels = PA.split_by_label("trials", var)

    # Plot single datapts
    if PLOT_TRIALS:
        # plot a random subset of trials
        ntrials = PAdata.X.shape[1]
        trials_plot = list(range(ntrials))
        if ntrials > n_trials_rand:
            if False:
                # determistic, but spread throughout trials
                from pythonlib.tools.listtools import random_inds_uniformly_distributed
                trials_plot = random_inds_uniformly_distributed(trials_plot, n_trials_rand)
            else:
                import random
                trials_plot = random.sample(trials_plot, n_trials_rand)

        for trial in trials_plot:
            frmat = PAdata.X[:, trial, :]
            PApca.statespace_pca_plot_projection(frmat, ax, dims_pc=dims_pc, color_for_trajectory=col,
                                        times=PAdata.Times, times_to_mark=times_to_mark,
                                         times_to_mark_markers=times_to_mark_markers,
                                         alpha = alpha_trial,
                                         markersize=3)

    # Overlay the mean trajectory for a level            
    frmean = np.mean(PAdata.X, axis=1)
    PApca.statespace_pca_plot_projection(frmean, ax, dims_pc=dims_pc, color_for_trajectory=col,
                                        times=PAdata.Times, times_to_mark=times_to_mark,
                                         times_to_mark_markers=times_to_mark_markers,
                                         alpha = alpha_mean,
                                         markersize=7, marker="P")
    # grid on, for easy comparisons
    ax.grid()

def plotwrapper_statespace_mult_events():
    # Extract PApca
    tmp = DF_PA_SPACES[(DF_PA_SPACES["event_wind"]==event_wind_pca) & (DF_PA_SPACES["bregion"]==bregion)]
    assert len(tmp)==1
    PApca = tmp["PApca"].item()

    # WHICH DATA?
    fig = plot_statespace_grpbyevent(PApca, SP, var_dat, vars_others_dat, dims_pc = dims_pc,
                                  time_windows_mean = time_windows_traj, alpha_mean=0.2,
                                  PLOT_TRIALS=PLOT_TRIALS)
    fig.savefig(f"{SAVEDIR}/eventpca_{event_wind}-{bregion}-var_{var}-OV_{[str(x) for x in vars_others]}.pdf")


def plotwrapper_statespace_single_event_bregion(DF_PA_SPACES, SP, event_wind_pca, bregion, event_dat,
                             var_dat, vars_others_dat, time_windows_traj=None,
                             savedir=None, dims_pc = (0,1),
                             PLOT_TRIALS=False, n_trials_rand=10,
                             alpha_mean=0.5, alpha_trial=0.2):
    """
    event_wind_pca = "03_samp_0.04_0.6", bregion_pca = "vlPFC_p",
    """
    from pythonlib.tools.plottools import saveMultToPDF


    # Extract PApca
    tmp = DF_PA_SPACES[(DF_PA_SPACES["event_wind"]==event_wind_pca) & (DF_PA_SPACES["bregion"]==bregion)]
    assert len(tmp)==1
    PApca = tmp["PApca"].item()

    _, sites = _preprocess_extract_plot_params(SP, PApca)

    LIST_FIG = []

    ##### 1) Plot all on a single axis
    DICT_DF_DAT, _, _ = SP._statespace_pca_extract_data(sites, event_dat, var_dat, vars_others_dat)

    if len(DICT_DF_DAT)==0:
        return

    if len(DICT_DF_DAT.keys())<8: # Otherwise it is too crowded
        fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
        _plot_statespace_dfmult_on_ax(DICT_DF_DAT, PApca, SP, time_windows_traj,
            ax, PLOT_TRIALS, dims_pc, alpha_mean, alpha_trial, n_trials_rand)
        LIST_FIG.append(fig)

    ##### 2) Plot
    if vars_others_dat is not None:
        fig = plot_statespace_grpbyvarsothers(event_dat, PApca, SP, var_dat, vars_others_dat,
                                           time_windows_mean = time_windows_traj, alpha_mean=alpha_mean,
                                           PLOT_TRIALS=PLOT_TRIALS, dims_pc=dims_pc, n_trials_rand=n_trials_rand,
                                           alpha_trial=alpha_trial)
        LIST_FIG.append(fig)
        # if savedir:
        #     fig.savefig(f"{savedir}/eventpca_{event_wind}-{bregion}-var_{var_dat}-OV_{[str(x) for x in vars_others_dat]}-eventdat_{event_dat}-1.pdf")

        ##### 2) Plot,  FLip the order of var and vars_others
        if "vars_others_tmp" in SP.DfScalar.columns:
            fig = plot_statespace_grpbyvarsothers(event_dat, PApca, SP, "vars_others_tmp", [var_dat],
                                               time_windows_mean = time_windows_traj, alpha_mean=alpha_mean,
                                               PLOT_TRIALS=PLOT_TRIALS, dims_pc=dims_pc, n_trials_rand=n_trials_rand,
                                               alpha_trial=alpha_trial)
            LIST_FIG.append(fig)

            # if savedir:    
            #     fig.savefig(f"{savedir}/eventpca_{event_wind}-{bregion}-var_{var_dat}-OV_{[str(x) for x in vars_others_dat]}-eventdat_{event_dat}-2.pdf")

    if savedir is not None:
        if vars_others_dat is not None:
            vars_others_dat_str = "-".join(vars_others_dat)
        else:
            vars_others_dat_str = "None"
        path = f"{savedir}/eventpca-{event_wind_pca}|{bregion}|eventdat_{event_dat}|dims_{dims_pc}"
        saveMultToPDF(path, LIST_FIG)

def _plot_pca_results(PApca):

    fig, ax = plt.subplots()
    w = PApca.Saved["pca"]["w"]

    # Cum variance explained by traning data
    ax.plot(np.cumsum(w)/np.sum(w), '-or')
    ax.set_title("Training data")
    ax.set_title('cumulative var expl.')
    ax.hlines(0.9, 0, len(w))
    ax.set_ylim(0, 1)
    return fig


def _plot_variance_explained_timecourse(PApca, SP, event_dat, var, vars_others=None,
    time_windows=None, Dimslist = (0,1, 2)):
    """ Plot(overlay) timecourse of variance explained by each dimension, by 
    reprojecting data onto the subspace.
    """
    # 1) How much variance (across levels) accounted for by the first N pcs?
    # - keep the time window constant

    # assert var is None, "not yet coded"
    assert vars_others is None, "not yet coded"
    assert False, "might not be working, is not giving reasonable results..."

    sites = PApca.Chans
    DICT_DF_DAT, _, _ = SP._statespace_pca_extract_data(sites, event_dat)
    dfall = DICT_DF_DAT["alldata"]
    w = PApca.Saved["pca"]["w"]

    # get time windows and popanal
    PAdata = SP.dataextract_as_popanal_good(dfall, chans_needed=sites)

    if time_windows is not None:
        PAdata.X, PAdata.Times = PAdata.agg_by_time_windows(time_windows=time_windows)

    # agg: take mean for each level.
    PAdata = PAdata.slice_and_agg_wrapper("trials", [var])

    # compute total variance
    Vtot = np.var(PAdata.X, axis=1, keepdims=True) # (ndims, 1, ntimes)
    Vtot = np.sum(Vtot, axis=0, keepdims=True) # (1, 1, ntimes), i.e., total variance across neurons at each timepoint

    # project to low-D
    frmat = PAdata.X
    X = PApca.reprojectInput(frmat, Ndim=None, Dimslist=Dimslist)

    # compute variance across levels
    V = np.var(X, axis=1, keepdims=True) # (ndims, 1, ntimes)

    # normalize V to total variance (of unprojeted data)
    Vfrac = V/Vtot

    # plot
    fig, axes = plt.subplots(2,2, sharex=False, figsize=(2*3.5, 2*2.5))

    # - raw variance
    ax = axes.flatten()[0]
    ax.set_title("raw var explained, each dim")
    # ax.set_ylabel("frac var explained")
    for idim in range(V.shape[0]):
        ntimes = V.shape[2]
        ax.plot(PAdata.Times, V[idim, 0, :], label=idim)
    ax.legend()
    ax.set_ylim(bottom=0)
    # ax.set_ylim(bottom=0, top=np.sum(w))

    # - normalized variance
    ax = axes.flatten()[1]
    ax.set_title("var explained, each dim")
    ax.set_ylabel("frac var explained")
    from pythonlib.tools.plottools import makeColors
    pcols = makeColors(len(Dimslist))
    for idim, col in zip(range(Vfrac.shape[0]), pcols):
        ntimes = Vfrac.shape[2]
        ax.plot(PAdata.Times, Vfrac[idim, 0, :], color=col, label=idim)

        # - overlay frac variance explained by each dim in training data
        pcdim = Dimslist[idim]
    #     ax.axhline(w[pcdim], linestyle="--", label=f"trainingdat:{pcdim}", color=col)
        ax.axhline(w[pcdim], linestyle="--", color=col)
    ax.legend()
    ax.set_ylim(0, 1)

    # sum of normalized variance
    ax = axes.flatten()[2]
    ax.set_title("sum (across dims)")
    ax.set_ylabel("sum frac var explained")
    Vfrac_sum = np.sum(Vfrac, axis=0, keepdims=True)
    ax.plot(PAdata.Times, Vfrac_sum[0, 0,:], color="k", label=idim)
    ax.legend()
    ax.set_ylim(0, 1)

#     # - overlay frac variance explained by each dim in training data
#     pcdim = Dimslist[idim]
# #     ax.axhline(w[pcdim], linestyle="--", label=f"trainingdat:{pcdim}", color=col)
#     ax.axhline(w[pcdim], linestyle="--", color=col)

    # Cum variance explained by traning data
    ax = axes.flatten()[3]
    ax.plot(np.cumsum(w)/np.sum(w), '-or')
    ax.set_title("Training data")
    ax.set_title('cumulative var expl.')
    ax.hlines(0.9, 0, len(w))
    ax.set_ylim(0, 1)
    return fig

def _load_pca_space(pca_trial_agg_grouping, animal, DATE):
    SAVEDIR = f"/gorilla1/analyses/recordings/main/pca/{animal}/{DATE}/aggby_{pca_trial_agg_grouping}"

    path = f"{SAVEDIR}/DF_PA_SPACES.pkl"
    with open(path, "rb") as f:
        DF_PA_SPACES = pickle.load(f)

    path = f"{SAVEDIR}/params_pca_space.yaml"
    params_pca_space = load_yaml_config(path)

    print("Loaded this already computed PCA space, with these params:")
    print(params_pca_space)

    return DF_PA_SPACES, params_pca_space, SAVEDIR

def plot_variance_explained_timecourse(SP, animal, DATE, pca_trial_agg_grouping, bregion, event_wind_pca, event_dat):
    """ Plot timecourse of variance explained
    """
    list_vars = [
        "seqc_0_shape",
        "seqc_0_loc",
        "gridsize"
    ]
    list_vars_others = [
        ["seqc_0_loc", "gridsize"],
        ["seqc_0_shape", "gridsize"],
        ["seqc_0_shape", "seqc_0_loc"],
    ]
    vars_others_dat = None
    Dimslist=list(range(3))

    for pca_trial_agg_grouping in list_vars:
        DF_PA_SPACES, params_pca_space, SAVEDIR = _load_pca_space(pca_trial_agg_grouping, animal, DATE)

        tmp = DF_PA_SPACES[(DF_PA_SPACES["bregion"]==bregion) & (DF_PA_SPACES["event_wind"]==event_wind_pca)]
        assert len(tmp)==1
        PApca = tmp.iloc[0]["PApca"]

        for var_dat in list_vars:

            savedir = f"{SAVEDIR}/FIGS/var_explained_timecourse"
            import os
            os.makedirs(savedir, exist_ok=True)

            fig = _plot_variance_explained_timecourse(PApca, SP, event_dat, var_dat, Dimslist=Dimslist);

            path = f"{savedir}/eventpca-{event_wind_pca}|{bregion}|eventdat_{event_dat}|var_{var_dat}|{bregion}.pdf"
            fig.savefig(path)
    #         assert False

            print("--- SAVING AT:", f"{savedir}/var_{var_dat}.pdf")
        plt.close("all")

