""" Plots on population data X, dimensions (channels, trials, timesteps)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def _heatmap_mat(Xarr, ax):
#     """
#     """
#     heatmap_mat(Xarr, ax, annotate_heatmap=False, labels_col=times_str, diverge=True, robust=True, zlims=zlims)


        

def heatmapwrapper_stratified_each_neuron_alltrials(PA, y_group_var=None):
    """ 
    Figure with multiple subplots, each a neuron(chan), and for that neuron, plot heatmap of FR,
    all trials, chronological, vs. time.
    And then overlay grouping labels on top of that (but doesnt use anything about grouping to decide how
    to order y axis.)

    Guaranteed to be sorted by trial (top is first trial)

    NOTE: figure can be large and slow.
    """

    PA = PA.sort_trials_by_trialcode() # first, sort with trialcode_scal
    dflab = PA.Xlabels["trials"]
    X = PA.X
    times = PA.Times
    task_kinds = dflab["task_kind"].values

    # Overlay boundaries in the plot
    if y_group_var is not None:
        from pythonlib.tools.pandastools import group_continuous_blocks_of_rows
        df_sorted, _, group_boundaries = group_continuous_blocks_of_rows(dflab, y_group_var, "trialcode_scal", 
                                                                         f"{y_group_var}_group", do_sort=False)
        inds_start = group_boundaries["Start"].tolist()
        groups = group_boundaries[f"{y_group_var}_group"].tolist()
        groups_general = [dflab.iloc[i][y_group_var] for i in inds_start]
    else:
        inds_start = None
        groups_general = None
    # else:
    #     # Then dont put groups

    if False:
        # False, since it shold already by sorted
        from pythonlib.tools.stringtools import trialcode_to_tuple
        trialcode_tuples = [trialcode_to_tuple(tc) for tc in dflab["trialcode"]]
        trialcode_scalars = [trialcode_to_scalar(tc) for tc in dflab["trialcode"]]
        sort_inds = [x[0] for x in sorted([(i, tct) for i, tct in enumerate(trialcode_tuples)], key=lambda x:x[1])]
        Xsorted = X[:, sort_inds, :]
        task_kinds = task_kinds[sort_inds]
    else:
        Xsorted = X

    ncols = 6
    n_chans = len(PA.Chans)
    nrows = int(np.ceil(n_chans/ncols))
    SIZE=3
    zlims = (-1, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE))
    for i, ind_neur in enumerate(range(Xsorted.shape[0])):
        ax = axes.flatten()[i]
        x = Xsorted[ind_neur, :, :]
        _heatmap_stratified_y_axis(x, times, ax, zlims=zlims, list_y_group_onsets=inds_start, list_group_labels=groups_general)
        ax.set_title(f"{i}-{PA.Chans[i]}", color="r")
    
    return fig

def heatmap_stratified_trials_grouped_by_neuron(PA, inds, ax, n_rand_trials=10, zlims=None):
    """
    Plot these trials, but showing each neuron in same plot.
    - y, trials, grouped by neuron (and mult trials withim that nerun). Think of this as a subplot fixed on a single
    variable level (represented by trials)

    NOTE: trial order will be increasing (lowest at top)

    NOTE: 
    PARAMS:
    - inds, rows of PA.
    """
    import random

    times = PA.Times

    # If many trials, take random subset
    if len(inds)>n_rand_trials:
        indsrand = random.sample(range(len(inds)), n_rand_trials)
        inds = [inds[i] for i in indsrand]
        inds = sorted(inds)

    # Get neural data just for these trials
    X = PA.X[:, inds, :]

    # reshape to (neurons x trials, times)
    chans = PA.Chans
    nchans = X.shape[0]
    ntrials = X.shape[1]
    ntimes = X.shape[2]
    nrows = ntrials * nchans
    Xarr = np.reshape(X, (nchans*ntrials, ntimes))
    neurons_row_starts = np.arange(0, nrows, ntrials).tolist()

    _heatmap_stratified_y_axis(Xarr, times, ax, neurons_row_starts, chans, zlims=zlims)

def heatmap_stratified_neuron_grouped_by_var(PA, ind_neur, ax, n_rand_trials, zlims, y_group_var, y_group_var_levels=None):
    """
    Plot this neuron, each plot:
    - y, trials grouped by levels of y_group_var. For each group, plot a random subsample, 
    grouped on y axis. where is sorted within y axis (based 
    on index row within PA).
    
    NOTE: trial order will be increasing (lowest at top)
    """
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    import random

    grpdict = grouping_append_and_return_inner_items_good(PA.Xlabels["trials"], [y_group_var], y_group_var_levels, 
                                                          must_get_all_groupouter_levels=False)

    if len(grpdict)==0 or grpdict is None:
        # no data to plot
        return 
    
    times = PA.Times

    # Collect data
    list_x = []
    list_labels = []
    list_onset = []
    on = 0
    # n_rand_trials = 8
    for grp, inds in grpdict.items():

        # If many trials, take random subset
        if len(inds)>n_rand_trials:
            indsrand = random.sample(range(len(inds)), n_rand_trials)
            inds = [inds[i] for i in indsrand]
            inds = sorted(inds)
        x = PA.X[ind_neur, inds, :]
        list_x.append(x)
        list_labels.extend([grp for _ in range(len(inds))])
        list_onset.append(on)
        on+=len(inds)

    Xarr = np.concatenate(list_x, axis=0)
    _heatmap_stratified_y_axis(Xarr, times, ax, list_onset, list(grpdict.keys()), zlims=zlims)

def _heatmap_stratified_y_axis(Xarr, times, ax, 
                               list_y_group_onsets=None, list_group_labels=None, 
                               zlims=None):
    """
    PARAMS:
    - Xarr, (nrows, ntimes), where rows represent stratified features (e..g, neuron--trial), but this function doesnt  need to know.
    - list_y_group_onsets, list of y values that mark onset of each group, where 0 puts a line at the very top.
    """
    from pythonlib.tools.snstools import heatmap_mat

    times_str = [f"{t:.2f}" for t in times]
    heatmap_mat(Xarr, ax, annotate_heatmap=False, labels_col=times_str, diverge=True, robust=True, zlims=zlims)
    ax.set_xlabel("times")

    # Put lines demarcating each neuron's data onset

    if list_y_group_onsets is not None:
        for i, row in enumerate(list_y_group_onsets):
            ax.axhline(row, color="k", linestyle="--", alpha=0.5, linewidth=1)
    
    # First, remove the current ticks
    ax.tick_params(axis='y',label1On=False)

    # Put labels for each group
    if list_group_labels is not None:
        assert len(list_y_group_onsets)==len(list_group_labels), "they need to match..."
        from pythonlib.tools.plottools import add_secondary_axis_label_nested
        onsets = np.array(list_y_group_onsets)
        onset_diffs = np.diff(onsets)
        onsets_mid = np.r_[onset_diffs, Xarr.shape[0]-onsets[-1]]/2 + onsets
        add_secondary_axis_label_nested(ax, onsets_mid, list_group_labels, "y")

def subsample_rand(X, n_rand):
    """ get random sample (dim 0), or return all if n_rand larget than X.
    """
    import random
    if n_rand>X.shape[0]:
        return X, list(range(X.shape[0]))

    inds = random.sample(range(X.shape[0]), n_rand)
    return X[inds, ...], inds

def plotNeurHeat(X, ax=None, barloc="right", robust=True, zlims = None,
                 times=None):
    """ plot heatmap for data X.
    X must be (neuron, time)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))

    
    # X = self.activations[pop][tasknum]
    X_nonan = X[:]
    X_nonan = X_nonan[~np.isnan(X_nonan)]
    minmax = (np.min(X_nonan), np.max(X_nonan))

    if times is None:
        times = list(range(X.shape[1]))
    # plot
    if zlims is not None: 
        sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
               robust=robust, vmin=zlims[0], vmax=zlims[1])
    else:
        sns.heatmap(X, ax=ax, cbar=False, cbar_kws = dict(use_gridspec=False,location=barloc), 
               robust=robust)
    # ax.set_title(f"{pop}|{minmax[0]:.2f}...{minmax[1]:.2f}")
    ax.set_xlabel(f"robust={robust}|{minmax[0]:.2f}...{minmax[1]:.2f}")
    ax.set_ylabel('neuron #')
    indtimes = list(range(len(times)))
    ax.set_xticks([indtimes[0], indtimes[-1]], labels=[times[0], times[-1]])
        
# def plotNeurTimecourseSingleTrace(x, times, ax):
#     """ plot a single fr trace on axis
#     PARAMS
#     """

def plot_smoothed_fr(frmat, times=None, ax=None, summary_method="mean", error_ver="sem",
    color="k"):
    """
    Low-level plot of smoothed fr, mean of frmat
    PARAMS:
    - frmat, (ntrials, time)
    """

    if error_ver=="sem":
        from scipy import stats
        if summary_method=="mean":
            Xmean = np.mean(frmat, axis=0)
        elif summary_method=="median":
            Xmean = np.median(frmat, axis=0)
        else:
            print(summary_method)
            assert False
        Xsem = stats.sem(frmat, axis=0)
    else:
        assert error_ver is None, "not coded"
        
    if times is None:
        times = np.arange(frmat.shape[1])

    fig, ax = plotNeurTimecourseErrorbar(Xmean, Xerror=Xsem, times=times,ax=ax, color=color)
    return fig, ax



def plotNeurTimecourse(X, times=None, ax=None, n_rand=None, marker="-", color="k",
    alpha=None):
    """ Plot overlaid timecourses. 
    - X, (neuron/trials, time)
    - times, vector of timestmaps, len as X.shape[1]. if None, then uses indices 0,1,2, ..
    - Xerror, (neuron/trials, time), to add errorbars)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    else:
        fig = None
    

    # X = self.activations[pop][tasknum]
    X_nonan = X[:]
    X_nonan = X_nonan[~np.isnan(X_nonan)]
    minmax = (np.min(X_nonan), np.max(X_nonan))
    
    if alpha is None:
        alpha=0.75

    if n_rand is not None:
        X, indsrand = subsample_rand(X, n_rand)
    if times is None:
        t = np.arange(X.shape[1])
    else:
        t = times
    ax.plot(t, X.T, marker, color=color, alpha=alpha)

    # if Xerror is not None:
    #     from pythonlib.tools.plottools import shadedErrorBar
    #     assert Xmean is not None
    #     shadedErrorBar(t, Xmean, yerr=Xerror, ax=ax)
    return fig, ax
    
def plotNeurTimecourseErrorbar(Xmean, Xerror, times=None, ax=None, color="k",
    alpha_fill=0.4):
    """ Plot timecourse + shaded error bar
    PARAMS:
    - Xmean, (times,)
    - Xerror, (times,)
    """
    from pythonlib.tools.plottools import shadedErrorBar
    assert len(Xmean.shape)==1
    assert len(Xmean)==len(Xerror)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))
    else:
        fig = None
    
    if times is None:
        t = np.arange(len(Xmean))
    else:
        t = times
    npts = len(t)
    if npts>1:
        # Then plot curve
        shadedErrorBar(t, Xmean, yerr=Xerror, ax=ax, color=color, alpha_fill=alpha_fill)
    else:
        # Plot point
        ax.errorbar(t, Xmean, Xerror, linestyle="", marker="o", color=color, alpha=alpha_fill)

    return fig, ax


# def plotStateSpace(X, dim1=None, dim2=None, plotndim=2, ax=None, color=None):
#     """ general, for plotting state space, will work
#     with trajectories or pts (with variance plotted)
#     X, shape should always be (neurons, time), if is larger, then
#     will raise error. 
#     - dim1, 2, list of inds to take. if None, then will take auto (for dim1
#     takes first N depending on plotver) and dim2 takes all. length of dim1 should
#     match the plotndim.
#     - plotndim, [2, 3] whether 2d or 3d
#     """
#     assert False, "use the one in dimreduction"
#     assert False, "copied over from drawnn.notebooks_analy.analy_everythinguptonow_021021 Not sure if works here."
    
#     if ax is None:
#         fig, ax = plt.subplots()
        
#     # check that input X is correct shape
#     assert len(X.shape)<=2
    
#     # how many neural dimensions>?
#     if dim1 is not None:
#         assert len(dim1)==plotndim
#     else:
#         dim1 = np.arange(plotndim)
    
#     # how many time bins?
#     if dim2 is None:
#         dim2 = np.arange(X.shape[1])    
#     # PLOT
#     if plotndim==2:
#         x1 = X[dim1[0], dim2]
#         x2 = X[dim1[1], dim2]
#         ax.scatter(x1, x2, c=color)
#         ax.plot(x1, x2, '-', color=color)
#         if len(x1)>1:
#             ax.plot(x1[0], x2[0], "ok") # mark onset
#     # elif plotndim==3:
#     #     assert False, not coded
#     #     # %matplotlib notebook
#     #     fig, axes = plt.subplots(1,2, figsize=(12,6))
#     #     from mpl_toolkits.mplot3d import Axes3D
    
#     #     # --- 1
#     #     ax = fig.add_subplot(121, projection='3d')
#     #     ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=[x for x in Mod.A.calcNumStrokes()])
#     #     # --- 2
#     #     tasks_as_nums = mapStrToNum(Mod.Tasks["train_categories"])[1]
#     #     ax = fig.add_subplot(122, projection='3d')
#     #     ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=tasks_as_nums)

#     # fig, ax= plt.subplots()
#     # for b in [0,1]:
#     #     X = Xmean[:,:,b]
#     #     plotStateSpace(X, ax=ax)


