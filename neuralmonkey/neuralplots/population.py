""" Plots on population data X, dimensions (channels, trials, timesteps)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def subsample_rand(X, n_rand):
    """ get random sample (dim 0), or return all if n_rand larget than X.
    """
    import random
    if n_rand>X.shape[0]:
        return X, list(range(X.shape[0]))

    inds = random.sample(range(X.shape[0]), n_rand)
    return X[inds, ...], inds

def plotNeurHeat(X, ax=None, barloc="right", robust=True, zlims = None):
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
        t = np.arange(len(X))
    else:
        t = times
    shadedErrorBar(t, Xmean, yerr=Xerror, ax=ax, color=color, alpha_fill=alpha_fill)

    return fig, ax


def plotStateSpace(X, dim1=None, dim2=None, plotndim=2, ax=None, color=None):
    """ general, for plotting state space, will work
    with trajectories or pts (with variance plotted)
    X, shape should always be (neurons, time), if is larger, then
    will raise error. 
    - dim1, 2, list of inds to take. if None, then will take auto (for dim1
    takes first N depending on plotver) and dim2 takes all. length of dim1 should
    match the plotndim.
    - plotndim, [2, 3] whether 2d or 3d
    """
    assert False, "use the one in dimreduction"
    assert False, "copied over from drawnn.notebooks_analy.analy_everythinguptonow_021021 Not sure if works here."
    import seaborn as sns
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # check that input X is correct shape
    assert len(X.shape)<=2
    
    # how many neural dimensions>?
    if dim1 is not None:
        assert len(dim1)==plotndim
    else:
        dim1 = np.arange(plotndim)
    
    # how many time bins?
    if dim2 is None:
        dim2 = np.arange(X.shape[1])    
    # PLOT
    if plotndim==2:
        x1 = X[dim1[0], dim2]
        x2 = X[dim1[1], dim2]
        ax.scatter(x1, x2, c=color)
        ax.plot(x1, x2, '-', color=color)
        if len(x1)>1:
            ax.plot(x1[0], x2[0], "ok") # mark onset
    elif plotndim==3:
        assert False, not coded
        # %matplotlib notebook
        fig, axes = plt.subplots(1,2, figsize=(12,6))
        from mpl_toolkits.mplot3d import Axes3D
    
        # --- 1
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=[x for x in Mod.A.calcNumStrokes()])
        # --- 2
        tasks_as_nums = mapStrToNum(Mod.Tasks["train_categories"])[1]
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=tasks_as_nums)

    # fig, ax= plt.subplots()
    # for b in [0,1]:
    #     X = Xmean[:,:,b]
    #     plotStateSpace(X, ax=ax)


