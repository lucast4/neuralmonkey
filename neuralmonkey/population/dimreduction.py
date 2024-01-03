""" Methods for working with neural populations, or any other high-D dataset, including 
dim reduction (mainly) and related plots.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def classify(Modthis, modver="logistic", pop= "hidden_init", ndim = 100,
            numsplits = 50, test_size = 0.05, yvar = "nstrokes_task", use_actual_neurons=False):
    """ 
    - use_actual_neurons, then instaed of doing PCA, uses entire population"""
    assert False, "clean this up, I just copied it from drawnn"
    if Modthis is None:
        return None, None, None

    taskcats = {"1line":(1,0), 
     "2line":(2,0),
     "3line":(3,0),
     "4line":(4,0),
     "L":(1,1),
     "linePlusL":(2,1),
     "2linePlusL":(3,1),
     "3linePlusL":(4,1)}

    out = []
    if use_actual_neurons:
        X = Modthis.A.activations[pop]
        fig = None
    else:
        # - get PCA proejctions
        fig = Modthis.pca(pop, ploton=True)
        X = Modthis.P[pop].reproject(Ndim=ndim)

    # - get class labels
    if yvar =="task_categories":
        y = Modthis.Tasks["train_categories"]
        y = mapStrToNum(y)[1]
    elif yvar == "nstrokes_beh":
        y = Modthis.A.calcNumStrokes()
    elif yvar == "nstrokes_task":
        y = [taskcats[t][0] for t in Modthis.Tasks["train_categories"]]
    elif yvar == "task_hand_cats_0":
        y = [taskcats[t][0] for t in Modthis.Tasks["train_categories"]]
    elif yvar == "task_hand_cats_1":
        y = [taskcats[t][1] for t in Modthis.Tasks["train_categories"]]
    else:
        assert False

    for n in range(numsplits):

        # === XVAL split
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)

        # - regression model
        from sklearn import linear_model
        # reg = linear_model.LinearRegression()

        if modver=="logistic":
            reg = linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial")
        elif modver=="ridge":
            reg = linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))
        elif modver=="ridgeclass":
            reg = linear_model.RidgeClassifierCV(alphas = np.logspace(-6, 6, 13))

        reg.fit(Xtrain, ytrain)
        trainscore = reg.score(Xtrain, ytrain)
        testscore = reg.score(Xtest, ytest)

        # --- collect
        out.append({
            "nsplit":n,
            "score":trainscore,
            "scorever":"train"
        })        
        out.append({
            "nsplit":n,
            "score":testscore,
            "scorever":"test"
        })        
    return out, reg, fig

def runclassify(Mod, ModBase, modver, pop, yvar, use_actual_neurons=False):
    assert False, "clean this up, I just copied it from drawnn"

    Out = []
    Reg = {}
    Figs = {}
    
    modellist = ["trained", "baseline"]
    if ModBase is None:
        modellist = ["trained"]

    for model in modellist:
        if model=="trained":
            Modthis = Mod
        elif model=="baseline":
            Modthis = ModBase
        else:
            assert False
        out, reg, fig = classify(Modthis, modver=modver, pop=pop, yvar=yvar, use_actual_neurons=use_actual_neurons)

        for o in out:
            o["model"]=model

        Reg[model] = reg
        Figs[model] = fig

        Out.extend(out)
    return Out, Reg, Figs


def plotStateSpace(X, dims_neural=(0,1), plotndim=2, ax=None, 
    color_for_trajectory="k", is_traj=False, text_plot_pt1=None, alpha=0.5,
    traj_mark_times_inds = None, traj_mark_times_markers = None,
    markersize=3, marker="o"):
    """ general, for plotting state space, will work
    with trajectories or pts (with variance plotted)
    PARAMS:
    - X, input array, shape (neurons, time), if is larger, then
    will raise error. Can think of time as either this being a 
    temporal trajcetory, or each column is a datapt.
    - dim1, dim2, list of inds to take, to slice each dimension. 
    if None:
    --- for dim1 takes first N depending on plotver (2 or 3)
    --- for dim2 takes all. length of dim1 should match the plotndim.
    - plotndim, int, {2, 3} whether 2d or 3d plot
    - is_traj, bool(False), if True, plots lines between dots.
    - text_plot_pt1, str, if not None, then plots this text at the location of 
    the first pt
    - list_color, list of colors, matching len(X).
    - traj_mark_times_inds, list of indices where you want to place a marker on 
    the trajecotry.
    - traj_mark_times_markers, list of markers, matching len of traj_mark_times_inds.
    If None, then marks with 0, 1, 2, 3...
    RETURNS:
    - 
    """
    import seaborn as sns
        
    if traj_mark_times_inds is not None:
        if traj_mark_times_markers is None:
            traj_mark_times_markers = [f"${i}$" for i in range(len(traj_mark_times_inds))] # use 0, 1, ...
        else:
            assert len(traj_mark_times_markers)==len(traj_mark_times_inds)
    if ax is None:
        fig, ax = plt.subplots()
        
    # check that input X is correct shape
    assert len(X.shape)<=2
    if len(X.shape)==1:
        X = X[:, None]
    
    # how many neural dimensions>?
    assert len(dims_neural)==plotndim

    alpha_marker = 3*alpha
    if alpha_marker>1:
        alpha_marker = 1

    # PLOT
    if plotndim==2:
        x1 = X[dims_neural[0], :]
        x2 = X[dims_neural[1], :]

        # dfthis = pd.DataFrame({"x":x1, "y":x2, "color":color_for_trajectory})
        # sns.scatterplot(data=dfthis, x="x", y="y", hue="color", ax=ax, alpha=alpha)
        if is_traj:
            ax.plot(x1, x2, '-', marker=".", color=color_for_trajectory, alpha=alpha, linewidth=2.25)
            ax.plot(x1[-1], x2[-1], "s", mfc="w", mec=color_for_trajectory, alpha=alpha_marker, markersize=markersize) # mark offset
            ax.plot(x1[0], x2[0], marker, mfc=color_for_trajectory, mec="k", alpha=alpha_marker, markersize=markersize) # mark onset
        else:
            # plot smaller markers
            ax.plot(x1[0], x2[0], marker, color=color_for_trajectory, alpha=alpha_marker, markersize=markersize) # mark onset
        
        # plot markers
        if is_traj and traj_mark_times_inds is not None:
            for ind,  marker in zip(traj_mark_times_inds, traj_mark_times_markers):
                # ax.plot(x1[ind], x2[ind], marker="d", color=color_for_trajectory)
                ax.plot(x1[ind], x2[ind], marker=marker, color=color_for_trajectory, markersize=markersize)

        if text_plot_pt1 is not None:
            ax.text(x1[0], x2[0], text_plot_pt1)

    elif plotndim==3:
        assert False, "not coded"
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

    else:
        assert False

    return x1, x2

# def plotStateSpace(X, dims_neural=(0,1), plotndim=2, ax=None, 
#     trajectory_color=None, is_traj=False, text_plot_pt1=None, color_dict=None,
#     hack_remove_outliers=False, alpha=0.5):
#     """ general, for plotting state space, will work
#     with trajectories or pts (with variance plotted)
#     PARAMS:
#     - X, input array, shape (neurons, time), if is larger, then
#     will raise error. Can think of time as either this being a 
#     temporal trajcetory, or each column is a datapt.
#     - dim1, dim2, list of inds to take, to slice each dimension. 
#     if None:
#     --- for dim1 takes first N depending on plotver (2 or 3)
#     --- for dim2 takes all. length of dim1 should match the plotndim.
#     - plotndim, int, {2, 3} whether 2d or 3d plot
#     - is_traj, bool(False), if True, plots lines between dots.
#     - text_plot_pt1, str, if not None, then plots this text at the location of 
#     the first pt
#     - list_color, list of colors, matching len(X). 
#     RETURNS:
#     - 
#     """
#     import seaborn as sns
    
#     if ax is None:
#         fig, ax = plt.subplots()
        
#     # check that input X is correct shape
#     assert len(X.shape)<=2
#     if len(X.shape)==1:
#         X = X[:, None]
    
#     # how many neural dimensions>?
#     assert len(dims_neural)==plotndim
#     # if dim1 is not None:
#     #     assert len(dim1)==plotndim
#     # else:
#     #     dim1 = np.arange(plotndim)
    
#     # # how many time bins?
#     # if dim2 is None:
#     #     dim2 = np.arange(X.shape[1])
    
#     if color_dict is not None:
#         assert list_color is not None, "give the categorcal variable for each edatprt in color"
#         # check that each category has a color
#         for cat in list_color:
#             assert cat in color_dict.keys()

#     # color = np.asarray(color)

#     # PLOT
#     if plotndim==2:
#         x1 = X[dims_neural[0], :]
#         x2 = X[dims_neural[1], :]

#         if hack_remove_outliers:
#             # quick and dirty
#             NSTD = 3.5
#             indsout = (x1 > np.mean(x1)+NSTD*np.std(x1)) | (x1 < np.mean(x1)-NSTD*np.std(x1)) | (x2 > np.mean(x2)+NSTD*np.std(x2)) | (x2 < np.mean(x2)-NSTD*np.std(x2))        
#             x1 = x1[~indsout]
#             x2 = x2[~indsout]
#             # assert sum(indsout)<4, "weird...?"
#             # color = color[~indsout]
#             # # print("Here")
#             # # # print(indsout.shape)
#             # # # print(~indsout)
#             # print(np.argwhere(indsout))

#             inds_bad = np.argwhere(indsout)
#             list_color = [list_color[i] for i in range(len(list_color)) if i not in inds_bad]

#         # ax.scatter(x1, x2, c=color)
#         # print(len(x1))
#         # print(len(x2))
#         # print(len(list_color))    
#         dfthis = pd.DataFrame({"x":x1, "y":x2, "color":list_color})
#         sns.scatterplot(data=dfthis, x="x", y="y", hue="color", ax=ax, palette=color_dict,
#             alpha=alpha)
#         if is_traj:
#             ax.plot(x1, x2, '-', color=list_color, alpha=alpha)
#             ax.plot(x1[0], x2[0], "ok", mfc=list_color) # mark onset
#             ax.plot(x1[-1], x2[-1], "ok", mfc="w") # mark offset

#         if text_plot_pt1 is not None:
#             ax.text(x1[0], x2[0], text_plot_pt1)

#     elif plotndim==3:
#         assert False, "not coded"
#         # %matplotlib notebook
#         fig, axes = plt.subplots(1,2, figsize=(12,6))
#         from mpl_toolkits.mplot3d import Axes3D
    
#         # --- 1
#         ax = fig.add_subplot(121, projection='3d')
#         ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=[x for x in Mod.A.calcNumStrokes()])
#         # --- 2
#         tasks_as_nums = mapStrToNum(Mod.Tasks["train_categories"])[1]
#         ax = fig.add_subplot(122, projection='3d')
#         ax.scatter(Xsub[:,0], Xsub[:,1], Xsub[:,2], c=tasks_as_nums)

#     else:
#         assert False

#     return x1, x2