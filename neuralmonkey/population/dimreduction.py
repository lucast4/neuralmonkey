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



def plotStateSpace(X, dim1=None, dim2=None, plotndim=2, ax=None, 
    color=None, is_traj=False, text_plot_pt1=None, color_dict=None,
    hack_remove_outliers=False):
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
    RETURNS:
    - 
    """
    import seaborn as sns
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # check that input X is correct shape
    assert len(X.shape)<=2
    if len(X.shape)==1:
        X = X[:, None]
    
    # how many neural dimensions>?
    if dim1 is not None:
        assert len(dim1)==plotndim
    else:
        dim1 = np.arange(plotndim)
    
    # how many time bins?
    if dim2 is None:
        dim2 = np.arange(X.shape[1])
    
    if color_dict is not None:
        assert color is not None, "give the categorcal variable for each edatprt in color"
        # check that each category has a color
        for cat in color:
            assert cat in color_dict.keys()

    # color = np.asarray(color)

    # PLOT
    if plotndim==2:
        x1 = X[dim1[0], dim2]
        x2 = X[dim1[1], dim2]

        if hack_remove_outliers:
            # quick and dirty
            NSTD = 3.5
            indsout = (x1 > np.mean(x1)+NSTD*np.std(x1)) | (x1 < np.mean(x1)-NSTD*np.std(x1)) | (x2 > np.mean(x2)+NSTD*np.std(x2)) | (x2 < np.mean(x2)-NSTD*np.std(x2))        
            x1 = x1[~indsout]
            x2 = x2[~indsout]
            # assert sum(indsout)<4, "weird...?"
            # color = color[~indsout]
            # # print("Here")
            # # # print(indsout.shape)
            # # # print(~indsout)
            # print(np.argwhere(indsout))

            inds_bad = np.argwhere(indsout)
            color = [color[i] for i in range(len(color)) if i not in inds_bad]

        # ax.scatter(x1, x2, c=color)    
        dfthis = pd.DataFrame({"x":x1, "y":x2, "color":color})
        sns.scatterplot(data=dfthis, x="x", y="y", hue="color", ax=ax, palette=color_dict)
        if is_traj:
            ax.plot(x1, x2, '-', color=color)
            ax.plot(x1[0], x2[0], "ok") # mark onset

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