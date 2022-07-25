""" Static represnetation of prims (clean behavior, e..,g, prims in grid), in population 
state space

See notebook: 220713_prims_state_space

"""

import numpy as np
import matplotlib.pyplot as plt
from ..population.dimreduction import plotStateSpace
import random
import os

def plot_results_state_space_by_group_(DatDict, DATAPLOT_GROUPING_VARS, ResDict, 
    plot_dims=[0,1], COLOR_BY = "shape", TEXT_LABEL_BY="group",
    overlay_strokes="beh", plot_mean=False, ax=None, color_dict=None):
    DatGrp = DatDict["DatGrp"]
    DS = DatDict["DS"]

    from pythonlib.tools.plottools import move_legend_outside_axes

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))

    if plot_mean:
        DATAVER = "X_timetrialmean"
    else:
        DATAVER = "X_timemean"

    list_x = []
    list_grp_for_color = []
    list_firstpointinds = []
    list_grp_for_text = []
    list_grp_alignedto_firstpoint = []
    for dat in DatGrp:
        x = dat[DATAVER]
        if len(x.shape)==1:
            x = x[:, None]

        def _get_group_(by):
            if by=="group":
                grp = dat["group"]
            elif by=="shape":
                indthis = DATAPLOT_GROUPING_VARS.index("shape_oriented")
                grp = dat["group"][indthis]
            elif by=="location":
                indthis = DATAPLOT_GROUPING_VARS.index("gridloc")
                grp = dat["group"][indthis]
            elif by=="gridsize":
                indthis = DATAPLOT_GROUPING_VARS.index("gridsize")
                grp = dat["group"][indthis]
            else:
                indthis = DATAPLOT_GROUPING_VARS.index(by)
                grp = dat["group"][indthis]
            return grp

        grp_for_color = _get_group_(COLOR_BY)
        grp_for_text = _get_group_(TEXT_LABEL_BY)
        grp = _get_group_("group")

        list_x.append(x)
        list_firstpointinds.append(len(list_grp_for_color)) # useful for plotting
        list_grp_alignedto_firstpoint.append(grp)
        list_grp_for_color.extend([grp_for_color for _ in range(x.shape[1])])
        list_grp_for_text.append(grp_for_text);

    # Concatenate to (nchans, ndatapts)
    X = np.concatenate(list_x, axis=1)
    assert len(list_grp_for_color)==X.shape[1]

    # Plot
    x1, x2 = plotStateSpace(X, dim1=plot_dims, ax=ax, color=list_grp_for_color, color_dict=color_dict)
    move_legend_outside_axes(ax)

    # Put text on first pts.
    for i, ind in enumerate(list_firstpointinds):
        ax.text(x1[ind], x2[ind], list_grp_for_text[i])

    # Plot strokes at each pt
    # PLOT DRAWINGS, with their location in state space
    # get stroke inds
    if overlay_strokes is not None:
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
        groupdict = grouping_append_and_return_inner_items(DS.Dat, DATAPLOT_GROUPING_VARS)
        for i, ind in enumerate(list_firstpointinds):
            loc = [x1[ind], x2[ind]]
            grp = list_grp_alignedto_firstpoint[i]

            list_indstroke = groupdict[grp]
            indstroke = random.choice(list_indstroke) # take random one
            strok = DS.extract_strokes(inds=[indstroke], ver_behtask=overlay_strokes)[0][:,:2]

            # center, to strok onset, then rescale
            pt0 = strok[0]
            strok = strok-pt0 # center to onset

            strok = strok/50 # rescale

            XLIM = ax.get_xlim()
            YLIM = ax.get_ylim()
            xdiff = XLIM[1] - XLIM[0]
            ydiff = YLIM[1] - YLIM[0]
            strok[:,0] =  strok[:,0] * xdiff/ydiff # so it looks correct even if aspect is not 1
            strok = strok + loc # move to new locatin

            # plot at that location, starting from strok onset
            ax.plot(strok[:,0], strok[:,1], '-k')
            ax.plot(strok[0,0], strok[0,1], 'ok')


def plot_results_state_space_by_group(DatDict, DATAPLOT_GROUPING_VARS, ResDict, 
    plot_dims=[0,1], COLOR_BY = "shape", TEXT_LABEL_BY="group",
    overlay_strokes="beh", plot_mean=False):
    
    DatGrp = DatDict["DatGrp"]
    DATAPLOT_GROUPING_VARS = ParamsDict["DATAPLOT_GROUPING_VARS"]
    DS = DatDict["DS"]

    from pythonlib.tools.plottools import move_legend_outside_axes

    fig, axes = plt.subplots(len(LIST_DATAVER),1, figsize=(15,len(LIST_DATAVER)*10))
    for ax, plot_mean in zip(axes.flatten(), [True, False]):
        plot_results_state_space_by_group_(DatDict, DATAPLOT_GROUPING_VARS, ResDict, 
            plot_dims=[0,1], COLOR_BY = "shape", TEXT_LABEL_BY="group",
            overlay_strokes="beh", plot_mean=plot_mean, ax=ax)

    return fig


def get_levels(df, category):
    """ Return sorted list of unique levels for this category (column name)
    """
    if category=="shape":
        category="shape_oriented"
    return sorted(df[category].unique().tolist())


def plotall_state_space(PA, DF, MS, REGIONS, VARNAME_SUBPLOT_LEVELS,
        dims_plot, overlay_strokes, plot_mean,
        DATAPLOT_GROUPING_VARS = ["shape_oriented", "gridloc", "gridsize"], 
        VERSION="pca",
        pca_trial_agg_grouping=["shape_oriented", "gridloc", "gridsize"], 
        pca_trial_agg_method = "grouptrials",
        pca_time_agg_method=None,
        DS=None):
    
    if overlay_strokes is not None:
        assert isinstance(overlay_strokes, str), "beh, task, etc"
    if overlay_strokes:
        assert DS is not None
        assert len(DS.Dat)==PA.X.shape[1], "n trials dont match"

    # Slice dataset to get specific subgroups
    from pythonlib.tools.pandastools import filterPandas, applyFunctionToAllRows
    from neuralmonkey.analyses.prims_state_space import plot_results_state_space_by_group, plot_results_state_space_by_group_
    from pythonlib.neural.population import compute_data_projections

    # # VARNAME_SUBPLOT_LEVELS = 'none_colorby_loc' # e.g., if gridloc, then each subplot is a different location
    # # VARNAME_SUBPLOT_LEVELS = 'none_colorby_shape' # e.g., if gridloc, then each subplot is a different location
    # # VARNAME_SUBPLOT_LEVELS = 'gridloc' # e.g., if gridloc, then each subplot is a different location
    # VARNAME_SUBPLOT_LEVELS = 'shape_oriented' # e.g., if gridloc, then each subplot is a different location

    # # Which area
    # # REGIONS = ["M1_m", "M1_l"]
    # REGIONS = ["dlPFC_p", "dlPFC_a"]
    # # REGIONS = ["preSMA_p", "preSMA_a"]

    # # Annotation/color
    # dims_plot = [0,1]
    # overlay_strokes = None
    # plot_mean = False # false means plot trials

    ####
    from pythonlib.tools.pandastools import filterPandasMultOrs
    from pythonlib.tools.plottools import color_make_pallete_categories
    import copy

    # - color by the variable that is NOT defining the subplot
    if VARNAME_SUBPLOT_LEVELS=="gridloc":
        COLOR_BY = "shape"
        TEXT_BY = "shape"
        LIST_LEVELS = get_levels(DF, VARNAME_SUBPLOT_LEVELS)
    elif VARNAME_SUBPLOT_LEVELS=="shape_oriented":
        COLOR_BY = "gridloc"
        TEXT_BY = "gridloc"
        LIST_LEVELS = get_levels(DF, VARNAME_SUBPLOT_LEVELS)
    elif VARNAME_SUBPLOT_LEVELS=="none_colorby_shape":
        COLOR_BY = "shape"
        TEXT_BY = "shape"
        LIST_LEVELS = [None]
    elif VARNAME_SUBPLOT_LEVELS=="none_colorby_loc":
        COLOR_BY = "gridloc"
        TEXT_BY = "gridloc"
        LIST_LEVELS = [None]
    else:
        assert False
    
    fig, axes = plt.subplots(3,3, sharex=True, sharey=True, figsize=(15, 15))

    # for sh, ax in zip(LIST_SHAPES, axes.flatten()):
    for level, ax in zip(LIST_LEVELS, axes.flatten()):

        # Subset of data to plot
    #     gridloc = None
    # #     shapes = ["Lcentered-4-0"]
    #     shapes = [sh]
    #     # shapes = None
    #     gridsize = None
        if VARNAME_SUBPLOT_LEVELS=="gridloc":
            list_varlevels = [
                None,
                [level],
                None
            ]
        elif VARNAME_SUBPLOT_LEVELS in ["shape", "shape_oriented"]:
            list_varlevels = [
                [level],
                None,
                None
            ]
        elif VARNAME_SUBPLOT_LEVELS in ["none_colorby_loc", "none_colorby_shape"]:
            # plot all data on single plot
            list_varlevels = [
                None,
                None,
                None
            ]
        else:
            assert False

        
    #     COLOR_BY = "location"
        if COLOR_BY=="shape":
            colby = "shape_oriented"
        else:
            colby = COLOR_BY
        pallete = color_make_pallete_categories(DF, colby)
        
        ######################
        # - extract data grouped by conjuction of all bars.
        DATAPLOT_GROUPING_VARS = ["shape_oriented", "gridloc", "gridsize"]
    #     _, _, DatGrpDfThis  = compute_data_projections(REGIONS, DATAPLOT_GROUPING_VARS, ploton=False)
        _, _, DatGrpDfThis = compute_data_projections(PA, DF, MS, VERSION, REGIONS, DATAPLOT_GROUPING_VARS, 
                                    pca_trial_agg_grouping, pca_trial_agg_method, 
                                    pca_time_agg_method, ploton=False)

        DatGrpSliced = filterPandasMultOrs(DatGrpDfThis, DATAPLOT_GROUPING_VARS, [list_varlevels], return_as="dict")

        # - Combine things into dicts
        DatDict, ResDict = {}, {}
        DatDict["DatGrp"] = DatGrpSliced
        DatDict["DS"] = DS

        # - Plot
        plot_results_state_space_by_group_(DatDict, DATAPLOT_GROUPING_VARS, ResDict, dims_plot, COLOR_BY, TEXT_BY,
                                         overlay_strokes=overlay_strokes, plot_mean=plot_mean, ax=ax,
                                          color_dict=pallete)
        # fig = plot_results_state_space_by_group(DatDict, ParamsDict, ResDict, dims_plot, COLOR_BY, TEXT_BY,
        #                                  overlay_strokes=overlay_strokes)

        ax.set_title(level)
    return fig

def plotall_state_space_iter_hyperparams(MS, DS, LIST_PCA_AGG, LIST_REGIONS, LIST_VARNAMES_SUBPLOT,
        DATAPLOT_GROUPING_VARS, SAVEDIR):
    """ Make many plots, each one variation of hyperparam combos
    PARAMS;
    - MS, MultSession
    - DS, DatStrokes
    - LIST_PCA_AGG, list of lists, each inner list a way to group data (mean) before constructing PCA
    - LIST_REGIONS, list of list of string, each inner list a set of brain regions to plot
    - LIST_VARNAMES_SUBPLOT, list of variables for levels for subplots.
    """

    from pythonlib.tools.pandastools import filterPandas, applyFunctionToAllRows
    from pythonlib.neural.population import extract_neural_snippets_aligned_to
    from neuralmonkey.analyses.prims_state_space import plot_results_state_space_by_group, plot_results_state_space_by_group_
    from neuralmonkey.analyses.prims_state_space import plotall_state_space


    # 1) What to align data to
    for align_to in ["go_cue", "stroke_onset"]:
    #     align_to = "go_cue"
        PAall = extract_neural_snippets_aligned_to(MS, DS, align_to=align_to)

        # Slice dataset to get specific subgroups
        for pca_trial_agg_grouping in LIST_PCA_AGG:

            for REGIONS in LIST_REGIONS:
                # 1) PCA space 
                # pca_trial_agg_grouping=["shape_oriented", "gridloc", "gridsize"]
            #     pca_trial_agg_grouping=["shape_oriented"]
                # pca_trial_agg_grouping=["gridloc"]

                # 2) Which area

                for VARNAME_SUBPLOT_LEVELS in LIST_VARNAMES_SUBPLOT:
        #             # 3) Plotting params
        #             # VARNAME_SUBPLOT_LEVELS = 'none_colorby_loc' # e.g., if gridloc, then each subplot is a different location
        #             # VARNAME_SUBPLOT_LEVELS = 'none_colorby_shape' # e.g., if gridloc, then each subplot is a different location
        #             VARNAME_SUBPLOT_LEVELS = 'gridloc' # e.g., if gridloc, then each subplot is a different location
        #             # VARNAME_SUBPLOT_LEVELS = 'shape_oriented' # e.g., if gridloc, then each subplot is a different location

                    # -- Annotation/color
                    dims_plot = [0,1]
                    overlay_strokes = None
                    # overlay_strokes = "beh"

                    for plot_mean in [True, False]:
                    #     plot_mean = True # false means plot trials

                        ##########################################
                        # SAVE PARAMS
                        # ParamsDict["DATAPLOT_GROUPING_VARS"] = DATAPLOT_GROUPING_VARS 
                        # ParamsDict["REGIONS"] = REGIONS

                        sdir = f"{SAVEDIR}/plot_results_state_space_pca/alignto_{align_to}-PCAby_{'__'.join(pca_trial_agg_grouping)}"
                        os.makedirs(sdir, exist_ok=True)

                        # 1) Compute the PCA space and 2) Plot
                        fig = plotall_state_space(PAall, DS.Dat, MS, REGIONS, VARNAME_SUBPLOT_LEVELS,
                                dims_plot, overlay_strokes, plot_mean,
                                DATAPLOT_GROUPING_VARS,
                                VERSION="PCA",
                                pca_trial_agg_grouping=pca_trial_agg_grouping,
                                pca_trial_agg_method = "grouptrials",
                                pca_time_agg_method=None,
                                DS=DS)
                        if REGIONS is None:
                            regions = ["ALL"]
                        else:
                            regions = REGIONS
                        fname = f"{sdir}/{'__'.join(regions)}-splotby_{VARNAME_SUBPLOT_LEVELS}-mean{plot_mean}.pdf"
                        print("Saving figure to: ", fname)
                        fig.savefig(fname)
                    plt.close("all")
