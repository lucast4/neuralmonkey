"""
For plotting of state space, focus on DPCA, including splitting subplots by one variable and coloring by the other.
And also scalar plots.

This is good repo of state-space plotting codes.

Notebook: 240128_snippets_demixed_PCA
"""

from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
from neuralmonkey.classes.session import load_mult_session_helper
import os
import pandas as pd
from neuralmonkey.analyses.state_space_good import snippets_extract_popanals_split_bregion_twind
from pythonlib.tools.plottools import savefig
from pythonlib.tools.plottools import rotate_x_labels
from pythonlib.tools.pandastools import applyFunctionToAllRows
from pythonlib.tools.listtools import stringify_list
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def plot_all_results_mult(DFRES, SAVEDIR):
    """ Plot summary across multiple brain areas (i.e., PAs)
    """
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    from pythonlib.tools.plottools import rotate_x_labels

    #########################
    # Plot summaries across cases
    marges = DFRES["marginalizations"].values[0]
    brs = DFRES["bregion"].unique().tolist()

    # Score each bregion
    for marg in marges:
        def F(x):
            """ sum of explained var over all dims"""
            return np.sum(x["explained_var"][marg])
        DFRES = applyFunctionToAllRows(DFRES, F, f"sumvar_{marg}")

    savedir = f"{SAVEDIR}/figures_summary"
    os.makedirs(savedir, exist_ok=True)

    nc = 2
    nr = int(np.ceil(len(marges)/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3), sharex=True, sharey=True)

    for marg, ax in zip(marges, axes.flatten()):
        for br in brs:
            dfthis = DFRES[DFRES["bregion"]==br]
            assert len(dfthis)==1

            exp_var = dfthis["explained_var"].values[0][marg]
            ax.plot(range(len(exp_var)), exp_var, "-o", label=br)
        ax.legend()
        ax.set_title(f"marg: {marg}")
        ax.set_xlabel("dim")
        rotate_x_labels(ax)

    savefig(fig, f"{savedir}/exp_var_bydims.pdf")


    nc = 2
    nr = int(np.ceil(len(marges)/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3), sharex=True, sharey=True)

    for marg, ax in zip(marges, axes.flatten()):
        sns.barplot(data=DFRES, x="bregion", y=f"sumvar_{marg}", ax=ax)
        ax.set_title(f"marg: {marg}")
        rotate_x_labels(ax)

    savefig(fig, f"{savedir}/exp_var_avg_over_dims.pdf")

def plot_all_results_single(dpca, Z, effect_vars, params_dpca, savedir):
    """All results from fitting/testing a single PA
    PARAMS:
    - dpca, resulting model
    - Z, data after transforming with model.
    """

    ############## PLOT RESULTS
    # def plot_results(dpca, Z, savedir):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from neuralmonkey.scripts.analy_dpca_script_quick import plothelper_get_variables

    ts, margs, map_var_to_idx, map_margstr_to_var, map_var_lev_to_pcol, map_var_to_lev = plothelper_get_variables(Z,
                                                                                                                  effect_vars,
                                                                                                                  params_dpca)

    map_grp_to_idx = params_dpca["map_grp_to_idx"]

    fontP = FontProperties()
    ndim = 4
    nmarg = len(margs)
    W = 6
    H = 5

    # Collect all marginalizations and order them and plot
    labels_all = []
    ratios_all = []
    tmp = []
    for lab, ratios in dpca.explained_variance_ratio_.items():
        labels_all.extend(lab)
        ratios_all.extend(ratios)

        tmp.extend([(lab, r, i) for i, r in enumerate(ratios)])

    # Sort
    tmp = sorted(tmp, key = lambda x:-x[1])
    import pandas as pd
    df_exp_var = pd.DataFrame(tmp, columns=["margstr", "exp_var_ratio", "idx_within_marg"])
    df_exp_var["rank"] = list(range(len(df_exp_var)))

    if False:
        fig, ax = plt.subplots()

        x = np.arange(len(df_exp_var))
        y = df_exp_var["exp_var_ratio"]
        color = df_exp_var["margstr"].tolist()

        ax.scatter(x=x, y=y, c=color)


    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10,5))
    sns.pointplot(data=df_exp_var[:15], x="rank", y="exp_var_ratio", hue="margstr", linestyles="none")
    ax.axhline(0, color="k", alpha=0.25)
    # sns.catplot(data=df_exp_var, x="rank", y="exp_var_ratio", hue="margstr")
    savefig(fig, f"{savedir}/explained_var_combined.pdf")

    plt.close("all")

    for indvar_to_color_by in [0,1]:
        # indvar_to_color_by = 0 # color each curve by this variable

        fig, axes = plt.subplots(nmarg, ndim, figsize=(ndim*W, nmarg*H), sharex=True, sharey=True)
        for i, marg in enumerate(margs):
            for j, dim in enumerate(range(ndim)):
                ax = axes[i][j]

                # One curve for each (shape/loc) combo
                for grp, idx in map_grp_to_idx.items():
                    z = Z[marg][dim, idx[0], idx[1], :]

                    # What color
                    # var = map_margstr_to_var[marg]
                    # pcol = map_var_lev_to_pcol[(var, grp[indvar_to_color_by])]

                    # which shape is this?
                    var = effect_vars[indvar_to_color_by]
                    pcol = map_var_lev_to_pcol[(var, grp[indvar_to_color_by])]
                    ax.plot(ts, z, "-", color=pcol, alpha=0.5, label=grp)

                ax.set_title(f"{marg}-d{dim}")
                # legend = ax.legend(loc=0, ncol=1, bbox_to_anchor=(0, 0, 1, 1),
                #        prop = fontP, fancybox=True,shadow=False,title='LEGEND', fontsize=2)2+
                # plt.setp(legend.get_title(),fontsize="xx-small")
                legend = ax.legend(fontsize="5")
                # plt.setp(legend.get_title(),fontsize="xx-small")
        savefig(fig, f"{savedir}/overview_latents-color_by_{indvar_to_color_by}.pdf")
        plt.close("all")

    ### Other plots
    fig, axes = plt.subplots(2,1)
    ax = axes.flatten()[0]

    for lab, ratios in dpca.explained_variance_ratio_.items():
        print(lab, ratios)
        ax.plot(ratios, "-o", label=lab)
    ax.legend()
    savefig(fig, f"{savedir}/explained_var_each_marg.pdf")


    ### State space plots
    from neuralmonkey.population.dimreduction import plotStateSpace

    maxs = []
    mins = []
    for m, z in Z.items():
        if not m=="t": # exclude t, since it is suusalyl very large
            maxs.append(np.max(z))
            mins.append(np.min(z))

    zmax = max(maxs)
    zmin = min(mins)
    print(zmin, zmax)
    # One curve for each (shape/loc) combo


    ####### 2D PLOTS, DIFF PAIRS OF DIMENSIONS
    # dims_pc = [0,1]
    for marg, var in map_margstr_to_var.items():
        if marg in Z.keys():
            # marg = "l"
            # indvar_to_color_by = 0
            for var_color_by in effect_vars:
                for SAME_AXES_ACROSS_MARGS in [False, True]:
                    if SAME_AXES_ACROSS_MARGS:
                        val_minmax = [zmin, zmax]
                    else:
                        val_minmax = None
                    fig = plot_statespace_2d_overlaying_all_othervar(Z, marg, var_color_by, params_dpca, val_minmax)

                    #
                    # indvar_to_color_by = effect_vars.index(var_color_by)
                    #
                    # fig, axes = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
                    # for ax, dims_pc in zip(axes.flatten(), list_dims):
                    #
                    #     for grp, idx in map_grp_to_idx.items(): # Iterate over all curves (conditions)
                    #         # z = Z[marg][dim, idx[0], idx[1], :]
                    #
                    #         # What color
                    #         # var = map_margstr_to_var[marg]
                    #         pcol = map_var_lev_to_pcol[(var_color_by, grp[indvar_to_color_by])]
                    #
                    #         i=idx[0]
                    #         j=idx[1]
                    #         X = Z[marg][:, i, j, :] # (ndims, ntimes)'
                    #
                    #         plotndim = len(dims_pc)
                    #         color_for_trajectory = pcol
                    #         is_traj = True
                    #         traj_mark_times_inds = [1]
                    #         x1, x2 = plotStateSpace(X, dims_neural=dims_pc, plotndim=plotndim, ax=ax,
                    #             color_for_trajectory=color_for_trajectory, is_traj=is_traj, alpha=0.3,
                    #                                 traj_mark_times_inds=traj_mark_times_inds)
                    #     ax.set_xlabel(f"dim {dims_pc[0]}")
                    #     ax.set_ylabel(f"dim {dims_pc[1]}")
                    #
                    #     if SAME_AXES_ACROSS_MARGS:
                    #         ax.set_xlim([zmin, zmax])
                    #         ax.set_ylim([zmin, zmax])

                    savefig(fig, f"{savedir}/2D-marg_{marg}-color_by_{var_color_by}-share_axes_{SAME_AXES_ACROSS_MARGS}.pdf")
                    plt.close("all")

    # See state_space_good.plot_statespace_grpbyvarsothers
    from neuralmonkey.population.dimreduction import plotStateSpace, statespace_plot_single

    ########### 3D PLOTS


    SAME_AXES_ACROSS_MARGS = False
    for marg, var in map_margstr_to_var.items():
        if marg in Z.keys():

            var_color_by = map_margstr_to_var[marg]
                                            # lev_effect = grp[map_var_to_idx[var_effect]]
                                            # pcol = map_var_lev_to_pcol[(var_effect, lev_effect)]
            indvar_to_color_by = map_var_to_idx[var_color_by]

            fig = plt.figure(figsize=(10,10))
            # ax = fig.gca(projection='3d')
            ax = plt.axes(projection="3d")

            for grp, idx in map_grp_to_idx.items(): # Iterate over all curves (conditions)
                # z = Z[marg][dim, idx[0], idx[1], :]

                # What color
                # var = map_margstr_to_var[marg]
                pcol = map_var_lev_to_pcol[(var_color_by, grp[indvar_to_color_by])]

                i=idx[0]
                j=idx[1]
                X = Z[marg][:, i, j, :] # (ndims, ntimes)'

                # plotndim = len(dims_pc)
                # color_for_trajectory = pcol
                # is_traj = True
                # traj_mark_times_inds = [1]
                # x1, x2 = plotStateSpace(X, dims_neural=dims_pc, plotndim=plotndim, ax=ax,
                #     color_for_trajectory=color_for_trajectory, is_traj=is_traj, alpha=0.3,
                #                         traj_mark_times_inds=traj_mark_times_inds)

                # ax = ax .gca(projection='3d')

                ax.plot(X[0,:], X[1,:], X[2,:], "-o", color=pcol, alpha=0.4)
                ax.plot(X[0,-1], X[1,-1], X[2,-1], "-s", mfc="w", color=pcol, alpha=0.4)
                ax.plot(X[0, 0], X[1,0], X[2,0], "-c", mfc="w", color=pcol, alpha=0.4)

                ax.view_init(50, -60)

            ax.set_xlabel(f"dim 0")
            ax.set_ylabel(f"dim 1")
            ax.set_zlabel(f"dim 2")

            if SAME_AXES_ACROSS_MARGS:
                ax.set_xlim([zmin, zmax])
                ax.set_ylim([zmin, zmax])

            savefig(fig, f"{savedir}/3D-marg_{marg}-color_by_{var_color_by}-share_axes_{SAME_AXES_ACROSS_MARGS}.pdf")
            plt.close("all")


    ###### SEPARATE SUBPLOTS BY OTHERVAR
    # NOTE: This should be replaced by trajgood_plot_colorby_splotby().
    for marg, var in map_margstr_to_var.items():
        if marg in Z.keys():
            for var_effect in effect_vars:
                for var_other in effect_vars:
                    if not var_effect == var_other:
                        # var_effect = "seqc_0_shape"
                        # var_other = "seqc_0_loc"
                        # var_effect = "seqc_0_loc"
                        # var_other = "seqc_0_shape"
                        levs_other = map_var_to_lev[var_other]

                        ### PLot params
                        times_to_mark = [0]
                        times_to_mark_markers = ["d"]
                        times = params_dpca["times"]
                        # text_plot_pt1 = "on"
                        text_plot_pt1 = None
                        markersize=7
                        marker="P"
                        time_bin_size = 0.05

                        ncols = 3
                        nrows = int(np.ceil(len(levs_other)/ncols))
                        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3.5, nrows*3.5))

                        dims_pc = [0,1]
                        for ax, levo in zip(axes.flatten(), levs_other):
                            ax.set_title(levo)

                            # plot on this axis all the trajs
                            list_lev = []
                            list_col = []
                            for grp, idx in map_grp_to_idx.items(): # Iterate over all curves (conditions)
                                # z = Z[marg][dim, idx[0], idx[1], :]

                                if grp[map_var_to_idx[var_other]]==levo:

                                    # What color
                                    # var = map_margstr_to_var[marg]
                                    lev_effect = grp[map_var_to_idx[var_effect]]
                                    pcol = map_var_lev_to_pcol[(var_effect, lev_effect)]
                                    list_lev.append(lev_effect)
                                    list_col.append(pcol)

                                    i=idx[0]
                                    j=idx[1]
                                    X = Z[marg][:, i, j, :] # (ndims, ntimes)'

                                    plotndim = len(dims_pc)
                                    color_for_trajectory = pcol
                                    is_traj = True
                                    traj_mark_times_inds = [1]

                                    x = X[dims_pc, :]
                                    statespace_plot_single(x, ax, color_for_trajectory,
                                                           times, times_to_mark, times_to_mark_markers,
                                                           time_bin_size = time_bin_size,
                                                           alpha=0.3,
                                                           markersize=6,
                                                           text_plot_pt1=text_plot_pt1)
                                    # x1, x2 = plotStateSpace(X, dims_neural=dims_pc, plotndim=plotndim, ax=ax,
                                    #     color_for_trajectory=color_for_trajectory, is_traj=is_traj, alpha=0.3,
                                    #                         traj_mark_times_inds=traj_mark_times_inds)
                                # ax.set_xlabel(f"dim {dims_pc[0]}")
                                # ax.set_ylabel(f"dim {dims_pc[1]}")

                            # overlay legend
                            # list_cols = makeColors(len(levels_var))
                            # levels_var = map_var_to_lev[var_effect]
                            # list_cols = map_var_lev_to_pcol[(var_effect, )]
                            from pythonlib.tools.plottools import legend_add_manual
                            legend_add_manual(ax, list_lev, list_col, 0.2)

                        savefig(fig, f"{savedir}/2DOthervarSplot-marg_{marg}-var_{var_effect}-ovar_{var_other}.pdf")
                        plt.close("all")


def plot_statespace_2d_overlaying_all_othervar(Z, marg, var_color_by, params_dpca,
                                               val_minmax=None, return_axes=False):
    """ Quick dirty helper for multiple 2d plots, each overlaying all conditions, coloring by
    <var_color_by>. Each subplot is a different set of pairwise dimensions
    """
    import matplotlib.pyplot as plt
    from neuralmonkey.population.dimreduction import plotStateSpace

    list_dims = [
        [0,1],
        [0,2],
        [0,3],
        [1,2]
    ]

    map_grp_to_idx = params_dpca["map_grp_to_idx"]
    map_var_lev_to_pcol = params_dpca["map_var_lev_to_pcol"]
    effect_vars = params_dpca["effect_vars"]
    indvar_to_color_by = effect_vars.index(var_color_by)

    fig, axes = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
    for ax, dims_pc in zip(axes.flatten(), list_dims):

        for grp, idx in map_grp_to_idx.items(): # Iterate over all curves (conditions)
            # z = Z[marg][dim, idx[0], idx[1], :]

            # What color
            # var = map_margstr_to_var[marg]
            pcol = map_var_lev_to_pcol[(var_color_by, grp[indvar_to_color_by])]

            i=idx[0]
            j=idx[1]
            X = Z[marg][:, i, j, :] # (ndims, ntimes)'

            plotndim = len(dims_pc)
            color_for_trajectory = pcol
            is_traj = True
            traj_mark_times_inds = [1]
            plotStateSpace(X, dims_neural=dims_pc, plotndim=plotndim, ax=ax,
                color_for_trajectory=color_for_trajectory, is_traj=is_traj, alpha=0.3,
                                    traj_mark_times_inds=traj_mark_times_inds)
        ax.set_xlabel(f"dim {dims_pc[0]}")
        ax.set_ylabel(f"dim {dims_pc[1]}")

        if val_minmax is not None:
        # if SAME_AXES_ACROSS_MARGS:
            ax.set_xlim(val_minmax)
            ax.set_ylim(val_minmax)
    if return_axes:
        return fig, axes
    else:
        return fig


def plothelper_get_variables(Z, effect_vars, params_dpca):
    """
    Stupid helper to add important params to params_dpca. Not useful to build on.
    """
    # Some derived variables
    from pythonlib.tools.plottools import makeColors
    import numpy as np

    # ts = np.arange(params_dpca["ntimes"])
    ts = params_dpca["times"]
    margs = list(Z.keys())

    # one plot for each

    # Color and marker based on features
    map_var_to_idx = {var:i for i, var in enumerate(effect_vars)}

    # Just for plotting.
    map_margstr_to_var = {}
    map_margstr_to_var["s"] = "seqc_0_shape"
    map_margstr_to_var["st"] = "seqc_0_shape"
    map_margstr_to_var["t"] = "seqc_0_shape"
    map_margstr_to_var["slt"] = "seqc_0_shape"

    map_margstr_to_var["sl"] = "seqc_0_shape"
    map_margstr_to_var["sz"] = "seqc_0_shape"

    map_margstr_to_var["l"] = "seqc_0_loc"
    map_margstr_to_var["lt"] = "seqc_0_loc"

    map_margstr_to_var["z"] = "gridsize"
    map_margstr_to_var["zt"] = "gridsize"
    map_margstr_to_var["szt"] = "seqc_0_shape"

    map_var_to_lev = params_dpca["map_var_to_lev"]

    map_var_lev_to_pcol = {}
    for i, var in enumerate(effect_vars):
        n = len(map_var_to_lev[var])
        pcols = makeColors(n)
        for lev, pc in zip(map_var_to_lev[var], pcols):
            print(var, lev, pc)
            map_var_lev_to_pcol[(var, lev)] = pc

    # params_dpca["ts"] = ts
    params_dpca["margs"] = margs
    params_dpca["map_var_to_idx"] = map_var_to_idx
    params_dpca["map_margstr_to_var"] = map_margstr_to_var
    params_dpca["map_var_lev_to_pcol"] = map_var_lev_to_pcol
    params_dpca["map_var_to_lev"] = map_var_to_lev
    params_dpca["effect_vars"] = effect_vars

    return ts, margs, map_var_to_idx, map_margstr_to_var, map_var_lev_to_pcol, map_var_to_lev

def preprocess_pa_to_frtensor(PA, effect_vars, keep_all_margs=False):
    """ Given PA, return data processed to be inputs to dPCA.
    Main output is data reshaped to be (ntrials, faet1, feat2, ntimes).
    """
    from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization
    from pythonlib.tools.pandastools import grouping_count_n_samples_quick
    import numpy as np
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

    # assert len(effect_vars)==2, "currently hard coded for this..."

    # Normalize PA as did for RSA stuff.
    # (1) Normalize PA.
    subtract_mean_each_level_of_var = None
    subtract_mean_at_each_timepoint = False
    PAnorm, PAscal, PAscalagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(PA, effect_vars,
                                                                                  subtract_mean_each_level_of_var,
                                                                                  plot_example_chan=None,
                                                                                  plot_example_split_var=None,
                                                                                  DO_AGG_TRIALS=False,
                                                                                  subtract_mean_at_each_timepoint=subtract_mean_at_each_timepoint)
    # (2) Shape into (trials, neurons, feat1, feat2, ...)
    # ie (trials, neurons, shape, loc, times)
    # currently (neurons, trials, times)
    # Iterate thru all feature possibilites, collecting data
    df = PAnorm.Xlabels["trials"]
    nchans = PAnorm.X.shape[0]
    ntimes = PAnorm.X.shape[2]
    times = PA.Times

    # Construct helper mappings.
    map_var_to_lev = {v:sorted(df[v].unique()) for v in effect_vars}

    var_ns = [len(map_var_to_lev[v]) for v in effect_vars]

    # map grp to indices
    if len(effect_vars)==1:
        map_grp_to_idx = {}
        for i, lev1 in enumerate(map_var_to_lev[effect_vars[0]]):
            map_grp_to_idx[(lev1,)]=(i)
        assert len(set(map_grp_to_idx.values())) == len(map_grp_to_idx.values())
    elif len(effect_vars)==2:
        map_grp_to_idx = {}
        for i, lev1 in enumerate(map_var_to_lev[effect_vars[0]]):
            for j, lev2 in enumerate(map_var_to_lev[effect_vars[1]]):
                map_grp_to_idx[(lev1, lev2)]=(i,j)
        assert len(set(map_grp_to_idx.values())) == len(map_grp_to_idx.values())
    else:
        assert False

    # Initialize array
    # - count max n trials across all conjs
    tmin, tmax = grouping_count_n_samples_quick(df, effect_vars)
    trialR = np.empty((tmax, nchans, *var_ns, ntimes))
    trialR[:] = np.nan
    print(trialR.shape)


    grpdict = grouping_append_and_return_inner_items(df, effect_vars)

    # for grp in grpdict:
    #     print(grp, " -- ", map_grp_to_idx[grp])
    # assert False

    # sorted([map_grp_to_idx[grp] for grp in grpdict]) ==

    min_len = 100000
    for grp, inds in grpdict.items():
        print(grp, inds)

        assert len(inds)>0

        if len(effect_vars)==1:
            print( map_grp_to_idx.keys())
            i = map_grp_to_idx[grp]
            x = PAnorm.X[:, inds, :]
            # print(x.shape)

            xthis = np.transpose(x, [1, 0, 2])
            assert ~np.any(np.isnan(xthis))
            assert xthis.shape[0]==len(inds)
            assert xthis.shape[1]==trialR.shape[1]
            assert xthis.shape[2]==trialR.shape[3]

            trialR[:len(inds), :, i, :] = xthis
        elif len(effect_vars)==2:
            i, j = map_grp_to_idx[grp]
            x = PAnorm.X[:, inds, :]
            trialR[:len(inds), :, i, j, :] = np.transpose(x, [1, 0, 2])
        else:
            assert False

        if len(inds)<min_len:
            min_len = len(inds)

    assert min_len>1, "need at least 2 trails... You should prune to remove this level from data before running"

    # Make sure filled up all trials.
    assert not np.all(np.isnan(trialR[-1, ...]))

    if len(effect_vars)==1:
        assert ~np.any(np.isnan(trialR[0, :, 0, 0]))
        assert ~np.any(np.isnan(trialR[0, 0, :, 0]))
        assert ~np.any(np.isnan(trialR[0, 0, 0, :]))
    elif len(effect_vars)==2:
        assert ~np.any(np.isnan(trialR[0, :, 0, 0, 0]))
        assert ~np.any(np.isnan(trialR[0, 0, :, 0, 0]))
        assert ~np.any(np.isnan(trialR[0, 0, 0, :, 0]))
        assert ~np.any(np.isnan(trialR[0, 0, 0, 0, :]))
    else:
        assert False

    # trial-average data
    R = np.nanmean(trialR,0)
    assert np.any(np.isnan(R))==False

    # center data
    if len(effect_vars)==1:
        R -=  np.mean(R.reshape((nchans,-1)),1)[:,None, None] # each neuron gets one mean scalar across all other dimensions
    elif len(effect_vars)==2:
        R -=  np.mean(R.reshape((nchans,-1)),1)[:,None,None, None] # each neuron gets one mean scalar across all other dimensions
    else:
        assert False

    ###### dPCA params
    # labels = "slt" # (shape, location, time)
    labels = ""
    for var in effect_vars:
        if var in ["shape", "seqc_0_shape"]:
            labels+="s"
        elif var=="seqc_0_loc":
            labels+="l"
        elif var=="gridsize":
            labels+="z"
        elif var in ["di_an_ci_binned", "angle_binned", "dist_angle"]:
            labels+="m" # m for motor.
        else:
            print(var)
            assert False
    labels+="t"

    if keep_all_margs:
        join = None
    else:
        if labels in ["st"]:
            join = {
                "s":["s", "st"],
            }
        elif labels in ["slt"]:
            join = {
                "s":["s", "st"],
                "l":["l", "lt"],
                "sl":["sl", "slt"],
            }
        elif labels=="szt":
            join = {
                "s":["s", "st"],
                "z":["z", "zt"],
                "sz":["sz", "szt"],
            }
        elif labels=="mt":
            join = {
                "m":["m", "mt"],
            }
        else:
            print(labels)
            assert False

    n_components = 8

    params_dpca = {
        "labels":labels,
        "join":join,
        "n_components":n_components,
        "ntimes":ntimes,
        "nchans":nchans,
        "times":PA.Times,
        "map_var_to_lev":map_var_to_lev,
        "map_grp_to_idx":map_grp_to_idx
    }

    return R, trialR, map_var_to_lev, map_grp_to_idx, params_dpca, PAnorm



if __name__=="__main__":

    from pythonlib.tools.plottools import savefig
    from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
    import os
    import sys

    SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/dPCA"

    ############### PARAMS
    animal = sys.argv[1]
    date = int(sys.argv[2])
    PLOT = True
    question = sys.argv[3]
    # question = "SP_shape_loc"
    exclude_bad_areas = True
    SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
    bin_by_time_dur = 0.1
    bin_by_time_slide = 0.05
    events_keep = None

    slice_agg_slices = None
    slice_agg_vars_to_split = None
    slice_agg_concat_dim = None

    LIST_TW = [
        [[(-0.3, 0.5)], ["06_on_strokeidx_0"]],
        [[(-0.1, 0.6)], ["03_samp"]],
    ]

    for list_time_windows, events_keep in LIST_TW:

        ################### RUN
        # date = 230615
        # animal = "Diego"
        which_level = "trial"
        combine_into_larger_areas = True
        MS = load_mult_session_helper(date, animal, spikes_version=SPIKES_VERSION)

        SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level = which_level,
            DEBUG=False)

        from neuralmonkey.analyses.rsa import rsagood_questions_dict

        # Load a question
        DictParamsEachQuestion = rsagood_questions_dict(animal, date)
        q_params = DictParamsEachQuestion[question]

        # Clean up SP and extract features
        HACK_RENAME_SHAPES = True
        D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
            ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"],
            HACK_RENAME_SHAPES=HACK_RENAME_SHAPES)


        ### PARAMS for SP --> PA
        # list_time_windows = [(-0.1, 0.6), (-0.5, 0.3), (-0.3, 0.5)]
        # events_keep = None

        if events_keep is None:
            events_keep = q_params["events_keep"]

        # Extract all popanals
        DFallpa = snippets_extract_popanals_split_bregion_twind(SP, list_time_windows,
                                                        list_features_extraction,
                                                                combine_into_larger_areas=combine_into_larger_areas,
                                                                events_keep=events_keep,
                                                                exclude_bad_areas=exclude_bad_areas)

        for pa in DFallpa["pa"].values:
            for feat in list_features_extraction:
                assert feat in pa.Xlabels["trials"].columns

        # Bin times if needed
        if bin_by_time_dur is not None:
            list_pa = []
            for pa in DFallpa["pa"].tolist():
                list_pa.append(pa.agg_by_time_windows_binned(bin_by_time_dur, bin_by_time_slide))
            DFallpa["pa"] = list_pa


        # Aggregate PA if needed
        from neuralmonkey.classes.population_mult import dfpa_slice_specific_windows, dfpa_group_and_split

        if slice_agg_slices is not None:
            # 1) slice
            print(" *** Before dfpa_slice_specific_windows")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print("slice_agg_slices:", slice_agg_slices)
            DFallpa = dfpa_slice_specific_windows(DFallpa, slice_agg_slices)

            # 2) agg (one pa per bregion)
            print(" *** Before dfpa_group_and_split")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print(slice_agg_vars_to_split)
            DFallpa = dfpa_group_and_split(DFallpa, vars_to_split=slice_agg_vars_to_split, concat_dim=slice_agg_concat_dim)

            print(" *** After dfpa_group_and_split")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print("Event, within pa:")

            for pa in DFallpa["pa"].tolist():
                print(pa.Xlabels["trials"]["event"].value_counts())
                print(pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
                assert isinstance(pa.Xlabels["trials"]["wl_ev_tw"].values[0], str)

        # ##### Option 1 - pick out a PA by hand
        #
        # # Take a single pa for now
        #
        # br = "PMv_m"
        # tw = (-0.6, 0.6)
        # ev = "06_on_strokeidx_0"
        # wl = "trial"
        #
        # # tw = (-0.3, -0.1)
        # # ev = "06_on_strokeidx_0"
        # # wl = "trial"
        # # effect_vars = ["seqc_0_shape", "gridsize"]
        #
        # a = DFallpa["bregion"]==br
        # b = DFallpa["twind"]==tw
        # c = DFallpa["event"]==ev
        # d = DFallpa["which_level"]==wl
        #
        # pa =DFallpa[(a & b & c & d)]["pa"].values[0]
        #

        if False:
            effect_vars = ["seqc_0_shape", "seqc_0_loc"]
            q_params["effect_vars"] = effect_vars
        else:
            effect_vars = q_params["effect_vars"]


        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from pythonlib.tools.plottools import makeColors

        for keep_all_margs in [False, True]:
            PLOT = True

            ##### savedir
            from pythonlib.tools.listtools import stringify_list
            a = stringify_list(list_time_windows, return_as_str=True)
            b = stringify_list(events_keep, return_as_str=True)

            SAVEDIR = f"{SAVEDIR_ANALYSES}/{animal}-{date}/{question}/{a}--{b}--{keep_all_margs}"


            RES = []
            for i, row in DFallpa.iterrows():

                pa = row["pa"]
                br = row["bregion"]
                wl = row["which_level"]
                ev = row["event"]
                tw = row["twind"]

                savedir = f"{SAVEDIR}/each_pa/{wl}-{ev}-{br}-{tw}"
                os.makedirs(savedir, exist_ok=True)
                print(" *** Saving to:", savedir)

                # Clean up PA
                from neuralmonkey.analyses.rsa import preprocess_rsa_prepare_popanal_wrapper
                pa, res_check_tasksets, res_check_effectvars = preprocess_rsa_prepare_popanal_wrapper(pa, **q_params)

                ########### SKIPPING ANALYSIS AND EXITING!!
                from pythonlib.tools.expttools import writeDictToTxt, writeDictToYaml
                path = f"{savedir}/res_check_effectvars.txt"
                writeDictToTxt(res_check_effectvars, path)
                path = f"{savedir}/res_check_tasksets.txt"
                writeDictToTxt(res_check_tasksets, path)

                # Preprocess
                R, trialR, map_var_to_lev, map_grp_to_idx, params_dpca, PAnorm = preprocess_pa_to_frtensor(pa, effect_vars, keep_all_margs=keep_all_margs)

                params_dpca["data_shape-trial_N_features_time"] = trialR.shape
                writeDictToTxt(params_dpca, f"{savedir}/params.yaml")

                print("---- OUTPUT")
                print(R.shape)
                print(trialR.shape)
                print(map_var_to_lev)
                print(map_grp_to_idx)
                print(params_dpca)

                #### Fit model
                from dPCA import dPCA

                # We then instantiate a dPCA model where the two parameter axis are labeled by 's' (stimulus) and 't' (time) respectively. We set regularizer to 'auto' to optimize the regularization parameter when we fit the data.
                labels = params_dpca["labels"]
                join = params_dpca["join"]
                n_components = params_dpca["n_components"]
                dpca = dPCA.dPCA(labels=labels, regularizer='auto', join=join, n_components=n_components)
                dpca.protect = ['t']

                # Now fit the data (R) using the model we just instatiated. Note that we only need trial-to-trial data when we want to optimize over the regularization parameter.
                Z = dpca.fit_transform(R, trialR)

                if PLOT:
                    plot_all_results_single(dpca, Z, effect_vars, params_dpca, savedir)

                RES.append({
                    "bregion":br,
                    "which_level":wl,
                    "event":ev,
                    "twind":tw,
                    "explained_var":dpca.explained_variance_ratio_,
                    "marginalizations":list(dpca.marginalizations.keys()),
                    "params_dpca":params_dpca,
                    "map_var_to_lev":map_var_to_lev,
                    "map_grp_to_idx":map_grp_to_idx,
                })

            from neuralmonkey.scripts.analy_dpca_script_quick import plot_all_results_mult
            DFRES = pd.DataFrame(RES)
            plot_all_results_mult(DFRES, SAVEDIR)