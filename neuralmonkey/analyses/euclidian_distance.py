"""
Methods for For computing and plots of eucldian dsitances, all should collect here.
Usually using approach of pairwise distances between trials, then splitting/grouping by different label relations.

Previously:
- state_space_good.py --> For scoring euclidian distance
- analy_euclidian_dist_pop_script.py --> Scripts for running entire pipeline (generic).
--- This is too unweildy, trying to be too flexible across expts.
Expt-specific plots
- psychometric
- char_sp
- shape_invariance.

HERE: collects all the functions that generate from PA.
"""
from pythonlib.tools.pandastools import append_col_with_grp_index
import numpy as np
from pythonlib.tools.plottools import savefig
import matplotlib.pyplot as plt

def timevarying_compute(PA, vars_group):
    """
    Generate dataset of parwise distances between each level of each variable in vars_group.
    Gets value for each time bin independelty, which makes this slow.
    This is the old method.
    PARAMS:
    - vars_group, list of str, variables that define groups.
    """

    version = "traj"
    DFDIST = PA.dataextractwrap_distance_between_groups(vars_group, version)

    # Add column names reflecting the "sameness" state of variables.
    for var in vars_group:
        DFDIST[f"{var}_same"] = DFDIST[f"{var}_1"] == DFDIST[f"{var}_2"]
        DFDIST = append_col_with_grp_index(DFDIST, [f"{var}_1", f"{var}_2"], f"{var}_12")
    
    for i in range(len(vars_group)):
        for j in range(len(vars_group)):
            if j>i:
                var1 = vars_group[0]
                var2 = vars_group[1]
                DFDIST = append_col_with_grp_index(DFDIST, [f"{var1}_same", f"{var2}_same"], f"same-{var1}|{var2}")

    return DFDIST

def timevarying_convert_to_scalar(DFDIST, twind_scalar):
    """
    Genreate a dataframe of scalar vallues suing this timw window,
    given the time-resolved distances in DFDIST.

    PARAMS:
    - DFDIST, holds distances for each time bin, is returned from timevarying_compute()
    """
    from pythonlib.tools.pandastools import aggregGeneral

    dfthis_sub = DFDIST[(DFDIST["time_bin"]>=twind_scalar[0]-0.001) & (DFDIST["time_bin"]<=twind_scalar[1]+0.001)].reset_index(drop=True)

    # Agg, averaging over time
    # dfscal = aggregGeneral(dfthis_sub, ["animal", "date", "combine_areas", "event", "bregion", "metaparams", "same-task|shape", "prune_version", "subspace_projection", "remove_drift", "raw_subtract_mean_each_timepoint", 
    #                                 "remove_singleprims_unstable"], values=["dist_mean", "dist_norm", "dist_yue_diff"])
    dfscal = aggregGeneral(dfthis_sub, ["labels_1", "labels_2"], values=["dist_mean", "dist_norm", "dist_yue_diff"], nonnumercols="all")

    return dfscal


def timevarying_compute_fast_to_scalar(PA, label_vars=("seqc_0_shape", "seqc_0_loc"),
                                       rsa_heatmap_savedir=None):
    """
    [Fast code] -- do all steps to extract dfdist, starting from PA.
    POTENTIAL PROBLOEM (is ok, emeprically): it doesnt get time-varying, it goes straight from (ndims, ntimes) --> scalar,
    for each trial. This is so that it can work with the distmat_construct_wrapper helper
    function, but it is really not necesary, and would be a quick way to get time-varying also,
    if needed --> This also means that the final dist_yue_diff uses the scalars after agging
    across time, instead of computing in each bin, then averaging them. May have slight difference,
    but not too much.
    """
    from pythonlib.tools.distfunctools import distmat_construct_wrapper
    from pythonlib.cluster.clustclass import Clusters


    # (ndims, ntrials, ntimes)
    # --> (ndims, ntimes) X ntrials.
    # trial1 = 0
    # trial2 = 1
    # x1 = pa.X[:, trial1, :] 
    # x2 = pa.X[:, trial2, :] 

    # x1.shape
    # import numpy as np

    # res = []
    # for trial1 in range(ntrials):
    #     print(trial1)
    #     for trial2 in range(ntrials):
    #         x1 = pa.X[:, trial1, :] 
    #         x2 = pa.X[:, trial2, :] 

    #         x1.shape

    #         (np.sum((x1 - x2)**2, axis=0))**0.5

    ###### ALTERANTIVE METHODS:
    if False:
        # (1) SLOWEST: Usual way of computing
        DFDIST = timevarying_compute(PA, vars_group)
        
        # (2) FAST! but not as fast.
        # It works, output is idneticla, but is about 20-40% slower (1.2 sec vs. 0.9 sec).
        DIAGONAL_VALUE = 0
        distmat = np.zeros((len(indsall), len(indsall))) + DIAGONAL_VALUE
        # distmat = distmat - np.inf

        indsall = list(range(pa.X.shape[1]))

        for i in range(1, len(indsall)):
            print(i)
            
            inds1 = indsall[:len(indsall)-i]
            inds2 = indsall[i:]

            xmat1 = pa.X[:, inds1, :]
            xmat2 = pa.X[:, inds2, :]
            xmat2.shape

            xdiff = np.sum((xmat1 - xmat2)**2, axis=0)**0.5 # (ntrials, ntimes)
            # mean over time
            xdiff_scal = np.mean(xdiff, axis=1) # (ntrials, )

            # # store into distance matrix
            # for i1, i2, d in zip(inds1, inds2, xdiff_scal):
            #     distmat[i1, i2] = d
            #     distmat[i2, i1] = d

            # TO compare this with the output of outer function
            fig, ax = plt.subplots()
            ax.imshow(distmat)
            fig, ax = plt.subplots()
            ax.imshow(dmat)

            fig, ax = plt.subplots()
            ax.imshow(dmat - distmat)

            np.min(dmat - distmat)


    # Collect each trial.
    ntrials = PA.X.shape[1]
    list_x = []
    for trial in range(ntrials):
        list_x.append(PA.X[:, trial, :])

    # Get distnace matrix.
    def dist_func(x1, x2):
        """
        x1, x2, (ndims, ntimes), a trial-slice.
        Return scalar distance, averaged over all time.
        """
        return np.mean((np.sum((x1 - x2)**2, axis=0))**0.5)
    dmat = distmat_construct_wrapper(list_x, list_x, dist_func)

    # Convert to Cl
    dflab = PA.Xlabels["trials"]
    list_lab = [tuple(x) for x in dflab.loc[:, label_vars].values.tolist()]
    params = {
        "label_vars":label_vars,
        "version_distance":"euclidian",
        "Clraw":None,
    }
    Cldist = Clusters(dmat, list_lab, list_lab, ver="dist", params=params)

    if rsa_heatmap_savedir is not None:
        # This fn also returns dfdist. The reason I dont keep ti is that here I get both directions. I am not sure
        # if this would do wierd things downstream
        zlims = None
        # zlims = [0, 0.3]
        _, CldistAgg = Cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False, return_as_clustclass=True)
        n = min([len(label_vars), 3])
        from itertools import permutations
        list_sort_order = sorted(permutations(range(n)))
        for sort_order in list_sort_order:

            fig, ax = CldistAgg.rsa_plot_heatmap(sort_order, zlims=zlims)
            savefig(fig, f"{rsa_heatmap_savedir}/rsa_heatmap-sort_order={sort_order}.pdf")

            varthis = label_vars[sort_order[0]]
            _, fig = CldistAgg.rsa_distmat_construct_theoretical(varthis, PLOT=True, sort_order=list_sort_order[0]) # use the same sort order for each var so can compare them
            savefig(fig, f"{rsa_heatmap_savedir}/rsa_heatmap-var={varthis}-sort_order={list_sort_order[0]}-THEOR.pdf")
            plt.close("all")
    
    # convert to cldist.
    dfdist = Cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=True)
    # for i in range(len(label_vars)):
    #     for j in range(len(label_vars)):
    #         if j>i:
    #             var1 = label_vars[0]
    #             var2 = label_vars[1]
    #             dfdist = append_col_with_grp_index(dfdist, [f"{var1}_same", f"{var2}_same"], f"same-{var1}|{var2}")

    return dfdist, Cldist


