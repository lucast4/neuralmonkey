""" Methods that take in firing rate over time, and returns metrics related to whether site is good,
e.g., stability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def score_firingrate_drift(frvals, times_frac, trials, ntrials_per_bin = 50, nsigma=3.5, 
                           savedir=None, savename=None):
    """
    Score the across-day drift in FR
    """

    assert frvals.shape[0] == times_frac.shape[0] == trials.shape[0]

    from pythonlib.tools.datetools import standardize_time_helper
    from sklearn.linear_model import LinearRegression
    # import pandas as pde

    def _threshold_fr(frvals):
        """
        """
        mu = np.mean(frvals)
        sig = np.std(frvals)
        thresh_upper = mu + nsigma*sig
        thresh_lower = mu - nsigma*sig
        inds_bad_bool = (frvals>thresh_upper) | ((frvals<thresh_lower))
        inds_bad = [int(x) for x in np.where(inds_bad_bool)[0]]
        return inds_bad, thresh_lower, thresh_upper, inds_bad_bool

    def _plot(_x, _y, ax):
        """
        """
        inds_bad, thresh_lower, thresh_upper, inds_bad_bool= _threshold_fr(_y)
        _x_bad = _x[inds_bad]
        _y_bad = _y[inds_bad]
        ax.plot(_x, _y, 'xk');
        ax.plot(_x_bad, _y_bad, 'or')
        # plot text next to each
        for t, fr in zip(_x_bad, _y_bad):
            ax.text(t, fr, f"{t:.2f}", color="r")
        ax.set_ylim(bottom=0.)
        # ax.set_title(f"fr_spread_index_across_bins={fr_spread_index_across_bins.item():.2f}")

    ### Extract data
    # (1) firing rates across trials (one scalar each trial)
    # frvals = np.array(list_fr)
    frvals_sq = frvals**0.5
    # trials_this = np.array(trials)
    # times_frac = np.array([standardize_time_helper(dt) for dt in dfthis["datetime"].tolist()])

    # (3) Trehshodl the firing rate
    inds_bad, thresh_lower, thresh_upper, inds_bad_bool = _threshold_fr(frvals_sq)
    
    ### Metrics (Score drift)_
    nbins = int(len(times_frac)/ntrials_per_bin)
    if nbins == 1:
        # SKIP THIS, too few trials.
        fr_spread_index_across_bins = None
        slope_over_intercept = None
    else:
        # from pythonlib.tools.nptools import bin_values_by_rank
        # bin_values_by_rank(times_frac, nbins)

        frvals_sq_no_outlier = frvals_sq[~inds_bad_bool]
        times_frac_no_outlier = times_frac[~inds_bad_bool]
        trials_this_no_outlier = trials[~inds_bad_bool]

        # (1) linear, across day
        reg = LinearRegression().fit(times_frac_no_outlier[:, None], frvals_sq_no_outlier[:, None])
        slope = reg.coef_.item()
        frmean = np.mean(frvals_sq_no_outlier)
        # intercept = reg.intercept_.item()
        slope_over_mean = slope/frmean # units = intercepts/day
        slope_over_mean = slope_over_mean/24 # units = intercepts/hour.
        # print(slope, intercept, slope_over_intercept)

        # (2) any block of time with very diff fr from others?
        # Any trial bins with deviation in fr? Get index of (max - min)/(mean) across 50-trial bins.
        dfrate = pd.DataFrame({"fr":frvals_sq_no_outlier, "times_frac":times_frac_no_outlier, "trials":trials_this_no_outlier})
        dfrate["times_frac_bin"] = pd.qcut(dfrate["times_frac"], nbins) # bin it

        fr_max_across_bins = np.max(dfrate.groupby("times_frac_bin").mean()["fr"])
        fr_min_across_bins = np.min(dfrate.groupby("times_frac_bin").mean()["fr"])
        fr_mean_across_bins = np.mean(dfrate.groupby("times_frac_bin").mean()["fr"])
        fr_spread_index_across_bins = (fr_max_across_bins - fr_min_across_bins)/fr_mean_across_bins        

        # (3) Any block with very high variance across trials?
        frstd_max_across_bins = np.max(dfrate.groupby("times_frac_bin").std()["fr"])
        frstd_min_across_bins = np.min(dfrate.groupby("times_frac_bin").std()["fr"])
        frstd_mean_across_bins = np.mean(dfrate.groupby("times_frac_bin").std()["fr"])
        frstd_spread_index_across_bins = (frstd_max_across_bins - frstd_min_across_bins)/frstd_mean_across_bins        

    ### Plots
    if savedir is not None:
        # for each chan and event, find outlier trials
        fig, axes = plt.subplots(5,1, figsize=(10,16))

        ax = axes.flatten()[0]
        ax.hist(frvals, 50, color="k");
        ax.set_xlabel("firing rate histogram")

        ax = axes.flatten()[1]
        ax.hist(frvals_sq, 50, log=True, color="g");
        ax.set_xlabel("firing rate histogram (sqrt)")

        ax = axes.flatten()[2]
        if False: # To make it quicekr. Dont need this
            # print(trials_this)
            # print(frvals)
            _plot(trials, frvals, ax)
            ax.set_ylabel("fr (hz)")
        ax.set_title(f"slope_over_mean={slope_over_mean:.2f}")

        ax = axes.flatten()[3]
        if False: # To make it quicekr. Dont need this
            _plot(trials, frvals_sq, ax)
            ax.set_ylabel("fr (hz**0.5)")
        ax.set_title(f"frstd_spread_index_across_bins={frstd_spread_index_across_bins:.2f}")

        # Plot against time.
        ax = axes.flatten()[4]
        _plot(times_frac, frvals_sq, ax)
        ax.set_ylabel("fr (hz**0.5)")
        ax.set_title(f"fr_spread_index_across_bins={fr_spread_index_across_bins:.2f}")

        # Save
        from pythonlib.tools.pandastools import savefig
        savefig(fig, f"{savedir}/{savename}.pdf")

    metrics = {
        "fr_spread_index_across_bins":fr_spread_index_across_bins, 
        "frstd_spread_index_across_bins":frstd_spread_index_across_bins, 
        "slope_over_mean":slope_over_mean
    }

    return metrics, inds_bad
