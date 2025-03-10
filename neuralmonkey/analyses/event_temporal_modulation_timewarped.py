"""
For figure giving overview of activity across areas, in terms of how related to 
image and motor aspects of trial,
using PA object

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pythonlib.tools.plottools import savefig
import seaborn as sns

def _get_twind_between_these_events(PAtrial, ev1, ev2, pad=0.005):
    ind1 = PAtrial.Params["events_all"].index(ev1)
    ind2 = PAtrial.Params["events_all"].index(ev2)
    t1 = PAtrial.Params["event_times_median"][ind1]
    t2 = PAtrial.Params["event_times_median"][ind2]
    twind = (t1+pad, t2-pad)
    return twind


def load_and_preprocess_PAtrialpop(animal, date):
    """
    PAtrialpop is a single PA, time-warped across events, and holding all bregions
    """
    from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED
    import pickle
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion, dfpa_concatbregion_preprocess_wrapper

    savedir = f"{PATH_DATA_NEURAL_PREPROCESSED}/PA_trialpop/{animal}-{date}"

    with open(f"{savedir}/PA.pkl", "rb") as f:
        PAtrial = pickle.load(f)

    # Hacky, old version had some numerical imprecision <0
    PAtrial.X[PAtrial.X<0] = 0.

    # Split into bregion-specific PA
    list_pa, bregions = PAtrial.split_by_label("chans", "bregion_combined")

    # faking it, overwrite DFalpa
    res = []
    for pa, region in zip(list_pa, bregions):
        print(region, " -- ", pa.X.shape)
        res.append({
            "which_level":"warped",
            # "which_level":"trial",
            "event":"03_samp",
            "event":"none",
            "bregion":region,
            # "twind":(-1.0, 1.8),
            "twind":"none",
            "pa":pa
        })
    DFallpa = pd.DataFrame(res)

    # Preprocess
    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

    preprocess_dfallpa(DFallpa)

    # base_twind = _get_twind_between_these_events("fixtch", "samp")
    # sort_twind = _get_twind_between_these_events("samp", "go_cue")

    # print(base_twind, sort_twind)

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/events_modulation_timewarped/{animal}-{date}-using_timewarped"
    os.makedirs(savedir, exist_ok=True)

    map_event_to_time = {ev:time for ev, time in zip(PAtrial.Params["events_all"], PAtrial.Params["event_times_median"])}

    return PAtrial, DFallpa, SAVEDIR, map_event_to_time


def preprocess_dfallpa(DFallpa):
    """

    """

    # Get baseline pa
    ### First, split each PA into train and test 
    # Will use test for both (i) getting mean adn std for normaliztaion and (ii) sorting neruosn.
    def compute_frac_split(n_trials, n_trials_small=50):
        n_trials_small = 50
        frac = n_trials_small/n_trials
        frac = np.min([np.max([frac, 0.05]), 0.2])
        return frac
    
    list_pa_large = []
    list_pa_small = []
    for pa in DFallpa["pa"].values:
        frac = compute_frac_split(len(pa.Trials))
        try:
            # pa_large, pa_small = pa.split_sample_stratified_by_label(["seqc_0_shape", "seqc_0_loc"], frac, False)
            pa_large, pa_small = pa.split_sample_stratified_by_label(["seqc_0_shape"], frac, False)
        except Exception as err:
            pa_large, pa_small = pa.split_train_test_random(frac)

        list_pa_large.append(pa_large)
        list_pa_small.append(pa_small)
    DFallpa["pa_large"] = list_pa_large
    DFallpa["pa_small"] = list_pa_small

    

def heatmap_bregions_events_wrapper(DFallpa, PAtrial, SAVEDIR):
    """
    Wrapper to make all kinds of heatmaps showing mean activity fore ach neruon (y axis) as function of time,
    split into subplots of (bregion, event).
    """

    savedir = f"{SAVEDIR}/heatmaps"
    os.makedirs(savedir, exist_ok=True)

    # Baseline is during samp
    base_twind = _get_twind_between_these_events(PAtrial, "fixtch", "samp")
    
    for HACK_ABS_VALUE in [False, True]:
        for base_norm_method in ["subtract", "zscore"]:
            for sort_by in ["value", "modulation"]:
                for e1, e2 in zip(PAtrial.Params["events_all"][:-1], PAtrial.Params["events_all"][1:]):
                    sort_twind = _get_twind_between_these_events(PAtrial, e1, e2)

                    fig1, axes1, fig2 = heatmap_bregions_events(DFallpa, "none", base_twind, "none", sort_twind, sort_by, 
                                                                    base_norm_method=base_norm_method, 
                                                                    which_level="warped", xsizemult = 4,
                                                                    HACK_ABS_VALUE=HACK_ABS_VALUE);

                    # overlay times of each event
                    for ax in axes1.flatten():
                        # overlay events
                        for ev, time in zip(PAtrial.Params["events_all"], PAtrial.Params["event_times_median"]):
                            ax.axvline(time, color="k", alpha=0.3)
                            ax.text(time, 0, ev)

                        # overlay windows for sort and base
                        ax.axvline(sort_twind[0], color="m", alpha=0.3)
                        ax.axvline(sort_twind[1], color="m", alpha=0.3)
                        
                        
                    plt.tight_layout()

                    # fig1
                    savefig(fig1, f"{savedir}/sortby={sort_by, e1, e2}-norm={base_norm_method}-hackabs={HACK_ABS_VALUE}.pdf")
                    plt.close("all")

                    savefig(fig2, f"{savedir}/val_distributions-sortby={sort_by, e1, e2}-norm={base_norm_method}-hackabs={HACK_ABS_VALUE}.pdf")
                    plt.close("all")


def heatmap_bregions_events(DFallpa, base_event, base_twind, sort_event, sort_twind, sort_by="value", 
                                   base_norm_method="zscore", mean_over_trials=True, which_level="trial", xsizemult=1,
                                   HACK_ABS_VALUE=False):
    """
    [GOOD] Overal plot of FR modulation in relation to different events.
    Multiple supblots, organized by bregion (rows) vs. events (cols) all with identical
    normalization, and sorting neurons (y-axis) using a specific event and twind, and sorting all events
    the same way.
    PARAMS
    - base_event, for norm = "03_samp"
    - base_twind, window for computing mean std for zscoring, e/g, (-1, -0.2)
    - sort_event, for deciding how to sort neurons, same way across events, e.g, "03_samp"
    - sort_twind, ... e.g,  (0.05, 1)
    - sort_by, str, what metric to use for sorting neruosn, eg. = "value"
    - mean_over_trials = True
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.neuralplots.population import heatmap_stratified_each_neuron_alltrials, heatmap_stratified_trials_grouped_by_neuron
    
    # PARAMS
    list_bregion = DFallpa["bregion"].unique().tolist()
    list_event = sorted(DFallpa["event"].unique().tolist())

    ### Second, normalization -- get mean and std for each bregion
    pa_field = "pa_small" # i.e. use held out for getting z-score and sort inds
    stats_base_dict = {}
    for bregion in list_bregion:
        pa = extract_single_pa(DFallpa, bregion, which_level=which_level, event=base_event, pa_field=pa_field)
        _, xmean, xstd = pa.norm_rel_base_window(base_twind, method=base_norm_method, return_stats=True)    
        stats_base_dict[bregion] = (xmean, xstd)

    # Get sortinds 
    # map_region_to_sortevent = {
    #     "M1":"06_on_strokeidx_0",
    #     "M1":"06_on_strokeidx_0",
    #     "M1":"06_on_strokeidx_0",
    #     "M1":"06_on_strokeidx_0",
    #     "M1":"06_on_strokeidx_0",

    #     'PMv', 'PMd', 'dlPFC', 'vlPFC', 'FP', 'SMA', 'preSMA
    # }
    map_region_to_sortevent = {region:sort_event for region in list_bregion}
    sortinds_dict = {}
    # sort_event = "06_on_strokeidx_0"
    # twind_sort = (2.2, 5)
        # Sort by modulation
    for bregion in list_bregion:
        event = map_region_to_sortevent[bregion]
        pa = extract_single_pa(DFallpa, bregion, which_level=which_level, event=event, pa_field=pa_field)

        if sort_by=="modulation":
            _, sortinds_chan = pa.sort_chans_by_modulation_over_time()
        elif sort_by=="value":
            # get zscore
            xmean, xstd = stats_base_dict[bregion]
            panorm = pa.norm_rel_base_apply(xmean, xstd, method=base_norm_method)
            # panorm = pa._norm_apply_zscore(xmean, xstd)        
            _, sortinds_chan = panorm.sort_chans_by_fr_in_window(sort_twind)
        else:
            print(sort_by)
            assert False

        sortinds_dict[bregion] = sortinds_chan

    ### MAKE PLOTS            
    pa_field = "pa_large"

    ncols = len(list_event)
    nrows = len(list_bregion)
    SIZE = 4
    if base_norm_method=="zscore":
        zlims = [-2, 2]
    elif base_norm_method=="subtract":
        zlims = [-0.6, 0.6]
    else:
        print(base_norm_method)
        assert False

    if HACK_ABS_VALUE:
        zlims = [0, zlims[1]]

    # Scale height of subplots to match n neurons per area.
    event = DFallpa["event"].unique().tolist()[0]
    map_region_to_nchans = {}
    for i, bregion in enumerate(list_bregion):
        pa = extract_single_pa(DFallpa, bregion, which_level=which_level, event=event, pa_field=pa_field)
        map_region_to_nchans[bregion] = len(pa.Chans)

    height_ratios = [map_region_to_nchans[bregion] for bregion in list_bregion]
    width_ratios = [1 for _ in range(ncols)] # all are same time bins.

    # PARAMS
    if mean_over_trials:
        ysizemult = 1.1
        add_hline_separator = False
        n_rand = 1
    else:
        ysizemult = 1.7
        add_hline_separator = True
        n_rand = 5

    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(ncols*SIZE*xsizemult, nrows*SIZE*ysizemult), 
                            sharex=True, gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios},
                            squeeze=False)

    list_x = []
    for i, bregion in enumerate(list_bregion):
        for j, event in enumerate(list_event):
            pa = extract_single_pa(DFallpa, bregion, which_level=which_level, event=event, pa_field=pa_field)
            ax = axes1[i][j]

            if False:
                # This norms each event separately (using its own window)
                twind_base = (-1, -0.1)
                panorm = pa.norm_rel_base_window(twind_base)        
            else:
                # Norm all events against pre-samp
                xmean, xstd = stats_base_dict[bregion]
                # panorm = pa._norm_apply_zscore(xmean, xstd)       
                panorm = pa.norm_rel_base_apply(xmean, xstd, method=base_norm_method) 
            
            if HACK_ABS_VALUE:
                panorm.X = np.abs(panorm.X)

            # Sort 
            sortinds_chan = sortinds_dict[bregion]
            panorm = panorm.slice_by_dim_indices_wrapper("chans", sortinds_chan)

            # For each chan, plots its mean
            chans_trials_times = panorm.X.shape
            if mean_over_trials:
                panorm = panorm.agg_wrapper("trials")
            
            print(bregion, event, " -- ", np.percentile(panorm.X.flatten(), [2.5, 5, 50, 95, 97.5]))

            inds = list(range(len(panorm.Trials)))
            heatmap_stratified_trials_grouped_by_neuron(panorm, inds, ax, n_rand_trials=n_rand, 
                                                        zlims=zlims, add_hline_separator=add_hline_separator)

            ax.set_title(f"{bregion} - {event} - chans_trials_times={chans_trials_times}")
            
            # Collect all fr
            list_x.append(panorm.X)
    plt.tight_layout()

    # Plot distribution of fr values, confirm that zlims are good
    x = np.array([])
    for xthis in list_x:
        x = np.concatenate([x, xthis.flatten()])
    
    fig2, ax = plt.subplots()

    lims = np.percentile(x, [1, 99])
    ax.hist(x, bins=200);
    print("X limits (1, 99)-percentile:", lims)
    for l in lims:
        ax.axvline(l)
    ax.set_title("all fr plotted in heatmaps, and [1 99] percentiles")
    ax.set_xlim([-4, 4])

    return fig1, axes1, fig2


def running_euclidian_compute(DFallpa, PAtrial):
    """
    Compyte euclidian distance at each timpoint vs. a base window.
    """

    from pythonlib.tools.distfunctools import euclidian_unbiased, compute_euclidian_dist_yue_diff
    import numpy as np

    if False:
        # Use this setting if you have DFallpa split by (bregion, event)
        # base_twind = (-0.25, -0.05)
        base_twind = (-0.25, -0.15)
    else:
        # Use this if have time-warped, i.e,. one pa for each bregion
        # warped. hand-inputed
        # - Use the end of samp, but leave a bit of room at end, so that can use this as control period.
        # base_twind = (1.2, 1.7)
        # base_twind = (2.0982590111096684 - 0.25, 2.0982590111096684-0.15)
        base_twind = _get_twind_between_these_events(PAtrial, "fixtch", "samp")
        base_twind = (base_twind[1]-0.4, base_twind[1]-0.2) # Leave the lsat 0.15 sec

    res = []
    for i, row in DFallpa.iterrows():

        bregion = row["bregion"]
        event = row["event"]

        if True:
            # held-out data
            pa_base = row["pa_small"]
            pa = row["pa_large"]
        else:
            pa_base = row["pa"]
            pa = row["pa"]

        print(bregion, event)
        
        # First, get baseline distribution, using subsampled data
        pa_base_scal = pa_base.slice_by_dim_values_wrapper("times", base_twind).agg_wrapper("times")
        x_base = pa_base_scal.X[:, :, 0]

        # Second, for each time bin, get distance to this baseline
        if False:
            dur = 0.05
            slide = 0.01
            pa_binned = pa.agg_by_time_windows_binned(dur, slide)
        else:
            pa_binned = pa.copy()

        times = pa_binned.Times
        ntimes = len(times)
        distances = []
        for i in range(ntimes):
            x = pa_binned.X[:, :, i] # (chans, trials)

            if False:
                # Simply unbiased euclidian
                d = euclidian_unbiased(x_base.T, x.T)
            else:
                # Dist yue diff
                Xthis = np.concatenate([x, x_base], axis=1).T # (ndat, ndims)
                labels_rows = [("dat",) for _ in range(x.shape[1])] + [("base",) for _ in range(x_base.shape[1])]
                label_vars = ("condition",)
                _dfdists = compute_euclidian_dist_yue_diff(Xthis, labels_rows, label_vars)
                d = _dfdists[(_dfdists["condition_1"]=="dat") & (_dfdists["condition_2"]=="base")]["dist_yue_diff"].values[0]
                
            distances.append(d)
        distances = np.array(distances)
        
        # Store it
        res.append({
            "bregion":bregion,
            "event":event,
            "distances":distances,
            "times":times,
        })
    dfdists = pd.DataFrame(res)
    
    return dfdists
    
def running_euclidian_scalar_get_windows(PAtrial, map_event_to_time):
    """
    Get twinds to use for scalar contrasts, for running eucldiina.
    "Contrast" means that each scalar score is difference between two windows,
    contrast[0] and contrast[1].
    """

    # Base twind? Use the lsat 0.15 of baseline.
    base_twind = _get_twind_between_these_events(PAtrial, "fixtch", "samp")
    base_wind_contrast = (base_twind[1]-0.15, base_twind[1]) # Take the lsat 0.15 sec

    stroke_dur = map_event_to_time["off_strokeidx_0"] - map_event_to_time["on_strokeidx_0"]
    reach_dur = map_event_to_time["on_strokeidx_0"] - map_event_to_time["first_raise"]

    # Hard coded windows...
    map_event_to_reltwind = {
        "samp":(0.05, 0.2),
        "go_cue":(-0.3, -0.1),
        "first_raise":(-0.1, reach_dur+0.1),
        "on_strokeidx_0":(0.25, stroke_dur-0.25)
    }

    map_event_to_twind = {ev:(map_event_to_time[ev]+rel_twind[0], map_event_to_time[ev]+rel_twind[1]) for ev, rel_twind in map_event_to_reltwind.items()}
    map_event_to_contrast = {ev:[base_wind_contrast, twind] for ev, twind in map_event_to_twind.items()}

    return map_event_to_contrast

def running_euclidian_plot_timecourse(DFallpa, PAtrial, dfdists, map_event_to_contrast, SAVEDIR):
    # Plot

    list_bregion = DFallpa["bregion"].unique().tolist()
    list_event = sorted(DFallpa["event"].unique().tolist())
    which_level = "warped"
    SIZE = 8
    ncols = len(list_event)
    nrows = len(list_bregion)
    xsizemult = 2
    ysizemult = 1

    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(ncols*SIZE*xsizemult, nrows*SIZE*ysizemult), 
                            sharex=True, sharey=True, squeeze=False)

    for i, bregion in enumerate(list_bregion):
        for j, event in enumerate(list_event):
            # pa = extract_single_pa(DFallpa, bregion, which_level=which_level, event=event, pa_field=pa_field)
            
            tmp = dfdists[(dfdists["bregion"]==bregion) & (dfdists["event"] == event)]
            assert len(tmp)==1
            times = tmp["times"].values[0]
            dists = tmp["distances"].values[0]

            ax = axes1[i][j]

            ax.set_title(f"{bregion} - {event}")

            ax.plot(times, dists)

            # overlay times of each event
            if True:
                # overlay events
                for ev, time in zip(PAtrial.Params["events_all"], PAtrial.Params["event_times_median"]):
                    ax.axvline(time, color="k", alpha=0.3)
                    ax.text(time, 0, ev)

                for ev, contrast in map_event_to_contrast.items():
                    from pythonlib.tools.plottools import plot_patch_rectangle_filled
                    _twind = contrast[1]
                    plot_patch_rectangle_filled(ax, _twind[0], _twind[1])
                    YLIM = ax.get_ylim()
                    ax.text(_twind[0], YLIM[-1], ev)

                    # Put the base
                    base_twind = contrast[0]
                    ax.axvline(base_twind[0], color="m", alpha=0.3, linestyle="--")
                    ax.axvline(base_twind[1], color="m", alpha=0.3, linestyle="--")
                    ax.text(base_twind[0], YLIM[-1], "base")

            ax.axhline(0, color="k", linestyle="--", alpha=0.5)
            ax.axvline(0, color="k", linestyle="--", alpha=0.5)
                
    plt.tight_layout()

    savedir = f"{SAVEDIR}/running_euclidian"
    os.makedirs(savedir, exist_ok=True)
    savefig(fig1, f"{savedir}/timecourse.pdf")


def running_euclidian_compute_scalar(dfdists, map_event_to_contrast, PLOT, SAVEDIR):
    """
    """
    from neuralmonkey.neuralplots.brainschematic import plot_df_from_longform, datamod_reorder_by_bregion

    ##### Get scalar scores using the windows
    dists = np.stack(dfdists["distances"].tolist()) # (nrows, ntimes)
    times = dfdists["times"].values[0]

    list_df = []
    for event, contrast in map_event_to_contrast.items():

        twind = contrast[0]
        vals1 = np.mean(dists[:, (times>twind[0]) & (times<twind[1])], axis=1)

        twind = contrast[1]
        vals2 = np.mean(dists[:, (times>twind[0]) & (times<twind[1])], axis=1)

        _df = dfdists.loc[:, ["bregion", "event"]].copy()
        
        _df["scalar_score"] = vals2 - vals1
        _df["event"] = event
        _df["contrast"] = [contrast for _ in range(len(_df))]

        list_df.append(_df)
    dfscalar = pd.concat(list_df).reset_index(drop=True)
    
    dfscalar = datamod_reorder_by_bregion(dfscalar)

    if PLOT:
        import seaborn as sns

        savedir = f"{SAVEDIR}/running_euclidian"
        os.makedirs(savedir, exist_ok=True)

        fig =sns.catplot(data=dfscalar, x="bregion", y="scalar_score", col="event", col_wrap=6, kind="bar", errorbar=("ci", 68))
        savefig(fig, f"{savedir}/catplot-1.pdf")

        fig = sns.catplot(data=dfscalar, x="event", y="scalar_score", col="bregion", col_wrap=6, kind="point", errorbar=("ci", 68))
        savefig(fig, f"{savedir}/catplot-2.pdf")

        # All on brain schematic, single
        plot_df_from_longform(dfscalar, "scalar_score", "event", savedir, "ALL", subplot_var_values=list(map_event_to_contrast.keys()))

        # Each event a differnet schematic
        for ev in dfscalar["event"].unique():
            dfthis = dfscalar[dfscalar["event"] == ev]
            plot_df_from_longform(dfthis, "scalar_score", "event", savedir, f"ev={ev}", subplot_var_values=list(map_event_to_contrast.keys()))
            # savefig(fig, f"{savedir}/brainschematic-ev={ev}.pdf")

    return dfscalar

def running_anova_compute(DFallpa):
    """
    Each (neuron, time) get anova for temporal modulation, within a window.
    """
    from pythonlib.tools.distfunctools import euclidian_unbiased, compute_euclidian_dist_yue_diff
    import numpy as np

    dur = 0.3
    slide = 0.02

    res = []
    for i, row in DFallpa.iterrows():

        bregion = row["bregion"]
        event = row["event"]
        pa = row["pa"]
        print(bregion, event)

        twindows = pa._agg_by_time_windows_binned_get_windows(dur, slide)    

        for i in range(twindows.shape[0]):
            twind = twindows[i, :]
            pathis = pa.slice_by_dim_values_wrapper("times", twind)

            from neuralmonkey.metrics.scalar import _calc_modulation_by_frsm_event_aligned_time
            nchans = len(pathis.Chans)
            for c in range(nchans):
                frmat = pathis.X[c, :, :] # (trials, times)
                r2 = _calc_modulation_by_frsm_event_aligned_time(frmat)

                res.append({
                    "bregion":bregion,
                    "event":event,
                    "chan":pathis.Chans[c],
                    "r2":r2,
                    "twind":twind,
                })
        
    dfmodu = pd.DataFrame(res)
    dfmodu["time"] = [np.mean(twind) for twind in dfmodu["twind"]]

    return dfmodu

def running_anova_scalar_get_windows(map_event_to_time):
    """
    Get twinds to use for scoring scalar
    """

    # Get twinds to use for scalar contrasts
    stroke_dur = map_event_to_time["off_strokeidx_0"] - map_event_to_time["on_strokeidx_0"]
    reach_dur = map_event_to_time["on_strokeidx_0"] - map_event_to_time["first_raise"]

    # Hard coded
    map_event_to_reltwind = {
        "samp":(-0.05, 0.3),
        "go_cue":(-0.05, 0.2),
        "first_raise":(-0.05, reach_dur-0.1),
        "on_strokeidx_0":(-0.05, 0.35)
    }

    map_event_to_twind = {ev:(map_event_to_time[ev]+rel_twind[0], map_event_to_time[ev]+rel_twind[1]) for ev, rel_twind in map_event_to_reltwind.items()}

    return map_event_to_twind

def running_anova_plot_timecourse(PAtrial, dfmodu, map_event_to_twind, SAVEDIR):
    """
    Plot results, running anove, one subplot per area, showing mean timeocurse of anova (across neurons).
    and overlaying event windows
    """
    import seaborn as sns
    fig = sns.relplot(dfmodu, x="time", y="r2", row="bregion", kind="line", aspect=2.5)

    # overlay events
    for ax in fig.axes.flatten():
        for ev, time in zip(PAtrial.Params["events_all"], PAtrial.Params["event_times_median"]):
            ax.axvline(time, color="k", alpha=0.3)
            ax.text(time, 0, ev)
        for ev, _twind in map_event_to_twind.items():
            from pythonlib.tools.plottools import plot_patch_rectangle_filled
            plot_patch_rectangle_filled(ax, _twind[0], _twind[1])
            YLIM = ax.get_ylim()
            ax.text(_twind[0], YLIM[-1], ev)

    savedir = f"{SAVEDIR}/running_anova"
    os.makedirs(savedir, exist_ok=True)
    savefig(fig, f"{savedir}/timecourse.pdf")

def running_anova_compute_scalar(dfmodu, map_event_to_twind, PLOT, SAVEDIR):
    """
    Compute scalar scores of anova within time windows, and make summary plots
    """
    from pythonlib.tools.pandastools import aggregGeneral
    from neuralmonkey.neuralplots.brainschematic import plot_df_from_longform, datamod_reorder_by_bregion

    # Score scalar
    list_df = []

    for event, twind in map_event_to_twind.items():
        # Ignore contrast[0], since this modulation doesnt need a "baseline" window.
        # twind = contrast[1]

        # Get scalar values in this time winodw
        _df = dfmodu[(dfmodu["time"]>twind[0]) & (dfmodu["time"]<twind[1])].reset_index(drop=True)
        from pythonlib.tools.pandastools import aggregGeneral
        _df_scal = aggregGeneral(_df, ["bregion", "event", "chan"], ["r2"])
        _df_scal["agg_event"] = event
        
        list_df.append(_df_scal)
        
    dfscalar = pd.concat(list_df).reset_index(drop=True)
    dfscalar = datamod_reorder_by_bregion(dfscalar)

    if PLOT:
        savedir = f"{SAVEDIR}/running_anova"
        os.makedirs(savedir, exist_ok=True)

        fig = sns.catplot(data=dfscalar, x="bregion", y="r2", col="agg_event", row="event", alpha=0.2, jitter=True)
        savefig(fig, f"{savedir}/catplot-1.pdf")

        fig = sns.catplot(data=dfscalar, x="bregion", y="r2", col="agg_event", row="event", kind="bar", errorbar=("ci", 68))
        savefig(fig, f"{savedir}/catplot-2.pdf")

        fig = sns.catplot(data=dfscalar, x="agg_event", y="r2", col="bregion", col_wrap=6, kind="bar", errorbar=("ci", 68))
        savefig(fig, f"{savedir}/catplot-3.pdf")

        from neuralmonkey.neuralplots.brainschematic import plot_df_from_longform
        plot_df_from_longform(dfscalar, "r2", "agg_event", savedir, savesuffix="ALL",
                              subplot_var_values=list(map_event_to_twind.keys()))
    
    return dfscalar


if __name__=="__main__":
    """
    Uses time-warped data (each row in PA is a single trial)
    Does all plots, inclding of the raw data, normalized different ways, and summary plots, both line-plots and scalar scores.

    """
    import sys
    from neuralmonkey.analyses.event_temporal_modulation_timewarped import load_and_preprocess_PAtrialpop
    from neuralmonkey.analyses.event_temporal_modulation_timewarped import heatmap_bregions_events_wrapper
    from neuralmonkey.analyses.event_temporal_modulation_timewarped import running_euclidian_compute, running_euclidian_scalar_get_windows, running_euclidian_plot_timecourse
    from neuralmonkey.analyses.event_temporal_modulation_timewarped import running_euclidian_compute_scalar
    from neuralmonkey.analyses.event_temporal_modulation_timewarped import running_anova_compute, running_anova_scalar_get_windows, running_anova_plot_timecourse, running_anova_compute_scalar

    animal = sys.argv[1]
    date = int(sys.argv[2])

    # animal = "Diego"
    # date = 230615
    PAtrial, DFallpa, SAVEDIR, map_event_to_time = load_and_preprocess_PAtrialpop(animal, date)

    # (1) Heatmaps (fr vs time)
    heatmap_bregions_events_wrapper(DFallpa, PAtrial, SAVEDIR)

    # (2) Euclidian distnace fro baseline
    dfdists = running_euclidian_compute(DFallpa, PAtrial)
    map_event_to_contrast = running_euclidian_scalar_get_windows(PAtrial, map_event_to_time)
    running_euclidian_plot_timecourse(DFallpa, PAtrial, dfdists, map_event_to_contrast, SAVEDIR)
    dfscalar = running_euclidian_compute_scalar(dfdists, map_event_to_contrast, PLOT=True, SAVEDIR=SAVEDIR)

    # (3) Modulation using anova method (aligned to events)
    dfmodu = running_anova_compute(DFallpa)
    map_event_to_twind = running_anova_scalar_get_windows(map_event_to_time)
    running_anova_plot_timecourse(PAtrial, dfmodu, map_event_to_twind, SAVEDIR)
    dfscalar = running_anova_compute_scalar(dfmodu, map_event_to_twind, PLOT=True, SAVEDIR=SAVEDIR)
