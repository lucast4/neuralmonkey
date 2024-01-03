

import numpy as np
from pythonlib.tools.plottools import makeColors
from pythonlib.tools.plottools import legend_add_manual
import matplotlib.pyplot as plt
import pickle
from pythonlib.tools.expttools import load_yaml_config

def _compute_PCA_space(SP, pca_trial_agg_grouping, pca_time_agg_method=None,
    list_event_window=None,
    pca_norm_subtract_condition_invariant = True,
    list_bregion = None,
    list_vars_others=None,
    do_balance=True):
    # pca_trial_agg_grouping = "epoch"
    # pca_trial_agg_grouping = "seqc_0_shape"
    # list_bregion = ["PMv_m"]
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
            PApca, fig, PA, sample_meta = SP._statespace_pca_compute_spaces(sites, event, pre_dur, post_dur, 
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
    PAdata = SP._dataextract_as_popanal_good(df, chans_needed=sites)    

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
    PAdata = SP._dataextract_as_popanal_good(dfall, chans_needed=sites)    

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
