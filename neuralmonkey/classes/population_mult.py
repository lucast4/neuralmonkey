"""
FOr loading and splitting/concatting previously extract PA datasets.
NOWADAYS not used much, since I am not sainvg PA, but isntad goings taight from SP --> Analyses... (saving dsiak sapc).
"""


from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import glob
import pickle
import pandas as pd
import numpy as np
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.tools.plottools import savefig
import os
import  matplotlib.pyplot as plt

# (animal, date, question) --> DFallPA

def load_handsaved_wrapper(animal=None, date=None, version=None, combine_areas=True, 
                           return_none_if_no_exist=False, use_time = True, question=None):
    """ Load a pre-saved DfallPA -- not systematic, just hand saved versions.
    """ 

    assert question is not None, "to not run into error of loading old pa"

    if animal is not None:
        # Load using params input
        norm = None
        if use_time:
            if version == "saccade_fix_on":
                t1 = -0.4
                t2 = 0.4
            else:
                t1 = -1.0
                t2 = 1.8
            if question is None:
                path = f"/lemur2/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{version}-kilosort_if_exists-norm={norm}-combine={combine_areas}-t1={t1}-t2={t2}.pkl"
            else:
                path = f"/lemur2/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{version}-kilosort_if_exists-norm={norm}-combine={combine_areas}-t1={t1}-t2={t2}-quest={question}.pkl"
                if not os.path.exists(path):
                    # Older, without "question" label. You should be the one to decide if this is acceptable.
                    path = f"/lemur2/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{version}-kilosort_if_exists-norm={norm}-combine={combine_areas}-t1={t1}-t2={t2}.pkl"
        else:
            path = f"/lemur2/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-{animal}-{date}-{version}-kilosort_if_exists-norm={norm}-combine={combine_areas}.pkl"
        
        # if animal == "Diego" and date == 230615 and version == "trial":
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230615-trial-kilosort_if_exists-norm=None-combine=True.pkl" # SP, shape vs. loc, all events, good.
        # elif animal == "Pancho" and date == 220715 and version == "trial":
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220715-trial-kilosort_if_exists-norm=None-combine=True.pkl" # SP, has all events.
        # elif animal == "Diego" and date == 230630 and version == "trial":
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230630-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 230630 and version == "stroke":
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230630-stroke-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 240612 and version == "trial":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240612-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 240612 and version == "stroke":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240612-stroke-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 240614 and version == "trial":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240614-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 240614 and version == "stroke":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240614-stroke-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 240619 and version == "trial":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240619-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Diego" and date == 240619 and version == "stroke":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240619-stroke-kilosort_if_exists-norm=None-combine=True.pkl"
        # elif animal == "Pancho" and date == 230623 and version == "trial":
        #     # PROBLEM - is not rule extraction! Just PIG
        #     path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-230623-trial-kilosort_if_exists-norm=None-combine=True.pkl"
    else:
        # Load by manully modifying code.

        # NEWER (After adding epoch variables)
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230823_RULESW_BASE_stroke.pkl"
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230928_RULE_BASE_stroke.pkl"
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230910_RULESW_BASE_stroke.pkl" # shape vs. color
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230911_RULE_BASE_stroke.pkl" #
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230817_RULE_BASE_stroke.pkl" # AnBmCk
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230929_RULE_BASE_stroke.pkl"
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Pancho_230320_RULE_BASE_stroke.pkl" # rowcol
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230922_RULESW_BASE_stroke.pkl" # AnBm vs. DIR vs. seqsup
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_231024_RULESW_BASE_stroke.pkl" #
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Pancho_220830_RULESW_BASE_stroke.pkl" # # (AB)n

        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Diego_230925_RULESW_BASE_stroke.pkl" # seqsup
        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa_Pancho_230811_RULESW_BASE_stroke.pkl" # AnBmCk

        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-231211-stroke-ks_nonorm.pkl" # char [DAN AND XUAN]
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-231211-stroke-tdt-norm=None-combine=True.pkl" # char [DAN AND XUAN]

        # path = f"/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Dolnik/DFallpa_KS_nonorm.pkl" # single prim, shapes (Diego,230615, trial)
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230615-stroke-ks_nonorm.pkl" # shapes (Diego,230615, strokes)
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230630-stroke-ks_nonorm.pkl" # PIG (strokes)
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220918-stroke-ks_nonorm.pkl" # Pancho SP (shape, loc, size)

        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220715-stroke-ks_nonorm.pkl" # PAncho, SP, strokes
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220715-stroke-tdt_nonorm.pkl" # Pancho, SP, tdt, more data.
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220715-trial-tdt_nonorm.pkl" # Pancho, SP (shapes vs loc, TRIAL)
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230618-trial-tdt_nonorm.pkl" # Diego, SP (many shapes) (Trial)
        
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230615-trial-kilosort_if_exists-norm=across_time_bins-combine=True.pkl"
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230615-trial-kilosort_if_exists-norm=None-combine=True.pkl" # SP, shape vs. loc, all events, good.
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220715-trial-kilosort_if_exists-norm=None-combine=True.pkl" # SP, has all events.
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Pancho-220608-trial-tdt-norm=None-combine=True.pkl" # SP, all events, good
        path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240516-trial-kilosort_if_exists-norm=None-combine=True.pkl" # Novel prims.

        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240604-trial-kilosort_if_exists-norm=None-combine=True-RAW.pkl"

        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230630-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230630-stroke-kilosort_if_exists-norm=None-combine=True.pkl"

        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-231211-trial-kilosort_if_exists-norm=None-combine=True.pkl" # CHAR
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-231211-stroke-kilosort_if_exists-norm=None-combine=True.pkl" 
    
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230616-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-230615-trial-kilosort_if_exists-norm=None-combine=True.pkl"
        
        # path = /home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/Xuan/DFallpa-Diego-240523-trial-kilosort_if_exists-norm=None-combine=True-t1=-1.0-t2=1.8.pkl # psycho prims (structured)

    # else:
    #     print(animal, date, version)
    #     assert False

    if not os.path.exists(path) and return_none_if_no_exist:
        return None
    else:
        # print(path)
        # assert False
        DFallpa = pd.read_pickle(path)
        return DFallpa

def dfallpa_preprocess_sitesdirty_single(PA, animal, date, plot_fr_after_replace_trials_dir=None):
    """
    Clean up PA using multiple methods, based on firing rate stats, using metrics and decisions already
    computed.
    
    REmoves channels that are unstable FR over day.
    For each chan, removes trials that are FR outliers, by replacing the trial with the mean over all other trials.

    RETURNS:
    - PA, a copy of PA, which has been cleaned.
    """
    import pandas as pd
    import pickle
    from pythonlib.tools.expttools import load_yaml_config
    from pythonlib.tools.pandastools import savefig
    from pythonlib.tools.plottools import rotate_x_labels

    # First, check that prepropcess data actually exist
    LOADDIR = f"/lemur2/lucas/neural_preprocess/sitesdirtygood_preprocess/{animal}-{date}-combsess"
    if not os.path.exists(LOADDIR):
        return None, None

    ### Load information about clean sites.
    if False:
        # Old, separate fore ach session
        path = f"/lemur2/lucas/neural_preprocess/sitesdirtygood_preprocess/{animal}-{date}-0/dfres.pkl"
        dfres = pd.read_pickle(path)

        path = f"/lemur2/lucas/neural_preprocess/sitesdirtygood_preprocess/{animal}-{date}-0/params.yaml"
        params = load_yaml_config(path)

        TRIALCODES = params["trialcodes"]
    else:
        # New, combining across sesions.
        path = f"{LOADDIR}/dfres.pkl"
        dfres = pd.read_pickle(path)

        path = f"{LOADDIR}/params_text.yaml"
        params = load_yaml_config(path)

        path = f"{LOADDIR}/sessions.pkl"
        with open(path, "rb") as f:
            SESSIONS = pickle.load(f)

        path = f"{LOADDIR}/trialcodes.pkl"
        with open(path, "rb") as f:
            TRIALCODES = pickle.load(f)

    PA = PA.copy()

    ### Sanity checks (clean site info matches PA)
    
    tmp = dfres["chan"].tolist() 
    if not all([ch in tmp for ch in PA.Chans]): # check that all chans in PA exist in dfres
        print("Chans in sitesdirty analysis: ", tmp)
        print("Chans in the current PA: ", PA.Chans)
        for ch in PA.Chans:
            print(ch, ch in tmp)
        assert False, "probably DFallpa is old. need to reextract"

    ### (1) Remove bad chans
    chans_bad_all = dfres[~dfres["good_chan"]]["chan"].tolist()
    chans_bad_this_pa = [ch for ch in PA.Chans if ch in chans_bad_all]
    chans_good_this_pa = [ch for ch in PA.Chans if ch not in chans_bad_all]

    if len(chans_bad_this_pa)>0:
        print("Removing these bad chans:", chans_bad_this_pa)
        PA = PA.slice_by_dim_values_wrapper("chans", chans_good_this_pa)

    # Plot exampel chans
    # PA.plotwrapper_smoothed_fr_split_by_label_and_subplots(37, "seqc_0_shape", "seqc_0_loc")

    ### (2) For each chan, remove the bad trials. Do so by replacing them with the average
    # Get list of bad trialcodes for this chan
    # chan = PA.Chans[11]
    # chan = 354
    dflab = PA.Xlabels["trials"]
    # plot_fr_after_replace_trials_dir = "/tmp"
    
    # Track what was changed
    map_chan_to_trialcodes_replaced = {}
    for chan in PA.Chans:
        map_chan_to_trialcodes_replaced[chan] = []
        dfthis = dfres[dfres["chan"] == chan]
        assert len(dfthis)==1
        _inds_bad = dfthis["inds_bad"].values[0] # indices into trials

        if len(_inds_bad)>0:
            trialcodes_bad = [TRIALCODES[i] for i in _inds_bad]

            # get the mean activity for this chan by taking its flanking n trials that are not bad trialcodes
            # ACTUALLY - get the mean activity for this chan by taking its good trials
            ind_chan = PA.Chans.index(chan)
            trialcodes_all = dflab["trialcode"].tolist()
            trialcodes_good = [tc for tc in trialcodes_all if tc not in trialcodes_bad]
            inds_trials_good = dflab[dflab["trialcode"].isin(trialcodes_good)].index.tolist()
            inds_trials_bad = dflab[dflab["trialcode"].isin(trialcodes_bad)].index.tolist()
            
            if len(inds_trials_bad)>0: # Then bad trials exist
                print("chan", chan, "Replacing these trials with mean good trial:", inds_trials_bad)

                if plot_fr_after_replace_trials_dir is not None:                
                    frvals_old = np.mean(PA.X[ind_chan, :, :], axis=1) # for plotting
                
                xmean_good = np.mean(PA.X[ind_chan, inds_trials_good, :], axis=0) # (ntimes,)

                # For each bad trial, replace it with xmean_good
                map_chan_to_trialcodes_replaced[chan].append(dflab.iloc[inds_trials_bad]["trialcode"].tolist())
                for ind_trial in inds_trials_bad:
                    PA.X[ind_chan, ind_trial, :] = xmean_good

                if plot_fr_after_replace_trials_dir is not None:
                    # Sanity check, compare before and after 
                    frvals_new = np.mean(PA.X[ind_chan, :, :], axis=1)
                    fig, ax = plt.subplots(figsize=(40, 5))

                    ax.plot(trialcodes_all, frvals_new, "or", alpha=0.8)
                    ax.plot(trialcodes_all, frvals_old, "sk", alpha=0.8)
                    rotate_x_labels(ax, 90)
                    savefig(fig, f"{plot_fr_after_replace_trials_dir}/fr_after_replace_trial-chan={chan}.pdf")
                    plt.close("all")    
    return PA, map_chan_to_trialcodes_replaced

def dfallpa_preprocess_fr_normalization(DFallpa, fr_normalization_method, savedir=None):
    """
    Apply normalization to firing rates, modifying the inputted DFallpa.
    :param DFallpa:
    :param fr_normalization_method:
    :param savedir:
    :return: Modifies DFallpa
    """
    from pythonlib.tools.plottools import savefig
    from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization

    if savedir is not None:
        path_to_save_example_fr_normalization = f"{savedir}/example_fr_normalization.png"
    else:
        path_to_save_example_fr_normalization = None

    # What low-level params?
    if fr_normalization_method=="each_time_bin":
        # Then demean in each time bin indepednently
        subtract_mean_at_each_timepoint = True
        subtract_mean_across_time_and_trial = False
    elif fr_normalization_method=="across_time_bins":
        # ALl time bins subtract the same scalar --> maintains temporal moudlation.
        subtract_mean_at_each_timepoint = False
        subtract_mean_across_time_and_trial = True
    else:
        print(fr_normalization_method)
        assert False

    # Iterate thru each pa, doing normalizatoin.
    list_panorm = []
    for i, pa in enumerate(DFallpa["pa"].tolist()):

        # Params for saving figure
        if path_to_save_example_fr_normalization is not None and i==0:
            plot_example_chan_number = pa.Chans[0]
            plot_example_split_var_string = None
            for var in ["seqc_0_shape", "shape", "seqc_0_loc", "gridloc", "task_kind", "epoch"]:
                if var in pa.Xlabels["trials"].columns:
                    plot_example_split_var_string = var
                    break
            assert plot_example_split_var_string is not None
        else:
            plot_example_chan_number = None
            plot_example_split_var_string = None

        # Do normalization
        PAnorm, _, _, fig, axes, _ = popanal_preprocess_scalar_normalization(pa, None,
                                                                                          DO_AGG_TRIALS=False,
                                                                                          plot_example_chan_number=plot_example_chan_number,
                                                                                            plot_example_split_var_string = plot_example_split_var_string,
                                                                                          subtract_mean_at_each_timepoint=subtract_mean_at_each_timepoint,
                                                                                          subtract_mean_across_time_and_trial=subtract_mean_across_time_and_trial)
        if path_to_save_example_fr_normalization is not None and i==0:
            savefig(fig, path_to_save_example_fr_normalization)
        list_panorm.append(PAnorm)

    # Replace all pa
    DFallpa["pa"] = list_panorm


def snippets_extract_popanals_split_bregion_twind(SP, list_time_windows, vars_extract_from_dfscalar,
                                                  SAVEDIR=None, dosave=False,
                                                  combine_into_larger_areas=False,
                                                  events_keep=None,
                                                  exclude_bad_areas=False):
    """ [GOOD] SP --> Multiple Popanals, each with speciifc (event, bregion, twind), and
    with all variables extracted into each pa.Xlabels["trials"]. The goal is that at can
    run all population analyses using these pa, without need for having beh datasets and
    all snippets in memory.
    Extraction of specific PopAnals for each conjunction of (twind, bregion).
    PARAMS:
    - list_time_windowsm, list of timw eindow, tuples .e.g, (-0.2, 0.2), each defining a specific
    extracvted PA.
    - EFFECT_VARS, list of str, vars to extract, mainly to make sure the etracted PA have all
    variables. If not SKIP_ANALY_PLOTTING, then these also determine which plots.
    - dosave, bool, def faulse since takes lots sapce, like 1-3g per wl.
    RETURNS:
    - DictBregionTwindPA, dict, mapping (bregion, twind) --> pa.
    All PAs guaradteeed to have iodentical (:, trials, times).
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index

    # SInce this is population, make sure all channels are present (no outliers removed)
    SP.datamod_append_outliers()

    if events_keep is None or len(events_keep)==0:
        events_keep = SP.Params["list_events_uniqnames"]

    if SAVEDIR is None and dosave:
        from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
        SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS"
        os.makedirs(SAVEDIR, exist_ok=True)

    ####################### EXTRACT DATA
    # list_features_extraction = list(set(list_features_extraction + EFFECT_VARS))
    list_bregion = SP.bregion_list(combine_into_larger_areas=combine_into_larger_areas)

    if not any([e in SP.DfScalar["event"].unique().tolist() for e in events_keep]):
        events_keep = sorted(SP.DfScalar["event"].unique().tolist())

    # 1) Extract population dataras
    DictEvBrTw_to_PA = {}
    print("These events:", events_keep)
    for event in events_keep:
        if event in SP.DfScalar["event"].tolist():
            print(event)
            # assert len(SP.Params["list_events_uniqnames"])==1, "assuming is strokes, just a single event... otherwise iterate"
            # event = SP.Params["list_events_uniqnames"][0]
            PA, _ = SP.dataextract_as_popanal_statespace(SP.Sites, event,
                                                         list_features_extraction=vars_extract_from_dfscalar,
                                                      which_fr_sm = "fr_sm", max_frac_trials_lose=0.02)

            assert len(PA.X)>0
            # print("These are requested sites:", SP.Sites)
            # print("These are extracted sites:", PA.Chans)

            # Split PA based on chans (e.g., bregions), times (e.g., different time slices) BEFORE doing downstream analyses
            DictBregionTwindPA = {}
            trials = None
            xlabels_times = None
            xlabels_trials = None
            for twind in list_time_windows:
                times = None
                for bregion in list_bregion:

                    print(event, bregion, twind)

                    # Bregion
                    chans_needed = SP.sitegetter_map_region_to_sites(bregion, exclude_bad_areas=exclude_bad_areas)
                    print("Sites for this bregion ", bregion)
                    print(chans_needed)
                    if len(chans_needed)>0:
                        pa = PA.slice_by_dim_values_wrapper("chans", chans_needed)
                        # Times
                        pa = pa.slice_by_dim_values_wrapper("times", twind)

                        assert len(pa.X)>0

                        # sanity check that all pa are identical
                        if trials is not None:
                            assert pa.Trials == trials
                        if times is not None:
                            # print(list(pa.Times))
                            # print(list(times))
                            assert list(pa.Times) == list(times)
                        if xlabels_trials is not None:
                            assert pa.Xlabels["trials"].equals(xlabels_trials)
                        if xlabels_times is not None:
                            assert pa.Xlabels["times"].equals(xlabels_times)

                        # # uiseful - a conjucntionv ariable for each tw
                        # from pythonlib.tools.pandastools import append_col_with_grp_index
                        # pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"],
                        #                                                 ["which_level", "event", "twind"],
                        #                                                 "wl_ev_tw",
                        #                                                 use_strings=False)
                        #
                        # Update all
                        trials = pa.Trials
                        times = pa.Times
                        xlabels_trials = pa.Xlabels["trials"]
                        xlabels_times = pa.Xlabels["times"]

                        # DictBregionTwindPA[(bregion, twind)] = pa
                        DictEvBrTw_to_PA[(SP.Params["which_level"], event, bregion, twind)] = pa
                        print(event, " -- ", bregion, " -- ", twind, " -- (data shape:)", pa.X.shape)
                    else:
                        print("Skipping bregion (0 channels): ", bregion)

    assert len(DictEvBrTw_to_PA)>0

    # Save it as dataframe
    tmp = []
    for k, v in DictEvBrTw_to_PA.items():

        # Make sure pa itself is keeping track of the outer varibles,
        # for sanity checks once you start splitting and grouping.
        v.Xlabels["trials"]["which_level"] = k[0]
        v.Xlabels["trials"]["event"] = k[1]
        v.Xlabels["trials"]["bregion"] = k[2]
        v.Xlabels["trials"]["twind"] = [k[3] for _ in range(len(v.Xlabels["trials"]))]

        tmp.append({
            "which_level":k[0],
            "event":k[1],
            "bregion":k[2],
            "twind":k[3],
            "pa":v
        })
    DFallpa = pd.DataFrame(tmp)

    if len(DFallpa)==0:
        print(list_time_windows, vars_extract_from_dfscalar,
              combine_into_larger_areas, events_keep, exclude_bad_areas)
        assert False, "probably params not compatible with each other"

    # # Sanity check
    # for i, row in DFallpa.iterrows():
    #     a = row["twind"]
    #     b = row["pa"].Xlabels["trials"]["twind"].values[0]
    #
    #     if not a==b:
    #         print(a, b)
    #         assert False, "this is old versio before 1/28 -- delete it and regenerate DFallpa"

    # Also note down size of PA, in a column
    list_shape =[]
    for i, row in DFallpa.iterrows():
        list_shape.append(row["pa"].X.shape)
    DFallpa["pa_x_shape"] = list_shape

    ## SAVE
    if dosave:
        import pickle
        mult_sing, sessions = SP.check_if_single_or_mult_session()
        sessions_str = "_".join([str(s) for s in sessions])
        if mult_sing == "mult":
            SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/mult_session"
            path = f"{SAVEDIR}/{SP.animal()}-{SP.date()}-{SP.Params['which_level']}-{sessions_str}.pkl"
        elif mult_sing=="sing":
            SAVEDIR = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/single_session"
            path = f"{SAVEDIR}/{SP.animal()}-{SP.date()}-{SP.Params['which_level']}-{sessions_str}.pkl"
        with open(path, "wb") as f:
            pickle.dump(DFallpa, f)
        print("Saved to: ", path)

    return DFallpa

def dfpa_update_pa_x_shape(DFallpa):
    """ simply update column pa_x_shape"""
    list_shape =[]
    for i, row in DFallpa.iterrows():
        list_shape.append(row["pa"].X.shape)
    DFallpa["pa_x_shape"] = list_shape

def dfallpa_preprocess_vars_conjunctions_extract(DFallpa, which_level):
    """
    Holding methods for:
    Modify all PA, extracting new columns that are usually used. A bit hacky, as they are
    hard-coded things which might be excessive.
    :param DFallpa:
    :param which_level:
    :return:
    - None, but modifies DFallpa
    """
    from pythonlib.tools.nptools import bin_values_by_rank, bin_values
    from pythonlib.tools.expttools import deconstruct_filename
    from pythonlib.tools.vectools import bin_angle_by_direction

    if which_level=="trial":
        pass # is all done in Dataset alrady

        # Extract information about the first stroke (semantic labels)
        # locations_allpa =[]
        # angles_allpa =[]
        # list_i_j = []
        # for i, pa in enumerate(DFallpa["pa"]):
        #     dflab = pa.Xlabels["trials"]
        #
        #     list_seqc_0_shsem = []
        #     list_seqc_0_locon = []
        #     list_seqc_0_shsemcat = []
        #     for j, row in dflab.iterrows():
        #
        #         # To get semantic label for shape, use task token
        #         assert False, "this is incorrect for char -- should use Tkbeh_stkbeh -- do this in Dataset..."
        #         Tk = row["Tkbeh_stktask"]
        #         Tk.features_extract_wrapper(features_get=["shape_semantic"])
        #         list_seqc_0_shsem.append(Tk.Tokens[0]["shape_semantic"])
        #
        #         # Expand shape
        #         # sh-x-y
        #         tmp = deconstruct_filename(Tk.Tokens[0]["shape_semantic"])
        #         shape_semantic_cat = tmp["filename_components_hyphened"][0] # e..g, ['test', '1', '2']
        #         list_seqc_0_shsemcat.append(shape_semantic_cat)
        #
        #         # To get motoro stuff
        #         Tk = row["Tkbeh_stkbeh"]
        #         Tk.features_extract_wrapper(features_get=["loc_on", "angle"])
        #         list_seqc_0_locon.append(Tk.Tokens[0]["loc_on"])
        #
        #         # Collect acorss pa
        #         locations_allpa.append(Tk.Tokens[0]["loc_on"])
        #         angles_allpa.append(Tk.Tokens[0]["angle"])
        #         list_i_j.append((i,j))
        #
        #     dflab["seqc_0_shapesem"] = list_seqc_0_shsem
        #     dflab["seqc_0_locon"] = list_seqc_0_locon
        #     dflab["seqc_0_locx"] = np.array(list_seqc_0_locon)[:,0]
        #     dflab["seqc_0_locy"] = np.array(list_seqc_0_locon)[:,1]
        #     dflab["seqc_0_shapesemcat"] = list_seqc_0_shsemcat
        #
        #
        # # BIN LOCATIONS
        # # Given list of locations, bin them in x and y
        # tmp = np.stack(locations_allpa)
        # xs = tmp[:,0]
        # ys = tmp[:,1]
        #
        # # xs_binned = bin_values_by_rank(xs, nbins=2)
        # # ys_binned = bin_values_by_rank(ys, nbins=2)
        # xs_binned = bin_values(xs, nbins=2)
        # ys_binned = bin_values(ys, nbins=2)
        # locations_allpa_binned = np.stack([xs_binned, ys_binned], axis=1)
        # # map from (i,j) to locations_binned
        # map_ij_to_locbinned ={}
        # assert len(list_i_j)==len(locations_allpa_binned)
        # for (i,j), loc_binned in zip(list_i_j, locations_allpa_binned):
        #     map_ij_to_locbinned[(i,j)] = loc_binned
        # # Place back into pa
        # colname = "seqc_0_loconbinned"
        # for i, pa in enumerate(DFallpa["pa"]):
        #     dflab = pa.Xlabels["trials"]
        #     vals = []
        #     for j, row in dflab.iterrows():
        #         vals.append(tuple(map_ij_to_locbinned[(i,j)].tolist()))
        #     dflab[colname] = vals
        #
        # # BIN ANGLES
        # angles_binned = bin_angle_by_direction(angles_allpa, num_angle_bins=4)
        #
        # map_ij_to_anglebinned ={}
        # map_ij_to_angle ={}
        # assert len(list_i_j)==len(angles_binned)
        # for (i,j), ab, a in zip(list_i_j, angles_binned, angles_allpa):
        #     map_ij_to_anglebinned[(i,j)] = ab
        #     map_ij_to_angle[(i,j)] = a
        #
        # # Place back into pa
        # colname = "seqc_0_anglebinned"
        # for i, pa in enumerate(DFallpa["pa"]):
        #     dflab = pa.Xlabels["trials"]
        #     vals = []
        #     for j, row in dflab.iterrows():
        #         vals.append(map_ij_to_anglebinned[(i,j)])
        #     dflab[colname] = vals
        #
        # colname = "seqc_0_angle"
        # for i, pa in enumerate(DFallpa["pa"]):
        #     dflab = pa.Xlabels["trials"]
        #     vals = []
        #     for j, row in dflab.iterrows():
        #         vals.append(map_ij_to_angle[(i,j)])
        #     dflab[colname] = vals
        #
        # ##### CONJUNCTIONS
        for i, pa in enumerate(DFallpa["pa"]):
            dflab = pa.Xlabels["trials"]
            dflab = append_col_with_grp_index(dflab, ["seqc_0_shape", "seqc_0_loc"], "seqc_0_shapeloc")
            dflab = append_col_with_grp_index(dflab, ["seqc_1_shape", "seqc_1_loc"], "seqc_1_shapeloc")
            dflab = append_col_with_grp_index(dflab, ["seqc_2_shape", "seqc_2_loc"], "seqc_2_shapeloc")
            pa.Xlabels["trials"] = dflab

    elif which_level in ["stroke", "stroke_off"]:

        ##### CONJUNCTIONS
        for i, pa in enumerate(DFallpa["pa"]):
            dflab = pa.Xlabels["trials"]

            dflab = append_col_with_grp_index(dflab, ["CTXT_shape_prev", "CTXT_loc_prev"], "CTXT_shapeloc_prev", strings_compact=True)
            dflab = append_col_with_grp_index(dflab, ["CTXT_shape_next", "CTXT_loc_next"], "CTXT_shapeloc_next", strings_compact=True)
            dflab = append_col_with_grp_index(dflab, ["CTXT_shapeloc_prev", "gridloc", "CTXT_shapeloc_next"], "CTXT_ALL_shape", strings_compact=True)

            dflab = append_col_with_grp_index(dflab, ["gridloc", "stroke_index"], "loc_si")
            dflab = append_col_with_grp_index(dflab, ["shape", "gridloc"], "shape_loc")
            dflab = append_col_with_grp_index(dflab, ["CTXT_loc_prev", "CTXT_shape_prev", "gridloc"], "CTXT_present_1")
            dflab = append_col_with_grp_index(dflab, ["CTXT_loc_prev", "CTXT_shape_prev", "shape"], "CTXT_present_1b")
            dflab = append_col_with_grp_index(dflab, ["CTXT_loc_prev", "CTXT_shape_prev", "gridloc", "shape"], "CTXT_present_2")
            dflab = append_col_with_grp_index(dflab, ["CTXT_loc_prev", "CTXT_shape_prev", "shape", "gridloc", "CTXT_loc_next"], "CTXT_ALL_1")
            dflab = append_col_with_grp_index(dflab, ["CTXT_loc_prev", "CTXT_shape_prev", "shape", "gridloc", "CTXT_shape_next"], "CTXT_ALL_2")
            dflab = append_col_with_grp_index(dflab, ["CTXT_loc_prev", "CTXT_shape_prev", "shape", "gridloc", "CTXT_loc_next", "CTXT_shape_next"], "CTXT_ALL_MAX")
            dflab = append_col_with_grp_index(dflab, ["task_kind", "stroke_index"], "tk_si")
            dflab = append_col_with_grp_index(dflab, ["gridloc", "stroke_index_semantic"], "loc_sis")

            # get location bin within the gridloc bin
            from pythonlib.dataset.dataset_strokes import DatStrokes
            ds = DatStrokes()
            ds.location_redefine_gridloc_locally(2, dflab, False)

            pa.Xlabels["trials"] = dflab
    else:
        print(which_level)
        assert False

def dfallpa_combine_trial_strokes_from_already_loaded_DFallpa():
    """

    :return:
    """
    assert False, "ignore, instead use dfallpa_extraction_load_wrapper_combine_trial_strokes"
    # NOTE: this works, but it is clunky, as does lots of reshaping and slicing in order
    # to get the pa. Better to extract the stroke PAs separated before concating to tirals.

    # Extract separate pa for each stroke index (from the single PA_STROKE), and place as new rows into the "trial" DFallPA.
    # Guarantees:
    # - channels match across all pa for a given brain region.
    # - checks that shape seuqences are the same (but this only checjs up to the n strokes in sequence stored in the trial PA).
    # Doesnt guarantee:
    # - trialcodes may not be aligned between strokes and trials pa. Skipped forcing this, since some trials may lack a stroke index..


    RES_DFMULT_NEW = []
    twind_trial = (-0.6, 0.6)
    twind_stroke = (-0.6, 0.6)
    DEBUG = False

    list_br = sorted(DFallpa["bregion"].unique().tolist())
    assert list_br == sorted(DFallpaSTROKE["bregion"].unique().tolist())

    for bregion in list_br:
        print(bregion, " .... ")
        wl = "trial"
        event_trial = "03_samp" # pick any, this just for the trialcodes...
        pa_trial = extract_single_pa(DFallpa, bregion, twind_trial, wl, event_trial)

        wl = "stroke"
        event_stroke = "00_stroke"
        pa_stroke = extract_single_pa(DFallpaSTROKE, bregion, twind_stroke, wl, event_stroke)

        ##### Check that channels match
        assert pa_stroke.Chans==pa_trial.Chans
        # Note: it's ok if time bins dont match.

        print(pa_trial.X.shape)
        print(pa_stroke.X.shape)

        ##### Visualize the match between trial and stroke pa (print each trialcode one by one).
        df_trial = pa_trial.Xlabels["trials"]
        df_stroke = pa_stroke.Xlabels["trials"]

        print("They have identical trialcodes: ", sorted(df_trial["trialcode"].unique().tolist()) == sorted(df_stroke["trialcode"].unique().tolist()))

        list_tc = set(df_trial["trialcode"].tolist() + df_stroke["trialcode"].tolist())
        for tc in list_tc:
            a = df_trial.loc[df_trial["trialcode"]==tc, ["seqc_0_shape", "seqc_1_shape"]].values.tolist()
            b = df_stroke.loc[df_stroke["trialcode"]==tc, ["stroke_index", "shape_oriented"]].values.tolist()

            if DEBUG:
                s = f"{tc} -- {a} -- {b}"
                print(s)

            # Check that shapes in order match
            tmp_b = [bb[1] for bb in b][:2]
            n = len(tmp_b)
            tmp_a = a[0][:n]
            assert tmp_a == tmp_b
        print("GOOD! trialcodes match exaclty, and sequence of shapes matches, checked to the extent that they are stored in df_trial")

        ##### GET SHARED LABELS (across trial and stroke)
        # Make a dataframe of variables, each row a trialcode, which maps from trialcode to variables
        # that will be used in general across trial and stroke level pa.
        n_strokes_get = 3
        resthis = []
        for tc in list_tc:

            resthis.append({
                "trialcode":tc,
            })

            map_var_to_value = {}
            for si_get in range(n_strokes_get):
                tmp = df_stroke[(df_stroke["trialcode"]==tc) & (df_stroke["stroke_index"]==si_get)]
                if len(tmp)==0:
                    sh = "IGN"
                    loc = ("IGN",)
                elif len(tmp)==1:
                    sh = tmp["shape_oriented"].values[0]
                    loc = tmp["gridloc"].values[0]
                else:
                    print(tmp)
                    assert False

                map_var_to_value[f"seqc_{si_get}_shape"] = sh
                map_var_to_value[f"seqc_{si_get}_loc"] = loc
                resthis[-1][f"seqc_{si_get}_shape"] = sh
                resthis[-1][f"seqc_{si_get}_loc"] = loc
        dflab = pd.DataFrame(resthis)

        ##### COLLECT - for each stroke index, extract a slice of pa_stroke that is just that stroke index.
        for stroke_index_get in range(max(df_stroke["stroke_index"])+1):
            inds_keep = df_stroke[df_stroke["stroke_index"]==stroke_index_get].index.tolist()
            pa_tmp = pa_stroke.slice_by_dim_indices_wrapper("trials", inds_keep, reset_trial_indices=True)
            # PA_EACH_STROKE_INDEX[stroke_index_get] = pa_tmp

            # Assign a new column for shape, which matches terminology of "trial" level data
            pa_tmp.labels_features_input_from_dataframe_merge_append(dflab)
            # pa_tmp.Xlabels["trials"][f"seqc_{stroke_index_get}_shape"] = pa_tmp.Xlabels["trials"]["shape_oriented"]
            # pa_tmp.Xlabels["trials"][f"seqc_{stroke_index_get}_loc"] = pa_tmp.Xlabels["trials"]["shape_oriented"]

            RES_DFMULT_NEW.append({
                "which_level":"trial",
                "event":f"06_on_STK_{stroke_index_get}",
                "bregion":bregion,
                "twind":twind_stroke,
                "pa":pa_tmp
            })

        # For every trial-level pa, append the general variables
        for i, row in DFallpa.iterrows():
            if row["bregion"] == bregion:
                try:
                    row["pa"].labels_features_input_from_dataframe_merge_append(dflab)
                except Exception as err:
                    print(row)
                    print("Probably event_trial should be changed to an event that has all the tcs that exist for this bregion...")

    # Keep all
    DFallpa = pd.concat([DFallpa, pd.DataFrame(RES_DFMULT_NEW)]).reset_index(drop=True)

    return DFallpa


def dfallpa_extraction_load_wrapper_combine_trial_strokes(animal, date,
                                                question_trial, question_stroke,
                                                list_time_windows,
                                                combine_into_larger_areas = True, exclude_bad_areas=True,
                                                SPIKES_VERSION="tdt",
                                                HACK_RENAME_SHAPES = True,
                                                fr_normalization_method="each_time_bin",
                                                  check_that_shapes_match=True,
                                                  check_that_locs_match=True,
                                                  events_keep_trials = None,
                                                          ):
    """
    Helper to load a single dataset across "trial" and "stroke" levels, and concatenate them
    into a single DFallPA.

    Will make sure that seqc_{}_shape and loc are present, making sure that seqc_{si}_shape
    and seqc_{si}_loc are correct (within each stroke pa) and match (across trial and stroke pas).

    GUarantees:
    - chans will match, within each bregion, across all pa
    - shapes and locations at each index will matchn between trial and strokes (if check... are both on).

    Otherwise, no guarantee that PAs will have same trials.

    RETURNS:
    - DFallpaALL, each row a single pa...
    """

    prune_low_fr_sites = False # or else chans may not match between trials and strokes

    if events_keep_trials is None:
        events_keep_trials = ['03_samp', '05_first_raise']
    DFallpaTRIALS = dfallpa_extraction_load_wrapper(animal, date, question_trial, list_time_windows, "trial",
                                                    events_keep_trials, combine_into_larger_areas, exclude_bad_areas,
                                                    SPIKES_VERSION=SPIKES_VERSION,
                                                    HACK_RENAME_SHAPES=HACK_RENAME_SHAPES,
                                                    fr_normalization_method=fr_normalization_method, prune_low_fr_sites=prune_low_fr_sites)

    DFallpaSTROKES = dfallpa_extraction_load_wrapper(animal, date, question_stroke, list_time_windows, "stroke", None,
                                                     combine_into_larger_areas, exclude_bad_areas,
                                                     SPIKES_VERSION=SPIKES_VERSION,
                                                     HACK_RENAME_SHAPES=HACK_RENAME_SHAPES,
                                                     strokes_split_into_multiple_pa=True,
                                                     fr_normalization_method=fr_normalization_method, prune_low_fr_sites=prune_low_fr_sites)


    # If you want to add general varaibles to all pa in both datasets, then do this, wherever
    # Snippets is present. Decided to skip this, as is better to make sure variables are
    # extracted earlier, when extract SP in SP.datasetbeh_preprocess_clean_by_expt.



    # # For each trialcode, extract beh info that applies across all data
    #
    # # - Get one pa at trial level
    # ev = DFallpa["event"].unique()[0]
    # pa = DFallpa[DFallpa["event"]==ev]["pa"].values[0]
    # dflab = pa.Xlabels[]
    # list_tc = pa
    # D = SP.datasetbeh_extract_dataset()
    # D.seqcontext_preprocess()
    #
    #
    # ntake = 5
    # cols_take = ["trialcode"] + [f"seqc_{i}_shape" for i in range(ntake)] + [f"seqc_{i}_loc" for i in range(ntake)]
    # dflab_all = D.Dat.loc[:, cols_take]
    # D.Dat["trialcode"]
    #
    # from pythonlib.tools.pandastools import slice_by_row_label
    # pathis["stroke_index"]
    # import numpy as np
    # # for each pa, append the same columns
    # for i, row in DFallpaSTROKES.iterrows():
    #     pathis = row["pa"]
    #     dflab_this = pathis.Xlabels["trials"]
    #
    #     tmp = dflab_this["stroke_index"].unique()
    #     assert len(tmp)==1
    #     si = tmp[0]
    #
    #     tcs = dflab_this["trialcode"].tolist()
    #
    #     # Chekc shapes
    #     shapes_in_dflab_all = slice_by_row_label(dflab_all, "trialcode", tcs, assert_exactly_one_each=True)[f"seqc_{si}_shape"]
    #     shapes_in_dflab_this = dflab_this["shape_oriented"]
    #     assert np.all(shapes_in_dflab_all==shapes_in_dflab_this), "probably a stroke was skipped before etraction to DS, therefore it skips an index..."
    #
    #     # Chekc locations
    #     shapes_in_dflab_all = slice_by_row_label(dflab_all, "trialcode", tcs, assert_exactly_one_each=True)[f"seqc_{si}_loc"]
    #     shapes_in_dflab_this = dflab_this["gridloc"]
    #     assert np.all(shapes_in_dflab_all==shapes_in_dflab_this), "probably a stroke was skipped before etraction to DS, therefore it skips an index..."
    #
    #     # update the columns in pa with the global variables.
    #     pathis.labels_features_input_from_dataframe_merge_append(dflab_all)


    # Check that, for each stroke's pa, its shape is match to the seqc_ shaope from trials.
    from pythonlib.tools.pandastools import slice_by_row_label
    import numpy as np

    # For each trialcode, extract beh info that applies across all data
    # - Get one pa at trial level
    ev = sorted(DFallpaTRIALS["event"].unique())[0] # take first, its most liekly to have all trialcodes.
    pa = DFallpaTRIALS[DFallpaTRIALS["event"]==ev]["pa"].values[0]
    dflab_trial = pa.Xlabels["trials"]

    # for each pa in strokes, check it against dflab_trial
    for i, row in DFallpaSTROKES.iterrows():
        pathis = row["pa"]
        dflab_stroke_this = pathis.Xlabels["trials"]

        tmp = dflab_stroke_this["stroke_index"].unique()
        assert len(tmp)==1
        si = tmp[0]
        tcs = dflab_stroke_this["trialcode"].tolist()

        # Chekc shapes
        if check_that_shapes_match:
            shapes_in_dflab_all = slice_by_row_label(dflab_trial, "trialcode", tcs, assert_exactly_one_each=True)[f"seqc_{si}_shape"]
            shapes_in_dflab_this = dflab_stroke_this["shape_oriented"]
            shapes_in_dflab_this_2 = dflab_stroke_this[f"seqc_{si}_shape"]
            if not np.all(shapes_in_dflab_all==shapes_in_dflab_this):
                for tc, sh1, sh2 in zip(tcs, shapes_in_dflab_all, shapes_in_dflab_this):
                    if not sh1==sh2:
                        print(tc, sh1, sh2)
                assert False, "probably either (i) you need to re-extract Snippets after you have just updated char cluster shapes (resaon: trial-data loads labels anew, while DS uses old labels), or (ii) a stroke was skipped before etraction to DS, therefore it skips an index..."
            if not np.all(shapes_in_dflab_all==shapes_in_dflab_this_2):
                for tc, sh1, sh2 in zip(tcs, shapes_in_dflab_all, shapes_in_dflab_this_2):
                    if not sh1==sh2:
                        print(tc, sh1, sh2)
                assert False, "probably either (i) you need to re-extract Snippets after you have just updated char cluster shapes (resaon: trial-data loads labels anew, while DS uses old labels), or (ii) a stroke was skipped before etraction to DS, therefore it skips an index..."

        # Chekc locations
        if check_that_locs_match:
            shapes_in_dflab_all = slice_by_row_label(dflab_trial, "trialcode", tcs, assert_exactly_one_each=True)[f"seqc_{si}_loc"]
            shapes_in_dflab_this = dflab_stroke_this["gridloc"]
            shapes_in_dflab_this_2 = dflab_stroke_this[f"seqc_{si}_loc"]
            assert np.all(shapes_in_dflab_all==shapes_in_dflab_this), "probably a stroke was skipped before etraction to DS, therefore it skips an index..."
            assert np.all(shapes_in_dflab_all==shapes_in_dflab_this_2), "probably a stroke was skipped before etraction to DS, therefore it skips an index..."

    # Check that channels match within each bregions across all datasets.
    map_br_to_chans = {}
    for i, row in DFallpaSTROKES.iterrows():
        br = row["bregion"]
        if br in map_br_to_chans:
            if not map_br_to_chans[br] == row["pa"].Chans:
                print(map_br_to_chans[br])
                print(row["pa"].Chans)
                print(row)
                assert False
        else:
            map_br_to_chans[br] = row["pa"].Chans
    for i, row in DFallpaTRIALS.iterrows():
        br = row["bregion"]
        if br in map_br_to_chans:
            if not map_br_to_chans[br] == row["pa"].Chans:
                print(map_br_to_chans[br])
                print(row["pa"].Chans)
                print(row)
                assert False
        else:
            map_br_to_chans[br] = row["pa"].Chans

    ##### CONCAT
    DFallpaALL = pd.concat([DFallpaTRIALS, DFallpaSTROKES]).reset_index(drop=True)
    # call all wl = trial
    DFallpaALL["which_level"] = "trial"

    return DFallpaALL

def dfallpa_extraction_load_wrapper_from_MS(MS, question, list_time_windows, which_level="trial", events_keep=None,
                                            combine_into_larger_areas=True, exclude_bad_areas=True,
                                            bin_by_time_dur=None, bin_by_time_slide=None, slice_agg_slices=None,
                                            slice_agg_vars_to_split=None, slice_agg_concat_dim="trials",
                                            HACK_RENAME_SHAPES=True, substrokes_plot_preprocess=True,
                                            strokes_split_into_multiple_pa=False,
                                            fr_normalization_method="each_time_bin", REGENERATE_SNIPPETS=True,
                                            path_to_save_example_fr_normalization=None, prune_low_fr_sites=True):
    """ Wrapper of dfallpa_extraction_load_wrapper for loading given already loaded
    MS,
    From SP (already saved) --> DFallpa
    (ie not an option to load already saved DFallpa  but could implement that).
    """
    from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
    from neuralmonkey.classes.session import load_mult_session_helper
    from neuralmonkey.analyses.rsa import rsagood_questions_dict

    animal = MS.animal()
    date = MS.date()
    assert isinstance(list_time_windows[0], (list, tuple))

    PRE_DUR = min([x[0] for x in list_time_windows])
    POST_DUR = max([x[1] for x in list_time_windows])

    # Load a question
    q_params = rsagood_questions_dict(animal, date, question=question)[question]
    print("Question:", question)
    print("These questions params:")
    for k, v in q_params.items():
        print(k, " -- ", v)
    assert which_level in q_params["list_which_level"], "or else might run into error later."

    # Keep only specific events - to make the following faster.
    if events_keep is None:
        events_keep = q_params["events_keep"]

    # Load previously generated
    SP, _ = load_and_concat_mult_snippets(MS, which_level = which_level, events_keep=events_keep,
        DEBUG=False,  REGENERATE_SNIPPETS=REGENERATE_SNIPPETS, PRE_DUR=PRE_DUR, POST_DUR=POST_DUR,
        prune_low_fr_sites=prune_low_fr_sites)

    # Run this early, before run further pruning stuff.
    SP.datamod_append_outliers()

    # Clean up SP and extract features
    D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
        ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"],
        substrokes_plot_preprocess=substrokes_plot_preprocess,
        HACK_RENAME_SHAPES=HACK_RENAME_SHAPES)

    if question=="PIG_BASE_saccade_fix_on":
        # want to extract clusterfix columns, which is a memory-intensive operation
        import gc
        gc.collect()
        del MS

        print("adding saccade-fixation columns...")
        SP._add_clusterfix_saccfix_columns(filter_outliers=True, filter_only_first_shapefix=False)
        list_features_extraction = ["seqc_0_shape", "seqc_0_loc", "seqc_1_shape", "seqc_1_loc", 
                            "seqc_2_shape", "seqc_2_loc", "seqc_3_shape", "seqc_3_loc",
                            "seqc_0_loc_on_clust", "seqc_1_loc_on_clust", "seqc_2_loc_on_clust", "seqc_3_loc_on_clust",
                            "trial_neural", "event_idx_within_trial", "between-stimonset-and-go",
                            "early-or-late-planning-period", "fixation-centroid",
                            "shape-fixation", "loc-fixation", "first-fixation-on-shape",
                            "shape-macrosaccade-index", "saccade-dir-angle", "saccade-dir-angle-bin",
                            "is-fixated-on-seqc0shape", "prev-shape-fixation", "prev-loc-fixation", "is-first-macrosaccade"] + list_features_extraction
        # NOTE: if change _add_clusterfix_saccfix_columns, must add new column names to above list.

    # If this is "strokes" SP, you have option of renaming events to the stroke index, allowing to
    # extract separate PA for each stroke index.
    if SP.Params["which_level"] in ["stroke", "stroke_off"] and strokes_split_into_multiple_pa:
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        if SP.Params["which_level"]=="stroke":
            pref = "06_on_STK"
        elif SP.Params["which_level"]=="stroke_off":
            pref = "06_off_STK"
        else:
            print(SP.Params)
            assert False, "code it"
        # elif SP.Params["which_level"]=="substroke":
        #     pref = "06_on_SS"
        # elif SP.Params["which_level"]=="substroke_iff":
        #     pref = "06_on_SS"

        def F(x):
            si = x["stroke_index"]
            ev = f"{pref}_{si}"
            return ev

        SP.DfScalar = applyFunctionToAllRows(SP.DfScalar, F, "event")
        SP.DfScalar["event_aligned"] = SP.DfScalar["event"]
        SP.Params["_list_events"] = sorted(SP.DfScalar["event"].unique().tolist())
        SP.Params["list_events_uniqnames"] = sorted(SP.DfScalar["event"].unique().tolist())

    ## Extract all popanals
    DFallpa = snippets_extract_popanals_split_bregion_twind(SP, list_time_windows,
                                                    list_features_extraction,
                                                    combine_into_larger_areas=combine_into_larger_areas,
                                                    events_keep=events_keep,
                                                    exclude_bad_areas=exclude_bad_areas)

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

    #################### Normalize PA firing rates if needed
    if fr_normalization_method is not None:
        if fr_normalization_method=="each_time_bin":
            # Then demean in each time bin indepednently
            subtract_mean_at_each_timepoint = True
            subtract_mean_across_time_and_trial = False
        elif fr_normalization_method=="across_time_bins":
            # ALl time bins subtract the same scalar --> maintains temporal moudlation.
            subtract_mean_at_each_timepoint = False
            subtract_mean_across_time_and_trial = True
        else:
            print(fr_normalization_method)
            assert False

        from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization
        list_panorm = []

        for i, pa in enumerate(DFallpa["pa"].tolist()):
            if path_to_save_example_fr_normalization is not None and i==0:
                plot_example_chan_number = pa.Chans[0]
                if which_level=="trial":
                    plot_example_split_var_string = "seqc_0_shape"
                elif which_level=="stroke":
                    plot_example_split_var_string = "shape"
                else:
                    plot_example_split_var_string = q_params["effect_vars"][0]
            else:
                plot_example_chan_number = None
                plot_example_split_var_string = None
            PAnorm, PAscal, PAscalagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(pa, None,
                                                                                              DO_AGG_TRIALS=False,
                                                                                              plot_example_chan_number=plot_example_chan_number,
                                                                                                plot_example_split_var_string = plot_example_split_var_string,
                                                                                              subtract_mean_at_each_timepoint=subtract_mean_at_each_timepoint,
                                                                                              subtract_mean_across_time_and_trial=subtract_mean_across_time_and_trial)
            if path_to_save_example_fr_normalization is not None and i==0:
                savefig(fig, path_to_save_example_fr_normalization)
            list_panorm.append(PAnorm)
        DFallpa["pa"] = list_panorm

    return DFallpa

def dfallpa_extraction_load_wrapper(animal, date, question, list_time_windows, which_level="trial", events_keep=None,
                                    combine_into_larger_areas=True, exclude_bad_areas=True, bin_by_time_dur=None,
                                    bin_by_time_slide=None, slice_agg_slices=None, slice_agg_vars_to_split=None,
                                    slice_agg_concat_dim="trials", LOAD_FROM_RSA_ANALY=False,
                                    rsa_ver_dist="euclidian_unbiased", rsa_subtr=None, rsa_agg=True, rsa_invar=None,
                                    SPIKES_VERSION="tdt", HACK_RENAME_SHAPES=True, substrokes_plot_preprocess=True,
                                    strokes_split_into_multiple_pa=False, fr_normalization_method="each_time_bin",
                                    path_to_save_example_fr_normalization=None, prune_low_fr_sites=True):

    """ [GOOD] Hihg level to extrqact
    DFallpa, with all preprocessing steps built in, must have already extgracted Snippets.
    By default this gets separate pa for each (event, bregion), but has many methods for
    slicing and aggregating across multiple PAs.
    """
    from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
    from neuralmonkey.classes.session import load_mult_session_helper
    from neuralmonkey.analyses.rsa import rsagood_questions_dict
    from neuralmonkey.classes.population_mult import snippets_extract_popanals_split_bregion_twind

    if LOAD_FROM_RSA_ANALY:
        # Saved in analy_rsa_script.py
        from neuralmonkey.analyses.rsa import rsagood_pa_vs_theor_wrapper_loadresults, rsagood_pa_vs_theor_single
        version_distance = rsa_ver_dist
        subtract_mean_each_level_of_var = rsa_subtr
        DO_AGG_TRIALS = rsa_agg
        vars_test_invariance_over_dict = rsa_invar

        DFallpa = rsagood_pa_vs_theor_wrapper_loadresults(animal, date, question,
                                                          version_distance, DO_AGG_TRIALS,
                                                          subtract_mean_each_level_of_var,
                                                          vars_test_invariance_over_dict)[0]
    else:
        # Generate it from saved Snippets
        assert list_time_windows is not None

        ############### PARAMS
        # animal = "Diego"
        # date = 230615
        # exclude_bad_areas = True
        # SPIKES_VERSION = "tdt" # since Snippets not yet extracted for ks
        # bin_by_time_dur = 0.05
        # bin_by_time_slide = 0.025

        ## Load Snippets
        MS = load_mult_session_helper(date, animal, spikes_version=SPIKES_VERSION)

        # If this is aligned to fixations, first extract them.
        if which_level == "saccade_fix_on":
            for sn in MS.SessionsList:
                if not sn.clusterfix_check_if_preprocessing_complete():
                    sn.extract_and_save_clusterfix_results()
            
        # Get DFallpa
        DFallpa = dfallpa_extraction_load_wrapper_from_MS(MS, question, list_time_windows, which_level, events_keep,
                                                          combine_into_larger_areas, exclude_bad_areas, bin_by_time_dur,
                                                          bin_by_time_slide, slice_agg_slices, slice_agg_vars_to_split,
                                                          slice_agg_concat_dim, HACK_RENAME_SHAPES,
                                                          substrokes_plot_preprocess=substrokes_plot_preprocess,
                                                          strokes_split_into_multiple_pa=strokes_split_into_multiple_pa,
                                                          fr_normalization_method=fr_normalization_method,
                                                          path_to_save_example_fr_normalization=path_to_save_example_fr_normalization,
                                                          prune_low_fr_sites=prune_low_fr_sites)

    return DFallpa


def load_dataset_mult_wl(animal, date, list_which_level):

    list_out = []
    list_params =[]
    for wl in list_which_level:
        out, params = load_dataset_single(animal, date, wl, return_as_df=True)
        list_out.append(out)
        list_params.append(params)

    DFallpa = pd.concat(list_out).reset_index(drop=True)

    return DFallpa, list_params


def load_dataset_single(animal, date, which_level):
    """ Load a single dataset, which is a dict of popanals, each keyed by
    (event, bregion, twind). WIll first look for PA created from SP concateed
    across mult sessions. if that doesnt eixst, then looks for single session.
    (In progrsss) And concatenates.
    RETURNS:
        - DictEvBrTw_to_PA, dict (which_level, event, bregion, twind): pa.
    """

    # First, look for data made from SP using multiple sessions. (concated SP)
    savedir = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/mult_session"
    path = f"{savedir}/{animal}-{date}-{which_level}-*.pkl"
    files = glob.glob(path)
    print("Found mult session: ", files)

    if len(files)==1:
        mult_or_single = "mult"
    elif len(files)==0:
        # Then look for singles
        savedir = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/SAVED_POPANALS/single_session"
        path = f"{savedir}/{animal}-{date}-{which_level}-*.pkl"
        files = glob.glob(path)

        print("Found single session: ", files)

        if len(files)==0:
            print()
            assert False, "doesnt exist.."

        mult_or_single = "single"
    else:
        assert False

    ##############
    if len(files)>1:
        print(files)
        assert False, "not yet coded.. should extract session num from filename, then before concat should append sess num as column in all pa.Xlabels[trials]"

        # # Load and concat all files
        # for fi in files:
        #     with open(fi, "rb") as f:
        #         DictEvBrTw_to_PA = pickle.load(f)
    elif len(files)==1:
        # Load and concat all files
        fi = files[0]
        with open(fi, "rb") as f:
            DFallpa = pickle.load(f)
    else:
        assert False

    # # Convert each key from (event, bregion, twind) to (wl, event, bregion, twind)
    # dicttmp = {}
    # for k, v in DictEvBrTw_to_PA.items():
    #     k = tuple([which_level] + list(k))
    #     dicttmp[k] = v
    # DictEvBrTw_to_PA = dicttmp

    # # condition, for entry to dataframe
    # if return_as_df:
    #     tmp = []
    #     for k, v in DictEvBrTw_to_PA.items():
    #
    #         # Make sure pa itself is keeping track of the outer varibles,
    #         # for sanity checks once you start splitting and grouping.
    #         v.Xlabels["trials"]["which_level"] = k[0]
    #         v.Xlabels["trials"]["event"] = k[1]
    #         v.Xlabels["trials"]["bregion"] = k[2]
    #         v.Xlabels["trials"]["twind"] = [k[3] for _ in range(len(v.Xlabels["trials"]))]
    #
    #         tmp.append({
    #             "which_level":k[0],
    #             "event":k[1],
    #             "bregion":k[2],
    #             "twind":k[3],
    #             "pa":v
    #         })
    #     DFallpa = pd.DataFrame(tmp)
    #     out = DFallpa
    # else:
    #     out = DictEvBrTw_to_PA

    Params = {
        "animal":animal,
        "date":date,
        "files":files,
        "mult_or_single":mult_or_single
    }
    return DFallpa, Params

def extract_single_pa(DFallpa, bregion, twind=None, which_level = "trial", event = "03_samp",
                      pa_field="pa"):
    """ Quick, get a isngle pa... failing if not found.
    """

    if twind is None:
        list_twind = DFallpa["twind"].unique().tolist()
    else:
        list_twind = [twind]
    # if twind is None:
    #     assert len(DFallpa["twind"].unique())==1
    #     twind = DFallpa["twind"].values[0]


    # assert twind is not None
    # display(DFallpa)
    # assert False
    a = DFallpa["which_level"]==which_level
    # print(a)
    # print(which_level, type(which_level))
    # assert False
    b = DFallpa["event"]==event
    c = DFallpa["bregion"]==bregion
    d = DFallpa["twind"].isin(list_twind)

    tmp = DFallpa[a & b & c & d]
    if not len(tmp)==1:
        print(DFallpa)
        print(len(tmp))
        print(which_level, event, bregion, list_twind)
        print(sum(a), sum(b), sum(c), sum(d))
        assert False
    pa = tmp[pa_field].values[0].copy()

    return pa
    
def dfpa_extract_single_window(DFallpa, which_level, event, twind):
    """ Return df with multiple pa (rows) all with the same specific
    values for wl, ev, and tw
    """
    a = DFallpa["which_level"] == which_level
    b = DFallpa["event"] == event
    c = DFallpa["twind"] == twind

    dfthis = DFallpa[(a & b & c)].reset_index(drop=True)
    assert len(dfthis)>0
    assert len(dfthis["bregion"].unique()) == len(dfthis)

    return dfthis

def dfpa_slice_specific_windows(DFallpa, list_pa_get):
    """ Return slice of DF which has onkly the specific onbinations of
    (wl, ev, tw) in list_pa_get
    PARAMS:
    - list_pa_get, list of tuples, each (wl, ev, tw), and thus each deefining a
    slice of DF (rows are bregions). Will collect then in order of items in list_pa_get
    Example:
    # list_pa_get = [
    #     ("stroke", "00_stroke", (-0.6, -0.5)),
    #     ("stroke", "00_stroke", (0.4, 0.6))
    # ]
    RETURNS:
    - df, concated slices, each one of the tuples of list_pa_get. N rows should be
    len(list_pa_get) * len(bregions). Will have same columnes as input df
    """
    list_df = []
    for wl, ev, tw in list_pa_get:
        list_df.append(dfpa_extract_single_window(DFallpa, wl, ev, tw))
    DFallpa_THIS = pd.concat(list_df).reset_index(drop=True)
    return DFallpa_THIS

def dfpa_group_and_split(DFallpa, vars_to_concat=None, vars_to_split=None,
                         DEBUG=False, concat_dim="trials", pa_column="pa"):
    """ Flexible method to concatenate PAs across all levels for
    given dimensions (vars_to_concat) and to maintain separate PA
    for each level of variables in vars_to_split.
    PARAMS:
    - vars_to_concat, list of str,
    - vars_to_split, list of str.
    (One of the above must be None, since they are redundant, just 2 methods
    to do same thing).
    RETURNS:
        DFallpa, with columns being the variables in vars_to_split, and each
        pa concated across vars_to_concat. For the columns which were grouped (concatted),
        replaces the value with "dummy", since the old values have been combined. THey
        are still accessible within the PA itself.
    - if concat across events, then event information will be retained in times, with
    Times = "event_num|time".
    EXAMPLE:
        vars_to_concat = ["which_level", "event", "twind"]
        vars_to_split = None
    """
    from neuralmonkey.classes.population import concatenate_popanals_flexible

    assert concat_dim in ["trials", "times"], "not coded yet"

    allvars = ["which_level", "event", "bregion", "twind"]

    # They are redundant informations.
    if vars_to_concat is None:
        assert vars_to_split is not None
        assert "bregion" not in vars_to_split, "For now, must have this, not sure how best to ###concat bregions, since they have diff chans..."
        vars_to_concat = [var for var in allvars if var not in vars_to_split]
    else:
        assert vars_to_split is None
        assert "bregion" not in vars_to_concat, "For now, must have this, not sure how best to ###concat bregions, since they have diff chans..."
        vars_to_split = [var for var in allvars if var not in vars_to_concat]

    # give a new conj var
    if "event" in vars_to_concat and concat_dim=="trials":
        # uiseful - a conjucntionv ariable for each tw
        from pythonlib.tools.pandastools import append_col_with_grp_index
        list_pa = DFallpa[pa_column].tolist()
        # list_pa_new = []
        for pa in list_pa:
            # print("HERERER", pa.Xlabels["trials"]["twind"].value_counts())
            pa.Xlabels["trials"] = append_col_with_grp_index(pa.Xlabels["trials"],
                                                            ["which_level", "event", "twind"],
                                                            "wl_ev_tw",
                                                            use_strings=True,
                                                            strings_compact=True)
            # print("HERERER", pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
        # assert False

    if False:
        for grp in DFallpa.groupby(vars_to_split):
            list_pa = grp[1]["pa"].tolist()
            # print(grp[1])

            # concatenate them
            pa_cat, twind_cat = concatenate_popanals_flexible(list_pa)
    else:
        def F(x):
            # concatenate them
            list_pa = x[pa_column].tolist()
            return concatenate_popanals_flexible(list_pa, concat_dim=concat_dim)[0]

        DFallpa = DFallpa.groupby(vars_to_split, as_index=False).apply(F).reset_index(drop=True)
        # tmp = DFallpa.groupby(vars_to_split, as_index=False).apply(F)
        # # DFallpa = pd.DataFrame({"pa":tmp})
        # DFallpa = pd.DataFrame({"pa":tmp}, index=tmp.index)


    # For the other columns which were concated, they are not presnet in output.
    # add them back, with "dummy" value
    # print(vars_to_concat)
    # print(vars_to_split)
    for var in vars_to_concat:
        DFallpa[var] = "dummy"

    # HACKY, it returns df with column named None insted of pa.
    DFallpa[pa_column] = DFallpa[None]
    del DFallpa[None]
    #
    # for pa in DFallpa["pa"]:
    #     print(pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
    # assert False

    # HACKY - Redefine event to be conj varoiable. useful for downstream analy
    if "event" in vars_to_concat and concat_dim=="trials":
        for pa in DFallpa["pa"].tolist():
            # print(pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
            # assert False
            pa.Xlabels["trials"]["event_orig"] = pa.Xlabels["trials"]["event"]
            pa.Xlabels["trials"]["event"] = pa.Xlabels["trials"]["wl_ev_tw"]

            a = pa.X.shape[1]
            b = len(pa.Xlabels["trials"])
            c = max(pa.Xlabels["trials"].index)+1
            if not a==b==c:
                print(a, b, c)
                assert False

    assert len(DFallpa)>0

    DFallpa["pa_x_shape"] = [row[pa_column].X.shape for _, row in DFallpa.iterrows()]
     
    if DEBUG:
        for pa in list_pa:
            print(pa.X.shape)

    if False: # No need to do this. This can fail for many legit reasions. E.g, diff events ahve diff not trials...
        # This is ok, since downstream will index into the dim values, and not just assume same shape
        # Sanity check that concated pas are identical across the concated dims
        from neuralmonkey.classes.population import check_get_common_values_this_dim
        list_pa = DFallpa["pa"].tolist()
        for dim in vars_to_concat:
            check_get_common_values_this_dim(list_pa, dim, assert_all_pa_have_same_values=True, dims_are_columns_in_xlabels=True)

    return DFallpa


def dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date, fr_mean_subtract_method = "across_time_bins",
        do_sitesdirty_extraction=True):
    """
    Apply seuqence of preprocessing steps to cases where multkiple events' PA were combined in DFallpa.
    I used this for decode moment stuff (around Jul 2024).
    Modifies DFallpa
    # fr_mean_subtract_method = "each_time_bin"
    PARAMS:
    - do_sitesdirty_extraction, bool, if True, then does (slow, like 10 min) extracation of sitesdirty preprocess metrics.
    Only does this if it can't find it already done and saved.
    """

    assert fr_mean_subtract_method in ["across_time_bins", "each_time_bin"]

    # (1) Prune to chans that are common across pa for each bregion (intersection of chans)|
    dfpa_match_chans_across_pa_each_bregion(DFallpa)
    
    # (2) Remove bad chans based on sitedirty preprocessing (e.g., drift)
    # First, check that preprocess data exist
    tmp, _ = dfallpa_preprocess_sitesdirty_single(DFallpa["pa"].values[0], animal, date)
    if tmp is None and do_sitesdirty_extraction:
        # Then do extraction of sitesdirty preprocess metrics. This can take time (like 10 min).
        from neuralmonkey.classes.session import load_mult_session_helper
        MS = load_mult_session_helper(date, animal)
        MS.sitesdirtygood_preprocess_wrapper(PLOT_EACH_TRIAL=True)
    
    # Try again
    tmp, _ = dfallpa_preprocess_sitesdirty_single(DFallpa["pa"].values[0], animal, date)
    if tmp is not None:
        # Then it exists -- run it.
        # savedir = "/tmp"
        list_pa = []
        for i, row in DFallpa.iterrows():
            PA = row["pa"]
            # plot_fr_after_replace_trials_dir = f"{savedir}/{row['bregion']}-{row['event']}"
            # os.makedirs(plot_fr_after_replace_trials_dir, exist_ok=True)
            plot_fr_after_replace_trials_dir = None
            PA, map_chan_to_trialcodes_replaced = dfallpa_preprocess_sitesdirty_single(PA, animal, date, plot_fr_after_replace_trials_dir)
            list_pa.append(PA)
        print("PA.X.shape, before and after dfallpa_preprocess_sitesdirty_single")
        for pa1, pa2 in zip(DFallpa["pa"].tolist(), list_pa):
            print(pa1.X.shape, " --> ", pa2.X.shape)
            # print(pa1.X.shape[0], " --> ", pa2.X.shape[0])
        # Replace PA
        DFallpa["pa"] = list_pa         

    # (2) Clean bad chans - based on fr modulation.
    dfpa_concatbregion_preprocess_clean_bad_channels(DFallpa, PLOT=False)

    # (3) Sqrt transform
    for pa in DFallpa["pa"]:
        pa.X = pa.X**0.5

    # (4) Normalize FR    
    PLOT=False
    # pa = DFallpa["pa"].values[10]
    # pa.plotNeurHeat(0)
    dfpa_concat_normalize_fr_split_multbregion_flex(DFallpa, fr_mean_subtract_method, PLOT)
    # pa = DFallpa["pa"].values[10]
    # pa.plotNeurHeat(0)
    

def dfpa_concatbregion_preprocess_clean_bad_channels(DFallpa, PLOT = False):
    """
    Concat all events fore ach bregion, compute scores of FR and FR modulation, and then 
    determine which channels are bad, then modify all pa in DFallpa to keep just those
    channels.

    TODO: In progress -- see TODO within. Tends to be conservative (keeps noise)
    """
    from pythonlib.tools.plottools import savefig

    if DFallpa["event"].unique().tolist() == ["00_stroke"]:
        # THis is the only event
        events_keep = ["00_stroke"]
    elif DFallpa["event"].unique().tolist() == ["fixon_preparation"]:
        # THis is the only event
        events_keep = ["fixon_preparation"]
    else:
        # Just use the events that ahve the same trialcodes and chans.
        # You must have already pruned chans to be same!
        # events_keep = DFallpa["event"].unique().tolist()
        events_keep = ["03_samp", "05_first_raise", "06_on_strokeidx_0"]

    list_bregion = DFallpa["bregion"].unique().tolist()

    # - smooth the fr
    dur = 0.2
    slide = 0.01

    savedir = f"/tmp/chans_pruning"
    os.makedirs(savedir, exist_ok=True)

    # For each bregion, track which channels to exclude
    # (keep if value is HIGHER than any of these criteria)
    # NOTE: these I determined empricallyl to be conservative, for Diego, 240619
    
    # with sm (0.2, 0.01)
    # THRESH_FR_RATIO_CLEAN = 0.5
    # THRESH_FR_RATIO_NOISY = 3.2
    # THRESH_FR_STD_MEAN = 0.1

    # No smoothing (empriical, not the closest look)
    THRESH_FR_RATIO_CLEAN = 0.65
    THRESH_FR_RATIO_NOISY = 5.5
    THRESH_FR_STD_MEAN = 0.2
    THRESH_FR_MOD_VS_MEAN = 0.2
    THRESH_FR_TRIALSTD_MEAN = 0.2
    # THRESH_FR_RATIO_CLEAN = 0.8
    # THRESH_FR_RATIO_NOISY = 65
    # THRESH_FR_STD_MEAN = 0.35
    # THRESH_FR_MOD_VS_MEAN = 0.35
    # THRESH_FR_TRIALSTD_MEAN = 0.35

    # Collect all chans for each bregion
    MAP_REGION_TO_CHANS_KEEP = {}
    for bregion in list_bregion:

        list_pa = DFallpa[
            (DFallpa["bregion"] == bregion) &
            (DFallpa["event"].isin(events_keep))
            ]["pa"].tolist()
        list_event = DFallpa[
            (DFallpa["bregion"] == bregion) &
            (DFallpa["event"].isin(events_keep))
            ]["event"].tolist()

        if len(list_pa)==0:
            print(events_keep)
            assert False, "probably need to update events_keep"

        # Smooth the fr
        if False:
            list_pa = [pa.agg_by_time_windows_binned(dur, slide) for pa in list_pa]

        # Confirm that trials are matched.
        trialcodes = None
        chans = None
        for pa in list_pa:
            if trialcodes is None:
                trialcodes = pa.Xlabels["trials"]["trialcode"].tolist()
                chans = pa.Chans
            else:
                assert trialcodes == pa.Xlabels["trials"]["trialcode"].tolist()
                assert chans == pa.Chans, "You must have already pruned chans to be same! See dfpa_concatbregion_preprocess_wrapper"
        
        # if bregion =="PMd":
        #     print(chans)
        #     print(len(list_pa))
        #     for pa in list_pa:
        #         print(pa.Chans)
        #     assert False

        # Get concated data
        X = np.concatenate([pa.X for pa in list_pa], axis=2) # (nchans, ntrials, ntimes_concatenated)
        
        # Get derived metrics
        # global mean fr for each chan
        fr_mean_all = np.mean(X.reshape(X.shape[0], -1), axis=1) # (nchans, )

        # "Modulation", two methods
        # i. for each trial, get std across time. Then average that std across trials.
        fr_std_trial_noisy = np.mean(np.std(X, axis=2), axis=1) # (nchans, )
        # ii. get mean fr (timecourse), then get std of that over time
        fr_std_trial_clean = np.std(np.mean(X, axis=1), axis=1)

        # std of trial-mean fr
        fr_std_trialmean = np.std(np.mean(X, axis=2), axis=1) # (nchans, )

        ### Plots
        if PLOT:
            for fr_std_version in ["noisy", "clean"]:
                if fr_std_version=="noisy":
                    fr_std_trial = fr_std_trial_noisy
                elif fr_std_version=="clean":
                    fr_std_trial = fr_std_trial_clean
                else:
                    assert False

                fig, axes = plt.subplots(2, 4, figsize=(4*7, 2*7))

                ax = axes.flatten()[0]
                s = ax.scatter(fr_mean_all, fr_std_trial, c=fr_std_trialmean, cmap="gray")
                for ch, x, y in zip(chans, fr_mean_all, fr_std_trial):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("mean fr (global)")
                ax.set_ylabel("mean of trial-std fr")
                ax.set_title("color=std of trial-mean fr")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                ax = axes.flatten()[1]
                s = ax.scatter(fr_mean_all, fr_std_trialmean, c=fr_std_trial, cmap="gray")
                for ch, x, y in zip(chans, fr_mean_all, fr_std_trialmean):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("mean fr (global)")
                ax.set_ylabel("std of trial-mean fr")
                ax.set_title("color=mean of trial-std fr")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                ax = axes.flatten()[2]
                s = ax.scatter(fr_std_trialmean, fr_std_trial, c=fr_mean_all, cmap="gray")
                for ch, x, y in zip(chans, fr_std_trialmean, fr_std_trial):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("std of trial-mean fr")
                ax.set_ylabel("mean of trial-std fr")
                ax.set_title("color=mean fr. (high y-value is good)")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                ax = axes.flatten()[3]
                fr_std_ratio = fr_std_trial/fr_std_trialmean
                s = ax.scatter(fr_mean_all, fr_std_ratio, c=fr_std_trial, cmap="gray")
                for ch, x, y in zip(chans, fr_mean_all, fr_std_ratio):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("mean fr (global)")
                ax.set_ylabel("fr std ratio (mean of trialstd / std of trialmean")
                ax.set_title("[goal: high y] color=mean of trial-std fr")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                ax = axes.flatten()[4]
                fr_std_over_fr_mean = fr_std_trial/fr_mean_all
                s = ax.scatter(fr_mean_all, fr_std_over_fr_mean, c=fr_std_trial, cmap="gray")
                for ch, x, y in zip(chans, fr_mean_all, fr_std_over_fr_mean):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("mean fr (global)")
                ax.set_ylabel("fr_std_over_fr_mean")
                ax.set_title("[goal: high y] color=mean of trial-std fr")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                ax = axes.flatten()[5]
                s = ax.scatter(fr_std_over_fr_mean, fr_std_ratio, c=fr_mean_all, cmap="gray")
                for ch, x, y in zip(chans, fr_std_over_fr_mean, fr_std_ratio):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("fr_std_over_fr_mean")
                ax.set_ylabel("fr std ratio (mean of trialstd / std of trialmean")
                ax.set_title("[goal: high both x and y] color=mean of fr")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                # ----------------------
                ax = axes.flatten()[6]
                s = ax.scatter(fr_std_trial_noisy, fr_std_trial_clean, c=fr_mean_all, cmap="gray")
                for ch, x, y in zip(chans, fr_std_trial_noisy, fr_std_trial_clean):
                    ax.text(x, y, ch, color="m", alpha=0.8, fontsize=7)
                plt.colorbar(s)
                ax.set_xlabel("mean of trial-std fr [noisy]")
                ax.set_ylabel("mean of trial-std fr [clean]")
                ax.set_title("color=mean fr")
                ax.set_xlim(xmin=0)
                ax.set_ylim(ymin=0)

                savefig(fig, f"{savedir}/stats-{bregion}-fr_std_version={fr_std_version}.pdf")
                plt.close("all")

        ### Final diagnostic scores
        fr_std_ratio_clean = fr_std_trial_clean/fr_std_trialmean
        fr_std_ratio_noisy = fr_std_trial_noisy/fr_std_trialmean
        fr_std_over_fr_mean = fr_std_trial_clean/fr_mean_all
        fr_mod_vs_mean = fr_std_trial_clean/fr_mean_all
        fr_trialstd_vs_mean = fr_std_trialmean/fr_mean_all

        # TODO: another metric which is:
        # THis might be best? Since the above incorreclty penalize you for modulation across trials

        # Save text file of the scores and chans
        from pythonlib.tools.listtools import stringify_list
        from pythonlib.tools.expttools import writeStringsToFile

        for scores, name in zip(
            [fr_std_ratio_clean, fr_std_ratio_noisy, fr_std_over_fr_mean, fr_mod_vs_mean, fr_trialstd_vs_mean],
            ["fr_std_ratio_clean", "fr_std_ratio_noisy", "fr_std_over_fr_mean", "fr_mod_vs_mean", "fr_trialstd_vs_mean"]):

            tmp = [(score, chan) for score, chan in zip(scores, chans)]
            score_chan_sorted = sorted(tmp, key=lambda x: x[0])
            score_chan_sorted_str = [stringify_list(x, return_as_str=True, separator="  --  ") for x in score_chan_sorted]
            
            fname = f"{savedir}/{bregion}-{name}.txt"
            writeStringsToFile(fname, score_chan_sorted_str)

        # Plot each chan
        if PLOT:
            from pythonlib.tools.plottools import share_axes
            sdir = f"{savedir}/{bregion}"
            os.makedirs(sdir, exist_ok=True)
            for chan in pa.Chans:
                fig, axes = plt.subplots(2, 2, figsize=(10,10))
                for pa, event, ax in zip(list_pa, list_event, axes.flatten()):
                    pa.plotwrapper_smoothed_fr_split_by_label("trials", "seqc_0_shape", chan=chan, ax=ax)
                    # ax.set_ylim(ymin=0)
                    ax.axhline(0)
                    ax.set_title(event)
                share_axes(axes, "y")
                savefig(fig, f"{sdir}/chan={chan}.pdf")
                plt.close("all")
        
        ### Decide which chans to keep
        a = fr_std_ratio_clean > THRESH_FR_RATIO_CLEAN
        b = fr_std_ratio_noisy > THRESH_FR_RATIO_NOISY
        c = fr_std_over_fr_mean > THRESH_FR_STD_MEAN
        d = fr_mod_vs_mean > THRESH_FR_MOD_VS_MEAN
        e = fr_trialstd_vs_mean > THRESH_FR_TRIALSTD_MEAN

        inds = a | b | c | d | e
        chans_keep = [chans[i] for i in np.argwhere(inds)[:,-1]]
        print("Keep, for ", bregion, " ...", sum(inds), "/", len(inds))
        MAP_REGION_TO_CHANS_KEEP[bregion] = chans_keep

    # print("----------")
    # for pa in DFallpa["pa"]:
    #     print(pa.Chans)
    # print(MAP_REGION_TO_CHANS_KEEP)
    # print(DFallpa)
    # print("----------")
    
    # Prune DFallpa to keep just those chans
    list_pa = []
    for i, row in DFallpa.iterrows():
        bregion = row["bregion"]
        pa = row["pa"]
        # print("----")
        # print(bregion)
        # print(pa.Chans)
        # print(MAP_REGION_TO_CHANS_KEEP[bregion])
        pathis = pa.slice_by_dim_values_wrapper("chans", MAP_REGION_TO_CHANS_KEEP[bregion])
        list_pa.append(pathis)
    DFallpa["pa"] = list_pa    


def dfpa_concat_normalize_fr_split_multbregion_flex(DFallpa, fr_mean_subtract_method = "across_time_bins", PLOT = False):
    """
    For each bregion, concat across events (in time dimension) and run normalization, and then split back.

    This (flex) is bettern than dfpa_concat_normalize_fr_split_multbregion, becuase the latter requires same n trials 
    across PA. Here does not.

    Works by finding the sinhgle rescale and subtract factors after concating X across PA. then applies those factors.

    Steps:
    1. Always rescales FR (soft)
    2. Optaionlly then cneters the FR (based on fr_mean_subtract_method)

    PARAMS:
    - fr_mean_subtract_method, str, how to center the FR
    
    RETURNS:
    - Modifies each pa in DFallpa (changing pa.X to be normalized), without changing anything else.
    """
    from neuralmonkey.analyses.state_space_good import _popanal_preprocess_normalize_softzscore_raw

    assert PLOT == False, "too many plots"

    for pa in DFallpa["pa"].tolist():
        assert np.all(pa.X>=0), "Detected that FR has already been normalized (centerized). Cannot run this again."

    # (1) First, do soft normalziation of FR

    list_bregion = DFallpa["bregion"].unique().tolist()
    for bregion in list_bregion:
        print("Running .. ", bregion)
            
        # (1) Collect rescale and center across all PA (order is first rescale, then centerize)
        # Determine mean and rescale factor
        list_pa = DFallpa[DFallpa["bregion"] == bregion]["pa"].tolist()
        X = np.concatenate([pa.X.reshape(pa.X.shape[0], -1) for pa in list_pa], axis=1) # (nchans, trials*times)
        _, DENOM, CENTER = _popanal_preprocess_normalize_softzscore_raw(X)

        # Apply this same DENOM and CENTER to all pa
        for pa in list_pa:
            if PLOT:
                pa.plotNeurHeat(0)

            pa.X = pa.X/DENOM[:, :, None]
            
            if PLOT:
                pa.plotNeurHeat(0)
            
            # Method for subtracting FR
            if fr_mean_subtract_method == "each_time_bin":
                # Can do this in PA independent of other PA
                _pa = pa.norm_subtract_trial_mean_each_timepoint()
                pa.X = _pa.X # becuase it returns copy
            elif fr_mean_subtract_method == "across_time_bins":
                # use global mean across all PA
                pa.X = pa.X - CENTER[:, :, None]
            else:
                assert fr_mean_subtract_method is None, "what is it?"
            
            if PLOT:
                pa.plotNeurHeat(0)


def dfpa_concat_pca_split_multbregion(DFallpa, sm_dur=0.1, sm_slide=0.01,
                                      npcs_keep_force=15,
                                      twind = None,
                                      pca_method = "trials",
                                      pcamean_var = None, pcamean_vars_grouping = None):
    """
    Project data to a single PC space across all events, for each bregion.

    For each bregion, concat, along times (retaining chans and trials), then run PCA, on (chans, -1) shape,
    then split back again into a new column in DFallpa, called pa_pca.
    
    Optionally smooth data before pca
    
    NOTE: only run once, and cannot rerun if fails, since does partial mods...

    PARAMS:
    - sm_dur, sm_slide, params for smoothing. make None to skip.
    - npcs_keep_force, take this many PCs
    - twind, (t1, t2), for slicing out data before concating
    - pca_method, str, see within.
    [if pca_method=="trial_means"]
    - pcamean_var, str, which variable to take mean over trials for.
    - pcamean_vars_grouping, grouping, within each group will take means for eahc level of pcamean_var

    RETURNS:
    - Modifies DFallpa, to add new column pa_pca, holding copy of pa, with data in pc space
    """
    PLOT=False

    # First, prep by adding pa_pca copy
    DFallpa["pa_pca"] = [pa.copy() for pa in DFallpa["pa"]]

    def preprocess_pa(pa):
        if twind is not None:
            pa = pa.slice_by_dim_values_wrapper("times", twind)
        else:
            pa = pa.copy()

        if sm_dur is not None:
            pa = pa.agg_by_time_windows_binned(sm_dur, sm_slide)
        
        return pa

    list_bregion = DFallpa["bregion"].unique().tolist()
    for bregion in list_bregion:
        print("Running .. ", bregion)

        # Slice out this bregion
        dfallpa = DFallpa[DFallpa["bregion"] == bregion].reset_index(drop=True)

        # (0) Do smoothing here first, becuase after concat will lost time dim information
        dfallpa["pa_sm"] = [preprocess_pa(pa) for pa in dfallpa["pa"]] # smoothed

        # (1) Concat all the PA into a singel pa
        DFALLPA = dfpa_group_and_split(dfallpa, ["event"], concat_dim="times", pa_column="pa_sm")

        # Plot example heatmaps, for sanity check comaprison with post
        if PLOT:
            for pa in DFALLPA["pa"]:
                pa.plotNeurHeat(0)

        # Pull out the single PA
        if len(DFALLPA)!=1:
            print(DFALLPA)
            assert False
        PA = DFALLPA["pa_sm"].values[0]

        # Do PCA
        if pca_method == "trials":
            # Data are trials.
            tbin_dur = "ignore"
            tbin_slide = None
            if True:
                # To plot things for each pc run
                plot_pca_explained_var_path = f"/tmp/plot_pca_explained_var_path-{bregion}.pdf"
                plot_loadings_path = f"/tmp/plot_loadings_path-{bregion}.pdf"
            else:
                plot_pca_explained_var_path = None
                plot_loadings_path = None
            norm_subtract_single_mean_each_chan = True
            _, PAfinal, _, _, _ = PA.dataextract_state_space_decode_flex(None, tbin_dur, tbin_slide, 
                                                                    "chans_x_trials_x_times",
                                                                    pca_reduce=True,
                                                                    plot_pca_explained_var_path=plot_pca_explained_var_path, 
                                                                    plot_loadings_path=plot_loadings_path,
                                                                    norm_subtract_single_mean_each_chan=norm_subtract_single_mean_each_chan,
                                                                    npcs_keep_force=npcs_keep_force)
            plt.close("all")
        elif pca_method == "trial_means":
            # first take mean over trials to group by a variable (e.g., shape), then do PCA as you would above.

            savedir_plots = f"/tmp/pca_plots"
            os.makedirs(savedir_plots, exist_ok=True)

            var_pca = pcamean_var
            vars_grouping = pcamean_vars_grouping
            n_pcs_subspace_max = npcs_keep_force

            # Fixed params
            raw_subtract_mean_each_timepoint = False
            pca_subtract_mean_each_level_grouping = True
            n_min_per_lev_lev_others = 4
            prune_min_n_levs = 2

            # PCA time window -- use specific smaller window for fitting PC
            # pca_twind = pcamean_pca_twind
            pca_twind = None
            pca_tbindur = "ignore"
            pca_tbin_slice = None

            # Data projection time window -- keep larger window for data
            # print(PA.Times)
            # proj_twind = (PA.Times[0]-0.1, PA.Times[-1]+0.1) # use entire window
            proj_twind=None
            proj_tbindur = "ignore"
            proj_tbin_slice = None

            _, PAfinal, _, _, _ = PA.dataextract_pca_demixed_subspace(var_pca, vars_grouping,
                                                            pca_twind, pca_tbindur,
                                                            savedir_plots=savedir_plots,
                                                            raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint,
                                                            pca_subtract_mean_each_level_grouping=pca_subtract_mean_each_level_grouping,
                                                            n_min_per_lev_lev_others=n_min_per_lev_lev_others, prune_min_n_levs = prune_min_n_levs,
                                                            n_pcs_subspace_max = n_pcs_subspace_max,
                                                            do_pca_after_project_on_subspace=False,
                                                            PLOT_STEPS=False, SANITY=False,
                                                            reshape_method="chans_x_trials_x_times",
                                                            pca_tbin_slice=pca_tbin_slice, 
                                                            proj_twind = proj_twind, proj_tbindur = proj_tbindur, 
                                                            proj_tbin_slice = proj_tbin_slice)
            
            plt.close("all")
        else:
            print(pca_method)
            assert False

        ### Split the neural activity back into separate events

        # Find the split indices
        ons = []
        offs = []
        for pa in dfallpa["pa_sm"]:
            if len(ons)==0:
                ons = [0]
                offs = [len(pa.Times)]
            else:
                ons.append(ons[-1] + len(pa.Times))
                offs.append(offs[-1] + len(pa.Times))

        # Do the split
        pa_combined = PAfinal
        for i, j, pa_sm, pa_pca in zip(ons, offs, dfallpa["pa_sm"].tolist(), dfallpa["pa_pca"].tolist()):
            x = pa_combined.X[:,:,i:j] # sliced data

            if not x.shape[1:] == pa_sm.X.shape[1:]:
                print(x.shape)
                print(pa_sm.X.shape[1:])
                assert False

            # Modiofuy the copy of the original that holds pca data
            pa_pca.replace_X(x, pa_sm.Times, list(range(npcs_keep_force)))


def dfpa_concat_normalize_fr_split_multbregion(DFallpa, fr_normalization_method ="across_time_bins", PLOT = False):
    """
    For each bregion, concat across events (in time dimension) and run normalization, and then split back.
    
    RETURNS:
    - Modifies each pa in DFallpa (changing pa.X to be normalized), without changing anything else.
    """
    list_bregion = DFallpa["bregion"].unique().tolist()
    for bregion in list_bregion:
        print("Running .. ", bregion)
        _DFallpa = DFallpa[DFallpa["bregion"] == bregion].reset_index(drop=True)
        dfpa_concat_normalize_fr_split_singlebregion(_DFallpa, fr_normalization_method, PLOT)
        

def dfpa_concat_normalize_fr_split_singlebregion(DFallpa, fr_normalization_method ="across_time_bins", PLOT = False):
    """
    Concatenate all pa across all rows of DFallpa on time dimension (so that chans and tirals stay same) into a single
    PA. Run fr normalization on this single PA, then split back into the original PA and update their firing rates.
    
    Useful if rows of DFallpa correspond to different events for the same bregion, and you want to ensure that normalizsation
    applies same to all events.

    HERE aserts that only one bregion across all rows, as this concatenates them all into one.

    RETURNS:
    - modifies DFallpa in place (i.e,, only changes each pa.X)
    """

    assert len(DFallpa["bregion"].unique())==1, "if multipel bregions, use dfpa_concat_normalize_fr_split_multbregion"
    
    if PLOT:
        print("Plots in order: (1) N pa before norm; (2) 1 pa after concat, before norm (3) 1 pa, after norm (4) N pa after splitting.")
        print("Sanity check --> compare (1) and (4) make sure they are the same activiyt, just normed.")


    # Plot example heatmaps, for sanity check comaprison with post
    if PLOT:
        for pa in DFallpa["pa"]:
            pa.plotNeurHeat(0)

    # (1) Concat all the PA into a singel pa
    from neuralmonkey.classes.population_mult import dfpa_group_and_split
    DFALLPA = dfpa_group_and_split(DFallpa, ["event"], concat_dim="times")

    # Plot example heatmaps, for sanity check comaprison with post
    if PLOT:
        for pa in DFALLPA["pa"]:
            pa.plotNeurHeat(0)

    # Do fr normalization
    from neuralmonkey.classes.population_mult import dfallpa_preprocess_fr_normalization
    dfallpa_preprocess_fr_normalization(DFALLPA, fr_normalization_method=fr_normalization_method)

    # Plot example heatmaps, for sanity check comaprison with post
    if PLOT:
        for pa in DFALLPA["pa"]:
            pa.plotNeurHeat(0)

    # Split the neural activity
    ons = []
    offs = []
    for pa in DFallpa["pa"]:
        len(pa.Times)
        if len(ons)==0:
            ons = [0]
            offs = [len(pa.Times)]
        else:
            ons.append(ons[-1] + len(pa.Times))
            offs.append(offs[-1] + len(pa.Times))

    pa_combined = DFALLPA["pa"].values[0]
    assert len(ons)==len(offs)==len(DFallpa)
    for i, j, pa in zip(ons, offs, DFallpa["pa"].tolist()):
        x = pa_combined.X[:,:,i:j]
        assert x.shape == pa.X.shape
        pa.X = x.copy()

    # Plot example heatmaps, to compare with pre
    if PLOT:
        for pa in DFallpa["pa"]:
            pa.plotNeurHeat(0)

def dfpa_match_chans_across_pa_each_bregion(DFallpa):
    """
    For each bregion, all pa in DFallpa will havea the identical chans, which are the intersection of chans
    across them.

    RETURNS:
    - Modifies DFallpa, replacing each pa with new (pa copies) that have the pruned chans, and also
    the column pa_x_shape, with the new size.
    """
    map_bregion_to_chans = {}
    for bregion in DFallpa["bregion"].unique():
        # keep only channels that are common across all pa
        _dfallpa = DFallpa[DFallpa["bregion"] == bregion]

        # Find chans common across pa
        chans_all = None
        for pa in _dfallpa["pa"]:
            print(bregion, " ... ", len(pa.Chans))
            if chans_all is None:
                chans_all = pa.Chans
            else:
                chans_all = [ch for ch in chans_all if ch in pa.Chans]
        
        map_bregion_to_chans[bregion] = chans_all
        print(bregion, " -- n chans final: ", len(chans_all))

    # Replace PA for each row of DFallpa
    list_pa = []
    list_pa_x_shape = []
    for i, row in DFallpa.iterrows():
        chans = map_bregion_to_chans[row["bregion"]]
        pa = row["pa"]
        panew = pa.slice_by_dim_values_wrapper("chans", chans)
        list_pa.append(panew)
        list_pa_x_shape.append(panew.X.shape)

    DFallpa["pa"] = list_pa
    DFallpa["pa_x_shape"] = list_pa_x_shape
        
def pa_split_into_multiple_event_timewindows_flex(PA, list_ev_tw,
                                                  how_deal_with_different_time_values="replace_with_first_pa_realigned"):
    """
    Split a single PA into multiple PA, each with its specific event (e.g., 03_samp) and time
    window (sliced into PA), and the concatenate them into a single PA along the "trials" axis.
    
    Useful if want to analyze similarity of representations across time, and you have PA that have already
    concatenated across events (i.e, the output of dfpa_group_and_split).

    Note; This can also work for unmodified PA, byut would have to chagne some hard-coded stuff, such as 
    "event_orig" below.

    Note: hard coded for 2 event_times, but could be expanded by changing code.
    
    PARAMS:
    - ev_tw_1, tuple([event_orig, twind]), where event_orig is string like "03_samp" and twind is 2-tyuple (e.g, (-0.2, -0.1))) 
    defining time window to keep. 
    - ev_tw_2, same as 1. NOTE: must have same duration time window or will fail.
    
    e.g.,:
    ev_tw_1 = ("03_samp", (-0.4, -0.2))
    ev_tw_2 = ("06_on_strokeidx_0", (0.1, 0.3))

    RETURNS:
    - PA copy, with shape like input but time just within the twinds, and evens just thoise evnets., and 
    times repalced with udummy var 0,1,2 ... And appends a column called "event_times" which is the unique conjunction
    of ev_tw
    """
    from neuralmonkey.classes.population import concatenate_popanals, concatenate_popanals_flexible

    assert "event_orig" in PA.Xlabels["trials"], "need to either run dfpa_group_and_split first, or modify code to not have this hard-coded."

    list_pa = []
    _shape = None
    for ev_tw in list_ev_tw:
        assert isinstance(ev_tw[0], str)
        assert isinstance(ev_tw[1], tuple)
        
        # Pull out PAs for each event
        pa = PA.slice_by_labels_filtdict({"event_orig":[ev_tw[0]]}) # keep event
        pa = pa.slice_by_dim_values_wrapper("times", ev_tw[1], time_keep_only_within_window=False) # keep time wind
        pa.Xlabels["trials"]["event_times_str"] = f"{ev_tw[0]}|{ev_tw[1][0]:.3f}|{ev_tw[1][1]:.3f}"

        if _shape is not None:
            if not _shape == pa.X.shape:
                print(_shape)
                print(pa.X.shape)
                print(ev_tw)
                assert False, "come up with better way to ensure same num time bins."

        _shape = pa.X.shape
        
        list_pa.append(pa)

    # concatenate them
    pa, twind = concatenate_popanals_flexible(list_pa, "trials", how_deal_with_different_time_values)

    return pa, twind


def pa_split_into_multiple_event_timewindows(PA, ev_tw_1, ev_tw_2):
    """
    Split a single PA into multiple PA, each with its specific event (e.g., 03_samp) and time
    window (sliced into PA), and the concatenate them into a single PA along the "trials" axis.
    
    Useful if want to analyze similarity of representations across time, and you have PA that have already
    concatenated across events (i.e, the output of dfpa_group_and_split).

    Note; This can also work for unmodified PA, byut would have to chagne some hard-coded stuff, such as 
    "event_orig" below.

    Note: hard coded for 2 event_times, but could be expanded by changing code.
    
    PARAMS:
    - ev_tw_1, tuple([event_orig, twind]), where event_orig is string like "03_samp" and twind is 2-tyuple (e.g, (-0.2, -0.1))) 
    defining time window to keep. 
    - ev_tw_2, same as 1. NOTE: must have same duration time window or will fail.
    
    e.g.,:
    ev_tw_1 = ("03_samp", (-0.4, -0.2))
    ev_tw_2 = ("06_on_strokeidx_0", (0.1, 0.3))

    RETURNS:
    - PA copy, with shape like input but time just within the twinds, and evens just thoise evnets., and 
    times repalced with udummy var 0,1,2 ... And appends a column called "event_times" which is the unique conjunction
    of ev_tw
    """
    from neuralmonkey.classes.population import concatenate_popanals

    assert "event_orig" in PA.Xlabels["trials"], "need to either run dfpa_group_and_split first, or modify code to not have this hard-coded."

    # Pull out PAs for each event
    pa1 = PA.copy()
    pa1 = pa1.slice_by_labels_filtdict({"event_orig":[ev_tw_1[0]]}) # keep event
    pa1 = pa1.slice_by_dim_values_wrapper("times", ev_tw_1[1], time_keep_only_within_window=False) # keep time wind
    pa1.Xlabels["trials"]["event_times_str"] = f"{ev_tw_1[0]}|{ev_tw_1[1][0]:.3f}|{ev_tw_1[1][1]:.3f}"

    pa2 = PA.copy()
    pa2 = pa2.slice_by_labels_filtdict({"event_orig":[ev_tw_2[0]]}) # keep event
    pa2 = pa2.slice_by_dim_values_wrapper("times", ev_tw_2[1], time_keep_only_within_window=False) # keep time wind
    pa2.Xlabels["trials"]["event_times_str"] = f"{ev_tw_2[0]}|{ev_tw_2[1][0]:.3f}|{ev_tw_2[1][1]:.3f}"

    # print(pa1.X.shape)
    # print(pa2.X.shape)

    if not pa1.X.shape ==pa2.X.shape:
        # print(pa1.X.shape)
        # print(pa2.X.shape)
        # print(pa1.Times)
        # print(pa2.Times)
        # assert False, "probably mismatch in times??"
        return None

    # concatenate them
    pa = concatenate_popanals([pa1, pa2], "trials", replace_times_with_dummy_variable=True)

    return pa



    # ### [OBSOLETE] Devo - methods to combine acorss different events and time windows... (e.g., visual vs. motor...)
    # # Forst, for each PA, make sure it has info about event, twind, and bregion
    # for key, pa in DictEvBrTw_to_PA.items():
    #     ev, br, tw = key
    #     pa.Xlabels["trials"]["event"] = ev
    #     pa.Xlabels["trials"]["bregion"] = br
    #     n = len(pa.Xlabels["trials"])
    #     pa.Xlabels["trials"]["twind"] = [tw for _ in range(n)]
    #
    #     print(pa.X.shape)
    # from neuralmonkey.classes.population import concatenate_popanals
    #
    # # make sure all twinds are same length, with same time from alignment.
    # if False: # DONT NEED TO RUN. this was hacky solution. code is fixed.
    #     for k, pa in DictEvBrTw_to_PA.items():
    #         tw = k[2]
    #         pa = pa.slice_by_dim_values_wrapper("times", tw)
    #         DictEvBrTw_to_PA[k] = pa
    #
    # def convert_dict_pa_to_only_bregion_twind_keys(DictEvBrTw_to_PA, event):
    #     """
    #     """
    #     # collect all that have this event
    #     DictBregionTwindPA = {}
    #     for key, pa in DictEvBrTw_to_PA.items():
    #         ev, br, tw = key
    #         if ev==event:
    #             bregion_twind = (br, tw)
    #             assert bregion_twind not in DictBregionTwindPA.keys()
    #             DictBregionTwindPA[bregion_twind] = pa
    #     return DictBregionTwindPA
    #
    # convert_dict_pa_to_only_bregion_twind_keys(DictEvBrTw_to_PA, "03_samp")
    #
    # def _extract_concatted_pa(DictEvBrTw_to_PA, events=None, bregions=None, twinds=None):
    #     """
    #     Returns a single pa concatting. If any dimejsions are None, then takes
    #     all.
    #     NOTE: This concats across the pa dim of "trials". therefore must have same
    #     num chans and timepoints. This means, for now, can only pick one brain region.
    #     :param events:
    #     :param bregions:
    #     :param twinds:
    #     :return:
    #     """
    #
    #     assert isinstance(bregions, (list, tuple)) and len(bregions)==1, "see docs above."
    #
    #     list_pa = []
    #     list_twinds =[]
    #     # collect all that have this event
    #     for key, pa in DictEvBrTw_to_PA.items():
    #         ev, br, tw = key
    #         # bregion_twind = key[1:3]
    #         # event = key[0]
    #         if events is not None and ev not in events:
    #             continue
    #         if bregions is not None and br not in bregions:
    #             continue
    #         if twinds is not None and tw not in twinds:
    #             continue
    #
    #         assert pa.Xlabels["trials"]["event"].unique()[0]==ev, "sanity"
    #         list_pa.append(pa)
    #         list_twinds.append(tw)
    #
    #     # if you are combining multiple times, then replace times iwth a
    #     # dummy variable
    #     replace_times_with_dummy_variable = len(set(list_twinds))>1
    #
    #     assert len(list_pa)>0, "didnt get any data"
    #     return concatenate_popanals(list_pa, "trials",
    #                                 replace_times_with_dummy_variable=replace_times_with_dummy_variable)
    #
    # _extract_concatted_pa(DictEvBrTw_to_PA, events=None, bregions=["PMv_m"], twinds=None)
    #
    #
    # ##### Methods to collect specific slices across popanals
    # list_bregions = SP.bregion_list()
    # # Method 1 - pick specific conjunction of even and twind
    # list_events = ['03_samp', '06_on_strokeidx_0']
    # list_twinds = [(0.3, 0.5), (-0.3, -0.1)]
    # # list_twinds = [(0.1, 0.3), (0.1, 0.3)]
    # # list_events = ['04_go_cue', '06_on_strokeidx_0']
    # # list_twinds = [(-0.3, -0.1), (-0.1, 0.1)]
    #
    # for br in list_bregions:
    #
    #     # one pa for this bregion
    #     list_pa = []
    #     for ev, tw in zip(list_events, list_twinds):
    #         pa = _extract_concatted_pa(DictEvBrTw_to_PA, events=[ev], bregions=[br], twinds=[tw])
    #         assert pa.Xlabels["trials"]["event"].unique()[0]==ev, "sanity"
    #         list_pa.append(pa)
    #
    #     key = (br, (-99, 99))
    #     DictBregionTwindPA[key] = concatenate_popanals(list_pa, "trials", replace_times_with_dummy_variable=True)
    #
    #
    #
    # # Method 2 - ONLY WORKS if using same time window for each event
    #
    # # key_name_prune_event = False
    #
    # # For each bregion, twind, collect across all the events for it
    #
    # list_events_collect = ["03_samp", "06_on_strokeidx_0"]
    #
    # DictBregionTwindPA = {}
    # list_pa = []
    # # collect all that have this event
    # for key, pa in DictEvBrTw_to_PA.items():
    #     bregion_twind = key[1:3]
    #     event = key[0]
    #     if event in list_events_collect:
    #         if bregion_twind in DictBregionTwindPA.keys():
    #             DictBregionTwindPA[bregion_twind].append((event, pa))
    #         else:
    #             DictBregionTwindPA[bregion_twind] = [(event, pa)]
    #
    # # Concat
    # for br_tw, val in DictBregionTwindPA.items():
    #     list_ev = [v[0] for v in val]
    #     list_pa = [v[1] for v in val]
    #     assert list_ev==list_events_collect
    #
    #     # map_idxpa_to_value = {i:ev for i, ev in enumerate(list_ev)}
    #     map_idxpa_to_value = list_ev
    #     map_idxpa_to_value_colname = "event"
    #     DictBregionTwindPA[br_tw] = concatenate_popanals(list_pa, "trials",
    #                                                      map_idxpa_to_value=map_idxpa_to_value,
    #                                                      map_idxpa_to_value_colname=map_idxpa_to_value_colname)
    #
    #
    #
    # pa = DictBregionTwindPA[key]
    # pa.Xlabels["trials"]["event"].value_counts()

def data_extract_raw_and_save(DFallpa, savepath):
    """
    Helper to extrat raw data (so that no dependencies on PA libarray) and
    save to savepath as pickle.
    :param DFallpa:
    :param savepath, usually should have "DFallpa_raw.pkl"
    :return:
    """
    import pandas as pd
    
    # 1. Extract raw data for each row
    datas = []
    chans = []
    times = []
    trials = []
    labels = []
    for pa in DFallpa["pa"]:
        d = pa.extract_activity_copy_all()
        datas.append(d["X"])
        chans.append(d["Chans"])
        times.append(d["Times"])
        trials.append(d["Trials"])
        labels.append(d["dflab"])

    # 2. Make a copy and append new columns
    DFallpa_raw = DFallpa.copy()
    DFallpa_raw["DATA_chan_trial_time"] = datas
    DFallpa_raw["chans"] = chans
    DFallpa_raw["times"] = times
    DFallpa_raw["trials"] = trials
    DFallpa_raw["labels"] = labels

    # 3. Drop columns that use PA class
    DFallpa_raw = DFallpa_raw.drop(["pa", "pa_x_shape"], axis=1)

    # 4. Remove label columns which use custom class objects

    list_labels = []
    did_print = False
    for dflab in DFallpa_raw["labels"]:
        keys_remove = []
        for k in dflab.keys():
            # print(k, " -- ", type(dflab[k].values[0]))
            if k in ["TokTask", "Tktask", "TokTask", "TokBeh", "Tkbeh_stktask", "Tkbeh_stkbeh"]:
                keys_remove.append(k)
        print("Removing these custom class columns from labels: ", keys_remove)
        dflab_raw = dflab.drop(keys_remove, axis=1)

        # Also convert Strokes to strokes
        if "Stroke" in dflab_raw.columns:
            list_stroke = []
            # list_tok_task = []
            for i, row in dflab_raw.iterrows():
                list_stroke.append(row["Stroke"]())
                # list_tok_task.append(row["TokTask"].data_extract_raw())
            dflab_raw["stroke"] = list_stroke
            dflab_raw = dflab_raw.drop(["Stroke"], axis=1)

        # Rest index
        dflab_raw = dflab_raw.reset_index(drop=True)

        list_labels.append(dflab_raw)

        if not did_print:
            # Print the final labels
            # - First, print the types of each column
            # dflab = DFallpa_raw["labels"].values[0]
            print("Final kept labels... ")
            columns = sorted(dflab_raw.columns)
            for col in columns:
                print(col, "   ====   ", type(dflab_raw[col].values[0]))
            did_print = True

    DFallpa_raw["labels"] = list_labels

    # 5. Save
    import pickle
    # # path = "/gorilla4/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa.pkl"
    # path = "/home/lucas/Dropbox/SCIENCE/FREIWALD_LAB/DATA/DFallpa.pkl"
    pd.to_pickle(DFallpa_raw, savepath)
    print("Saved to path:", savepath)
