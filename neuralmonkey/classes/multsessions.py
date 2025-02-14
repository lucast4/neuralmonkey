""" A collection of sessions, 
Does not touch the raw data (therfore Session is the interface between this processed
and raw data)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

class MultSessions(object):
    """docstring for MultSessions"""
    def __init__(self, list_sessions):
        """
        PARAMS:
        - list_sessions, list of Session objects.
        """

        assert len(list_sessions)>0
        
        self.SessionsList = list_sessions
        self.Datasetbeh = None
        self._generate_index() 


    ################ PARAMS
    def spikes_version(self):
        """
        REturn the common spikes version across sn. fails if they are different
        """
        tmp = [sn.SPIKES_VERSION for sn in self.SessionsList]
        assert len(set(tmp))==1, "diff spike versions..."
        return tmp[0]
        
    def animal(self):
        """ return the common animal
        if different animals, then error
        """
        animals = list(set([sn.Animal for sn in self.SessionsList]))
        assert len(animals)==1
        return animals[0]


    def date(self):
        """ return the common date
        if different dates, then error
        """

        dates = list(set([sn.Date for sn in self.SessionsList]))
        assert len(dates)==1
        return dates[0]


    ################ Indexing
    def _generate_index(self):
        """ run once to generate indexes
        """

        # Map from trialcode to session
        self._MapperTrialcode2SessTrial = {}
        self._MapperInd2Trialcode = {}
        ind = 0
        for sessnum, SN in enumerate(self.SessionsList):
            for trialcode, trial in SN._MapperTrialcode2TrialToTrial.items():
                self._MapperTrialcode2SessTrial[trialcode] = (sessnum, trial)
                self._MapperInd2Trialcode[ind] = trialcode
                ind+=1
        print("Generated index mappers!")

    # def index_convert(self, index, output="sn_trial"):
    def index_convert_trial_trialcode_flex(self, index, output="sn_trial"):
        """ 
        Flexible, helps convert multiple possible ways of indexing trials into 
        Session object and trial within that Session.
        PARAMS:
        - index, different possibilites:
        --- ind, int counts from 0 to N, for N trials, in temporal order
        --- (sess num, trial_within_sess), tuple
        --- trialcode, str, date-<beh_session>-<beh_trial>
        RETURN:
        - Depends on output:
        --- [if output=="sn_trial"], Session, trial in sess, sessnum
        --- [if output=="trialcode"], trialcode
        """

        if isinstance(index, int):
            # incremental index over entire lsti of sessions.
            trialcode = self._MapperInd2Trialcode[index]
            sessnum, trial_in_sess = self._MapperTrialcode2SessTrial[trialcode]
        elif isinstance(index, str):
            # trialcode
            sessnum, trial_in_sess = self._MapperTrialcode2SessTrial[index]
        elif isinstance(index, tuple):
            # (sessnum trial)
            assert len(index)==2
            sessnum, trial_in_sess = index
        else:
            print(index)
            assert False, "code it"

        SN = self.SessionsList[sessnum]
        return SN, trial_in_sess, sessnum

    def session_extract_largest(self):
        """ Get the sn with the most trials"""
        # pull out the largest sn
        list_n = []
        for i, sn in enumerate(self.SessionsList):
            list_n.append(len(sn.Datasetbeh.Dat))
        i = list_n.index(max(list_n))
        sn = self.SessionsList[i]
        return sn


    ########### DATASET beh
    def datasetbeh_extract(self):
        """ Get concatenated dataset, 
        First time will extract and cache.
        RETURNS:
        - D, concatted dataset
        """
        from pythonlib.dataset.analy_dlist import concatDatasets

        if self.Datasetbeh is None:
            # Load it for one time.

            if len(self.SessionsList)==1:
                # Just use the one.
                D = self.SessionsList[0].Datasetbeh
            else:
                from pythonlib.dataset.dataset import load_dataset_daily_helper

                # Get for this day.
                # Better than concatting -- load from scratch, can't trust concatDatasets, sometimes for cases
                # where use prims labeled by cluster, it changes...
                # Merge datasets
                D = load_dataset_daily_helper(self.animal(), self.date())

                # prune to just the trialcodes that exist
                trialcodes = []
                for sn in self.SessionsList:
                    trialcodes.extend(sn.Datasetbeh.Dat["trialcode"].unique().tolist())
                trialcodes = list(set(trialcodes))

                D.Dat = D.Dat[D.Dat["trialcode"].isin(trialcodes)].reset_index(drop=True)

                # Dlist = []
                # for sn in self.SessionsList:
                #     Dlist.append(sn.Datasetbeh)
                # D = concatDatasets(Dlist)

            self.Datasetbeh = D

        return self.Datasetbeh

    #################### UTILS
    def prune_remove_sessions_too_few_trials(self, min_n_trials):
        """ for each suessions removes if trials (clean) 
        (doesnt care whetehr exists in beh dataset) 
        not enough.
        RETURNS:
        - modifies self.SessionsList in place
        """
        def _good(sn):
            return len(sn.get_trials_list(True, True, False))>=min_n_trials

        print("[MS] prune_remove_sessions_too_few_trials...")
        print("Starting  num sessions: ", len(self.SessionsList))
        self.SessionsList = [sn for sn in self.SessionsList if _good(sn)]
        print("Ending num sessions: ", len(self.SessionsList))


    # def session_generate_single_nodata(self):
    #     """ Generate a single sn for use in analyses that aggregated across
    #     sessions, where sn is used for thigns that are common across
    #     sessions, such as site info, and for methods.
    #     Takes the first session, and clears all the neural data, and
    #     does sanity check that all sessions have same metadata
    #     """
        
    ################### STATS
    def sitestats_fr_extract_good(self, site, keep_within_events_flanking_trial=False):
        """
        Extract firing rate across ttrials, along with time within day
        EXAMPLE:
            sites = MS.sitegetter_all()
            MS.sitestats_fr_extract_good(sites[0])
        """

        list_fr =[]
        list_trials = []
        list_times_frac = []
        list_trialcodes = []
        sessions = []
        for i, sn in enumerate(self.SessionsList):
            frvals, trials, times_frac, trialcodes = sn.sitestats_fr_extract_good(site, 
                                                                      keep_within_events_flanking_trial=keep_within_events_flanking_trial)
            list_fr.append(frvals)
            list_trials.append(trials)
            list_times_frac.append(times_frac)
            list_trialcodes.extend(trialcodes)
            sessions.extend([i for _ in range(len(frvals))])

        frvals = np.concatenate(list_fr, axis=0)
        trials = np.concatenate(list_trials, axis=0)
        times_frac = np.concatenate(list_times_frac, axis=0)
        trialcodes = list_trialcodes

        return frvals, trials, times_frac, trialcodes, sessions

    def sitesdirtygood_preprocess_firingrate_drift(self, chan, savedir=None,
                                                   ntrials_per_bin = 50, nsigma=3.5):
        """
        Methods to score the across-day drift in FR. 
        NOTE:
        To skip plotting each trial, set savedir = None
        """
        from neuralmonkey.metrics.goodsite import score_firingrate_drift

        ### Extract data
        frvals, trials, times_frac, trialcodes, sessions = self.sitestats_fr_extract_good(chan, keep_within_events_flanking_trial=True)
        metrics, inds_bad = score_firingrate_drift(frvals, times_frac, trials, ntrials_per_bin, nsigma, 
                                                   savedir=savedir, savename=self.sitegetter_summarytext(chan))

        return frvals, trials, times_frac, metrics, inds_bad, trialcodes, sessions


    def sitesdirtygood_preprocess_wrapper(self, SAVEDIR=None, nsigma=3.5, how_combine="intersect", PLOT_EACH_TRIAL=True):
        """
        Wrapper for all things for good pruning of chans and trails within chans, based on firing rates
        stats -- outliers.
        
        For each chan, gets stats based on (fr over trials), such as drift and variance fr across blocks of trials.
        
        For each chan, get list of trials which are bad, in having mean fr higher or lower than mean trial by nsigma*STD.

        Apply threshold for different metrics, keeping only those chans that pass for all metrics.

        PARAMS:
        - how_combine, how dealw ith cases where sessions have diff chans?

        TODO:
        - deal with cases where most trials are low FR, and subset are high. Is partly dealt with by frac_trials_bad, but this
        is not best way to do ti. Should look at distribution of fr, within each slice/block of trials.
        - within-trials noisy periods -- these may escape the current, which looks just at fr for each trial. 
        """
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.pandastools import savefig
        from pythonlib.tools.checktools import check_objects_identical
        import pickle

        # Manually input thresholds
        # These based on Diego,, 240508.
        map_var_to_thresholds = {
            # "frstd_spread_index_across_bins":(0, 1),
            # "slope_over_mean":(-0.18, 0.18),
            # "fr_spread_index_across_bins":(0, 0.4),
            # "frac_trials_bad":(0, 0.012)
            "frstd_spread_index_across_bins":(0, 1.15),
            "slope_over_mean":(-0.19, 0.19),
            "fr_spread_index_across_bins":(0, 0.65),
            "frac_trials_bad":(0, 0.018)
        }

        if SAVEDIR is None:
            from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED
            SAVEDIR = f"{PATH_DATA_NEURAL_PREPROCESSED}/sitesdirtygood_preprocess/{self.animal()}-{self.date()}-combsess"
            os.makedirs(SAVEDIR, exist_ok=True)

        ### COLLECT ALL DATA        
        if PLOT_EACH_TRIAL:
            savedir = f"{SAVEDIR}/fr_over_trials"
            os.makedirs(savedir, exist_ok=True)
        else:
            savedir = None

        sites = self.sitegetter_all(how_combine=how_combine)
        res = []
        # sites = sites[:5]
        trials, times_frac, trialcodes, sessions = None, None, None, None
        for chan in sites:
            print(chan)
            
            # FR stability/drift
            frvals, _t, _tf, metrics, inds_bad, _tc, _s = self.sitesdirtygood_preprocess_firingrate_drift(
                chan, savedir, nsigma=nsigma)
            
            if trials is None:
                trials = _t
                times_frac = _tf
                trialcodes = _tc
                sessions = _s
            else:
                assert check_objects_identical(trials, _t)
                assert check_objects_identical(times_frac, _tf)
                assert check_objects_identical(trialcodes, _tc)
                assert check_objects_identical(sessions, _s)

            # FR outliers, specific trials.
            res.append({
                "chan":chan,
                "frvals":frvals,
                # "trials":trials,
                # "times_frac":times_frac,
                "inds_bad":inds_bad,
                "bregion":self.sitegetterKS_map_site_to_region(chan),
                "bregion_combined":self.sitegetterKS_map_site_to_region(chan, region_combined=True)
            })
            for name, val in metrics.items():
                res[-1][name] = val

            plt.close("all")

        ### Generate dataframe and add things to it
        dfres = pd.DataFrame(res)

        # Quantify, frac trials that are outliers
        def F(x):
            return len(x)
        dfres["n_inds_bad"] = dfres["inds_bad"].apply(F)
        n_trials = len(trials)
        dfres["frac_trials_bad"] = dfres["n_inds_bad"]/n_trials

        # Which chans pass threshodl
        # Check which chans are excluded
        keeps = None
        for var, threshes in map_var_to_thresholds.items():
            if all(dfres[var].isna()):
                continue
            _keeps = (dfres[var]>=threshes[0]) & (dfres[var]<=threshes[1])
            if keeps is None:
                keeps = _keeps
            else:
                keeps = keeps & _keeps
        dfres["good_chan"] = keeps

        #### SAVE DATA
        pd.to_pickle(dfres, f"{SAVEDIR}/dfres.pkl")
        dfres.to_csv(f"{SAVEDIR}/dfres.csv")

        params = {
            "nsigma":nsigma,
            "animal":self.animal(),
            "date":self.date(),
            "spikes_version":self.spikes_version(),
            "trials":trials,
            "chans":sites,
            "chans_good":dfres[dfres["good_chan"]]["chan"].tolist(),
            "trialcodes":trialcodes,
            "sessions":sessions,
        }
        from pythonlib.tools.expttools import writeDictToYaml, writeDictToTxtFlattened
        writeDictToYaml(params, f"{SAVEDIR}/params.yaml")
        writeDictToTxtFlattened(params, f"{SAVEDIR}/params_text.yaml")

        with open(f"{SAVEDIR}/trials.pkl", "wb") as f:
            pickle.dump(trials, f)

        with open(f"{SAVEDIR}/times_frac.pkl", "wb") as f:
            pickle.dump(times_frac, f)
            
        with open(f"{SAVEDIR}/trialcodes.pkl", "wb") as f:
            pickle.dump(trialcodes, f)

        with open(f"{SAVEDIR}/sessions.pkl", "wb") as f:
            pickle.dump(sessions, f)

        ### Plot summary across sites
        savedir = f"{SAVEDIR}/summary_figures"
        os.makedirs(savedir, exist_ok=True)

        # plot distribution over chans, of different metrics.
        vars = ["frstd_spread_index_across_bins", "slope_over_mean", "fr_spread_index_across_bins", "frac_trials_bad"]
        
        # Keep only those vars that dont have na. na is becuase not enoughd data to do bins.
        vars = [v for v in vars if ~np.all(dfres[v].isna())]
        
        # (1) val vs. site (catplot)
        for yvar in vars:
            # yvar = "fr_spread_index_across_bins"
            fig = sns.catplot(data=dfres, x="chan", y=yvar, alpha=0.5, hue="bregion", aspect=10, height=4)
            rotateLabel(fig, 70)
            savefig(fig, f"{savedir}/catplot-yvar={yvar}.pdf")

        # (2) Pairplot
        fig = sns.pairplot(data = dfres, x_vars=vars, y_vars=vars, hue="bregion", height=4)
        savefig(fig, f"{savedir}/pairplot.pdf")

        # (3) Pairplot, with text labeling chans
        # overlay with text
        ct = 0
        chans = dfres["chan"].tolist()
        assert chans == sites
        fig, axes = plt.subplots(3,3, figsize=(15, 15))
        for i in range(len(vars)):
            for j in range(len(vars)):
                if j>i:
                    var1 = vars[i]
                    var2 = vars[j]

                    ax = axes.flatten()[ct]
                    if ct==0:
                        ax.set_title("red=bad chans. red lines=thresholds")
                    ct+=1

                    # overlay thresholds
                    thresh_lower, thresh_upper = map_var_to_thresholds[var1]
                    ax.axvline(thresh_lower, color="r", alpha=0.3)
                    ax.axvline(thresh_upper, color="r", alpha=0.3)

                    thresh_lower, thresh_upper = map_var_to_thresholds[var2]
                    ax.axhline(thresh_lower, color="r", alpha=0.3)
                    ax.axhline(thresh_upper, color="r", alpha=0.3)

                    ax.axhline(0)
                    ax.axvline(0)

                    # Plot data
                    x = dfres[var1].values
                    y = dfres[var2].values
                    good_chans = dfres["good_chan"].values

                    ax.plot(x[~good_chans], y[~good_chans], ".r", alpha=0.7)
                    ax.plot(x[good_chans], y[good_chans], ".k", alpha=0.4) # label the bad ones
                    
                    for ch, xx, yy in zip(chans, x, y):
                        # if good:
                        col = 0.1 + 0.8*np.random.rand(3)
                        # else:
                        #     col = "r"
                        ax.text(xx, yy, ch, color=col, alpha=0.65, fontsize=8)

                    # Labels
                    ax.set_xlabel(var1)
                    ax.set_ylabel(var2)
        savefig(fig, f"{savedir}/pairplot-labeling_chans.pdf")
        plt.close("all")

        return dfres, params, trials, times_frac, trialcodes, sessions, SAVEDIR

    def elephant_spiketrain_from_values(self, spike_times, time_on, time_off):
        """ Convert array of times to a SpikeTrain instance
        """
        sn = self.SessionsList[0]
        return sn.elephant_spiketrain_from_values(spike_times, time_on, time_off)

    def _popanal_generate_from_raw(self, frate_mat, times, chans, df_label_trials=None,
        list_df_label_cols_get=None):
        """
        """
        sn = self.SessionsList[0]
        return sn._popanal_generate_from_raw(frate_mat, times, chans, df_label_trials, list_df_label_cols_get)

    def popanal_timewarp_rel_events_INTERPRATES(self, events=None, sites=None,
                                               predur_rel_first_event = -1, postdur_rel_last_event = 1,
                                               remove_trials_outlier_time_intervals=True, PLOT=False):
        """
        Wrapper to generate PA that has fr time-warped to fit the "median" trial, based on linear
        interpolation of firing rates between events. Each trial must have all events. 
        
        PARAMS:
        - events, list of str. if give 4 events, then will actuayl ahve 6, including 2 flankers, whose timing
        is given by predur_rel_first_event, and postdur_rel_last_event.
        RETURNS:
        - PA, with times in actual raw times. Event-info is in PA.Params
        """
 
        ### For each session extract
        res =[]
        # ind_trial_0 = 0
        # events = None
        if sites is None:
            sites = self.sitegetter_all(how_combine="intersect")
        trials = None
        for sn in self.SessionsList:
            # Extract relevant data for each session
            _events, _sites, times_array, _, dflab, list_pa = sn._popanal_generate_timewarped_rel_events_extract_raw(
                events, sites, trials, predur_rel_first_event, postdur_rel_last_event, get_dfspikes=False)

            # Check that globally these params match
            if events is None:
                events = _events
            else:
                assert events == _events
        
            if sites is None:
                sites = _sites
            else:
                assert sites == _sites

            res.append({
                # "events":events,
                # "trials":trials,
                # "sites":sites,
                "times_array":times_array,
                "list_pa":list_pa,
                "dflab":dflab,
            })

            # Shift the ind_trials so that they reflect the global order of trials in times_array.
            # ind_trials_global = list(range(ind_trial_0, ind_trial_0 + times_array.shape[0]))
            # dfspikes["ind_trial"] = [ind_trials_global[i] for i in dfspikes["ind_trial"]]
            # ind_trial_0 += times_array.shape[0]

        ### Collect across all sessions
        times_array = np.concatenate([x["times_array"] for x in res], axis=0) # (ntrials, nevents)
        # dfspikes = pd.concat([x["dfspikes"] for x in res], axis=0).reset_index(drop=True) # (ntrials, nevents)
        LIST_PA = []
        for r in res:
            LIST_PA.extend(r["list_pa"])
        dflab = pd.concat([x["dflab"] for x in res], axis=0).reset_index(drop=True) # (ntrials, nevents)

        ### Remove outlier trials, any trial with at least one interval that is diff from others trials.
        if remove_trials_outlier_time_intervals:
            from pythonlib.tools.pandastools import remove_outlier_values
            times_array_diff = np.diff(times_array[:, 1:-1], axis=1) # exclude the flankers.
            df = pd.DataFrame(times_array_diff)
            inds_good = remove_outlier_values(df)
            # - prune
            times_array = times_array[inds_good,:]
            LIST_PA = [LIST_PA[i] for i in inds_good]
            dflab = dflab.iloc[inds_good].reset_index(drop=True)

        # Warp each pa to the median trial
        if True:
            # Get median correctly -- first get median of each segment, and then cumsum those.
            diffs_median = np.median(np.diff(times_array, axis=1), axis=0)
            times_median = np.cumsum(diffs_median)
            anchor_times_template_all = np.insert(times_median, 0, 0.)
        else:
            # is OK, but is less correct
            anchor_times_template_all = np.median(times_array, axis=0)
        anchor_times_template = anchor_times_template_all[1:-1] # get inner times.
        period = 0.0025
        times_template = np.arange(anchor_times_template_all[0], anchor_times_template_all[-1], period)

        assert len(LIST_PA)==times_array.shape[0]
        list_pa_warp = []
        fig1 = None
        fig2 = None
        for i, pa in enumerate(LIST_PA):
            # Get inner anchor times
            anchor_times_this_trial = times_array[i, 1:-1]
            if PLOT and i==0:
                plotthis = True
            else:
                plotthis = False
            assert len(pa.Trials)==1, "must be..."
            pa_warp, _fig1, _fig2 = pa.dataextract_timewarp_piecewise_linear(anchor_times_this_trial, times_template, 
                                                        anchor_times_template, PLOT=plotthis)
            if plotthis:
                fig1 = _fig1
                fig2 = _fig2
            list_pa_warp.append(pa_warp)
        
        # Concatnaet
        from neuralmonkey.classes.population import concatenate_popanals
        PA = concatenate_popanals(list_pa_warp, "trials", 
                                assert_otherdims_have_same_values=True, 
                                assert_otherdims_restrict_to_these=("chans", "times"),
                                all_pa_inherit_times_of_pa_at_this_index=0)

        # Get beh data
        assert len(dflab)==PA.X.shape[1]
        PA.Xlabels["trials"] = dflab

        # Store bregions
        # -- NOTE: This works.. list_pa, bregions = PA.split_by_label("chans", "bregion_combined")
        res =[]
        for site in PA.Chans:
            res.append(
                {"bregion_combined": self.sitegetterKS_map_site_to_region(site, region_combined=True),
                "bregion":self.sitegetterKS_map_site_to_region(site, region_combined=False),
                "chan":site
            })


        # Plot event timings
        if PLOT:
            from pythonlib.tools.plottools import set_y_tick_labels
            fig3, ax = plt.subplots()
            times_array_diff = np.diff(times_array[1:-1], axis=1)
            times_median_diff = np.diff(anchor_times_template_all)
            events_all = ["ONSET"] + events + ["OFFSET"]
            events_pairs = [f"{e1}-{e2}" for e1, e2 in zip(events_all[:-1], events_all[1:])]

            # print(anchor_times_template_all)
            # print(times_array_diff.shape)
            # print(len(events_pairs))
            # print(events_pairs)
            for i in range(times_array_diff.shape[1]):
                nrows = times_array_diff.shape[0]
                ax.plot(times_array_diff[:, i], np.ones(nrows)*i, "x", alpha=0.25, label=events_pairs[i])
                ax.plot(times_median_diff[i], i, "o", alpha=0.7)
                # ax.set_yticklabels(range(len(events_pairs)), labels=events_pairs)

            set_y_tick_labels(ax, events_pairs)
            ax.set_xlim(left=0.)
            ax.set_xlabel("segment duration (sec)")
            ax.set_xlabel("circle = median")

            # ax.legend()
        else:
            fig3 = None

        # Finally, bin over time, to make data smaller
        if True:
            dur = 0.01
            slide = 0.005
            PA = PA.agg_by_time_windows_binned(dur, slide)

        # Store params related to the events
        PA.Params["version"] = "time_warped_to_events_INTERPRATES"
        PA.Params["event_times_array"] = times_array
        PA.Params["event_times_median"] = anchor_times_template_all
        PA.Params["events_inner"] = events
        PA.Params["events_all"] = ["ONSET"] + events + ["OFFSET"]
        PA.Params["ONSET_predur_rel_first_event"] = predur_rel_first_event
        PA.Params["OFFSET_postdur_rel_lst_event"] = postdur_rel_last_event

        # save
        PA.Xlabels["chans"] = pd.DataFrame(res)        

        return PA, fig1, fig2, fig3

    def popanal_timewarp_rel_events_SHIFTSPIKES(self, events=None, sites=None,
                                               predur_rel_first_event = -1, postdur_rel_last_event = 1,
                                               PLOT=False):
        """
        Wrapper to generate PA that has fr time-warped to fit the "median" trial, based on linear
        interpolation of spike timings between events. Each trial must have all events. 
        NOTE: this can blow up the firing rate, beucas it modfies spike time..
        PARAMS:
        - events, list of str. if give 4 events, then will actuayl ahve 6, including 2 flankers, whose timing
        is given by predur_rel_first_event, and postdur_rel_last_event.
        RETURNS:
        - PA, with times in actual raw times. Event-info is in PA.Params
        """ 

        assert False, "use INTERPRATES instead. THis works (but need to update a bit, some of the return signatures are wrong), but fr can blow up"

        ### For each session extract
        res =[]
        ind_trial_0 = 0
        events = None
        # sites = None
        sites = self.sitegetter_all(how_combine="intersect")
        for sn in self.SessionsList:
            # Extract relevant data for each session
            trials = None
            _events, _sites, times_array, dfspikes, dflab = sn._popanal_generate_timewarped_rel_events_extract_raw(
                events, sites, trials, predur_rel_first_event, postdur_rel_last_event, min_interval_dur=0.1)

            # Check that globally these params match
            if events is None:
                events = _events
            else:
                assert events == _events
        
            if sites is None:
                sites = _sites
            else:
                assert sites == _sites

            res.append({
                # "events":events,
                # "trials":trials,
                # "sites":sites,
                "times_array":times_array,
                "dfspikes":dfspikes,
                "dflab":dflab,
            })

            # Shift the ind_trials so that they reflect the global order of trials in times_array.
            ind_trials_global = list(range(ind_trial_0, ind_trial_0 + times_array.shape[0]))
            dfspikes["ind_trial"] = [ind_trials_global[i] for i in dfspikes["ind_trial"]]
            ind_trial_0 += times_array.shape[0]

        # Collect across all sessions
        times_array = np.concatenate([x["times_array"] for x in res], axis=0) # (ntrials, nevents)
        dfspikes = pd.concat([x["dfspikes"] for x in res], axis=0).reset_index(drop=True) # (ntrials, nevents)
        dflab = pd.concat([x["dflab"] for x in res], axis=0).reset_index(drop=True) # (ntrials, nevents)

        PA, fig = self._popanal_generate_timewarped_rel_events_do_warp(events, sites, times_array, dfspikes, dflab, 
                                                                predur_rel_first_event, postdur_rel_last_event, PLOT)
        
        return PA, fig

    def _popanal_generate_timewarped_rel_events_do_warp(self, events, sites, times_array, dfspikes, dflab, 
                                                                predur_rel_first_event, postdur_rel_last_event, PLOT=False):
        """
        Wrapper to generate PA that has fr time-warped to fit the "median" trial, based on linear
        interpolation between events. Each trial must have all events. 
        PARAMS:
        - events, list of str. if give 4 events, then will actuayl ahve 6, including 2 flankers, whose timing
        is given by predur_rel_first_event, and postdur_rel_last_event.
        RETURNS:
        - PA, with times in actual raw times. Event-info is in PA.Params
        """
        from elephant.kernels import GaussianKernel        
        from elephant.statistics import time_histogram, instantaneous_rate,mean_firing_rate
        from quantities import s
        from neo.core import SpikeTrain
        from neuralmonkey.classes.session import SMFR_TIMEBIN, _SMFR_SIGMA
        
        ### Prep params
        # Get median times
        assert False, "Run (1) remove_trials_outlier_time_intervals and (2) Get median correctly"
        times_median = np.median(times_array, axis=0)
        list_ind_trial = list(range(times_array.shape[0]))
        print(list_ind_trial)
        print(dfspikes)
        assert list_ind_trial == sorted(dfspikes["ind_trial"].unique())
        
        def convert_spiketime_to_timecanonical(st, ind_trial, DEBUG=False):
            """
            Given a time and a trial, convert it to common coordinate system, 0...1...2, where these are the evenst, 
            and fraction is fraction with nthe window
            """
            times = times_array[ind_trial]
            event_ind_this_time_occurs_after = np.max(np.argwhere((st-times)>0))
            time_delta = st - times[event_ind_this_time_occurs_after]
            time_interval = times[event_ind_this_time_occurs_after+1] - times[event_ind_this_time_occurs_after]
            time_delta_frac = time_delta/time_interval
            time_canonical = event_ind_this_time_occurs_after + time_delta_frac

            if DEBUG:
                print(st, times, event_ind_this_time_occurs_after, time_delta_frac)
            
            assert time_delta_frac>=0
            assert time_delta_frac<=1
            return time_canonical
                
        def _get_spike_times(site, ind_trial, spikes_version="spike_times_warped"):
            """ get spike times for this (site, trial)"""
            tmp = dfspikes[(dfspikes["site"] == site) & (dfspikes["ind_trial"] == ind_trial)]
            assert len(tmp)==1
            return tmp[spikes_version].values[0]

        ### METHODS -- Convert from canonical time to projected onto median trial event times
        times_median_canonical = np.arange(len(times_median)) # (0, 1, 2, ...)
        map_segment_to_segmentdur = {} # each segment (0,1 ,2..) to dur in sec
        for i, dur in enumerate(np.diff(times_median)):
            map_segment_to_segmentdur[i] = dur

        def project_spiketimescanonical_to_median(spike_times_canonical):
            """
            e.g., convert 0.1 (first segment, 0.1 frac within) to the actual raw time relative to the
            median trial.
            """
            spike_times_warped = np.zeros(spike_times_canonical.shape)-999

            for segment_idx in times_median_canonical[:-1]:

                # get all inds in this segment
                mask = ((spike_times_canonical-segment_idx)>=0) & ((spike_times_canonical-segment_idx)<1)

                # get their fraction within the segment, and add that to the median time onset
                fracs = (spike_times_canonical[mask] - segment_idx) # frac  
                segment_dur = map_segment_to_segmentdur[segment_idx]
                segment_onset = times_median[segment_idx]

                assert np.all(spike_times_warped[mask] == -999), "already filled..."

                spike_times_warped[mask] = segment_onset + fracs*segment_dur
            assert np.all(spike_times_warped>-999), "missed items"
            return spike_times_warped

        ### Now time-warp each spike time to canonical coordiantes then to warped.
        list_spike_times_canonical = []
        list_spike_times_warped = []
        for i, row in dfspikes.iterrows():
            ind_trial = row["ind_trial"]
            site = row["site"]
            spike_times_raw = row["spike_times_raw"]

            print("Times --> canonical, site:", site)
            spike_times_canonical = np.array([convert_spiketime_to_timecanonical(st, ind_trial) for st in spike_times_raw])

            # Project to median (final warp)
            print("Canonical --> warped, site:", site)
            spike_times_warped = project_spiketimescanonical_to_median(spike_times_canonical)

            list_spike_times_canonical.append(spike_times_canonical)
            list_spike_times_warped.append(spike_times_warped)
        dfspikes["spike_times_canonical"] = list_spike_times_canonical
        dfspikes["spike_times_warped"] = list_spike_times_warped

        # PLOT EXAMPLE
        if PLOT:
            site = sites[0]
            trials_plot = list_ind_trial[:30]

            fig, axes = plt.subplots(1, 4, figsize=(40, 8))

            ax = axes.flatten()[0]
            ax.set_title("original times")
            ax.set_xlabel(f"Site={site}")
            ax.set_ylabel("Trials (subset)")
            for ind_trial in range(len(trials_plot)):
                times = times_array[ind_trial, :]
                ax.plot(times, np.ones(len(times))*ind_trial, "ok")

                # overlay spikes
                spike_times = _get_spike_times(site, ind_trial, "spike_times_raw")
                ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

            ax = axes.flatten()[1]
            ax.set_title("original times, subtract first time")
            for ind_trial in range(len(trials_plot)):
                times = times_array[ind_trial, :]
                times = times_array[ind_trial, :]
                time_start = times[0]
                times = times - time_start
                ax.plot(times, np.ones(len(times))*ind_trial, "ok")

                # overlay spikes
                spike_times = _get_spike_times(site, ind_trial, "spike_times_raw")
                spike_times = spike_times - time_start
                ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

            ax = axes.flatten()[2]
            ax.set_title("canonical times")
            for ind_trial in range(len(trials_plot)):
                times = times_array[ind_trial, :]

                # times = times_array[ind_trial, :]
                times = times_median_canonical
                ax.plot(times, np.ones(len(times))*ind_trial, "ok")

                # overlay spikes
                spike_times = _get_spike_times(site, ind_trial, "spike_times_canonical")
                ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

            ax = axes.flatten()[3]
            ax.set_title("warped times (final)")
            for ind_trial in range(len(trials_plot)):
                times = times_array[ind_trial, :]

                # times = times_array[ind_trial, :]
                times = times_median
                ax.plot(times, np.ones(len(times))*ind_trial, "ok")

                # overlay spikes
                spike_times = _get_spike_times(site, ind_trial, "spike_times_warped")
                ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")
        else:
            fig = None

        ### Generate PA -- using warped aligned times, generate a single PA
        t_on = times_median[0]
        t_off = times_median[-1]
        list_pa = []
        for ind_trial in list_ind_trial:
            
            print("Generating PA, ind_trial: ", ind_trial)
            
            # Collect spike trains over all sites
            list_spiketrain = []
            for site in sites:
                # dat = self.datall_TDT_KS_slice_single_bysite(site, trial)
                st = _get_spike_times(site, ind_trial, spikes_version="spike_times_warped")
                list_spiketrain.append(self.elephant_spiketrain_from_values(st, t_on, t_off))
                
            # Convert spike train to smoothed FR
            frate = instantaneous_rate(list_spiketrain, sampling_period=SMFR_TIMEBIN*s, 
                kernel=GaussianKernel(_SMFR_SIGMA*s))

            # Convert to popanal
            times = np.array(frate.times)
            pa = self._popanal_generate_from_raw(frate.T.magnitude, times, sites, df_label_trials=None)
            list_pa.append(pa)

        from neuralmonkey.classes.population import concatenate_popanals
        PA = concatenate_popanals(list_pa, "trials", 
                                # assert_otherdims_have_same_values=True, 
                                # assert_otherdims_restrict_to_these=("chans", "times"),
                                assert_otherdims_have_same_values=False,   # no need, it must be by design
                                assert_otherdims_restrict_to_these=("chans", "times"),
                                all_pa_inherit_times_of_pa_at_this_index=0)

        # Get beh data
        PA.Xlabels["trials"] = dflab

        # Store params related to the events
        PA.Params["version"] = "time_warped_to_events"
        PA.Params["event_times_array"] = times_array
        PA.Params["event_times_median"] = times_median
        PA.Params["event_times_median_canonical"] = times_median_canonical
        PA.Params["events_inner"] = events
        PA.Params["events_all"] = ["ONSET"] + events + ["OFFSET"]
        PA.Params["ONSET_predur_rel_first_event"] = predur_rel_first_event
        PA.Params["OFFSET_postdur_rel_lst_event"] = postdur_rel_last_event

        # Store bregions
        # -- NOTE: This works.. list_pa, bregions = PA.split_by_label("chans", "bregion_combined")
        res =[]
        for site in PA.Chans:
            res.append(
                {"bregion_combined": self.sitegetterKS_map_site_to_region(site, region_combined=True),
                "bregion":self.sitegetterKS_map_site_to_region(site, region_combined=False),
                "chan":site
            })

        # save
        PA.Xlabels["chans"] = pd.DataFrame(res)        

        return PA, fig


    ################### SITES
    # (Generally asserts that all sessions have same channels ...)
    def sitegetter_all(self, list_regions=None, clean=True, how_combine="assert_identical"):
        """ Gets list of sites. Runs this for each sessin and checsk that 
        all are identicayul before returning
        PARAMS:
        - how_combine, if sessions have diff sites, how to combine? 
        """

        list_list_sites = []
        for SN in self.SessionsList:
            list_sites = SN.sitegetterKS_map_region_to_sites_MULTREG(list_regions, clean)
            list_list_sites.append(list_sites)

        if how_combine=="assert_identical":
            # check that all lists are same
            for i, sites1 in enumerate(list_list_sites):
                for ii, sites2 in enumerate(list_list_sites):
                    if ii>i:
                        if set(sites1)!=set(sites2):
                            print(len(sites1), sites1)
                            print(len(sites2), sites2)
                            assert False

            return list_list_sites[0]
        elif how_combine=="intersect":
            # get the intersection
            sites_intersect = list_list_sites[0]
            for sites_this in list_list_sites[1:]:
                sites_intersect = [s for s in sites_intersect if s in sites_this]
            return sites_intersect
        elif how_combine=="union":
            sites = []
            for sitesthis in list_list_sites:
                sites.extend(sitesthis)
            sites = list(set(sites))
            return sites
        else:
            print(how_combine)
            assert False

    def sitegetterKS_map_site_to_region(self, site, region_combined=False):
        """
        GFEt the region for this site. fails if sn have different region.
        """

        region = None
        for sn in self.SessionsList:
            _region = sn.sitegetterKS_map_site_to_region(site, region_combined)
            if region is None:
                region = _region
            else:
                assert region == _region
        return region

    def sitegetter_summarytext(self, chan):
        """ Return string summarizing this chan. Fails if the sn give diffewrent neamse. 
        """

        st = None
        for sn in self.SessionsList:
            _st = sn.sitegetter_summarytext(chan)
            if st is None:
                st = _st
            else:
                assert st == _st, f"{st} vs {_st}"
        
        return st

    def print_summary_sessions(self):
        """ Helper to print summary of n trial and sites for each sn
        """
        print("=== N trials per session")
        for i, sn in enumerate(self.SessionsList):
            print("sess", i, len(sn.get_trials_list(True)))

        print("=== N units per session")
        for i, sn in enumerate(self.SessionsList):
            print("\n====== SESSION NUM: ", i)
            sn.sitegetter_print_summary_nunits_by_region()
