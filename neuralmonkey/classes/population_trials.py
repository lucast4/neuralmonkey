import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from neuralmonkey.neuralplots.population import plotNeurHeat, plotNeurTimecourse
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.pandastools import _check_index_reseted

class PopTrials():
    """ 
    Holds data across trials, events, neurons, and times.
    Allows for trials with differnet lenghts.
    """

    def __init__(self, DFTRIAL):
        """ 
        PARAMS:
        - DFTRIAL, dataframe, where rows are trials, each holding pa, SN, and other columns of interst
        """

        columns_needed = ['SN', 'trialcode', 'pa', 'spike_times', 'event_times']
        for col in columns_needed:
            assert col in DFTRIAL.columns

        _check_index_reseted(DFTRIAL)

        # Check that all channels are identical across trials...
        chans = None
        for pa in DFTRIAL["pa"]:
            if chans is None:
                chans = pa.Chans
            else:
                assert chans == pa.Chans        

        self.Dat = DFTRIAL

        self.Datasetbeh = None

    def index_trialcode_to_idx(self, tc):
        """
        Return the row index in self.Dat for this tc. 
        Fails if doesnt exist.
        """

        tmp = self.Dat[self.Dat["trialcode"] == tc].index.tolist()
        assert len(tmp)==1
        idx = tmp[0]
        return idx
    
    def index_chan_to_idx(self, chan):
        """
        Get the index for this channel, in self.Chans and self.LabelChans
        """
        assert chan in self.Chans
        return self.Chans.index(chan)

    def extract_pa(self, idx):
        """
        Extract PA (all neurons,times) for this trial index
        """
        return self.Dat.iloc[idx]["pa"]

    def extract_sn(self, idx):
        """
        Get SN and its trial (within session) for
        this row index --> (sn, trial_sn)
        """
        sn = self.Dat.iloc[idx]["SN"]
        trial_sn = self.Dat.iloc[idx]["trial"]
        return sn, trial_sn

    def extract_spike_times(self, idxtrial, chan):
        """
        Retrun spike times, as (nspikes,) array, in
        seconds, in the original times relative to trial onset
        (NOT aligned to any chosen event yet)
        """
        return self.Dat.iloc[idxtrial]["spike_times"][chan]

    def extract_event_times(self, idxtrial):
        """
        Get events for this trial
        RETURNS:
        - dict[event:times] for this trial, where times are arrays
        """        
        event_times = self.Dat.iloc[idxtrial]["event_times"] # dict[event:times]
        return event_times

    def extract_stroke_times(self, idxtrial, onset_or_offset):
        """
        Get lsit of stroke time for this trial, either alinged to "onset" or "offset",
        determined by onset_or_offset

        RETURNS;
        - list of scalar times (sec)
        """

        if onset_or_offset in ["on", "onset"]:
            return self.Dat.iloc[idxtrial]["stroke_ons"]
        elif onset_or_offset in ["off", "offset"]:
            return self.Dat.iloc[idxtrial]["stroke_offs"]
        else:
            assert False, "input was wrong"

    def extract_stroke_peanut_times(self, idxtrial):
        """
        Get times of peanuts during stroke. "Peanuts" are ink dots that 
        compose the line that is drawn by monkey. They are all contained within 
        stroke.

        RETURNS;
        - List of array of times where list if len n strokes
        """

        sn, trialsn = self.extract_sn(idxtrial)
        peanut_times = sn._behcode_extract_times(75, trialsn, refrac_period=0)

        # get peanut times within each stroke
        stroke_ons = self.extract_stroke_times(idxtrial, "on")
        stroke_offs = self.extract_stroke_times(idxtrial, "off")
        strokes_peanut_times = []
        for _, (s_on, s_off) in enumerate(zip(stroke_ons, stroke_offs)):
            pnuts = peanut_times[(peanut_times>=s_on) & (peanut_times<=s_off)]
            strokes_peanut_times.append(np.array(pnuts))

        return strokes_peanut_times

    def datasetbeh_extract_D(self):
        """
        Get Dataset() class, holding data across all self.Dat
        where D.Dat rows are aligned exactly with each trial in self.Dat
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        if (self.Datasetbeh is not None) and (self.Datasetbeh.Dat["trialcode"].tolist() == self.Trials):
            pass
        else:
            # Collect behavior
            list_D = []
            for sn in set(self.Dat["SN"]):
                list_D.append(sn.Datasetbeh)
            list_D = list(set(list_D))

            from pythonlib.dataset.analy_dlist import concatDatasets
            if len(list_D)>1:
                D = concatDatasets(list_D)
            else:
                D = list_D[0]

            # Make sure dataset matches self
            # Slice beh to match self.Dat
            D.Dat = slice_by_row_label(D.Dat, "trialcode", self.Dat["trialcode"].tolist(), 
                                            assert_exactly_one_each=True)

            self.Datasetbeh = D

        return self.Datasetbeh

    def datasetbeh_extract_df_trialaligned(self):
        """
        Get dataframe of behavior, where rows in dataframe are
        aligned exactly with each trial in self.Dat
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        D = self.datasetbeh_extract_D()
        return D.Dat
    
    def datasetbeh_abort_stroke_times_extract(self):
        """
        THis determines whether abort happend on the last stroke, using method based on 
        the stored kind of abort. 
        
        More accurate would be to use the errorcode, WN, and other features to compute it directly.
        But this is quick and works arlgiht. It is conservative.

        RETURNS:
        - dfabort, one row per trial
        """

        dfbeh = self.datasetbeh_extract_df_trialaligned()

        # good_stroke_durations = []
        # aborted_stroke_durations = []

        # times_peanut_good = []
        # times_peanut_aborted = []

        res = []
        for idxtrial, row in dfbeh.iterrows():
            
            stroke_peanut_times = self.extract_stroke_peanut_times(idxtrial)
            # stroke_ons = self.extract_stroke_times(idxtrial, "on")
            stroke_offs = self.extract_stroke_times(idxtrial, "off")
            
            time_last_peanut = stroke_peanut_times[-1][-1]
            dur_last_stroke_using_peanut = stroke_peanut_times[-1][-1] - stroke_peanut_times[-1][0]

            if row["trial_end_method"] == "pressed_done_button":
                # Success
                assert row["online_abort"] == None
                aborted_during_last_stroke = False

                # dur_aborted_stroke = stroke_offs[-1] - stroke_ons[-1]
                # good_stroke_durations.append(dur_aborted_stroke)
                # times_peanut_good.append(dur_last_stroke_using_peanut)

                abort_kind_semantic = "none"

            elif row["trial_end_method"] == "postscene_hotkey_abort":
                # Success
                aborted_during_last_stroke = False
                abort_kind_semantic = "postscene_hotkey"

            elif row["trial_end_method"] == "online_abort" and row["online_abort"] == "hotkey_x":
                # Failed him within trial. Need to look more closely to determine when.
                # Don't assume it was last stroke.
                aborted_during_last_stroke = False
                abort_kind_semantic = "hotkey_x_unsure_when"

            elif row["trial_end_method"] == "online_abort" and row["online_abort"] == "too_far_from_ink":
                # Abort during stroke, stroke-level quality error
                aborted_during_last_stroke = True
                abort_kind_semantic = "duringstroke_too_far_from_ink"

            elif row["trial_end_method"] == "online_abort" and row["online_abort"] == "failed_rule_objectclass":
                # Abort during stroke, due to rule.
                
                if False:
                    print("------------")
                    for pnut_times, off in zip(stroke_peanut_times, stroke_offs):
                        print(off - pnut_times[-1])

                # dur_aborted_stroke = stroke_offs[-1] - stroke_ons[-1]
                # aborted_stroke_durations.append(dur_aborted_stroke)
                # times_peanut_aborted.append(dur_last_stroke_using_peanut)

                # If this failed, save things about it
                aborted_during_last_stroke = True
                abort_kind_semantic = "duringstroke_failedrule"

            else:
                assert False
            
            res.append({
                "idxtrial":idxtrial,
                "trialcode":row["trialcode"],
                "aborted_during_last_stroke":aborted_during_last_stroke,
                "time_of_abort_using_peanuts":time_last_peanut,
                "dur_last_stroke_using_peanut":dur_last_stroke_using_peanut,
                "abort_kind_semantic":abort_kind_semantic,
            })

        dfabort = pd.DataFrame(res)

        if False:
            import seaborn as sns
            sns.displot(data=dfabort, x="dur_last_stroke_using_peanut", hue="abort_kind_semantic", kind="kde")
        return dfabort

    def plot_timecourse_onetrial(self, idx, ax):
        """
        Plots smoothed firing rates for all chans, overlaied
        for a given trial.
        """
        pa = self.extract_pa(idx)
        sn, trial_sn = self.extract_sn(idx)
        pa.plotNeurTimecourse(0, ax=ax, alpha=0.25)
        sn.plotmod_overlay_trial_events(ax, trial_sn)        

    def plot_raster_create_figure_blank(self, duration, n_raster_lines, n_subplot_rows=1, nsubplot_cols=1, 
                                        reduce_height_for_sm_fr=False, sharex=True, sharey=False, force_scale_height=None):
        """ Generate blank figure at right size for rasters"""
        sn, _ = self.extract_sn(0)
        fig, axes, kwargs = sn._plot_raster_create_figure_blank(duration, n_raster_lines, n_subplot_rows,
            nsubplot_cols, reduce_height_for_sm_fr, sharex, sharey, force_scale_height)
        return fig, axes, kwargs

    def plot_raster_multchans(self, idxtrial, chans, ax, overlay_trial_events=True, pcol = "k"):
        """
        Plot for this trial, multiple chans, each a row,
        """
        
        # Collect data for each chan
        list_ylabel = []
        for i, ch in enumerate(chans):
            st = self.extract_spike_times(idxtrial, ch)
            self._plot_raster_line(ax, st, yval=i, color=pcol, linelengths=1, alpha=0.5)

            # collect for ylabel
            bregion = self.info_chan_to_bregion(ch)
            list_ylabel.append(f"{ch}|{bregion}")
                               
        ax.set_yticks(range(len(list_ylabel)))
        ax.set_yticklabels(list_ylabel);
        ax.set_xlabel('time rel. trial onset (sec)');
        ax.set_ylabel('site');
        ax.set_title(f"idxtrial: {idxtrial}")

        if overlay_trial_events:
            self.plotmod_overlay_trial_events_singletrial(ax, idxtrial)

        # XLIM = ax.get_xlim()

        if False: # Redundant, since I am already plotting regions
            # - Overlay brain regions
            self.plotmod_overlay_brainregions(ax, chans)

    def plot_raster_multtrials(self, list_idxtrials, chan, ax, alignto=None,
        raster_linelengths=0.9, alpha_raster = 0.9, overlay_trial_events=True,
        ylabel_trials=True, plot_rasters=True, xmin = None, xmax = None,
        overlay_strokes=True):
        """ Plot raster, for a channel, but multiple trials, on this axis.
        PARAMS:
        - list_idxtrials, list of indices into self. will plot them in order, from bottom to top
        - alignto, str, even to align to.
        """

        list_align_time = []
        for i, idxtrial in enumerate(list_idxtrials):
            
            # get time of this event (first instance)
            if alignto:
                timesthis = self.extract_event_times(idxtrial)[alignto]
                # if long, then take the first one
                assert len(timesthis)>0
                alignto_time = timesthis[0]
            else:
                alignto_time = None

            list_align_time.append(alignto_time)
                
            # Rasters
            if plot_rasters: # THis hould be independento f overlay_trial_egents, since latter I use for preprocessing, to make plots of events.
                spikes = self.extract_spike_times(idxtrial, chan)
                self._plot_raster_line(ax, spikes, i, alignto_time=alignto_time, 
                    linelengths=raster_linelengths, alpha=alpha_raster)

        if overlay_trial_events:
            self.plotmod_overlay_trial_events_singletrial_multrows(ax, list_idxtrials, list_align_time,
                ylabel_trials, xmin=xmin, xmax=xmax, overlay_strokes=overlay_strokes)
        
        ax.set_title(self.info_chan_to_summarytext(chan)) 

    def plot_rasters_block_by_var(self, chan, vars_group, event_align, overlay_trial_events = True):
        """
        Plot raster, where y axis is blobked into diff conjunctive levels of vars_group
        In a single raster plot, group the rows using meaningful variables
        PARAMS;
        - vars_group, list of str, variables for beh
        -- e.g., ["supervision_stage_concise", "aborted"]
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

        # Get beh
        dfbeh = self.datasetbeh_extract_df_trialaligned()

        # Rasters, grouped into two
        # get trials grouped by var_group 
        grpdict = grouping_append_and_return_inner_items_good(dfbeh, vars_group)

        list_list_st = []
        list_list_evtimes = []
        list_labels = []
        list_idxtrials_flat = []
        list_evtimes_flat = []
        for grp, inds in grpdict.items():
            
            list_st = []
            list_evtimes = []
            for idxtrial in inds:
                st = self.extract_spike_times(idxtrial, chan)
                evtime = self.extract_event_times(idxtrial)[event_align][0]

                # realign to event
                st = st - evtime

                list_st.append(st)
                list_evtimes.append(evtime)

            # Global lists
            list_list_st.append(list_st)
            list_labels.append(grp)
            list_list_evtimes.append(list_evtimes)
            # - flat
            list_idxtrials_flat.extend(inds)
            list_evtimes_flat.extend(list_evtimes)

        # Plot
        sn, _ = self.extract_sn(0)
        dur = 10
        ntrials = len(list_idxtrials_flat)
        nrows = 1
        ncols = 1
        fig, axes, kwargs = self.plot_raster_create_figure_blank(dur, ntrials, nrows, ncols)
        # fig, ax = plt.subplots(figsize=(10, 20))
        ax = axes.flatten()[0]
        list_list_trials = None
        sn.plot_raster_spiketimes_blocked(ax, list_list_st, list_labels, list_list_trials, 
            list_list_evtimes, overlay_trial_events=False)

        if overlay_trial_events:
            # Overlay events
            list_yvals = list(range(len(list_idxtrials_flat)))
            self.plotmod_overlay_trial_events_singletrial_multrows(ax, 
                                                                list_idxtrials_flat, 
                                                                list_evtimes_flat, 
                                                                None, 
                                                                list_yvals)


    def _plot_raster_line(self, ax, spiketimes, yval, color='k', alignto_time=None, 
                          linelengths = 0.85, alpha=0.4, linewidths=None):
        """
        Plot a single raster line of spiketimes (array of times) at this yval
        """
        sn, _ = self.extract_sn(0)
        sn._plot_raster_line(ax, spiketimes, yval, color, alignto_time, linelengths, alpha, linewidths)

    def plotmod_overlay_brainregions(self, ax, chans):
        """
        Overlya labels for brain regions on y axis, where chans correspond to rows, ie
        the labels of each row.
        """
        sn, _ = self.extract_sn(0)
        sn.plotmod_overlay_brainregions(ax, chans)

    def plotmod_overlay_trial_events_singletrial(self, ax, idxtrial, alignto_time=None, **kwargs):
        """
        Overlay events for this trial.
        Entire plot is a single trial, so the event markers span the trial
        """
        sn, trialsn = self.extract_sn(idxtrial)
        sn.plotmod_overlay_trial_events(ax, trialsn, alignto_time=alignto_time, **kwargs)

    def plotmod_overlay_trial_events_singletrial_multrows(self, ax, list_idxtrial, list_align_time=None,
            ylabel_trials=None, list_yvals=None, xmin=None, xmax =None, overlay_strokes=True,
            clear_old_yticks = True):
        """
        Overlay events for multiple trials.
        Plot arrowheads for multiple trials, each a row in the plot
        PARAMS;
        - list_align_time, list of times that are defined as 0.
        """

        if list_yvals is None:
            # start from bottom
            list_yvals = list(range(len(list_idxtrial)))

        if list_align_time is None:
            list_align_time = [None for _ in range(len(list_idxtrial))]

        assert len(list_align_time)==len(list_idxtrial)

        # subsample trials to label
        if ylabel_trials is None or ylabel_trials==True:
            # just label as the trial indices
            ylabel_trials = list_idxtrial
        else:
            assert len(ylabel_trials)==len(list_idxtrial)

        for i, (yval, idxtrial, alignto_time) in enumerate(zip(list_yvals, list_idxtrial, list_align_time)):

            # - overlay beh things
            # Auto determine alpha for markers, based on num trials
            ALPHA_MARKERS = 1-np.clip(len(list_idxtrial)/ 90, 0.63, 0.82)
            if i==0:
                include_text = True
            else:
                include_text = False

            # Plot
            sn, trialsn = self.extract_sn(idxtrial)
            sn.plotmod_overlay_trial_events(ax, trialsn, alignto_time=alignto_time, only_on_edge="bottom", 
                                            YLIM=[yval-0.3, yval+0.5], which_events=["key_events_correct"], 
                                            include_text=include_text, text_yshift = -0.5, alpha=ALPHA_MARKERS,
                                            xmin = xmin, xmax =xmax)
            
            if overlay_strokes:
                ALPHA_STROKES = 0.8*ALPHA_MARKERS
                sn.plotmod_overlay_trial_events(ax, trialsn, alignto_time=alignto_time, 
                                                YLIM=[yval-0.4, yval-0.3], which_events=["strokes"], alpha=ALPHA_STROKES,
                                                xmin = xmin, xmax =xmax)

        if clear_old_yticks:
            ticks_current = []
            labels_current =[]
        else:
            ticks_current = list(ax.get_yticks())
            labels_current = list(ax.get_yticklabels())

        ax.set_yticks(ticks_current+list_yvals, labels=labels_current+ylabel_trials, 
            fontsize=5)

        # ax.set_xbound(xmin, xmax)
        if xmin is not None:
            ax.set_xlim(xmin=xmin)
        if xmax is not None:
            ax.set_xlim(xmax=xmax)


    def info_bregion_to_chans(self, bregion):
        """
        Return list of chans for this bregion
        """
        return self.LabelChans[self.LabelChans["region"] == bregion]["chan_pt"].tolist()

    def info_chan_to_bregion(self, chan):
        """
        Get the bregion for this chan, returned as string
        """
        idx_chan = self.index_chan_to_idx(chan)
        return self.LabelChans.iloc[idx_chan]["region"]

    def info_chan_to_summarytext(self, chan):
        """
        Return chan|region, a string summarizing this chan
        """
        idx = self.index_chan_to_idx(chan)
        region = self.LabelChans.iloc[idx]["region"]
        return f"{chan}|{region}"

    def pa_extract_align_to_event(self, event, time_pre, time_post):
        """
        Helper to extract a PA, holding all chans and trials, aligned to this event
        across trials, with time window (time_pre, time_post are both positive.)
        """ 
        from neuralmonkey.classes.population import concatenate_popanals

        assert time_pre>0
        assert time_post>0

        # Get PA across trials, aligned to this event
        list_pa = []
        for idxtrial in range(len(self.Dat)):
            pa = self.extract_pa(idxtrial)
            event_times = self.extract_event_times(idxtrial)[event]
            
            if len(event_times)==0:
                print("Skipping, becuase is missing event: ", idxtrial)
                continue
            elif len(event_times)>1:
                print(idxtrial)
                print(event_times)
                assert False, "how deal with this? Should extract new snip per event instance"
            
            # Get single pa
            time = event_times[0]
            twind = (time-time_pre, time+time_post)
            pathis = pa.slice_by_dim_values_wrapper("times", twind)
            list_pa.append(pathis)

        # Reset all times, and round them
        sampling_period = np.round(np.mean(np.diff(pathis.Times))*1000)/1000
        # - Replace all times with this time relative to alignement.
        for pa in list_pa:
            # sampling period, to acocunt for random variation in alignment across snips.
            TIMES = (pa.Times - pa.Times[0]) - time_pre + sampling_period/2 # times all as [-predur, ..., postdur]
            pa.Times = TIMES

        assert len(set([len(pa.Times) for pa in list_pa]))==1, "times not aligned?"

        PA = concatenate_popanals(list_pa, "trials", 
                                assert_otherdims_have_same_values=True, 
                                assert_otherdims_restrict_to_these=("chans", "times"),
                                all_pa_inherit_times_of_pa_at_this_index=0)
        
        # Extract trials data
        df = self.datasetbeh_extract_df_trialaligned()
        assert len(df) == len(PA.Trials)
        PA.Xlabels["trials"] = df

        return PA

    def pa_extract_align_to_stroke(self, time_pre, time_post, onset_or_offset = "onset"):
        """
        Helper to extract a PA, holding all chans and trials, where each returned row
        is aligned to a different stroke, and choice of either onset or offset of 
        stroke. 
        
        (time_pre, time_post are both positive.)

        """
        from neuralmonkey.classes.population import concatenate_popanals
        from pythonlib.dataset.dataset_strokes import preprocess_dataset_to_datstrokes
        assert time_pre>0
        assert time_post>0

        # The features to extract for each stroke.
        feature_columns_take = ["trialcode", "stroke_index", "shape", "gridloc",  "gridsize", "loc_on", "CTXT_loc_prev", "CTXT_shape_prev", 
                                "CTXT_loc_next", "CTXT_shape_next", "shape_semantic", "shape_semantic_grp", 
                                "time_duration", "gap_from_prev_dur", "gap_from_prev_dist", "gap_from_prev_angle",
                                "gap_to_next_dist", "gap_to_next_angle", "epoch"]

        ### Extract stroke-level features    
        D = self.datasetbeh_extract_D()

        # If want to get grammar/syntax chunk data for each stroke:
        if False:
            D.grammarparses_successbinary_score_wrapper()
            for ind in range(len(D.Dat)):
                D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(ind)
            # TODO: incorproate these varibles. They are not included in DS, so it is a bit of work.
        DS = preprocess_dataset_to_datstrokes(D, "all_no_clean")

        ### For each trial determine whether it is abort (and if so, whether aborted on last stroke)
        dfabort = self.datasetbeh_abort_stroke_times_extract()
        assert len(dfabort) == len(self.Dat)

        ### Get PA across trials, aligned to this event
        list_pa = []
        list_feat = []
        list_abort_value = []
        list_abort_reason = []

        for idxtrial in range(len(self.Dat)):
            pa = self.extract_pa(idxtrial)
            tc = self.Dat.iloc[idxtrial]["trialcode"]

            stroke_times = self.extract_stroke_times(idxtrial, onset_or_offset)
            
            assert len(stroke_times)>0, "missing strokes, shuldnt happen"

            # Was this abort?
            assert dfabort.iloc[idxtrial]["trialcode"] == tc
            stroke_abort_values = [False for _ in range(len(stroke_times))]
            stroke_abort_reason = ["ignore" for _ in range(len(stroke_times))]
            if dfabort.iloc[idxtrial]["aborted_during_last_stroke"]:
                stroke_abort_values[-1] = True
                stroke_abort_reason[-1] = dfabort.iloc[idxtrial]["abort_kind_semantic"]
            
            for idxstroke, (time, abort_value, abort_reason) in enumerate(zip(stroke_times, stroke_abort_values, stroke_abort_reason)):    
                
                ### Get single pa
                twind = (time-time_pre, time+time_post)
                pathis = pa.slice_by_dim_values_wrapper("times", twind)
                list_pa.append(pathis)

                ### Get stroke-level features
                # Find this stroke in DS.Dat
                _ind = (DS.Dat["trialcode"] == tc) & (DS.Dat["stroke_index"] == idxstroke).tolist()
                # Sanity check you got the correct row (DS times equal self times)
                if onset_or_offset == "onset":
                    assert DS.Dat[_ind]["time_onset"].values[0] == time
                elif onset_or_offset == "offset":
                    assert DS.Dat[_ind]["time_offset"].values[0] == time
                else:
                    assert False

                # Store features
                list_feat.append(DS.Dat.loc[_ind, feature_columns_take])

                # Store abort features for this stroke
                list_abort_value.append(abort_value)
                list_abort_reason.append(abort_reason)

        ### Reset all times, and round them
        sampling_period = np.round(np.mean(np.diff(pathis.Times))*1000)/1000
        # - Replace all times with this time relative to alignement.
        for pa in list_pa:
            # sampling period, to acocunt for random variation in alignment across snips.
            TIMES = (pa.Times - pa.Times[0]) - time_pre + sampling_period/2 # times all as [-predur, ..., postdur]
            pa.Times = TIMES

        assert len(set([len(pa.Times) for pa in list_pa]))==1, "times not aligned?"

        PA = concatenate_popanals(list_pa, "trials", 
                                assert_otherdims_have_same_values=True, 
                                assert_otherdims_restrict_to_these=("chans", "times"),
                                all_pa_inherit_times_of_pa_at_this_index=0)

        # Assign stroke-level features
        dflab = pd.concat(list_feat, axis=0).reset_index(drop=True)
        dflab["abort_value"] = list_abort_value
        dflab["abort_reason"] = list_abort_reason

        PA.Xlabels["trials"] = dflab

        return PA

    def save(self, savedir, suffix=None):
        """
        """
        import pickle
        
        if suffix:
            path = f"{savedir}/PT-{suffix}.pkl"
        else:
            path = f"{savedir}/PT.pkl"

        with open(path, "wb") as f:
            pickle.dump(self, f)