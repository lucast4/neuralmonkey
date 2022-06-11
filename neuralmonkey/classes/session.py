""" Holds a single recording file
 - i.e, combines neural, beh, and camera
"""
import tdt
import numpy as np
import matplotlib.pyplot as plt
from ..utils.timeseries import dat_to_time
from ..utils.monkeylogic import getTrialsTaskAsStrokes, getTrialsStrokes

class Session(object):
    """
    Operates over all trials for this entire session (day), even if multiple rec files for today

    """
    
    def __init__(self, datestr, beh_expt_list, beh_sess_list, 
            beh_trial_map_list,
            sites_garbage = None,
            expt="Lucas512-220520-115835", animal="Pancho"):
        """
        PARAMS:
        - datestr, string, YYMMDD, e.g, "220609"
        - beh_expt_list, lsit of string, for each beh session you wish to load.
        - beh_trial_map_list
        e.g., [(20,0), (1,200)] means that the first fd's trial 20 maps onto trial 0 neural and
        the second fd starts (trial 1) from trial 200 neural
        - expt, string, expt in Synapse (TDT)
        - animal, string
        - TrialBehAtNeuralZero, int, what trial was ml on the first (zero) trial for 
        neural? If started botrh nerual and beh at same time, then this will be 1.
        """

        self.Animal = animal
        self.ExptSynapse = expt
        self.Date = datestr
        self.Paths = None

        self.Chans = range(1, 257)
        self.Rss = [2, 3]
        self.DatAll = None

        self.SitesGarbage = sites_garbage

        # Behavior stuff
        self.BehDate = datestr
        self.BehExptList = beh_expt_list
        self.BehSessList = beh_sess_list
        self.BehTrialMapList = beh_trial_map_list
        self.BehFdList = []
        self.BehTrialMapListGood = None
        self.load_behavior()

        # self._initialize_params()
        self._initialize_paths()

        # Load raw things
        self.load_tdt_tank()
        # Load raw data
        self.load_spike_times() # Load all spike times

        # Load beh 
        self.load_behavior()

        # Find the times of all trial onsets (inneural data)
        # 1. get all onset and offset times
        self.TrialsOnset = self.extract_behcode_times(9)
        self.TrialsOffset = self.extract_behcode_times(18)



    ####################### PREPROCESS THINGS
    # def _initialize_params(self):
    def _initialize_paths(self, PATHBASE = "/mnt/LT_neural_1"):

        # 1) find all recordings for this date
        from pythonlib.tools.expttools import findPath, deconstruct_filename
        path_hierarchy = [
            [self.Animal],
            [self.Date]
        ]
        paths = findPath(PATHBASE, path_hierarchy)
        assert len(paths)==1, 'not yhet coded for combining sessions'

        fnparts = deconstruct_filename(paths[0])
        print(fnparts)
        pathdict = {
            # "raws":f"{PATHBASE}/{ANIMAL}/{DATE}/{ANIMAL}-{DATETIME}"
            "raws":paths[0],
            "tank":f"{paths[0]}/{fnparts['filename_final_noext']}",
            "final_dir_name":fnparts["filename_final_noext"],
            "time":fnparts["filename_components_hyphened"][2]
            }

        self.Paths = pathdict
        self.PathRaw = pathdict["raws"]
        self.PathTank = pathdict["tank"]


    ####################### EXTRACT RAW DATA (AND STORE)
    def load_behavior(self):
        """ Load monkeylogic data for this neural recording
        Could be either filedata, or dataset, not sure yet , but for now go with filedata
        """ 
        from ..utils.monkeylogic import loadSingleDataQuick
        for e, s, trialmap in zip(self.BehExptList, self.BehSessList, self.BehTrialMapList):

            # Load filedata
            # a = "Pancho"
            # d = 220531
            # e = "charneuraltrain1b"
            # s = 1
            fd = loadSingleDataQuick(self.Animal, self.Date, e, s)

            self.BehFdList.append(fd)

    def load_tdt_tank(self):
        """ Holds all non-nueral signals. is aligned to the neural
        """
        print("** Loading data from tdt tank")
        data_tank = tdt.read_block(self.PathTank, evtype = ["epocs", "streams"])
        self.DatTank = data_tank

    def load_spike_times(self):
        """ Load and strore all spike times (across all trials, and chans)
        Not yet aligned to trials, etc.
        """
        # Extract database of spikes (not sliced by trials yet)
        # chans = range(1,256+1)
        # rss = [2,3]

        def load_spike_times_(rs, chan, ver="spikes_tdt_quick"):
            """ Return spike times, pre-extracted elsewhere (matlab)
            in secs
            """
            import scipy.io as sio
            PATH_SPIKES = f"{self.PathRaw}/spikes_tdt_quick"
            fn = f"{PATH_SPIKES}/RSn{rs}-{chan}"
            mat_dict = sio.loadmat(fn)
            return mat_dict["spiketimes"]

        def load_spike_times_mult(rss, chans, ver="spikes_tdt_quick"):
            DatSpikes = []
            for rs in rss:
                for ch in chans:
                    st = load_spike_times_(rs, ch, ver)
                    DatSpikes.append({
                        "rs":rs,
                        "chan":ch,
                        "spike_times":st})

            print("Got spikes for rss/chans")
            print(rss)
            print(chans)
            return DatSpikes

        DatSpikes = load_spike_times_mult(self.Rss, self.Chans)
        self.DatSpikes = DatSpikes


    def load_raw(self, rss, chans, trial0, pre_dur=1., post_dur = 1.,
            get_raw=True):
        """ Get all raw data across channels for this trial
        PARAMS:
        - get_raw, bool( True), whether to actually load. if False, then makes teh
        hodler but doesnt load.
        """

        TIME_ML2_TRIALON = self.ml2_get_trial_onset(trialtdt = trial0)
        def extract_raw_(rs, chans, trial0, pre_dur=1., post_dur = 1.):
            """ Extract raw data for this trial, across any number of chans,
            by default will get these chans for both RSs, 
            """
            # --
            assert isinstance(chans, list)
            T1, T2 = self.extract_timerange_trial(trial0, pre_dur, post_dur)
            n = len(chans)
            
            DatRaw = []
            if get_raw:
                # data = tdt.read_block(PATH, channel=channels, t1=T1, t2=T2)
                ev = f"RSn{rs}"
                data = tdt.read_sev(self.PathRaw, channel=chans, event_name=ev, t1=T1, t2=T2)
                assert data.time_ranges[0]==T1
                assert data.time_ranges[1]==T2

                # Reshape so dim is not (n,) when only one chan
                if len(chans)==1:
                    data[ev]["data"] = data[ev]["data"][None,:]
                
            #     time_ranges = data.time_ranges
                # Convert to list of dict
                # print(chans)
                # print(n)
                # print(data[ev]["data"])
                # print(data[ev]["data"].shape)
                # assert False
                assert n==len(data[ev]["channels"])
                assert n== data[ev]["data"].shape[0]
                
                for i in range(n):
                    ch = data[ev]["channels"][i]
                    assert ch==chans[i]

                    # generate the timebins (start from 0 sec)
                    raw = data[ev]["data"][i, :]
                    fs = data[ev]["fs"]
                    t = self.dat_to_time(raw, fs) # starts from 0
                    
                    # sanity check the bins
                    dt1 = data.time_ranges[-1] - data.time_ranges[0]
                    dt2 = t[-1]-t[0]
                    assert dt1-dt2<0.001, "error in computing time bins"
                    
                    # Reconvert time so that time=0sec is defined at eventcode 9 (ml2)
                    
                    if True:
                        # more fool-proof version
                        t = t + T1 # - first convert time to real time (entire session)
                        t, raw = self.extract_windowed_data_bytrial(t, trial0, raw) # - second, use the general windowing function
                    else:
                        # This works
                        t = t - pre_dur - TIME_ML2_TRIALON # shifts time base

                    DatRaw.append({
                        "rs":rs,
                        "chan":ch,
                        "trial0":trial0,
                        "pre_dur":pre_dur,
                        "post_dur":post_dur,
                        "time_range":data.time_ranges,
                        "raw":raw,
                        "tbins0":t,
                        "fs":fs}),
            else:
                # Dummy...
                for i in range(n):
                    ch = chans[i]

                    # generate the timebins (start from 0 sec)
                    raw = None
                    fs = None
                    t = None
                    
                    DatRaw.append({
                        "rs":rs,
                        "chan":ch,
                        "trial0":trial0,
                        "pre_dur":pre_dur,
                        "post_dur":post_dur,
                        "time_range":np.array([T1, T2]),
                        "raw":raw,
                        "tbins0":t,
                        "fs":fs}),
            return DatRaw
        
        DatRaw = []
        for rs in rss:
            print(rs)
            datraw = extract_raw_(rs, chans, trial0, pre_dur, post_dur)
            DatRaw.extend(datraw)
        return DatRaw


    ###################### EXTRACT SPIKES (RAW, PRE-CLIUSTERED)
    def load_spike_waveforms_(self, rs, chan, ver="spikes_tdt_quick"):
        """ Return spike waveforms, pre-extracted
        """
        import scipy.io as sio
        PATH_SPIKES = f"{self.PathRaw}/spikes_tdt_quick"
        fn = f"{PATH_SPIKES}/RSn{rs}-{chan}-snips_subset"
        mat_dict = sio.loadmat(fn)
        return mat_dict["snips"]




    ########################## GET FROM PRE-EXTRACTED DATA
    def extract_behcode_times(self, codenum, trial0=None, first_instance_only=False):
        """ Extract the times that this codenum occured, in sec
        PARAMS
        - codenum, int
        - trial0, int, if given, then will get times reltaive to this trial
        - first_instance_only, bool, if True, then only returns the first time
        it occurs.
        """    
        timesall, codes = self.extract_data_tank_epocs("behcode")
        inds = codes==codenum
        times = timesall[inds]
        
        if trial0 is not None:
            times, _ = self.extract_windowed_data_bytrial(times, trial0)
        
        if first_instance_only:
            if len(times)>1:
                times = times[0:1] # stay as an array
        return times
        
    def extract_behcode_times_sequence(self, codes, trial0):
        """ Get the times of this sequence of codes, aligned to this trial,
        taking only the first instance of each code.
        If a code doesnt occur, gives it a np.nan
        RETURNS:
        - times, np array len of codes. elements are either seconds or np.nan
        NOTE: order will not matter
        """
        
        times = np.empty(len(codes))
        for i, c in enumerate(codes):
            t = self.extract_behcode_times(c, trial0, True)
            if len(t)>0:
                times[i] = t[0]
            else:
                times[i] = np.nan
        return times    

    def extract_data_tank_streams(self, which, trial0=None):
        """ Extract tank data, epocs, in flexible manner.
        PARAMS:
        - which, string name, will be mapped to the specific key in data
        - crosstime, whether to use "onset", "offset", or "mean" times
        - trial0, int, optional trial to slice data (will recompute the 0 rel trial onset).
        """
        
        keynames = {
            "mic":"Mic1",
            "duplicate1":"dup1",
            "duplicate2":"dup2",
            "pd1":"PhDi",
            "pd2":"PhD2",
            "eyex":"Eyee",
            "eyey":"Eyee",
            "eyediam":"Eyee",
        }
        key = keynames[which]
        dat = self.DatTank["streams"][key]
        
        
        # Times, Values
        fs = dat["fs"]
        vals = dat["data"]
        
        # some vals are arrays
        if which=="eyex":
            vals = vals[0,:]
        elif which=="eyey":
            vals = vals[1,:]
        elif which=="eyediam":
            vals = vals[2, :]
        
        # get times
        times = self.dat_to_time(vals, fs)
        
        # Slice a single tyrial?
        if trial0 is not None:
            times, vals = self.extract_windowed_data_bytrial(times, trial0, vals)
        
        return times, vals


    def extract_data_tank_epocs(self, which, crosstime="onset", trial0=None):
        """ Extract tank data, epocs, in flexible manner.
        PARAMS:
        - which, string name, will be mapped to the specific key in data
        - crosstime, whether to use "onset", "offset", or "mean" times
        - trial0, int, optional trial to slice data (will recompute the 0 rel trial onset).
        """
        
        keynames = {
            "camtrialon":"Ton_",
            "camframe":"CIn_",
            "camtrialoff":"Tff_",
            "rew":"Rew_",
            "rewon":"Rew_",
            "rewoff":"Rew_",
            "behcode":"SMa1",
            "strobe":"S_ML",
        }
        key = keynames[which]
        dat = self.DatTank["epocs"][key]
        
        # Force using onset or offset
        if which in ["rewon"]:
            crosstime = "onset"
        elif which in ["rewoff"]:
            crosstime = "offset"
        
        # Times
        if crosstime in ["onset", "offset"]:
            times = dat[crosstime]
        elif crosstime=="mean":
            # take mean of on and off
            ons = dat["onset"]
            offs = dat["offset"]
            times = (ons+offs)/2
        else:
            assert False
            
        # Values
        vals = dat["data"]
        
        # Slice a single tyrial?
        if trial0 is not None:
            times, vals = self.extract_windowed_data_bytrial(times, trial0, vals)
        
        return times, vals

    def extract_raw_and_spikes(self, rss, chans, trialtdt, get_raw = False):
        """ [GET ALL DATA] Extract raw (unfiltered) and spikes for this trial
        PARAMS:
        - rss, list of ints for rs
        - chans, list of chans (within this rs)
        - trialtdt, tdt trials
        """
        
        # raw (get all chans) [from disk]
        DatRaw = self.load_raw(rss, chans, trialtdt, get_raw=get_raw)
        
        # spikes (from pre-extracted spikes)
        for d in DatRaw:
    #         spike_times = datspikes_slice_single(d["rs"], d["chan"], d["time_range"])
            spike_times = self.datspikes_slice_single(d["rs"], d["chan"], trial0=trialtdt)
            d["spike_times"] = spike_times
        
        # change name, since is both raw and spikes
        # append to DatAll
        if self.DatAll is None:
            self.DatAll = DatRaw
        else:
            for d in DatRaw:
                if not self.datall_this_exists(d["rs"], d["chan"], d["trial0"]):
                    # then append
                    self.DatAll.append(d)
        return self.DatAll


    # def load_all_data_this_trial_(self, trial0, rss, chans):
    #     # Extract both raw and spikes for a given channel and trial
    #     # chans = list(range(1, 256+1))
    #     # rss = [2,3]
    #     # trial0 = 267
    #     DatAll = self.extract_raw_and_spikes(rss, chans, trial0)



    ###################### WINDOW THE DATA based on trials, etc
    def extract_windowed_data(self, times, twind, vals=None, recompute_time_rel_onset=True, time_to_add=0.):
        """ Prune data (time, vals) to include only those with
        times within twind. Also changes times to be relative to twind[0]
        """
        inds = (times>=twind[0]) & (times<=twind[1])
        if vals is not None:
            assert times.shape==vals.shape
            vals = vals[inds]
        times = times[inds]
        
        # get times relative to windo wonset
        if recompute_time_rel_onset:
            times = times - twind[0]
            
        # shift the times
        times = times + time_to_add
        
        return times, vals

    def extract_windowed_data_bytrial(self, times, trial0, vals=None, recompute_time_rel_onset=True, pre_dur=1., post_dur=1.):
        """ Given generic data, window it by a given trial.
        Prune data (time, vals) to include only those with
        times within twind. Also changes times to be relative to trial onset (regardless of pre_dur
        PARAMS:
        - pre_dur, post_dur, time extra to extract. NOTE: doesnt affect what is called 0, which is always trial onset
        """
        
        # Get window
        t1, t2 = self.extract_timerange_trial(trial0, pre_dur, post_dur)
        TIME_ML2_TRIALON = self.ml2_get_trial_onset(trialtdt = trial0)
        time_to_add = -pre_dur # so that is zeroed on time of trial onset
        
        # shift all tdt things so that by definition the time of beh code 9 are identical between tdt and ml2
        time_to_add = time_to_add - TIME_ML2_TRIALON
        return self.extract_windowed_data(times, [t1, t2], vals, recompute_time_rel_onset, time_to_add = time_to_add)

    def extract_timerange_trial(self, trial0, pre_dur=1., post_dur=1.):
        T1 = self.TrialsOnset[trial0]-pre_dur
        T2 = self.TrialsOffset[trial0]+post_dur
        return T1, T2


    #################### DATALL
    def datall_this_exists(self, rs, chan, trial0):
        """ returns True if this combo exist sin self.DatAll,
        false otherwise
        """
        for D in self.DatAll:
            if D["rs"]==rs and D["chan"]==chan and D["trial0"]==trial0:
                return True
        return False

    def datall_slice_single(self, rs, chan, trial0):
        """ Slice a single chans data
        """
        for D in self.DatAll:
            if D["rs"]==rs and D["chan"]==chan and D["trial0"]==trial0:
                return D
        print(rs, chan, trial0)

        assert False, 'this combo of rs and chan and trial0 doesnt exist in DatAll!'



    ###################### SPIKES
    # Extract spike times within a given time window
    def datspikes_slice_windowed(self, DatSpikes, twind):
        """ extract spike times only within this window, and recompute start
        time to be rel to T1
        """
        assert False, "should try to use window by trial, so that time base is correct"
        DatSpikesWindowed = []
        for D in DatSpikes:
    #         times = D["spike_times"]
    #         times = extract_windowed_data(times, twind)[0]
            
            # Make a new containner , so dont overwrite DatSpikes
            Dnew = {}
            for k, v in D.items():
                Dnew[k] = D[k]
            
            Dnew["spike_times"] = extract_windowed_data(Dnew["spike_times"], twind)[0]
            DatSpikesWindowed.append(Dnew)
            
        return DatSpikesWindowed
        

    def datspikes_slice_single(self, rs, chan, trial0=None, twind=None):
        """ Slice a single chans spike times, 
        optionally relative to a time window
        PARAMS:
        - rs, chan, ints
        - trial0, int, trial to reslice and reform time base to. 
        - twind, (2,) array, time window to slice out.
        NOTE: use trial0 if want time base to match up so that behcode 9 is 0 sec.
        NOTE: can only pass in one of trial0 or twind
        """
        
        for D in self.DatSpikes:
            if D["rs"]==rs and D["chan"]==chan:
                spiketimes = D["spike_times"]
                # optionally window it
                if twind is not None:
                    assert trial0 is None
                    spiketimes = self.extract_windowed_data(spiketimes, twind)[0]
                elif trial0 is not None:
                    assert twind is None
                    spiketimes = self.extract_windowed_data_bytrial(spiketimes, trial0)[0]    
                return spiketimes
                

        print(rs, chan)
        assert False, 'this combo of rs and chan doesnt exist in DatSpikes!'
        
    ####################### HELP CALC THINGS
    def dat_to_time(self, vals, fs):
        return dat_to_time(vals, fs)

    ####################### CONVERSIONS BETWEEN BEH AND NEURAKL
    def beh_get_fdnum_trial(self, trialtdt):
        from ..utils.conversions import get_map_trial_and_set
        ntrials = len(self.TrialsOnset)
        assert trialtdt < ntrials

        dict_trial2_to_set_and_trial1 = get_map_trial_and_set(self.BehTrialMapList, ntrials)
        if False:
            # plot
            for k, v in dict_trial2_to_set_and_trial1.items():
                print(k, v)
        
        if self.BehTrialMapListGood is None:
            self.BehTrialMapListGood = dict_trial2_to_set_and_trial1
        [fd_setnum, fd_trialnum] = self.BehTrialMapListGood[trialtdt]
        return fd_setnum, fd_trialnum
        
    def beh_get_fd_trial(self, trialtdt):
        """ Return the fd and trial linked to this tdt trial
        """
        fd_setnum, fd_trialnum = self.beh_get_fdnum_trial(trialtdt)
        fd = self.BehFdList[fd_setnum]
        return fd, fd_trialnum

    # def convert_trialnum(self, trialtdt=None, trialml=None):
    #     """ Given one of neural (tdt) or beh (ml) trials, convert to the other one.
    #     Assumes that no skipped trials.
    #     """
    #     assert False, "fix this so it uses all the fd. returns fd and trial"
    #     if trialtdt is None: # then try to get trialtdt
    #         return trialml-self.TrialBehAtNeuralZero
    #     elif trialml is None:
    #         # return trialtdt+1
    #         return trialtdt+self.TrialBehAtNeuralZero

    def ml2_get_trial_onset(self, trialtdt):
        from ..utils.monkeylogic import ml2_get_trial_onset
        # convert to trialml
        fd, trialml = self.beh_get_fd_trial(trialtdt)
        return ml2_get_trial_onset(fd, trialml)

    ######################## BRAIN STUFF
    def sitegetter_brainregion(self, region=None, clean=False):
        regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
                            "PMd_p", "PMd_a", "dlPFC_p", "dlPFC_a", 
                            "vlPFC_p", "vlPFC_a", "FP_p", "FP_a",
                            "SMA_p", "SMA_a", "preSMA_p", "preSMA_a"]
        dict_sites ={}
        for i, name in enumerate(regions_in_order):
            dict_sites[name] = list(range(1+32*i, 1+32*(i+1)))
        if False:
            for k, v in dict_sites.items():
                print(k, v)
            
        # do clean
        if clean:
            assert self.SitesGarbage is not None, "you need to enter which are bad sites"
            for k, v in dict_sites.items():
                # remove any sites that are bad
                dict_sites[k] = [vv for vv in v if vv not in self.SitesGarbage]

        if region=="list_regions":
            return regions_in_order
        elif region=="mapper" or region is None:
            return dict_sites
        elif isinstance(region, int):
            return dict_sites[regions_in_order[region]]
        else:
            return dict_sites[region]

    def convert_site_to_rschan(self, sitenum):
        """ Cnvert sites (1-512) to rs and chan
        e.g., 512 --> (3, 256)
        """
        if sitenum>256:
            rs = 3
            chan = sitenum - 256
        else:
            rs = 2
            chan = sitenum
        return rs, chan

    ############################ BEH STUFF (ML)
    # def events_extract_stroke_onsoffs(self, trial0):
    #     """ Get the times of stroke onsets and offsets.
    #     """
    #     fd, trialml = self.beh_get_fd_trial(trial0)
    #     # trialml = convert_trialnum(trialtdt=trial0)
    #     strokes = getTrialsStrokes(fd, trialml) # includes go and done

    ####################### PLOTS (generic)
    
    def plot_spike_waveform(self, ax, waveforms, nplot = 200, fs=None):
        """ Plot multipel spikes waveforms overlaid
        PARAMS:
        - waveforms, np array, (n, ntimebins),
        - nplot, how many
        - fs, Optional, if want to plot with correct time base
        """
            
        if nplot>waveforms.shape[0]:
            nplot = waveforms.shape[0]
        
        if fs is not None:
            t = self.dat_to_time(waveforms[0], fs)
            ax.set_xlabel('time')
        else:
            t = np.arange(waveforms.shape[1])
            ax.set_xlabel('bins')
            
        ax.plot(t, waveforms[:nplot, :].T, alpha=0.3)
        
    
    def plot_raster_line(self, ax, times, yval, color='k'):
        """ plot a single raster line at times at row yval
        """
        
    #     ax.plot(times, yval*np.ones(time.shape), '.', color=color, alpha=0.55)
        y = yval*np.ones(times.shape)
        ax.eventplot([times], lineoffsets=yval, color=color, alpha=0.55)
        
        # plot as hashes
        
    def plotmod_overlay_trial_events(self, ax, trial0):
        """ Overlines trial events in vertical lines
        Time is rel trial onset (ml2 code 9)
        Run this after everything else, so get propoer YLIM.
        """
        from ..utils.monkeylogic import getTrialsOnsOffs
        # Vertical lines for beh codes
        list_codes = [9, 11, 16, 91, 92, 62, 73, 50]
        times_codes = self.extract_behcode_times_sequence(list_codes, trial0)
        # names_codes = [beh_codes[c] for c in list_codes] 
        names_codes = ["on", "fixcue", "fix", "samp", "go", "done", "fb_vs", "rew"]
        
        # Also include times of strokes
        fd, trialml = self.beh_get_fd_trial(trial0)
        ons, offs = getTrialsOnsOffs(fd, trialml)
        times_codes = np.append(times_codes, ons)
        times_codes = np.append(times_codes, offs)
        names_codes.extend(["Son" for _ in range(len(ons))])
        names_codes.extend(["Soff" for _ in range(len(ons))])
        
        
        YLIM = ax.get_ylim()
        for time, name in zip(times_codes, names_codes):
            if name in ["Son"]:
    #             ax.axvline(time, color="b", ls="--", alpha=0.5)
                ax.axvline(time, color="b", ls="-", alpha=0.5)
            elif name in ["Soff"]:
    #             ax.axvline(time, color="m", ls="--", alpha=0.5)
                ax.axvline(time, color="m", ls="-", alpha=0.5)
            else:
                ax.axvline(time, color="r", ls="-")
            ax.text(time, YLIM[0], name, rotation="vertical")


        

    def plot_trial_timecourse_summary(self, ax, trial0, number_strokes=True):
        # trialml = convert_trialnum(trialtdt=trial0)
        from ..utils.monkeylogic import getTrialsStrokes
        fd, trialml = self.beh_get_fd_trial(trial0)
        strokes = getTrialsStrokes(fd, trialml) # includes go and done
        
        # Beh strokes (ml2)
        for i, s in enumerate(strokes):
            x = s[:,0]
            y = s[:,1]
            t = s[:,2]
            ax.plot(t, x, label="x")
            ax.plot(t, y, label="y")
            
            if number_strokes:
                ax.text(np.mean(t), np.max(np.r_[x,y]), i)
            
        self.plotmod_overlay_trial_events(ax, trial0)


    ###################### PLOTS (specific)
    def plot_specific_trial_overview(self, trialtdt):
        """ Plots _everytnig_ about this trial, aligned
        """

        # 1) plot each on a separate line
        list_rs = [D["rs"] for D in self.DatAll]
        list_chans = [D["chan"] for D in self.DatAll]
        list_bregion = self.sitegetter_brainregion("list_regions")

        fig, axes = plt.subplots(7,1, figsize=(15, 28), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1,1,1,12]})

        # -- Epochs
        ax = axes.flatten()[0]
        ax.set_title("epocs")
        list_plot = ["camframe", "camtrialon", "camtrialoff", "rewon", "rewoff", "behcode"]
        for i, pl in enumerate(list_plot):
            times, vals = self.extract_data_tank_epocs(pl, trial0=trialtdt)
            ax.plot(times, np.ones(times.shape)+i, 'x', label=pl)
            if pl=="behcode":
                for t, b in zip(times, vals):
                    ax.text(t, 1+i+np.random.rand(), int(b))
        ax.legend()
        # ax.set_ylim(-1, i+2)
        # Behcode
        # times, vals = extract_data_tank_epocs("behcode", trial0=trialtdt)
        # ax.plot(times, np.ones(times.shape)+i, 'x', label=pl)
        self.plotmod_overlay_trial_events(ax, trialtdt)

        # - phdi
        ax = axes.flatten()[1]
        ax.set_title("photodiodes")
        list_plot = ["pd1", "pd2"]
        for i, pl in enumerate(list_plot):
            times, vals = self.extract_data_tank_streams(pl, trial0=trialtdt)
            ax.plot(times, vals, '-', label=pl)
        ax.legend()
        self.plotmod_overlay_trial_events(ax, trialtdt)

        # -- Eye
        ax = axes.flatten()[2]
        ax.set_title("eyes")
        list_plot = ["eyex","eyey","eyediam"]
        for i, pl in enumerate(list_plot):
            times, vals = self.extract_data_tank_streams(pl, trial0=trialtdt)
            ax.plot(times, vals, '-', label=pl)
        ax.legend()
        self.plotmod_overlay_trial_events(ax, trialtdt)

        # -- audio
        ax = axes.flatten()[3]
        ax.set_title("audio")
        list_plot = ["mic"]
        for i, pl in enumerate(list_plot):
            times, vals = self.extract_data_tank_streams(pl, trial0=trialtdt)
            ax.plot(times, vals, '-', label=pl)
        ax.legend()
        self.plotmod_overlay_trial_events(ax, trialtdt)

        # Beh strokes (ml2)
        ax = axes.flatten()[4]
        self.plot_trial_timecourse_summary(ax, trialtdt)

        # ax.set_title("touch data")
        # for s in strokes:
        #     x = s[:,0]
        #     y = s[:,1]
        #     t = s[:,2]
        #     ax.plot(t, x, label="x")
        #     ax.plot(t, y, label="y")
            
        # A single raw channel 
        ax = axes.flatten()[5]
        import random
        chanrand = random.randint(1, 256)
        rsrand = random.randint(2, 3)
        ax.set_title(f"ranbdom raw data: rs{rsrand}-ch{chanrand}")
        D = self.datall_slice_single(rsrand, chanrand, trialtdt)
        t = D["tbins0"]
        raw = D["raw"]
        st = D["spike_times"]
        ax.plot(t, raw)
        # spikes
        ax.plot(st, np.ones(st.shape), 'xr')
        self.plotmod_overlay_trial_events(ax, trialtdt)

        # -- Rasters
        ax = axes.flatten()[6]
        list_ylabel = []
        cnt = 0
        for i, (rs, chan) in enumerate(zip(list_rs, list_chans)):
            d = self.datall_slice_single(rs, chan, trialtdt)
            st = d["spike_times"]
            if rsrand==rs and chanrand==chan:
                # the random one plotted, color diff 
                pcol = 'r';
            else:
                pcol = 'k'
            self.plot_raster_line(ax, st, yval=i, color=pcol)
            
            if i%32==0:
                ax.axhline(i-0.5)
                ax.text(-0.5, i-0.5, list_bregion[cnt], size=15, color="b")
                cnt+=1
            
            # collect for ylabel
            list_ylabel.append(f"{rs}-{chan}")
        ax.set_yticks(range(len(list_ylabel)))
        ax.set_yticklabels(list_ylabel);
        ax.set_xlabel('time rel. trial onset (sec)');
        ax.set_ylabel('site');
        ax.set_title(f"trialtdt: {trialtdt}")
        self.plotmod_overlay_trial_events(ax, trialtdt)    


    def plot_raster_trial_sites(self, trialtdt, list_sites, site_to_highlight=None):
        """ Plot a single raster for this trial, across these sites
        PARAMS:
        - site_to_highlight, bool, if True, colors it diff
        """
        from pythonlib.drawmodel.strokePlots import plotDatStrokes
        
        list_rschan = [self.convert_site_to_rschan(s) for s in list_sites]
        # fig, axes = plt.subplots(2,2, figsize=(17, 15), sharex=False, 
        #                          gridspec_kw={'height_ratios': [1,8], 'width_ratios':[8,1]})
        fig, axes = plt.subplots(2,2, figsize=(30, 10), sharex=False, 
                                 gridspec_kw={'height_ratios': [1,8], 'width_ratios':[8,1]})
        
        # -- Rasters
        ax = axes.flatten()[2]
        list_ylabel = []
        cnt = 0
        print(list_rschan)
        
        if site_to_highlight is not None:
            rsrand, chanrand = self.convert_site_to_rschan(site_to_highlight)

        for i, (rs, chan) in enumerate(list_rschan):
            d = self.datall_slice_single(rs, chan, trialtdt)
            st = d["spike_times"]
            if site_to_highlight is not None:
                if rsrand==rs and chanrand==chan:
                    # the random one plotted, color diff 
                    pcol = 'r';
                else:
                    pcol = 'k'
            else:
                pcol = "k"
            self.plot_raster_line(ax, st, yval=i, color=pcol)

            # collect for ylabel
            list_ylabel.append(f"{rs}-{chan}")
        ax.set_yticks(range(len(list_ylabel)))
        ax.set_yticklabels(list_ylabel);
        ax.set_xlabel('time rel. trial onset (sec)');
        ax.set_ylabel('site');
        ax.set_title(f"trialtdt: {trialtdt}")
        self.plotmod_overlay_trial_events(ax, trialtdt)
        XLIM = ax.get_xlim()
        
        # Beh strokes (ml2)
        ax = axes.flatten()[0]
        self.plot_trial_timecourse_summary(ax, trialtdt)
        ax.set_xlim(XLIM)
        
        # Final drawing
        fd, trialml = self.beh_get_fd_trial(trialtdt)

        ax = axes.flatten()[3]
        strokes = getTrialsStrokes(fd, trialml)
        plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, number_from_zero=True)
        
        # Image
        ax = axes.flatten()[1]
    #     plotTrialSimple(fd, trialml, ax=ax, plot_task_stimulus=True, nakedplot=True, plot_drawing_behavior=False)
        strokestask = getTrialsTaskAsStrokes(fd, trialml)
        plotDatStrokes(strokestask, ax, clean_task=True)
        
    #     plotTrialSimple(fd, trialml, ax=ax, plot_task_stimulus=True, nakedplot=True, plot_drawing_behavior=False)

        return fig, axes
        
