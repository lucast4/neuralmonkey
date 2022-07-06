""" Holds a single recording file
 - i.e, combines neural, beh, and camera
"""
import tdt
import numpy as np
import matplotlib.pyplot as plt
from ..utils.timeseries import dat_to_time
# from ..utils.monkeylogic import getTrialsTaskAsStrokes
from pythonlib.drawmodel.strokePlots import plotDatStrokes
from ..utils import monkeylogic as mkl
import pickle
import os


class Session(object):
    """
    Operates over all trials for this entire session (day), even if multiple rec files for today

    """
    
    def __init__(self, datestr, beh_expt_list, beh_sess_list, 
            beh_trial_map_list=None,
            sites_garbage = None,
            expt="Lucas512-220520-115835", animal="Pancho", 
            path_base = "/mnt/hopfield_data01/ltian/recordings", 
            path_local = "/data2/recordings",
            rec_session=0, do_all_copy_to_local=False):
        """
        PARAMS:
        - datestr, string, YYMMDD, e.g, "220609"
        - beh_expt_list, lsit of string, for each beh session you wish to load.
        - beh_trial_map_list
        e.g., [(20,0), (1,200)] means that the first fd's trial 20 maps onto trial 0 neural and
        the second fd starts (trial 1) from trial 200 neural. Pass in None to try to autoamtically 
        figure out. does so by assuming that neural and beh recording starting on the same trial.
        - expt, string, expt in Synapse (TDT)
        - animal, string
        - TrialBehAtNeuralZero, int, what trial was ml on the first (zero) trial for 
        neural? If started botrh nerual and beh at same time, then this will be 1.
        - path_base, full path to the base, where recordings are in <path_base>/<animal>...
        - path_local, local path with fast read/write, for saving tank data and spikes (first time loading).
        subsequent times will try to load from here.
        - rec_session, int, which session, from 0, 1, ... Currently Session only holds one rec
        session.
        - do_all_copy_to_local, bool, if True, then does various things (once) which can take a while, but will
        speed up loading in future. even if this False, will do this for spikes and tdt tank
        """

        self.Animal = animal
        self.ExptSynapse = expt
        self.Date = datestr
        self.Paths = None

        self.Chans = range(1, 257)
        self.Rss = [2, 3]
        self.DatAll = None

        self.SitesGarbage = sites_garbage
        self.SitesAll = range(1, 513)

        # Neural stuff
        self.RecPathBase = path_base
        self.RecSession = rec_session
        self.RecPathBaseLocal = path_local

        # Behavior stuff
        self.BehDate = datestr
        self.BehExptList = beh_expt_list
        self.BehSessList = beh_sess_list
        self.BehTrialMapList = beh_trial_map_list
        self.BehFdList = []
        self.BehTrialMapListGood = None

        # self._initialize_params()
        self._initialize_paths()

        # for k, v in self.Paths.items():
        #     print(k, ' -- ' , v)
        # assert False

        # Load raw things
        print("== Loading TDT tank")
        self.load_tdt_tank()
        print("== Done")

        # Save tank data and spikes locally for faster reload.


        # Find the times of all trial onsets (inneural data)
        # 1. get all onset and offset times
        self.TrialsOnset = self.behcode_extract_times(9)
        self.TrialsOffset = self.behcode_extract_times(18)

        # Load beh 
        print("== Loading ml2 behavior")
        self.load_behavior()
        print("== Done")

        # Load raw data
        print("== Loading spike times")
        self.load_spike_times() # Load all spike times
        print("== Done")

        # behcodes
        beh_codes = {
                9:"start",
                10:"fix cue",
                11:"fix cue visible",
                13:"frame skip",
                14:"manual rew",
                15:"guide",
                16:'FixationOnsetWTH',
                17:'FixationDoneSuccessWTH',
                18:"end",
                19:'FixationRaiseFailWTH',
                20:"go (draw)",
                21:"guide_on_GA",
                30:'DelayWhatIsThis', 
                40:'GoWhatIsThis',
                41:"samp1 on",
                42:"samp1 off",
                45:"done",
                46:"post",
                50:"reward",
                51:"free reward",
                61:'DoneButtonVisible',
                62:'DoneButtonTouched',
                63:'DragAroundSuccess',
                64:'DragAroundAbort',
                65:'DragAroundFirstAbortNow',
                70:'hotkey_x',
                71:'DAstimevent_firstpres',
                72:'DAstimoff_finibeforepause',
                73:'DAstimoff_fini',
                74:'DAsamp1_visible_change',
                75:'DAnewpnutthisframe',
                76:'DAsound_samp1touched',
                78:'DAsound_gotallink',
                80:'ttl_trialon',
                81:'ttl_trialoff',
                91:'GAstimevent_firstpres', 
                92:'GAstimoff_fini', 
                101:'fix_square_on',
                102:'fix_square_off',
                103:'fix_square_on_pd',
                111:'photodiode_force_off',
                200:'skipped_movie_frame'
            }
        self.BehCodes = beh_codes

        if do_all_copy_to_local:
            # Copy spike waveforms saved during tdt thresholding and extraction
            self.load_and_save_spike_waveform_images()
            # Load raw and dupl and compare them (sanity check)
            self.plot_raw_dupl_sanity_check()


    ####################### PREPROCESS THINGS
    # def _initialize_params(self):
    def _initialize_paths(self):

        # 1) find all recordings for this date
        from pythonlib.tools.expttools import findPath, deconstruct_filename
        path_hierarchy = [
            [self.Animal],
            [self.Date]
        ]
        paths = findPath(self.RecPathBase, path_hierarchy)
        # assert len(paths)==1, 'not yhet coded for combining sessions'
        paththis = paths[self.RecSession]

        fnparts = deconstruct_filename(paththis)
        print(fnparts)
        final_dir_name = fnparts["filename_final_noext"]

        # Local cached
        pathbase_local = f"{self.RecPathBaseLocal}/{self.Animal}/{self.Date}/{final_dir_name}"
        import os
        os.makedirs(pathbase_local, exist_ok=True)

        pathdict = {
            "raws":paththis,
            "tank":f"{paththis}/{fnparts['filename_final_noext']}",
            "spikes":f"{paththis}/spikes_tdt_quick",
            "final_dir_name":final_dir_name,
            "time":fnparts["filename_components_hyphened"][2],
            "pathbase_local":pathbase_local,
            "tank_local":f"{pathbase_local}/data_tank.pkl",
            "spikes_local":f"{pathbase_local}/data_spikes.pkl",
            "datall_local":f"{pathbase_local}/data_datall.pkl",
            "figs_local":f"{pathbase_local}/figs"
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
        for e, s in zip(self.BehExptList, self.BehSessList):

            # Load filedata
            # a = "Pancho"
            # d = 220531
            # e = "charneuraltrain1b"
            # s = 1
            fd = loadSingleDataQuick(self.Animal, self.Date, e, s)

            self.BehFdList.append(fd)

        # Try to automatically determine trial map list
        if False:
            # in progress. one issue is that neural can have more trial onsets than in beh, with some
            # neural trials not even recorded in beh (e..g, the last trial before exit)
            print(len(self.TrialsOnset))
            for fd in self.BehFdList:
                print(len(mkl.getIndsTrials(fd)))

            # Todo from here: infer the mapping.
            assert False




    def load_tdt_tank(self):
        """ Holds all non-nueral signals. is aligned to the neural
        """

        # First, try to load from local (much faster)
        import os
        if os.path.exists(self.Paths["tank_local"]):
            print("** Loading tank data from local (previusly cached)")
            with open(self.Paths["tank_local"], "rb") as f:
                data_tank = pickle.load(f)
            self.DatTank = data_tank
        else:
            print("** Loading data from tdt tank")
            # data_tank = tdt.read_block(self.PathTank, evtype = ["epocs", "streams"])
            data_tank = tdt.read_block(self.PathTank, evtype = ["epocs"])
            self.DatTank = data_tank

            # save this for later
            self._savelocal_tdt_tank()
            # with open(self.Paths["tank_local"], "wb") as f:
            #     pickle.dump(data_tank, f)


    def _savelocal_tdt_tank(self):
        """ save this for later
        """
        print("Saving TDT Tank locally to: ", self.Paths["tank_local"])
        with open(self.Paths["tank_local"], "wb") as f:
            pickle.dump(self.DatTank, f)


    def load_spike_times(self):
        """ Load and strore all spike times (across all trials, and chans)
        Not yet aligned to trials, etc.
        """
        # Extract database of spikes (not sliced by trials yet)
        # chans = range(1,256+1)
        # rss = [2,3]

        # First, try to load from local (much faster)
        import os
        if os.path.exists(self.Paths["spikes_local"]):
            # Load quickly from local
            print("** Loading spike data from local (previusly cached)")
            with open(self.Paths["spikes_local"], "rb") as f:
                DatSpikes = pickle.load(f)
            self.DatSpikes = DatSpikes
        else:
            # Load from server
            def load_spike_times_(rs, chan, ver="spikes_tdt_quick"):
                """ Return spike times, pre-extracted elsewhere (matlab)
                in secs
                """
                import scipy.io as sio
                fn = f"{self.Paths['spikes']}/RSn{rs}-{chan}"
                mat_dict = sio.loadmat(fn)
                return mat_dict["spiketimes"]

            def load_spike_times_mult(rss, chans, ver="spikes_tdt_quick"):
                DatSpikes = []
                for rs in rss:
                    for ch in chans:
                        print(rs, ch)
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

            # Save for faster loading later
            self._savelocal_spikes()

    def load_and_save_spike_waveform_images(self, spikes_ver = "spikes_tdt_quick"):
        """ Copies images from server to local, of extracted spikes
        """
        from pythonlib.tools.expttools import deconstruct_filename
        import glob
        import shutil
        import os

        # 1) target folder
        sdir = f"{self.Paths['pathbase_local']}/{spikes_ver}"
        os.makedirs(sdir, exist_ok=True)

        # 2) copy over images
        images = glob.glob(f"{self.Paths['spikes']}/*.png")
        for im in images:
            x = deconstruct_filename(im)
            
            targ = f"{sdir}/{x['filename_final_ext']}"
            if os.path.exists(targ):
                print(f"Skipping, since already copied: {targ}")
            else:
                shutil.copy2(im, targ)
                print(f"Copied : {x['filename_final_ext']} to {targ}")
        
        print("DONE!")
        

    def _savelocal_spikes(self):
        """  Save for faster loading later
        """
        print("Saving spikes locally to: ", self.Paths["spikes_local"])
        with open(self.Paths["spikes_local"], "wb") as f:
            pickle.dump(self.DatSpikes, f)


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
                    raw = np.empty(0)
                    fs = None
                    t = np.empty(0)
                    
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

    def extract_raw_and_spikes_helper(self, trials=None, sites=None, get_raw=False):
        """ to quickly get a subset of trials, all sites, etc.
        PARAMS:
        - trials, list of ints, if None, then gets all.
        - sites, list of ints, if None, then gets all.
        NOTE:
        - if trials and sites are None, then automatically saves and loads locally.
        """

        if sites is None and trials is None:
            # Then try to load locally.
            import os
            if os.path.exists(self.Paths["datall_local"]):
                # Load quickly from local
                print("** Loading datall from local (previusly cached)")
                with open(self.Paths["datall_local"], "rb") as f:
                    self.DatAll = pickle.load(f)
        else:
            # Then extract from loaded spikes and tank data.
            if sites is None:
                # get all sites
                sites = self.SitesAll
            if trials is None:
                trials = range(len(SN.TrialsOnset))

            # convert sites to rs and chan
            rss, chans = [], []
            for s in sites:
                rs, ch = self.convert_site_to_rschan(s)
                if rs not in rss:
                    rss.append(rs)
                if ch not in chans:
                    chans.append(ch)

            for t in trials:
                print(f"Extrcting data for trial {t}")
                self.extract_raw_and_spikes(rss, chans, t, get_raw=get_raw)

            # Save
            self._savelocal_datall()


    def extract_raw_and_spikes(self, rss, chans, trialtdt, get_raw = False):
        """ [GET ALL DATA] Extract raw (unfiltered) and spikes for this trial
        PARAMS:
        - rss, list of ints for rs
        - chans, list of chans (within this rs)
        - trialtdt, tdt trials
        TODO: 
        - dont extract if already gottne.
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
                # Don't inlcude it if it is already extracted
                # if getting raw, then will overwrite if it was gotten but without raw.
                d_old = self.datall_slice_single(d["rs"], d["chan"], d["trial0"])
                if d_old is None:
                    # then doestn exist, append
                    self.DatAll.append(d)
                elif len(d_old["raw"])==0 and get_raw:
                    # Then previous didnt have raw, so overwrite it
                    self.datall_replace_single(d["rs"], d["chan"], d["trial0"], Dnew=d)
                else:
                    # skip
                    pass

                # if not self.datall_this_exists(d["rs"], d["chan"], d["trial0"],
                #     also_check_if_has_raw_data=get_raw):
                #     # then append
                #     self.DatAll.append(d)
        return self.DatAll


    # def load_all_data_this_trial_(self, trial0, rss, chans):
    #     # Extract both raw and spikes for a given channel and trial
    #     # chans = list(range(1, 256+1))
    #     # rss = [2,3]
    #     # trial0 = 267
    #     DatAll = self.extract_raw_and_spikes(rss, chans, trial0)

    def _savelocal_datall(self):
        """ save this for later
        This is what is extracted by extract_raw_and_spikes. it is all data aligned to each trial.
        """
        print("Saving DatAll (raw and spikes) locally to: ", self.Paths["tank_local"])
        with open(self.Paths["datall_local"], "wb") as f:
            pickle.dump(self.DatAll, f)

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
    def datall_this_exists(self, rs, chan, trial0, also_check_if_has_raw_data=False):
        """ returns True if this combo exist sin self.DatAll,
        false otherwise
        - also_check_if_has_raw_data, bool, if true, then to return True must also 
        hvae raw data extracted
        """
        D = self.datall_slice_single(rs, chan, trial0)
        if D is None:
            return False
        if also_check_if_has_raw_data:
            if len(D["raw"])==0:
                return False
            else:
                return True
        else:
            return True


    def datall_replace_single(self, rs, chan, trial0, Dnew):
        """ If this (rs, chan, trial0) exists, replace it with Dnew
        MODIFIES:
        - self.DatAll, at a single index
        """

        Dold, idx = self.datall_slice_single(rs, chan, trial0, return_index=True)
        if Dold is None:
            # then doesnt exist, do mnothing
            pass
        else:
            self.DatAll[idx] = Dnew


    def datall_slice_single(self, rs, chan, trial0, return_index=False):
        """ Slice a single chans data.
        PARAMS:
        - rs, chan, trial0, ints
        RETURNS:
        - if exists:
        --- siungle item from self.DatAll if it exists, 
        --- [if return_index]:index (in self.DatAll)
        - if doesnt exist:
        --- None, 
        --- [return_index] None
        """

        if self.DatAll is None:
            return None
        for i, D in enumerate(self.DatAll):
            if D["rs"]==rs and D["chan"]==chan and D["trial0"]==trial0:
                if return_index:
                    return D, i
                else:
                    return D
        if return_index:
            return None, None
        else:
            return None


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
    def _beh_get_fdnum_trial(self, trialtdt):
        """ Get the filedata indices and trial indices (beh) for
        this neural trial (trialtdt)
        """
        from ..utils.conversions import get_map_trial_and_set
        ntrials = len(self.TrialsOnset)
        assert trialtdt < ntrials, "This tdt trial doesnt exist, too large..."

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
        fd_setnum, fd_trialnum = self._beh_get_fdnum_trial(trialtdt)
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
    def sitegetter_summarytext(self, site):
        """ Return a string that useful for labeling
        """

        info = self.sitegetter_thissite_info(site)
        bregion = info["region"]
        rs = info["rs"]
        chan = info["chan"]
        return f"{site}|{bregion}|{rs}-{chan}"

    def sitegetter_brainregion_chan(self, region, chan):
        """ Given a regin and chan (1-256) return its site (1-512)
        """ 

        # which rs?
        sites = self.sitegetter_brainregion(region)
        if all([s<257 for s in sites]):
            rs = 2
        elif all([s<513 for s in sites]):
            rs = 3
        else:
            print(sites, region)
            assert False

        site = self.convert_rschan_to_site(rs, chan)
        assert site in sites, "this site not in this brain region!!"
        return site


    def sitegetter_thissite_info(self, site):
        """ returns info for this site in a dict
        """

        # Get the brain region
        dict_sites = self.sitegetter_brainregion()
        regionthis = None
        for bregion, sites in dict_sites.items():
            if site in sites:
                regionthis = bregion
                break
        assert regionthis is not None

        # Get the rs and chan
        rs, chan = self.convert_site_to_rschan(site)

        return {
            "region":regionthis,
            "rs":rs,
            "chan":chan}

    def sitegetter_all(self, list_regions=None, clean=True):
        """ Get all sites, in order
        MNOTE: will not be in order of list_regions, but will be in order in channels.
        PARAMS:
        - list_regions, get only these regions. leave None to get all
        """
        bregion_mapper = self.sitegetter_brainregion("mapper", clean=clean)
        if list_regions is None:
            bm = bregion_mapper
        else:
            bm = {br:sites for br, sites in bregion_mapper.items() if br in list_regions}

        sites = [s for br, ss in bm.items() for s in ss]
        return sites

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

    def convert_rschan_to_site(self, rs, chan):
        """
        PARAMS:
        - rs, {2,3}
        - chan, 1-256
        """
        assert chan<257
        if rs==2:
            return chan
        elif rs==3:
            return chan + 256

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
    def behcode_extract_times(self, code, trial0=None, 
            first_instance_only=False, shorthand=False):
        """ Extract the times that this code occured, in sec, as recorded by TDT 
        _NOT_ as recorded in ml2. these are almost identical. These are the "correct" values
        wrt alignemnet with neural data.
        PARAMS
        - code, eitehr either string or int
        - trial0, int, if given, then will get times reltaive to this trial
        - first_instance_only, bool, if True, then only returns the first time
        it occurs.
        """    

        if isinstance(code, str):
            codenum = self.behcode_convert(codename=code, shorthand=shorthand)
        else:
            codenum = code

        timesall, codes = self.extract_data_tank_epocs("behcode")
        inds = codes==codenum
        times = timesall[inds]
        
        if trial0 is not None:
            times, _ = self.extract_windowed_data_bytrial(times, trial0)
        
        if first_instance_only:
            if len(times)>1:
                times = times[0:1] # stay as an array
        return times
        
    def behcode_extract_times_sequence(self, codes, trial0):
        """ Get the times of this sequence of codes, aligned to this trial,
        taking only the first instance of each code.
        If a code doesnt occur, gives it a np.nan
        RETURNS:
        - times, np array len of codes. elements are either seconds or np.nan
        NOTE: order will not matter
        """
        
        times = np.empty(len(codes))
        for i, c in enumerate(codes):
            t = self.behcode_extract_times(c, trial0, True)
            if len(t)>0:
                times[i] = t[0]
            else:
                times[i] = np.nan
        return times    

    def behcode_shorthand(self, full=None, short=None):
        """ Convert between
        """

        dict_full2short = {
            "start":"on",
            "fix cue visible":"fixcue",
            "FixationOnsetWTH":"fix",
            "GAstimevent_firstpres":"samp", 
            "GAstimoff_fini":"go",
            "DoneButtonTouched":"done",
            "DAstimoff_fini":"fb_vs", 
            "reward":"rew"}

        if full:
            assert not short
            return dict_full2short[full]
        else:
            assert not full
            for k, v in dict_full2short.items():
                if v==short:
                    return k
            assert False

    def behcode_convert(self, codenum=None, codename=None, shorthand=False):
        """ convert between num (int) and name (str)
        Name can be the formal name in BehCodes, or shorthand
        PARAMS:
        - shorthand, bool, then 
        --- if ask for name, returns codename that is shorthand. 
        --- if ask for num, then assumes you passed in shortand
        """

        if codenum:
            assert codename is None
            name = self.BehCodes[codenum]
            if shorthand:
                return self.behcode_shorthand(full=name)
            else:
                return name
        else:
            assert codenum is None
            if shorthand:
                # convert to full
                codename = self.behcode_shorthand(short=codename)
            for k, v in self.BehCodes.items():
                if v==codename:
                    return k    
            assert False
        

    ###################### GET TEMPORAL EVENTS
    def events_get_time(self, event, trial):
        """ Return the time in trial for this event
        PARAMS:
        - event, either string or tuple.
        - eventkind, string, if None, then tries to find it automatically.
        """

        if isinstance(event, tuple):
            eventkind = event[0]            
            if eventkind=="strokes":
                # event = (strokes, 1, "on")
                strokenum = event[1] # 0, 1, .. -1
                timepoint = event[2] # on, off
                ons, offs = self.strokes_extract_ons_offs(trial)
                if timepoint=="on":
                    alignto_time = ons[strokenum]
                elif timepoint=="off":
                    alignto_time = offs[strokenum]
                else:
                    assert False
            else:
                print(eventkind)
                assert False
        elif isinstance(event, str):
            # THis is behcode shorthand
            code = self.behcode_convert(codename=event, shorthand=True)
            alignto_time = self.behcode_extract_times(code, trial, first_instance_only=True)
        elif isinstance(event, int):
            # Then is behcode
            alignto_time = self.behcode_extract_times(code, trial, first_instance_only=True)
        else:
            assert False

        return alignto_time


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
        
    
    def plot_raster_line(self, ax, times, yval, color='k', alignto_time=None, 
        linelengths = 0.85, alpha=0.4):
        """ plot a single raster line at times at row yval
        PARAMS:
        - alignto_time, in sec, this becomes new 0
        """
        
        if alignto_time:
            t = times-alignto_time
        else:
            t = times
    #     ax.plot(times, yval*np.ones(time.shape), '.', color=color, alpha=0.55)
        y = yval*np.ones(t.shape)
        ax.eventplot([t], lineoffsets=yval, color=color, alpha=alpha, linelengths=linelengths)
        
        # plot as hashes
        
    def strokes_extract_ons_offs(self, trialtdt):
        from ..utils.monkeylogic import getTrialsOnsOffsAllowFail
        fd, trialml = self.beh_get_fd_trial(trialtdt)
        ons, offs = getTrialsOnsOffsAllowFail(fd, trialml)
        return ons, offs

    def plotmod_overlay_brainregions(self, ax, list_sites):
        """ 
        PARAMS:
        - list_sites, list of ints, must be lnegth of the num raster lines, and must
        be ordered matchng the figure
        """

        XLIM = ax.get_xlim()
        # 1) plot horizonatl line on figure demarcatring transtion ebtwenn areas
        list_regions = [self.sitegetter_thissite_info(site)["region"] for site in list_sites]
        region_previous = None
        for i, region_this in enumerate(list_regions):
            if region_this==region_previous:
                # then have not transitioned between regions
                continue
            else:
                ax.axhline(i-0.5)
                ax.text(XLIM[0]-1.5, i, region_this, size=15, color="b")
            region_previous = region_this






    def plotmod_overlay_trial_events(self, ax, trial0, strokes_patches=True, 
            alignto_time=None, only_on_edge=None, YLIM=None, alpha = 0.2,
            which_events=["trial", "strokes"], include_text=True, text_yshift = 0.):
        """ Overlines trial events in vertical lines
        Time is rel trial onset (ml2 code 9)
        Run this after everything else, so get propoer YLIM.
        PARAMS:
        - strokes_patches, bool, then fills in patches for each stroke. otherwise places vert lines
        at onset and ofset.
        - only_on_edge, whethe and how to plot on edge, without blocking the figure, either:
        --- None, skip
        --- "top", or "bottom"
        - YLIM, to explicitly define, useful if doing one per raster line.
        - which_events, list of str, control whhic to plot
        - text_yshift, shfit text by this amnt.
        """

        if "trial" in which_events:
            # Vertical lines for beh codes
            list_codes = [9, 11, 16, 91, 92, 62, 73, 50]
            times_codes = self.behcode_extract_times_sequence(list_codes, trial0)
            # names_codes = [beh_codes[c] for c in list_codes]
            names_codes = [self.behcode_convert(c, shorthand=True) for c in list_codes] 
            # names_codes = ["on", "fixcue", "fix", "samp", "go", "done", "fb_vs", "rew"]
            colors_codes = ["k",  "m",       "b",    "r",   "y",  "g",     "m",     "k"]
        else:
            list_codes = []
            times_codes = []
            names_codes = [] 
            colors_codes = []


        # Also include times of strokes
        if strokes_patches==False and "strokes" in which_events:
            ons, offs = self.strokes_extract_ons_offs(trial0)
            times_codes = np.append(times_codes, ons)
            times_codes = np.append(times_codes, offs)
            names_codes.extend(["Son" for _ in range(len(ons))])
            colors_codes.extend(["b" for _ in range(len(ons))])
            names_codes.extend(["Soff" for _ in range(len(offs))])
            colors_codes.extend(["m" for _ in range(len(offs))])
        
        if alignto_time is not None:
            times_codes = times_codes - alignto_time

        if not YLIM:
            YLIM = ax.get_ylim()

        for time, name, col in zip(times_codes, names_codes, colors_codes):
            if only_on_edge:
                if only_on_edge=="top":
                    ax.plot(time, YLIM[1], "v", color=col)
                    y_text = YLIM[1]
                elif only_on_edge=="bottom":
                    ax.plot(time, YLIM[0], "^", color=col)
                    y_text = YLIM[0]
                else:
                    assert False
            else:
        #         if name in ["Son"]:
        # #             ax.axvline(time, color="b", ls="--", alpha=0.5)
        #             ax.axvline(time, color="b", ls="-", alpha=0.5)
        #         elif name in ["Soff"]:
        # #             ax.axvline(time, color="m", ls="--", alpha=0.5)
        #             ax.axvline(time, color="m", ls="-", alpha=0.5)
        #         else:
                ax.axvline(time, color=col, ls="--", alpha=0.7)
                y_text = YLIM[0]
            if include_text:
                y_text = y_text + text_yshift
                ax.text(time, y_text, name, rotation="vertical", fontsize=14)

        # color in stroke times

        if strokes_patches and "strokes" in which_events:
            from matplotlib.patches import Rectangle
            ons, offs = self.strokes_extract_ons_offs(trial0)
            if alignto_time:
                ons = [o - alignto_time for o in ons]
                offs = [o - alignto_time for o in offs]

            for on, of in zip(ons, offs):
                if only_on_edge:
                    if only_on_edge=="top":
                        ax.hlines(YLIM[1], on, of, color="r")
                    elif only_on_edge=="bottom":
                        ax.hlines(YLIM[0], on, of, color="r")
                    else:
                        assert False
                        # ax., YLIM[1], "v", color=col)
                else:
                    rect = Rectangle((on, YLIM[0]), of-on, YLIM[1]-YLIM[0], 
                        linewidth=1, edgecolor='r',facecolor='r', alpha=alpha)
                    ax.add_patch(rect)


    def plot_trial_timecourse_summary(self, ax, trial0, number_strokes=True):
        # trialml = convert_trialnum(trialtdt=trial0)
        strokes = self.strokes_extract(trial0)

        # Beh strokes (ml2)
        for i, s in enumerate(strokes):
            x = s[:,0]
            y = s[:,1]
            t = s[:,2]
            ax.plot(t, x, label="x", color='b')
            ax.plot(t, y, label="y", color='r')
            
            if number_strokes:
                ax.text(np.mean(t), np.max(np.r_[x,y]), i)
            
        self.plotmod_overlay_trial_events(ax, trial0)

    def plot_final_drawing(self, ax, trialtdt, strokes_only=False):
        """ plot the drawing
        PARAMS:
        - strokes_only, then just pnuts
        """
        strokes = self.strokes_extract(trialtdt, peanuts_only=strokes_only)
        plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, number_from_zero=True)


    def strokes_extract(self, trialtdt, peanuts_only=False):
        from ..utils.monkeylogic import getTrialsStrokes, getTrialsStrokesByPeanuts
        fd, trialml = self.beh_get_fd_trial(trialtdt)
        if peanuts_only:
            strokes = getTrialsStrokesByPeanuts(fd, trialml)
        else:
            strokes = getTrialsStrokes(fd, trialml)
        return strokes


    def plot_taskimage(self, ax, trialtdt):
    #     plotTrialSimple(fd, trialml, ax=ax, plot_task_stimulus=True, nakedplot=True, plot_drawing_behavior=False)
        from ..utils.monkeylogic import getTrialsTaskAsStrokes
        fd, trialml = self.beh_get_fd_trial(trialtdt)
        strokestask = getTrialsTaskAsStrokes(fd, trialml)
        plotDatStrokes(strokestask, ax, clean_task=True)
        
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
        if D is not None:
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

    def plot_raster_multrials_onesite(self, list_trials, site, alignto=None, SIZE=0.5):
        """ Plot one site, mult trials, overlaying for each trial its major events
        PARAMS:
        - list_trials, list of int
        - site, int (512)
        - alignto, str or None, how to adjust times to realign.
        """

        nrows = len(list_trials)
        ncols = 2
        alpha_raster = 0.4
        fig, axes = plt.subplots(1, ncols, sharex=True, figsize=(25, SIZE*nrows), 
                               gridspec_kw={'width_ratios': [9,1]})
        ax = axes.flatten()[0]
        for i, trial in enumerate(list_trials):
            
            # get time of this event (first instance)
            if alignto:
                alignto_time = self.events_get_time(alignto, trial)
            else:
                alignto_time = None

            # Rasters
            rs, chan = self.convert_site_to_rschan(site)
            D = self.datall_slice_single(rs, chan, trial0=trial)
            spikes = D['spike_times']
            self.plot_raster_line(ax, spikes, i, alignto_time=alignto_time, linelengths=0.8, alpha=alpha_raster)
            
            # - overlay beh things
        #     SN.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, only_on_edge="top")
            if i==0:
                include_text = True
            else:
                include_text = False
            self.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, only_on_edge="bottom", 
                                            YLIM=[i-0.3, i+0.5], which_events=["trial"], 
                                            include_text=include_text, text_yshift = -0.5)
        #     SN.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
        #                                     YLIM=[i-0.3, i+0.3], which_events=["strokes"], only_on_edge="top")
        #     SN.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
        #                                     YLIM=[i-0.3, i+0.3], which_events=["strokes"], only_on_edge="top")
        #     SN.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
        #                                     YLIM=[i-0.4, i-0.2], which_events=["strokes"], alpha=0.3)
        #     SN.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
        #                                     YLIM=[i+0.2, i+0.4], which_events=["strokes"], alpha=0.3)
            self.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
                                            YLIM=[i-0.4, i-0.2], which_events=["strokes"], alpha=0.3)
            
        ax.set_yticks(range(len(list_trials)))
        ax.set_yticklabels(list_trials);
        ax.set_ylabel('trial');
        ax.set_title(self.sitegetter_summarytext(site))

            # Final drawing
        #     ax = axes.flatten()[2*i + 1]
        #     SN.plot_final_drawing(ax, trial)
        # Plot each drawing
        SIZE = 2
        fig_draw, axes_draw = plt.subplots(nrows, 2, sharex=True, figsize=(5, SIZE*nrows))
        for i, trial in enumerate(list_trials):
            
            # Final drawing
            ax = axes_draw.flatten()[1+ i*2]
            ax.set_title(f"trial_{trial}")
            self.plot_final_drawing(ax, trial, strokes_only=True)
            
            # task image
            ax = axes_draw.flatten()[i*2]
            ax.set_title(f"trial_{trial}")
            self.plot_taskimage(ax, trial)
            

        return fig, axes, fig_draw, axes_draw


    def plot_raster_oneetrial_multsites(self, trialtdt, list_sites, site_to_highlight=None,
            WIDTH=20, HEIGHT = 10):
        """ Plot a single raster for this trial, across these sites
        PARAMS:
        - site_to_highlight, bool, if True, colors it diff
        """
        
        list_rschan = [self.convert_site_to_rschan(s) for s in list_sites]
        # fig, axes = plt.subplots(2,2, figsize=(17, 15), sharex=False, 
        #                          gridspec_kw={'height_ratios': [1,8], 'width_ratios':[8,1]})
        fig, axes = plt.subplots(2,2, figsize=(WIDTH, HEIGHT), sharex=False, 
                                 gridspec_kw={'height_ratios': [1,8], 'width_ratios':[8,1]})
        
        # -- Rasters
        ax = axes.flatten()[2]
        list_ylabel = []
        cnt = 0
        
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
            self.plot_raster_line(ax, st, yval=i, color=pcol, linelengths=1, alpha=0.5)

            # collect for ylabel
            list_ylabel.append(f"{rs}-{chan}")
        ax.set_yticks(range(len(list_ylabel)))
        ax.set_yticklabels(list_ylabel);
        ax.set_xlabel('time rel. trial onset (sec)');
        ax.set_ylabel('site');
        ax.set_title(f"trialtdt: {trialtdt}| nsites: {len(list_sites)}")
        self.plotmod_overlay_trial_events(ax, trialtdt)
        XLIM = ax.get_xlim()
        # - Overlay brain regions
        self.plotmod_overlay_brainregions(ax, list_sites)
        
        # Beh strokes (ml2)
        ax = axes.flatten()[0]
        self.plot_trial_timecourse_summary(ax, trialtdt)
        ax.set_xlim(XLIM)
        
        # Final drawing
        ax = axes.flatten()[3]
        self.plot_final_drawing(ax, trialtdt, strokes_only=True)
        
        # Image
        ax = axes.flatten()[1]
        self.plot_taskimage(ax, trialtdt)

    #     plotTrialSimple(fd, trialml, ax=ax, plot_task_stimulus=True, nakedplot=True, plot_drawing_behavior=False)

        return fig, axes

    def plot_raw_dupl_sanity_check(self, trial = 0):
        """ Plot dupl (saved in TDT tank) on top of raw (RS4)
        to confirm they are identical, or maybe shifted by a sample or two.
        Need to have extracted raw data first (for chan 1)
        PARAMS:
        - trial, int, Extract raw for an example trial
        """

        # For saving
        sdir = f"{self.Paths['figs_local']}/sanity_checks"
        os.makedirs(sdir, exist_ok=True)

        # One plot for each rs
        rss = [2,3]
        chan = 1
        for zoom in [True, False]:
            fig, axes = plt.subplots(2,1,figsize=(10,8), sharex=True)
            for rs, ax in zip(rss, axes.flatten()):

                # Extract dupl
                if rs==2:
                    duplver = "duplicate1"
                elif rs==3:
                    duplver = "duplicate2"
                else:
                    assert False
                tdup, valsdup = self.extract_data_tank_streams(duplver, trial0=trial)

                # Extract raw if not already done
                if not self.check_raw_extracted(rs, chan, trial):
                    # Then extract
                    sites = [self.convert_rschan_to_site(rs, chan) for rs in rss]
                    self.extract_raw_and_spikes_helper(trials=[trial], sites=sites, get_raw=True)
                raw = self.datall_slice_single(rs, chan, trial)
                t = raw["tbins0"]
                vals = raw["raw"]

                if zoom:
                    ind1 = 100
                    tdup = tdup[ind1:ind1+10]
                    valsdup = valsdup[ind1:ind1+10]
                    t = t[ind1:ind1+10]
                    vals = vals[ind1:ind1+10]

                ax.plot(tdup, valsdup, color='r', label="dupl")
                ax.plot(t, vals, color='k', label="rs4")
                ax.set_title(f"{duplver}(r) and rs4 (k)")
                ax.legend()

            path = f"{sdir}/duplvsraw_trial_{trial}-chan_{chan}-zoom_{zoom}.pdf"
            fig.savefig(path)
            print(f"Saved at: {path}")


    #################### CHECK THINGS
    def check_what_extracted(self, rs, chan, trial):
        """ Return a dict of what has been extracted for this rs, chan, trial
        combo, in self.DatAll
        """
        dat = self.datall_slice_single(rs, chan, trial)

        outdict = {}

        if dat is None:
            outdict["spikes"] = False
            outdict["raw"] = False
        else:
            outdict["spikes"] = dat["spike_times"] is not None
            outdict["raw"] = len(dat["raw"])>0

        return outdict

    def check_raw_extracted(self, rs, chan, trial):
        """ Check whether this rs, chan, trial has extracted raw data yet
        RETURNS:
        - bool
        """
        dat = self.check_what_extracted(rs, chan, trial)
        return dat["raw"]

    ##################### PRINT SUMMARIES
    def print_summarize_datall(self, only_print_if_has_raw=False):
        """ Print out what data extracted 
        PARAMS;
        - only_print_if_has_raw, bool, then only prints cases with raw neural data extracted
        """
        for D in self.DatAll:
            gotspikes = D["spike_times"] is not None
            gotraw = len(D["raw"])>0
            if only_print_if_has_raw and gotraw==False:
                continue
            print(f"{D['rs']}-{D['chan']}-t{D['trial0']}-spikes={gotspikes} - raw={gotraw}")
