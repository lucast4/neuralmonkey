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

PATH_NEURALMONKEY = "/data1/code/python/neuralmonkey/neuralmonkey"

def load_session_helper(DATE, dataset_beh_expt, rec_session=0, animal="Pancho", expt="*"):
    """ Load a single recording session.
    PARAMS:
    - DATE, str, "yymmdd"
    - dataset_beh_expt, string, name of beh expt to link to this.
    - rec_session, int, within-day session, 0,1,2, ... 
    assumes one-to-one mapping between neural and beh sessions.
    - animal, str
    - expt, for finding raw beh data
    RETURNS:
    - SN, session
    """

    # 1) Find the raw beh data (filedata)
    # Assume that the beh sessions increment in order, matching the neural sessions
    sessdict = mkl.getSessionsList(animal, datelist=[DATE])
    # assume that beh sessions are indexed by neural rec sessions
    # beh_session = rec_session+1    
    beh_session = sessdict[DATE][rec_session][0]
    print("Beh Sessions that exist on this date: ", DATE, sessdict)
    print("taking this one :", beh_session)
    # beh_session = rec_session+1 # 1-indexing.
    if False:
        # get all sessions
        beh_expt_list = [sess_expt[1] for sess_expt in sessdict[DATE]]
        beh_sess_list = [sess_expt[0] for sess_expt in sessdict[DATE]]
    else:
        # Get the single session assued to map onto this neural.
        beh_expt_list = [sess_expt[1] for sess_expt in sessdict[DATE] if sess_expt[0]==beh_session]
        print("Found these beh expt names: ", beh_expt_list)
        assert(len(beh_expt_list))==1, "must be error, multiple sessions with same session num"
        beh_sess_list = [beh_session]
        beh_trial_map_list = [(1, 0)]

    print("Loading these beh expts:", beh_expt_list)
    print("Loading these beh sessions:",beh_sess_list)
    print("Loading this neural session:", rec_session)

    SN = Session(DATE, beh_expt_list, beh_sess_list, beh_trial_map_list, 
        rec_session = rec_session, dataset_beh_expt=dataset_beh_expt)

    # Extract spikes
    SN.extract_raw_and_spikes_helper()

    return SN

class Session(object):
    """
    Operates over all trials for this entire session (day), even if multiple rec files for today

    """
    
    def __init__(self, datestr, beh_expt_list, beh_sess_list, 
            beh_trial_map_list=None,
            sites_garbage = None,
            expt="Lucas512-220520-115835", animal="Pancho", 
            path_base = "/mnt/hopfield_data01/ltian/recordings", 
            path_local = "/data3/recordings",
            rec_session=0, do_all_copy_to_local=False, 
            do_sanity_checks=False, do_sanity_checks_rawdupl=False,
            dataset_beh_expt= None):
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
        self.DatSpikeWaveforms = {}

        self.SitesAll = range(1, 513)
        self.SitesMetadata = {}

        # Neural stuff
        self.RecPathBase = path_base
        self.RecSession = rec_session
        self.RecPathBaseLocal = path_local

        # PopAnal stuff
        self.PopAnalDict = {}

        # Behavior stuff
        self.BehDate = datestr
        self.BehExptList = beh_expt_list
        self.BehSessList = beh_sess_list
        self.BehTrialMapList = beh_trial_map_list
        self.BehFdList = []
        self.BehTrialMapListGood = None

        # Behavior (Dataset Class) stuff
        self.DatasetbehExptname = dataset_beh_expt

        # For caching mapping from (site, trial) to index in self.DatAll
        self._MapperSiteTrial2DatAllInd = {}

        # self._initialize_params()
        self._initialize_paths()
        print("== PATHS for this expt: ")
        for k, v in self.Paths.items():
            print(k, ' -- ' , v)

        # Metadat about good sites, etc. Run this first before other things.
        assert sites_garbage is None, "use metadata instead"
        # then look for metadata
        self.load_metadata_sites()
        # else:
        #     self.SitesGarbage = sites_garbage
        # if self.SitesGarbage is not None:
        #     assert np.all(np.diff(self.SitesGarbage)>0.), "you made mistake entering chanels (assuming going in order)?"

        # Load raw things
        print("== Loading TDT tank")
        self.load_tdt_tank(include_streams = do_all_copy_to_local)
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

        if do_sanity_checks:
            self.plot_behcode_photodiode_sanity_check()

        if do_sanity_checks_rawdupl:
            # Load raw and dupl and compare them (sanity check)
            self.plot_raw_dupl_sanity_check()

        # Precompute mappers (quick)
        self._generate_mappers_quickly()

        # Various cleanups
        self._cleanup()


    ####################### PREPROCESS THINGS
    def _cleanup(self):
        """ Various things to run at end of each initialization
        - Sanity checks, etc. SHould be quick.
        """

        # If any spike chans are None, that means there was error in loading it. Mark this as
        # a bad site.
        sites_bad = []
        for dat in self.DatSpikes:
            if dat["spike_times"] is None:
                rs = dat["rs"]
                chan = dat["chan"]
                site = self.convert_rschan_to_site(rs, chan)
                print("[cleanup] Found a bad site (rs, chan, site): ", rs, chan, site)
                sites_bad.append(site)

        print("Saved all bad site to self.SitesErrorSpikeDat")
        self.SitesMetadata["sites_error_spikes"] = sites_bad

    def load_metadata_sites(self, dirty_kinds = ["sites_garbage", "sites_low_fr", "sites_error_spikes"]):
        """ Load info about which sites are garbage, hand coded
        PARAMS:
        - dirty_kinds, list str, sites marked as any of these kinds will be designated "dirty"
        """
        from pythonlib.tools.expttools import load_yaml_config
        import os

        path = f"{self.Paths['metadata_units']}/{self.Date}.yaml"
        if os.path.exists(path):
            print("Found! metada path : ", path)
            out = load_yaml_config(path)
            for k, v in out.items():
                self.SitesMetadata[k] = v
        else:
            print("Sites metada path doesnt exist: ", path)

        self._sitesdirty_update()


    def _sitesdirty_update(self, dirty_kinds = ["sites_garbage", "sites_low_fr", "sites_error_spikes"]):
        sites_dirty = []
        print("updating self.SitesDirty with: ", dirty_kinds)
        for kind in dirty_kinds:
            if kind in self.SitesMetadata.keys():
                sites = self.SitesMetadata[kind]
                sites_dirty.extend(sites)
        self.SitesDirty = sorted(set(sites_dirty))


    def _generate_mappers_quickly(self):
        """ generate mappers, which are dicts for mapping, e.g.,
        between indices"""

        # 1) map from trialcode to trial
        self._MapperTrialcode2TrialToTrial = {}

        for trial in self.get_trials_list():
            trialcode = self.datasetbeh_get_trialcode(trial)
            assert trialcode not in self._MapperTrialcode2TrialToTrial.keys(), "diff trials give same trialcode, not possible."
            self._MapperTrialcode2TrialToTrial[trialcode] = trial
        print("Generated self._MapperTrialcode2TrialToTrial!")



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
        assert len(paths)>0, "maybe you didn't mount server?"
        paththis = paths[self.RecSession]

        fnparts = deconstruct_filename(paththis)
        print(fnparts)
        final_dir_name = fnparts["filename_final_noext"]

        # Local cached
        pathbase_local = f"{self.RecPathBaseLocal}/{self.Animal}/{self.Date}/{final_dir_name}"
        import os
        os.makedirs(pathbase_local, exist_ok=True)

        def _get_spikes_raw_path():
            """ checks to find path to folder holding spikes data, in order of most to 
            least desired version. Returns None if doesnt find. 
            """
            from pythonlib.tools.expttools import load_yaml_config

            # First is saved path, the one where already got spikes from?
            if os.path.exists(f"{pathbase_local}/data_spikes.pkl"):
                # Then load old paths to raw spikes
                path_paths = f"{pathbase_local}/paths.yaml"

                if os.path.exists(path_paths):
                    paths_old = load_yaml_config(f"{pathbase_local}/paths.yaml")
                    print(paths_old)
                    return paths_old["spikes"]
                else:
                    print("then is old version, before saved paths every time save spikes") 
                    # Return the old version, which was 5.5 (blank)
                    return f"{paththis}/spikes_tdt_quick"

            # Second, if have not yet extracted spikes.
            for suffix in ["-4.5", "", "-3.5"]: 
                path_maybe = f"{paththis}/spikes_tdt_quick{suffix}"
                if os.path.exists(path_maybe):
                    return path_maybe

            # Didn't find spikes, return None
            return None

        pathdict = {
            "raws":paththis,
            "tank":f"{paththis}/{fnparts['filename_final_noext']}",
            # "spikes":f"{paththis}/spikes_tdt_quick",
            "spikes":_get_spikes_raw_path(),
            "final_dir_name":final_dir_name,
            "time":fnparts["filename_components_hyphened"][2],
            "pathbase_local":pathbase_local,
            "tank_local":f"{pathbase_local}/data_tank.pkl",
            "spikes_local":f"{pathbase_local}/data_spikes.pkl",
            "datall_local":f"{pathbase_local}/data_datall.pkl",
            "mapper_st2dat_local":f"{pathbase_local}/mapper_st2dat.pkl",
            "figs_local":f"{pathbase_local}/figs",
            "metadata_units":f"{PATH_NEURALMONKEY}/metadat/units"
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



    def load_tdt_tank(self, include_streams=False):
        """ Holds all non-nueral signals. is aligned to the neural
        BY default only loads epocs (not streams) as streams is large and takes a while
        PARAMS:
        - include_streams, bool, if True, then checks the pre-extracted includes stremas. if 
        not then tries to load fand save from server.
        RETURNS:
        - modifies self.DatTank
        """
        import os

        # Excluding dupl, which dont need, other than for sanity check.
        streams_to_extract = ["Mic1", "PhDi", "PhD2", "Eyee"]

        def tank_includes_streams(data_tank):
            """ returns True if Tank includes all desired streams
            """
            for stream in streams_to_extract:
                if stream not in data_tank["streams"].keys():
                    return False
            return True

        # First, try to load from local (much faster)
        if os.path.exists(self.Paths["tank_local"]):
            with open(self.Paths["tank_local"], "rb") as f:
                data_tank = pickle.load(f)
            if include_streams and not tank_includes_streams(data_tank):
                # Then discard this and load from server.
                LOAD_FROM_SERVER=True
            else:
                print("** Loading tank data from local (previusly cached)")
                LOAD_FROM_SERVER=False
                self.DatTank = data_tank
        else:
            LOAD_FROM_SERVER = True

        if LOAD_FROM_SERVER:
            print("** Loading data from tdt tank")
            # data_tank = tdt.read_block(self.PathTank, evtype = ["epocs", "streams"])
            if include_streams:
                print(" - Loading streams...", self.Date, self.ExptSynapse)
                data_tank_streams = tdt.read_block(self.PathTank, evtype = ["streams"], store=streams_to_extract)
                print(" - Loading epocs...")
                data_tank_epocs = tdt.read_block(self.PathTank, evtype = ["epocs"])
                data_tank = data_tank_epocs
                data_tank["streams"] = data_tank_streams["streams"]
            else:
                print(" - Loading epocs...")
                data_tank = tdt.read_block(self.PathTank, evtype = ["epocs"])
            self.DatTank = data_tank

            # save this for later
            self._savelocal_tdt_tank()

    def load_tdt_tank_specific(self, store, t1, t2):
        """
        Load from raw tank data specific segment. Only take specific, or else
        will take a while. 
        Save in same format as datall
        PARAMS:
        - store, str, like "PhDi"
        - t1, t2, time in sec
        """
            # {'rs': 2,
            #  'chan': 1,
            #  'trial0': 0,
            #  'pre_dur': 1.0,
            #  'post_dur': 1.0,
            #  'time_range': array([[38.48519424],
            #         [50.01179392]]),
            #  'raw': array([  0, 924, 927, ..., 487, 488, 495], dtype=int16),
            #  'tbins0': array([-1.0105611 , -1.01052014, -1.01047918, ..., 10.51595154,
            #         10.5159925 , 10.51603346]),
            #  'fs': 24414.0625,
            #  'spike_times': array([-0.91684206, -0.72330606, -0.46226798, -0.45358446, -0.25943406,
            #         -0.21306734, -0.18754926, -0.0916619 ,  0.0635765 ,  0.06488722,
            #          0.16388754,  0.1839989 ,  0.22037138,  0.36000402,  0.37913234,
            #          0.58323602,  0.64627346,  0.69227154,  1.03060114,  1.03367314,
            #          1.05271954,  1.15421842,  1.25465234,  1.30462354,  1.36573586,
            #          1.42963346,  1.44798354,  1.44962194,  1.6143221 ,  1.71586194,
            #          1.73511314,  1.78668178,  1.8148213 ,  1.87269778,  1.99819922,
            #          2.05619858,  2.08605842,  2.14192786,  2.36188306,  2.36806802,
            #          2.4705909 ,  2.64176274,  2.65544338,  2.76804242,  2.92827794,
            #          3.15417234,  3.20410258,  3.24502162,  3.32862098,  3.34819986,
            #          3.35135378,  3.36265874,  3.37019538,  3.37515154,  3.38170514,
            #          3.42401682,  3.50597778,  3.54603666,  3.5824501 ,  3.82386834,
            #          4.03514002,  4.0536949 ,  4.15462034,  4.22642322,  4.23137938,
            #          4.23805586,  4.24911506,  4.30002834,  4.67284626,  5.03771794,
            #          5.1166069 ,  5.18795922,  5.32456082,  5.59035026,  5.64687506,
            #          5.7082741 ,  5.7191285 ,  5.8506101 ,  5.95620498,  6.0472181 ,
            #          6.2413685 ,  6.38374546,  6.40815762,  6.45636754,  6.50252946,
            #          6.53746834,  6.65531026,  6.70106258,  6.7996533 ,  7.11459474,
            #          7.18193298,  7.22350738,  7.25107346,  7.29920146,  7.3159541 ,
            #          7.34904978,  7.3644917 ,  7.38910866,  7.40917906,  7.45411218,
            #          7.50215826,  7.53865362,  7.61799314,  7.6637045 ,  8.25131666,
            #          8.44763794,  8.7057269 ,  8.79960722,  8.8097653 ,  8.90835602,
            #          8.97315474,  9.05048722,  9.07551378,  9.14383506,  9.24643986,
            #          9.44640658,  9.56101266,  9.61507986,  9.65440146,  9.76974482,
            #          9.86903186,  9.93342098, 10.09042066, 10.22276242])}   
        d = tdt.read_block(self.PathTank, store=[store], t1=t1, t2=t2)
        # d["streams"]["PhDi"] = 
        #     name:   'PhDi'
        #     code:   1766090832
        #     size:   138
        #     type:   33025
        #     type_str:   'streams'
        #     ucf:    False
        #     fs: 1017.2526245117188
        #     dform:  2
        #     start_time: 0.0
        #     data:   array([ 6, 16, 16, ...,  7,  6, 10], dtype=int16)
        #     channel:    [1]            

        out = {
            "store":store,
            "datatype":"stream",
            "time_range":np.array([t1, t2]), 
            "fs":d["streams"][store]["fs"],
            "data":d["streams"][store]["data"]
            }

        return out

    def _savelocal_tdt_tank(self):
        """ save this for later
        """
        print("Saving TDT Tank locally to: ", self.Paths["tank_local"])
        with open(self.Paths["tank_local"], "wb") as f:
            pickle.dump(self.DatTank, f)


    def _load_spike_times(self, rs, chan, ver="spikes_tdt_quick", 
            return_none_if_fail=True):
        """ Load specific site inforamtion
        """
        """ Return spike times, pre-extracted elsewhere (matlab)
        in secs
        """
        import scipy.io as sio
        import scipy
        import zlib
        fn = f"{self.Paths['spikes']}/RSn{rs}-{chan}"
        print(f"Loading this spikes file: {fn}.mat")
        if return_none_if_fail:
            try:
                mat_dict = sio.loadmat(fn)
                return mat_dict["spiketimes"]
            except zlib.error as err:
                print("[scipy error] Skipping spike times for (rs, chan): ", rs, chan)
                return None
            except Exception as err:
                print(err)
                print("Failed for this rs, chan: ",  rs, chan)
                assert False
        else:
            mat_dict = sio.loadmat(fn)
            return mat_dict["spiketimes"]


    def load_spike_times(self):
        """ Load and strore all spike times (across all trials, and chans)
        Not yet aligned to trials, etc.
        NOTE:
        - if loading error, then spiketimes are None. Otherwise is np array of
        times in seconds.
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
            def load_spike_times_mult(rss, chans, ver="spikes_tdt_quick"):
                DatSpikes = []
                for rs in rss:
                    for ch in chans:
                        print("load_spike_times", rs, ch)
                        st = self._load_spike_times(rs, ch, ver)
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

        # also save note of where the path
        from pythonlib.tools.expttools import writeDictToYaml
        writeDictToYaml(self.Paths, f"{self.Paths['pathbase_local']}/paths.yaml")


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
            T1, T2 = self.extract_timerange_trial(trial0, pre_dur, post_dur) # T1, T2 after after applying the time range.
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
                        t, raw = self.extract_windowed_data_bytrial(t, trial0, raw)[:2] # - second, use the general windowing function
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
        site = self.convert_rschan_to_site(rs, chan)

        # Decide if extract from saved
        if site not in self.DatSpikeWaveforms.keys():
            import zlib
            import scipy.io as sio

            PATH_SPIKES = self.Paths["spikes"]
            # PATH_SPIKES = f"{self.PathRaw}/spikes_tdt_quick"
            fn = f"{PATH_SPIKES}/RSn{rs}-{chan}-snips_subset"
            try:
                mat_dict = sio.loadmat(fn)
                waveforms = mat_dict["snips"]
            except zlib.error as err:
                print("[scipy error] Skipping load_spike_waveforms_ for (rs, chan): ", rs, chan)
                waveforms = None
            except Exception as err:
                print(err)
                print("[load_spike_waveforms] Failed for this rs, chan: ",  rs, chan)
                print(fn)
                assert False

            self.DatSpikeWaveforms[site] = waveforms

        return self.DatSpikeWaveforms[site]


    def spikewave_compute_stats(self, waveforms):
        """ 
        PARAMS;
        - waveforms, (n waveforms,timebins) array.
        RETURNS:
        - outdict, dict of stats
        """

        outdict = {}

        # 2) min and max voltages
        outdict["volt_max"] = np.max(waveforms, axis=1)
        outdict["volt_min"] = np.min(waveforms, axis=1)

        return outdict

    ########################## GET FROM PRE-EXTRACTED DATA
    def extract_data_tank_streams(self, which, trial0=None):
        """ Extract tank data, epocs, in flexible manner.
        PARAMS:
        - which, string name, will be mapped to the specific key in data
        - crosstime, whether to use "onset", "offset", or "mean" times
        - trial0, int, optional trial to slice data (will recompute the 0 rel trial onset).
        RETURNS
        - times, vals
        - OR: None, if this data doesnt eixst.
        """
        
        keynames = {
            "mic":"Mic1",
            "duplicate1":"dup1",
            "duplicate2":"dup2",
            "pd1":"PhDi", # Lucas
            "pd2":"PhD2", # G and Y
            "eyex":"Eyee",
            "eyey":"Eyee",
            "eyediam":"Eyee",
        }
        key = keynames[which]
        if key not in self.DatTank["streams"].keys():
            print("Did not find in self.DatTank: ", key, ", so skipping...")
            return None

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
            times, vals = self.extract_windowed_data_bytrial(times, trial0, vals)[:2]
        
        return times, vals, fs


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
            "behcode":"SMa1" if "SMa1" in self.DatTank["epocs"].keys() else "Bode",
            "strobe":"S_ML",
        }
        key = keynames[which]
        if key not in self.DatTank["epocs"].keys():
            print("********* failing extract_data_tank_epocs")
            print(self.DatTank["epocs"].keys())
            print(key)
            print(which)
            print(trial0)
            assert False
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
            times, vals = self.extract_windowed_data_bytrial(times, trial0, vals)[:2]

        return times, vals

    def extract_raw_and_spikes_helper(self, trials=None, sites=None, get_raw=False):
        """ to quickly get a subset of trials, all sites, etc.
        PARAMS:
        - trials, list of ints, if None, then gets all.
        - sites, list of ints, if None, then gets all.
        NOTE:
        - if trials and sites are None, then automatically saves and loads locally.
        """

        LOADED = False
        #DO_SAVE = False
        if sites is None and trials is None:
            # Then try to load locally.
            import os
            if os.path.exists(self.Paths["datall_local"]):
                
                # Load quickly from local
                print("** Loading datall from local (previusly cached)")
                with open(self.Paths["datall_local"], "rb") as f:
                    self.DatAll = pickle.load(f)

                # Add timing info, since might not be done
                self.datall_cleanup_add_things()

                # == Load mapper
                if os.path.exists(self.Paths["mapper_st2dat_local"]):
                    print("** Loading _MapperSiteTrial2DatAllInd from local (previusly cached)")
                    with open(self.Paths["mapper_st2dat_local"], "rb") as f:
                        self._MapperSiteTrial2DatAllInd = pickle.load(f)
                else:
                    # generate mapper, slice each one and this will autoamtically extract
                    self.mapper_extract("sitetrial_to_datallind")
                
                # dont rerun
                LOADED = True

        if not LOADED:
            # Then extract from loaded spikes and tank data.
            if sites is None:
                # get all sites
                sites = self.SitesAll
            if trials is None:
                trials = self.get_trials_list(only_if_ml2_fixation_success=True)
                # trials = range(len(self.TrialsOnset))

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

            # generate mapper, slice each one and this will autoamtically extract
            self.mapper_extract("sitetrial_to_datallind")
            
            # Save
            self._savelocal_datall()


    def extract_raw_and_spikes(self, rss, chans, trialtdt, get_raw = False):
        """ [GET ALL DATA] Extract raw (unfiltered) and spikes for this trial
        PARAMS:
        - rss, list of ints for rs
        - chans, list of chans (within this rs)
        - trialtdt, tdt trials
        NOTE: This will extract even if already gotten. This is because simply checking
        whether is gotten takes long time.
        """
        
        # raw (get all chans) [from disk]
        DatRaw = self.load_raw(rss, chans, trialtdt, get_raw=get_raw)
        
        # spikes (from pre-extracted spikes)
        for i, d in enumerate(DatRaw):
    #         spike_times = datspikes_slice_single(d["rs"], d["chan"], d["time_range"])
            spike_times, time_dur, time_on, time_off = self.datspikes_slice_single(d["rs"], d["chan"], trial0=trialtdt)
            d["spike_times"] = spike_times 
            d["time_dur"] = time_dur
            d["time_on"] = time_on
            d["time_off"] = time_off


        # change name, since is both raw and spikes
        if self.DatAll is None:
            self.DatAll = DatRaw
        else:
            # OLD VERSION - checkign if gotten before runing. this takes a whiel
            # for d in DatRaw:
            #     # Don't inlcude it if it is already extracted
            #     # if getting raw, then will overwrite if it was gotten but without raw.
            #     d_old = self.datall_slice_single(d["rs"], d["chan"], d["trial0"])
            #     if d_old is None:
            #         # then doestn exist, append
            #         self.DatAll.append(d)
                    
            #         # save the index
            #         index = len(self.DatAll)
            #         site = self.convert_rschan_to_site(d["rs"], d["chan"])
            #         trial = d["trial0"]
            #         if (site, trial) not in self._MapperSiteTrial2DatAllInd[(site, trial)]:
            #             self._MapperSiteTrial2DatAllInd[(site, trial)] = index
            #         else:
            #             assert self._MapperSiteTrial2DatAllInd[(site, trial)] == index

            #     elif len(d_old["raw"])==0 and get_raw:
            #         # Then previous didnt have raw, so overwrite it
            #         self.datall_replace_single(d["rs"], d["chan"], d["trial0"], Dnew=d)
            #     else:
            #         # skip
            #         pass

            #     # if not self.datall_this_exists(d["rs"], d["chan"], d["trial0"],
            #     #     also_check_if_has_raw_data=get_raw):
            #     #     # then append
            #     #     self.DatAll.append(d)

            for d in DatRaw:
                site = self.convert_rschan_to_site(d["rs"], d["chan"])
                trial = d["trial0"]

                if (site, trial) in self._MapperSiteTrial2DatAllInd.keys():
                    # Then is already done
                    d_old = self.datall_slice_single_bysite(site, trial)
                    if len(d_old["raw"])==0 and get_raw:
                        # Then previous didnt have raw, so overwrite it
                        self.datall_replace_single(d["rs"], d["chan"], trial, Dnew=d)
                    else:
                        # skip, don't replace.
                        pass
                else:
                    # Brand new. append it.
                    self.DatAll.append(d)
                    
                    # save the index
                    index = len(self.DatAll)-1
                    if (site, trial) not in self._MapperSiteTrial2DatAllInd.keys():
                        self._MapperSiteTrial2DatAllInd[(site, trial)] = index
                    else:
                        assert self._MapperSiteTrial2DatAllInd[(site, trial)] == index

        # Add timing information
        self.datall_cleanup_add_things()

        return self.DatAll

    def datall_cleanup_add_things(self):
        """ Quick processing, things to add to datall in case not already 
        added 
        - Also makes self.DatAllDf (dataframe version).
        """
        print("DOING: datall_cleanup_add_things")
        import pandas as pd

        # Time info
        if "time_dur" not in self.DatAll[0].keys():
            self._datall_compute_timing_info()

        # sites
        for Dat in self.DatAll:
            if "site" not in Dat.keys():
                site = self.convert_rschan_to_site(Dat["rs"], Dat["chan"])
                Dat["site"] = site

        # no spikes outside of time window
        for Dat in self.DatAll:
            st = Dat["spike_times"]
            t_on = Dat["time_on"]
            t_off = Dat["time_off"]
            if st is not None and t_on is not None and t_off is not None:
                if np.any(st<t_on) or np.any(st>t_off):
                    print(Dat)
                    assert False, "due to jitter in ml2 vs. tdt?"

        self.DatAllDf = pd.DataFrame(self.DatAll)
        print("Generated self.DatAllDf")

    def _datall_compute_timing_info(self):
        """
        Add info into self.DatAll reflecting the time on and off of data, and
        the duration. i.e., for each trial. self.DatAll must already be extracted.
        RETURNS:
        - in self.DatAll, each element modified with new keys.
        """

        dicttrials = {}
        def _get_this_trial(trial):
            if trial not in dicttrials.keys():
                _, _, time_dur, time_on, time_off = self.extract_windowed_data_bytrial([], trial)
                dicttrials[trial] = (time_dur, time_on, time_off)
            return dicttrials[trial]

        for Dat in self.DatAll:
            trial = Dat["trial0"]
            time_dur, time_on, time_off = _get_this_trial(trial)
            Dat["time_dur"] = time_dur
            Dat["time_on"] = time_on
            Dat["time_off"] = time_off

            # OLD
            t_dur = Dat["time_range"][1]- Dat["time_range"][0] # entire data, sec.
            t_on = -Dat["pre_dur"] # e.g., -1
            t_off = time_dur + t_on 
            assert np.abs(t_dur - time_dur)<0.02
            assert np.abs(t_on - time_on)<0.02
            assert np.abs(t_off - time_off)<0.02


        # assert False, "dont use this, it is off by the jitter between ML2 and TDT. instead, use time extracted from extract_windowed_data_bytrial "
        # for Dat in self.DatAll:
        #     time_dur = Dat["time_range"][1]- Dat["time_range"][0] # entire data, sec.
        #     t_on = -Dat["pre_dur"] # e.g., -1
        #     t_off = time_dur + t_on 
        #     Dat["time_dur"] = time_dur
        #     Dat["time_on"] = t_on
        #     Dat["time_off"] = t_off
        print("DONE! _datall_compute_timing_info")


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
        self.datall_cleanup_add_things()

        print("Saving DatAll (raw and spikes) locally to: ", self.Paths["datall_local"])
        with open(self.Paths["datall_local"], "wb") as f:
            pickle.dump(self.DatAll, f)

        print("Saving _MapperSiteTrial2DatAllInd locally to: ", self.Paths["mapper_st2dat_local"])
        with open(self.Paths["mapper_st2dat_local"], "wb") as f:
            pickle.dump(self._MapperSiteTrial2DatAllInd, f)


    ###################### WINDOW THE DATA based on trials, etc
    def extract_windowed_data(self, times, twind, vals=None, recompute_time_rel_onset=True, time_to_add=0.):
        """ Prune data (time, vals) to include only those with
        times within twind. Also changes times to be relative to twind[0]
        RETURNS:
        - times, eitehr:
        --- in original time base (if recompute_time_rel_onset is False)
        --- times so that 0 is aligned to twind[0], if recompute_time_rel_onset is True)
        ---- times that arbitrary timepoint (time_to_add) is aligned to the twind[0]
        - vals, aligned to times
        - time_dur, time_on, time_off, scalars, the "ground-truth" for the window that contains the data.
        """
        
        inds = (times>=twind[0]) & (times<=twind[1])
        if vals is not None:
            assert times.shape==vals.shape
            vals = vals[inds]
        times = times[inds]
        
        # get times relative to windo wonset
        if recompute_time_rel_onset:
            # zero is now defined to be aligned to twind[0]
            times = times - twind[0]
            
            # shift the times
            times = times + time_to_add
        else:
            assert time_to_add==0., "not sure what this wuld mean for downstream code.."

        # save time window information
        time_dur, time_on, time_off = self._extract_windowed_data_get_time_bounds(twind, recompute_time_rel_onset, time_to_add)
        
        return times, vals, time_dur, time_on, time_off

    def _extract_windowed_data_get_time_bounds(self, twind, recompute_time_rel_onset=True, time_to_add=0.):
        if recompute_time_rel_onset:
            # save time window information
            time_dur = twind[1] - twind[0]
            time_on = 0 + time_to_add
            time_off = time_on + time_dur
        else:
            assert time_to_add==0., "not sure what this wuld mean for downstream code.."

            # save time window information
            time_dur = twind[1] - twind[0]
            time_on = twind[0]
            time_off = twind[1]     
        return time_dur, time_on, time_off       


    def extract_windowed_data_bytrial(self, times, trial0, vals=None, recompute_time_rel_onset=True, pre_dur=1., post_dur=1.):
        """ Given generic data, window it by a given trial.
        Prune data (time, vals) to include only those with
        times within twind. Also changes times to be relative to trial onset (regardless of pre_dur
        PARAMS:
        - pre_dur, post_dur, time extra to extract. NOTE: doesnt affect what is called 0, which is always trial onset
        """

        assert recompute_time_rel_onset==True, "some code assumes this true. I dont see any reason to change this. if want to change, then make new function that does this"
        
        # Get window
        t1, t2 = self.extract_timerange_trial(trial0, pre_dur, post_dur)
        TIME_ML2_TRIALON = self.ml2_get_trial_onset(trialtdt = trial0)
        time_to_add = -pre_dur # so that is zeroed on time of trial onset
        
        # shift all tdt things so that by definition the time of beh code 9 are identical between tdt and ml2
        time_to_add = time_to_add - TIME_ML2_TRIALON

        if len(times)>0:
            times, vals, time_dur, time_on, time_off = self.extract_windowed_data(times, [t1, t2], vals, recompute_time_rel_onset, time_to_add = time_to_add)
        else:
            time_dur, time_on, time_off = self._extract_windowed_data_get_time_bounds([t1, t2], recompute_time_rel_onset, time_to_add = time_to_add)

        return times, vals, time_dur, time_on, time_off

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


    def datall_slice_single_bysite(self, site, trial0, return_index=False):
        """ Like datall_slice_single, but input site instead of (rs, chan)
        """

        rs, chan = self.convert_site_to_rschan(site)
        return self.datall_slice_single(rs, chan, trial0, return_index)

    def datall_slice_single(self, rs, chan, trial0, return_index=False, method="new"):
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
            if return_index:
                return None, None
            else:
                return None

        site = self.convert_rschan_to_site(rs, chan)
        index = None
        if (site, trial0) not in self._MapperSiteTrial2DatAllInd.keys():
            # Then extract
            assert False, "first run extract_raw_and_spikes_helper to pre-save self.DatAll and self._MapperSiteTrial2DatAllInd"
            
            # self.mapper_extract("sitetrial_to_datallind", save=True)

        index = self._MapperSiteTrial2DatAllInd[(site, trial0)]
        dat = self.DatAll[index]    
        assert dat["rs"] == rs
        assert dat["chan"] == chan
        assert dat["trial0"] == trial0

        if return_index:
            return dat, index
        else:
            return dat

    def mapper_extract(self, version, save=True):
        """ construct mapper (do this oine time)
        """
                    

        if version=="sitetrial_to_datallind":
            print("Extracting _MapperSiteTrial2DatAllInd")
            trialprint = -1
            for index, Dat in enumerate(self.DatAll):
                site = self.convert_rschan_to_site(Dat["rs"], Dat["chan"])
                trial = Dat["trial0"]
                if trial!=trialprint:
                    print("trial: ", trial)
                    trialprint = trial
                self._MapperSiteTrial2DatAllInd[(site, trial)] = index
            if save:
                self._savelocal_datall()
        else:
            print(version)
            assert False, "code it"

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
        RETURNS:
        spiketimes, time_dur, time_on, time_off
        """
        
        for D in self.DatSpikes:
            if D["rs"]==rs and D["chan"]==chan:
                spiketimes = D["spike_times"]

                if spiketimes is not None:
                    # optionally window it
                    if twind is not None:
                        assert trial0 is None
                        spiketimes, _, time_dur, time_on, time_off = self.extract_windowed_data(spiketimes, twind)
                    elif trial0 is not None:
                        assert twind is None
                        spiketimes, _, time_dur, time_on, time_off = self.extract_windowed_data_bytrial(spiketimes, trial0)    
                    else:
                        assert False, "must be one or the other"
                    return spiketimes, time_dur, time_on, time_off
                else:
                    return None, None, None, None
                
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
        ntrials = len(self.get_trials_list(only_if_ml2_fixation_success=False))
        # ntrials = len(self.TrialsOnset)
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
        if fd is None or fd_trialnum is None:
            print("ERrory in beh_get_fd_trial: ", self.Date, self.ExptSynapse, trialtdt)
            assert False
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
    def sitegetter_print_summary_nunits_by_region(self):
        """ Prints num units (clean) per region
        and total units etc
        """
        sites_all =[]
        for area, sites in self.sitegetter_brainregion(clean=True).items():
            print(area, " : ", len(sites))
            sites_all.append(len(sites))
        print(" ------- ")
        print("TOTAL: ", sum(sites_all))
        print("MIN: ", min(sites_all))
        print("MAX: ", max(sites_all))
        print("MEAN: ", np.mean(sites_all))

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
        - list_regions, get only these regions. leave None to get all. if None,
        then returns all sites.
        RETURNS:
        - list of sites
        """
        bregion_mapper = self.sitegetter_brainregion("mapper", clean=clean)
        if list_regions is None:
            bm = bregion_mapper
        else:
            bm = {br:sites for br, sites in bregion_mapper.items() if br in list_regions}

        sites = [s for br, ss in bm.items() for s in ss]
        return sites

    def sitegetter_brainregion(self, region=None, clean=False):
        """ Flexible getter of channels based on region
        PARAMS:
        - region, flexible input. see within code
        - clean, whether to remove garbage chanels
        RETURNS:
        - out, depends on type of region
        """
        # Hard coded
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
            assert self.SitesDirty is not None, "you need to enter which are bad sites in SitesDirty"
            for k, v in dict_sites.items():
                # remove any sites that are bad
                dict_sites[k] = [vv for vv in v if vv not in self.SitesDirty]

        if region=="list_regions":
            return regions_in_order
        elif region=="mapper" or region is None:
            return dict_sites
        elif isinstance(region, int):
            return dict_sites[regions_in_order[region]]
        elif isinstance(region, list):
            # then this is list of str
            list_chans = []
            for reg in region:
                sites = self.sitegetter_brainregion(reg, clean=clean)
                list_chans.extend(sites)
            return list_chans
        elif isinstance(region, str):
            return dict_sites[region]
        else:
            assert False

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
        else:
            assert False

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

    ########################### Stats for each site
    def sitestats_fr_single(self, sitenum, trial):
        """ get fr (sp/s)
        - if fr doesnt exist in self.DatAll, then will add it to the dict.
        """
        dat = self.datall_slice_single_bysite(sitenum, trial)
        if "fr" not in dat.keys():
            nspikes = len(dat["spike_times"])
            dur = dat["time_dur"]
            dat["fr"] = nspikes/dur
        return dat["fr"]

    def sitestats_fr(self, sitenum):
        """ gets fr across all trials. Only works if you have already extracted fr 
        into DatAll (and its dataframe)
        """
        list_fr = []
        

        # if first_extraction:
        #     # then dont use dataframe, this both extracts and returns.
        #     # This both easfasdfxtracts and returns
        #     for trial in self.get_trials_list(True):
        #         fr = self.sitestats_fr_single(sitenum, trial)
        #         list_fr.append(fr)
        # else:
        # Use Dataframe, is much faster than iterating over trials

        trials = self.get_trials_list()
        dfthis = self.DatAllDf[(self.DatAllDf["site"]==sitenum) & (self.DatAllDf["trial0"].isin(trials))]

        # Confirm that has already been extracted and saved
        ERROR = False        
        if "fr" not in dfthis.columns:
            ERROR = True
        elif dfthis["fr"].isna().any():
            ERROR = True
        if ERROR:
            # Now raise error, first extract
            print("First extract all fr using sitestats_fr_get_and_save(True)")
            assert False

        # Get fr
        list_fr = dfthis["fr"].tolist()

        stats = {}
        stats["list_fr"] = list_fr
        stats["fr_mean"] = np.mean(list_fr)
        return stats

    def sitestats_fr_get_and_save(self):
        """ Gets fr for all sites and saves in self.DatAll, and saves
        to disk. This is more for computation than for returning anything useufl. 
        Run this once."""
        for site in self.sitegetter_all(clean=False):
            
            if site%50==0:
                print(site)
            
            for trial in self.get_trials_list(True):
                self.sitestats_fr_single(site, trial)
                # list_fr.append(fr)
            # self.sitestats_fr(site) # run this to iterate over all trials, and save to datall
        
        self._savelocal_datall()

    def sitestats_get_low_fr_sites(self, low_fr_thresh=2):
        """ FInds sites with mean fr less than threshold; 
        useful for pruning neurons, etc.
        """
        frmeans = []
        sites =[]
        for site in self.sitegetter_all(clean=False):
            # if site%50==0:
            #     print(site)
            frmeans.append(self.sitestats_fr(site)["fr_mean"])
            sites.append(site)
        frmeans = np.asarray(frmeans)
        sites = np.asarray(sites)

        # Save
        fig, axes = plt.subplots(2,2, figsize=(15,8))

        # 1) Histogram of fr
        ax = axes.flatten()[0]
        ax.hist(frmeans, 100)
        ax.set_title('mean fr across sites')

        # 1) Histogram of fr (zzoming in)
        ax = axes.flatten()[2]
        ax.hist(frmeans, np.linspace(0, 10, 10))
        ax.set_title('mean fr across sites')

        # 2) Each site, plot fr
        ax = axes.flatten()[1]
        ax.plot(sites, frmeans, 'ok')
        ax.set_xlabel('site num');
        ax.set_ylabel('fr mean')

        ####### sort sites by fr and print in order
        indsort = np.argsort(frmeans)
        frmeans_sorted = frmeans[indsort]
        sites_sorted = sites[indsort]

        # print("-- SITES, sorted (increasing fr): ")
        # print(sites_sorted)
        # print("-- FR, sorted (increasing fr): ")
        # print(frmeans_sorted)
        site_fr = [f"{s}[{fr:.0f}]" for s, fr in zip(sites_sorted, frmeans_sorted)]
        print("-- SITE(FR), sorted (increasing fr): ")
        print(site_fr)
        

        ####### Get low Fr sites
        sites_lowfr = sites[frmeans<low_fr_thresh]
        print("Low FR sites: ", sites_lowfr)
        print("Num sites failing threshold: ", sum(frmeans<low_fr_thresh))

        sites_lowfr_str = [str(s) for s in sites_lowfr]
        sites_lowfr_str = ', '.join(sites_lowfr_str)        
        print("low fr sites, as strings: ")
        print(sites_lowfr_str)



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
            times = self.extract_windowed_data_bytrial(times, trial0)[0]
        
        if first_instance_only:
            if len(times)>1:
                times = times[0:1] # stay as an array
        return times
        
    def behcode_canonical_sequence(self):
        """ Gets the standard sequence for a full trial.
        """
        list_codes = [9, 11, 16, 91, 92, 62, 73, 50, 18]
        return list_codes

    def behcode_extract_times_sequence(self, trial0, codes=None):
        """ Get the times of this sequence of codes, aligned to this trial,
        taking only the first instance of each code.
        If a code doesnt occur, gives it a np.nan
        RETURNS:
        - times, np array len of codes. elements are either seconds or np.nan
        NOTE: order will not matter
        """
        
        if codes is None:
            # The default "important" events within a trial
            list_codes = self.behcode_canonical_sequence()

        times = np.empty(len(codes))
        for i, c in enumerate(codes):
            t = self.behcode_extract_times(c, trial0, True)
            if len(t)>0:
                times[i] = t[0]
            else:
                times[i] = np.nan
        return times    

    def behcode_get_stream_crossings_in_window(self, trial, behcode,
        t_pre=0.01, t_post=0.2, whichstream="pd1", do_smooth=True, smooth_win=0.01,
        ploton=False, cross_dir="both", force_single_output=False):
        """ Given this behcode, and time window around this code,
        finds all cases where this stream has trheshold crossings, where
        threshold are determined automatically (based on prctiles of 
        vals in this trial.)
        PARAMS:
        - behcode, int, will by default find the first instance of this code
        - t_pre, t_post, time in sec relative to behcode time, for windowing 
        the stream
        - whichstream, str, name of the stream
        - do_smooth, bool, whether to smooth data before computing things.
        - smooth_win, smoothing window in sec.
        - cross_dir, str, which direction crossing to take. either {"both", "up", "down"}
        - false_single_output, bool, if True, then aserts that only one, and returns the number
        RETURNS:
        - TCROSS, VCROSS, array of times and values at crossings
        - time_behcode, time of behc ode
        - valminmax, (2,) array of vals min and max
        - threshold, value used for crossing
        TODO: use histogram to determine local threshold in each time window. problem is that the globally-defined threshold is not super accurate")
        NOTE: times are all relative to beh code 9 (i.e., trial onset)
        """

        
        # For each pd expected event, find its beh code, then look for the pd signal

        # 1) Extract the stream siganl
        times, vals, fs = self.extract_data_tank_streams(whichstream, trial)

        # 2) Smooth if desired
        n = int(smooth_win * fs)
        vals_sm = np.convolve(vals, np.ones(n)/n, mode="same")
        if False:
            fig, ax = plt.subplots(1,1, figsize=(15,10))
            ax.plot(times, vals, '-k');
            ax.plot(times, vals_sm, '-r');

        # 3) Tresholding: Get min and max values. Threshold is in betweem them.
        valminmax = np.percentile(vals, [1, 99])
        threshold = np.mean(valminmax)

        # 4) Find temporal window for this beh code.
        if False:
            # For a given window, find threshold crossings
            codes_nums = self.behcode_canonical_sequence()
            codes_times = self.behcode_extract_times_sequence(trial, codes_nums)   
            for num, time in zip(codes_nums, codes_times):
                print(num, time)

        # - Specific code
        time_behcode = self.behcode_extract_times(behcode, trial, True)

        # - window around the code
        if len(time_behcode)>0:   
            inds = (times>=time_behcode-t_pre) & (times<=time_behcode+t_post)
            timesthis = times[inds]
            valsthis = vals_sm[inds]

            # Get all threshold crossings
            indscross = np.where(np.diff(valsthis>threshold))[0]
            
            # - take mean of the immediately preceding and following time bins for each
            # crossing.
            timecross = (timesthis[indscross] + timesthis[indscross+1])/2
            valscross = (valsthis[indscross] + valsthis[indscross+1])/2

            # What directions are crossings?
            if valsthis[0]<threshold:
                # this will be [positive-going, neg-going, etc...]
                timecross_up = timecross[0::2]
                timecross_dn = timecross[1::2]
                valscross_up = valscross[0::2]
                valscross_dn = valscross[1::2]
            else:
                # other direction
                timecross_up = timecross[1::2]
                timecross_dn = timecross[0::2]
                valscross_up = valscross[1::2]
                valscross_dn = valscross[0::2]

            ncross = len(timecross)

            if ploton:
                fig, axes = plt.subplots(1,2, figsize=(15,5))

                ax = axes.flatten()[0]
                ax.plot(timesthis, valsthis)
                ax.plot(timecross, valscross, 'xk')
                ax.plot(timecross_up, valscross_up, 'ob')
                ax.plot(timecross_dn, valscross_dn, 'or')
                ax.axhline(threshold)
                ax.set_title('b=upcross, r=dncross')

                ax = axes.flatten()[1]
                edges = np.linspace(np.min(vals_sm), np.max(vals_sm), 50)
                ax.hist(vals_sm, bins=edges, density=True, histtype="step")
                ax.hist(valsthis, bins=edges, density=True, histtype="step")
                ax.set_title("values in window and entire trial")

            # Time of first cross relative to behcode
            # TODO

            if cross_dir=="both":
                TCROSS = timecross
                VCROSS = valscross
            elif cross_dir=="up":
                TCROSS = timecross_up
                VCROSS = valscross_up
            elif cross_dir in ["dn", "down"]:
                TCROSS = timecross_dn
                VCROSS = valscross_dn
            else:
                assert False

            if force_single_output:
                assert len(TCROSS)==1
                assert len(VCROSS)==1
                TCROSS = TCROSS[0]
                VCROSS = VCROSS[0]
            return TCROSS, VCROSS, time_behcode, valminmax, threshold
        else:
            return np.array([]), np.array([]), time_behcode, valminmax, threshold

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
            
    ############### Beh Dataset (extracted)
    def datasetbeh_load(self, dataset_beh_expt=None, 
            dataset_beh_rule="null", remove_online_abort=False,
            remove_trials_not_in_dataset=False):
        """ Load pre-extracted beahviuioral dataset and algined
        trial by trial to this recording session. 
        Tries to automtiaclly get the exptname, and rule, but you might
        have to give it manualyl.
        - Will prune self.Dat to only keep trials that are in Datset (which tehemselves
        are trials where there was successfuly touhc)
        - Will do sanity check that every clean 
        RETURNS:
        - self.DatasetBeh, and returns
        """
        from pythonlib.dataset.dataset import Dataset
        from pythonlib.dataset.dataset_preprocess.general import preprocessDat

        # 1) Load Dataset
        if self.DatasetbehExptname is None:
            # you must enter it
            expt = dataset_beh_expt
        else:
            # load saved
            expt = self.DatasetbehExptname
        assert expt is not None
        D = Dataset([], remove_online_abort=remove_online_abort)
        D.load_dataset_helper(self.Animal, expt, rule=dataset_beh_rule)
        D.load_tasks_helper()

        D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt)

        # 2) keep only the dataset trials that are included in recordings
        trials = self.get_all_existing_site_trial_in_datall("trial")                                               
        list_trialcodes = [self.datasetbeh_get_trialcode(t) for t in trials]
        print("- Keeping only dataset trials that exist in self.Dat")
        print("Starting length: ", len(D.Dat))
        D.filterPandas({"trialcode":list_trialcodes}, "modify")
        print("Ending length: ", len(D.Dat))

        # 3) Prune trials in self to remove trials that dont have succesfuly fix and touch.
        trials_neural_to_remove = []
        for trial_neural, trialcode in enumerate(list_trialcodes):
            if trialcode not in D.Dat["trialcode"].tolist():
                # then this is only acceptable if this trial is not succesful fix or touch
                fd, t = self.beh_get_fd_trial(trial_neural)
                suc = mkl.getTrialsFixationSuccess(fd, t)
                touched = mkl.getTrialsTouched(fd, t)
                # tem = mkl.getTrialsOutcomesWrapper(fd,t)["trial_end_method"]
                if touched and suc:
                    print(list_trialcodes)
                    print(D.Dat["trialcode"].tolist())
                    print(trialcode)
                    assert False, "some neural data not found in beh Dataset..."
                else:
                    # remove this trial from self.Dat, since it has no parallele in dataset
                    trials_neural_to_remove.append(trial_neural)
        # - remove the trials.
        if remove_trials_not_in_dataset:
            assert False, "code it..."

        # --
        self.Datasetbeh = D
        return self.Datasetbeh

    def datasetbeh_plot_example_drawing(self, trial):
        """ Plot drawing from both neural and dataset data
        Mostly for sanity check, and for saving this method
        PARAMS:
        - trial, for neural
        """
        # PLOT EXAMPLE, SHOWING MATCH across neural and beh datasets

        # 1) for each tdt trial, get its trialcode in beh
        trialcode = self.datasetbeh_get_trialcode(trial)
        D = self.Datasetbeh

        # 2) Find this trialcode in Dataset
        dat = D.Dat[D.Dat["trialcode"]==trialcode]
        trial_dat = dat.index.values[0]

        # 3) Sanity check, plot using both dataset and neural
        D.plotSingleTrial(trial_dat)

        fig, ax = plt.subplots(1,1)
        # SN.plot_trial_timecourse_summary(ax, trial)
        self.plot_final_drawing(ax,trial)





    ###################### GET TEMPORAL EVENTS
    def events_get_time_all(self, trial, list_events = ["stim_onset", "go_cue", "first_raise", "on_stroke_1"]):
        """
        Get dict of times of important events. Uses variety of methods, including
        (i) photodiode (ii) motor behavior, (iii) beh codes, wherever appropriate.
        - All times relative to behcode 9 (trial onset) by convention.
        PARAMS:
        - list_events, list of str, each a label for an event. only gets those in this list.
        """

        
        list_events_skip_if_no_fixation = ["go_cue", "first_raise", "on_stroke_1"]
        dict_events = {}

        for event in list_events:
            # 1) Skip this, if no fixation success
            if not self.beh_fixation_success(trial) and event in list_events_skip_if_no_fixation:
                time = None
            else:
                if event=="stim_onset":
                    # Use photodiode
                    behcode = 91
                    stream = 'pd1'
                    cross_dir = 'up'
                    t_pre = 0
                    t_post = 0.2
                    time, _,_,_,_ = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                              cross_dir=cross_dir, t_pre=t_pre,
                                                              t_post=t_post,
                                                              ploton=False, force_single_output=True)
                elif event=="go_cue":
                    # Use photodiode
                    behcode = self.behcode_convert(codename="go", shorthand=True)
                    stream = 'pd2'
                    cross_dir = 'down'
                    t_pre = 0
                    t_post = 0.2
                    time, _,_,_,_ = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                              cross_dir=cross_dir, t_pre=t_pre,
                                                              t_post=t_post,
                                                              ploton=False, force_single_output=True)
                elif event=="first_raise":
                    # Offset of the stroke that is overalpping in time with the go cue.
                    fd, t = self.beh_get_fd_trial(trial)
                    time = mkl.getTrialsTimesOfMotorEvents(fd, t)["raise"]
                # elif event=="raise_last_stroke":
                elif event=="on_stroke_1":
                    # onset of first stroke (touch)
                    ons, offs = self.strokes_extract_ons_offs(trial)
                    if len(ons)==0:
                        time = None
                    else:
                        time = ons[0]
                else:
                    assert False
                assert time is not None

            # store time.
            dict_events[event] = time
        return dict_events





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


    ######################## SPIKE TRAIN STUFF
    def _spiketrain_as_elephant(self, site, trial, save = True):
        """ Get this site and trial as elephant (Neo) SpikeTrain object
        RETURNS:
        - st, a SpikeTrain object
        (Can be None, if the spiketimes is None)
        """
        from neo.core import SpikeTrain
        from quantities import s

        # extract dat
        rs, chan = self.convert_site_to_rschan(site)
        dat = self.datall_slice_single(rs, chan, trial)
        assert dat is not None, "doesnt exist..."

        stimes = dat["spike_times"]
        if stimes is None:
            st = None
        else:
            # Convert to spike train
            st = SpikeTrain(dat["spike_times"]*s, t_stop=dat["time_off"], t_start=dat["time_on"])

        if save:
            dat = self.datall_slice_single_bysite(site, trial)
            dat["spiketrain"] = st

        return st

    def spiketrain_as_elephant_batch(self):
        """ Generate and save SpikeTrain for all site and trial
        RETURNS:
        - adds "spiketrain" as key in self.DatAll
        """

        ADDED = False # track whether datall is updated.
        for i, Dat in enumerate(self.DatAll):
            if i%500==0:
                print("spiketrain_as_elephant_batch, datall index: ", i)
            if "spiketrain" not in Dat.keys():
                # if Dat["trial0"]%50==0:
                #     print(Dat["trial0"])
                if "site" in Dat.keys():
                    site = Dat["site"]
                else:
                    site = self.convert_rschan_to_site(Dat["rs"], Dat["chan"])
                st = self._spiketrain_as_elephant(site, Dat["trial0"])
                Dat["spiketrain"] = st
                ADDED = True

        print("FINISHED - extracting spiketrain for all trials in self.DatAll")
        if ADDED:
            self._savelocal_datall()

    ####################### GENERATE POPANAL for a trial
    def popanal_generate_save_trial(self, trial, gaussian_sigma = 0.1, 
            sampling_period=0.01, print_shape_confirmation=False,
            clean_chans=True, overwrite=False):
        """ Genreate a single PopAnal object for this trial.
        Holds data across all sites
        PARAMS:
        - trial, int
        - gaussian_sigma, in sec, sigma for guassian kernel for smoothibng to 
        instant rate.
        - sampling_period, how much to slide kernel. this becomes the new sampling rate for 
        instant frate
        RETURNS:
        - PA
        NOTE:
        - saves in self.PopAnalDict[trial] = PA
        """
        # Given a trial, get a PopAnal object
        # trial = 10
        # gaussian_sigma = 0.1
        from elephant.kernels import GaussianKernel
        from elephant.statistics import time_histogram, instantaneous_rate,mean_firing_rate
        from quantities import s
        from pythonlib.neural.population import PopAnal

        if trial not in self.PopAnalDict.keys() or overwrite==True:
            # Get all spike trains for a trial
            list_sites = self.sitegetter_all(clean=clean_chans)
            list_spiketrain = []
            for site in list_sites:
                dat = self.datall_slice_single_bysite(site, trial)
                list_spiketrain.append(dat["spiketrain"])
                
            # Convert spike train to smoothed FR
            frate = instantaneous_rate(list_spiketrain, sampling_period=sampling_period*s, kernel=GaussianKernel(gaussian_sigma*s))

            # Convert to popanal
            PA = PopAnal(frate.T.magnitude, frate.times, chans = list_sites,
                spike_trains = [list_spiketrain], print_shape_confirmation=print_shape_confirmation)

            self.PopAnalDict[trial] = PA

        # Return
        return self.PopAnalDict[trial]


    ######################## STUFF RELATED TO INDEXING
    def get_all_existing_site_trial_in_datall(self, version="site_trial"):
        """ Returns info for what exists in datall.
        PARAMS:
        - version, str, decide what is in each tuple in the output list (see RETURNS)
        RETURNS:
        - list_all, sorted list of tuples, where each tuple is defined by version
        (all self explantory)
        """

        list_all = []

        # list_all = []
        # list_rschan = []
        # list_site = []
        # list_trial = []
        for Dat in self.DatAll:
            rs = Dat["rs"]
            chan = Dat["chan"]
            site = self.convert_rschan_to_site(rs, chan)
            trial = Dat["trial0"]

            # store different kinds of info
            if version=="site_trial":
                list_all.append((site, trial))
            elif version=="trial":
                list_all.append(trial)
            elif version=="site":
                list_all.append(site)
            else:
                assert False, "code it"

            # list_all.append((rs, chan, site, trial))
            # list_rschan.append((rs, chan))
            # list_site.append(site)
            # list_trial.append(trial)

        list_all = sorted(list(set(list_all)))

        # list_all = sorted(list(set(list_all)))
        # list_rschan = sorted(list(set(list_rschan)))
        # list_site = sorted(list(set(list_site)))
        # list_trial = sorted(list(set(list_trial)))

        return list_all

    def beh_fixation_success(self, trial):
        """Returns True if fixation succes (mkl data) for this trial.
        """
        from ..utils.monkeylogic import getTrialsFixationSuccess
        fd, trialml2 = self.beh_get_fd_trial(trial)
        if trialml2 not in fd["trials"].keys():
            return False
        suc = getTrialsFixationSuccess(fd, trialml2)
        return suc

    def get_trials_list(self, only_if_ml2_fixation_success=False):
        """
        Get list of ints, trials,
        PARAMS:
        - only_if_ml2_fixation_success, then keeps onl trials where the corresponding
        ml2 beh trial had fixation success. Also skips trials that dont exist in filedata at all.
        """
        trials = range(len(self.TrialsOffset))
        if only_if_ml2_fixation_success:
            trials = [t for t in trials if self.beh_fixation_success(t)]

        return trials



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
        
    def plot_spike_waveform_stats_multchans(self, XLIM=None, saveon=True):
        """
        Plot waveform stats for all channels in a grid, each will
        be a histogram.
        PARAMS:
        - XLIM, limit for histograms
        - sdir, string path. if None then doesnt save
        """
        from pythonlib.tools.plottools import subplot_helper

        # list_sites = None # none, to get all
        list_sites = self.sitegetter_all(clean=False)

        if XLIM is None:
            sharex = False
        else:
            sharex = True
        ncols = 16
        nrowsmax = 4
        getax, figholder, nplots = subplot_helper(ncols, nrowsmax, len(list_sites), SIZE = 3,
                                                 sharex=sharex, xlim=XLIM)

        # fig, axes = plt.subplots(
        for i, s in enumerate(list_sites):
            rs, chan = self.convert_site_to_rschan(s)
            print("plot_spike_waveform_stats_multchans", rs, chan)
            
            # extarct spike
            waveforms = self.load_spike_waveforms_(rs, chan)

            #### PLOTS
            # 2) stats
            ax = getax(i)
            outdict = self.spikewave_compute_stats(waveforms)
            ax.hist(outdict["volt_max"], bins=100);
            ax.hist(outdict["volt_min"], bins=100);
            ax.set_title('volt min/max')

        plt.close("all")

        # save all figs
        if saveon:
            sdir = f"{self.Paths['figs_local']}/waveforms_stats"
            os.makedirs(sdir, exist_ok=True)
            print("Saving spike waveform stats to: ", sdir)
            for i in range(nplots):
                fig, axes = figholder[i]
                fig.savefig(f"{sdir}/stats-subset{i}.jpg")
        return figholder

    def _plot_spike_waveform_multchans(self, list_sites, YLIM, saveon, prefix):
        """
        Plot waveforms for all channels in a grid
        PARAMS:
        - sdir, string path. if None then doesnt save
        - LIST_YLIM,
        """
        from pythonlib.tools.plottools import subplot_helper

        ncols = 16
        nrowsmax = 4
        getax, figholder, nplots = subplot_helper(ncols, nrowsmax, len(list_sites), SIZE = 3,
                                                 sharex=True, ylim = YLIM)

        # fig, axes = plt.subplots(
        for i, s in enumerate(list_sites):
            rs, chan = self.convert_site_to_rschan(s)
            print("_plot_spike_waveform_multchans", rs, chan)
            
            # extarct spike
            waveforms = self.load_spike_waveforms_(rs, chan)

            #### PLOTS
            # 1) wavefore
            ax = getax(i)
            self.plot_spike_waveform(ax, waveforms)
            ax.set_title(f"s{s}({rs}-{chan})")

            plt.close("all")


        # save all figs
        if saveon:
            sdir = f"{self.Paths['figs_local']}/waveforms_overlay"
            os.makedirs(sdir, exist_ok=True)
            print("Saving spike overlays to: ", sdir)
            for i in range(nplots):
                fig, axes = figholder[i]
                if YLIM is not None:
                    suff = f"{YLIM[0]}to{YLIM[1]}"
                else:
                    suff = None
                fig.savefig(f"{sdir}/{prefix}-ylim{suff}-subset{i}.jpg")

        return figholder


    def plot_spike_waveform_multchans(self, saveon=True,
            clean=False,
            LIST_YLIM = [
                [-300, 150],
                [-200, 100],
                [-400, 200],
                None]):
        """
        Plot waveforms for all channels in a grid
        PARAMS:
        - sdir, string path. if None then doesnt save
        - LIST_YLIM,
        """
        from pythonlib.tools.plottools import subplot_helper

        # list_sites = None # none, to get all
        list_sites = self.sitegetter_all(clean=clean)

        ncols = 16
        nrowsmax = 4
        prefix = f"all-clean_{clean}"
        for YLIM in LIST_YLIM:
            assert isinstance(YLIM, list), "mistake..."
            figholder = self._plot_spike_waveform_multchans(list_sites, YLIM, saveon, prefix)

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
            times_codes = self.behcode_extract_times_sequence(trial0, list_codes)
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
        """ Overlays events onto an axis
        """
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
    def plot_rasters_all(self, ax, trial, list_sites=None, site_to_highlight=None):
        """ Plot all sites onto a single axes, for this trial, aligned rasters
        PARAMS;
        - site_to_highlight, int, then will be diff color - if what want to link to
        other example figures.
        """

        list_ylabel = []
        cnt = 0
        
        if site_to_highlight is not None:
            rsrand, chanrand = self.convert_site_to_rschan(site_to_highlight)
        if list_sites is None:
            list_sites = self.sitegetter_all()

        for i, site in enumerate(list_sites):
            d = self.datall_slice_single_bysite(site, trial)
            st = d["spike_times"]
            assert st is not None, "corrupted file..., make this sitegarbage temporarily?"
            pcol = "k"
            if site_to_highlight is not None:
                if site==site_to_highlight:
                    # the random one plotted, color diff 
                    pcol = 'r';
            self.plot_raster_line(ax, st, yval=i, color=pcol, linelengths=1, alpha=0.5)

            # collect for ylabel
            rs, chan = self.convert_site_to_rschan(site)
            list_ylabel.append(f"{site}|{rs}-{chan}")
        ax.set_yticks(range(len(list_ylabel)))
        ax.set_yticklabels(list_ylabel);
        ax.set_xlabel('time rel. trial onset (sec)');
        ax.set_ylabel('site');
        ax.set_title(f"trial: {trial}| nsites: {len(list_sites)}")
        self.plotmod_overlay_trial_events(ax, trial)
        XLIM = ax.get_xlim()
        # - Overlay brain regions
        self.plotmod_overlay_brainregions(ax, list_sites)
        
    def plot_epocs(self, ax, trial, list_epocs=["camframe", "camtrialon", "camtrialoff", 
        "rewon", "rewoff", "behcode"]):
        """ Plot discrete events onto axes, for this trial
        """
        
        # -- Epochs
        ax.set_title("epocs")
        for i, pl in enumerate(list_epocs):
            times, vals = self.extract_data_tank_epocs(pl, trial0=trial)
            ax.plot(times, np.ones(times.shape)+i, 'x', label=pl)
            if pl=="behcode":
                for t, b in zip(times, vals):
                    ax.text(t, 1+i+np.random.rand(), int(b))
        ax.legend()
        self.plotmod_overlay_trial_events(ax, trial)


    def plot_stream(self, ax=None, trial=0, which="pd1"):
        """ Plot this trial and stream on ax.
        """
        if ax is None:
            fig, ax = plt.subplots(1,1)
        out = self.extract_data_tank_streams(which, trial0=trial)
        if out is not None:
            times, vals, fs = out
            ax.plot(times, vals, '-', label=which)

    # def plotwrapper_specific_trial_overview(self, trialtdt):
    #     """ Plots _everytnig_ about this trial, aligned
    #     """

    #     # assert False, "clean up and combine with drawing"
    #     fig, axes = plt.subplots(7,1, figsize=(15, 28), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1,1,1,12]})

    #     # -- Epochs
    #     ax = axes.flatten()[0]
    #     ax.set_title("epocs")
    #     list_plot = ["camframe", "camtrialon", "camtrialoff", "rewon", "rewoff", "behcode"]
    #     for i, pl in enumerate(list_plot):
    #         times, vals = self.extract_data_tank_epocs(pl, trial0=trialtdt)
    #         ax.plot(times, np.ones(times.shape)+i, 'x', label=pl)
    #         if pl=="behcode":
    #             for t, b in zip(times, vals):
    #                 ax.text(t, 1+i+np.random.rand(), int(b))
    #     ax.legend()
    #     # ax.set_ylim(-1, i+2)
    #     # Behcode
    #     # times, vals = extract_data_tank_epocs("behcode", trial0=trialtdt)
    #     # ax.plot(times, np.ones(times.shape)+i, 'x', label=pl)
    #     self.plotmod_overlay_trial_events(ax, trialtdt)

    #     def plot_stream(streamname, ax):
    #         out = self.extract_data_tank_streams(streamname, trial0=trialtdt)
    #         if out is not None:
    #             times, vals = out
    #             # times, vals = self.extract_data_tank_streams(pl, trial0=trialtdt)
    #             ax.plot(times, vals, '-', label=streamname)

    #     # - phdi
    #     ax = axes.flatten()[1]
    #     ax.set_title("photodiodes")
    #     list_plot = ["pd1", "pd2"]
    #     for i, pl in enumerate(list_plot):
    #         plot_stream(pl, ax)
    #         # out = self.extract_data_tank_streams(pl, trial0=trialtdt)
    #         # if out is not None:
    #         #     times, vals = out
    #         #     # times, vals = self.extract_data_tank_streams(pl, trial0=trialtdt)
    #         #     ax.plot(times, vals, '-', label=pl)
    #     ax.legend()
    #     self.plotmod_overlay_trial_events(ax, trialtdt)

    #     # -- Eye
    #     ax = axes.flatten()[2]
    #     ax.set_title("eyes")
    #     list_plot = ["eyex","eyey","eyediam"]
    #     for i, pl in enumerate(list_plot):
    #         plot_stream(pl, ax)
    #     ax.legend()
    #     self.plotmod_overlay_trial_events(ax, trialtdt)

    #     # -- audio
    #     ax = axes.flatten()[3]
    #     ax.set_title("audio")
    #     list_plot = ["mic"]
    #     for i, pl in enumerate(list_plot):
    #         plot_stream(pl, ax)
    #     ax.legend()
    #     self.plotmod_overlay_trial_events(ax, trialtdt)

    #     # Beh strokes (ml2)
    #     ax = axes.flatten()[4]
    #     self.plot_trial_timecourse_summary(ax, trialtdt)

    #     # A single raw channel 
    #     import random
    #     ax = axes.flatten()[5]
    #     site = random.choice(self.sitegetter_all())
    #     ax.set_title(f"ranbdom raw data: site{site}")
    #     D = self.datall_slice_single_bysite(site, trialtdt)
    #     if D is not None:
    #         t = D["tbins0"]
    #         raw = D["raw"]
    #         st = D["spike_times"]
    #         if raw is not None:
    #             ax.plot(t, raw)
    #             # spikes
    #             ax.plot(st, np.ones(st.shape), 'xr')
    #             self.plotmod_overlay_trial_events(ax, trialtdt)

    #     # -- Rasters
    #     ax = axes.flatten()[6]
    #     list_ylabel = []
    #     cnt = 0
    #     for i, (rs, chan) in enumerate(zip(list_rs, list_chans)):
    #         d = self.datall_slice_single(rs, chan, trialtdt)
    #         st = d["spike_times"]
    #         if rsrand==rs and chanrand==chan:
    #             # the random one plotted, color diff 
    #             pcol = 'r';
    #         else:
    #             pcol = 'k'
    #         self.plot_raster_line(ax, st, yval=i, color=pcol)
            
    #         if i%32==0:
    #             ax.axhline(i-0.5)
    #             try:
    #                 ax.text(-0.5, i-0.5, list_bregion[cnt], size=15, color="b")
    #             except Exception as err:
    #                 pass
    #             cnt+=1
            
    #         # collect for ylabel
    #         list_ylabel.append(f"{rs}-{chan}")
    #     ax.set_yticks(range(len(list_ylabel)))
    #     ax.set_yticklabels(list_ylabel);
    #     ax.set_xlabel('time rel. trial onset (sec)');
    #     ax.set_ylabel('site');
    #     ax.set_title(f"trialtdt: {trialtdt}")
    #     self.plotmod_overlay_trial_events(ax, trialtdt)    

    def plotwrapper_raster_multrials_onesite(self, list_trials, site, alignto=None, 
            SIZE=0.5):
        """ Plot one site, mult trials, overlaying for each trial its major events
        PARAMS:
        - list_trials, list of int. if None, then plots 20 random
        - site, int (512)
        - alignto, str or None, how to adjust times to realign.
        """

        if list_trials is None:
            import random
            nrand = 20
            list_trials = sorted(random.sample(self.get_trials_list(True), nrand))
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


    def plotwrapper_raster_oneetrial_multsites(self, trialtdt, 
            list_sites=None, site_to_highlight=None,
            WIDTH=20, HEIGHT = 10):
        """ Plot a single raster for this trial, across these sites
        PARAMS:
        - site_to_highlight, bool, if True, colors it diff
        """
        
        # fig, axes = plt.subplots(2,2, figsize=(WIDTH, HEIGHT), sharex=False, 
        #                          gridspec_kw={'height_ratios': [1,8], 'width_ratios':[8,1]})
        fig1, axes = plt.subplots(10, 1, figsize=(15, 28), sharex=True, 
            gridspec_kw={'height_ratios': [1, 1, 1, 1,1,1,1,12,1, 1]})

        # -- Epochs (events)
        ax = axes.flatten()[0]
        self.plot_epocs(ax, trialtdt)
        # XLIM = ax.get_xlim()

        # Streams
        ax = axes.flatten()[1]
        for stream in ["pd1", "pd2"]:
            self.plot_stream(ax, trialtdt, stream)
        ax.set_title("photodiodes")
        ax.legend()
        self.plotmod_overlay_trial_events(ax, trialtdt)
        # ax.set_xlim(XLIM)

        ax = axes.flatten()[2]
        for stream in ["eyex","eyey","eyediam"]:
            self.plot_stream(ax, trialtdt, stream)
        ax.set_title("eyes")
        # ax.set_xlim(XLIM)

        ax = axes.flatten()[3]
        for stream in ["mic"]:
            self.plot_stream(ax, trialtdt, stream)
        ax.set_title("mic")
        # ax.set_xlim(XLIM)

        # A single raw channel 
        import random
        ax = axes.flatten()[5]
        site = random.choice(self.sitegetter_all())
        ax.set_title(f"ranbdom raw data: site{site}")
        D = self.datall_slice_single_bysite(site, trialtdt)
        if D is not None:
            t = D["tbins0"]
            raw = D["raw"]
            st = D["spike_times"]
            if raw is not None:
                ax.plot(t, raw)
                # spikes
                ax.plot(st, np.ones(st.shape), 'xr')
                self.plotmod_overlay_trial_events(ax, trialtdt)
        
        # Beh strokes (ml2)
        ax = axes.flatten()[6]
        self.plot_trial_timecourse_summary(ax, trialtdt)
        # ax.set_xlim(XLIM)

        # -- Rasters
        ax = axes.flatten()[7]
        self.plot_rasters_all(ax, trialtdt, list_sites)
        # ax.set_xlim(XLIM)

        # Another plot for the beh and image
        fig2, axes = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))

        # -- Final drawing
        ax = axes.flatten()[0]
        self.plot_final_drawing(ax, trialtdt, strokes_only=True)
        
        # -- Image
        ax = axes.flatten()[1]
        self.plot_taskimage(ax, trialtdt)

        return fig1, fig2

    ####################################################
    def plot_behcode_photodiode_sanity_check(self):
        """ Checks that each instance of a beh code is matched to a close
        by photodiode crossing of the correct direction and timing
        RETURNS:
        - makes and saves plots
        """
        import pandas as pd
        import seaborn as sns
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        import os

        # Check whether aleady done
        sdir = f"{self.Paths['figs_local']}/sanity_checks/phdi_check"
        print(sdir)
        if False:
            if os.path.exists(sdir):
                print("Skipping plot_behcode_photodiode_sanity_check, becuase already done")
                return
        
        print("Running plot_behcode_photodiode_sanity_check")    
        os.makedirs(sdir, exist_ok=True)

        # For each trial, get the time of pd crossing
        list_behcode = [11, 91, 92, 73]
        list_whichstream = ["pd2", "pd1", "pd1", "pd1"]
        t_post = 0.4 # time from beh code to search
        t_pre = 0

        for behcode, stream in zip(list_behcode, list_whichstream):
            # behcode = 91
            # stream = 'pd1'
            # crosdir = "up"

            outdict = []
            
            if stream=="pd1":
                crosdir = "up"
            elif stream=="pd2":
                crosdir = "down"
            else:
                assert False
                
            for trial in self.get_trials_list(True):
                timecross, valscross, time_behcode, valminmax, threshold = \
                    self.behcode_get_stream_crossings_in_window(trial, behcode, 
                        whichstream=stream, cross_dir=crosdir, t_pre=t_pre, 
                        t_post=t_post)

                if trial%20==0:
                    print(trial)

                outdict.append({
                    "behcode":behcode,
                    "stream":stream,
                    "trial":trial,
                    "timescross":timecross,
                    "valscross":  valscross,
                    "time_behcode":time_behcode,
                    "valminmax":valminmax,
                    "threshold":threshold,
                    "crossdir":crosdir
                })

            ################# Plot distribution of times and magnitudes across trials.
            dfthis = pd.DataFrame(outdict)
            
            # ======= 1) plot all, n crosses
            def F(x):
                return len(x["timescross"])
            dfthis = applyFunctionToAllRows(dfthis, F, "n_crosses")
            
            fig = sns.pairplot(data=dfthis, vars=["n_crosses"])
            fig.savefig(f"{sdir}/ncrosses-code_{behcode}-codename_{self.behcode_convert(codenum=behcode, shorthand=True)}-stream_{stream}.pdf")
            
            # ===== 2) Prune to only those with crosses, and plot those
            dfthis_pruned = dfthis[dfthis["n_crosses"]>0]
            
            def F(x):
                return x["timescross"][0]
            dfthis_pruned = applyFunctionToAllRows(dfthis_pruned, F, "time_first_cross")

            def F(x):
                return x["valscross"][0]
            dfthis_pruned = applyFunctionToAllRows(dfthis_pruned, F, "val_first_cross")

            def F(x):
                return x["timescross"][0] - x["time_behcode"][0]
            dfthis_pruned = applyFunctionToAllRows(dfthis_pruned, F, "time_behcode_to_firstcross")

            def F(x):
                if x["crossdir"]=="up":
                    FLOOR = x["valminmax"][0]
                elif x["crossdir"]=="down":
                    FLOOR = x["valminmax"][1]
                else:
                    assert False
                return x["val_first_cross"] - FLOOR
            dfthis_pruned = applyFunctionToAllRows(dfthis_pruned, F, "val_first_cross_rel_floor")

            fig = sns.pairplot(data=dfthis_pruned, vars=["time_first_cross", "val_first_cross", "val_first_cross_rel_floor", "n_crosses", "time_behcode_to_firstcross"])
            fig.savefig(f"{sdir}/stats_crossings-code_{behcode}-codename_{self.behcode_convert(codenum=behcode, shorthand=True)}-stream_{stream}.pdf")
            
            # === 3) save the dataframe
            path= f"{sdir}/dataframe-code_{behcode}-codename_{self.behcode_convert(codenum=behcode, shorthand=True)}-stream_{stream}.pkl"
            dfthis.to_pickle(path)


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

    #################### LINKING TO BEH DATASET
    def datasetbeh_get_trialcode(self, trial):
        """ get trialcode for this trial(tdt)
        RETURNS:
        - trialcode, a string
        """

        date = self.Date
        index_sess, trial_ml = self._beh_get_fdnum_trial(trial)
        session_ml = self.BehSessList[index_sess]

        trialcode = f"{date}-{session_ml}-{trial_ml}"
        return trialcode
        
    def datasetbeh_trialcode_to_trial(self, trialcode):
        """ given trialcode (string) return trial in self.Dat
        """
        return self._MapperTrialcode2TrialToTrial[trialcode]



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

    def print_summarize_expt_params(self):
        """ summarize things like expt name, aniaml, date, etc."""
        print("Animal: ", self.Animal)
        print("ExptSynapse: ", self.ExptSynapse)
        print("Date: ", self.Date)
        print("RecPathBase: ", self.RecPathBase)
        print("final_dir_name: ", self.Paths["final_dir_name"])
