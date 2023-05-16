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
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles

from pythonlib.globals import PATH_NEURALMONKEY, PATH_DATA_NEURAL_RAW, PATH_DATA_NEURAL_PREPROCESSED
# PATH_NEURALMONKEY = "/data1/code/python/neuralmonkey/neuralmonkey"


assert os.path.exists(PATH_DATA_NEURAL_RAW), "might have to mount servr?"


# SMFR_SIGMA = 0.025
SMFR_SIGMA = 0.040 # 4/29/23
SMFR_TIMEBIN = 0.01

BEH_CODES = {
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

# whether must pass fixation success to call it a trial.
# leave as true, since original datall extraction was also this.
SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS = True 


def load_mult_session_helper(DATE, animal, dataset_beh_expt=None, expt = "*", MINIMAL_LOADING=True):
    """ Hacky, iterates over range(10) sessions, concatenations into a single MultSessions
    for this date.
    """

    from neuralmonkey.classes.multsessions import MultSessions

    # get list of existing sessions
    from neuralmonkey.utils.directory import find_rec_session_paths

    sessionslist = find_rec_session_paths(animal, DATE)

    SNlist = []
    for rec_session in range(len(sessionslist)):
        # go thru many, if doesnt exist will not do it.
        # rec_session = 1 # assumes one-to-one mapping between neural and beh sessions.
        print("session: ", rec_session)

        # ============= RUN
        # beh_session = rec_session+1 # 1-indexing.
        # sessdict = mkl.getSessionsList(animal, datelist=[date])

        # print("ALL SESSIONS: ")
        # print(sessdict)

        SN = load_session_helper(DATE, dataset_beh_expt, rec_session, animal, expt,
            MINIMAL_LOADING=MINIMAL_LOADING)
        SNlist.append(SN)
        print("Extracted successfully for session: ", rec_session)

    assert len(SNlist)>0, "did not find any neural sessions..."

    # Combine into all sessions
    MS = MultSessions(SNlist)
    return MS


def load_session_helper(DATE, dataset_beh_expt=None, rec_session=0, animal="Pancho", 
    expt="*", do_all_copy_to_local=False,
    extract_spiketrain_elephant=False, DEBUG_TIMING=False,
    MINIMAL_LOADING = False, BAREBONES_LOADING=False):
    """ Load a single recording session.
    PARAMS:
    - DATE, str, "yymmdd"
    - dataset_beh_expt, string, name of beh expt to link to this. If None, then skips this.
    - rec_session, int, within-day session, 0,1,2, ... 
    assumes one-to-one mapping between neural and beh sessions.
    - animal, str
    - expt, for finding raw beh data
    - extract_<stuff>, data you want to extract
    RETURNS:
    - SN, session
    """
    from ..utils.monkeylogic import session_map_from_rec_to_ml2, session_map_from_rec_to_ml2_ntrials_mapping

    if dataset_beh_expt is not None:
        assert len(dataset_beh_expt)>1, "if skip, then make this None"

    # 1) Find the raw beh data (filedata)
    beh_session, exptname, sessdict = session_map_from_rec_to_ml2(animal, DATE, rec_session)

    # # Assume that the beh sessions increment in order, matching the neural sessions
    # sessdict = mkl.getSessionsList(animal, datelist=[DATE])
    # # assume that beh sessions are indexed by neural rec sessions
    # # beh_session = rec_session+1    

    # # Which beh session maps to this neural session?
    # session_map = load_session_mapper(animal, int(DATE))
    # if session_map is not None:
    #     beh_session = session_map[rec_session]
    # else:
    #     beh_session = sessdict[DATE][rec_session][0]
    # print("Beh Sessions that exist on this date: ", DATE, sessdict)
    # print("taking this one :", beh_session)

    # beh_session = rec_session+1 # 1-indexing.
    if False:
        # get all sessions
        beh_expt_list = [sess_expt[1] for sess_expt in sessdict[DATE]]
        beh_sess_list = [sess_expt[0] for sess_expt in sessdict[DATE]]
    else:
        # Get the single session assued to map onto this neural.
        # beh_expt_list = [sess_expt[1] for sess_expt in sessdict[DATE] if sess_expt[0]==beh_session]
        # print("Found these beh expt names: ", beh_expt_list)
        # assert(len(beh_expt_list))==1, "must be error, multiple sessions with same session num"
        beh_expt_list = [exptname]
        beh_sess_list = [beh_session]
        beh_trial_map_list = [(1, 0)]

    print("Loading these beh expts:", beh_expt_list)
    print("Loading these beh sessions:",beh_sess_list)
    print("Loading this neural session:", rec_session)



    try:
        SN = Session(DATE, beh_expt_list, beh_sess_list, beh_trial_map_list, 
            rec_session = rec_session, dataset_beh_expt=dataset_beh_expt, 
            extract_spiketrain_elephant=extract_spiketrain_elephant,
            do_all_copy_to_local=do_all_copy_to_local, DEBUG_TIMING=DEBUG_TIMING,
            MINIMAL_LOADING= MINIMAL_LOADING, BAREBONES_LOADING=BAREBONES_LOADING)
    except AssertionError as err:
        print("FAILED loading session:", DATE, rec_session)
        print("Possible that this one session maps to multiple beh sessions. try loading it automatically.")
        beh_expt_list, beh_sess_list, beh_trial_map_list = session_map_from_rec_to_ml2_ntrials_mapping(
            animal, DATE, rec_session)
        print("ATTEMPTING RELOAD WITH THESE BEH SESSIONS:")
        print("beh sessions: ", beh_expt_list, beh_sess_list)
        print("beh_trial_map_list: ", beh_trial_map_list)
        SN = Session(DATE, beh_expt_list, beh_sess_list, beh_trial_map_list, 
            rec_session = rec_session, dataset_beh_expt=dataset_beh_expt, 
            extract_spiketrain_elephant=extract_spiketrain_elephant,
            do_all_copy_to_local=do_all_copy_to_local, DEBUG_TIMING=DEBUG_TIMING, 
            MINIMAL_LOADING= MINIMAL_LOADING)

    # if not MINIMAL_LOADING and RESAVE_CACHE:
    #     # Save cached
    #     self._savelocalcached_extract()
    #     self._savelocalcached_save()
            

    ############ EXTRACT STUFF
    # if not MINIMAL_LOADING and extract_raw_and_spikes_helper:
    #     # Extract spikes
    #     SN.extract_raw_and_spikes_helper()
    
    # Load dataset beh
    # if not MINIMAL_LOADING and dataset_beh_expt is not None:
    #     SN.datasetbeh_load_helper(dataset_beh_expt)

    return SN

class Session(object):
    """
    Operates over all trials for this entire session (day), even if multiple rec files for today
    """
    
    def __init__(self, datestr, beh_expt_list, beh_sess_list, 
            beh_trial_map_list=None,
            sites_garbage = None,
            expt="Lucas512-220520-115835", animal="Pancho", 
            path_base = PATH_DATA_NEURAL_RAW, 
            path_local = f"{PATH_DATA_NEURAL_PREPROCESSED}/recordings",
            rec_session=0, do_all_copy_to_local=False, 
            extract_spiketrain_elephant=False, 
            do_sanity_checks=False, do_sanity_checks_rawdupl=False,
            dataset_beh_expt= None, DEBUG_TIMING=False,
            MINIMAL_LOADING = False, BAREBONES_LOADING=False):
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
        Loading version:
        - BAREBONES_LOADING, just the metadata, like paths, trials, and sites. This cant do 
        analyses, use just for checking metadata. Doesn't require any preprocessing already don.
        - MINIMAL_LOADING, the final state, after have completed all preprocessing. This requires
        having already run load_and_preprocess_single_session(). Is quick to load, and can 
        run all analyses.
        - FULL_LOADING, intermediate state. loads all raw that is needed to save cached files
        which will allow MINIMAL_LOADING. Takes a long time to load and large data. This won't be
        allowed if you have already completed preproicessing.
        """
        from pythonlib.tools.expttools import makeTimeStamp

        if BAREBONES_LOADING:
            assert MINIMAL_LOADING == False, "must do one or the other"
        if MINIMAL_LOADING:
            assert BAREBONES_LOADING == False, "must do one or the other"
        FULL_LOADING = (MINIMAL_LOADING==False) and (BAREBONES_LOADING==False)
        assert sum([(BAREBONES_LOADING==True), (MINIMAL_LOADING==True), (FULL_LOADING==True)])==1

        if BAREBONES_LOADING:
            self._LOAD_VERSION = "BAREBONES_LOADING"
        elif MINIMAL_LOADING:
            self._LOAD_VERSION = "MINIMAL_LOADING"
        elif FULL_LOADING:
            self._LOAD_VERSION = "FULL_LOADING"
        else:
            assert False


        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, STARTING AT",  ts)
    
        # self._MINIMAL_LOADING = MINIMAL_LOADING
        # self._BAREBONES_LOADING = BAREBONES_LOADING
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

        # behcodes
        self.BehCodes = BEH_CODES
        # self._initialize_params()

        # Caching things:
        self._CachedTrialOnset = {}
        self._CachedStrokesPeanutsOnly = {}
        self._CachedStrokes = {}
        self._CachedStrokesTask = {}
        self._CachedDatSlice = {}
        self._CachedTouchData = {}
        self._CachedTrialsList = {}

        # Debug mode?
        self._DEBUG_PRUNE_SITES = False

        # Initialize paths
        self._initialize_paths()
        print("== PATHS for this expt: ")
        for k, v in self.Paths.items():
            print(k, ' -- ' , v)
        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, COMPLETED", "self._initialize_paths()", ts)


        # Metadat about good sites, etc. Run this first before other things.
        assert sites_garbage is None, "use metadata instead"
        # then look for metadata
        self.load_metadata_sites()
        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, COMPLETED:", "self.load_metadata_sites()", ts)
        # else:
        #     self.SitesGarbage = sites_garbage
        # if self.SitesGarbage is not None:
        #     assert np.all(np.diff(self.SitesGarbage)>0.), "you made mistake entering chanels (assuming going in order)?"

        # Load raw things
        print("== Loading TDT tank")
        self.load_tdt_tank(include_streams = do_all_copy_to_local)
        print("== Done")
        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, COMPLETED", "self.load_tdt_tank()", ts)

        # Find the times of all trial onsets (inneural data)
        # 1. get all onset and offset times
        self.TrialsOnset = self._behcode_extract_times(9)
        self.TrialsOffset = self._behcode_extract_times(18)

        # Sanity check: are you using the correct loading version?
        # Not allowed to do full loading if you already have gotten to final preprocessed state
        if self._LOAD_VERSION == "FULL_LOADING":
            check_dict = self._check_preprocess_status()
            assert check_dict["allow_full_loading"], "you have already completed preproicessing. use MINIMAL_LOADING instead"
        elif self._LOAD_VERSION == "MINIMAL_LOADING":
            check_dict = self._check_preprocess_status()
            assert check_dict["exists_2"], "you have not yet saved cached files. rerun load_and_preprocess_single_session"

        if MINIMAL_LOADING or FULL_LOADING:
            # Load event timing data
            print("== Trying to load events data") 
            succ = self._loadlocal_events() # Load all spike times
            if MINIMAL_LOADING:
                assert succ==True, "must have already saved local events."
            if DEBUG_TIMING:
                ts = makeTimeStamp()
                print("@@@@ DEBUG TIMING, COMPLETED", "self._loadlocal_events()", ts)
            print("== Done")

        if MINIMAL_LOADING: 
            # not FULL_LOADING. Expect to load all previously cached data.
            assert do_all_copy_to_local == False
            assert do_sanity_checks_rawdupl == False
            assert extract_spiketrain_elephant == False

            # Load previously cached.
            print("** MINIMAL_LOADING, therefore loading previuosly cached data")
            self._savelocalcached_load()

            # Hacky things to do, since cannot do in oeroginal extract of dataset.
            self.Datasetbeh.supervision_epochs_extract_orig()


        if FULL_LOADING:
            # The initial step. This saves and caches whatever is required.
            # Load spikes. THIS IS ONLY USED TO EXTRACT FURTHER DATA. not needed if already
            print("== Loading spike times")
            self.load_spike_times() # Load all spike times
            if DEBUG_TIMING:
                ts = makeTimeStamp()
                print("@@@@ DEBUG TIMING, COMPLETED", "self.load_spike_times()", ts)
            print("== Done")

            # Load beh 
            print("== Loading ml2 behavior")
            self.load_behavior()
            if DEBUG_TIMING:
                ts = makeTimeStamp()
                print("@@@@ DEBUG TIMING, COMPLETED", "self.load_behavior()", ts)
            print("== Done")

            # Check trial mapping between tdt and ml2
            self._beh_prune_trial_number()
            self._beh_validate_trial_number()
            self._beh_validate_trial_mapping(ploton=True, do_update_of_mapper=True, 
                fail_if_not_aligned=False)
            
            if DEBUG_TIMING:
                ts = makeTimeStamp()
                print("@@@@ DEBUG TIMING, COMPLETED", "self._beh_validate_trial_mapping()", ts)
                print("@@@@ DEBUG TIMING, COMPLETED", "self._beh_validate_trial_number()", ts)

            if do_all_copy_to_local:
                # Copy spike waveforms saved during tdt thresholding and extraction
                self.load_and_save_spike_waveform_images()
                if DEBUG_TIMING:
                    ts = makeTimeStamp()
                    print("@@@@ DEBUG TIMING, COMPLETED", "load_and_save_spike_waveform_images()", ts)

            if False:
                # Skip for now, need to fix:
                # at this line, gets error that list index out of range: FLOOR = x["valminmax"][0]
                if do_sanity_checks:
                    self.plot_behcode_photodiode_sanity_check()

            if do_sanity_checks_rawdupl:
                # Load raw and dupl and compare them (sanity check)
                print("RUNNIGN plot_raw_dupl_sanity_check")
                self.plot_raw_dupl_sanity_check()
                if DEBUG_TIMING:
                    ts = makeTimeStamp()
                    print("@@@@ DEBUG TIMING, COMPLETED", "self.plot_raw_dupl_sanity_check()", ts)

            # Precompute mappers (quick)
            print("RUNNIGN _generate_mappers_quickly_datasetbeh")
            self._generate_mappers_quickly_datasetbeh()
            if DEBUG_TIMING:
                ts = makeTimeStamp()
                print("@@@@ DEBUG TIMING, COMPLETED", "self._generate_mappers_quickly_datasetbeh()", ts)

            # Extract raw and spikes
            print("RUNNIGN extract_raw_and_spikes_helper")
            self.extract_raw_and_spikes_helper()

            # Extract 
            # Get spike trains for all trials.
            if extract_spiketrain_elephant:
                print("RUNNIGN spiketrain_as_elephant_batch")
                self.spiketrain_as_elephant_batch()
                if DEBUG_TIMING:
                    ts = makeTimeStamp()
                    print("@@@@ DEBUG TIMING, COMPLETED", "self.spiketrain_as_elephant_batch()", ts)

            # Load beh dataset
            print("RUNNIGN datasetbeh_load_helper")
            self.datasetbeh_load_helper(dataset_beh_expt)            

        # if not BAREBONES_LOADING:
        #     # Load event timing data
        #     print("== Trying to load events data") 
        #     succ = self._loadlocal_events() # Load all spike times
        #     if MINIMAL_LOADING:
        #         assert succ==True, "must have already saved local events."
        #     if DEBUG_TIMING:
        #         ts = makeTimeStamp()
        #         print("@@@@ DEBUG TIMING, COMPLETED", "self._loadlocal_events()", ts)
        #     print("== Done")

        #     if not MINIMAL_LOADING:
                
        #         # Load spikes. THIS IS ONLY USED TO EXTRACT FURTHER DATA. not needed if already
        #         print("== Loading spike times")
        #         self.load_spike_times() # Load all spike times
        #         if DEBUG_TIMING:
        #             ts = makeTimeStamp()
        #             print("@@@@ DEBUG TIMING, COMPLETED", "self.load_spike_times()", ts)
        #         print("== Done")

        #         # Load beh 
        #         print("== Loading ml2 behavior")
        #         self.load_behavior()
        #         if DEBUG_TIMING:
        #             ts = makeTimeStamp()
        #             print("@@@@ DEBUG TIMING, COMPLETED", "self.load_behavior()", ts)
        #         print("== Done")

        #         # Check trial mapping between tdt and ml2
        #         self._beh_validate_trial_number()
        #         self._beh_validate_trial_mapping(ploton=True, do_update_of_mapper=True, 
        #             fail_if_not_aligned=False)
        #         if DEBUG_TIMING:
        #             ts = makeTimeStamp()
        #             print("@@@@ DEBUG TIMING, COMPLETED", "self._beh_validate_trial_mapping()", ts)
        #             print("@@@@ DEBUG TIMING, COMPLETED", "self._beh_validate_trial_number()", ts)

        #         if do_all_copy_to_local:
        #             # Copy spike waveforms saved during tdt thresholding and extraction
        #             self.load_and_save_spike_waveform_images()
        #             if DEBUG_TIMING:
        #                 ts = makeTimeStamp()
        #                 print("@@@@ DEBUG TIMING, COMPLETED", "load_and_save_spike_waveform_images()", ts)

        #         if False:
        #             # Skip for now, need to fix:
        #             # at this line, gets error that list index out of range: FLOOR = x["valminmax"][0]
        #             if do_sanity_checks:
        #                 self.plot_behcode_photodiode_sanity_check()

        #         if do_sanity_checks_rawdupl:
        #             # Load raw and dupl and compare them (sanity check)
        #             print("RUNNIGN plot_raw_dupl_sanity_check")
        #             self.plot_raw_dupl_sanity_check()
        #             if DEBUG_TIMING:
        #                 ts = makeTimeStamp()
        #                 print("@@@@ DEBUG TIMING, COMPLETED", "self.plot_raw_dupl_sanity_check()", ts)

        #         # Precompute mappers (quick)
        #         print("RUNNIGN _generate_mappers_quickly_datasetbeh")
        #         self._generate_mappers_quickly_datasetbeh()
        #         if DEBUG_TIMING:
        #             ts = makeTimeStamp()
        #             print("@@@@ DEBUG TIMING, COMPLETED", "self._generate_mappers_quickly_datasetbeh()", ts)

        #         # Extract raw and spikes
        #         print("RUNNIGN extract_raw_and_spikes_helper")
        #         self.extract_raw_and_spikes_helper()

        #         # Extract 
        #         # Get spike trains for all trials.
        #         if extract_spiketrain_elephant:
        #             print("RUNNIGN spiketrain_as_elephant_batch")
        #             self.spiketrain_as_elephant_batch()
        #             if DEBUG_TIMING:
        #                 ts = makeTimeStamp()
        #                 print("@@@@ DEBUG TIMING, COMPLETED", "self.spiketrain_as_elephant_batch()", ts)

        #         # Load beh dataset
        #         print("RUNNIGN datasetbeh_load_helper")
        #         self.datasetbeh_load_helper(dataset_beh_expt)
        #     else:
        #         assert do_all_copy_to_local == False
        #         assert do_sanity_checks_rawdupl == False
        #         assert extract_spiketrain_elephant == False

        #         # Load previously cached.
        #         print("** MINIMAL_LOADING, therefore loading previuosly cached data")
        #         self._savelocalcached_load()


        # Various cleanups
        self._cleanup()
        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, COMPLETED", "self._cleanup()", ts)
            
        # Initialize mappers
        self.MapSiteToRegionCombined = {}
        self.MapSiteToRegion = {}


    ####################### PREPROCESS THINGS
    def _cleanup(self):
        """ Various things to run at end of each initialization
        - Sanity checks, etc. SHould be quick.
        """

        if self._LOAD_VERSION == "FULL_LOADING":
        # if not self._MINIMAL_LOADING and not self._BAREBONES_LOADING: # becuase doesnt load datspikes if is minimal loading.
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

    def _check_preprocess_status(self):
        """ Check the status of preprocessing, which is these steps in
        load_and_preprocess_single_session()
        RETURNS:
        - check_dict, dict holding whether each step (key) is done or not (bools).
        """

        # 1st preprocessing
        exists_1 = os.path.exists(self.Paths["datall_local"])

        # 2nd preprocessing (caching)
        exists_2 = self._savelocalcached_check_done()

        # if you have already done 2nd and deleted 1st, then would be waste of time
        # to do full loading.
        preprocessing_all_complete = not exists_1 and exists_2 
        allow_full_loading = not preprocessing_all_complete

        check_dict = {
            "exists_1":exists_1,
            "exists_2":exists_2,
            "allow_full_loading":allow_full_loading,
            "preprocessing_all_complete":preprocessing_all_complete
        }
        return check_dict


    def load_metadata_sites(self):
        """ Load info about which sites are garbage, hand coded
        PARAMS:
        """
        from pythonlib.tools.expttools import load_yaml_config
        import os

        path = f"{self.Paths['metadata_units']}/{self.Date}.yaml"
        if os.path.exists(path):
            print("Found! metada path : ", path)
            out = load_yaml_config(path)

            # if anything is list of lists, then each inner list is a diff session.
            # so take the union.
            list_sites_keys = ["sites_garbage", "sites_low_fr"]
            for key in list_sites_keys:
                if key in out.keys() and len(out[key])>0:
                    if isinstance(out[key][0], list):
                        print("Starting lengths:")
                        for list_vals in out[key]:
                            print(len(list_vals))
                        values_union = [val for list_vals in out[key] for val in list_vals]
                        # take 
                        out[key] = sorted(set(values_union))
                        print("Union length")
                        print(len(out[key]))

            # Save it
            for k, v in out.items():
                self.SitesMetadata[k] = v
        else:
            print("Sites metada path doesnt exist: ", path)

        self._sitesdirty_update()

    def sitesdirty_filter_by_spike_magnitude(self, 
            # MIN_THRESH = 90, # before 2/12/23
            MIN_THRESH = 70, # this seems better, dont throw out some decent units.
            plot_results=False, plot_spike_waveform=False, 
            update_metadata=True):
        """ Prune channels with small spikes
        Can do this dynamaically, is pretty quick.
        PARAMS:
        - MIN_THRESH, scalar, all sites with spk peak to trough (some sumary stats) 
        less than this are retrns inot
        RETURNS:
        - dataframe, with columns "site     spk_peak_to_trough  keep",
        - updates self.SitesMetadata["sites_low_spk_magn"] = sites_remove (if
        update_metadata==True)
        ===
        """
        import pandas as pd
        import numpy as np
        import seaborn as sns


        # get current sites
        sites = self.sitegetter_all(clean=False)
        res = []
        for s in sites:
            rs, chan = self.convert_site_to_rschan(s)
            spk = self.load_spike_waveforms_(rs, chan) # (nspk, ntimebins), e.g., (1000,30)

            if plot_spike_waveform:
                fig, ax = plt.subplots(1,1)
                self.plot_spike_waveform(ax, spk)
                assert False, "or else too many plots."
        
            # for each spike, get peak to trough
            minvals = np.min(spk, axis=1)
            maxvals = np.max(spk, axis=1)
            vals = maxvals - minvals
            spk_peak_to_trough = np.percentile(vals, [90])[0] # useful in some cases where the singal spike is rleatively lower frequency.

            res.append({
                "site":s,
        #         "spk_peak_to_trough":np.mean(spk_peak_to_trough)
                "spk_peak_to_trough": spk_peak_to_trough,
                "keep": spk_peak_to_trough>=MIN_THRESH
            })
        dfthis = pd.DataFrame(res)
        # print for each site
        # plot each site, coloring based on whether crossed threshold

        # OUTPUT
        sites_remove = dfthis[dfthis["keep"]==False]["site"].tolist()
        # Update metadata
        self.SitesMetadata["sites_low_spk_magn"] = sites_remove

        # Print
        print("Printing whether spikes gotten (o) or not (-) because of spike peak to trough")
        for i in range(len(dfthis)):
            s = dfthis.iloc[i]["site"]
            pt = dfthis.iloc[i]["spk_peak_to_trough"]
            keep = dfthis.iloc[i]["keep"]
            
            if keep:
                print("o ", s, pt)
            else:
                print("- ", s, pt)

        # histogram
        if plot_results:
            fig, axes = plt.subplots(2,2)
            ax = axes.flatten()[0]
            ax.hist(dfthis["spk_peak_to_trough"], bins=20)

            sns.relplot(data=dfthis, x="site", y="spk_peak_to_trough", hue="keep", aspect=2)
            plt.grid(True)


        return dfthis

    def _sitesdirty_update(self, dirty_kinds = None):

        if dirty_kinds is None:
            # dirty_kinds = ("sites_garbage", "sites_low_fr",  # before 2/13/23
            #     "sites_error_spikes", "sites_low_spk_magn")
            dirty_kinds = ("sites_garbage", 
                "sites_error_spikes", "sites_low_spk_magn")

        sites_dirty = []
        print("updating self.SitesDirty with: ", dirty_kinds)
        for kind in dirty_kinds:
            if kind=="sites_low_spk_magn" and "sites_low_spk_magn" not in self.SitesMetadata.keys():
                # Then extract it
                self.sitesdirty_filter_by_spike_magnitude()
                sites = self.SitesMetadata[kind]
                sites_dirty.extend(sites)
            elif kind in self.SitesMetadata.keys():
                sites = self.SitesMetadata[kind]
                sites_dirty.extend(sites)
            else:
                print("[_sitesdirty_update] skipping! since did not find: ", kind)
        self.SitesDirty = sorted(set(sites_dirty))


    def _generate_mappers_quickly_datasetbeh(self):
        """ generate mappers, which are dicts for mapping, e.g.,
        between indices"""

        # 1) map from trialcode to trial
        self._MapperTrialcode2TrialToTrial = {}

        for trial in self.get_trials_list():
            trialcode = self.datasetbeh_trial_to_trialcode(trial)
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

        # REmove paths that say "IGNORE"
        paths = [p for p in paths if "IGNORE" not in p]
        # assert len(paths)==1, 'not yhet coded for combining sessions'
        if len(paths)==0:
            print("***^^*")
            print(paths)
            print(self.RecSession)
            print(self.RecPathBase, path_hierarchy)
            print(self.Animal, self.Date)
            print(self.print_summarize_expt_params())
            assert False, "maybe you didn't mount server?"
        if len(paths)<self.RecSession+1:
            print("******")
            print(paths)
            print(self.RecSession)
            print(self.RecPathBase, path_hierarchy)
            print(self.Animal, self.Date)
            print(self.print_summarize_expt_params())
            assert False, "why mismatch?"

        paththis = paths[self.RecSession]
        # print(paths, self.RecSession)
        # assert False

        fnparts = deconstruct_filename(paththis)
        print(fnparts)
        final_dir_name = fnparts["filename_final_noext"]

        # Local cached
        pathbase_local = f"{self.RecPathBaseLocal}/{self.Animal}/{self.Date}/{final_dir_name}"
        import os
        os.makedirs(pathbase_local, exist_ok=True)

        # Local cached (processed), single-trial separated)
        cached_dir = f"{pathbase_local}/cached"
        os.makedirs(cached_dir, exist_ok=True)

        def _get_spikes_raw_path():
            """ checks to find path to folder holding spikes data, in order of most to 
            least desired version. Returns None if doesnt find. 
            """
            from pythonlib.tools.expttools import load_yaml_config

            # First is saved path, the one where already got spikes from?
            oldpath = None
            if os.path.exists(f"{pathbase_local}/data_spikes.pkl"):
                # Then load old paths to raw spikes
                path_paths = f"{pathbase_local}/paths.yaml"

                if os.path.exists(path_paths):
                    paths_old = load_yaml_config(path_paths)
                    oldpath = paths_old["spikes"]
                else:
                    print("then is old version, before saved paths every time save spikes") 
                    # Return the old version, which was 5.5 (blank)
                    oldpath = f"{paththis}/spikes_tdt_quick"
            if oldpath is not None:
                return oldpath

            # Second, if have not yet extracted spikes.
            for suffix in ["-4", "-4.5", "", "-3.5"]: 
                path_maybe = f"{paththis}/spikes_tdt_quick{suffix}"
                # if os.path.exists(path_maybe):
                if checkIfDirExistsAndHasFiles(path_maybe)[0]:
                    print("FOund this path for spikes: ", path_maybe)
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
            "events_local":f"{pathbase_local}/events_photodiode.pkl",
            "mapper_st2dat_local":f"{pathbase_local}/mapper_st2dat.pkl",
            "figs_local":f"{pathbase_local}/figs",
            "metadata_units":f"{PATH_NEURALMONKEY}/metadat/units",
            "cached_dir":f"{pathbase_local}/cached",
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
            try:
                fd = loadSingleDataQuick(self.Animal, self.Date, e, s)
            except Exception as err:
                print("=======")
                self.print_summarize_expt_params()
                raise err


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

        if self.Paths['spikes'] is None:
            print(self.Paths)
            self.print_summarize_expt_params()
            assert False
        fn = f"{self.Paths['spikes']}/RSn{rs}-{chan}"
        print(f"Loading this spikes file: {fn}.mat")
        if return_none_if_fail:
            try:
                mat_dict = sio.loadmat(fn)
                return mat_dict["spiketimes"]
            except zlib.error as err:
                print("[scipy error] Skipping spike times for (rs, chan): ", rs, chan)
                self.print_summarize_expt_params()
                raise
                # return None
            except Exception as err:
                print(err)
                print("[_load_spike_times] Failed for this rs, chan: ",  rs, chan)
                self.print_summarize_expt_params()
                raise
                # assert False
        else:
            mat_dict = sio.loadmat(fn)
            return mat_dict["spiketimes"]


    def _savelocalcached_extract(self):
        """
        Extract cached data from raw or loaded, i.e,, current session must be not MINIMAL_LOADING.
        RETURNS;
        - objects all starting with _Cached, e.g, self._CachedStrokesTask
        """
        assert self._LOAD_VERSION=="FULL_LOADING"
        # assert self._MINIMAL_LOADING == False
        # assert self._BAREBONES_LOADING == False

        # get trials
        self.get_trials_list(False, False)
        self.get_trials_list(False, True)
        self.get_trials_list(True, False)
        self.get_trials_list(True, True)

        for trial in self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS):
            self._CachedTrialOnset[trial] = self.ml2_get_trial_onset(trial)

        for trial in self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS):
            self._CachedStrokesTask[trial] = self.strokes_task_extract(trial)
            self._CachedStrokes[trial] = self.strokes_extract(trial, peanuts_only=False)
            self._CachedStrokesPeanutsOnly[trial] = self.strokes_extract(trial, peanuts_only=True)
            self._CachedTouchData[trial] = self.beh_extract_touch_data(trial)


    def _savelocalcached_save(self, save_dataset_beh=True, save_datslices=True):
        """
        Save to disk all cached data in self._Cached... This saves quickly.
        """

        # ONLY ALLOWED to do this if this was not using MINIMAL loading. Otherwise not sure
        # if did correct sanity checks (which si only possible wihtout minimal locading)
        assert self._LOAD_VERSION == "FULL_LOADING"
        # assert self._MINIMAL_LOADING == False
        # assert self._BAREBONES_LOADING == False

        pathdir = self.Paths["cached_dir"]

        def _save_this(this, filename):
            # _CachedTrialOnset
            path = f"{pathdir}/{filename}.pkl"
            with open(path, "wb") as f:
                pickle.dump(this, f)

        _save_this(self._CachedTrialOnset, "trial_onsets")
        _save_this(self._CachedTouchData, "touch_data")
        _save_this(self._CachedStrokes, "strokes")
        _save_this(self._CachedStrokesPeanutsOnly, "strokes_peanutsonly")
        _save_this(self._CachedStrokesTask, "strokes_task")
        _save_this(self._CachedTrialsList, "trials_list")

        if save_dataset_beh:
            self.Datasetbeh.save(pathdir)

        if save_datslices:
            list_trials = self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS)
            # list_trials = self.get_trials_list(False)
            path = f"{pathdir}/datall_site_trial"
            os.makedirs(path, exist_ok=True)
            for trial in list_trials:
                if trial%20==0:
                    print("trial:", trial)
                for site in self.sitegetter_all(clean=False):
                    this = self.datall_slice_single_bysite(site, trial)
                    paththis = f"{path}/datslice_trial{trial}_site{site}.pkl"
                    with open(paththis, "wb") as f:
                        pickle.dump(this, f)
                       
    def _savelocalcached_check_done(self):
        """
        Check if all things done for saving local cached data. Returns True if all things have been saved
        """

        pathdir = self.Paths["cached_dir"]

        def _check_this(filename):
            path = f"{pathdir}/{filename}.pkl"
            return os.path.exists(path)

        # _CachedTrialOnset
        for x in ["trials_list", "trial_onsets", "strokes", "strokes_peanutsonly", "strokes_task", "touch_data", "dataset_beh"]:
            if _check_this(x)==False:
                return False

        if self._savelocalcached_checksaved_datslice()==False:
            return False

        return True


    def _savelocalcached_load(self):
        """
        Load from disk the cached data (quick).
        """

        pathdir = self.Paths["cached_dir"]

        def _load_this(filename):
            path = f"{pathdir}/{filename}.pkl"
            with open(path, "rb") as f:
                out = pickle.load(f)
            return out

        # _CachedTrialOnset
        self._CachedTrialsList = _load_this("trials_list")
        self._CachedTrialOnset = _load_this("trial_onsets")
        self._CachedStrokes = _load_this("strokes")
        self._CachedStrokesPeanutsOnly = _load_this("strokes_peanutsonly")
        self._CachedStrokesTask = _load_this("strokes_task")
        self._CachedTouchData = _load_this("touch_data")

        paththis = f"{pathdir}/dataset_beh.pkl"
        with open(paththis, "rb") as f:
            self.Datasetbeh = pickle.load(f)

        self.Datasetbeh.LockPreprocess = True
        self._generate_mappers_quickly_datasetbeh()

    def _savelocalcached_loadextract_datslice(self, trial, site, only_check_if_exists=False):
        """
        Load a specific data slice (trial x site) from disk. 
        PARAMS:
        - only_check_if_exists, bool, if True, then doesnt load, just returns True/False 
        for whrther file exists.
        RETURNS:
        -- if saved/exits, then dict for this (trial,site), and adds to self._CachedDatSlice
        -- if doesnt exist, then returns None
        """
        import os

        if (trial, site) in self._CachedDatSlice.keys():
            # 1. Then return already extracted
            return self._CachedDatSlice[(trial, site)]
        else:
            pathdir = self.Paths["cached_dir"]
            # Try to load
            path = f"{pathdir}/datall_site_trial"
            paththis = f"{path}/datslice_trial{trial}_site{site}.pkl"

            if only_check_if_exists:
                return os.path.exists(paththis)
            else:
                if os.path.isfile(paththis):
                    # 2. Then exists, load it.
                    try:
                        with open(paththis, "rb") as f:
                            dat = pickle.load(f)
                    except Exception as err:
                        print(paththis)
                        raise err
                    self._CachedDatSlice[(trial, site)] = dat
                    return dat
                else:
                    # 3. Doesnt exist, return None.
                    return None

    def _savelocalcached_checksaved_datslice(self):
        """ Returns bool, whether all trials/sites have their
        data extraacted. Doesnt check that the files are not corrupted..
        NOTE: this is more general. checks whether oyu have completed caching
        """

        # First, check that you have cached trials. If not, 
        if not os.path.exists(f"{self.Paths['cached_dir']}/trials_list.pkl"):
            return False
        else:
            # load the trials
            import pickle
            path = f"{self.Paths['cached_dir']}/trials_list.pkl"
            with open(path, "rb") as f:
                self._CachedTrialsList = pickle.load(f)


        trials = self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS)
        sites = self.sitegetter_all(clean=True)

        for t in trials:
            for s in sites:
                exists = self._savelocalcached_loadextract_datslice(t, s, only_check_if_exists=True)
                if not exists:
                    return False
        return True


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

        if self.Paths['spikes'] is None:
            print(self.Paths)
            self.print_summarize_expt_params()
            assert False

        # 2) copy over images
        images = glob.glob(f"{self.Paths['spikes']}/*.png")
        for im in images:
            x = deconstruct_filename(im)
            
            targ = f"{sdir}/{x['filename_final_ext']}"
            if os.path.exists(targ):
                print(f"Skipping, since already copied: {targ}")
            else:
                print(f"Copying... : {x['filename_final_ext']} to {targ}")
                shutil.copy2(im, targ)
        
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

            if self.Paths['spikes'] is None:
                print(self.Paths)
                self.print_summarize_expt_params()
                assert False

            PATH_SPIKES = self.Paths["spikes"]

            if PATH_SPIKES is None:
                # Then have not extracted spikes yet...
                self.print_summarize_expt_params()
                assert False, "need to first extract spikes ...."
                
            # PATH_SPIKES = f"{self.PathRaw}/spikes_tdt_quick"
            fn = f"{PATH_SPIKES}/RSn{rs}-{chan}-snips_subset"
            try:
                mat_dict = sio.loadmat(fn)
                waveforms = mat_dict["snips"]
            except zlib.error as err:
                print("[scipy error] failed load_spike_waveforms_ for (rs, chan): ", rs, chan)
                self.print_summarize_expt_params()
                # waveforms = None
                raise
            except Exception as err:
                print(err)
                print("[load_spike_waveforms] Failed for this rs, chan: ",  rs, chan)
                print(fn)
                self.print_summarize_expt_params()
                raise
                # assert False

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
    # def convert_epoc_to_stream(self, ):

    def extract_data_tank_streams(self, which, trial0=None, ploton=False):
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

        if ploton:
            fig, ax = plt.subplots(1,1)
            ax.plot(times, vals)
            ax.set_title(f"{which}-trial_{trial0}")
        
        return times, vals, fs


    def extract_reward_stream(self, trial, fs = 1000., ploton=False):
        """ 
        - return reward triggers as_timeseries, 
        RETURNSL:
         times, vals, periodic like streams,
        where vals are 0,1 for on/off
        """
        from ..utils.timeseries import convert_discrete_events_to_time_series

        # event trimes
        ontimes = self.extract_data_tank_epocs("rewon", trial0=trial)[0]
        offtimes = self.extract_data_tank_epocs("rewoff", trial0=trial)[0]

        # time range of entire trial
        t0, tend = self.extract_timerange_trial_final(trial)

        # convert
        # event_onsets = [0.123, 0.450]
        # event_offsets = [0.300, 0.490]
        # onset of this trial cannot be positive, if so, then is remnant of previous trial.
        # remove it.
        clamp_onset_to_this = 0.
        times, vals = convert_discrete_events_to_time_series(t0, tend, 
            ontimes, offtimes, fs, ploton=ploton, clamp_onset_to_this=clamp_onset_to_this)



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

    def extract_raw_and_spikes_helper(self, trials=None, sites=None, 
        get_raw=False, save=True):
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
                print("** Loading datall from local (previusly cached) (reading from disk ....")
                with open(self.Paths["datall_local"], "rb") as f:
                    self.DatAll = pickle.load(f)
                    if self.DatAll is None:
                        self.print_summarize_expt_params()
                        assert False, "reextract DatAll..."
                print("Done loading!")

                # Add timing info, since might not be done
                self.datall_cleanup_add_things()

                # == Load mapper
                if os.path.exists(self.Paths["mapper_st2dat_local"]):
                    print("** Loading datall from local (previusly cached) (reading from disk ....")
                    with open(self.Paths["mapper_st2dat_local"], "rb") as f:
                        self._MapperSiteTrial2DatAllInd = pickle.load(f)
                else:
                    # generate mapper, slice each one and this will autoamtically extract
                    self.mapper_extract("sitetrial_to_datallind", save=save)
                
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
                self._extract_raw_and_spikes(rss, chans, t, get_raw=get_raw) 

            # generate mapper, slice each one and this will autoamtically extract
            self.mapper_extract("sitetrial_to_datallind", save=save)
            
            # Save
            if save:
                self._savelocal_datall()


    def _extract_raw_and_spikes(self, rss, chans, trialtdt, get_raw = False):
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

        return self.DatAll

    ##################### DEBUGGING
    def debug_event_photodiode_detection():
        assert False, "just notes here"
        t = 348
        alignto = "first_raise"
        sn = MS.SessionsList[0]
        sn.events_get_time_using_photodiode(t, list_events=[alignto], overwrite=True, plot_beh_code_stream=True)

    ####################### DATALL operations
    def datall_cleanup_add_things(self, only_generate_dataframe=False):
        """ Quick processing, things to add to datall in case not already 
        added 
        - Also makes self.DatAllDf (dataframe version).
        """
        print("DOING: datall_cleanup_add_things")
        import pandas as pd

        if not only_generate_dataframe:
            # Time info
            if "time_dur" not in self.DatAll[0].keys():
                print("Running self._datall_compute_timing_info")
                self._datall_compute_timing_info()
                print("Done _datall_compute_timing_info ")

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

        # clean up, add timing information, etc
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


    def extract_windowed_data_bytrial(self, times, trial0, vals=None, 
            recompute_time_rel_onset=True, pre_dur=1., post_dur=1.):
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
        
    def extract_timerange_trial_final(self, trial):
        """ Return the "final" times, which is aligned to trial onset (ml2), and
        used across all data
        RETURNS:
        - time_on, time_off, times in seconds, relative to trial onset.
        """
        _, _, _, time_on, time_off = self.extract_windowed_data_bytrial([], trial)
        return time_on, time_off

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

        site = self.convert_rschan_to_site(rs, chan)

        # First, try to load from cached.
        dat = self._savelocalcached_loadextract_datslice(trial0, site)
        if dat is not None:
            # FOudn it in cache!
            if return_index:
                return dat, None    
            else:
                return dat
        else:
            if self.DatAll is None:
                if return_index:
                    return None, None
                else:
                    return None

            index = None
            if (site, trial0) not in self._MapperSiteTrial2DatAllInd.keys():
                # Then extract
                print("/////////")
                print(site, trial0)
                print(self._MapperSiteTrial2DatAllInd.keys())
                self.print_summarize_expt_params()
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
        NOTE: only run this if modify the indices in self.DatAll...
        NOTE: it will be smart about saving, only if there is any mod.
        """
                    

        if version=="sitetrial_to_datallind":
            print("Extracting _MapperSiteTrial2DatAllInd")
            trialprint = -1
            ADDED_ITEM = False
            for index, Dat in enumerate(self.DatAll):
                site = self.convert_rschan_to_site(Dat["rs"], Dat["chan"])
                # site = Dat["site"]
                trial = Dat["trial0"]
                if (site, trial) not in self._MapperSiteTrial2DatAllInd.keys():
                    ADDED_ITEM = True
                    if trial!=trialprint:
                        print("trial: ", trial)
                        trialprint = trial
                    self._MapperSiteTrial2DatAllInd[(site, trial)] = index
            if save and ADDED_ITEM:
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

                if spiketimes is None:
                    print("****")
                    print(self.convert_site_to_rschan(sitenum), trial)
                    print(self.print_summarize_expt_params())
                    assert False, "figure out why this is None. probably in loading (scipy error) and I used to replace errors with None. I should re-extract the spikes."
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
                
        print(rs, chan)
        assert False, 'this combo of rs and chan doesnt exist in DatSpikes!'
        
    ####################### HELP CALC THINGS
    def dat_to_time(self, vals, fs):
        return dat_to_time(vals, fs)

    ####################### CONVERSIONS BETWEEN BEH AND NEURAKL
    def _beh_prune_trial_number(self):
        """ perpocess, quick ways to prune the trials, based on comparison of neural and ml2_beh data
        to remove excess trials that will cause later.
        """
        trials_all = self.get_trials_list(False, False)
        trials_exist_in_ml2 = self.get_trials_list(False, True)

        if trials_all == trials_exist_in_ml2 + [max(trials_exist_in_ml2)+1]:
            # then extra trial gotten in neural. remove it.
            print("-- pruning off the last trial, beucase not found in beh")
            self.TrialsOffset = self.TrialsOffset[:-1]
            self.TrialsOnset = self.TrialsOnset[:-1]
            # Clear cache
            self._CachedTrialsList = {}

    def _beh_validate_trial_number(self):
        """ Confirms that each neural trial has a corresponding mapping that exists in ml2 data.
        In combination with self._beh_validate_trial_mapping(), this ensures that each nbeural trials
        is mapped to _correct_ beh trial. Fails (assertion) if this check fails."""
        trials_all = self.get_trials_list(False, False)
        trials_exist_in_ml2 = self.get_trials_list(False, True)

        if not trials_all==trials_exist_in_ml2:
            # check whether this session is allowed to fail this.
            from ..utils.monkeylogic import _load_sessions_corrupted
            sessdict = _load_sessions_corrupted()
            value = (int(self.Date), self.RecSession)
            if value in sessdict[self.Animal]:
                # then ok, expect to fail
                print("_beh_validate_trial_number failed, but OK becuase is expected!!")
            else:
                print("**&*&**")
                print(trials_all)
                print(trials_exist_in_ml2)
                print([t for t in trials_all if t not in trials_exist_in_ml2])
                print([t for t in trials_exist_in_ml2 if t not in trials_all])
                self.print_summarize_expt_params()
                print("_beh_validate_trial_number failed!!")
                assert False, "there exist neural trials which are not succesuflly matched to beh trial"
        else:
            print("_beh_validate_trial_number passed!!")


    def _beh_validate_trial_mapping(self, ploton=True, do_update_of_mapper=False,
        fail_if_not_aligned=False):
        """ Use cross correlations of trial durations to validate the 
        mapping betwen tdt and ml2 trials. Goal is to (i) verify that the mapping is correc tand
        (2) try to update mapping autoamtically.
        PARAMS:
        - do_update_of_mapper, bool(False), if true, then updates the mapping
        if crosscorr lag is not 0, and other critera, see below, in order to make sure
        is not doing a drastic change. Makes change to self.BehTrialMapList.
        and self.BehTrialMapListGood. After updating, checks again and
        fails if not now aligned.
        - fail_if_not_aligned, bool (False), if true, then must be algined or fails.
        RETURNS:
        - lagshift, int, if not 0, then lag to max crosscorr of trials durs. positive
        means should look at earleir ml2 beh.
        NOTE: THis gets EVERY neural trial (that has a corresponding ml2 trial, ie.. if
        ml2 is damaged, ingnores those trials).
        """
        from scipy.signal import correlate, correlation_lags

        ACCEPTABLE_VARIATION = 0.1 # seconds, in std in difference in trials durations.

        ##### Get array of trial durations vs trial num
        # 1) tdt. this is including appended pre and post durs.
        # trials = self.get_trials_list(True) 
        trials = self.get_trials_list(False, True) 
        def get_trial_dur(t):
            T1, T2 = self.extract_timerange_trial(t)
            return T2 - T1
            # OLD: led to recursion.
            # trange = self.extract_timerange_trial_final(t) 
            # return trange[1] - trange[0]

        # Get the durations in TDT.
        trials_exist = []
        durs_exist = []
        for t in trials:
            # try:
            durs_exist.append(get_trial_dur(t))
            trials_exist.append(t)
            # except:
            #     pass

        # 2) ml2. only for those trials that exist for tdt.
        durs_exist_ml2 = []
        trials_exist_ml2 = []
        for trial in trials_exist:
            fd, t = self.beh_get_fd_trial(trial)
            tmp = mkl.getTrialsBehCodes(fd, t)
            codes = tmp["num"]
            times = tmp["time"]

            def get_code_time(codethis):
                tmp = [t for t, c in zip(times, codes) if c==codethis]
                assert len(tmp)==1
                return tmp[0]
            
            tstart = get_code_time(9)
            tend = get_code_time(18)
            dur = tend - tstart
            
            trials_exist_ml2.append(t)
            durs_exist_ml2.append(dur)
                

        # check if cross correlation peaks at 0 lag
        if len(durs_exist)==0 or len(durs_exist_ml2)==0:
            print(";;;;;;;;;;")
            print(1, durs_exist)
            print(2, durs_exist_ml2)
            print("trials", trials)
            print("trials_exist", trials_exist)
            self.print_summarize_expt_params()
            assert False, "probably refering to incorrent beh files?"
        corr = correlate(durs_exist, durs_exist_ml2)
        lags = correlation_lags(len(durs_exist), len(durs_exist_ml2))
        lagshift = lags[np.argmax(corr)] # add this to ml2 tirals
        print("-- This is the lag for optimal xcorr of tdt and ml2 trial durations (+ means should look further back into beh trials):")
        print(lagshift)

        # Check the std of the difference in durations
        # (dont use mean diff, since tdt trials have appended pre and post times)
        variation = np.std(np.array(durs_exist) - np.array(durs_exist_ml2))
        print("-- This is the variation in (tdt - ml2) durations across trials. shodl be clsoe to 0")        
        print(variation)


        ## Plots
        def _doplot():
            """ Quick plots for diagnostic. plots overlays trial durations vs. trials"""
            fig, axes = plt.subplots(1,1,figsize=(10,5))
            plt.plot(trials_exist, durs_exist, '-ok', label="tdt")
            plt.plot(trials_exist, durs_exist_ml2, '-or', label="ml2(mapped from tdt trials)")
            # plt.plot(trials_exist_ml2, durs_exist_ml2, '-gx', alpha=0.5, label="ml2(ml2 trials)")
            plt.title('Goal: k and r should overlap')
            plt.ylabel('trial durations')
            plt.xlabel('trials(tdt)')
            plt.legend()   

        def _summarize():
            fd, _ = self.beh_get_fd_trial(0)
            print("* n trials: ", len(self.get_trials_list()), len(mkl.getIndsTrials(fd)))
            print(self.BehTrialMapList)
            print(self.BehTrialMapListGood)
            print(lagshift)
            print(variation)
            print(corr[np.argmax(corr)-3:np.argmax(corr)+4])
            _doplot()

        if ploton:
            _doplot()

        if variation<ACCEPTABLE_VARIATION:
            if lagshift!=0:
                self.print_summarize_expt_params()
                assert False, "variation says they are aligned... (but lagshift doesnt)"
        else:
            if lagshift==0:
                self.print_summarize_expt_params()
                assert False, "variation says misaligned, but lagshift says algined..."

        # Check if aligned and do things.
        if lagshift!=0:
            if fail_if_not_aligned:
                _summarize()
                print("000000000000000")
                self.print_summarize_expt_params()
                assert False, "Not aligned!!!"
            if do_update_of_mapper:
                # This only runs if you have entered the default otherwise doesnt try to overwtite.
                if len(self.BehTrialMapList)>1 or self.BehTrialMapList[0]!=(1, 0) or np.abs(lagshift)>5:
                    # Then this is not default!
                    _summarize()
                    print("0000000000")
                    self.print_summarize_expt_params()
                    assert False, "the mapping you inputed is incorrect. too differnet to attempt to modify autoamtically."
                else:
                    # Update this default mapoping
                    print("Updating self.BehTrialMapList from")           
                    print(self.BehTrialMapList)

                    mapper = self.BehTrialMapList[0]
                    self.BehTrialMapList[0] = tuple([mapper[0]-lagshift, mapper[1]])     

                    print("to: ")
                    print(self.BehTrialMapList)

                    # Force an update
                    print("Old BehTrialMapListGood: ")
                    print(self.BehTrialMapListGood)
                    self.BehTrialMapListGood = None
                    self._beh_get_fdnum_trial_generate_mapper()
                    print("New BehTrialMapListGood: ")
                    print(self.BehTrialMapListGood)

                    # Check again, and fail if not fixed
                    self._beh_validate_trial_mapping(ploton=False, fail_if_not_aligned=True)
                    print("SUCCESFULLY realigned trial mappings!")
            else:
                # bad, but dont want to do update
                print("BAD ALIGNMENT! But you did not choose to update the mapper")
        else:
            print("GREAT!! mapping is correct. Doing nothing")

        return lagshift, variation



    def _beh_get_fdnum_trial_generate_mapper(self):
        """ Generate mapping between tdt and ml2
        RETURNS:
        - modifies: self.BehTrialMapListGood
        """
        from neuralmonkey.utils.conversions import get_map_trial_and_set

        # RECURSIVE. Stop
        # ntrials = len(self.get_trials_list(only_if_ml2_fixation_success=False, only_if_has_valid_ml2_trial=True))
        ntrials = len(self.get_trials_list(only_if_ml2_fixation_success=False, only_if_has_valid_ml2_trial=False))
        # ntrials = len(self.TrialsOnset) 
        self.BehTrialMapListGood = get_map_trial_and_set(self.BehTrialMapList, ntrials)

    def _beh_get_fdnum_trial(self, trialtdt):
        """ Get the filedata indices and trial indices (beh) for
        this neural trial (trialtdt).
        PARAMS:
        - doreset, then resets self.BehTrialMapListGood
        """

        if self.BehTrialMapListGood is None:
            self._beh_get_fdnum_trial_generate_mapper()

        # assert trialtdt < ntrials, "This tdt trial doesnt exist, too large..."
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
        """ return the onset of this trial in seconds, for the beh data,
        reltiave toe onset of the neural data
        """

        if trialtdt in self._CachedTrialOnset.keys():
            return self._CachedTrialOnset[trialtdt]
        else:
            from ..utils.monkeylogic import ml2_get_trial_onset as ml2gto
            # convert to trialml
            fd, trialml = self.beh_get_fd_trial(trialtdt)
            return ml2gto(fd, trialml)


    ######################## BRAIN STUFF
    def sitegetter_print_summary_nunits_by_region(self):
        """ Prints num units (clean) per region
        and total units etc
        """
        sites_all =[]
        for area, sites in self._sitegetter_generate_mapper_region_to_sites(clean=True).items():
            print(area, " : ", len(sites))
            sites_all.append(len(sites))
        print(" ------- ")
        print("TOTAL: ", sum(sites_all))
        print("MIN: ", min(sites_all))
        print("MAX: ", max(sites_all))
        print("MEAN: ", np.mean(sites_all))

        print("--------")
        print("BAD SITES (n sites)")
        ntot = 0
        for k, v in self.SitesMetadata.items():
            if k!="done_garbage":
                print(k, len(v))
                ntot+=len(v)
        print("Total (accounting for overlap): ", len(self.SitesDirty))
        # print("Total: ", ntot)

        # sn.sitegetter_all(["dlPFC_p", "dlPFC_a"])
        print("------")
        print("Summary for each overall region")
        regions_summary = self.sitegetter_get_brainregion_list()
        max_prev = 0
        print("region, nunits, --, min(sitenum), max(sitenum)")
        for regsum in regions_summary:
            sites = self.sitegetter_map_region_to_sites(regsum)
            print(regsum, len(sites), "----", min(sites), max(sites))
            min_this = min(sites)
            assert min_this > max_prev
            max_prev = min_this


    def sitegetter_summarytext(self, site):
        """ Return a string that useful for labeling
        """

        info = self.sitegetter_thissite_info(site)
        bregion = info["region"]
        rs = info["rs"]
        chan = info["chan"]
        return f"{site}|{bregion}|{rs}-{chan}"

    def sitegetter_brainregion_chan(self, region, chan):
        """ Given a regin (e.g., M1_m) and chan (1-256) return its site (1-512)
        """ 
        assert False, "rewrite using sitegetter_map_region_to_site"
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


    def _sitegetter_get_map_brainregion_to_site(self):
        """ Retgurn dict mapping from regions to sites.
        Hard coded.
        RETURNS:
        - dict[region] = list of sites
        """
        # regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
        #         "PMd_p", "PMd_a", "SMA_p", "SMA_a", 
        #         "dlPFC_p", "dlPFC_a", "vlPFC_p", "vlPFC_a", 
        #         "preSMA_p", "preSMA_a", "FP_p", "FP_a"]
        regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
                "PMd_p", "PMd_a", "dlPFC_p", "dlPFC_a", 
                "vlPFC_p", "vlPFC_a", "FP_p", "FP_a", 
                "SMA_p", "SMA_a", "preSMA_p", "preSMA_a"]
        dict_sites ={}
        for i, name in enumerate(regions_in_order):
            dict_sites[name] = list(range(1+32*i, 1+32*(i+1)))
        return dict_sites

    def _sitegetter_generate_mapper_region_to_sites(self, clean=True,
        combine_into_larger_areas=False):
        """ Generate dict mapping from region to sites, with added flexiblity of paras
        PARAMS:
        - clean, bool, whether to remove bad sites
        - combine_into_larger_areas, bool,
        RETURNS:
        - dict_sites[sitename] = list of ints.
        """

        # Get default sites
        dict_sites = self._sitegetter_get_map_brainregion_to_site()

        # Remove bad sites?
        if clean:
            assert self.SitesDirty is not None, "you need to enter which are bad sites in SitesDirty"
            for k, v in dict_sites.items():
                # remove any sites that are bad
                dict_sites[k] = [vv for vv in v if vv not in self.SitesDirty]

        if combine_into_larger_areas:
            regions_specific = dict_sites.keys()
            # regions_in_order = ["M1", "PMv", "PMd", "SMA", "dlPFC", "vlPFC",  "preSMA", "FP"]
            regions_in_order = ["M1", "PMv", "PMd", "dlPFC", "vlPFC", "FP",  "SMA", "preSMA"]
            def _regions_in(summary_region):
                """ get list of regions (e.g, ["dlPFC_a", 'dlPFC_p']) that are in this summary region (e.g., dlPFC)
                """
                return [reg for reg in regions_specific if reg.find(summary_region)==0]
            
            dict_sites_new = {}
            for reg in regions_in_order:
                regions_specific_this = _regions_in(reg)
                sites_this = [s for reg in regions_specific_this for s in dict_sites[reg]]
                dict_sites_new[reg] = sites_this
            dict_sites = dict_sites_new

        return dict_sites

    def sitegetter_get_brainregion_list(self, combine_into_larger_areas=False):
        """ Get list of str, names of all brain regions.
        """
        dict_sites = self._sitegetter_generate_mapper_region_to_sites(clean=False,
            combine_into_larger_areas=combine_into_larger_areas)
        return list(dict_sites.keys())

    def sitegetter_map_region_to_sites(self, region, clean=True):
        """ Given a region (string) map to a list of ints (sites)
        """
        mapper = self._sitegetter_generate_mapper_region_to_sites(clean, False)
        if region not in mapper.keys():
            mapper = self._sitegetter_generate_mapper_region_to_sites(clean, True)
        sites = mapper[region]

        if self._DEBUG_PRUNE_SITES:
            sites = [sites[0]]

        return sites

        # if region in mapper.keys():
        #     return mapper[region]
        # else:
        #     mapper = self._sitegetter_generate_mapper_region_to_sites(clean, True)
        #     return mapper[region]


    def sitegetter_map_site_to_region(self, site, region_combined=False):
        """ REturn the regino (str) for this site (int, 1-512)
        PARAMS:
        - region_combined, bool, if true, then uses gross areas (e.g, M1) but
        if False, then uses specific area for each array (e.g., M1_l)
        """

        if region_combined:
            Mapper = self.MapSiteToRegionCombined
        else:
            Mapper = self.MapSiteToRegion

        if len(Mapper)==0:
            # Generate it
            dict_sites = self._sitegetter_generate_mapper_region_to_sites(
                clean=False, combine_into_larger_areas=region_combined) # clean=False, since maping from sites to reg.
            for bregion, slist in dict_sites.items():
                for s in slist:
                    Mapper[s] = bregion

        return Mapper[site]
        

    def sitegetter_thissite_info(self, site, clean=False):
        """ returns info for this site in a dict
        INCLUDES even dirty sites
        """

        # Get the brain region
        dict_sites = self._sitegetter_generate_mapper_region_to_sites(clean=clean) 
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

    def _sitegetter_sort_sites_by(self, sites, by, take_top_n=None):
        """ Sort sites by some method and optionally return top n
        PARAMS:
        - sites, list of ints
        - by, str, method for sorting sites
        - take_top_n, eitehr None (ignore) or int, take top N after sorting.
        RETURNS:
        - sites_sorted, list of ints, sites sorted and (optiaolly) pruned to top n
        """

        if by=="fr":
            # Get fr for all sites
            frate_all = [self.sitestats_fr(s)["fr_mean"] for s in sites]

            # Sort them, in decresaing order of fr
            tmp = [(fr, s) for fr, s in zip(frate_all, sites)]
            tmp = sorted(tmp, key = lambda x: -x[0])
        else:
            print(by)
            assert False, "not coded"

        if take_top_n is None:
            take_top_n = len(sites)

        # get this many of the top sites (by average fr)
        sites_sorted = [tmp[i][1] for i in range(take_top_n)]

        return sites_sorted

    def sitegetter_all(self, list_regions=None, clean=True):
        """ Get all sites, in order
        MNOTE: will be in order of list_regions
        PARAMS:
        - list_regions, get only these regions. leave None to get all. if None,
        then returns all sites.
        RETURNS:
        - list of sites
        """

        if list_regions is None:
            list_regions = self.sitegetter_get_brainregion_list(combine_into_larger_areas=False)

        sites = []
        for region in list_regions:
            sites_this = self.sitegetter_map_region_to_sites(region, clean=clean)
            sites.extend(sites_this)

        # if list_regions is None:
        #     bm = self.sitegetter_brainregion("mapper", clean=clean)
        #     sites = [s for br, ss in bm.items() for s in ss]
        # else:
        #     assert isinstance(list_regions, list)           
        #     tmp = [self.sitegetter_brainregion(reg, clean=clean) for reg in list_regions]
        #     sites = [site for list_sites in tmp for site in list_sites]

            # bm = {br:sites for br, sites in bregion_mapper.items() if br in list_regions}

        return sites



    def sitegetter_brainregion(self, region=None, clean=True):
        """ Flexible mapping from region to site
        PARAMS:
        - region, Either string (specific region) or list of strings (concats the sites)
        - clean, whether to remove garbage chanels
        RETURNS:
        - 
        - out, depends on type of region
        """
        # Hard coded
        # regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
        #                     "PMd_p", "PMd_a", "dlPFC_p", "dlPFC_a", 
        #                     "vlPFC_p", "vlPFC_a", "FP_p", "FP_a",
        #                     "SMA_p", "SMA_a", "preSMA_p", "preSMA_a"]
        # regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
        #                     "PMd_p", "PMd_a", "SMA_p", "SMA_a", 
        #                     "dlPFC_p", "dlPFC_a", "vlPFC_p", "vlPFC_a", 
        #                     "preSMA_p", "preSMA_a", "FP_p", "FP_a"]

        assert False, "new version"

        # do clean
        if clean:
            assert self.SitesDirty is not None, "you need to enter which are bad sites in SitesDirty"
            for k, v in dict_sites.items():
                # remove any sites that are bad
                dict_sites[k] = [vv for vv in v if vv not in self.SitesDirty]

        # Get sites
        if region=="list_regions":
            assert False, "_sitegetter_generate_mapper_region_to_sites"
            return regions_in_order
        elif region=="mapper" or region is None:
            assert False, "_sitegetter_generate_mapper_region_to_sites"
            return dict_sites
        elif isinstance(region, int):
            assert False, "why do this"
            return dict_sites[regions_in_order[region]]
        elif isinstance(region, list):
            # then this is list of str
            list_chans = []
            for reg in region:
                sites = self.sitegetter_map_region_to_sites(reg, clean=clean)
                list_chans.extend(sites)
            return list_chans
        elif isinstance(region, str):
            return self.sitegetter_map_region_to_sites(region)
            # if region in dict_sites.keys():
            #     # Then is one of the main regions
            #     return dict_sites[region]
            # else:
            #     # Then could be a summary region
            #     def _regions_in(summary_region):
            #         """ get list of regions (e.g, ["dlPFC_a", 'dlPFC_p']) that are in this summary region (e.g., dlPFC)
            #         """
            #         return [reg for reg in regions_in_order if reg.find(summary_region)==0]
            #     return self.sitegetter_all(list_regions=_regions_in(region), clean=clean)
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
            if dat["spike_times"] is None:
                print("****")
                print(sitenum, trial)
                print(self.print_summarize_expt_params())
                assert False, "figure out why this is None. probably in loading (scipy error) and I used to replace errors with None. I should re-extract the spikes."
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

        # Confirm that has already been extracted and saved
        def _check_for_fr(dfthis):
            ERROR = False        
            if "fr" not in dfthis.columns:
                print("fr not in dfthis.columns, ", sitenum)
                ERROR = True
            elif dfthis["fr"].isna().any():
                inds_with_na = np.where(dfthis["fr"].isna())[0]
                print("fr has na for theses indices in dfthis: ", inds_with_na, "this sitenum: ", sitenum)
                print("Running extraction for these cases")
                print(inds_with_na)

                # This computes fr for each specific case, and stores in self.DatAll.
                # still have to run datall_cleanup_add_things below to push this to dataframe
                for ind in inds_with_na:
                    sitethis = dfthis.iloc[ind]["site"]
                    trialthis = dfthis.iloc[ind]["trial0"]
                    self.sitestats_fr_single(sitethis, trialthis)

                ERROR = True
            return ERROR

        # 1) check if you have fr in dataframe
        dfthis = self.DatAllDf[(self.DatAllDf["site"]==sitenum) & (self.DatAllDf["trial0"].isin(trials))]
        ERROR = _check_for_fr(dfthis)

        # 2) try again, after cleaning up
        if ERROR:
            # Then try cleaning up, maybe havent updated the dataframe yet
            self.datall_cleanup_add_things(only_generate_dataframe=True)
        dfthis = self.DatAllDf[(self.DatAllDf["site"]==sitenum) & (self.DatAllDf["trial0"].isin(trials))]
        ERROR = _check_for_fr(dfthis)

        # 3) If still error, then haven't extract into self.DatAll...
        if ERROR:
            # Now raise error, first extract
            print("dddddddddddddddddddddddddddd")
            self.print_summarize_expt_params()
            print("First extract all fr using sitestats_fr_get_and_save(True)")
            assert False

        # Get fr
        list_fr = dfthis["fr"].tolist()
        list_trials = dfthis["trial0"].tolist()

        stats = {}
        stats["list_fr"] = list_fr
        stats["list_trials"] = list_trials
        stats["fr_mean"] = np.mean(list_fr)
        return stats

    def sitestats_fr_get_and_save(self, save=True):
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
        if save:
            self._savelocal_datall()

    def sitestats_get_low_fr_sites(self, low_fr_thresh=2, savedir=None):
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
        site_fr = [f"{s}[{fr:.1f}]" for s, fr in zip(sites_sorted, frmeans_sorted)]
        print("-- SITE(FR), sorted (increasing fr): ")
        print(site_fr)

        
        ####### Get low Fr sites
        sites_lowfr = sites[frmeans<low_fr_thresh]
        nsites_fail = sum(frmeans<low_fr_thresh)
        print("Low FR sites: ", sites_lowfr)
        print("Num sites failing threshold: ", nsites_fail)

        sites_lowfr_sorted = sites_sorted[frmeans_sorted<low_fr_thresh]

        sites_lowfr_sorted_str = [str(s) for s in sites_lowfr_sorted]
        sites_lowfr_sorted_str = ', '.join(sites_lowfr_sorted_str)


        sites_lowfr_str = [str(s) for s in sites_lowfr]
        sites_lowfr_str = ', '.join(sites_lowfr_str)        
        print("low fr sites, as strings: ")
        print(sites_lowfr_str)

        # summarize
        summary = {
            "string_site_fr_sorted_increasing":site_fr,
            'nsites_fail':nsites_fail,
            "nums_site_lowfr":sites_lowfr,
            "nums_site_lowfr_sorted":sites_lowfr_sorted,
            "string_site_lowfr_sorted":sites_lowfr_sorted_str,
            "string_site_lowfr":sites_lowfr_str,
            "sites_sorted":sites_sorted,
            "frmeans_sorted":frmeans_sorted
        }

        frdict = {s:fr for s, fr in zip(sites_sorted, frmeans_sorted)}

        if savedir is not None:
            from pythonlib.tools.expttools import writeDictToYaml
            writeDictToYaml(summary, f"{savedir}/summary-frthresh_{low_fr_thresh}.yaml")

            fig.savefig(f"{savedir}/frhist-frthresh_{low_fr_thresh}.pdf")

        return fig, summary, frdict

    def plotbatch_sitestats_fr_overview(self, LIST_FR_THRESH = (2, 4.5, 10, 15, 40),
            skip_if_done=False):
        """ [preprocessing] Plots and saves fr across sites
        , including (1) printing the sites (2) histograms, 
        (3) example rasters (sites, across triasl).
        Useful for deciding what sites to throw out due to
        low fr.
        PARAMS:
        - LIST_FR_THRESH, list of nums, will make plots and save text for each of
        these possible threshold. [2, 4.5, 40], good for 5.5, 4.5, 3.5 x STD (by eye)
        NOTE: skips if finds that directory already exists (DONE)
        """
        # Plot 
        import os
        import random

        savedir = f"{self.Paths['figs_local']}/fr_distributions"

        if skip_if_done and checkIfDirExistsAndHasFiles(savedir)[1]:
        # if os.path.exists(savedir):
            return

        for fr_thresh in LIST_FR_THRESH:
            os.makedirs(savedir, exist_ok=True)
            fig, summary, frdict = self.sitestats_get_low_fr_sites(fr_thresh, savedir=savedir)
            
        # Plot example rasters, sampled uniformly across fr values.
        sites_plot = summary["sites_sorted"][::20]
        n = 40
        trials_plot = self.get_trials_list(True)
        if len(trials_plot)>n:
            trials_plot = random.sample(trials_plot, n)

        for s in sites_plot:
            fig, axes, fig_draw, axes_draw = self.plotwrapper_raster_multrials_onesite(trials_plot, s);
            fr = frdict[s]
            fig.savefig(f"{savedir}/exampleraster-fr_{fr}-site_{s}.pdf")


    ############################ BEH STUFF (ML)
    # def events_extract_stroke_onsoffs(self, trial0):
    #     """ Get the times of stroke onsets and offsets.
    #     """
    #     fd, trialml = self.beh_get_fd_trial(trial0)
    #     # trialml = convert_trialnum(trialtdt=trial0)

    def behcode_extract_times_semantic(self, codename, trial, 
        skip_constraints_that_call_self=False):
        """ [ALL BEHCODE EXTRACTION GOES THRU THIS]
        To extract behcode for semantically labeled event. This wraps methods
        that try to ensure this is the correct code, espeically in cases
        where code is found multiple times in a trial (not common), and other
        single-case errors. 
        PARAMS:
        - codename, shorthand name, as in self.behcode_shorthand()
        - skip_constraints_that_call_self, if True, then skips any call to 
        self.behcode_extract_times_semantic(), to avoide infinite recursinion.
        RETURNS:
        - list of numerical times. empty if no times found. 
        """

        # 1) convert from string name to code and get times
        times =self._behcode_extract_times(codename, trial, shorthand=True,
            refrac_period = 0.01)        
        info = self._behcode_expectations()

        # 3b) if multiple times found, resolve it.
        if len(times)>1 and codename in info["dict_constraints"].keys():
            constraints = info["dict_constraints"][codename]
            for con in constraints: # lsit of 2-tuples
                rule = con[0] # str
                prm = con[1] # flexible

                if rule=="after":
                    if not skip_constraints_that_call_self:
                        # then only keep times after the behcode in prm
                        codename_preceding = prm # str
                        # get time for this codename
                        # (skip constraints so that dont get into infinite recurstion)
                        times_other = self.behcode_extract_times_semantic(codename_preceding, 
                            trial, 
                            skip_constraints_that_call_self=True)
                        assert not len(times_other)==0
                        # it should be after the last instance of times_other
                        times = times[times > max(times_other)] # take only times following this t
                else:
                    print(constraints)
                    print(codename, trial, times)
                    assert False, "not coded"

        # 4) sanity checks
        # - Take the first time?
        list_take_first_time = info["list_take_first_time"]
        if len(times)>1 and codename in list_take_first_time:
            if False: # ok to ignore this
                # first, enforce that you apply constraints before resorting to this
                assert codename in info["dict_constraints"].keys()
            return times[:1]
        elif len(times)>1:
            if codename not in info["list_events_that_can_occur_multiple_times"]:
                assert False, "should be solved by constraints above"
            return times
        elif len(times)==0:
            list_must_exist_each_trial_shorthand = info["list_must_exist_each_trial_shorthand"]
            if codename in list_must_exist_each_trial_shorthand:
                print(codename, trial)
                assert False, "expected to find a time..."
            return []
        else:
            # length 1
            return times

    def _behcode_extract_times(self, code, trial0=None, 
            first_instance_only=False, shorthand=False, refrac_period=0.01):
        """ [ALL BEHCODE EXTRACTION GOES THRU THIS] 
        Extract the times that this code occured, in sec, as recorded by TDT 
        _NOT_ as recorded in ml2. these are almost identical. These are the "correct" values
        wrt alignemnet with neural data. 
        Autoatmically does a few things to ensure these are the most relevant codees:
        - refrac_period removes close ones, which occured early on (e.g., July 2022)
        PARAMS
        - code, eitehr either string or int
        - trial0, int, if given, then will get times reltaive to this trial
        - first_instance_only, bool, if True, then only returns the first time
        it occurs.
        - refrac_period, removes codes which occur close in time to previuos code.
        RETURNS:
        - np array of times
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

        # refrac period
        from pythonlib.tools.listtools import remove_values_refrac_period
        if refrac_period>0.:
            inds_keep, inds_remove = remove_values_refrac_period(times, refrac_period)
            # then remove them
            times = times[inds_keep]

        return times
        
    def _behcode_canonical_sequence(self):
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
            list_codes = self._behcode_canonical_sequence()

        times = np.empty(len(codes))
        for i, c in enumerate(codes):
            t = self._behcode_extract_times(c, trial0, True)
            if len(t)>0:
                times[i] = t[0]
            else:
                times[i] = np.nan
        return times    


    def behcode_get_stream_crossings_in_window(self, trial, behcode=None,
        t_pre=0.01, t_post=0.2, whichstream="pd1", do_smooth=True, smooth_win=0.015,
        ploton=False, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
        assert_single_crossing_this_trial=False, 
        assert_expected_direction_first_crossing=None,
        refrac_period_between_events=0.01,
        take_first_behcode_instance=False, 
        take_first_crossing_for_each_behcode=False,
        allow_no_crossing_per_behcode_instance_if=None):
        """ 
        Good code for extracting times of crossings of timestamp signals (e..g, photodiode) aligned to 
        behc odes. Many options for assertions or taking specific kidns of signals. 
        PARAMS:
        - behcode, {int, str, None}, if int, then finds all times of this code and all crosses. if None, then
        takes entire trial. if str, then assumes this is a shorthand name, and uses the smart
        method self.behcode_extract_times_semantic. [PREFERRED: str]
        - t_pre, t_post, to define time window (note t_pre should be positive to get time preceding).
        - do_smooth, smooth the signal before looking for crossings. Useful to avoid jitter.
        - cross_dir_to_take, string, in {'up', 'down', 'both', 'mean'}, determines which crossings to consider
        and ignroes others. both means consider all. mean says take mean of each pair, starting from (0,1)
        - assert_single_crossing_per_behcode_instance, then fails if a behcode evists that has anything other than one crossing
        - assert_single_crossing_this_trial, then fails if this trial finds != 1 net crossing (across code isntance)
        - assert_expected_direction_first_crossing, bool, fails if for any behcode instance, the frist corssing is not
        in this direction.
        - refrac_period_between_events, if events occur wthin this gap, only the first is kept. 0.01 is 
        reasonable, less than a single frame. (only applies to events within a single beh code)
        - take_first_behcode_instance, bool, then takes crossings only for the first behcode in the trial.
        - take_first_crossing_for_each_behcode, if behcode has crossing, just take the first one.
        - allow_no_crossing_per_behcode_instance_if, string, mnethod to allow excpets for cases when
        dont find stream crossings, but expected to sometimes not find, therefore allow to except. If None, then
        don't allow excptions.
        RETURNS:
        - list of dict, where each dict holds array of times for a single instance of this behcode.
        """
        from pythonlib.tools.timeseriestools import get_threshold_crossings

        assert t_post > -t_pre, "window is negative length..."

        # 1) Extract the stream siganl
        if whichstream=="touch_binary":
            times, vals = self.beh_extract_touching_binary(trial)
            vals_sm = vals
        elif whichstream=="touch_in_fixsquare_binary":
            times, vals = self.beh_extract_touch_in_fixation_square(trial)
            vals_sm = vals
        elif whichstream=="touch_done_button":
            times, vals = self.beh_extract_touch_in_done_button(trial)
            vals_sm = vals
        elif whichstream=="reward":
            times, vals = self.extract_reward_stream(trial)
            vals_sm = vals
        else:
            times, vals, fs = self.extract_data_tank_streams(whichstream, trial)
            # 2) Smooth if desired
            n = int(smooth_win * fs)
            vals_sm = np.convolve(vals, np.ones(n)/n, mode="same")
            if False:
                fig, ax = plt.subplots(1,1, figsize=(15,10))
                ax.plot(times, vals, '-k');
                ax.plot(times, vals_sm, '-r');

        # print(times, vals, vals_sm)
        # assert

        # 3) Tresholding: Get min and max values. Threshold is in betweem them.
        valminmax_global = np.percentile(vals, [0.01, 99.99])
        valmid_global = np.mean(valminmax_global)
        valrange_global = valminmax_global[1] - valminmax_global[0]

        # # - Specific code
        if isinstance(behcode, int):
            # check each time found beh code
            times_events = self._behcode_extract_times(behcode, trial, refrac_period=refrac_period_between_events)
            if take_first_behcode_instance:
                times_events = times_events[0:1]
        elif isinstance(behcode, str):
            times_events = self.behcode_extract_times_semantic(behcode, trial)
        else:
            # check entire trial, regardless of beh codes.
            times_events = [None]

        # Collect into list of dicts
        out = []
        for timethis in times_events:

            if timethis is None:
                # Then you want the entire trial
                timesthis = times
                valsthis = vals_sm
            else:
                # Slice to the desired time window around this event'
                inds = (times>=timethis-t_pre) & (times<=timethis+t_post)
                timesthis = times[inds]
                valsthis = vals_sm[inds]

            # compute local threshold
            valminmax = np.percentile(valsthis, [1, 99])

            # In some cases this is a binary signal, and yoh have window with no change.
            if valminmax[0]==valminmax[1]:
                # assert False "fix valminmax is same valuye throughotu."
                # DEFAULT:
                out.append({
                    "timecrosses":np.array([]),
                    "valcrosses":np.array([]),
                    "time_of_behcode":timethis,
                    "valminmax":valminmax,
                    "threshold":np.nan,
                    })
            else:
                # does this local window have sufficient difference between min and max? if not, then
                # assume there is no crossing.
                MIN_RATIO = 0.125
                ratio_minmax = (valminmax[1] - valminmax[0])/(valminmax_global[1] - valminmax_global[0]) 
                # asdasd
                # print(ratio_minmax, MIN_RATIO)
                if ratio_minmax<MIN_RATIO:
                    
                    # might be bad. check that most vals are close to extremes. if so, then ok
                    vals_frac_of_range = (valsthis - valminmax[0])/(valminmax[1] - valminmax[0])
                    if np.any(np.isnan(vals_frac_of_range)):
                        print("-------")
                        print("Trial:", trial)
                        print(valminmax)
                        print(ratio_minmax)
                        plt.figure()
                        plt.title('all data rel event time (plot made becuase failure.)')
                        plt.plot(times, vals_sm, ':k')
                        plt.plot(timesthis, valsthis, '-r')
                        plt.xlabel('red = time aligned to event')
                        self.print_summarize_expt_params()
                        assert False, "failed to find crossing for this instance."
                    vals_frac_of_range_abs = np.abs(vals_frac_of_range - 0.5)
                    frac_pts_close_to_extremes = np.sum(vals_frac_of_range_abs>0.25)/len(vals_frac_of_range_abs)
                    
                    if frac_pts_close_to_extremes<0.40:
                        # THen this is not a good window. no crosings 
                        if assert_single_crossing_per_behcode_instance:
                            if allow_no_crossing_per_behcode_instance_if is None:
                                # Then dont allow for any exceptions
                                print("Trial:", trial)
                                print(valminmax)
                                print(vals_frac_of_range_abs)
                                print(ratio_minmax)
                                print(frac_pts_close_to_extremes)
                                plt.figure()
                                plt.plot(times, vals_sm, ':k')
                                plt.plot(timesthis, valsthis, '-r')
                                self.print_summarize_expt_params()
                                assert False, "failed to find crossing for this instance."
                            elif allow_no_crossing_per_behcode_instance_if=="positive_val":
                                # val_diff_from_global_mid = valminmax[0] - valmid_global
                                val_diff = valminmax[0] - valminmax_global[0]
                                val_diff_ratio_to_global = val_diff/valrange_global
                                if val_diff_ratio_to_global>0.75:
                                    # Then good. the vals are close to the peak of the global values. this
                                    # means it is a postiive val. i..e, 0.75 means current vals are 0.75 of the
                                    # way from global min to global max.
                                    continue    
                            elif allow_no_crossing_per_behcode_instance_if=="negative_val":
                                val_diff = valminmax[1] - valminmax_global[0]
                                val_diff_ratio_to_global = val_diff/valrange_global
                                if val_diff_ratio_to_global<0.75:
                                    # Then good. the vals are close to the peak of the global values. this
                                    # means it is a postiive val. i..e, 0.75 means current vals are 0.75 of the
                                    # way from global min to global max.
                                    continue    
                            else:
                                print(allow_no_crossing_per_behcode_instance_if)
                                # Then dont allow for any exceptions
                                print("Trial:", trial)
                                print(valminmax)
                                print(vals_frac_of_range_abs)
                                print(ratio_minmax)
                                print(frac_pts_close_to_extremes)
                                plt.figure()
                                plt.plot(times, vals_sm, ':k')
                                plt.plot(timesthis, valsthis, '-r')
                                self.print_summarize_expt_params()
                                assert False, "failed to find crossing for this instance."

                        else:
                            # print("skipped")
                            # ok, skip it.
                            continue
                    else:
                        # OK, collect crossings
                        threshold = np.mean(valminmax)
                        try:
                            TCROSS, VCROSS = get_threshold_crossings(
                                timesthis, valsthis, threshold, cross_dir_to_take=cross_dir_to_take, 
                                expected_direction_of_first_crossing=assert_expected_direction_first_crossing, 
                                force_single_output=assert_single_crossing_per_behcode_instance, 
                                ploton=ploton, take_first_crossing=take_first_crossing_for_each_behcode)
                        except Exception as err:
                            print("Trial:", trial)
                            print(valminmax)
                            print(vals_frac_of_range_abs)
                            print(ratio_minmax)
                            print(frac_pts_close_to_extremes)
                            plt.figure()
                            plt.plot(times, vals_sm, ':k')
                            plt.plot(timesthis, valsthis, '-r')
                            self.print_summarize_expt_params()
                            raise err

                        # dont include this if is is very similar to others

                        out.append({
                            "timecrosses":TCROSS,
                            "valcrosses":VCROSS,
                            "time_of_behcode":timethis,
                            "valminmax":valminmax,
                            "threshold":threshold
                            })

                else:
                    # collect crossings
                    threshold = np.mean(valminmax)
                    TCROSS, VCROSS = get_threshold_crossings(
                        timesthis, valsthis, threshold, cross_dir_to_take=cross_dir_to_take, 
                        expected_direction_of_first_crossing=assert_expected_direction_first_crossing, 
                        force_single_output=assert_single_crossing_per_behcode_instance, 
                        ploton=ploton, take_first_crossing=take_first_crossing_for_each_behcode)

                    out.append({
                        "timecrosses":TCROSS,
                        "valcrosses":VCROSS,
                        "time_of_behcode":timethis,
                        "valminmax":valminmax,
                        "threshold":threshold
                        })

        if assert_single_crossing_this_trial:
            # then only one total crossing allowed
            # print(out)
            allcrossings = [t for o in out for t in o["timecrosses"]]
            # print(allcrossings)
            assert len(allcrossings)==1

        # # [IN PROGRESS< difficulty is expanding times in TCROPSS] Exclude crossings that are identical
        # def _already_exists(o):
        #     for item in out:
        # inds_keep = [i, o for i, o in enumerate(out) if not _already_exists(o)]
        # for o in out:
        #     if already_exists:
        #         continue


        return out


    def _behcode_expectations(self):
        """ Return dict holding expectations how behcodes are related. Useful for
        computing times, based on these constraints.
        """

        if not hasattr(self, "BehCodesExpectations"):
            # Generate new 
           
            # Initialize constraints, events should occur in order.
            dict_constraints ={}
            dict_full2short = self._behcode_get_dict_main_events()
            events_in_order = list(dict_full2short.values())
            for ev1, ev2 in zip(events_in_order[:-1], events_in_order[1:]):
                # if ev2 in ["on", "fixcue", "doneb", "post", "rew"]:
                if ev2 in ["on", "doneb", "post", "rew"]:
                    # skip these, they are not as straighfoawrd. entere by hadn
                    continue
                dict_constraints[ev2] = [("after", ev1)]

            # Ente by hadn
            dict_constraints["doneb"] = [("after", "samp")]
            dict_constraints["post"] = [("after", "go")]

            # behcodes for which you should take the first time (after appying constains)
            # (note: you should really only do this if you have defined constarints)
            list_events_that_can_occur_multiple_times = ["seqon", "rew"]
            list_take_first_time = [ev for ev in events_in_order if ev not in list_events_that_can_occur_multiple_times]

            # behcodes for which you wish to fail if you dont find it on every trial
            list_must_exist_each_trial_shorthand = [
                "on",
                "fixcue"
            ]

            info = {
                "dict_constraints":dict_constraints,
                "list_take_first_time":list_take_first_time,
                "list_must_exist_each_trial_shorthand":list_must_exist_each_trial_shorthand,
                "list_events_that_can_occur_multiple_times": list_events_that_can_occur_multiple_times
            }
            

            self.BehCodesExpectations = info

        return self.BehCodesExpectations

    def _behcode_get_dict_main_events(self):
        dict_full2short = {
            "start":"on",
            "fix cue visible":"fixcue",
            "FixationOnsetWTH":"fixtch", # prviously "fix"
            "GAstimevent_firstpres":"samp", 
            "GAstimoff_fini":"go", # previsouly go
            "DAsamp1_visible_change":"seqon",  # previuslyt "sampseq"
            "DoneButtonTouched":"doneb", # previsouljy "done"
            "DAstimoff_fini":"post", # preivusly "fb_vis"
            "reward":"rew"} # previously rew
        return dict_full2short

    def _behcode_shorthand(self, full=None, short=None):
        """ Convert between full and short, both directions
        PARAMS:
        - full, string name
        - short, string name.
        NOTE: only allowed to pass in full or short. will convert to
        the other.
        """

        dict_full2short = self._behcode_get_dict_main_events()

        # These hodl the main events for a trial.

        if full:
            assert not short
            if full in dict_full2short.keys():
                return dict_full2short[full]
            else:
                return full
        else:
            assert not full
            for k, v in dict_full2short.items():
                if v==short:
                    return k
            print(short)
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
                return self._behcode_shorthand(full=name)
            else:
                return name
        else:
            assert codenum is None
            if shorthand:
                # convert to full
                codename = self._behcode_shorthand(short=codename)
            for k, v in self.BehCodes.items():
                if v==codename:
                    return k    
            assert False
            
    ############### Beh Dataset (extracted)
    def datasetbeh_load_helper(self, dataset_beh_expt):
        """ Helps beucase can either be "main" dataset, in which case
        use dataset_beh_expt, or daily datyaset, in which case use the 
        date (gets it auto, doesnt use dataset_beh_expt)
        If cannot load datset, then tghrows error.
        PARAMS:
        - dataset_beh_expt, str, eitehr the name of expt, or None. Iether
        way will first try to get daily dataset. if fails, then tries getting
        main using this expt. 
        """
        self.DatasetbehExptname = dataset_beh_expt # need this, to work.

        # First try daily, its faster.    
        try:
            try:
                # If that doesnt work, then use the daily dataset, which 
                # is generated autmatoiocalyl (after like sep 2022)
                self.datasetbeh_load(version="daily") # daily 
                print("**Loaded dataset! daily")
            except:
                # Try loading using "null" rule, which is common
                assert dataset_beh_expt is not None, "assuming you wanted to get daily, but somehow failed and got to here... check that daily dataset actually exists."
                self.datasetbeh_load(dataset_beh_expt=dataset_beh_expt, 
                    version="main")
                print("**Loaded dataset! using rule: null")
        except Exception as err:
            print("probably need to pass in correct rule to load dataset.")
            raise err


    def datasetbeh_load(self, dataset_beh_expt=None, 
            remove_online_abort=False,
            remove_trials_not_in_dataset=False,
            version = "daily"):
        """ Load pre-extracted beahviuioral dataset and algined
        trial by trial to this recording session. 
        Tries to automtiaclly get the exptname, and rule, but you might
        have to give it manualyl.
        - Will prune self.Dat to only keep trials that are in Datset (which tehemselves
        are trials where there was successfuly touhc)
        - Will do sanity check that every clean 
        - Will do preprocess too.
        PARAMS:
        - version, str, either "main" or "daily". if main, then uses dataset_beh_expt, if
        daily, then uses self.Date
        RETURNS:
        - self.DatasetBeh, and returns
        """
        from pythonlib.dataset.dataset import Dataset
        from pythonlib.dataset.dataset_preprocess.general import preprocessDat
        from pythonlib.dataset.dataset import load_dataset, load_dataset_daily_helper

        assert self.DatAll is not None, "need to load first, run SN.extract_raw_and_spikes_helper()"

        # 1) Load Dataset
        if version=="daily":
            D = load_dataset_daily_helper(self.Animal, self.Date)
        elif version=="main":
            if self.DatasetbehExptname is None:
                # you must enter it
                expt = dataset_beh_expt
            else:
                # load saved
                expt = self.DatasetbehExptname
            D = load_dataset(self.Animal, expt)
        else:
            print(version)
            assert False

        # print("Loading this dataset: ", self.Animal, expt, dataset_beh_rule)
        # D = Dataset([], remove_online_abort=remove_online_abort)
        # D.load_dataset_helper(self.Animal, expt, rule=dataset_beh_rule)
        # D.load_tasks_helper()

        # 10/25/22 - done automaticlaly.
        # if not D._analy_preprocess_done:
        #     D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt)

        # 2) keep only the dataset trials that are included in recordings
        if False:
            # old
            trials = self.get_all_existing_site_trial_in_datall("trial")                                               
        else:
            trials = self.get_trials_list(True, True)
        list_trialcodes = [self.datasetbeh_trial_to_trialcode(t) for t in trials]
        print("- Keeping only dataset trials that exist in self.Dat")
        print("Starting length: ", len(D.Dat))
        D.filterPandas({"trialcode":list_trialcodes}, "modify")
        print("Ending length: ", len(D.Dat))

        # 3) Prune trials in self to remove trials that dont have succesfuly fix and touch.
        trials_neural_to_remove = []
        # for trial_neural, trialcode in (list_trialcodes):
        #     fd, t = self.beh_get_fd_trial(trial_neural)
        #     outcome = mkl.getTrialsOutcomesWrapper(fd, t)
        #     t2 = trials[trial_neural]
        #     print(outcome.keys())
        #     print(trial_neural, trialcode, t, t2)
        # assert False
        for trial_neural, trialcode in zip(trials, list_trialcodes):
            if trialcode not in D.Dat["trialcode"].tolist():
                # then this is only acceptable if this trial is not succesful fix or touch
                fd, t = self.beh_get_fd_trial(trial_neural)
                suc = mkl.getTrialsFixationSuccess(fd, t)

                # NOTE: this make some trials called "touched" even though no pnut touch
                # error, since these are excluded from Dataset
                touched = mkl.getTrialsTouched(fd, t)
                # tem = mkl.getTrialsOutcomesWrapper(fd,t)["trial_end_method"]
                if touched and suc:
                    print(outcome)
                    pnuts = mkl.getTrialsStrokesByPeanuts(fd, t)
                    print(pnuts)
                    print(list_trialcodes)
                    print(D.Dat["trialcode"].tolist())
                    print(trialcode)
                    assert False, "some neural data not found in beh Dataset..."
                else:
                    outcome = mkl.getTrialsOutcomesWrapper(fd, t)
                    print("Removing this trial because it is not in Dataset:", trial_neural, trialcode, t, outcome["beh_evaluation"]["trialnum"])
                    # remove this trial from self.Dat, since it has no parallele in dataset
                    trials_neural_to_remove.append(trial_neural)
        # - remove the trials.
        if remove_trials_not_in_dataset:
            assert False, "code it..."

        # --
        self.Datasetbeh = D
        self.Datasetbeh.LockPreprocess = True
        return self.Datasetbeh

    def datasetbeh_plot_example_drawing(self, trial):
        """ Plot drawing from both neural and dataset data
        Mostly for sanity check, and for saving this method
        PARAMS:
        - trial, for neural
        """
        # PLOT EXAMPLE, SHOWING MATCH across neural and beh datasets

        # 1) for each tdt trial, get its trialcode in beh
        trialcode = self.datasetbeh_trial_to_trialcode(trial)
        D = self.Datasetbeh

        # 2) Find this trialcode in Dataset
        dat = D.Dat[D.Dat["trialcode"]==trialcode]
        trial_dat = dat.index.values[0]

        # 3) Sanity check, plot using both dataset and neural
        D.plotSingleTrial(trial_dat)

        fig, ax = plt.subplots(1,1)
        # SN.plot_trial_timecourse_summary(ax, trial)
        self.plot_final_drawing(ax,trial)

        return fig, ax


    def datasetbeh_extract_dataframe(self, list_trials):
        """
        Returns D.Dat, with idnices exactly matching list_trials,
        in order, etc.
        """

        assert False, 'in progress'
        trialcodes = [sn.dataset_beh_trial_to_trialcode(t) for t in list_trials]



    ###################### GET TEMPORAL EVENTS
    def eventsanaly_helper_pre_postdur_for_analysis(self, do_prune_by_inter_event_times=False,
        just_get_list_events=False):
        """ 
        Help get pre and post-dur for this event, making sure the not impinge on adjacent
        events. Also acts as repository of pre and post durations, as well as events in order.
        PARAMS:
        - do_prune_by_inter_event_times, if True, then shortens windows if they extend past the
        empirical intervals.
        - just_get_list_events, then just get the dict_events_bounds, ignore trying to get distribtuion of
        interavals. This solves problem where fails if doesnt have all events (vcan't plot intervals.)
        RETURNS:
        - dict_events_bounds, holding events in order (keys) and their [predur postdur] as lists
        where predur is negative
        --- e.g, {'fix_touch': [-0.6, 0.6],
             'samp': [-0.6, 0.6],
             'go_cue': [-0.6, 0.23034612376199384],
             'first_raise': [-0.23034612376199384, 0.6],
             'off_stroke_last': [-0.6, 0.6],
             'doneb': [-0.6, 0.5601144949105435],
             'post': [-0.5601144949105435, 0.20174269626695995],
             'reward_all': [-0.20174269626695995, 0.6]}
        - dict_int_bounds, dict, keys are events, vals are [bounds_pre, bounds_post], where each are
        list of low and high percentile of duration from that event to pre and post event.
        """
        import numpy as np

        # these are ideal, ignoring the actual intervals (will be pruned in a bit)
        # Symmetrical, useful for analysis of alignemnet, events, segmentation in time, etc.
        dict_events_bounds = {
            "fixcue":[-0.6, 0.6], # onset of fixation cue
            "fix_touch":[-0.6, 0.6], # button-touch
            "samp":[-0.6, 0.6], # image response
            "go_cue":[-0.6, 0.6], # movement onset.
            "first_raise":[-0.6, 0.6], # image response
            "on_strokeidx_0":[-0.6, 0.6], # image response
            "off_stroke_last":[-0.6, 0.6], # image response
            "doneb":[-0.6, 0.6], # image response    
            "post":[-0.6, 0.6], # image response    
            "reward_all":[-0.6, 0.6], # image response    
        }

        if just_get_list_events:
            return dict_events_bounds, None

        # Get all events
        # eventsthis = ["on", "fixcue", "fix_touch", "samp", "go_cue", "first_raise", "on_strokeidx_0"]
        eventsthis = list(dict_events_bounds.keys())

        # only include trials with all these events
        trials = self.get_trials_list(True, True, True, events_that_must_include=eventsthis)
        trials_ignore = self.get_trials_list(True, True, True)

        print("--- Only keeping trials with all events...")
        print("This many trials (all good):", len(trials_ignore))
        print("This many trials (only if include all events):", len(trials))

        dfevents, _ = self.eventsdataframe_extract_timings(eventsthis)

        # keep only these trials
        dfeventsthis = dfevents[dfevents["trial"].isin(trials)]

        intervalnames = [f"{a}--{b}" for a, b in zip(eventsthis[:-1], eventsthis[1:])]

        # keep only those with all events
        list_vals = []
        for x in dfeventsthis["times_ordered_flat"].tolist():
            if len(x)==len(eventsthis):
                list_vals.append(x)
        # intervals = np.diff(np.stack(dfeventsthis["times_ordered_flat"]), axis=1)
        intervals = np.diff(np.stack(list_vals), axis=1)

        # Plot intervals
        fig, axes = plt.subplots(2,1)

        # 1) include all intervals kinds
        ax = axes.flatten()[0]
        for j in range(intervals.shape[1]):
            vals = intervals[:,j]
            label = intervalnames[j]
            ax.hist(vals, label=label, bins=15, histtype="step", log=True)
            minmax = np.percentile(vals, [1, 99])
            print("minmax for", label, "   =   ", minmax)
        ax.set_xlim(left=0)
        ax.legend()

        # 2) exlcude long intervals
        ax = axes.flatten()[1]
        for j in range(intervals.shape[1]):
            vals = intervals[:,j]
            label = intervalnames[j]
            minmax = np.percentile(vals, [1, 99])
            print("minmax for", label, "   =   ", minmax)
            if minmax[1]>2:
                print("SKIPPING, as too long")
                continue
            ax.hist(vals, label=label, bins=15, histtype="step", log=True)
        # ax.set_xlim([0, np.max(intervals)+0.2])
        ax.set_xlim(left=0)
        ax.legend()

        # Collect distributions for each interval between adjacent eventsz
        dict_int_bounds = {}
        prctiles = [50, 97.5] # really only the lower number matters. will cap intervals 
        # to not exceed this durations
        for i, event in enumerate(eventsthis):
            
            if i>0:
                # get interval to previuos event
                int_pre = np.percentile(intervals[:, i-1], prctiles)
            else:
                int_pre = None
            
            if i<len(eventsthis)-1:
                # get interval to next event
                int_post = np.percentile(intervals[:, i], prctiles)
            else:
                int_post = None
                
            # Save it
            dict_int_bounds[event] = [int_pre, int_post]
                        

        if do_prune_by_inter_event_times:
            # Update the pre and postdur for each event, using empirical values
            for event, bounds_hand in dict_events_bounds.items():
                
                # predur
                minmax_dur = dict_int_bounds[event][0]
                dur = bounds_hand[0]
                if minmax_dur is None:
                    # then dont change anythin
                    pass    
                elif -dur > minmax_dur[0]:
                    # prune it
                    dur = -minmax_dur[0]
                else:
                    # duration ok
                    pass
                bounds_hand[0] = dur
                
                # postdur
                minmax_dur = dict_int_bounds[event][1]
                dur = bounds_hand[1]
                if minmax_dur is None:
                    # then dont change anythin
                    pass    
                elif dur > minmax_dur[0]:
                    # prune it
                    dur = minmax_dur[0]
                else:
                    # duration ok
                    pass
                bounds_hand[1] = dur


        # plot durations
        fig, ax = plt.subplots()
        for i, (ev, bounds) in enumerate(dict_events_bounds.items()):
            ax.plot(bounds, [i, i], 'o-')
        ax.axvline(0)
        list_ev = list(dict_events_bounds.keys())
        ax.set_yticks(range(len(list_ev)), labels=list_ev);       
        ax.set_ylabel('predur (left) and postdur (right)') 

        return dict_events_bounds, dict_int_bounds

    def events_get_times_as_array(self, trial, list_events):
        """ 
        return as array, where take first crossing if exists., and
        is nan if doesnt. in order inputed in list_events
        """
        eventsdict = self.events_get_time_using_photodiode(trial, list_events=list_events)
        out = np.empty(len(list_events))
        for i, ev in enumerate(list_events):
            times = eventsdict[ev]
            if len(times)==0:
                out[i] = np.nan
            else:
                out[i] = times[0]
        return out
    def events_does_trial_include_all_events(self, trial, list_events):
        """
        REturns True if this trial includes each event at least one time
        """
        events_array = self.events_get_times_as_array(trial, list_events)
        return ~np.any(np.isnan(events_array))



    def events_get_time_sorted(self, trial, 
        list_events = ("stim_onset", "go", "first_raise", "on_stroke_1")):
        """ Get times of these events, sorted both by (i) their first occurances within the trial
        and (ii) first flatten to array of times, and sort that.
        RETURNS:
        (see example)
        EXAMPLE:
        ({'stim_onset': [1.7369399032181718],
              'go_cue': [3.1092637157717036],
              'first_raise': [3.351, 1.8],
              'on_stroke_1': [3.616]},
             [1.7369399032181718, 1.8, 3.1092637157717036, 3.616],
             [[1.7369399032181718], [3.351, 1.8], [3.1092637157717036], [3.616]],
             ['stim_onset', 'first_raise', 'go_cue', 'on_stroke_1'],
             [0, 2, 1, 3],
             [1.7369399032181718, 1.8, 3.1092637157717036, 3.351, 3.616],
             ['stim_onset', 'first_raise', 'go_cue', 'first_raise', 'on_stroke_1'],
             [0, 2, 1, 2, 3])
        """
        eventsdict = self.events_get_time_using_photodiode(trial, list_events=list_events)
        # x = self.events_get_time_using_photodiode(trial, list_events=list_events)
    
        ####### flatten and sort
        time_events_flat = [(time, ev) for ev in list_events for time in eventsdict[ev]]
        time_events_flat = sorted(time_events_flat, key=lambda x:x[0])

        def _this(x):
            if len(x)>0:
                return x[0]
            else:
                return np.nan
        time_events_flat_first_unsorted = [_this(eventsdict[ev]) for ev in list_events]
        time_events_flat_first_unsorted = np.asarray(time_events_flat_first_unsorted)

        times_ordered_flat = [x[0] for x in time_events_flat]
        events_ordered_flat = [x[1] for x in time_events_flat]

        ####### sort by first instance 
        # sort, and only keep if found
        time_alltimes_ev = [(min(eventsdict[ev]), eventsdict[ev], ev) for ev in list_events if len(eventsdict[ev])>0]
        time_alltimes_ev = sorted(time_alltimes_ev, key=lambda x:x[0]) # sort by first occurance

        # reextract event sand times
        events_ordered_by_firsttime = [x[2] for x in time_alltimes_ev]
        firsttimes_ordered_by_firsttime = [x[0] for x in time_alltimes_ev]
        alltimes_ordered_by_firsttime = [x[1] for x in time_alltimes_ev]

        # map from events to indices
        eventinds_ordered_by_firsttime = [list_events.index(ev) for ev in events_ordered_by_firsttime]
        eventinds_ordered_flat = [list_events.index(ev) for ev in events_ordered_flat]

        # out = [eventsdict, firsttimes_ordered_by_firsttime, alltimes_ordered_by_firsttime, \
        #     events_ordered_by_firsttime, eventinds_ordered_by_firsttime, \
        #     times_ordered_flat, events_ordered_flat]

        return eventsdict, firsttimes_ordered_by_firsttime, alltimes_ordered_by_firsttime, \
            events_ordered_by_firsttime, eventinds_ordered_by_firsttime, \
            times_ordered_flat, events_ordered_flat, eventinds_ordered_flat, time_events_flat_first_unsorted


    def events_get_time_using_photodiode_and_save(self):
        """ Extract and save for all trials. using all events. GOod for preprocessing.
        """

        list_events = self.events_default_list_events()

        for trial in self.get_trials_list(True, True):
            # Extract (it skips extraction if already exists)
            self.events_get_time_using_photodiode(trial, list_events)

        # save
        self._savelocal_events()


    def _loadlocal_events(self):
        """ Load into EventsTimeUsingPhd, skip if doesnt exist
        RETURNS:
        - succesly_loaded, bool.
        """
        path = self.Paths["events_local"]
        print(f"Loading this events (pd) locally to: ", path)

        if os.path.exists(path):
            with open(path, "rb") as f:
                self.EventsTimeUsingPhd = pickle.load(f)
            return True
        else:
            print("_loadlocal_events DOESNT EXIST")
            return False


    def _savelocal_events(self):
        """  Save for faster loading later
        """
        print("Saving events (pd) locally to: ", self.Paths["events_local"])
        with open(self.Paths["events_local"], "wb") as f:
            pickle.dump(self.EventsTimeUsingPhd, f)
        print("DONE")

    def events_read_time_from_cached(self, trial, event):
        """ Helper to read pre-computed event times. This is better than reading
        directly from self.EventsTimeUsingPhd, beucase here you can use either the old
        or new eventmanes. (it translate between new and old eventnames)
        PARAMS:
        - trial, int
        - event, string name.
        RETURNS:
        - times, list of times, deals with situation where event was saved
        using old or new names.
        - or None, reason,
        if either of folliwing suations:
        --- 1. events not previously cached
        --- 2. cached, but this (trial, event) not gotten
        where None is None
        reason is string, either "events_not_cached" or "trial_event_not_found", or "found"
        """

        if not hasattr(self, "EventsTimeUsingPhd"):
            self.EventsTimeUsingPhd = {}
            return None, "events_not_cached"
        if len(self.EventsTimeUsingPhd)==0:
            return None, "events_not_cached"


        map_event_newname_to_oldname = {
            "stim_onset":"samp",
            "fix_touch":"fixtch", 
            "go_cue":"go", 
            "samp_sequence_onset":"seqon",
            "done_button":"doneb",
            "post_screen_onset":"post"
        }

        def _query(key):
            """ Returns either the times or None (if key not in dict)
            """
            if key in self.EventsTimeUsingPhd.keys():
                return self.EventsTimeUsingPhd[key], "found"
            else:
                return None, "trial_event_not_found"

        # 1) try the input
        key = (trial, event)
        out, reason = _query(key)
        if out is None:
            # 2) then try see if this is old/new name
            if event in map_event_newname_to_oldname.keys():
                # This is old name. try the new name
                ev = map_event_newname_to_oldname[event]
                key = (trial, ev)
                return _query(key)
            elif event in map_event_newname_to_oldname.values():
                # Then this is new name, try old.
                ev = [k for k, v in map_event_newname_to_oldname.items() if v==event][0]
                key = (trial, ev)
                return _query(key)
            else:
                return out, reason
        else:
            return out, reason


    def events_get_time_using_photodiode(self, trial, 
        list_events = ("stim_onset", "go", "first_raise", "on_stroke_1"),
        overwrite=False, plot_beh_code_stream = False,
        do_reextract= False):
        """
        [GOOD] Get dict of times of important events. Uses variety of methods, including
        (i) photodiode (ii) motor behavior, (iii) beh codes, wherever appropriate.
        - All times relative to behcode 9 (trial onset) by convention.
        PARAMS:
        - list_events, list of str, each a label for an event. only gets those in this list.
        - force_single_output, if True, then asserts that ther eis one and only one crossing.
        - do_reextract, bool, if True, then tries to reextract if doesnt find this event..useful 
        because some events data are pre-saved, and new code might be better at extracting it.
        RETURNS:
        - then returns a single dict, keys, are the list_events, each a list of times. THis is empty
        if this event doesnt make sense (e..g, no fixation) or  not detected for any reason.
        SAVES:
        - in cached: self.EventsTimeUsingPhd
        - to disk (because takes a while): 
        """

        list_events_skip_if_no_fixation = ["samp", "stim_onset", "go", "seqon", 
            "first_raise", "on_stroke_1", "off_stroke_last", "doneb", "post", 
            "rew", "reward_all"]

        def _extract_times(out):
            """
            REturns flattened array of times
            """
            allcrossings = [t for o in out for t in o["timecrosses"]]
            return allcrossings

        def _resort_to_behcode_time(event, padding):
            """ If truly fail pd, then resort to behcode time, plus
            padding
            PARAMS;
            - event, str, semantic
            """                        

            times = self.behcode_extract_times_semantic(event, trial)
            times = [t+padding for t in times]
            return times

 
        def compute_times_from_scratch(event):
            """ Returns times, list of scalars. Empty list if no times found.
            """
            # 1) Skip this, if no fixation success
            if not self.beh_fixation_success(trial) and event in list_events_skip_if_no_fixation:
                times = []
            else:
                if event=="on":
                    # start of the trial. use the pd crossing, whihc reflect screen color chanigng.
                    # but not that crucuail to get a precise value for trail onset.
                    # NOTE: this sometimes fails for trial 0 (photodiode is in diff state?)

                    try:
                        # assert False, "this doesnt work that well for the first trial of the day... fix that before use"
                        out = self.behcode_get_stream_crossings_in_window(trial, 9, t_pre=0, t_post = 0.15, whichstream="pd2", 
                                                ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                                                assert_single_crossing_this_trial = True,
                                                assert_expected_direction_first_crossing = "up")
                        times = _extract_times(out)
                    except AssertionError:
                        if trial<20:
                            # then is early trial, when I know this sometimes fails.
                            times = _resort_to_behcode_time(event, 0.)
                        else:
                            raise err

                elif event=="fixcue":
                    # fixation cue visible.
                    # out = self.behcode_get_stream_crossings_in_window(trial, 103, t_pre=0.015, t_post = 0.16, whichstream="pd2", 
                    #                         ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=True,
                    #                         assert_single_crossing_this_trial = True,
                    #                         assert_expected_direction_first_crossing = None)
                    out = self.behcode_get_stream_crossings_in_window(trial, event, t_pre=0.015, t_post = 0.16, whichstream="pd2", 
                                            ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=True,
                                            assert_single_crossing_this_trial = True,
                                            assert_expected_direction_first_crossing = None)
                    times = _extract_times(out)

                elif event in ["fixtch", "fix_touch"]:
                    behcode = "fixtch" 
                    try:
                        # onset of touch of fixation cue, based on detection of finger on screen.
                        # NOTE: if fails, likely t_pre should be even larger (since could touch but fail to trigger the eventcode for some time)
                        # out = self.behcode_get_stream_crossings_in_window(trial, 16, t_pre=0.15, t_post = 0.25, whichstream="touch_in_fixsquare_binary", 
                        #                                           ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                        #                                             assert_single_crossing_this_trial = True,
                        #                                              assert_expected_direction_first_crossing = "up")              
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.15, t_post = 0.25, whichstream="touch_in_fixsquare_binary", 
                                                                  ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                                                                    assert_single_crossing_this_trial = True,
                                                                      assert_expected_direction_first_crossing = "up",
                                                                      refrac_period_between_events=0.05)              
                    except AssertionError as err:
                        # Try with larger window
                        assert len(self._behcode_extract_times(behcode, trial, shorthand=True))<2, "then doent want to expand window"

                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.5, t_post = 1, whichstream="touch_in_fixsquare_binary", 
                                                                  ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                                                                    assert_single_crossing_this_trial = True,
                                                                     assert_expected_direction_first_crossing = "up",
                                                                     refrac_period_between_events=0.05)              

                    times = _extract_times(out)


                elif event in ["samp", "stim_onset"]:
                    # Use photodiode
                    # behcode = 91
                    behcode = "samp"
                    stream = 'pd1'
                    cross_dir = 'up'
                    t_pre = -0.005
                    t_post = 0.16
                    assert_single_crossing_this_trial = True
                    assert_expected_direction_first_crossing = None
                    take_first_behcode_instance = False

                    try:
                        VER = 1
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                                  cross_dir_to_take=cross_dir, t_pre=t_pre,
                                                                  t_post=t_post,
                                                                  ploton=plot_beh_code_stream, assert_single_crossing_per_behcode_instance=True, 
                                                                  assert_single_crossing_this_trial = True) 
                        times = _extract_times(out)
                        times_behcode = self.behcode_extract_times_semantic(behcode, trial)

                    except AssertionError as err:
                        # on certain days, this pd was bad. therefore extract the photodiode time + padding.
                        print("*******************")
                        VER = 2
                        times = self.behcode_extract_times_semantic(behcode, trial)
                        times_behcode = times

                        print(VER, ' == ' , times, ' -- ' , times_behcode)
                        assert False, "decide on delay to add to times, based on looking thru many trials."



                elif event in ["go", "go_cue"]:
                    # Use photodiode
                    # behcode = self.behcode_convert(codename="go", shorthand=True)
                    behcode = "go"
                    stream = 'pd2'
                    cross_dir = 'down'
                    t_pre = -0.01
                    t_post = 0.16
                    out = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                              cross_dir_to_take=cross_dir, t_pre=t_pre,
                                                              t_post=t_post,
                                                              ploton=plot_beh_code_stream, assert_single_crossing_per_behcode_instance=True, 
                                                              assert_single_crossing_this_trial = True,
                                                              take_first_behcode_instance=True)
                    times = _extract_times(out)

                elif event=="first_raise":
                    # Offset of the stroke that is overalpping in time with the go cue.

                    if False:
                        # OLD method, not necessariyl accurate 
                        fd, t = self.beh_get_fd_trial(trial)
                        # assert False, 'this not accurate'
                        times = [mkl.getTrialsTimesOfMotorEvents(fd, t)["raise"]]
                    else:
                        whichstream = "touch_in_fixsquare_binary" # problem is sometimes gets out of box slightly, then this counts as raise even though not (and counts 2 raises)
                        # behcode = self.behcode_convert(codename="go", shorthand=True)
                        behcode = "go"
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, 
                                                t_pre=0.35, t_post = 2., whichstream=whichstream, 
                                          ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=False,
                                        assert_single_crossing_this_trial = False,
                                        assert_expected_direction_first_crossing = "down",
                                        take_first_behcode_instance=True,
                                        take_first_crossing_for_each_behcode=True) 
                        times = _extract_times(out)

                        if len(times)==0:

                            # on rare occasions, he doesnt lift for a loing time. 
                            out = self.behcode_get_stream_crossings_in_window(trial, behcode, 
                                                    t_pre=0.35, t_post = 12., whichstream=whichstream, 
                                              ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=False,
                                            assert_single_crossing_this_trial = False,
                                            assert_expected_direction_first_crossing = "down",
                                            take_first_behcode_instance=True,
                                            take_first_crossing_for_each_behcode=True) 
                            times = _extract_times(out)

                        # take the first time, sometimes can go in and out of squiare..
                        times = times[0:1]

                elif event in ["samp_sequence_onset", "seqon"]:
                    # for sequence traiing, when samp1 is shown (e..g a stroke turning on)
                    # use negative t_pre becuase sometimes previous peak bleeds into the current (when
                    # this signal is being used to signal dot changes)
                    # behcode = 74
                    behcode = "seqon"
                    out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=-0.017, t_post = 0.15, whichstream="pd1", 
                                          ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False, 
                                                              assert_single_crossing_this_trial = False,
                                                                assert_expected_direction_first_crossing = None,
                                                                take_first_crossing_for_each_behcode=True)
                    # collect all the times.
                    times = _extract_times(out)

                elif event=="on_stroke_1":
                    # onset of first stroke (touch)
                    ons, _ = self.strokes_extract_ons_offs(trial)
                    if len(ons)==0:
                        times = []
                    else:
                        times = [ons[0]]

                elif "on_strokeidx_" in event:
                    # e.g., on_strokeidx_2 means onset of 3rd stroke
                    # Returns empty, if this idx doesnt exist.
                    assert event.find("on_strokeidx_")==0

                    # - which stroke id
                    idx = int(event[13:])

                    # onset of idx stroke (touch)
                    ons, _ = self.strokes_extract_ons_offs(trial)
                    if len(ons)<idx+1:
                        times = []
                    else:
                        times = [ons[idx]]


                elif event=="off_stroke_last":
                    # offset of the last stroke (touch)
                    _, offs = self.strokes_extract_ons_offs(trial)
                    if len(offs)==0:
                        times = []
                    else:
                        times = [offs[-1]]


                elif event in ["done_button", "doneb"]:
                    # Onset of touch of done button, based on touchgin within square
                    # NOTE: This can sometimes (rarely) be very diff (many sec) from the fb assocaited
                    # with done. this is when he holds finger by done button, but doesnt trigger...

                    # The t_pre is long becuase seomtimes he succesfuly touches, but not registered. Should coutn those.
                    # force_must_find_crossings = False # because rarely will touch, but finger not in there (w.g. water)

                    # behcode = 62
                    behcode = "doneb"
                    try:
                        # NOte: tpre very large because seomtimes he touches but isnt correctly registered.
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=2.5, t_post = 1, 
                                                whichstream="touch_done_button", 
                                                ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
                                                assert_single_crossing_this_trial = False,
                                                assert_expected_direction_first_crossing = "up",
                                                take_first_behcode_instance=True,
                                                take_first_crossing_for_each_behcode=True,
                                                refrac_period_between_events=0.05) 
                        times = _extract_times(out)
                    except AssertionError as err:
                        # try expanding the window

                        assert len(self._behcode_extract_times("doneb", trial, shorthand=True))<2, "then doent want to expand window"
                        # NOte: tpre very large because seomtimes he touches but isnt correctly registered.
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=12, t_post = 2, 
                                                whichstream="touch_done_button", 
                                                ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
                                                assert_single_crossing_this_trial = False,
                                                assert_expected_direction_first_crossing = "up",
                                                take_first_behcode_instance=True,
                                                take_first_crossing_for_each_behcode=True,
                                                refrac_period_between_events=0.05) 
                        times = _extract_times(out)

                    times = times[0:1] # take the first time, sometimes can go in and out of squiare

                elif event in ["post_screen_onset", "post"]:
                    # onset of the post-screen, which is offset of the "pause" after you report done
                    # Avoid, since sometimes pd1 fails.. 
                    # (e..g, if doesnt make any touches postfix...)
                        # then try other method. this fails if dont find.

                    # First, try very loose with both
                    # behcode = 73
                    behcode = "post"
                    out1 = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.2, t_post = 0.4, whichstream="pd2", 
                      ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=False,
                        assert_single_crossing_this_trial = False,
                         assert_expected_direction_first_crossing = "down",
                         take_first_crossing_for_each_behcode=True)
                    out2 = self.behcode_get_stream_crossings_in_window(trial, 46, t_pre=0.2, t_post = 0.4, whichstream="pd1", 
                              ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
                                assert_single_crossing_this_trial = False,
                                assert_expected_direction_first_crossing = None, # make this none, since pd1 sometimes really bad.
                                take_first_crossing_for_each_behcode=True)
                    if len(out1)==1 and len(out2)==1:
                        # confirm they are the same
                        times1 = _extract_times(out1)
                        times2 = _extract_times(out2)
                        if len(times1)==0 or len(times2)==0:
                            BAD = True
                        else:
                            if np.abs(times1[0] - times2[0])<0.1:
                                # then good
                                times = times1
                                BAD = False
                            else:
                                BAD = True
                    else:
                        BAD = True

                    if BAD:
                        # Try first, then second
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.2, t_post = 0.35, whichstream="pd2", 
                          ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=False,
                            assert_single_crossing_this_trial = False,
                             assert_expected_direction_first_crossing = "down",
                             take_first_crossing_for_each_behcode=True)
                        if len(out)==0:
                            out = self.behcode_get_stream_crossings_in_window(trial, 46, t_pre=0.2, t_post = 0.35, whichstream="pd1", 
                                  ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                                    assert_single_crossing_this_trial = True,
                                    assert_expected_direction_first_crossing = "up")

                        times = _extract_times(out)


                elif event=="rew":
                    # For reward, it is possible that eventcodes do not match to signals, if press hotkey, too close
                    # in time to other rewards. Chweck that this is the case --> eventcode should be during a positive
                    # value.
                    # Also, is possible in early days before clamped reward down to 0 if it was below the value
                    # that could activete solenoid (like 50ms).
                    # Onsets of reward pulses
                    # behcode = 50
                    behcode = event
                    out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.01, t_post = 0.04, whichstream="reward", 
                    # out = self.behcode_get_stream_crossings_in_window(trial, 50, t_pre=1, t_post = 1, whichstream="reward", 
                                          ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                                            assert_single_crossing_this_trial = False,
                                            assert_expected_direction_first_crossing = None, 
                                            allow_no_crossing_per_behcode_instance_if="positive_val")
                    # use None for assert_expected_direction_first_crossing, because sometimes the preceding
                    # reward can be very close in time, if manually rewarded.
                    times = _extract_times(out)
                elif event=="reward_ons_manual":
                    # do same as reward_ons, with eventcode = 14;
                    assert False, "code it"
                elif event=="reward_all":
                    # Ingore beh codes. just use entire trial and get crossings.
                    out = self.behcode_get_stream_crossings_in_window(trial, None, t_pre=0.01, t_post = 0.04, whichstream="reward", 
                                                                ploton=plot_beh_code_stream, cross_dir_to_take="up", 
                                                                assert_single_crossing_per_behcode_instance=False,
                                                                assert_single_crossing_this_trial = False,
                                                                assert_expected_direction_first_crossing = "up", 
                                                                allow_no_crossing_per_behcode_instance_if=None)                
                    times = _extract_times(out)
                else:
                    print(event)
                    assert False, "This event doesnt exist!!"
                    
                assert times is not None
            return times


        ###############################
        dict_events = {}
        for event in list_events:

            if overwrite:
                # then recompute times.
                RECOMPUTE = True
            else:
                # then try to load from cached
                times, reason = self.events_read_time_from_cached(trial, event)

                if times is None:
                    # This means you have not cached the result (not even attempted. if attempted, then times would be empty list)
                    RECOMPUTE = True
                else:
                    # You cached result and found it., could still be []

                    # If you want to retry, e.g., if previous cached was innacurate.
                    if len(times)==0 and do_reextract:
                        print("Trying to reextract (trial, event):", trial, event)
                        times = self.events_get_time_using_photodiode(trial, 
                            list_events=[event], overwrite=True)[event]
                    RECOMPUTE = False

            # key = (trial, event)
            # if not overwrite and key in self.EventsTimeUsingPhd.keys():
            #     times=  self.EventsTimeUsingPhd[(trial, event)]
            #     if len(times)==0:
            #         # if its empty, try to recompute it.. (e.g., it is a saved version, and now code is updated..)
            #         RECOMPUTE = True
            #     else:
            #         RECOMPUTE = False
            # else:
            #     RECOMPUTE = True

            if RECOMPUTE:
                try:
                    times = compute_times_from_scratch(event) 
                except Exception as err:
                    print(">>>>>>>>>>>>>>>")
                    print("trial, event:", trial, event)
                    self.print_summarize_expt_params()
                    raise err
                self.EventsTimeUsingPhd[(trial, event)] = times

            # Store for output
            dict_events[event] = times

        # if any events didnt get anythjing, try to reextract
        # if do_reextract:
        #     for ev, times in dict_events.items():
        #         if len(times)==0 and RECOMPUTE==False:
        #             print("Trying to reextract (trial, event):", trial, ev)
        #             dict_events[ev] = self.events_get_time_using_photodiode(trial, 
        #                 list_events=[ev], overwrite=True)[ev]

        return dict_events


    def events_get_time_helper(self, event, trial):
        """ Return the time in trial for this event
        PARAMS:
        - event, either string or tuple.
        - eventkind, string, if None, then tries to find it automatically.
        RETURNS:
        - list of numbers, one for each detection of this even in this trial, sorted.
        """

        try:
            # Better version using photodiode or motor
            times = self.events_get_time_using_photodiode(trial, [event])[event] 
        except:
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
                alignto_time = self._behcode_extract_times(code, trial, first_instance_only=True)
            elif isinstance(event, int):
                # Then is behcode
                alignto_time = self._behcode_extract_times(event, trial, first_instance_only=True)
            else:
                assert False
            times = [alignto_time]

        return times

    def events_default_list_events(self, include_stroke_endpoints=True):
        """
        Return canocnialy events that make up a trial, these all have accurate timing
        (e.g., photodiode).
        PARAMS;
        - include_stroke_endpoints, if True, then includes on of first stroke and offset of
        last.
        RETURNS:
        - list_events, list of strings. 
        """
        if include_stroke_endpoints:
            list_events = ["fixcue", "fixtch", "samp", "go", 
                "first_raise", "on_stroke_1", "seqon", 
                "off_stroke_last", "doneb", 
                "post", "reward_all"]
        else:
            list_events = ["fixcue", "fixtch", "samp", "go", 
                "first_raise", "seqon", "doneb", 
                "post", "reward_all"]
        return list_events

    def eventsdataframe_sanity_check(self, DEBUG=False):
        """
        Sanityc check of timing of extracted events (accurate timing). Extract
        all events, then check ones that are weiord (in ordering and/or missing
        events). Save text files that summarize all the categories of event strings,
        and plot rasters for all trials split by these categories. This is useful
        for finding mistakes in timing extraction.
        PARAMS:
        - DEBUG, bool, then runs on subset of data. good since this is slwo.
        """
        from pythonlib.tools.expttools import writeDictToYaml
        import os

        # 1) Extract events dataframe
        dfeventsn, list_events = self.eventsdataframe_extract_timings(DEBUG =DEBUG)

        # 2) Sanity checks of events counts categories

        def log_this(dict_log, key, val):
            """ Log a falure of some sort
            PARAMS;
            - dict_log, holds log data
            - key, dict_log[key] appends val or creates new
            if doesnt exist. (usually the name of the thing to log)
            - val, usually trial num
            """
            if isinstance(key, list):
                key = "".join([str(item) for item in key])
                
            if key in dict_log.keys():
                dict_log[key].append(val)
            else:
                dict_log[key] = [val]
                
        # Initialize all logs
        log_events_incorrect_order = {}
        log_events_missing_unexplained = {}
        log_events_in_chron_order_firstinstance = {}
        log_events_in_chron_order_allinstances = {}

        for ind in range(len(dfeventsn)):
            
            dfthis = dfeventsn.iloc[ind]
            trial = int(dfthis["trial"])
            
            # 1) note if events (first occurance)  do not occur in chron order within a trial
            eventinds_in_chron_order = dfthis["eventinds_ordered_flat"]
            if not np.all(np.diff(eventinds_in_chron_order)>=0):
                log_this(log_events_incorrect_order, eventinds_in_chron_order, trial)
                
            # 2) if missing certain events, confirm thru other params that this was supposed to happen
            for ev in list_events:
                if len(dfthis[ev])==0:
                    # then no times found for this event. check that is ok
                    if ev=="doneb":
                        if dfthis["trial_end_method"]=="pressed_done_button":
                            # then expect done button...
                            log_this(log_events_missing_unexplained, ev, trial)
                    elif ev in ["rew", "reward_all"]:
                        if dfthis["reward"]!=0:
                            # then got reward...
                            log_this(log_events_missing_unexplained, ev, trial)
                    elif ev in ["on_stroke_1", "off_stroke_last"]:
                        strokes = self.strokes_extract(trial, peanuts_only=True)
                        if len(strokes)>0:
                            # then strokes exist. 
                            log_this(log_events_missing_unexplained, ev, trial)
                    elif ev=="seqon":
                        # TODO: check that this is NOT a sequence mask trial.
                        pass
                    elif ev in ["fixcue", "fixtch", "samp", "go", "first_raise", "post"]:
                        # this is a problem, not expected to miss this
                        log_this(log_events_missing_unexplained, ev, trial)
                    else:
                        print(ev)
                        assert False, "which is this?"

            # TODO: if time for done occurs before go, then it must have been an aborrt...
                        
            # 3) Save info of each trial kind
            log_this(log_events_in_chron_order_firstinstance, dfthis["eventinds_in_chron_order"], trial)
            log_this(log_events_in_chron_order_allinstances, dfthis["eventinds_ordered_flat"], trial)
            


        ##### For each category of trial (based on detected events), plot raster across trials`
        savedir = f"{self.Paths['figs_local']}/eventcodes_trial_structure"
        os.makedirs(savedir, exist_ok=True)
        print("SAVING AT: ", savedir)

        list_categories = list(log_events_in_chron_order_allinstances.keys())
        for category in list_categories:
            trials = log_events_in_chron_order_allinstances[category]
            fig, axes, _, _, = self.plotwrapper_raster_multrials_onesite(trials, plot_beh=False, 
                                                    plot_rasters=False, SIZE=0.15, alignto="go")
            
            # save
            fig.savefig(f"{savedir}/events_raster-eventscat_{category}.pdf")
            
        ###### save dicts

        # save reference for what the event indices stand for
        map_index_to_events = {i:ev for i, ev in enumerate(list_events)}
        path = f"{savedir}/map_index_to_events.yaml"
        writeDictToYaml(map_index_to_events, path)

        path = f"{savedir}/log_events_incorrect_order.yaml"
        writeDictToYaml(log_events_incorrect_order, path)

        path = f"{savedir}/log_events_missing_unexplained.yaml"
        writeDictToYaml(log_events_missing_unexplained, path)

        path = f"{savedir}/log_events_in_chron_order_firstinstance.yaml"
        writeDictToYaml(log_events_in_chron_order_firstinstance, path)

        path = f"{savedir}/log_events_in_chron_order_allinstances.yaml"
        writeDictToYaml(log_events_in_chron_order_allinstances, path)

        # return things
        return dfeventsn, list_events, savedir

    def datasetbeh_trial_outcome(self, trial):
        """ Return the outcomes for this trial, from self.DatasetBeh
        """
        ind = self.datasetbeh_trial_to_datidx(trial)
        
        out = {
            "trial_end_method":self.Datasetbeh.Dat.iloc[ind]["trial_end_method"],
            "rew_total":self.Datasetbeh.Dat.iloc[ind]["rew_total"],
            "online_abort":self.Datasetbeh.Dat.iloc[ind]["online_abort"],
            "abort_params":self.Datasetbeh.Dat.iloc[ind]["abort_params"],
            "score_final":self.Datasetbeh.Dat.iloc[ind]["score_final"],
        }

        return out



    def eventsdataframe_extract_timings(self, list_events=None, DEBUG=False):
        """
        Get a dataframe across all trials holding information about key timing of events.
        I used this for sanity checks (e.g., plotting events rasters across trials).
        PARAMS:
        - DEBUG = False # to just take subnset of trials, fastser
        RETURNS:
        - dataframe, where each row is a trial, holding timing of major events.
        """
        import pandas as pd

        DatEvents = [] # list of dicts.
        if DEBUG:
            trials = self.get_trials_list(True)[:20]
        else:
            trials = self.get_trials_list(True)
            
        if list_events is None:
            list_events = self.events_default_list_events()

        for trialthis in trials:
            if trialthis%50==0:
                print(trialthis)
            
            eventsdict, firsttimes_ordered_by_firsttime, alltimes_ordered_by_firsttime, \
                    events_ordered_by_firsttime, eventinds_ordered_by_firsttime, \
                    times_ordered_flat, events_ordered_flat, eventinds_ordered_flat, time_events_flat_first_unsorted = \
                    self.events_get_time_sorted(trialthis, list_events=list_events)

            # a code for n cases for each event
            list_n = [len(eventsdict[ev]) for ev in list_events]
            eventscode = "".join([str(x) for x in list_n])

            eventsdict["list_n"] = list_n
            eventsdict["eventscode"] = eventscode
            
            # Collect times (flattened across all events)
            eventsdict["times_ordered_flat"] = times_ordered_flat
            eventsdict["eventinds_ordered_flat"] = eventinds_ordered_flat
            
            # Collect times (ordered by first case)
            eventsdict["firsttimes_in_chron_order"] = firsttimes_ordered_by_firsttime
            eventsdict["eventinds_in_chron_order"] = eventinds_ordered_by_firsttime
            
            # trial info
            if False:
                fd, tml2 = self.beh_get_fd_trial(trialthis)
                outcome = mkl.getTrialsOutcomesWrapper(fd, tml2)
                eventsdict["trial_end_method"] = outcome["trial_end_method"]
                eventsdict["reward"] = outcome["beh_evaluation"]["rew_total"][0][0]
            else:
                outcome = self.datasetbeh_trial_outcome(trialthis)
                eventsdict["trial_end_method"] = outcome["trial_end_method"]
                eventsdict["reward"] = outcome["rew_total"]

            eventsdict["trial"] = trialthis
            
            eventsdict["time_events_flat_first_unsorted"] = time_events_flat_first_unsorted
            
            DatEvents.append(eventsdict)
        
        dfeventsn = pd.DataFrame(DatEvents)
        return dfeventsn, list_events


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

    def spiketrain_as_elephant_batch(self, save=True):
        """ Generate and save SpikeTrain for all site and trial
        RETURNS:
        - adds "spiketrain" as key in self.DatAll
        """

        ADDED = False # track whether datall is updated.

        if not hasattr(self, "DatAllDf"):
            assert False, "need to first run extract_raw_and_spikes_helper to extract DatAllDf"
        
        if "spiketrain" in self.DatAllDf.columns and not np.any(self.DatAllDf["spiketrain"].isna()):
            # then already gotten. skip
            pass
        else:
            for i, Dat in enumerate(self.DatAll):
                if "spiketrain" not in Dat.keys():
                    if i%500==0:
                        print("spiketrain_as_elephant_batch, datall index: ", i)
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
        if ADDED and save:
            self._savelocal_datall()

    ####################### GENERATE POPANAL for a trial
    def _popanal_generate_from_raw(self, frate_mat, times, chans, df_label_trials=None,
        df_label_cols_get=None):
        """ Low level code to generate from inputed raw fr data
        PARAMS:
        - frate_mat, shape (chans, trials, times)
        - times, shape (times,)
        - df_label_trials, either None (ignroes) or df labeling each trial, 
        whwere the rows match frmat[-, :, -]
        - df_label_cols_get, which cols of df_label_trials to take.
        RETURNS:
        - PopAnal object
        """
        from neuralmonkey.classes.population import PopAnal

        PA = PopAnal(frate_mat, times, chans = chans, print_shape_confirmation=False)

        # Input labels
        if df_label_trials is not None:
            assert df_label_cols_get is not None
            PA.labels_features_input_from_dataframe(df_label_trials, df_label_cols_get, dim="trials")

        return PA


    def _popanal_generate_alldata_bystroke(self, DS, sites, 
        pre_dur, post_dur, fail_if_times_outside_existing,
        use_combined_region, features_to_get_extra=None):
        """ Low level 
        """
        # 1) Get trials and stroke inds
        trials = []
        trialcodes = []
        strokeids = []
        for ind in range(len(DS.Dat)):
        
            if ind%200==0:
                print("index strokes: ", ind)
                
            tc = DS.Dat.iloc[ind]["dataset_trialcode"]
            si = DS.Dat.iloc[ind]["stroke_index"]
            trial_neural = self.datasetbeh_trialcode_to_trial(tc) 

            trials.append(trial_neural)
            trialcodes.append(tc)
            strokeids.append(si)

        # 2) generate PA (Align to each stroke)
        pa = self.smoothedfr_extract_timewindow_bystroke(trials, strokeids, sites, 
            pre_dur=pre_dur, post_dur=post_dur, 
            fail_if_times_outside_existing=fail_if_times_outside_existing) 

        # 3) Assign stroke-level features
        print("Sanity check, extracting trialcode into pa.Xlabel [trials]")
        list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
            'stroke_index', 'stroke_index_fromlast', 'stroke_index_semantic', 
            'shape_oriented', 'ind_taskstroke_orig', 'gridloc',
            'gridloc_x', 'gridloc_y', 'h_v_move_from_prev']

        if features_to_get_extra is not None:
            assert isinstance(features_to_get_extra, list)
            list_cols = list(set(list_cols + features_to_get_extra))
        pa.labels_features_input_from_dataframe(DS.Dat, list_cols, dim="trials")
        assert all([c in pa.Xlabels["trials"].columns for c in list_cols])
            
        # also extract index and call it a new name
        pa.labels_features_input_from_dataframe(DS.Dat, ["index"], dim="trials", overwrite=False)
        pa.Xlabels["trials"]["index_DS"] = pa.Xlabels["trials"]["index"]
        del pa.Xlabels["trials"]["index"]

        # Sanity check, input order matches output order
        assert pa.Xlabels["trials"]["dataset_trialcode"].tolist() == trialcodes
        # Rename it trialcode
        pa.Xlabels["trials"]["trialcode"] = pa.Xlabels["trials"]["dataset_trialcode"]
        pa.Xlabels["trials"] = pa.Xlabels["trials"].drop("dataset_trialcode", 1)
        
        print("Extracting dataset features into pa.Xlabel [chans]")
        regions = [self.sitegetter_map_site_to_region(s, region_combined=use_combined_region) for s in sites]
        pa.labels_input("regions_combined", regions, dim="chans")

        return pa

    def popanal_generate_alldata_bystroke(self, DS, sites, align_to_stroke=True,
        align_to_alternative=[], pre_dur=-0.2, post_dur=0.2, 
        fail_if_times_outside_existing=True,
        use_combined_region=True, features_to_get_extra=None):
        """ Return list of PA, aligned to each thing in align_to_alternative, or aligned
        to each stroke (a single PA)
        PARAMS:
        - DS, DatStrokes instance
        - align_to_stroke, bool, if true, then ignore align_to_alternative
        - align_to_alternative, list of str, dictates len of output
        RETURNS:
        - ListPA, list of PA objets
        """

        assert post_dur - pre_dur > 0.001

        # 2) generate PA
        if align_to_stroke:
            # Return the single pa, aligned to each stroke in
            pa = self._popanal_generate_alldata_bystroke(DS, sites, 
                pre_dur, post_dur, fail_if_times_outside_existing,
                use_combined_region, features_to_get_extra=features_to_get_extra)
            ListPA = [pa]
        else:
            assert False, "this is HACKY. this only uses DS.Dat to collect one datapt for each stroke in DS.Dat. This should either be trial level or stroke level"

            # just use trials, align to specific item in each trial
            # (note: would use this over popanal_generate_alldata because here 
            # saves the stroke-level features)
            assert len(align_to_alternative)>0, "need to pass in list of str, events to align to"

            # Which trials?
            trials = []
            trialcodes = []
            for ind in range(len(DS.Dat)):
                tc = DS.Dat.iloc[ind]["dataset_trialcode"]
                trial_neural = self.datasetbeh_trialcode_to_trial(tc)
                trials.append(trial_neural)
                trialcodes.append(tc)

            # Collect
            ListPA = self.popanal_generate_alldata(trials, sites, align_to_alternative, 
                pre_dur, post_dur)

            # Assign stroke-level features
            print("Extracting dataset features into pa.Xlabel [trials]")
            # list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
            # list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
            #     'stroke_index', 'stroke_index_fromlast', 'shape_oriented', 'ind_taskstroke_orig', 'gridloc',
            #     'gridloc_x', 'gridloc_y', 'h_v_move_from_prev']
            list_cols = []
            if features_to_get_extra is not None:
                assert isinstance(features_to_get_extra, list)
                list_cols = list(set(list_cols + features_to_get_extra))
            for pa in ListPA:
                pa.labels_features_input_from_dataframe(DS.Dat, list_cols, dim="trials")
                # Sanity check, input order matches output order
                # assert pa.Xlabels["trials"]["dataset_trialcode"].tolist() == trialcodes
                assert pa.Xlabels["trials"]["trialcode"].tolist() == trialcodes

                assert all([c in pa.Xlabels["trials"].columns for c in list_cols])

        return ListPA


    def popanal_generate_alldata(self, trials, sites,
        events = ("fixtch", "samp", "go", "first_raise", "on_stroke_1"),
        pre_dur=-0.5, post_dur=0.8, 
        columns_to_input = ("trialcode", "epoch", "character", "supervision_stage_concise"),
        use_combined_region = True):
        """ GOOD wrapper to generate multiple PA objects holding smoothed FR
        across chans and trials, aligned to specific events in trial. Also assigns
        features/labels into the PA, allowing for slicing afterwards

        """

        # pre_dur = -0.8
        # post_dur = 0.8
        # events = ["fixtch", "samp", "go", "first_raise", "on_stroke_1"]
        import pandas as pd
        from pythonlib.tools.pandastools import slice_by_row_label

        # Extract a PA for each event
        print("Generating PA")
        ListPA = []
        assert False, "fix this: now pa outped from smoothedfr_extract_timewindow can be shorter than trials since skips trials that lack this event. use the new trials in downstream code."
        for ev in events:
            print("Extracting pa for: ", ev)
            pa = self.smoothedfr_extract_timewindow(trials, sites, alignto=ev, pre_dur = pre_dur, post_dur = post_dur)
            ListPA.append(pa)
            
        ##### Sanity checks
        # 1) same trials and sites for all pa
        for pa1, pa2 in zip(ListPA[:-1], ListPA[1:]):
            assert pa1.Chans==pa2.Chans
            assert pa1.Trials==pa2.Trials
            
        #### INPUT VARIABLES
        print("Extracting dataset features into pa.Xlabel [trials]")
        # 2) Input desired variables associated with each trial
        trialcodes = [self.datasetbeh_trial_to_trialcode(t) for t in trials]
        dfthis = slice_by_row_label(self.Datasetbeh.Dat, "trialcode", trialcodes)
        assert dfthis["trialcode"].tolist() == trialcodes
        # store each val
        for pa in ListPA:
            pa.labels_features_input_from_dataframe(dfthis, columns_to_input, "trials")
            # for col in columns_to_input:
            #     values = dfthis[col].tolist()
            #     pa.labels_input(col, values, dim="trials")                

        # 3) Input variables associated with each chan
        # - for each chan, map to bregion
        print("Extracting dataset features into pa.Xlabel [chans]")
        regions = [self.sitegetter_map_site_to_region(s, region_combined=use_combined_region) for s in sites]
        for pa in ListPA:
            pa.labels_input("regions_combined", regions, dim="chans")

        return ListPA

    def elephant_spiketrain_to_smoothedfr(self, spike_times, 
        time_on, time_off, 
        gaussian_sigma = SMFR_SIGMA, # changed to 0.025 on 4/3/23. ,
        sampling_period=SMFR_TIMEBIN):
        """
        Convert spiketrain to smoothed fr
        PARAMS;
        - spike_times, array-like of scalar times (seconds).
        - time_on, time_off, scalar times, boundaries of spike times.
        RETURNS:
        - times, (1, tbins)
        - rates, (1, tbins)
        """
        from elephant.kernels import GaussianKernel        
        from elephant.statistics import time_histogram, instantaneous_rate,mean_firing_rate
        from quantities import s
        from neo.core import SpikeTrain

        spiketrain = SpikeTrain(spike_times*s, t_stop=time_off, t_start=time_on)

        frate = instantaneous_rate(spiketrain, sampling_period=sampling_period*s, 
            kernel=GaussianKernel(gaussian_sigma*s))
        
        return frate.times[None, :], frate.T.magnitude

    def popanal_generate_save_trial(self, trial, 
            # gaussian_sigma = 0.1, 
            # gaussian_sigma = 0.025, # changed to 0.025 on 4/3/23. 
            # sampling_period=0.005, # changed to 0.005 from 0.01 on 4/18/23.
            gaussian_sigma = SMFR_SIGMA, # made global on 4/23
            sampling_period = SMFR_TIMEBIN, # made global on 4/23
            print_shape_confirmation=False,
            clean_chans=True, overwrite=False,
            return_sampling_period = False):
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
        from neuralmonkey.classes.population import PopAnal

        if trial not in self.PopAnalDict.keys() or overwrite==True:
            # Get all spike trains for a trial
            list_sites = self.sitegetter_all(clean=clean_chans)
            list_spiketrain = []
            for site in list_sites:
                dat = self.datall_slice_single_bysite(site, trial)
                if "spiketrain" not in dat.keys():
                    print("Generating spike train! (site, trial): ", site, trial)
                    self._spiketrain_as_elephant(site, trial, save=True)
                list_spiketrain.append(dat["spiketrain"])
                
            # Convert spike train to smoothed FR
            frate = instantaneous_rate(list_spiketrain, sampling_period=sampling_period*s, 
                kernel=GaussianKernel(gaussian_sigma*s))

            # Convert to popanal
            PA = PopAnal(frate.T.magnitude, frate.times, chans = list_sites,
                spike_trains = [list_spiketrain], print_shape_confirmation=print_shape_confirmation)
            # PA.Params["frate_sampling_period"] = sampling_period
            self.PopAnalDict[trial] = PA

        # Return
        if return_sampling_period:
            return self.PopAnalDict[trial], sampling_period
        else:
            return self.PopAnalDict[trial]


    ###################### SMOOTHED FR
    def smoothedfr_extract_timewindow_bystroke(self, trials, strokeids, 
        sites, pre_dur=-0.1, post_dur=0.1,
        fail_if_times_outside_existing=True):
        """ Extract smoothed fr dataset for these trials and strokeids
        """
        from quantities import s
        from .population import PopAnal
        
        assert isinstance(pre_dur, (float, int))
        assert isinstance(pre_dur, (float, int))
        assert len(trials)==len(strokeids)

        list_xslices = []
        # 1) extract each trials' PA. Use the slicing tool in PA to extract snippet
        for tr, indstrok in zip(trials, strokeids):
            # extract popanal
            pa, sampling_period = self.popanal_generate_save_trial(tr, return_sampling_period=True)   

            # slice to desired channels
            pa = pa._slice_by_chan(sites) 
  
            # slice to time window
            if False:
                # Then align to onset of stroke that is in DS
                # Sanity check (confirm that timing for neural is same as timing saved in dataset)
                ons, offs = SNthis.strokes_extract_ons_offs(trial_neural)
                timeon_neural = ons[indstrok]
                timeoff_neural = offs[indstrok]    
                timeon = DS.Dat.iloc[ind]["time_onset"]
                timeoff = DS.Dat.iloc[ind]["time_offset"]
                assert np.isclose(timeon, timeon_neural)
                assert np.isclose(timeoff, timeoff_neural)
                time_align = timeon
            else:                
                alignto = f"on_strokeidx_{indstrok}"
                time_align = self.events_get_time_using_photodiode(tr, 
                    list_events=[alignto])[alignto]
                time_align = time_align[0] # take first time in list of times.
            t1 = time_align + pre_dur
            t2 = time_align + post_dur
            pa = pa._slice_by_time_window(t1, t2, return_as_popanal=True,
                fail_if_times_outside_existing=fail_if_times_outside_existing,
                subtract_this_from_times=time_align)
            
            # save this slice
            list_xslices.append(pa)

        # 2) Concatenate all PA into a single PA
        if not fail_if_times_outside_existing:
            assert False, "fix this!! if pre_dur extends before first time, then this is incorrect. Should do what?"
        
        # Replace all times with this time relative to alignement.
        for pa in list_xslices:
            # sampling period, to acocunt for random variation in alignment across snips.
            TIMES = (pa.Times - pa.Times[0]) + pre_dur + sampling_period/2 # times all as [-predur, ..., postdur]
            pa.Times = TIMES

        # get list of np arrays
        if False:
            TIMES = (list_xslices[0].Times - list_xslices[0].Times[0]) + pre_dur*s # times all as [-predur, ..., postdur]
            Xall = np.concatenate([pa.X for pa in list_xslices], axis=1) # concat along trials axis. each one is (nchans, 1, times)
            PAall = PopAnal(Xall, TIMES, sites, trials=trials)
        else:
            from neuralmonkey.classes.population import concatenate_popanals

            # then concat
            PAall = concatenate_popanals(list_xslices, "trials", 
                assert_otherdims_have_same_values=True, 
                assert_otherdims_restrict_to_these=("chans", "times"),
                all_pa_inherit_times_of_pa_at_this_index=0)

        return PAall

    def smoothedfr_extract_timewindow(self, trials, sites, alignto, 
        pre_dur=-0.1, post_dur=0.1,
        fail_if_times_outside_existing=True):
        """ [GOOD] Extract snippet of neural data temporally windows to have same time bins, 
        works even across trials. Time window defined relative to an event marker (alginto).
        TAKES THE FIRST time for alignto, found in the trial.
        PARAMS:
        - trials, list of ints
        - sites, list of ints, gets all combos of trials and sites
        - alignto, str, event marker.
        - pre_dur, post_dur, time in sec relative to alignto. make pre_dur negative if want to
        get time before
        RETURNS:
        - PopAnal object, with PA.X shape (sites, trials, timebins), one per trial.
        - trials_gotten, list of ints, the actual trials in PA, in order.
        NOTE: will only include trials that have this event, so output may be shorter than trials.
        """
        from quantities import s
        from .population import PopAnal
        
        assert isinstance(pre_dur, (float, int))
        assert isinstance(pre_dur, (float, int))
        assert len(trials)>0

        list_xslices = []
        # 1) extract each trials' PA. Use the slicing tool in PA to extract snippet
        trials_gotten = []
        for tr in trials:

            # Skip trial if doesnt have this event
            has_event = self.events_does_trial_include_all_events(tr, [alignto])
            if has_event:

                trials_gotten.append(tr)
                time_align = self.events_get_time_using_photodiode(tr, list_events=[alignto])[alignto]

                # extract popanal
                pa, sampling_period = self.popanal_generate_save_trial(tr, return_sampling_period=True) 

                # slice to desired channels
                pa = pa._slice_by_chan(sites)
     
                # slice to time window
                # if len(time_align)==0:
                #     # Try reextracting, could be updated code solved this.
                #     self.events_get_time_using_photodiode(tr, list_events=[alignto], overwrite=True)
                #     # try eagain                
                #     time_align = self.events_get_time_using_photodiode(tr, list_events=[alignto])[alignto]
                if len(time_align)==0:
                    # now its relaly fucked.
                    # run this to make the stream plot
                    self.events_get_time_using_photodiode(tr, list_events=[alignto], overwrite=True, plot_beh_code_stream=True)
                    print(sites)
                    print(tr)
                    print(alignto)
                    assert False, "didnt find this ewvent..."
                time_align = time_align[0] # take first time in list of times.
                t1 = time_align + pre_dur
                t2 = time_align + post_dur
                pa = pa._slice_by_time_window(t1, t2, return_as_popanal=True,
                    fail_if_times_outside_existing=fail_if_times_outside_existing,
                    subtract_this_from_times=time_align)
                
                # save this slice
                list_xslices.append(pa)

        # 2) Concatenate all PA into a single PA
        if not fail_if_times_outside_existing:
            assert False, "fix this!! if pre_dur extends before first time, then this is incorrect. Should do what?"

        # # Replace all times with this time relative to alignement.
        # for pa in list_xslices:
        #     TIMES = (pa.Times - pa.Times[0]) + pre_dur*s # times all as [-predur, ..., postdur]
        #     pa.Times = TIMES

        # Replace all times with this time relative to alignement.
        for pa in list_xslices:
            # sampling period, to acocunt for random variation in alignment across snips.
            TIMES = (pa.Times - pa.Times[0]) + pre_dur + sampling_period/2 # times all as [-predur, ..., postdur]
            pa.Times = TIMES

        # get list of np arrays
        if False:
            TIMES = (list_xslices[0].Times - list_xslices[0].Times[0]) + pre_dur*s # times all as [-predur, ..., postdur]
            Xall = np.concatenate([pa.X for pa in list_xslices], axis=1) # concat along trials axis. each one is (nchans, 1, times)
            PAall = PopAnal(Xall, TIMES, sites, trials=trials_gotten)
        else:
            from neuralmonkey.classes.population import concatenate_popanals

            # then concat
            PAall = concatenate_popanals(list_xslices, "trials", 
                assert_otherdims_have_same_values=True, 
                assert_otherdims_restrict_to_these=("chans", "times"),
                all_pa_inherit_times_of_pa_at_this_index=0)
 
        return PAall, trials_gotten


    def smoothedfr_extract(self, trials, sites):
        """ Extract smoothed fr, ignoreing trying to clip all trials to same length,
        in a dataframe. First gets the PopAnal represntation, if not already gotten, so
        might take a while first time its run
        PARAMS:
        - trials, list of ints,
        - sites, list of ints
        RETURNS:
        - df, pandas dataframe with site, trial, fr, times, as columns
        """
        import pandas as pd

        out = []
        for t in trials:
            pa = self.popanal_generate_save_trial(t) # pa.X --> (chans, 1, time)
            pathis = pa._slice_by_chan(sites) # slice to these sites
            times = pathis.Times
            assert len(pathis.X.shape)==3
            nchans = pathis.X.shape[0]
            for i in range(nchans):
                x = pathis.X[i,:,:].squeeze()
                assert pathis.Chans[i]==sites[i]
                out.append({
                    "trial":t,
                    "site":pathis.Chans[i],
                    "fr":x,
                    "times":times
                    })

            # if len(pathis.X.squeeze())!=len(pathis.Chans):
            #     print(pathis.X.shape)
            #     print(pathis.X.squeeze().shape)
            #     print(len(pathis.Chans))
            #     assert False, "i am confused"
            # # extract fr
            # for i, (x, chan) in enumerate(zip(pathis.X.squeeze(), pathis.Chans)):
            #     # x.shape, (ntime, )
            #     assert chan==sites[i], "confised"
            #     out.append({
            #         "trial":t,
            #         "site":chan,
            #         "fr":x,
            #         "times":times
            #         })

        return pd.DataFrame(out)



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

    def beh_fixation_success(self, trial, use_stroke_as_proxy=True):
        """Returns True if fixation succes (mkl data) for this trial.
        PARAMS:
        - use_stroke_as_proxy, then returns True if this trial has touch data
        after go cue.
        """
        if use_stroke_as_proxy:
            strokes = self.strokes_extract(trial, peanuts_only=True)
            return len(strokes)>0
        else:
            from ..utils.monkeylogic import getTrialsFixationSuccess
            fd, trialml2 = self.beh_get_fd_trial(trial)
            if trialml2 not in fd["trials"].keys():
                return False
            suc = getTrialsFixationSuccess(fd, trialml2)
            return suc

    def get_trials_list(self, only_if_ml2_fixation_success=False,
        only_if_has_valid_ml2_trial=True, only_if_in_dataset=False, 
        events_that_must_include=None,
        dataset_input = None, nrand=None):
        """
        Get list of ints, trials,
        PARAMS:
        - only_if_ml2_fixation_success, then keeps onl trials where the corresponding
        ml2 beh trial had fixation success. Also skips trials that dont exist in filedata at all.
        - only_if_has_valid_ml2_trial, then skips if the mapper refers to a session or trial outside domain
        (e.g.,negetive). can happen if mapper is incorrect, or missing some beh trials from start of day, etc.
        A legit reason is if ml2 is corrupted..
        - only_if_in_dataset, if True, then only keeps trials that are in self.DatasetBeh
        - events_that_must_include, list of str names of events. only inclues trials that have at least 
        one instance of eaech event. time in trial doesnt matter.
        - dataset_input, dataset to use for pruning, if only_if_in_dataset==True.
        if this None, then uses self.DatasetBeh
        """
        if events_that_must_include is None:
            events_that_must_include = []

        assert not isinstance(only_if_in_dataset, list), "sanity check, becasue I moved order of args..."

        key = (only_if_ml2_fixation_success, only_if_has_valid_ml2_trial)
        if key in self._CachedTrialsList:
            trials = self._CachedTrialsList[key]
        else:
            trials = list(range(len(self.TrialsOffset)))
            
            # 1) only tirals with actual beahvior
            if only_if_ml2_fixation_success:
                trials = [t for t in trials if self.beh_fixation_success(t)]

            # 2) only if there is valid ml2 trial (e..g, excludes if it is a 
            # negative trial, meaning that this neural needs to look at the previous beh data)
            if only_if_has_valid_ml2_trial:
                trials_keep = []
                for t in trials:
                    fdind, trialind = self._beh_get_fdnum_trial(t)
                    if fdind<0:
                        # refering to sess that doesnt exist
                        continue
                    elif trialind<1:
                        # ml2 trials are 1-indexed
                        continue
                    else:
                        # check that this trial exists in beh
                        fd, fdtrial = self.beh_get_fd_trial(t)
                        if fdtrial > fd["params"]["n_trialoutcomes"]:
                            # Then this trial doesnt exist in beh. could be corrupted beh data...
                            continue
                        else:
                            # keep
                            trials_keep.append(t)
                trials = trials_keep

            # Store it.
            self._CachedTrialsList[key] = trials

        if only_if_in_dataset:
            trials_keep = []
            for t in trials:
                if self.datasetbeh_trial_to_datidx(t, dataset_input=dataset_input) is None:
                    # exclud
                    pass
                else:
                    trials_keep.append(t)
            trials = trials_keep

        if len(events_that_must_include)>0:
            trials = self._get_trials_list_if_include_these_events(trials, events_that_must_include)

        if nrand is not None:
            # take randmo subset, ordered.
            if nrand < len(trials):
                import random
                trials = sorted(random.sample(trials, nrand))

        return trials

    def _get_trials_list_if_include_these_events(self, trials, events_that_must_include):
        """ only inclues trials that have at least 
        one instance of eaech event. time in trial doesnt matter.
        """
        trials_keep = []
        for t in trials:
            if self.events_does_trial_include_all_events(t, events_that_must_include):
                trials_keep.append(t)
        return trials_keep  


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

            if waveforms is None:
                print("*****")
                print(rs, chan)
                print(self.print_summarize_expt_params())
                assert False, "need to re-extract waveforms?"

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
            LIST_YLIM = (
                [-300, 150],
                [-200, 100],
                [-400, 200],
                None)):
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

    def _plot_raster_trials_blocked_generate_list_trials(self, trials_pool, dataset_group):
        """
        Using dataset categories, generatelist of trial blocks, each
        with same vale for some category
        """

        D = self.Datasetbeh

        # get the subsets of trials_all for each level
        dict_lev_trials = D.grouping_get_inner_items(dataset_group, sort_keys=True)
        # convert to neural trials
        dict_lev_trials = {lev:[self.datasetbeh_datidx_to_trial(i) for i in tri] for lev, tri in dict_lev_trials.items()}
        # make sure is in acceptable trials
        dict_lev_trials = {lev:[i for i in tri if i in trials_pool] for lev, tri in dict_lev_trials.items()}    

        # Extract into separate items
        list_labels = []
        list_list_trials = []
        for lev, tri in dict_lev_trials.items():
            list_labels.append(lev)
            list_list_trials.append(tri)

        return list_list_trials, list_labels, 

    def plot_raster_spiketimes_blocked(self, ax, list_list_spiketimes, list_labels=None,
                                    list_list_trials = None, list_list_evtimes = None,
                                    overlay_trial_events=True,                                
                                   xmin = None, xmax = None, alpha_raster=0.8):
        """
        Plot rasters, hierarhcially inputted, giving the spiketimes directly.
        Is simialr to plot_raster_trials_blocked, but there pass in trials.
        PARAMS;
        - list_list_spiketimes, list of list of spike times
        - list_labels, list of str labels, matching len(list_list_spiketimes)
        - list_list_trials, list of lsit of ints, matching each datapt in list_list_spiketimes,
        for overlayign events.
        - list_list_evtimes, list of list of scalar times, in the original time in trial,
        that is now defined as 0 for the matching spiektime. for overlayign events.
        """

        if list_labels is not None:
            assert len(list_labels)==len(list_list_spiketimes)
        

        # 1. Concatenate trials from different inner lists (i.e, blocks), keeping track of 
        # thier boundaires.
        list_st_plotting_order = []
        list_index_first_trial_in_block = []
        idx_first_tracker = 0
        for list_st in list_list_spiketimes:
            list_st_plotting_order.extend(list_st)

            list_index_first_trial_in_block.append(idx_first_tracker)
            idx_first_tracker = idx_first_tracker+len(list_st) # update the tracker
        
        # Plot rasters
        self._plot_raster_line_mult(ax, list_st_plotting_order, xmin = xmin, xmax = xmax, 
            alpha_raster=alpha_raster)

        # Overlay trial events
        if overlay_trial_events:
            # Flatten
            list_trials_flat = [t for X in list_list_trials for t in X]
            list_evtimes_flat = [t for X in list_list_evtimes for t in X]
            self.plotmod_overlay_trial_events_mult(ax, list_trials_flat, list_evtimes_flat, 
                xmin=xmin, xmax=xmax) 

        # Plot y markers splitting the blocks
        ymarks = [y-0.5 for y in list_index_first_trial_in_block]
        self.plotmod_overlay_y_events(ax, ymarks, list_labels, True, textcolor="m")
        
        if xmin is not None:
            ax.set_xlim(xmin=xmin)
        if xmax is not None:
            ax.set_xlim(xmax=xmax)


    def plot_raster_trials_blocked(self, ax, list_list_trials, site, list_labels=None,
                                   alignto=None, overlay_trial_events=True,                                
                                   sort_trials_within_blocks=True,
                                   xmin = None, xmax = None, DEBUG=False,
                                   alpha_raster=0.8):
        """
        Plot raster across trials, where trials come in blocks, useful if the trials
        correspond to distinct levelts of some feature (e.g., epoch, shape), and want to 
        plot all on same axis
        PARAMS:
        - list_list_trials, inner lists of list of ints (trials). each list is a "block" that
        will be plotted together (starting from bottom)
        - site, int
        - list_labels, list of str, same length as list_list_trials, for annotating.
        - sort_trials_within_blocks, bool, then trials within blocks will beosrted so that
        increase from bottom to top.
        """

        if list_labels is not None:
            if not len(list_labels)==len(list_list_trials):
                # to avoid confusion
                list_labels = None

        # 1. Concatenate trials from different inner lists (i.e, blocks), keeping track of 
        # thier boundaires.
        list_trials_plotting_order = []
        list_index_first_trial_in_block = []
        idx_first_tracker = 0
        for trials in list_list_trials:
            if sort_trials_within_blocks:
                trials = sorted(trials) # sort the trials within each block
            list_trials_plotting_order.extend(trials)

            list_index_first_trial_in_block.append(idx_first_tracker)
            idx_first_tracker = idx_first_tracker+len(trials) # update the tracker
        
        # Plot rasters
        self.plot_raster_trials(ax, list_trials_plotting_order, site, 
                                overlay_trial_events=overlay_trial_events,
                                alignto=alignto, xmin=xmin, xmax=xmax,
                                alpha_raster=alpha_raster)
            
        # Plot y markers splitting the blocks
        ymarks = [y-0.5 for y in list_index_first_trial_in_block]
        self.plotmod_overlay_y_events(ax, ymarks, list_labels, True, textcolor="m")
        
        if DEBUG:
            print(len(list_trials_plotting_order))
            print("index of first trials in each block: ", list_index_first_trial_in_block)

    def _plot_raster_line_mult(self, ax, list_spiketimes, alignto_time=0., 
        raster_linelengths=0.9, alpha_raster = 0.9, 
        xmin = None, xmax = None, ylabel_trials=None):
        """ Low-level code to plot raster, for these trials, on this axis, 
        inputing spiktimes directly.
        PARAMS:
        - list_spiketimes, list of list of scalar times. will plot them in order, from bottom to top
        """

        for i, spiketimes in enumerate(list_spiketimes):
            self._plot_raster_line(ax, spiketimes, i, alignto_time=alignto_time, 
                linelengths=raster_linelengths, alpha=alpha_raster)
        
        # subsample trials to label
        if ylabel_trials:
            assert len(ylabel_trials)==len(list_spiketimes)
            if len(ylabel_trials)>20:
                n = len(ylabel_trials)
                step = int(np.ceil(n/15))
                inds = range(0, n, step)
                ax.set_yticks(inds)
                ax.set_yticklabels([ylabel_trials[i] for i in inds]);
            else:
                ax.set_yticks(range(len(ylabel_trials)))
                ax.set_yticklabels(ylabel_trials);
        ax.set_ylabel('trial');

        if xmin is not None:
            ax.set_xlim(xmin=xmin)
        if xmax is not None:
            ax.set_xlim(xmax=xmax)


    def plot_raster_trials(self, ax, list_trials, site, alignto=None,
        raster_linelengths=0.9, alpha_raster = 0.9, overlay_trial_events=True,
        ylabel_trials=True, plot_rasters=True, xmin = None, xmax = None,
        overlay_strokes=True):
        """ Plot raster, for these trials, on this axis.
        PARAMS:
        - list_trials, list of indices into self. will plot them in order, from bottom to top
        """

        if plot_rasters:
            list_align_time = []
            for i, trial in enumerate(list_trials):
                
                # get time of this event (first instance)
                if alignto:
                    timesthis = self.events_get_time_helper(alignto, trial) 
                    # if long, then take the first one
                    assert len(timesthis)>0
                    alignto_time = timesthis[0]
                else:
                    alignto_time = None

                list_align_time.append(alignto_time)
                
                # Rasters
                # rs, chan = self.convert_site_to_rschan(site)
                D = self.datall_slice_single_bysite(site, trial)
                # D = self.datall_slice_single(rs, chan, trial0=trial)
                spikes = D["spike_times"]
                self._plot_raster_line(ax, spikes, i, alignto_time=alignto_time, 
                    linelengths=raster_linelengths, alpha=alpha_raster)
            

            self.plotmod_overlay_trial_events_mult(ax, list_trials, list_align_time,
                ylabel_trials, xmin, xmax)
        
            if site is not None:
                ax.set_title(self.sitegetter_summarytext(site)) 


    def _plot_raster_create_figure_blank(self, duration, n_raster_lines, n_subplot_rows=1,
            nsubplot_cols=1, reduce_height_for_sm_fr=False, sharex=True):
        """ Helper to genreate figure with correct size, based on duration and num rows
        RETURNS:
        - fig,
        - axes,
        - kwargs, to pass into self.plot_raster_trials(..., **kwargs)
        """

        # assert n_raster_lines<1500
        # assert n_subplot_rows<10
        # assert nsubplot_cols<10

        if nsubplot_cols<1:
            nsubplot_cols = 1
        if n_subplot_rows<1:
            n_subplot_rows = 1
        
        aspect = 0.8 * (duration/4) # empriically, 0.8 is good for a window of 4sec
        if aspect<0.6:
            aspect = 0.6
        if aspect>1.5:
            aspect = 1.5

        height_cell = n_raster_lines * 0.025
        if reduce_height_for_sm_fr:
            # make it wider
            height_cell = 0.9*height_cell
            aspect = 1.1 * aspect

        if height_cell < 3.5:
            height_cell = 3.5
        if height_cell > 10:
            height_cell = 10

        width_cell = aspect * height_cell

        height = n_subplot_rows * height_cell
        width = nsubplot_cols * width_cell

        if np.isnan(height):
            height = 1
        if np.isnan(width):
            width = 1

        if False:
            print(n_raster_lines, height_cell, n_subplot_rows)
            print(duration, width_cell)
            print(width, height, aspect)
            # assert False
        fig, axes = plt.subplots(n_subplot_rows, nsubplot_cols, sharex=sharex, 
            figsize = (width, height), squeeze=False)

        kwargs = {
            "alpha_raster":0.7
        }
        return fig, axes, kwargs


    def _plot_raster_line(self, ax, times, yval, color='k', alignto_time=None,
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
        
    def strokes_extract_ons_offs(self, trialtdt, ver="new"):
        """ Return strolkes during drawing period (peanuts)
        RETURNS:
        - ons, offs, lists holding scalar times. empty lists if nostrokes.
        NOTE: confirmed for a single session that new == old.
        """
        if ver=="new":
            # new version, allows using cached strokes, without need for drawmonkey
            strokes = self.strokes_extract(trialtdt, peanuts_only=True)
            ons = [s[0,2] for s in strokes]
            offs = [s[-1,2] for s in strokes]
        elif ver=="old":
            # old version.
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

    def plotmod_overlay_y_events(self, ax, yvals, labels=None, span_xlim=True, 
            color="m", textcolor="m"):
        """ Horizontal Marks on Y axis, flexible, to demarkate thnings like important trials, etc.
        PARAMS;
        - yvals, list of values, will place marker at each one
        - labels, liist of string, to place next to each marker.
        - span_xlim, if True, then line across plot. if False, then a marker onb the y axis.
        """

        XLIM = ax.get_xlim()

        if labels is not None:
            assert len(yvals)==len(labels)
        else:
            labels = [None for _ in range(len(yvals))]

        for val, lab in zip(yvals, labels):
            if span_xlim:
                ax.axhline(val, color=color)
            else:
                ax.plot(XLIM[0], val, ">", color=color)
            if lab is not None:
                ax.text(XLIM[0], val, f"{lab}", color=textcolor)



    def plotmod_overlay_beh_codes(self, ax, trial0, list_codes):
        """ Overlay all times of the given beh codes. This is a bare function, just 
        for behcodes. Plots all instances, not just the first.
        PARAMS:
        - list_codes, list of ints
        """


        for code in list_codes:
            times_codes = self._behcode_extract_times(code, trial0)
            name = self.behcode_convert(code, shorthand=True)
            # names_codes = [self.behcode_convert(code, shorthand=True) for _ in range(len(times_codes))] 

            for time in times_codes:
                ax.axvline(time, color="k", ls="--", alpha=0.7)
                YLIM = ax.get_ylim()
                y_text = YLIM[0]
                ax.text(time, y_text, name, rotation="vertical", fontsize=14)

    def plotmod_overlay_trial_events_mult(self, ax, list_trials, list_align_time,
        ylabel_trials=None, list_yvals=None, xmin=None, xmax =None, overlay_strokes=True,
        clear_old_yticks = True):
        """
        Flexible helper to plot events for specified trials, at specific alignemnet times.
        PARAMS:
        - list_trials, neural trials
        - list_align_time, list of times, one for each trial, will recenter that trial so that
        its time in list_align_time is plotted at 0.
        """

        if list_yvals is None:
            # start from bottom
            list_yvals = list(range(len(list_trials)))

        assert len(list_align_time)==len(list_trials)

        for i, (yval, trial, alignto_time) in enumerate(zip(list_yvals, list_trials, list_align_time)):

            # - overlay beh things
            ALPHA_MARKERS = 1-np.clip(len(list_trials)/ 90, 0.63, 0.82)
            # ALPHA_MARKERS = 0.05
            # Auto determine alpha for markers, based on num trials

        #     SN.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, only_on_edge="top")
            if i==0:
                include_text = True
            else:
                include_text = False
            self.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, only_on_edge="bottom", 
                                            YLIM=[yval-0.3, yval+0.5], which_events=["key_events_correct"], 
                                            include_text=include_text, text_yshift = -0.5, alpha=ALPHA_MARKERS,
                                            xmin = xmin, xmax =xmax
                                            )
            # overlay_strokes = False
            if overlay_strokes:
                ALPHA_STROKES = 0.8*ALPHA_MARKERS
                self.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
                                                YLIM=[yval-0.4, yval-0.3], which_events=["strokes"], alpha=ALPHA_STROKES,
                                                xmin = xmin, xmax =xmax)
        
        # subsample trials to label
        if ylabel_trials is None:
            ylabel_trials = list_trials
        elif ylabel_trials==True:
            ylabel_trials = list_trials
        else:
            assert len(ylabel_trials)==len(list_trials)

        # if len(ylabel_trials)>20:
        #     n = len(ylabel_trials)
        #     step = int(np.ceil(n/15))
        #     inds = range(0, n, step)
        #     ax.set_yticks(inds)
        #     ax.set_yticklabels([ylabel_trials[i] for i in inds]);
        # else:
        #     ax.set_yticks(range(len(ylabel_trials)))
        #     ax.set_yticklabels(ylabel_trials);
        # ax.set_ylabel('trial');
        if clear_old_yticks:
            ticks_current = []
            labels_current =[]
        else:
            ticks_current = list(ax.get_yticks())
            labels_current = list(ax.get_yticklabels())

        ax.set_yticks(ticks_current+list_yvals, labels=labels_current+ylabel_trials, 
            fontsize=5)
        # # ax.set_yticklabels(ylabel_trials);
        # ax.set_ylabel('trial');

        # ax.set_xbound(xmin, xmax)
        if xmin is not None:
            ax.set_xlim(xmin=xmin)
        if xmax is not None:
            ax.set_xlim(xmax=xmax)
        # plt.axis('scaled')



    def plotmod_overlay_trial_events(self, ax, trial0, strokes_patches=True, 
            alignto_time=None, only_on_edge=None, YLIM=None, alpha = 0.15,
            which_events=("key_events_correct", "strokes"), include_text=True, 
            text_yshift = 0., xmin=None, xmax=None):
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

        # Strokes should be lower alpha, so not obscure spikes.
        alpha_st = alpha
        if alpha_st>0.2:
            alpha_st = 0.2 

        for ev in which_events:
            assert ev in ["behcodes", "key_events_correct", "strokes"], "doesnt exist..."
        ###### 1) behcodes, old version whcih used specific sequenc eof codes. this is not perfectly accurate.
        # so is replaced by key_events_correct
        if "behcodes" in which_events:
            # Vertical lines for beh codes (only the first instance)
            list_codes = [9, 11, 16, 91, 92, 62, 73, 50]
            times_codes = self.behcode_extract_times_sequence(trial0, list_codes) # np array
            # names_codes = [beh_codes[c] for c in list_codes]
            names_codes = [self.behcode_convert(c, shorthand=True) for c in list_codes] 
            # names_codes = ["on", "fixcue", "fixtch", "samp", "go", "doneb", "fb_vs", "rew"]
            colors_codes = ["c",  "m",  "b",    "r",   "y",  "g",     "m",    "c"]
        else:
            # list_codes = []
            times_codes = np.array([])
            names_codes = [] 
            colors_codes = []

        ###### 2) key events, determeined using actual voltage clock signals or touch, etc.
        if "key_events_correct" in which_events:
            # key events in trial, with correct timing based on timing signals (e..g, photodiode)
            
            # events = ["fixcue", "fixtch", "samp", "go", "first_raise", "seqon", "doneb", "post", "rew"]
            # colors_codes = ["k",  "m",       "r",    "b",  "c",  "y",  "g",     "m",     "k"]
            events = self.events_default_list_events(include_stroke_endpoints=False)
            # events = ["fixcue", "fixtch", "samp", "go", "first_raise", 
            #     "seqon", "doneb", "post", "rew"]
            color_map = {
                "fixcue":"g",
                "fixtch":"m",
                "samp":"r",
                "go":"b",
                "first_raise":"c",
                "seqon":"y",
                "doneb":"g",
                "post":"m", 
                "rew":"k",
                "reward_all":"b"
            }
            # colors_codes = ["k",  "m",       "r",    "b",  "c",  "y",  "g",     "m",     "k"]
            dict_events = self.events_get_time_using_photodiode(trial0, events)
            
            # collect all event times into single arrays
            for ev in dict_events:
                for t in dict_events[ev]:
                    times_codes = np.append(times_codes, t)
                    names_codes.append(ev)
                    colors_codes.append(color_map[ev])
            


        ###### 3) Also include times of strokes
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

        ############## Plot marker for each event
        for time, name, col in zip(times_codes, names_codes, colors_codes):
            if np.isnan(time):
                continue
            if xmin is not None and time<xmin:
                continue
            if xmax is not None and  time>xmax:
                continue
            if only_on_edge:
                if only_on_edge=="top":
                    ax.plot(time, YLIM[1], "v", color=col, alpha=alpha)
                    y_text = YLIM[1]
                elif only_on_edge=="bottom":
                    ax.plot(time, YLIM[0], "^", color=col, alpha=alpha)
                    y_text = YLIM[0]
                else:
                    assert False
            else:
                ax.axvline(time, color=col, ls="--", alpha=0.7)
                y_text = YLIM[0]
            if include_text:
                y_text = y_text + text_yshift
                ax.text(time, y_text, name, rotation="vertical", fontsize=10, color="m", alpha=0.5)

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
                        ax.hlines(YLIM[1], on, of, color="r", alpha=alpha)
                    elif only_on_edge=="bottom":
                        ax.hlines(YLIM[0], on, of, color="r", alpha=alpha)
                    else:
                        assert False
                        # ax., YLIM[1], "v", color=col)
                else:
                    rect = Rectangle((on, YLIM[0]), of-on, YLIM[1]-YLIM[0], 
                        linewidth=1, edgecolor='r',facecolor='r', alpha=alpha_st)
                    ax.add_patch(rect)

    def plot_trial_timecourse_summary(self, ax, trial0, number_strokes=True,
        overlay_trial_events=True):
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

        # Also plot binary whether is touching
        times, touching = self.beh_extract_touching_binary(trial0)
        ax.plot(times, 100*touching, 'x-k', label="touching")
        
        if overlay_trial_events:
            self.plotmod_overlay_trial_events(ax, trial0)

    def plot_final_drawing(self, ax, trialtdt, strokes_only=False):
        """ plot the drawing
        PARAMS:
        - strokes_only, then just pnuts
        """
        strokes = self.strokes_extract(trialtdt, peanuts_only=strokes_only)
        plotDatStrokes(strokes, ax, clean_ordered_ordinal=True, number_from_zero=True)

    def beh_extract_touch_in_fixation_square(self, trial, window_delta_pixels = 38.5,
        ploton=False):
        """ Return binary wherther is touching fixation
        Evaluates if is touching within fixation square
        PARAMS:
        - window_delta_pixels, scalar, the box is sides 2*window_delta_pixels, 
        if finger in this then is touching fix. 30. is emprically good.
        RETURNS:
        - times,
        - touch, 1 where is touchign in square, 0 outside.
        """

        # 1) is touching within fix params?
        fd, t = self.beh_get_fd_trial(trial)
        fixcue_params = mkl.getTrialsFix(fd, t)
        x = fixcue_params["fixpos_pixels"][0]
        y = fixcue_params["fixpos_pixels"][1]
        window_x = [x-window_delta_pixels, x+window_delta_pixels]
        window_y = [y-window_delta_pixels, y+window_delta_pixels]

        # get times that touch is close to fixation button
        xyt = self.beh_extract_touch_data(trial)
        times = xyt[:,2]
        x = xyt[:,0]
        y = xyt[:,1]
        touchingfix = (x>=window_x[0]) & (x<=window_x[1]) & (y>=window_y[0]) & (y<=window_y[1])

        if ploton:
            # overlay with times of touch (any location)
            times2, touching2 = self.beh_extract_touching_binary(trial) 
            plt.figure()
            plt.plot(times, touchingfix*200, "-k", label="in fixation square")
            plt.plot(times2, touching2*200, ":m", label="touching screen")
            plt.plot(times, x, '--b', label="x")
            plt.plot(times, y, '--r', label="y")
            plt.axhline(window_x[0], color="b", label="xmin")
            plt.axhline(window_x[1], color="b", label="xmax")
            plt.axhline(window_y[0], color="r", label="ymin")
            plt.axhline(window_y[1], color="r", label="ymax")
            plt.legend()

        return times, touchingfix.astype(int)

    def beh_extract_touch_in_done_button(self, trial, window_delta_pixels = 40.,
        ploton=False):
        """ Return binary wherther finger is in done button, based solely on 
        location of touch
        PARAMS:
        - window_delta_pixels, scalar, the box is sides 2*window_delta_pixels, 
        if finger in this then is touching fix. 30. is emprically good.
        RETURNS:
        - times,
        - touch, nparray, 1 where is touchign in square, 0 outside.
        (if not using done button, still returns this, just all 0)
        """

        fd, t = self.beh_get_fd_trial(trial)
        # get times that touch is close to fixation button
        xyt = self.beh_extract_touch_data(trial)
        times = xyt[:,2]
        xtouch = xyt[:,0]
        ytouch = xyt[:,1]

        if mkl.getTrialsDoneButtonMethod(fd, t)=="skip":
            # Then no done button, return empty
            return times, np.zeros((len(times),)).astype(int)     

        # 1) is touching within fix params?
        # --- get window
        donebutton_pos = mkl.getTrialsDoneButtonPos(fd, t)
        x = donebutton_pos[0]
        y = donebutton_pos[1]
        window_x = [x-window_delta_pixels, x+window_delta_pixels]
        window_y = [y-window_delta_pixels, y+window_delta_pixels]

        # --- is touch in window?
        touchingdone = (xtouch>=window_x[0]) & (xtouch<=window_x[1]) & (ytouch>=window_y[0]) & (ytouch<=window_y[1])

        if ploton:
            # overlay with times of touch (any location)
            times2, touching2 = self.beh_extract_touching_binary(trial)
            plt.figure()
            plt.plot(times, touchingdone*200, "-k", label="in done square")
            plt.plot(times2, touching2*200, ":m", label="touching screen")
            plt.plot(times, xtouch, '--b', label="x")
            plt.plot(times, ytouch, '--r', label="y")
            plt.axhline(window_x[0], color="b", label="xmin")
            plt.axhline(window_x[1], color="b", label="xmax")
            plt.axhline(window_y[0], color="r", label="ymin")
            plt.axhline(window_y[1], color="r", label="ymax")
            plt.legend()

        return times, touchingdone.astype(int)



    def beh_extract_touch_data(self, trial):
        """ Extract touch data, raw.
        RETURNS:
        - times, array of times
        - x, array of x coords (nan if not touchgin)
        - y, array of y coorrds
        """
        if trial in self._CachedTouchData.keys():
            return self._CachedTouchData[trial]
        else:   
            fd, trialml = self.beh_get_fd_trial(trial)
            xyt = mkl.getTrialsTouchData(fd, trialml)
            # times, touching = mkl.getTrialsTouchingBinary(fd, trialml)
            return xyt
            # return xyt[:,2], xyt[:,0], xyt[:,1]


    def beh_extract_touching_binary(self, trialtdt):
        """ Return time series of whether is touching or not.
        RETURNS:
        times, array of time bins
        touching, array binary, whether is touching.
        """
        import numpy as np
        xyt = self.beh_extract_touch_data(trialtdt)
        x = xyt[:,0]
        touching = 1-np.isnan(x).astype(int)
        times = xyt[:,2]
        # fd, trialml = self.beh_get_fd_trial(trialtdt)
        # # xyt = mkl.getTrialsTouchData(fd, trialml)
        # times, touching = mkl.getTrialsTouchingBinary(fd, trialml)
        return times, touching


    def strokes_task_extract(self, trial):
        """ Extract the strokes for this task
        """
        if trial in self._CachedStrokesTask.keys():
            return self._CachedStrokesTask[trial]
        else:
            from ..utils.monkeylogic import getTrialsTaskAsStrokes
            fd, trialml = self.beh_get_fd_trial(trial)
            strokestask = getTrialsTaskAsStrokes(fd, trialml)
            return strokestask
            # plotDatStrokes(strokestask, ax, clean_task=True)

    def strokes_extract(self, trialtdt, peanuts_only=False):
        from ..utils.monkeylogic import getTrialsStrokes, getTrialsStrokesByPeanuts
        if peanuts_only:
            if trialtdt in self._CachedStrokesPeanutsOnly.keys():
                strokes = self._CachedStrokesPeanutsOnly[trialtdt]
            else:
                fd, trialml = self.beh_get_fd_trial(trialtdt)
                strokes = getTrialsStrokesByPeanuts(fd, trialml)
        else:
            if trialtdt in self._CachedStrokes.keys():
                strokes = self._CachedStrokes[trialtdt]
            else:
                fd, trialml = self.beh_get_fd_trial(trialtdt)
                strokes = getTrialsStrokes(fd, trialml)
        return strokes


    def plot_taskimage(self, ax, trialtdt):
    #     plotTrialSimple(fd, trialml, ax=ax, plot_task_stimulus=True, nakedplot=True, plot_drawing_behavior=False)
        strokestask = self.strokes_task_extract(trialtdt)
        # from ..utils.monkeylogic import getTrialsTaskAsStrokes
        # fd, trialml = self.beh_get_fd_trial(trialtdt)
        # strokestask = getTrialsTaskAsStrokes(fd, trialml)
        plotDatStrokes(strokestask, ax, clean_task=True)
        
    ###################### PLOTS (specific)
    def plot_rasters_all(self, ax, trial, list_sites=None, site_to_highlight=None,
        overlay_trial_events=True):
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
            self._plot_raster_line(ax, st, yval=i, color=pcol, linelengths=1, alpha=0.5)

            # collect for ylabel
            rs, chan = self.convert_site_to_rschan(site)
            list_ylabel.append(f"{site}|{rs}-{chan}")
        ax.set_yticks(range(len(list_ylabel)))
        ax.set_yticklabels(list_ylabel);
        ax.set_xlabel('time rel. trial onset (sec)');
        ax.set_ylabel('site');
        ax.set_title(f"trial: {trial}| nsites: {len(list_sites)}")
        if overlay_trial_events:
            self.plotmod_overlay_trial_events(ax, trial)
        XLIM = ax.get_xlim()
        # - Overlay brain regions
        self.plotmod_overlay_brainregions(ax, list_sites)
        
    def plot_epocs(self, ax, trial, list_epocs=("camframe", "camtrialon", "camtrialoff", 
        "rewon", "rewoff", "behcode"), overlay_trial_events=True, 
        overlay_trial_events_notpd=False):
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
        if overlay_trial_events:
            self.plotmod_overlay_trial_events(ax, trial)
        
        if overlay_trial_events_notpd:
            # Then plot the behcode times (not the actual photodiode)
            which_events = ["behcodes"]
            self.plotmod_overlay_trial_events(ax, trial, which_events=which_events)

    def plot_stream(self, ax=None, trial=0, which="pd1"):
        """ Plot this trial and stream on ax.
        """
        if ax is None:
            fig, ax = plt.subplots(1,1)
        out = self.extract_data_tank_streams(which, trial0=trial)
        if out is not None:
            times, vals, fs = out
            ax.plot(times, vals, '-', label=which)


    def plotwrapper_raster_multrials_onesite(self, list_trials=None, site=None, alignto=None, 
            SIZE=0.5, SIZE_HEIGHT_TOTAL = None, SIZE_WIDTH_TOTAL = 22., 
            plot_beh=True, plot_rasters=True, 
            alpha_raster = 0.9,
            xmin = None, xmax = None,
            raster_linelengths=0.9,
            overlay_trial_events=True,
            ylabel_trials=True, overlay_strokes=True, 
            nrand_trials = 20):
        """ Plot one site, mult trials, overlaying for each trial its major events
        PARAMS:
        - list_trials, list of int. if None, then plots 20 random
        - site, int (512). only allowed to be None if plot_rasters is None
        - alignto, str or None, how to adjust times to realign.
        - SIZE, height of each row. 
        - SIZE_HEIGHT_TOTAL, totla of all rows. if not None, then overwrites SIZE.
        - xmax, will limit the plot to this max value.
        """ 

        if site is None:
            assert plot_rasters==False

        if list_trials is None:
            list_trials = self.get_trials_list(True)
            if len(list_trials)>nrand_trials:
                import random
                list_trials = sorted(random.sample(list_trials, nrand_trials))
        nrows = len(list_trials)
        ncols = 1

        if SIZE_HEIGHT_TOTAL is not None:
            SIZE = SIZE_HEIGHT_TOTAL/nrows

        # old, when including drawing.
        # fig, axes = plt.subplots(1, ncols, sharex=True, figsize=(SIZE_WIDTH_TOTAL, SIZE*nrows), 
        #                        gridspec_kw={'width_ratios': [9,1]})
        fig, axes = plt.subplots(1, ncols, sharex=True, squeeze=False, figsize=(SIZE_WIDTH_TOTAL, SIZE*nrows))
        ax = axes.flatten()[0]

        self.plot_raster_trials(ax, list_trials, site, alignto,
            raster_linelengths, alpha_raster, overlay_trial_events, 
            ylabel_trials, plot_rasters, xmin, xmax, overlay_strokes=overlay_strokes)

        # Final drawing
            #     ax = axes.flatten()[2*i + 1]
            #     SN.plot_final_drawing(ax, trial)
            # Plot each drawing
        if plot_beh:
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
        else:
            fig_draw = None
            axes_draw = None
                

        return fig, axes, fig_draw, axes_draw


    def plotwrapper_raster_oneetrial_multsites(self, trialtdt, 
            list_sites=None, site_to_highlight=None,
            WIDTH=20, HEIGHT = 10, overlay_trial_events=True,
            overlay_trial_events_notpd=False):
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
        self.plot_epocs(ax, trialtdt, overlay_trial_events=overlay_trial_events,
            overlay_trial_events_notpd=overlay_trial_events_notpd)
        # XLIM = ax.get_xlim()

        # Streams
        ax = axes.flatten()[1]
        for stream in ["pd1", "pd2"]:
            self.plot_stream(ax, trialtdt, stream)
        ax.set_title("photodiodes")
        ax.legend()
        if overlay_trial_events:
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
                if overlay_trial_events:
                    self.plotmod_overlay_trial_events(ax, trialtdt)
        
        # Beh strokes (ml2)
        ax = axes.flatten()[6]
        ax.set_title("beh strokes")
        self.plot_trial_timecourse_summary(ax, trialtdt, overlay_trial_events=overlay_trial_events)
        # ax.set_xlim(XLIM)

        # -- Rasters
        ax = axes.flatten()[7]
        self.plot_rasters_all(ax, trialtdt, list_sites, overlay_trial_events=overlay_trial_events)
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

    def plotwrapper_raster_onesite_eventsubplots(self, site, trials=None, 
            ntrials = 200, list_alignto=None, sdir=None, xmin=-2, xmax=2):
        """ A single plot of rasters, aligned to each event in list_alignto, each a 
        seaprate subplots. 
        PARAMS;
        - site, int
        - trials, eitehr lsit of int, or None, in which case will sample random trials 
        (ntrials)
        - ntrials, int, how many random trials to get, if trials is None
        - list_alignto, list of str, events. or None, to get default events
        NOTE:
        - each trial will be enforced to have each event.
        """


        if list_alignto is None:
            # list_alignto = ["fixtch", "samp", "go", 
            #     "on_stroke_1", "off_stroke_last", "doneb", "reward_all"]
            list_alignto = ["fixtch", "samp", "go", 
                "on_stroke_1", "off_stroke_last", "reward_all"]

        # only keep trials with all events
        if trials is None:
            # get random trials.
            trials = self.get_trials_list(True, True, True, list_alignto, nrand=ntrials)
        else:
            trials = self._get_trials_list_if_include_these_events(trials, list_alignto)

        assert len(trials)>0

        # FIgure size params
        dur = xmax-xmin
        ncols = 6
        nrows = int(np.ceil(len(list_alignto)/ncols))

        ntrials = len(trials)
        fig, axes, kwargs = self._plot_raster_create_figure_blank(dur, ntrials, nrows, ncols)

        # make plot
        for i, (alignto, ax) in enumerate(zip(list_alignto, axes.flatten())):
        #     ax = axes.flatten()[0]
            self.plot_raster_trials(ax, trials, site, alignto, xmin = xmin, xmax=xmax, **kwargs);
            ax.set_title(alignto)
            if i==0:
                ax.set_ylabel(self.sitegetter_summarytext(site))

        if sdir:
            # fig.savefig(f"{sdir}/rastersevents-{self.sitegetter_summarytext(site)}.png", anti_alias=True, dpi=300) # aa and dpi dont help. 300 is default
            fig.savefig(f"{sdir}/rastersevents-{self.sitegetter_summarytext(site)}.png")
            
        return fig


    def plotwrapper_raster_onesite_eventsubplots_blocked(self, site, dataset_group,
            trials = None, ntrials = 200, list_alignto=None, sdir=None, xmin=-2, xmax=2):
        """ 
        """


        if list_alignto is None:
            # list_alignto = ["fixtch", "samp", "go", 
            #     "on_stroke_1", "off_stroke_last", "doneb", "reward_all"]
            list_alignto = ["fixtch", "samp", "go", 
                "on_stroke_1", "off_stroke_last", "reward_all"]

        # only keep trials with all events
        if trials is None:
            # get random trials.
            trials = self.get_trials_list(True, True, True, list_alignto, nrand=ntrials)
        else:
            trials = self._get_trials_list_if_include_these_events(trials, list_alignto)
        list_list_trials, list_labels = self._plot_raster_trials_blocked_generate_list_trials(trials, dataset_group)

        # FIgure size params
        dur = xmax-xmin
        ncols = 6
        nrows = int(np.ceil(len(list_alignto)/ncols))

        ntrials = len(trials)
        fig, axes, kwargs = self._plot_raster_create_figure_blank(dur, ntrials, nrows, ncols)

        # make plot
        for i, (alignto, ax) in enumerate(zip(list_alignto, axes.flatten())):
        #     ax = axes.flatten()[0]
            self.plot_raster_trials_blocked(ax, list_list_trials, site, list_labels, 
                alignto, xmin = xmin, xmax=xmax, **kwargs);
            ax.set_title(alignto)
            if i==0:
                ax.set_ylabel(self.sitegetter_summarytext(site))

        if sdir:
            # fig.savefig(f"{sdir}/rastersevents-{self.sitegetter_summarytext(site)}.png", anti_alias=True, dpi=300) # aa and dpi dont help. 300 is default
            fig.savefig(f"{sdir}/rastersevents-{self.sitegetter_summarytext(site)}.png")
            
        return fig


    def plotwrapper_smoothed_multtrials_multsites_timewindow(self, sites, trials, 
            alignto="go", pre_dur=-0.5, post_dur=2, ax=None,
            plot_indiv=True, plot_summary=False, error_ver="sem",
            pcol_both = None, pcol_indiv = "k", pcol_summary="r",
            xmin=None, xmax=None, summary_method="median"):
        """ Plot smoothed FR, across sites and trials, aligned to event 
        First extracts this data, then plots
        """

        assert pre_dur < post_dur
        # 1) Extract the data 
        pa = self.smoothedfr_extract_timewindow(trials, sites, alignto, pre_dur, post_dur)

        # 2) plot
        if pcol_both is not None:
            pcol_indiv = pcol_both
            pcol_summary = pcol_both
        fig1, ax1, fig2, ax2 = pa.plotwrapper_smoothed_fr(ax=ax, plot_indiv=plot_indiv, plot_summary=plot_summary, 
            error_ver=error_ver, pcol_indiv=pcol_indiv, pcol_summary=pcol_summary, summary_method=summary_method)

        if xmin is not None:
            for axthis in [ax1, ax2]:
                if axthis is not None:
                    axthis.set_xlim(left=xmin)
        if xmax is not None:
            for axthis in [ax1, ax2]:
                if axthis is not None:
                    axthis.set_xlim(right=xmax)

        return fig1, ax1, fig2, ax2


    def plotwrapper_smoothed_multtrials_multsites(self, sites, trials, ax, YLIM=None):
        """ Helper for various ways of plotting smoothed fr, a single plot. 
        Could call this multiple times with different YLIMS to stack multiple traces on single
        plot
        PARAMS;
        - sites, list of ints
        - trials, list of ints
        - YLIM, [lower, higher] or None, if former, then rescales so that all fr are fit within
        these y coordinates. all fr traces will retain their relative scales to each other. Useful
        if want to stack traces on same plot.
        """

        assert isinstance(sites, list)
        assert isinstance(trials, list)

        # Get the fr for ths site across trials
        df = self.smoothedfr_extract(trials, sites)

        # if YLIM is not None:
        #     frmax = 

        # OPTION 1: each fr trace different length
        # plot each fr trace on the axis
        for i in range(len(df)):
            times = df.iloc[i]["times"]
            fr = df.iloc[i]["fr"]

            ax.plot(times, fr)

        # OPTION 2: each fr trace is same length, they are in PopAnal object


    ################################################### SUMMARY PLOTS
    def plotbatch_rastersevents_blocked(self, sdir, ntrials = 250, dataset_group = "ep_tknd_sup"):
        """ each site plots many trials, each a separaet event(subplot)
        Withn each subplot, groups trials by category, defined by dataset_group, a column
        in self.DatasetBeh
        """

        if dataset_group=="ep_tknd_sup":
            self.Datasetbeh.grouping_append_col(["epoch", "task_kind", "supervision_stage_concise"], "ep_tknd_sup")
            # self.Datasetbeh.Dat["ep_tknd_sup"].value_counts()
        sites = self.sitegetter_all()
        print("Plotting plotbatch_rastersevents_blocked ...")
        for s in sites:
            if s%10==0:
                print(s)
            fig = self.plotwrapper_raster_onesite_eventsubplots_blocked(s, dataset_group, 
                ntrials=ntrials, sdir=sdir)
            plt.close("all")


    def plotbatch_rastersevents_each_site(self, sdir, ntrials = 150):
        """ each site plots many trials, each a separaet event(subplot)
        NOTE: about 10x faster for .png vs. .pdf
        """
        sites = self.sitegetter_all()
        for s in sites:
            fig = self.plotwrapper_raster_onesite_eventsubplots(s, ntrials=ntrials, sdir=sdir)
            # print("saving fig", s)
            # fig.savefig(f"{sdir}/rastersevents-{self.sitegetter_summarytext(s)}.png", dpi=300)
            plt.close("all")

    def plotbatch_alltrails_for_each_site(self, sdir):
        """ one plot for each site, raster across all trials
        """
        SIZE_HEIGHT_TOTAL = 15.
        SIZE_WIDTH_TOTAL = 20.
        XMIN = -4
        XMAX = 8
        
        trials = self.get_trials_list(True)
        print("This many good trials: ", len(trials))
        print("Ending on trial: ", max(trials))
        sites = self.sitegetter_all()

        # LIST_ALIGN_TO = ["go", "samp"]
        LIST_ALIGN_TO = ["go"]
        for s in sites:
            bregion = self.sitegetter_thissite_info(s)["region"]
            for ALIGNTO in LIST_ALIGN_TO:
                print("generating fig", s, ALIGNTO)
                # fig = self.plotwrapper_raster_multrials_onesite(trials, s, alignto =ALIGNTO,
                #                                               SIZE_HEIGHT_TOTAL=SIZE_HEIGHT_TOTAL, 
                #                                               SIZE_WIDTH_TOTAL=SIZE_WIDTH_TOTAL, 
                #                                               plot_beh=False, xmin = XMIN, xmax = XMAX, nrand_trials=200)[0]
                fig = self.plotwrapper_raster_multrials_onesite(None, s, alignto =ALIGNTO,
                                                              SIZE_HEIGHT_TOTAL=SIZE_HEIGHT_TOTAL, 
                                                              SIZE_WIDTH_TOTAL=SIZE_WIDTH_TOTAL, 
                                                              plot_beh=False, xmin = XMIN, xmax = XMAX, nrand_trials=200)[0]
                print("saving fig", s, ALIGNTO)
                fig.savefig(f"{sdir}/eachsite_alltrials-{s}_{bregion}-alignedto_{ALIGNTO}.png")
            plt.close("all")

    def plot_behcode_photodiode_sanity_check(self, skip_if_done=True):
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
        if skip_if_done:
            # if os.path.exists(sdir):
            if checkIfDirExistsAndHasFiles(sdir)[1]:
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
                out = self.behcode_get_stream_crossings_in_window(trial, behcode, 
                        whichstream=stream, cross_dir_to_take=crosdir, t_pre=t_pre, 
                        t_post=t_post)

                timecross = [t for o in out for t in o["timecrosses"]]
                valscross = [v for o in out for v in o["valcrosses"]]
                time_behcode = [o["time_of_behcode"] for o in out]
                valminmax = [o["valminmax"] for o in out]
                threshold = [o["threshold"] for o in out]

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
            if len(dfthis_pruned)>0:
            
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
    # def datasetbeh_get_trialcode(self, trial):
    #     """ get trialcode for this trial(tdt)
    #     RETURNS:
    #     - trialcode, a string
    #     """

    #     date = self.Date
    #     index_sess, trial_ml = self._beh_get_fdnum_trial(trial)
    #     session_ml = self.BehSessList[index_sess]

    #     trialcode = f"{date}-{session_ml}-{trial_ml}"
    #     return trialcode

    def datasetbeh_trial_to_trialcode(self, trial):
        """ get trialcode for this trial(tdt)
        RETURNS:
        - trialcode, a string
        """
        date = self.Date
        index_sess, trial_ml = self._beh_get_fdnum_trial(trial)
        session_ml = self.BehSessList[index_sess]

        trialcode = f"{date}-{session_ml}-{trial_ml}"
        return trialcode

    def datasetbeh_trial_to_datidx(self, trial, dataset_input=None):
        """ returns the index in self.Datasetbeh correspodning to
        this trial. If doesnt exist, then returns None.
        This is accurate even if self.Datasetbeh is changed.
        - dataset_input, which datsaet to query. useful to pass in a pruned
        datsaet if you want to check whether this trial exists (i.e., returns None)
        """
        tc = self.datasetbeh_trial_to_trialcode(trial)
        # print("get_trials_list - datasetbeh_trial_to_datidx", trial, tc)

        if dataset_input is None:
            dfcheck = self.Datasetbeh
        else:
            dfcheck = dataset_input

        dfthis = dfcheck.Dat[dfcheck.Dat["trialcode"]==tc]

        # if len(dfthis)==0:
        #     print(trial, tc)
        #     assert False, "didnt find this in datasetbeh"
        if len(dfthis)>1:
            print(trial, tc, dfthis)
            assert False, "bug, cant find > 1 row"
        elif len(dfthis)==0:
            # didnt find it, doesnt exits
            return None
        else:
            return dfthis.index[0]

    def datasetbeh_trialcode_to_datidx(self, tc):
        """convert trialcode (date-sess-trial) of beh --> index in self.Datasetbeh
        """
        trial = self.datasetbeh_trialcode_to_trial(tc)
        idx = self.datasetbeh_trial_to_datidx(trial)
        assert idx is not None
        return idx

    def datasetbeh_datidx_to_trial(self, datidx):
        """ returns, for this index in self.Datasetbeh, the
        neural trial """

        trialcode = self.Datasetbeh.Dat.iloc[datidx]["trialcode"]
        return self.datasetbeh_trialcode_to_trial(trialcode)

    def datasetbeh_trialcode_to_trial(self, trialcode):
        """ given trialcode (string) return trial in neural data
        """
        return self._MapperTrialcode2TrialToTrial[trialcode]

    def datasetbeh_trialcode_to_trial_batch(self, list_trialcodes):
        """ Given list of trialcodes, get list of trials in neural data
        """
        return [self.datasetbeh_trialcode_to_trial(tc) for tc in list_trialcodes]

    def datasetbeh_trialcode_prune_within_session(self, list_trialcodes):
        """ Returns subset of list_trialcodes, just those that exist in this session
        """
        assert len(self._MapperTrialcode2TrialToTrial)>0, "cannot check without this..."
        return [trialcode for trialcode in list_trialcodes if trialcode in self._MapperTrialcode2TrialToTrial.keys()]
        


    ##################### SNIPPETS
    def snippets_extract_bystroke(self, sites, DS, pre_dur= -0.4, post_dur= 0.4,
        features_to_get_extra=None, fr_which_version="sqrt", SANITY_CHECK=False):
        """ Helper to extract snippets in flexible way, saligend to each stroke onset.
        PARAMS:
        - sites, list of ints to extract.
        - DS, DatasetStrokes, generated from sn.DatasetBeh
        - features_to_get_extra, list of str, features to extract from DS. fails if these dont
        already exist in DS.
        - SANITY_CHECK, if True, checks alignment across diff columns of df during extraction, no mistakes.
        RETURNS:
        - dataframe, each row a (chan, event).
        """

        import pandas as pd
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        event_name = "00_stroke"
        
        list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
            'stroke_index', 'stroke_index_fromlast', 'stroke_index_semantic', 
            'shape_oriented', 'ind_taskstroke_orig', 'gridloc',
            'gridloc_x', 'gridloc_y']

        OUT = []
        trials = []
        strokeids = []
        list_ind = []
        for ind in range(len(DS.Dat)):

            # print(ind)
            if ind%50==0:
                print("index strokes: ", ind)

            tc = DS.Dat.iloc[ind]["dataset_trialcode"]
            si = DS.Dat.iloc[ind]["stroke_index"]
            trial_neural = self.datasetbeh_trialcode_to_trial(tc) 
            event = f"on_strokeidx_{si}"
            event_time = self.events_get_time_helper(event, trial_neural)[0]

            trials.append(trial_neural)
            strokeids.append(si)
            for s in sites:

                # get spiketimes
                # print(s, "i")
                spike_times = self._snippets_extract_single_snip(s, trial_neural, 
                    event_time, pre_dur, post_dur)
                # get smoothed fr


                # get metadat 
        #         for col in list_cols:
        #             DS.    

                # save it
                OUT.append({
                    "index_DS":ind,
                    "trialcode":tc,
                    "chan":s,
                    "event_aligned":event_name,
                    # "fr_sm":fr_sm, # (1, time)
                    # "fr_sm_times":fr_sm_times,
                    "spike_times":spike_times,
                    "trial_neural":trial_neural,
                    "event_time":event_time
                })

                list_ind.append(ind)
                
        # Get smoothed fr. this is MUCH faster than computing above.
        print("Extracting smoothed FR for all data...")
        fail_if_times_outside_existing = True
        pa = self.smoothedfr_extract_timewindow_bystroke(trials, strokeids, sites, 
            pre_dur=pre_dur, post_dur=post_dur, 
            fail_if_times_outside_existing=fail_if_times_outside_existing) 
        # print(pa.X.shape) # (chans, trials, tbins)
        # print(pa.Trials)
        # print(pa.Chans) 
        # deal out time to each site and trial.
        print("Inserting smoothed FR into dataset...")
        ct = 0
        for i in range(len(DS.Dat)):
            for j in range(len(sites)):
                fr_sm = pa.X[j, i, :]
                fr_sm_times = pa.Times
                OUT[ct]["fr_sm"] = fr_sm[None, :]
                OUT[ct]["fr_sm_times"] = fr_sm_times[None, :]
                ct+=1

                if SANITY_CHECK:
                    assert OUT[ct-1]["chan"] == sites[j]
                    assert OUT[ct-1]["index_DS"] == i

        # ----
        df = pd.DataFrame(OUT)

        # get every column in DS
        # for col in DS.Dat.columns:
        print("Appending other columns into dataset...")
        for col in list_cols:
            df[col] = DS.Dat.iloc[list_ind][col].tolist()
            # OUT[-1][col] = DS.Dat.iloc[ind][col]
        for col in features_to_get_extra:
            df[col] = DS.Dat.iloc[list_ind][col].tolist()
            # OUT[-1][col] = DS.Dat.iloc[ind][col]

        if SANITY_CHECK:
            assert df["trialcode"].tolist() == DS.Dat.iloc[list_ind]["dataset_trialcode"].tolist()

        # Compute fr scalar
        if False:
            # moved to snippets
            dur = post_dur - pre_dur
            def F(x):
                inds = (x["spike_times"]>=pre_dur) & (x["spike_times"]<=post_dur)
                nspk = sum(inds)
                rate = nspk/dur
                return rate
            df = applyFunctionToAllRows(df, F, "fr_scalar_raw")

            # tgransform the fr if desired
            if fr_which_version=="raw":
                df["fr_scalar"] = df["fr_scalar_raw"] 
            elif fr_which_version=="sqrt":
                df["fr_scalar"] = df["fr_scalar_raw"]**0.5
            else:
                print(fr_which_version)
                assert False

            df["fr_sm_sqrt"] = df["fr_sm"]**0.5

        return df

    def snippets_extract_bytrial(self, sites, trials, events,
        features_to_get_extra=None, pre_dur= -0.4, post_dur= 0.4):
        """ Helper to extract snippets in flexible way, saligend to specific events.
        PARAMS:
        - sites, list of ints to extract.
        - features_to_get_extra, list of str, features to extract from D. fails if these dont
        already exist in D.
        RETURNS:
        - dataframe, each row a (chan, event). If any trial doesnt have this events, the trial is excluded.
        """

        import pandas as pd
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        # Extract snippets across all trials, sites, and events.
        # list_cols = ['task_kind']
        list_cols = []

        OUT = []
        list_events_uniqnames = []
        for i_e, e in enumerate(events):
                
            if i_e<10:
                idx_str = f"0{i_e}"
            else:
                idx_str = f"{i_e}"
            event_unique_name = f"{idx_str}_{e}"
            list_events_uniqnames.append(event_unique_name)

            # Skip if this trial doesnt have this event
            pa, trials_this_event = self.smoothedfr_extract_timewindow(trials, sites, e, pre_dur, post_dur)

            if len(trials_this_event)==0:
                # Then this event is not exist in this dataset
                assert pa is None
                print(f"[SN.snippets_extract_bytrial] SKIPPING event, since no data: {e}")
                continue
            else:                
                print(f"[SN.snippets_extract_bytrial] TRIALS extracted for event: {e}: {trials_this_event}")

            assert len(trials_this_event)==pa.X.shape[1]
            # print(len(trials))
            # print(len(sites))
            # print(pa.X.shape) # (sites, trials, tbins)

            for i_t, t in enumerate(trials_this_event):

                if t%100==0:
                    print(t)
                # print(t)

                tc = self.datasetbeh_trial_to_trialcode(t)
                ind_dataset = self.datasetbeh_trial_to_datidx(t)

                for i_s, s in enumerate(sites):

                    # get eventtime
                    list_event_time = self.events_get_time_helper(e, t)
                    # take the first
                    if len(list_event_time)>0:
                        event_time = list_event_time[0]
                    else:
                        print(t, s, e)
                        assert False, "no event"
                    
                    # get spiketimes
                    spike_times = self._snippets_extract_single_snip(s, t, 
                        event_time, pre_dur, post_dur)

                    # get smoothed fr
                    fr_sm = pa.X[i_s, i_t, :]
                    fr_sm_times = pa.Times

                    # if False:
                    #     pa = self.smoothedfr_extract_timewindow([t], [s], e, pre_dur, post_dur)
                    #     fr_sm = pa.X[0, 0,:] # (ntime,)
                    #     fr_sm_times = pa.Times # (ntime,)
                    # else:
                    #     fr_sm = np.array([1])
                    #     fr_sm_times = np.array([1])


                    # save it
                    OUT.append({
                        "trialcode":tc,
                        "chan":s,
                        "event_aligned":event_unique_name,
                        "_event_aligned":e,
                        # "event_aligned_unique":event_unique_name,
                        "spike_times":spike_times,
                        "trial_neural":t,
                        "event_time":event_time,
                        "fr_sm":fr_sm[None, :],
                        "fr_sm_times":fr_sm_times[None, :]
                    })

                    # get metadat 
                    for col in features_to_get_extra:
                        if col not in OUT[-1].keys():
                            OUT[-1][col] = self.Datasetbeh.Dat.iloc[ind_dataset][col]

        df = pd.DataFrame(OUT)

        return df, list_events_uniqnames

    def _snippets_extract_single_snip(self, site, trial, event_time,
            pre_dur, post_dur):
        """ Extract a single snippet's spike times. aligned to event_time.
        """
        dat = self.datall_slice_single_bysite(site, trial)
        spike_times = dat["spike_times"]
        time_on = dat["time_on"]
        time_off = dat["time_off"]
        # spike_times = dat["spiketrain"]
        
        # recenter s times to event
        spike_times = spike_times - event_time
        time_on = time_on - event_time
        time_off = time_off - event_time

        # get windowed spike times
        if True:
            # use popanal
            spike_times = spike_times[(spike_times >= pre_dur) & (spike_times <= post_dur)]
        else:
            # get smoothed fr
            # print(s, "ii")
            if False:
                fr_sm_times, fr_sm = self.elephant_spiketrain_to_smoothedfr(spike_times, 
                    time_on, time_off)
            else:
                fr_sm_times, fr_sm = None, None
        return spike_times


    def subsample_trials(self, n_keep):
        """ Keep n_keep trials, evenly distributed.
        NOTE: is approx n_keep.
        THIS is permanent...
        NOTE: only works if using minimal loading
        """
        assert self._LOAD_VERSION=="MINIMAL_LOADING"

        for k, trials in self._CachedTrialsList.items():
            incr = int(len(trials)/n_keep)
            self._CachedTrialsList[k] = trials[::incr]
            print("pruned trials:", k, "to ", trials)


    #################### CHECK THINGS
    def check_which_fix_cue_version(self, trial):
        """ Determine for this trial which cue version it was, these
        changed over the dates (improvements)
        RETURNS:
        - ver, string name of version
        """

        assert False, "IN PROGRESS, below is just placehoder.."

        if np.any(self._behcode_extract_times(132, trial)):
            # Then this has separation in time
            # touch_fix -> (delay) -> fix cue changes color -> (delay) -> samp
            return ""

        else:
            # old version, 
            # touch fix -> (no delay) -> fix cue changes color -> (delay) -> samp

            pass


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

    def check_preprocess_all(self):
        """ SUmmarize whether each preprocess step has been done
        """

        CheckPreprocess = {}

        # bad sites
        CheckPreprocess["sites_garbage"] = len(self.SitesMetadata["sites_garbage"])>0
        CheckPreprocess["sites_low_fr"] = len(self.SitesMetadata["sites_low_fr"])>0

        self.CheckPreprocess = CheckPreprocess



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
        print("**")
        print("Animal: ", self.Animal)
        print("ExptSynapse: ", self.ExptSynapse)
        print("Date: ", self.Date)
        print("RecSession", self.RecSession)
        print("RecPathBase: ", self.RecPathBase)
        print("BehExpt: ", self.BehExptList)
        print("BehExptSess: ", self.BehSessList)
        if self.Paths is not None:
            print("final_dir_name: ", self.Paths["final_dir_name"])
