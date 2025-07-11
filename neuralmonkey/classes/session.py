""" Holds a single recording file
 - i.e, combines neural, beh, and camera
"""
import tdt
import numpy as np
import scipy
import matplotlib.pyplot as plt
from ..utils.timeseries import dat_to_time
# from ..utils.monkeylogic import getTrialsTaskAsStrokes
from pythonlib.drawmodel.strokePlots import plotDatStrokes
from ..utils import monkeylogic as mkl
import pickle
import os
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from pythonlib.tools.exceptions import NotEnoughDataException, DataMisalignError
from pythonlib.tools.expttools import checkIfDirExistsAndHasFiles
from pythonlib.tools.stroketools import strokesInterpolate2
from pythonlib.globals import PATH_NEURALMONKEY, PATH_DATA_NEURAL_RAW, PATH_DATA_NEURAL_PREPROCESSED, PATH_MATLAB, PATH_SAVE_CLUSTERFIX
# PATH_NEURALMONKEY = "/data1/code/python/neuralmonkey/neuralmonkey"
import pandas as pd
import seaborn as sns
from pythonlib.tools.plottools import savefig

LOCAL_LOADING_MODE = False
LOCAL_PATH_PREPROCESSED_DATA = f"{PATH_DATA_NEURAL_PREPROCESSED}/recordings"
if LOCAL_LOADING_MODE:
    # debugging code.
    PATH_DATA_NEURAL_RAW = "/tmp"
else:
    if not os.path.exists(PATH_DATA_NEURAL_RAW):
        print(PATH_DATA_NEURAL_RAW)
        assert False, "might have to mount servr?"

_REGIONS_IN_ORDER = ("M1_m", "M1_l", "PMv_l", "PMv_m",
                "PMd_p", "PMd_a", "dlPFC_p", "dlPFC_a", 
                "vlPFC_p", "vlPFC_a", "FP_p", "FP_a", 
                "SMA_p", "SMA_a", "preSMA_p", "preSMA_a")
_REGIONS_IN_ORDER_COMBINED = ("M1", "PMv", "PMd", "dlPFC", "vlPFC", "FP",  "SMA", "preSMA")

MAP_COMBINED_REGION_TO_REGION = {
    "M1":["M1_m", "M1_l"],
    "PMv":["PMv_l", "PMv_m"],
    "PMd":["PMd_p", "PMd_a"],
    "dlPFC":["dlPFC_p", "dlPFC_a"],
    "vlPFC":["vlPFC_p", "vlPFC_a"],
    "FP":["FP_p", "FP_a"],
    "SMA":["SMA_p", "SMA_a"],
    "preSMA":["preSMA_p", "preSMA_a"]}
MAP_COMBINED_REGION_TO_REGION = {k:tuple(v) for k, v in MAP_COMBINED_REGION_TO_REGION.items()}

MAP_REGION_TO_COMBINED_REGION = {}
for regcomb, regions in MAP_COMBINED_REGION_TO_REGION.items():
    for reg in regions:
        MAP_REGION_TO_COMBINED_REGION[reg] = regcomb
MAP_REGION_TO_COMBINED_REGION

# SMFR_SIGMA = 0.025
# SMFR_SIGMA = 0.040 # 4/29/23
_SMFR_SIGMA = 0.025 # 4/20/24, # since you can always smoother further later on.
SMFR_TIMEBIN = 0.01

PRE_DUR_TRIAL = 1.
POST_DUR_TRIAL = 1.

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

# Each event has a hard-coded numerical prefix
MAP_EVENT_TO_PREFIX = {
    "stroke":"00",
    "fixon_preparation":"00",
    "fixcue":"00",
    "rulecue2":"02",
    "samp":"03",
    "go_cue":"04",
    "first_raise":"05",
    "on_strokeidx_0":"06",
    "doneb":"08",
    "post":"09",
    "reward_all":"10"
}

DATASETBEH_CACHED_USE_BEHTOUCH = True

HACK_TOUCHSCREEN_LAG = True

def load_mult_session_helper(DATE, animal, dataset_beh_expt=None, expt = "*", 
    MINIMAL_LOADING=True,
    units_metadat_fail_if_no_exist=False,
    spikes_version="kilosort_if_exists", fr_sm_std=None):
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
            MINIMAL_LOADING=MINIMAL_LOADING,
            units_metadat_fail_if_no_exist=units_metadat_fail_if_no_exist,
            spikes_version=spikes_version, fr_sm_std=fr_sm_std)
        SNlist.append(SN)
        print("Extracted successfully for session: ", rec_session)
    assert len(SNlist)>0, "did not find any neural sessions..."

    # Combine into all sessions
    MS = MultSessions(SNlist)

    # Sanity check that the sites match (had issues with KS)
    sites = MS.sitegetter_all(how_combine="intersect")
    for s in sites:
        MS.sitegetter_summarytext(s)

    return MS


def load_session_helper(DATE, dataset_beh_expt=None, rec_session=0, animal="Pancho", 
    expt="*", do_all_copy_to_local=False,
    extract_spiketrain_elephant=False, DEBUG_TIMING=False,
    MINIMAL_LOADING = False, BAREBONES_LOADING=False,
    ACTUALLY_BAREBONES_LOADING = False,
    units_metadat_fail_if_no_exist=False,
    do_if_spikes_incomplete="ignore",
    spikes_version="kilosort_if_exists",
    fr_sm_std=None):
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
    if ACTUALLY_BAREBONES_LOADING or LOCAL_LOADING_MODE:
        beh_sess_list, beh_expt_list, beh_trial_map_list = None, None, None
    else:
        beh_sess_list, beh_expt_list, _, beh_trial_map_list = session_map_from_rec_to_ml2(animal, DATE, rec_session) 

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

    if False:
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

    if not ACTUALLY_BAREBONES_LOADING:
        print("------------------------------")
        print("Loading this neural session:", rec_session)
        print("Loading these beh expts:", beh_expt_list)
        print("Loading these beh sessions:", beh_sess_list)
        print("Using this beh_trial_map_list:", beh_trial_map_list)

    if beh_trial_map_list == [(1, 0)]:
        # Defualt
        ALLOW_RETRY=True
    else:
        # since you want to be able to fail, since to debug this.
        ALLOW_RETRY=False

    # # Hard code sessions where multiple sec sessions to a single beh session, this leads to
    # # error where the rec sessions doesnt know which beh trial to start at.
    # if animal=="Pancho" and int(DATE)==221024 and rec_session==1:
    #     # two rec sessions, one beh session
    #     beh_trial_map_list = [(667, 0)]
    #     ALLOW_RETRY=False
    # elif animal=="Pancho" and int(DATE)==220719:
    #     # one rec session, two beh sessions
    #     beh_trial_map_list = [(1,0), (1,45)]
    #     ALLOW_RETRY=False
    #     beh_expt_list = ["priminvar3e", "priminvar3e"]
    #     beh_sess_list = [1,2]

    #     print(beh_expt_list)
    #     print(beh_sess_list)
    #     print(sessdict[DATE])
    #     assert False
    try:
        SN = Session(DATE, beh_expt_list, beh_sess_list, beh_trial_map_list,
            animal =animal,  
            rec_session = rec_session, dataset_beh_expt=dataset_beh_expt, 
            extract_spiketrain_elephant=extract_spiketrain_elephant,
            do_all_copy_to_local=do_all_copy_to_local, DEBUG_TIMING=DEBUG_TIMING,
            MINIMAL_LOADING= MINIMAL_LOADING, BAREBONES_LOADING=BAREBONES_LOADING,
            units_metadat_fail_if_no_exist=units_metadat_fail_if_no_exist,
            do_if_spikes_incomplete=do_if_spikes_incomplete,
            ACTUALLY_BAREBONES_LOADING=ACTUALLY_BAREBONES_LOADING,
            spikes_version=spikes_version, fr_sm_std=fr_sm_std)
    except DataMisalignError as err:
        if ALLOW_RETRY:
            print("FAILED loading session:", DATE, rec_session)
            print("Possible that this one session maps to multiple beh sessions. try loading it automatically.")
            beh_expt_list, beh_sess_list, beh_trial_map_list = session_map_from_rec_to_ml2_ntrials_mapping(
                animal, DATE, rec_session)
            print("ATTEMPTING RELOAD WITH THESE BEH SESSIONS:")
            print("beh sessions: ", beh_expt_list, beh_sess_list)
            print("beh_trial_map_list: ", beh_trial_map_list)
            SN = Session(DATE, beh_expt_list, beh_sess_list, beh_trial_map_list, 
                animal =animal,  
                rec_session = rec_session, dataset_beh_expt=dataset_beh_expt, 
                extract_spiketrain_elephant=extract_spiketrain_elephant,
                do_all_copy_to_local=do_all_copy_to_local, DEBUG_TIMING=DEBUG_TIMING, 
                MINIMAL_LOADING= MINIMAL_LOADING,
                units_metadat_fail_if_no_exist=units_metadat_fail_if_no_exist,
                do_if_spikes_incomplete=do_if_spikes_incomplete,
                ACTUALLY_BAREBONES_LOADING=ACTUALLY_BAREBONES_LOADING,
                spikes_version=spikes_version, fr_sm_std=fr_sm_std)
        else:
            raise err
    except Exception as err:
        raise err

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
            MINIMAL_LOADING = False, BAREBONES_LOADING=False,
            ACTUALLY_BAREBONES_LOADING = False,
            units_metadat_fail_if_no_exist=False,
            do_if_spikes_incomplete="ignore",
            spikes_version="kilosort_if_exists",
            fr_sm_std=None):
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
        - ACTUALLY_BAREBONES_LOADING, make an instance just for the methods. Not even metadata.
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

        if isinstance(datestr, int):
            datestr = str(datestr)

        if (animal=="Diego" and datestr=="231001") or (animal=="Pancho" and datestr=="231020"):
            print("This day has flipped cue-image onset. It will fail photodiode extraction. ")
            print("See Dropbox/Diego_231001_cue_stim_flipped.png and Dropbox/Diego_231001_cue_stim_normal.png")
            print("Solve this using debug_eventcode...ipynb")
            assert False

        if (animal=="Pancho" and datestr=="220831"):
            print("This day fails, becuase beh (session 3) = rec sessions 3 and 4. Should fix this.")
            assert False

        # if BAREBONES_LOADING:
        #     assert MINIMAL_LOADING == False, "must do one or the other"
        # if MINIMAL_LOADING:
        #     assert BAREBONES_LOADING == False, "must do one or the other"
        FULL_LOADING = (MINIMAL_LOADING==False) and (BAREBONES_LOADING==False) and (ACTUALLY_BAREBONES_LOADING==False)
        assert sum([(ACTUALLY_BAREBONES_LOADING==True), (BAREBONES_LOADING==True), (MINIMAL_LOADING==True), (FULL_LOADING==True)])==1, "must do one or the other"

        if ACTUALLY_BAREBONES_LOADING:
            self._LOAD_VERSION = "ACTUALLY_BAREBONES_LOADING"
        elif BAREBONES_LOADING:
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
        self.Datasetbeh = None

        # For caching mapping from (site, trial) to index in self.DatAll
        self._MapperSiteTrial2DatAllInd = {}
        self._MapperTrialcode2TrialToTrial = {}

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
        self._DEBUG_PRUNE_TRIALS = False

        # Beh cached stuff
        self._BehEyeAlignOffset = None
        self._IGNORE_SANITY_CHECKS = False

        # INitialize empty data
        self.EventsTimeUsingPhd = {}

        # Other flags
        self._FORCE_GET_TRIALS_ONLY_IN_DATASET = False

        # WHICH SPIKES VERSION - store this in spikes_version
        if not self._LOAD_VERSION == "MINIMAL_LOADING":
            # Only do kilosort if this is MINIMAL loading -- ie. meaning
            # that all preprocessing and local stuff has been completed.
            # Needs it to be TDT to complete preprocessing (I think?).
            spikes_version = "tdt"
        else:
            if spikes_version=="kilosort_if_exists":
                # Check if ks exists.
                if self.spiketimes_ks_check_if_exists():
                    spikes_version = "kilosort"
                else:
                    spikes_version = "tdt"
                print("USING THIS SPIKES VERSION: ", spikes_version)

        # spikes versino (initialize as tdt always)
        self.SPIKES_VERSION = "tdt"
        assert self.SPIKES_VERSION=="tdt", "should always start with this, as it indexes tdt channels"

        if ACTUALLY_BAREBONES_LOADING:
            return

        # Initialize paths
        self._initialize_paths(do_if_spikes_incomplete=do_if_spikes_incomplete)

        print("== PATHS for this expt: ")
        for k, v in self.Paths.items():
            print(k, ' -- ' , v)
        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, COMPLETED", "self._initialize_paths()", ts)

        if not LOCAL_LOADING_MODE:
            # print(beh_expt_list, beh_sess_list, beh_trial_map_list)
            assert len(beh_expt_list) == len(beh_sess_list)
            assert len(beh_expt_list) == len(beh_trial_map_list), "these all equal the num beh sessions that have data relevant for this neural session"

        # Immediately fail for these exceptions
        if self.Animal=="Pancho" and int(self.Date)==230124:
            assert False, "WS8 out of space -- Lost about 100min in middle of day."

        # Metadat about good sites, etc. Run this first before other things.
        assert sites_garbage is None, "use metadata instead"
        # then look for metadata
        if spikes_version=="tdt":
            do_sitesdirty_update = True
        else:
            # Then no need.
            do_sitesdirty_update = False
        self.load_metadata_sites(fail_if_no_exist=units_metadat_fail_if_no_exist,
                                 do_sitesdirty_update=do_sitesdirty_update)
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
        self._beh_prune_trial_offset_onset()

        if False: # Actually no need, since now I am loading cached trials. This false means will throw error if try to load trials before load cahced
            # Load behavior, useful for many things -- I previously excluded for MINIMAL, but turns out it might be important
            print("== Loading ml2 behavior")
            self.load_behavior()

        # Sanity check: are you using the correct loading version?
        # Not allowed to do full loading if you already have gotten to final preprocessed state
        if self._LOAD_VERSION == "FULL_LOADING":
            check_dict = self._check_preprocess_status()
            assert check_dict["allow_full_loading"], "you have already completed preproicessing. use MINIMAL_LOADING instead"
        elif self._LOAD_VERSION == "MINIMAL_LOADING":

            ########## MUST DO THESE HERE, SINCE IT EXTRACTS TRIALS THAT ARE THEN USED IN _check_preprocess_status
            # not FULL_LOADING. Expect to load all previously cached data.
            assert do_all_copy_to_local == False
            assert do_sanity_checks_rawdupl == False
            assert extract_spiketrain_elephant == False

            # Load previously cached.
            print("** MINIMAL_LOADING, therefore loading previuosly cached data")
            self._savelocalcached_load()
            ###################

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
            # Load dataset beh
            self._savelocalcached_load_dataset_beh()

            # Hacky things to do, since cannot do in oeroginal extract of dataset.
            self.Datasetbeh.supervision_epochs_extract_orig()

            # Delete behavior, if it had been loaded above to support preprocesing and loading.
            self.BehFdList =[]

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
            try:
                self._beh_prune_trial_number()
                self._beh_validate_trial_number()
                self._beh_validate_trial_mapping(ploton=True, do_update_of_mapper=True, 
                    fail_if_not_aligned=False)
            except AssertionError as err:
                raise DataMisalignError

            if DEBUG_TIMING:
                ts = makeTimeStamp()
                print("@@@@ DEBUG TIMING, COMPLETED", "self._beh_validate_trial_mapping()", ts)
                print("@@@@ DEBUG TIMING, COMPLETED", "self._beh_validate_trial_number()", ts)

            # Load beh dataset
            # (Run here early, since it is needed for downstream stuff).
            print("RUNNIGN datasetbeh_load_helper")
            self.datasetbeh_load_helper(dataset_beh_expt)

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

            # # Load beh dataset
            # print("RUNNIGN datasetbeh_load_helper")
            # self.datasetbeh_load_helper(dataset_beh_expt)

        # Various cleanups
        self._cleanup()
        # self._datasetbeh_remove_neural_trials_missing_beh()

        if DEBUG_TIMING:
            ts = makeTimeStamp()
            print("@@@@ DEBUG TIMING, COMPLETED", "self._cleanup()", ts)
            
        # Initialize mappers
        self.MapSiteToRegionCombined = {}
        self.MapSiteToRegion = {}
        # self.MapDatasetKeyTo

        # # Check if ks exists.
        # if spikes_version=="kilosort_if_exists":
        #     if self.spiketimes_ks_check_if_exists():
        #         spikes_version = "kilosort"
        #     else:
        #         spikes_version = "tdt"

        # Extract spikes if needed/.
        if spikes_version=="tdt":
            # pass, no need to load
            pass
        elif spikes_version=="kilosort":
            # Load 
            self.spiketimes_ks_load()
            self.spiketimes_ks_extract_alltrials()
        else:
            print(spikes_version)
            assert False, "Code it"

        # Switch to the spikesversion
        # Do it here so that dont inadvertantly cache data for the wrong spikes version.
        self.SPIKES_VERSION=spikes_version
        self._SPIKES_VERSION_INPUTED = spikes_version

        if fr_sm_std is not None:
            assert fr_sm_std>0. and fr_sm_std<0.1, "mistake? this is unexpted window size"
            self.SMFR_SIGMA = fr_sm_std
        else:
            self.SMFR_SIGMA = _SMFR_SIGMA

    ####################### PREPROCESS THINGS
    def _datasetbeh_remove_neural_trials_missing_beh(self):
        """ Does ONE THING --> removes neural trials which dont exist in dataset.
        This is useful since dataset beh might prune trials which have bad strokes, etc.
        ACTUALLY - this now doesnt do anything. Istead,  just turns on flag to always only return trials that
        are in Dataset.
        """

        if True:
            # Always do this...
            self._FORCE_GET_TRIALS_ONLY_IN_DATASET = True
            self._FORCE_GET_TRIALS_ONLY_IN_DATASET_NTRIALS = len(self.Datasetbeh.Dat) # as sanity check later, to make sure that dataset has not been pruned since now.
        else:
            # All this does is check that not too many trials misisng. This is not needed, and often fails.
            if self.Datasetbeh is not None:
                if False:
                    # This is now done within loading of dataset
                    self.Datasetbeh._cleanup_preprocess_each_time_load_dataset()

                if False: # 2/4/24/ now done in _cleanup_reloading_saved_state()
                    # Important, to reset all tokens, some which mgith be incompleted, using
                    # old code, e.g., gridloc_local
                    self.Datasetbeh.behclass_preprocess_wrapper(skip_if_exists=False)

                # sanity check that every neuiral trial has a dataset trial
                # UPDATE: it is possible for some cached neural trials to not exist in dataset (e.g.,
                # recent code cleans up strokes in dataset).
                # Update trials list by pruning trials missing from dataset.
                # trials = self.get_trials_list(True)
                trials = self.get_trials_list(True)
                n_trials_missing_from_dataset = 0
                for t in trials:
                    # print(t, sn.datasetbeh_trial_to_datidx(t))
                    if self.datasetbeh_trial_to_datidx(t) is None:
                        n_trials_missing_from_dataset += 1
                        if False: # Older code, would raise error and print info.
                            tc = self.datasetbeh_trial_to_trialcode(t)
                            dfcheck = self.Datasetbeh
                            for _t in self.get_trials_list(True):
                            # for _t in self.get_trials_list(False):
                                print(_t, self.datasetbeh_trial_to_trialcode(_t))
                            print(tc)
                            print(t)
                            print(dfcheck.Dat["trialcode"])
                            dfthis = dfcheck.Dat[dfcheck.Dat["trialcode"]==tc]
                            print(dfthis)
                            assert False
                if n_trials_missing_from_dataset>0:
                    if False:
                        # Previously, would force fail, but that was when self.get_trials_list(True) used True. Now I turned to
                        # false, which means many neural trials will correctly not be in dataste. I have to do this becuase
                        # the True flag requires dataset, and it will fail. BUt this is ok, since now I always load dataset with nerual data.
                        if (n_trials_missing_from_dataset/len(trials)) > 0.05:
                            print("threw out >5% trials, missing from dataset. figure out why")
                            print(n_trials_missing_from_dataset)
                            print(len(trials))
                            assert False
                    # Then turn on flag to only keep trials in dataset. Otherwise might have error.
                    self._FORCE_GET_TRIALS_ONLY_IN_DATASET = True
                    self._FORCE_GET_TRIALS_ONLY_IN_DATASET_NTRIALS = len(self.Datasetbeh.Dat) # as sanity check later, to make sure that dataset has not been pruned since now.

        # Sanity check -- every neural trial should have trial in dataset
        if False: # Actually skip this -- it is perfeclty fine for a neural trial to not have beh trial, this occurs
            # when actually it is a bad trial.
            for trial in self.get_trials_list(True):
                if self.datasetbeh_trial_to_datidx(trial) is None:
                    print(self.Datasetbeh.Dat["trialcode"].tolist())
                    print(trial)
                    print(self.datasetbeh_trial_to_trialcode(trial))
                    print(self._FORCE_GET_TRIALS_ONLY_IN_DATASET)
                    print(self._FORCE_GET_TRIALS_ONLY_IN_DATASET_NTRIALS)
                    assert False, f"dont knwo why -- this neural trail could not be found in Dataset... {trial}"

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

        # Make sure "done button" event occurs after go cue in time (possiuble to get contamination so
        # it uses fixation time)

        def _check_increasing_up_to_nans(x):
            """ Check that array has increasinve values, until it hits a nan, after which
            they are all nans
            RETURNS:
            - None, but throws AssertionError if fails
            EXamples:
            - [1, 2, 3,4, np.nan, np.nan, np.nan] --> good
            - [1, 2, 3,4, np.nan, np.nan, 5 ] --> bad
            - [1, 2, 3,2, np.nan, np.nan, np.nan] --> bad
            """
            if np.isnan(x[1]):
                assert np.all(np.isnan(x[2:]))
            elif np.isnan(x[2]):
                assert np.all(np.isnan(x[3:]))
                assert np.all(np.diff(x[:2])>0)
            elif np.isnan(x[3]):
                assert np.all(np.isnan(x[4:]))
                assert np.all(np.diff(x[:3])>0)
            elif np.isnan(x[4]):
                assert np.all(np.isnan(x[5:]))
                assert np.all(np.diff(x[:4])>0)

        if self._LOAD_VERSION == "MINIMAL_LOADING":
            # Make sure that events are increasing, and if trial is cut short, then none of the events after the time
            # of cutting exist
            DO_SAVE = False

            def _raise_error(trial, tmp, err):
                print("Trial, ", trial)
                print(tmp)
                self.print_summarize_expt_params()
                print("timings are not possible. bug.")

                print("This trialcode:")
                print(self.datasetbeh_trial_to_trialcode(trial))
                raise err


            for trial in self.get_trials_list(True):                
                tmp = self.events_get_times_as_array(trial, ["fixcue", "fixtch", "samp", "go", "first_raise", "doneb", "post"])
                
                try:
                    _check_increasing_up_to_nans(tmp)

                except AssertionError as err:
                    # try recomputing all the event times
                    self.events_get_time_using_photodiode_and_save(list_trial = [trial], do_save=False)
                    
                    # check again
                    tmp = self.events_get_times_as_array(trial, ["fixcue", "fixtch", "samp", "go", "first_raise", "doneb", "post"])
                    
                    try:
                        _check_increasing_up_to_nans(tmp)
                    except Exception as err:
                        _raise_error(trial, tmp, err)

                    DO_SAVE = True

                except Exception as err:
                    _raise_error(trial, tmp, err)

            if DO_SAVE:
                self._savelocal_events()

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


    def load_metadata_sites(self, fail_if_no_exist=True, do_sitesdirty_update=True):
        """ Load info about which sites are garbage, hand coded
        PARAMS:
        """
        from pythonlib.tools.expttools import load_yaml_config
        import os

        # path = f"{self.Paths['metadata_units']}/{self.Date}.yaml"
        path = f"{PATH_NEURALMONKEY}/metadat/units/{self.Date}.yaml"
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
            if fail_if_no_exist:
                assert False, "make the /metadat/units/<date>.yaml file by hand."
        
        if do_sitesdirty_update:
            # Auto.
            dirty_kinds = None
        else:
            # Skip.
            dirty_kinds = []
        self._sitesdirty_noisy_update(dirty_kinds=dirty_kinds)

    def sitesdirtygood_preprocess_wrapper(self, SAVEDIR=None, nsigma=3.5):
        """
        Wrapper for all things for good pruning of chans and trails within chans, based on firing rates
        stats -- outliers.
        
        For each chan, gets stats based on (fr over trials), such as drift and variance fr across blocks of trials.
        
        For each chan, get list of trials which are bad, in having mean fr higher or lower than mean trial by nsigma*STD.

        Apply threshold for different metrics, keeping only those chans that pass for all metrics.
        """

        assert False, "obsolete -- use the one in MultSessions"
        
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.pandastools import savefig

        # Manually input thresholds
        # These based on Diego,, 240508.
        map_var_to_thresholds = {
            "frstd_spread_index_across_bins":(0, 1),
            "slope_over_mean":(-0.18, 0.18),
            "fr_spread_index_across_bins":(0, 0.4),
            "frac_trials_bad":(0, 0.012)
        }

        if SAVEDIR is None:
            from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED
            SAVEDIR = f"{PATH_DATA_NEURAL_PREPROCESSED}/sitesdirtygood_preprocess/{self.Animal}-{self.Date}-{self.RecSession}"
            os.makedirs(SAVEDIR, exist_ok=True)

        ### First, extract spikes data for each (site, trial)
        # NOTE: if this is already done, will be very fast.
        self.sitestats_fr_get_and_save(save=False)

        ### COLLECT ALL DATA        
        savedir = f"{SAVEDIR}/fr_over_trials"
        os.makedirs(savedir, exist_ok=True)

        sites = self.sitegetterKS_all_sites()
        trials = self.get_trials_list()
        res = []
        for chan in sites:
            print(chan)
            
            # FR stability/drift
            frvals, trials_this, times_frac, metrics, inds_bad = self.sitesdirtygood_preprocess_firingrate_drift(
                chan, trials, savedir, nsigma=nsigma)

            # FR outliers, specific trials.
            res.append({
                "chan":chan,
                "frvals":frvals,
                "trials":trials_this,
                "times_frac":times_frac,
                "inds_bad":inds_bad,
            })
            for name, val in metrics.items():
                res[-1][name] = val

            plt.close("all")

        dfres = pd.DataFrame(res)
        dfres["bregion"] = [self.sitegetterKS_map_site_to_region(chan) for chan in dfres["chan"].tolist()]

        ### Generate dataframe and add things to it
        # Quantify, frac trials that are outliers
        def F(x):
            return len(x)
        dfres["n_inds_bad"] = dfres["inds_bad"].apply(F)
        dfres["n_trials"] = [len(trials) for trials in dfres["trials"]]
        dfres["frac_trials_bad"] = dfres["n_inds_bad"]/dfres["n_trials"]

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
            "animal":self.Animal,
            "ExptSynapse":self.ExptSynapse,
            "date":self.Date,
            "RecSession":self.RecSession,
            "spikes_version":self.SPIKES_VERSION,
            "trials":trials,
            "chans":sites,
            "chans_good":dfres[dfres["good_chan"]]["chan"].tolist(),
            "trialcodes":[self.datasetbeh_trial_to_trialcode(t) for t in trials],
        }
        from pythonlib.tools.expttools import writeDictToYaml, writeDictToTxtFlattened
        writeDictToYaml(params, f"{SAVEDIR}/params.yaml")
        writeDictToTxtFlattened(params, f"{SAVEDIR}/params_text.yaml")

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

        return dfres, params, SAVEDIR

    def sitesdirtygood_preprocess_firingrate_drift(self, chan, trials, savedir=None,
                                                   ntrials_per_bin = 50, nsigma=3.5):
        """
        Score the across-day drift in FR
        """
        from pythonlib.tools.datetools import standardize_time_helper
        from sklearn.linear_model import LinearRegression
        # import pandas as pde
        from neuralmonkey.metrics.goodsite import score_firingrate_drift

        # def _threshold_fr(frvals):
        #     """
        #     """
        #     mu = np.mean(frvals)
        #     sig = np.std(frvals)
        #     thresh_upper = mu + nsigma*sig
        #     thresh_lower = mu - nsigma*sig
        #     inds_bad_bool = (frvals>thresh_upper) | ((frvals<thresh_lower))
        #     inds_bad = [int(x) for x in np.where(inds_bad_bool)[0]]
        #     return inds_bad, thresh_lower, thresh_upper, inds_bad_bool

        # def _plot(_x, _y, ax):
        #     """
        #     """
        #     inds_bad, thresh_lower, thresh_upper, inds_bad_bool= _threshold_fr(_y)
        #     _x_bad = _x[inds_bad]
        #     _y_bad = _y[inds_bad]
        #     ax.plot(_x, _y, 'xk');
        #     ax.plot(_x_bad, _y_bad, 'or')
        #     # plot text next to each
        #     for t, fr in zip(_x_bad, _y_bad):
        #         ax.text(t, fr, f"{t:.2f}", color="r")
        #     ax.set_ylim(bottom=0.)
        #     # ax.set_title(f"fr_spread_index_across_bins={fr_spread_index_across_bins.item():.2f}")

        ### Extract data
        frvals, trials, times_frac, trialcodes = self.sitestats_fr_extract_good(chan, trials, extract_using_smoothed_fr=False, keep_within_events_flanking_trial=True)
        # # (1) firing rates across trials (one scalar each trial)
        # list_fr = []
        # for t in trials:
        #     list_fr.append(self.sitestats_fr_single(chan, t, extract_using_smoothed_fr=False, keep_within_events_flanking_trial=True))
        # frvals = np.array(list_fr)
        # # frvals_sq = frvals**0.5

        # # (2) times (frac of day)
        # trials = np.array(trials)
        # dfthis = self.datasetbeh_extract_dataframe(trials)
        # times_frac = np.array([standardize_time_helper(dt) for dt in dfthis["datetime"].tolist()])

        metrics, inds_bad = score_firingrate_drift(frvals, times_frac, 
                                                                               trials, ntrials_per_bin, 
                                                                               nsigma, savedir=savedir, savename=self.sitegetter_summarytext(chan))

        return frvals, trials, times_frac, metrics, inds_bad

        # # (3) Trehshodl the firing rate
        # inds_bad, thresh_lower, thresh_upper, inds_bad_bool = _threshold_fr(frvals_sq)
        
        # ### Metrics (Score drift)_
        # nbins = int(len(times_frac)/ntrials_per_bin)
        # if nbins == 1:
        #     # SKIP THIS, too few trials.
        #     fr_spread_index_across_bins = None
        #     slope_over_intercept = None
        # else:
        #     # from pythonlib.tools.nptools import bin_values_by_rank
        #     # bin_values_by_rank(times_frac, nbins)

        #     frvals_sq_no_outlier = frvals_sq[~inds_bad_bool]
        #     times_frac_no_outlier = times_frac[~inds_bad_bool]
        #     trials_this_no_outlier = trials_this[~inds_bad_bool]

        #     # (1) linear, across day
        #     reg = LinearRegression().fit(times_frac_no_outlier[:, None], frvals_sq_no_outlier[:, None])
        #     slope = reg.coef_.item()
        #     frmean = np.mean(frvals_sq_no_outlier)
        #     # intercept = reg.intercept_.item()
        #     slope_over_mean = slope/frmean # units = intercepts/day
        #     slope_over_mean = slope_over_mean/24 # units = intercepts/hour.
        #     # print(slope, intercept, slope_over_intercept)

        #     # (2) any block of time with very diff fr from others?
        #     # Any trial bins with deviation in fr? Get index of (max - min)/(mean) across 50-trial bins.
        #     dfrate = pd.DataFrame({"fr":frvals_sq_no_outlier, "times_frac":times_frac_no_outlier, "trials":trials_this_no_outlier})
        #     dfrate["times_frac_bin"] = pd.qcut(dfrate["times_frac"], nbins) # bin it

        #     fr_max_across_bins = np.max(dfrate.groupby("times_frac_bin").mean()["fr"])
        #     fr_min_across_bins = np.min(dfrate.groupby("times_frac_bin").mean()["fr"])
        #     fr_mean_across_bins = np.mean(dfrate.groupby("times_frac_bin").mean()["fr"])
        #     fr_spread_index_across_bins = (fr_max_across_bins - fr_min_across_bins)/fr_mean_across_bins        

        #     # (3) Any block with very high variance across trials?
        #     frstd_max_across_bins = np.max(dfrate.groupby("times_frac_bin").std()["fr"])
        #     frstd_min_across_bins = np.min(dfrate.groupby("times_frac_bin").std()["fr"])
        #     frstd_mean_across_bins = np.mean(dfrate.groupby("times_frac_bin").std()["fr"])
        #     frstd_spread_index_across_bins = (frstd_max_across_bins - frstd_min_across_bins)/frstd_mean_across_bins        

        # ### Plots
        # # for each chan and event, find outlier trials
        # fig, axes = plt.subplots(5,1, figsize=(10,16))

        # ax = axes.flatten()[0]
        # ax.hist(frvals, 50, color="k");
        # ax.set_xlabel("firing rate histogram")

        # ax = axes.flatten()[1]
        # ax.hist(frvals_sq, 50, log=True, color="g");
        # ax.set_xlabel("firing rate histogram (sqrt)")

        # ax = axes.flatten()[2]
        # # print(trials_this)
        # # print(frvals)
        # _plot(trials_this, frvals, ax)
        # ax.set_ylabel("fr (hz)")
        # ax.set_title(f"slope_over_mean={slope_over_mean:.2f}")

        # ax = axes.flatten()[3]
        # _plot(trials_this, frvals_sq, ax)
        # ax.set_ylabel("fr (hz**0.5)")
        # ax.set_title(f"frstd_spread_index_across_bins={frstd_spread_index_across_bins:.2f}")

        # # Plot against time.
        # ax = axes.flatten()[4]
        # _plot(times_frac, frvals_sq, ax)
        # ax.set_ylabel("fr (hz**0.5)")
        # ax.set_title(f"fr_spread_index_across_bins={fr_spread_index_across_bins:.2f}")

        # ### SAVE
        # if savedir is not None:
        #     from pythonlib.tools.pandastools import savefig
        #     savefig(fig, f"{savedir}/{self.sitegetter_summarytext(chan)}.pdf")

        # metrics = {
        #     "fr_spread_index_across_bins":fr_spread_index_across_bins, 
        #     "frstd_spread_index_across_bins":frstd_spread_index_across_bins, 
        #     "slope_over_mean":slope_over_mean
        # }

        # return frvals, trials_this, times_frac, metrics, inds_bad



    def sitesdirty_filter_by_spike_magnitude(self, 
            # MIN_THRESH = 90, # before 2/12/23
            # MIN_THRESH = 70, # this seems better, dont throw out some decent units.
            MIN_THRESH = 50, # 2/8/24 - this best, for population analyses, keep more.
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
        sites = self.sitegetterKS_map_region_to_sites_MULTREG(clean=False)
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

            if False:
                if keep:
                    print("o ", s, pt)
                else:
                    print("- ", s, pt)

        # histogram
        if plot_results:
            fig, axes = plt.subplots(2,1)
            ax = axes.flatten()[0]
            ax.hist(dfthis["spk_peak_to_trough"], bins=40)

            sns.relplot(data=dfthis, x="site", y="spk_peak_to_trough", hue="keep", aspect=2)
            plt.grid(True)


        return dfthis

    def _sitesdirty_noisy_update(self, dirty_kinds = None):
        """
        Filter sites to find those that are "dirty" based on flexibly criteria (dirty_kinds).
        :param dirty_kinds:
        :return:
        """
        if dirty_kinds is None:
            # dirty_kinds = ("sites_garbage", "sites_low_fr",  # before 2/13/23
            #     "sites_error_spikes", "sites_low_spk_magn")
            if LOCAL_LOADING_MODE:
                # cant access server for spike magnitude info
                dirty_kinds = ("sites_garbage", 
                    "sites_error_spikes")
            else:
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
        """ Run a single time to generate mappers, which are dicts for mapping
        between indices for trialcode (datasetbeh) and trial (neural)
        RETURNS:
           - Updates self._MapperTrialcode2TrialToTrial
        """
        if hasattr(self, "_MapperTrialcode2TrialToTrial") and len(self._MapperTrialcode2TrialToTrial)>0:
            # SANITY CHECK that it mathes current. if not, then error --> misalign betwene nerual and dataset beh
            for trial in self.get_trials_list():
                trialcode = self._datasetbeh_trial_to_trialcode_from_raw(trial)
                assert self._MapperTrialcode2TrialToTrial[trialcode] == trial
        else:
            # Generate from scratch
            # 1) map from trialcode to trial
            self._MapperTrialcode2TrialToTrial = {}

            for trial in self.get_trials_list():
                trialcode = self._datasetbeh_trial_to_trialcode_from_raw(trial)
                assert trialcode not in self._MapperTrialcode2TrialToTrial.keys(), "diff trials give same trialcode, not possible."
                self._MapperTrialcode2TrialToTrial[trialcode] = trial
            print("Generated self._MapperTrialcode2TrialToTrial!")

    # def _initialize_params(self):
    def _initialize_paths(self, do_if_spikes_incomplete="ignore"):
        """
        Initalize paths to data, including searching for raw data
        PARAMS:
        - do_if_spikes_incomplete, str, what to do if do not find complete set of 
        sorted spike data.
        """

        # 1) find all recordings for this date
        from pythonlib.tools.expttools import findPath, deconstruct_filename
        path_hierarchy = [
            [self.Animal],
            [self.Date]
        ]
        if LOCAL_LOADING_MODE:
            paths = findPath(LOCAL_PATH_PREPROCESSED_DATA, path_hierarchy)
        else:
            paths = findPath(self.RecPathBase, path_hierarchy)

        if False: # Actually, no need, becuase evne if there are more preprcessing folders, the code below only picks out those
            # that match the raw rec data folders. So is fine.

            # Sanity check that the number of rec sessions on server (raw neural) matches the number of preprocessed data
            # If not, then this is a bug -- you probabyl pruned a sessions on server and left it in place in the preprocessed data
            if self._LOAD_VERSION == "MINIMAL_LOADING": 
                # THen this means preprocessing across all sessions hsould be finalized at this point.
                _paths1 = findPath(LOCAL_PATH_PREPROCESSED_DATA, path_hierarchy)
                _paths2 = findPath(self.RecPathBase, path_hierarchy)

                # If paths are not the 
                if len(_paths1) != len(_paths2):
                    print(_paths1)
                    print(_paths2)
                    assert False, "Fix this -- align the paths. PRobably need to delete a preprocess date?"
                for p1, p2 in zip(_paths1, _paths2):
                    if deconstruct_filename(p1)["filename_final_noext"]!=deconstruct_filename(p2)["filename_final_noext"]:
                        print(_paths1)
                        print(_paths2)
                        assert False, "Fix this -- align the paths. PRobably need to delete a preprocess date?"

        # REmove paths that say "IGNORE"
        paths = [p for p in paths if "IGNORE" not in p]
        # assert len(paths)==1, 'not yhet coded for combining sessions'
        if len(paths)==0:
            print("***^^*")
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

        ####### LOAD PRESAVED PATHS
        if LOCAL_LOADING_MODE:
            from pythonlib.tools.expttools import load_yaml_config
            pathbase_local = f"{self.RecPathBaseLocal}/{self.Animal}/{self.Date}/{final_dir_name}"
            pathdict = load_yaml_config(f"{pathbase_local}/paths.yaml")
            pathdict["cached_dir"] = cached_dir
            self.Paths = pathdict

            # Update paths with local directory
            paths_updated = {}
            # keys_to_remove = ["raws", "tank", "spikes"]
            for k, v in self.Paths.items():
                print(k, "---", v)
                if "/mnt" in v:
                    # Then this is a server path. remove it
                    continue
                else:
                    # Keep it. decide if to Replace string in path
                    s1 = "/gorilla1/neural_preprocess/recordings/"
                    ind1 = v.find(s1)

                    s2 = "/home/lucast4/code/neuralmonkey/neuralmonkey/"
                    ind2 = v.find(s2)
                    if ind1>-1:
                        v_new = f"{LOCAL_PATH_PREPROCESSED_DATA}/{v[ind1+len(s1):]}"
                    elif ind2>-1:
                        v_new = f"{PATH_NEURALMONKEY}/{v[ind2+len(s2):]}"
                    else:
                        # this is not a path to change
                        v_new = v
                    paths_updated[k] = v_new

            print("LOCAL LOADING - updated self.Paths:")
            for k, v in paths_updated.items():
                print(k, " ==== ", v)
            self.Paths = paths_updated

        else:
            def _get_spikes_raw_path():
                """ checks to find path to folder holding spikes data, in order of most to 
                least desired version. Returns None if doesnt find. 
                """
                from pythonlib.tools.expttools import load_yaml_config
                from pythonlib.tools.expttools import count_n_files_in_dir

                NCHANS = 512

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
                for suffix in ["-4", "-3.5", "-4.5", ""]: 
                    path_maybe = f"{paththis}/spikes_tdt_quick{suffix}"
                    # if os.path.exists(path_maybe):
                    # print("CHECKING ... ", path_maybe, checkIfDirExistsAndHasFiles(path_maybe))
                    if checkIfDirExistsAndHasFiles(path_maybe)[0]:
                        # count how many files
                        # nfiles, list_files = count_n_files_in_dir(path_maybe, "png") # This failed.
                        nfiles, list_files = count_n_files_in_dir(path_maybe, "mat") # This is the iomportant file
                        if nfiles == NCHANS*2: # x2, becuase there are 2 .mat files per chan.
                            print("FOund this path for spikes: ", path_maybe)
                            return path_maybe
                
                # Didn't find spikes, return None
                print("DIdnt find spikes directory")
                return None

            if self.Animal=="Pancho":
                metadata_units = f"{PATH_NEURALMONKEY}/metadat/units"
            else:
                metadata_units = f"{PATH_NEURALMONKEY}/metadat/units_{self.Animal}"

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
                "metadata_units":metadata_units,
                "cached_dir":f"{pathbase_local}/cached",
                }

            self.Paths = pathdict
            self.PathRaw = pathdict["raws"]
            self.PathTank = pathdict["tank"]

            # First, check if you have all spikes laready extracted, if not, then 
            # reextract
            def _missing_spikes():
                """ Returns True if any channel is missing spikes... bsaed on filenames in spikes folder."""
                for site in self.sitegetterKS_map_region_to_sites_MULTREG(clean=False):
                    if self._spikes_check_file_exists_tdt(site)==False: 
                        print("++ MISSING THIS SITE's SPIKE DATA:", site)
                        return True
                return False

            if self.Paths['spikes'] is None or _missing_spikes():
                # Then do not have complete set...
                if do_if_spikes_incomplete=="fail":
                    print("------ spikes not gotten...")
                    print("self.Paths['spikes']", self.Paths['spikes'])
                    print(_missing_spikes())
                    self.print_summarize_expt_params()
                    assert False, "Missing some spikes!!!"
                elif do_if_spikes_incomplete=="extract_quick_tdt":
                    # Reextract it using quick thresholding (tdt)
                    print("-- Extracting spikes thresholded (TDT)!! (becuase did not find spikes data...)")
                    from neuralmonkey.utils.matlab import spikes_extract_quick_tdt
                    try:
                        spikes_extract_quick_tdt(self.Animal, self.Date)
                        print("-- Successfully completed spikes extraction!!")
                    except Exception as err:
                        self.print_summarize_expt_params()
                        print(err)
                        assert False
                    
                    # # CHeck again is you are missing spikes
                    # if _missing_spikes():
                    #     print("** STILL MISSING SPIKES! Probably havent transfered all sev file sto server??")
                    #     self.print_summarize_expt_params()
                    #     assert False

                    # Now try reinitializing paths
                    self.Paths = {}
                    self.PathRaw = {}
                    self.PathTank = {}
                    self._initialize_paths(do_if_spikes_incomplete="fail")

                elif do_if_spikes_incomplete=="ignore":
                    # Is ok. do nothing
                    pass
                else:
                    print(do_if_spikes_incomplete)
                    self.print_summarize_expt_params()
                    assert False

    ####################### EXTRACT RAW DATA (AND STORE)
    def load_behavior(self):
        """ Load monkeylogic data for this neural recording
        Could be either filedata, or dataset, not sure yet , but for now go with filedata
        """ 
        from ..utils.monkeylogic import loadSingleDataQuick
        if len(self.BehFdList)==0:
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



    def load_tdt_tank(self, include_streams=False, force_reload_from_raw=False):
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

        if force_reload_from_raw:
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

    def _spikes_check_file_exists_tdt(self, site):
        """ Returns True if the tdt spike thresholding has been done
        """
        
        if self.Paths['spikes'] is None:
            print("self.Paths['spikes'] is None")
            print("self.Paths:", self.Paths)
            return False

        rs, chan = self.convert_site_to_rschan(site)
        fn = f"{self.Paths['spikes']}/RSn{rs}-{chan}.mat"
        return os.path.exists(fn)

    def _load_spike_times(self, rs, chan, ver="spikes_tdt_quick", 
            return_none_if_fail=True, actually_return_none_if_fail=False):
        """ Load specific site inforamtion
        Return spike times, pre-extracted elsewhere (matlab)
        RETURNS:
        - spike times, array (nspikes, ) in sec.
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
                if actually_return_none_if_fail:
                    return None
                else:
                    print("[scipy error] Skipping spike times for (rs, chan): ", rs, chan)
                    self.print_summarize_expt_params()
                    raise
            except Exception as err:
                if actually_return_none_if_fail:
                    return None
                else:
                    print(err)
                    print("[_load_spike_times] Failed for this rs, chan: ",  rs, chan)
                    self.print_summarize_expt_params()
                    raise
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
        # for k in [(False, True), (True, False), (True, True), (False, False)]:
        #     if k not in self._CachedTrialsList.keys():
        #         print(k)
        #         print(self._CachedTrialsList)
        #         assert False

        for trial in self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS):
            self._CachedTrialOnset[trial] = self.ml2_get_trial_onset(trial)

        for trial in self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS):
            self._CachedStrokesTask[trial] = self.strokes_task_extract(trial)
            self._CachedStrokes[trial] = self.strokes_extract(trial, peanuts_only=False)
            self._CachedStrokesPeanutsOnly[trial] = self.strokes_extract(trial, peanuts_only=True)
            self._CachedTouchData[trial] = self.beh_extract_touch_data(trial)


    def _savelocalcached_save(self, save_dataset_beh=True, save_datslices=True,
            ONLY_EXTRA_STUFF=False):
        """
        Save to disk all cached data in self._Cached... This saves quickly.
        """

        # ONLY ALLOWED to do this if this was not using MINIMAL loading. Otherwise not sure
        # if did correct sanity checks (which si only possible wihtout minimal locading)
        if not ONLY_EXTRA_STUFF:
            assert self._LOAD_VERSION == "FULL_LOADING"
        # assert self._MINIMAL_LOADING == False
        # assert self._BAREBONES_LOADING == False

        pathdir = self.Paths["cached_dir"]

        def _save_this(this, filename):
            # _CachedTrialOnset
            path = f"{pathdir}/{filename}.pkl"
            with open(path, "wb") as f:
                pickle.dump(this, f)

        _save_this(self.BehTrialMapListGood, "BehTrialMapListGood")
        _save_this(self.BehTrialMapList, "BehTrialMapList")
        _save_this(self.BehSessList, "BehSessList")
        _save_this(self._MapperTrialcode2TrialToTrial, "_MapperTrialcode2TrialToTrial")

        if not ONLY_EXTRA_STUFF:
            _save_this(self._CachedTrialOnset, "trial_onsets")
            _save_this(self._CachedTouchData, "touch_data")
            _save_this(self._CachedStrokes, "strokes")
            _save_this(self._CachedStrokesPeanutsOnly, "strokes_peanutsonly")
            _save_this(self._CachedStrokesTask, "strokes_task")

            for k in [(False, True), (True, False), (True, True), (False, False)]:
                if k not in self._CachedTrialsList.keys():
                    print(k)
                    print(self._CachedTrialsList)
                    assert False
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
                    for site in self.sitegetterKS_map_region_to_sites_MULTREG(clean=False):
                        this = self.datall_TDT_KS_slice_single_bysite(site, trial)
                        paththis = f"{path}/datslice_trial{trial}_site{site}.pkl"
                        with open(paththis, "wb") as f:
                            pickle.dump(this, f)
                       
    def _savelocalcached_check_done(self, datslice_quick_check=False): 
        """
        Check if all things done for saving local cached data. Returns True if all things have been saved
        """

        pathdir = self.Paths["cached_dir"]

        def _check_this(filename):
            path = f"{pathdir}/{filename}.pkl"
            return os.path.exists(path)

        # _CachedTrialOnset
        # files_check = ["trials_list", "trial_onsets", "strokes", "strokes_peanutsonly", "strokes_task", "touch_data", "dataset_beh"]
        files_check = ["trials_list", "trial_onsets", "strokes_task", "touch_data"] # 3/28 - reloading dataset every time. So ignore whether beh stuff is done.
        for x in files_check:
            if _check_this(x)==False:
                return False

        # Check that you got each trials list
        import pickle
        path = f"{self.Paths['cached_dir']}/trials_list.pkl"
        with open(path, "rb") as f:
            _CachedTrialsList = pickle.load(f)
            for k in [(False, True), (True, False), (True, True), (False, False)]:
                if k not in _CachedTrialsList.keys():
                    return False

        if self._savelocalcached_checksaved_datslice(datslice_quick_check=datslice_quick_check)==False:
            return False 

        return True


    def _savelocalcached_load(self):
        """
        Load from disk the cached data (quick). NOTE: all are indexed by "trial" neural, and therefore you must load those cached trial nums.
        """

        pathdir = self.Paths["cached_dir"]

        def _load_this(filename):
            path = f"{pathdir}/{filename}.pkl"
            with open(path, "rb") as f:
                out = pickle.load(f)
            return out

        # Related to neural data
        if True:
            self._CachedTrialsList = _load_this("trials_list")
            # # add (datset) to key.
            # tmp = {}
            # for k, v in self._CachedTrialsList.items():
            #     if len(k)==2:
            #         k = tuple([False] + list(k))
            #     tmp[k] = v
            # self._CachedTrialsList = tmp

            for a,b in ([(True, False), (False, True), (False, False), (True, True)]):
                if (a,b) not in self._CachedTrialsList.keys():
                    print(self._CachedTrialsList.keys())
                    assert False, "this might run into issues -- datslices use these cached trials. Prob mistake in prerpocessing --- run the load and save again."
        else:
            # 3/28/24 - Instead, load beh and regenrate these, since datset is reloaded eveyr time and was runnig into bugs.
            # NOTE: dont do this -- might lead to misalignemnet of trials, ie., not sure "trial" is uniwque
            self.load_behavior()
            for a,b in ([(True, False), (False, True), (False, False), (True, True)]):
                if (a,b) not in self._CachedTrialsList.keys():
                    self.get_trials_list(False, False)
                    self.get_trials_list(False, True)
                    self.get_trials_list(True, False)
                    self.get_trials_list(True, True)

        self._CachedTrialOnset = _load_this("trial_onsets") # diff between neural and beh on each trial.

        # Related to dataset beh
        if False:
            # Don't load this, inastead, code now just reads from Dataset directly.
            self._CachedStrokes = _load_this("strokes")
            self._CachedStrokesPeanutsOnly = _load_this("strokes_peanutsonly")
            self._CachedStrokesTask = _load_this("strokes_task")
        self._CachedTouchData = _load_this("touch_data")
        self._CachedStrokesTask_SANITY = _load_this("strokes_task") # For sanithc cehkc of mappoing dataset - neural.

        try:
            self.BehTrialMapListGood = _load_this("BehTrialMapListGood")
            self.BehTrialMapList = _load_this("BehTrialMapList")
            self.BehSessList = _load_this("BehSessList")
            self._MapperTrialcode2TrialToTrial = _load_this("_MapperTrialcode2TrialToTrial")
        except FileNotFoundError as err:
            # just skip this. old data, didnt save tehse caches.
            # assert False, "Missing cached data --> reextract!!"
            if False:
                # NOTE: This _should_ work (i compared the outcome to if dont), for
                # self.BehTrialMapList [(1, 0)
                # self.BehTrialMapListGood
                # self._MapperTrialcode2TrialToTrial!
                # self.BehSessList
                # BUT skip, since I'll just let the old code work, which did "pass"

                # Then renerate these by loading first behavioral data.
                self._beh_get_fdnum_trial_generate_mapper()
                self._generate_mappers_quickly_datasetbeh()
            else:
                pass

    def _savelocalcached_load_dataset_beh(self, dataset_version="raw"):
        """
        Load dataset beh and preprocess, imclding pruning beh and neral trials to match./
        Separating this from _savelocalcached_load so that the latter can run quickly, to load data useful for
        other loading steps.
        :param dataset_version:
        :return:
        """

        assert dataset_version=="raw", "always reload, to include latest things."
        if dataset_version=="cached":
            # Then load cached version
            paththis = f"{pathdir}/dataset_beh.pkl"
            with open(paththis, "rb") as f:
                self.Datasetbeh = pickle.load(f)
                self.Datasetbeh._cleanup_preprocess_each_time_load_dataset()
        elif dataset_version=="raw":
            # Load from scratch. This is useful if dataset beh has changed..
            # Don't run cleanup, since loading the dataset forces a cleanup.
            # This is default from 3/13/24 on
            datasetbeh_exptname = None # None was used orignally during extraction. This means use daily dataset.
            # self.datasetbeh_load_helper(self.DatasetbehExptname, FORCE_AFTER_MINIMAL=True)
            self.datasetbeh_load_helper(datasetbeh_exptname)

            # Clear cached beh data that might now be innacurate.
            self._CachedStrokesPeanutsOnly = {}
            self._CachedStrokes = {}
            self._CachedStrokesTask = {}
            if not DATASETBEH_CACHED_USE_BEHTOUCH:
                # This is not affected by dataset preprocessing, so can safely use the cached data
                self._CachedTouchData = {}

            # Remove events which are using old cached dataset
            self._events_prune_cached_which_use_datasetbeh()

        else:
            print(dataset_version)
            assert False

        if False: # Done above, during dataset loading
            self.Datasetbeh.LockPreprocess = True
            self._generate_mappers_quickly_datasetbeh()

        ################################
        # Sanity check alignement between dataset beh and neural data
        from pythonlib.tools.pandastools import _check_index_reseted
        # 1. Check dataset index formating
        _check_index_reseted(self.Datasetbeh.Dat)
        # 2. Most direct check of raw data -- time of go cue, sourced indepednently for the two datasests, so
        # this is a good test of match.
        gos1 = []
        gos2 = []
        for trial in self.get_trials_list(True):
            idx = self.datasetbeh_trial_to_datidx(trial)
            if idx is None:
                # Then this trial is not in dtaset
                print(trial)
                print(self.datasetbeh_trial_to_trialcode(trial))
                print(self.Datasetbeh.Dat["trialcode"].tolist())
                assert False, "dataset misising this trialcode.."
            assert self.datasetbeh_trial_to_trialcode(trial) == self.Datasetbeh.Dat.iloc[idx]["trialcode"]
            self.Datasetbeh.Dat.iloc[idx]["strokes_beh"]

            go1 = self.events_get_time_helper("go", trial, assert_one=True)[0]
            go2 = self.Datasetbeh.Dat.iloc[idx]["motorevents"]["go_cue"]

            assert go1-go2<1 # emprically, go1 tends to be ~0.15 earlier than go2. There is such high variance in gos across
            # trials, that a threshold of 1 will surely catch misalignment between neural and beh data.

            gos1.append(go1)
            gos2.append(go2)

        # if Mimmial loading, do sanity check that cached matches dataset
        if self._LOAD_VERSION == "MINIMAL_LOADING":
            # STRONG TEST -- check that strokes_task are identical between datsaet and neural data.
            for t in self.get_trials_list(True):
                idx = self.datasetbeh_trial_to_datidx(t)
                strokes_task_1 = self._CachedStrokesTask_SANITY[t]
                strokes_task_2 = self.Datasetbeh.Dat.iloc[idx]["strokes_task"]
                assert len(strokes_task_1)==len(strokes_task_2)
                for s1, s2 in zip(strokes_task_1, strokes_task_2):
                    assert np.all(s1==s2)
            # Now can delete..
            del self._CachedStrokesTask_SANITY

            #TODO: check match between self.EventsTimeUsingPhd and dataset...

        if False: # Debug, plotting distribution of go times.
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,2)
            axes.flatten()[0].plot(gos1, gos2, "xk")
            axes.flatten()[1].hist([g1-g2 for g1, g2 in zip(gos1, gos2)], bins=50)

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
                        # print("Loading... ", trial, site)
                    except Exception as err:
                        print(paththis)
                        raise err
                    self._CachedDatSlice[(trial, site)] = dat
                    return dat
                else:
                    # 3. Doesnt exist, return None.
                    return None

    def _savelocalcached_checksaved_datslice(self, datslice_quick_check=False):
        """ Returns bool, whether all trials/sites have their
        data extraacted. Doesnt check that the files are not corrupted..
        NOTE: this is more general. checks whether oyu have completed caching
        """

        if datslice_quick_check:
            # Then just check that the directory exists and is not empty
            path = f"{self.Paths['cached_dir']}/datall_site_trial"
            return checkIfDirExistsAndHasFiles(path)
        else:
            # First, check that you have cached trials. If not, 
            if not os.path.exists(f"{self.Paths['cached_dir']}/trials_list.pkl"):
                return False
            else:
                # load the trials
                import pickle
                path = f"{self.Paths['cached_dir']}/trials_list.pkl"
                with open(path, "rb") as f:
                    self._CachedTrialsList = pickle.load(f)

                    # If trials need to be skipped, then this wont work. so dont do it.
                    _, do_skip_trials = self._get_trials_list_skipped_trials(trials=[]) # enter dummy trials, as youjust want t know if this date has sipped trials...
                    if not do_skip_trials:
                        # Sanity check, that the trials in cached trials list are not incocnsitent with reloaded data.
                        # Should relaly just reconstruct each time from raw, since sometimes (rarely) there mistakes due
                        # to older data missing some trials (saw one case whwere those passing fixation success were empty)
                        self.load_behavior()
                        
                        trials = list(range(len(self.TrialsOffset)))
                        trials_missing = [t for t in trials if t not in self._CachedTrialsList[(False, True)]]
                        if len(trials_missing)>5 and len(trials_missing)/len(self._CachedTrialsList[(False, True)]) > 0.04:
                        # if not self._CachedTrialsList[(False, True)] == trials:
                            print("trials cached: ", self._CachedTrialsList[(False, True)])
                            print("trials reloaded: ", trials)
                            print("trials in reloaded that not exist in cahced: ", trials_missing)
                            assert False, "figure out why there is mismatch between cached trials and now-reloaded from raw (prob just rerun local caching)"

                        trials_pass_fixation = [t for t in trials if self.beh_fixation_success(t)]
                        trials_missing = [t for t in trials_pass_fixation if t not in self._CachedTrialsList[(True, True)]]
                        if len(trials_missing)>5 and len(trials_missing)/len(self._CachedTrialsList[(True, True)]) > 0.08:
                            # Is ok to miss a few, this can be explained by trials with no beh strokes matched to task.
                            print("trials cached: ", self._CachedTrialsList[(True, True)])
                            print("trials reloaded: ", trials_pass_fixation)
                            print("trials in reloaded that not exist in cahced: ", trials_missing)
                            assert False, "figure out why there is mismatch between cached trials and now-reloaded from raw (prob just rerun local caching)"

            # FOrce must_use_cached_trials, beecuase datslice are only saved for those cached trials.
            trials = self.get_trials_list(SAVELOCALCACHED_TRIALS_FIXATION_SUCCESS, must_use_cached_trials=True)
            sites = self.sitegetterKS_map_region_to_sites_MULTREG(clean=True)
            for t in trials:
                for s in sites:
                    exists = self._savelocalcached_loadextract_datslice(t, s, only_check_if_exists=True)
                    if not exists:
                        return False
            return True


    ########################################### KILOSORT LOADING OF SPIKETIMES
    def ks_plot_compare_tdt_to_kilosort(self, sdir):
        """
        Compare spikes for clusters (ks) matcjhed to their sites (tdt).
        Useful for sanity check of good clustering by ks.
        """

        assert self.SPIKES_VERSION=="kilosort"

        os.makedirs(sdir, exist_ok=True)

        try:
            self._IGNORE_SANITY_CHECKS = True

            ## Compare the tdt sites that were gotten

            # tdt spikes
            # sn.SPIKES_VERSION="tdt"
            list_sites_tdt = self.sitegetterKS_map_region_to_sites_MULTREG(force_tdt_sites=True)

            # any sites that have ks cluster but not kept in tdt?
            list_sites_ks = []
            for dat in self.DatSpikesSessByClust.values():
                list_sites_ks.append(dat["site"])

            list_sites_ks = sorted(set(list_sites_ks))
            print("Sites (KS):", list_sites_ks)

            print("KS sites not in tdt:")
            print([s for s in list_sites_ks if s not in list_sites_tdt])
            print(len([s for s in list_sites_ks if s not in list_sites_tdt]))

            print("TDT sites not in ks:")
            print([s for s in list_sites_tdt if s not in list_sites_ks])
            print(len([s for s in list_sites_tdt if s not in list_sites_ks]))

            ######## PLOT
            list_sites_both = sorted(set(list_sites_tdt + list_sites_ks))
            trials = self.get_trials_list(True)[::10]
            for site_tdt in list_sites_both:
                fig, axes = plt.subplots(4,1, figsize=(10, 18))

                # tdt
                self.SPIKES_VERSION="tdt"
                ax = axes.flatten()[0]
                self.plot_raster_trials(ax, trials, site_tdt, alignto="go", xmin=-3, xmax=7, alpha_raster=0.5)
                ax.set_ylabel("TDT")

                # ks
                self.SPIKES_VERSION="kilosort"
                list_clust = self.ks_convert_sites_to_clusts([site_tdt])
                for i, clust in enumerate(list_clust):
                    ax = axes.flatten()[i+1]
                    label = self.DatSpikesSessByClust[clust]["label_final"]
                    print(label)
                    self.plot_raster_trials(ax, trials, clust, alignto="go", xmin=-3, xmax=7, alpha_raster=0.5)
                    ax.set_ylabel(f"KILOSORT - {label}")

                fig.savefig(f"{sdir}/site_tdt_{site_tdt}.png")
                plt.close("all")

        except Exception as err:
            self._IGNORE_SANITY_CHECKS = False
            self.SPIKES_VERSION = "kilosort"
            raise err

        self._IGNORE_SANITY_CHECKS = False
        self.SPIKES_VERSION = "kilosort"

    def ks_convert_sites_to_clusts(self, list_site, sort_by_clust=False):
        """ Return list of clulster ids for this l ist of sites,
        PARAMS;
        - sort_by_clust. (so they lose ordering wrt sites)
        """
        clusts = []
        for s in list_site:
            clusts.extend(self._MapperKsortSiteToClustids[s])

        if sort_by_clust:
            clusts = sorted(clusts)
        
        return clusts

    def spiketimes_ks_check_if_exists(self):
        """ Returns True if finalized (curated) kilosort
        data exists, i.e, on server, and if the num sessions used in ks concatenation matches
        the num neural rec sessions -- ie can find alignment times.
        """
        from os.path import isfile

        # Check that files exust
        BASEDIR = "/mnt/Freiwald/kgupta/neural_data/postprocess/final_clusters/"
        clusters_final = f"{BASEDIR}/{self.Animal}/{self.Date}/DATSTRUCT_CLEAN_MERGED.mat"
        A = isfile(clusters_final)

        if A:
            # Check that the num sessions concated to make ks matches the num sessions you used for rs4. If not then I havent coded
            # up how to match neural session to onset times of ks.
            from neuralmonkey.utils.directory import rec_session_durations_extract_kilosort
            from pythonlib.tools.exceptions import NotEnoughDataException
            try:
                rec_session_durations_extract_kilosort(self.Animal, self.Date)
                B = True
            except NotEnoughDataException as err:
                B = False
        else:
            B = False

        return A and B

    def spiketimes_ks_compute_timeglobal_thissession_start(self):
        """ Return the global time (i.e. after concatting multiple sessions)
        that the rec for this rec session srtarted, based on log files in RS4 data stores.
        RETURNS:
            - {rs:onset}
        """
        from neuralmonkey.utils.directory import rec_session_durations_extract_kilosort
        # time_onset_this_session = out["onsets"][self.RecSession]
        # return time_onset_this_session
        # return out["onsets_using_rs4_each_rs"]
        out = rec_session_durations_extract_kilosort(self.Animal, self.Date)
        return {rs:onsets[self.RecSession] for rs, onsets in out["onsets_using_rs4_each_rs"].items()}

    def spiketimes_ks_load(self, PRINT=False, SANITY=True):
        """ Load all kilosort data fro this session. This is manually curated and
        finalized (using kspostprocess module).
        PARAMS;
        RETURNS:
        - stores in self.DatSpikesSessByClust, map from clust_id_global to dict of
        data kept exactly as in the kilosort data
        EXAMPLE:
        - self.DatSpikesSessByClust[0]
            {'GOOD': True,
             'Q': array(0.00462743),
             'RSn': array(2.),
             'amps_wf': array([2007., 2508., 2333., 2115., 2234., 2455., 2319., 2419., 2362.,
                    ..., 
                    3017., 4075., 3250., 3849., 3075.]),
             'batch': array(1.), # 1 -indexed
             'chan': array(2.), # 1 -indexed (e..g, 1-64)
             'chan_global': array(2.), # 1 -indexed (=site)
             'clust': array(0.), # IGNORE
             'clust_before_merge': array(nan),
             'clust_group_id': array(2.),
             'clust_group_name': 'good',
             'index': array(1.),
             'index_before_merge': array(nan),
             'isbimod': False,
             'isi_violation_pct': array(0.00018474),
             'label_final': 'su',
             'label_final_int': array(2.),
             'sharpiness': array(6.45553999),
             'snr_aligned': array(8.26876623),
             'snr_final': array(8.61565813),
             'snr_not_aligned': array(8.61565813),
             'times_sec_all': array([6.67115588e-01, 3.08170784e+00, 3.08715552e+00, ...,
                    1.02175754e+04, 1.02187145e+04, 1.02187212e+04]),
             'times_sec_all_BEFORE_REMOVE_DOUBLE': None,
             'waveforms': array([[ 358.,  182.,  -14., ..., -300.,  -56., -302.],
                    [ 296.,  431.,  436., ..., -108., -334., -180.],
                    ...,
                    [ 187.,  139.,    3., ..., -154., -410., -271.],
                    [  72.,   52.,  258., ..., -236., -586., -386.]])}
        """

        assert self.SPIKES_VERSION=="tdt", "if not, might have circular error, as assues that sites means tdt site"
        CLUST_STARTING_INDEX = 1000 # So that site nums for ks will not get confused with TDT nums
        # whihc end at 512

        import mat73
        # import scipy.io as sio
        # import scipy
        # import zlib
        BASEDIR = "/mnt/Freiwald/kgupta/neural_data/postprocess/final_clusters/"
        clusters_final = f"{BASEDIR}/{self.Animal}/{self.Date}/DATSTRUCT_CLEAN_MERGED.mat"

        ## Load all data
        # DATSTRUCT = sio.loadmat(clusters_final)
        res = mat73.loadmat(clusters_final, use_attrdict=True)
        DATSTRUCT = res["DATSTRUCT"]

        # Sanity check: DATSETUCT is dict of lists, each len num clust
        n = None
        for k,v in DATSTRUCT.items():
            if n is not None:
                assert len(v) == n
            n = len(v)

        #### SPIKE TIMES --> SUBTRACT duration in previous sessions, so that the start of this session
        # corresponds to timepoint 0. THis only matters for sessions 1+.
        # get time of this session onset
        # time_global_start_of_this_session = self.spiketimes_ks_compute_timeglobal_thissession_start()
        onsets_using_rs4_each_rs = self.spiketimes_ks_compute_timeglobal_thissession_start()

        # assert isinstance(DATSTRUCT["times_sec_all"], list), "i am assuming this... for this code below."
        list_times = []
        list_clustinds_without_spikes = []
        for i, (times, rs) in enumerate(zip(DATSTRUCT["times_sec_all"], DATSTRUCT["RSn"])):
            times = times - onsets_using_rs4_each_rs[int(rs)]
            times_good = times[times>=0]
            if len(times_good)==0:
                # This probably means you lost this cluster some point during day, and
                # so this session has no spikes. Solution: Exclude this cluster
                list_clustinds_without_spikes.append(i)
                # Allow to pass, then count how many clust this hapens to.
                # print([(k, v[i]) for k, v in DATSTRUCT.items()])
                # print(times)
                # print(rs)
                # print(onsets_using_rs4_each_rs[int(rs)])
                # self.print_summarize_expt_params()
                # assert False, "why no spikes"
            list_times.append(times_good)
        DATSTRUCT["times_sec_all"] = list_times

        # Did any clsuters not have spikes? Is ok if is rare...
        assert len(list_clustinds_without_spikes)<5, "why so many clusts are mnimssing spikes?"
        assert len(list_clustinds_without_spikes)/len(DATSTRUCT["times_sec_all"]) < 0.1, "why so many clusts are mnimssing spikes?"

        if False: # To look at all clusts that did ntot have spikes.
            if len(list_clustinds_without_spikes)>0:
                print("SOME CLUSTERS DID NOT HAVE SPIKES WITHIN THIS SESSION!! is ok, will skip them")
                print(len(list_clustinds_without_spikes), list_clustinds_without_spikes)
                print("n clusts exist: ", len(DATSTRUCT["times_sec_all"]))
                for clust in list_clustinds_without_spikes:
                    times = (DATSTRUCT["times_sec_all"][clust])
                    rs = (DATSTRUCT["RSn"][clust])
                    print(" ---- clust num: ", clust)
                    print(onsets_using_rs4_each_rs[int(rs)])
                    print(times)
                assert False

        # Some metadata
        nbatches = int(np.max(DATSTRUCT["batch"]))
        chans_per_batch = 256/nbatches
        # rs =3
        # batch_1indexed = 4
        # chan_within_batch_1indexed=64
        # 256*(rs-2) + chans_per_batch*(batch_1indexed-1) + chan_within_batch_1indexed

        ## Extract into format of dict[clustnum]
        self.DatSpikesSessByClust = {}

        # keys_extract = [k for k in DATSTRUCT.keys() if k not in ["chan_global", "clust"]]
        keys_extract = DATSTRUCT.keys()
        keys_exclude = ["GOOD", "clust", "clust_before_merge", "clust_group_id", "clust_group_name",
            'index', 'index_before_merge', 'times_sec_all_BEFORE_REMOVE_DOUBLE', 'waveforms']
        keys_convert_to_int = ["RSn", "batch", "chan", "chan_global", "label_final_int"]

        # index by a global cluster id
        clustid_glob = 0 + CLUST_STARTING_INDEX - 1
        for ind in range(len(DATSTRUCT["times_sec_all"])):
            
            # IMPORTANT to start here, so that across sessions the mapping frmo clustid_glob to index in DATSRUCT is maintained.
            # (the continue can fuck things up)
            clustid_glob+=1
            
            if len(DATSTRUCT["times_sec_all"][ind])==0:
                # Then this clust had no times within this session. Skip it.
                assert ind in list_clustinds_without_spikes, "sanity check failed"
                continue

        #     print("-------------------------------------")
        #     for k in DATSTRUCT.keys():
        #         print(k, " -- ", DATSTRUCT[k][ind])
            if PRINT:
                print("-------------------------------------")
                tmp = [DATSTRUCT[k][ind] for k in ["RSn", "batch", "chan", "chan_global", "clust", "label_final", "label_final_int"]]
                print(tmp)
                    
            if False:
                # new key:
                site = int(DATSTRUCT["chan_global"][ind])
                clust = int(DATSTRUCT["clust"][ind])
                assert site == DATSTRUCT["chan_global"][ind]
                assert clust == DATSTRUCT["clust"][ind]
                key = (site, clust)
            else:
                key = clustid_glob
            
            self.DatSpikesSessByClust[key] = {}
            for k in DATSTRUCT.keys():
                if k not in keys_exclude:
                    if k in keys_convert_to_int:
                        self.DatSpikesSessByClust[key][k] = int(DATSTRUCT[k][ind])
                    else:
                        self.DatSpikesSessByClust[key][k] = DATSTRUCT[k][ind]


            # Rename things to common scheme
            self.DatSpikesSessByClust[key]["clust_id"] = clustid_glob
            
            self.DatSpikesSessByClust[key]["spike_times"] = self.DatSpikesSessByClust[key]["times_sec_all"]
            del self.DatSpikesSessByClust[key]["times_sec_all"]

            self.DatSpikesSessByClust[key]["site"] = self.DatSpikesSessByClust[key]["chan_global"]
            del self.DatSpikesSessByClust[key]["chan_global"]

            ## Sanity checks
            chan_within_batch_1indexed = int(DATSTRUCT["chan"][ind])
            rs = int(DATSTRUCT["RSn"][ind])
            batch_1indexed = int(DATSTRUCT["batch"][ind])
            chan_within_batch = int(DATSTRUCT["chan"][ind]) # e.g, 1-64
            
            # convert to global chan
            site = 256*(rs-2) + chans_per_batch*(batch_1indexed-1) + chan_within_batch_1indexed
            chan_within_rs = chans_per_batch*(batch_1indexed-1) + chan_within_batch_1indexed

            self.DatSpikesSessByClust[key]["chan_within_batch_1indexed"] = chan_within_batch_1indexed
            self.DatSpikesSessByClust[key]["chan_within_rs"] = chan_within_rs
            self.DatSpikesSessByClust[key]["batch_1indexed"] = batch_1indexed
            self.DatSpikesSessByClust[key]["chan_within_batch"] = chan_within_batch

            if SANITY:
                
                # saved global chan
                site_saved = int(DATSTRUCT["chan_global"][ind])
                # identity
                assert site == site_saved
                # (rs, chan) == site
                assert self.convert_rschan_to_site(rs, chan_within_rs) == site
            
            # increment clustic
            # clustid_glob+=1

    def spiketimes_ks_extract_alltrials(self, cluster_labels_keep=("su", "mua")):
        """
        Kilosort - Extract and save trial-windowed spike times for all clusters
        PARAMS:
        - cluster_labels_keep, if not None, then keeps only clusters that have lavel in this list.
        RETURNS:
        - self.DatSpikesSliceClustTrial, dict indexed by (clust, trial), returning
        inner dict like this:"
            {'spike_times': array([-0.95138138,  3.05266383,  7.61773822,  7.62076926,  7.65222654,
                     7.70617087]),
             'time_dur': 12.241269759999994,
             'time_on': -1.0123358,
             'time_off': 11.228933959999994,
             'trial0': 0,
             'pre_dur': 1.0,
             'post_dur': 1.0,
             'spiketrain': None,
             'site': 2,
             'clustnum_glob': 0}
        - self._MapperKsortSiteToClustids, map from site(1-512) to list of clust ids

        TODO: Currently for sess>0 sometimes like1-2ms mmisalignement between ks spike times and spike wavefporms.
        Should solve this by either (i) forcing alignement to tdt spikes (ii) figureing out more carefully why this happen,
        my guesss is that tank and rs4 are slightly off. timestamps from from tank, so this would expalin the problem
        (solution: read the dupl from tank and do align of neural to dupl. OR see if duration of dupl or other trace
        reveals the alignment problem). ACTUALLY doesnt seem liek this can expalin, since tank durations are usually
        very diff (~100ms) from RS4...
        Can see this if run

            twind_plot = None
            twind_plot = [0.2, 0.25]
            sn_KS.plot_raw_overlay_spikes_on_raw_filtered(site, trial, twind_plot)
            sn_KS.ks_plot_compare_overlaying_spikes_on_raw_filtered(site, trial)

        """
        assert self.SPIKES_VERSION=="tdt", "if not, might have circular error, as assues that sites means tdt site"

        # [Extraction] for each clust and trial, extract and save
        list_clust = self.DatSpikesSessByClust.keys()
        list_trial = self.get_trials_list(True)

        DatSliceAll = {}
        self._MapperKsortSiteToClustids = {}

        for clust in list_clust:
            print(clust)

            # - all spikes for this sess
            datthis = self.spiketimes_ks_slice_session(clust)
                
            if datthis["label_final"] not in cluster_labels_keep:
                continue

            # save map from site(1-512) to clustids
            site = datthis["site"] 
            if site in self._MapperKsortSiteToClustids.keys():
                self._MapperKsortSiteToClustids[site].append(clust)
            else:
                self._MapperKsortSiteToClustids[site] = [clust]

            for trial in list_trial:
                
                # Given a trial, extract slice
                spike_times, time_dur, time_on, time_off = self._datspikes_slice_single(datthis["spike_times"], trial)
                # assert len(spike_times)>0, "is it possible, not a single spike?"

                # save this (trial, clust) combo
                key = (clust, trial)
                DatSliceAll[key] = {}                

                DatSliceAll[key]["spike_times"] = spike_times 
                DatSliceAll[key]["time_dur"] = time_dur
                DatSliceAll[key]["time_on"] = time_on
                DatSliceAll[key]["time_off"] = time_off 
                DatSliceAll[key]["trial0"] = trial
                DatSliceAll[key]["pre_dur"] = PRE_DUR_TRIAL
                DatSliceAll[key]["post_dur"] = POST_DUR_TRIAL
        #         datslice["time_range"] 
                DatSliceAll[key]["spiketrain"] = None # to extract later
                DatSliceAll[key]["site"] = datthis["site"]

                # Cluster related infor
                DatSliceAll[key]["clustnum_glob"] = clust
                DatSliceAll[key]["label_final"] = datthis["label_final"]
                DatSliceAll[key]["snr_final"] = datthis["snr_final"]
                DatSliceAll[key]["batch_1indexed"] = datthis["batch_1indexed"]
                DatSliceAll[key]["chan_within_batch"] = datthis["chan_within_batch"]

                # DatSliceAll.append(datslice)
                
        #         # save the index
        #         index = len(DatSliceAll)-1
        #         if (site, trial) not in self._MapperSiteTrial2DatAllInd.keys():
        #             self._MapperSiteTrial2DatAllInd[(site, trial)] = index
        #         else:
        #             assert self._MapperSiteTrial2DatAllInd[(site, trial)] == index

        self.DatSpikesSliceClustTrial = DatSliceAll

        # Make sure all sites are represented in mapper, even if they dont have any clusters.
        sites = self.sitegetterKS_map_region_to_sites_MULTREG(clean=False)
        mapper = {}
        for s in sites:
            if s in self._MapperKsortSiteToClustids.keys():
                mapper[s] = self._MapperKsortSiteToClustids[s]
            else:
                mapper[s] = []
        self._MapperKsortSiteToClustids = mapper


    def spiketimes_ks_slice_session(self, clust_id=None, site=None):
        """
        RETURNS:
        - an inner dict holding data for this clustid
        """

        if clust_id is None:
            # then index by site
            assert False, "code it"
        else:
            dat = self.DatSpikesSessByClust[clust_id]
            assert dat["clust_id"] == clust_id
            return dat  

    def spiketimes_bin_counts(self, spike_times, t_start, t_end, bin_size, plot=False, 
                              assert_all_times_within_bounds=True):
        """
        Bin spike times into counts per bin.

        Parameters
        ----------
        spike_times : array-like
            Array of spike times (in seconds or ms, as long as consistent with t_start, t_end, bin_size).
        t_start : float
            Start time of the window.
        t_end : float
            End time of the window.
        bin_size : float
            Bin size.

        Returns
        -------
        counts : np.ndarray
            Array of spike counts for each bin.
        bin_edges : np.ndarray
            The edges of the bins (length = len(counts) + 1).
        bin_centers : np.ndarray
            The centers of the bins (length = len(counts)).
        """
        
        if assert_all_times_within_bounds and len(spike_times)>0:
            assert np.min(spike_times) >= t_start, "Spike times should be >= t_start"
            assert np.max(spike_times) <= t_end, "Spike times should be <= t_end"

        bin_edges = np.arange(t_start, t_end+0.001, bin_size)
        counts, _ = np.histogram(spike_times, bins=bin_edges)
        bin_centers = bin_edges[:-1] + bin_size / 2 # Center of each bin
        assert len(bin_centers)==len(counts)

        if not len(spike_times)==np.sum(counts):
            print(spike_times)
            print(counts)
            print(bin_centers)
            assert False

        if plot:
            if False:
                print("Spike counts:", counts)
                print("Bin edges:", bin_edges)

            # Plot the spike counts histogram:
            plt.figure(figsize=(10, 5))
            plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', edgecolor='black')
            plt.xlabel('Time (s)')
            plt.ylabel('Spike counts')
            plt.title('Binned Spike Counts')

            # Overlay the spikes
            plt.eventplot(spike_times, orientation='horizontal', color='red')

            plt.show()
        return counts, bin_edges, bin_centers

    ########################################### TDT LOADING
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


    def load_raw(self, rss, chans, trial0, pre_dur=PRE_DUR_TRIAL, post_dur = POST_DUR_TRIAL,
            get_raw=True):
        """ Get all raw data across channels for this trial
        PARAMS:
        - get_raw, bool( True), whether to actually load. if False, then makes teh
        hodler but doesnt load. 
        """

        for ch in chans:
            assert ch <=256, "Inpout chan (1-256)"
        TIME_ML2_TRIALON = self.ml2_get_trial_onset(trialtdt = trial0)
        def extract_raw_(rs, chans, trial0):
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
            datraw = extract_raw_(rs, chans, trial0)
            DatRaw.extend(datraw)
        return DatRaw


    ###################### EXTRACT SPIKES (RAW, PRE-CLIUSTERED)
    def load_spike_waveforms(self, site):
        """
        Helper to load spike waveform from raw. 
        Might be obsolete.
        """
        rs, chan = self.convert_site_to_rschan(site)
        assert self.SPIKES_VERSION=="tdt", "make this auto detect whtehr get tdt or ks. if get ks, then have to extract from raw. See Paolo Emilio stuff."
        self.load_spike_waveforms_(rs, chan)

    def load_spike_waveforms_(self, rs, chan, ver="spikes_tdt_quick"):
        """ Return spike waveforms, pre-extracted
        """
        site = self.convert_rschan_to_site(rs, chan)

        # Decide if extract from saved
        if site not in self.DatSpikeWaveforms.keys():
            import zlib
            import scipy.io as sio

            if self.Paths['spikes'] is None:
                print("+++++++++++++++=")
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
                print("Loading file: ", fn)
                mat_dict = sio.loadmat(fn)
                if "snips" not in mat_dict.keys():
                    print(mat_dict)
                    print("Redo extraction of spikes?..")
                    raise err
                waveforms = mat_dict["snips"]
            except zlib.error as err:
                print("[scipy error] failed load_spike_waveforms_ for (rs, chan): ", rs, chan)
                self.print_summarize_expt_params()
                # waveforms = None
                raise err
            except Exception as err:
                print(err)
                print("[load_spike_waveforms] Failed for this rs, chan: ",  rs, chan)
                print(fn)
                self.print_summarize_expt_params()
                raise err
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
        RETURNS
        - times, vals or None, None if doesnt exist, and is allowed to pass.
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
        
        keys_which_might_be_missing = ["Tff_"]
        if key not in self.DatTank["epocs"].keys():
            if key in keys_which_might_be_missing:
                # If is expected to be missing sometimes (bug in Synapse,,,)
                # This means there was no data (e.g., loose BNC).
                return None, None   
            else:
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
        get_raw=False, get_spikes=True, save=True):
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
                self._extract_raw_and_spikes(rss, chans, t, get_raw=get_raw, get_spikes=get_spikes)

            # generate mapper, slice each one and this will autoamtically extract
            self.mapper_extract("sitetrial_to_datallind", save=save)
            
            # Save
            if save:
                self._savelocal_datall()


    def _extract_raw_and_spikes(self, rss, chans, trialtdt, get_raw = False, get_spikes=True):
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

        if get_spikes:
            # spikes (from pre-extracted spikes)
            for i, d in enumerate(DatRaw):
                spike_times, time_dur, time_on, time_off = self.datspikes_slice_single(d["rs"], d["chan"], trial0=trialtdt)
                d["spike_times"] = spike_times
                d["time_dur"] = time_dur
                d["time_on"] = time_on
                d["time_off"] = time_off


        def _dat_replace_single(self, rs, chan, trial0, Dnew):
            """ If this (rs, chan, trial0) exists, replace it with Dnew
            MODIFIES:
            - self.DatAll, at a single index
            """

            Dold, idx = self._datallTDT_slice_single(rs, chan, trial0, return_index=True)
            if Dold is None:
                # then doesnt exist, do mnothing
                pass
            else:
                self.DatAll[idx] = Dnew


        # change name, since is both raw and spikes
        if self.DatAll is None:
            self.DatAll = DatRaw
        else:
            for d in DatRaw:
                site = self.convert_rschan_to_site(d["rs"], d["chan"])
                trial = d["trial0"]

                if (site, trial) in self._MapperSiteTrial2DatAllInd.keys():
                    # Then is already done
                    d_old = self.datall_TDT_KS_slice_single_bysite(site, trial)
                    if len(d_old["raw"])==0 and get_raw:
                        # Then previous didnt have raw, so overwrite it
                        self._dat_replace_single(d["rs"], d["chan"], trial, Dnew=d)
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
    def debug_event_photodiode_detection(self):
        assert False, "just notes here"
        t = 348
        alignto = "first_raise"
        sn = MS.SessionsList[0]
        sn.events_get_time_using_photodiode(t, list_events=[alignto], do_reextract_even_if_saved=True, plot_beh_code_stream=True)

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
    def _timewindow_window_this_data(self, times, twind, vals=None, recompute_time_rel_onset=True, time_to_add=0.):
        return self._extract_windowed_data(times, twind, vals, recompute_time_rel_onset, time_to_add)

    def _extract_windowed_data(self, times, twind, vals=None, recompute_time_rel_onset=True, time_to_add=0.):
        """ Prune data (time, vals) to include only those with
        times within twind. Also changes times to be relative to twind[0]. Does not do anything clever!
        NOTE: This will NOT be accurate if you want to get relative tiems for trials. To do so,
        use extract_windowed_data_bytrial
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
            recompute_time_rel_onset=True, pre_dur=PRE_DUR_TRIAL, post_dur=POST_DUR_TRIAL): 
        """ Given generic data, window it by a given trial.
        Prune data (time, vals) to include only those with
        times within twind. Also changes times to be relative to trial onset (regardless of pre_dur
        PARAMS:
        - pre_dur, post_dur, time extra to extract. NOTE: doesnt affect what is called 0, which is always trial onset
        """

        assert recompute_time_rel_onset==True, "some code assumes this true. I dont see any reason to change this. if want to change, then make new function that does this"
        assert PRE_DUR_TRIAL==1 and POST_DUR_TRIAL==1, "all data has this hard coded;.."
        # Get window
        t1, t2 = self.extract_timerange_trial(trial0, pre_dur, post_dur)
        TIME_ML2_TRIALON = self.ml2_get_trial_onset(trialtdt = trial0)
        time_to_add = -pre_dur # so that is zeroed on time of trial onset
        
        # shift all tdt things so that by definition the time of beh code 9 are identical between tdt and ml2
        time_to_add = time_to_add - TIME_ML2_TRIALON

        if len(times)>0:
            times, vals, time_dur, time_on, time_off = self._extract_windowed_data(times, [t1, t2], vals, recompute_time_rel_onset, time_to_add = time_to_add)
        else:
            time_dur, time_on, time_off = self._extract_windowed_data_get_time_bounds([t1, t2], recompute_time_rel_onset, time_to_add = time_to_add)

        return times, vals, time_dur, time_on, time_off

    def extract_timerange_trial(self, trial0, pre_dur=PRE_DUR_TRIAL, post_dur=POST_DUR_TRIAL):
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
    # def datall_this_exists(self, rs, chan, trial0, also_check_if_has_raw_data=False):
    #     """ returns True if this combo exist sin self.DatAll,
    #     false otherwise
    #     - also_check_if_has_raw_data, bool, if true, then to return True must also 
    #     hvae raw data extracted
    #     """
    #     assert False, "use datall_slice_single_bysite for kilosort"
    #     D = self._datall_slice_single(rs, chan, trial0)
    #     if D is None:
    #         return False
    #     if also_check_if_has_raw_data:
    #         if len(D["raw"])==0:
    #             return False
    #         else:
    #             return True
    #     else:
    #         return True



    def datall_TDT_KS_slice_single_bysite(self, site_or_clust, trial0, return_index=False,
                                          IGNORE_SANITY_CHECKS = False):
        """ [KILOSORT OR TDT WORKS] 
        This is the only place where spikes data are extracted for analysis!!!
        Like datall_slice_single, but input site instead of (rs, chan)
        """

        if not self._IGNORE_SANITY_CHECKS:
            assert self.SPIKES_VERSION == self._SPIKES_VERSION_INPUTED, "not allowed to change spikes version once you load! (Instead, load a new session with flag spikes_version changed"

        if self.SPIKES_VERSION=="tdt":
            rs, chan = self.convert_site_to_rschan(site_or_clust)
            return self._datallTDT_slice_single(rs, chan, trial0, return_index)
        elif self.SPIKES_VERSION=="kilosort":
            key = (site_or_clust, trial0)
            return self.DatSpikesSliceClustTrial[key]
        else:
            print(self.SPIKES_VERSION)
            assert False, "Code it!"

    def _datallTDT_slice_single(self, rs, chan, trial0, return_index=False, method="new"):
        """ [TDT] Slice a single chans data.
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
        
    def _datspikes_slice_single(self, spiketimes_sess, trial0=None, twind=None):
        """ Slice a single chans spike times, 
        optionally relative to a time window
        PARAMS:
        - spiketimes, (nspikes, ) array, in seconds, across entire session
        - trial0, int, trial to reslice and reform time base to. 
        - twind, (2,) array, time window to slice out.
        NOTE: use trial0 if want time base to match up so that behcode 9 is 0 sec.
        NOTE: can only pass in one of trial0 or twind
        RETURNS:
        spiketimes, time_dur, time_on, time_off
        """

        if twind is not None:
            assert trial0 is None
            spiketimes, _, time_dur, time_on, time_off = self._extract_windowed_data(spiketimes_sess, twind)
        elif trial0 is not None:
            assert twind is None
            spiketimes, _, time_dur, time_on, time_off = self.extract_windowed_data_bytrial(spiketimes_sess, trial0)    
        else:
            assert False, "must be one or the other"
        return spiketimes, time_dur, time_on, time_off
                
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
                    print(self.print_summarize_expt_params())
                    assert False, "figure out why this is None. probably in loading (scipy error) and I used to replace errors with None. I should re-extract the spikes."
                
                return self._datspikes_slice_single(spiketimes, trial0, twind)
                
        print(rs, chan)
        assert False, 'this combo of rs and chan doesnt exist in DatSpikes!'
        

    ####################### HELP CALC THINGS
    def dat_to_time(self, vals, fs):
        return dat_to_time(vals, fs)

    ####################### CONVERSIONS BETWEEN BEH AND NEURAKL
    def _beh_prune_trial_offset_onset(self):
        """
        Prunes self.TrialsOffset or self.TrialsOnset to make sure they are
        the same length. computes trial times to ensur eit makes sense.
        """

        if not len(self.TrialsOffset)==len(self.TrialsOnset):

            offs = np.array(self.TrialsOffset)
            ons = np.array(self.TrialsOnset)

            if len(self.TrialsOnset)==len(self.TrialsOffset)-1:
                # one extract offset...

                diffs1 = offs[1:] - ons
                mean1 = np.mean(diffs1)
                std1 = np.std(diffs1)

                diffs2 = offs[:-1] - ons
                mean2 = np.mean(diffs2)
                std2 = np.std(diffs2)

                if mean1>0 and mean2<0:
                    self.TrialsOffset = self.TrialsOffset[1:]
                elif mean1<0 and mean2>0:
                    self.TrialsOffset = self.TrialsOffset[:-1]
                else:
                    print(mean1, std1, mean2, std2)
                    assert False

            elif len(self.TrialsOnset)==len(self.TrialsOffset)+1:
                # one extra onset...

                diffs1 = offs - ons[1:]
                mean1 = np.mean(diffs1)
                std1 = np.std(diffs1)
                min1 = np.min(diffs1)
                max1 = np.max(diffs1)

                diffs2 = offs - ons[:-1]
                mean2 = np.mean(diffs2)
                std2 = np.std(diffs2)
                min2 = np.min(diffs2)
                max2 = np.max(diffs2)

                if mean1>0 and mean2<0:
                    self.TrialsOnset = self.TrialsOnset[1:]
                elif mean1<0 and mean2>0:
                    self.TrialsOnset = self.TrialsOnset[:-1]
                else:
                    if max1<30 and min1>1 and min2>5 and max2>120:
                        # Theese are so unliekly unless is this alignemnt
                        # Eg these are times for on and off:
                            # [ 30.04301312 163.7029888  176.922624   188.17630208 205.02122496]
                            # [173.53125888 185.73729792 201.29517568 220.50480128 235.91272448]
                        self.TrialsOnset = self.TrialsOnset[1:]
                    elif max2<30 and min2>1 and min1>5 and max1>120:
                        # Theese are so unliekly unless is this alignemnt
                        self.TrialsOnset = self.TrialsOnset[:-1]
                    else:
                        print(mean1, std1, mean2, std2)
                        print(min1, max1, min2, max2)
                        print(ons[:5])
                        print(offs[:5])
                        assert False
            else:
                print("-----")
                print(len(self.TrialsOnset))
                print(len(self.TrialsOffset))
                print(len(self.TrialsOffset)+1)
                print(len(self.TrialsOffset)==len(self.TrialsOffset)+1)
                print(len(self.TrialsOffset)==len(self.TrialsOffset)-1)
                print("-----")
                print(self.TrialsOnset)
                print(self.TrialsOffset)
                assert False, "what to do?"

        assert len(self.TrialsOffset)==len(self.TrialsOnset)


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

        if False: # Tried this hacky, but still failed... (for loading and saving locally).
            if (int(self.Date))==220610 and self.RecSession==0 and self.Animal=="Pancho":
                # HACKY - I restarted ML2 after around 40 trails, but kept recording in. Need
                # to throw out those trials...
                if all([t in trials_all for t in trials_exist_in_ml2]):
                    print("_beh_validate_trial_number passed!!")
                return None

        if not trials_all==trials_exist_in_ml2:
            # check whether this session is allowed to fail this.
            from ..utils.monkeylogic import _load_sessions_corrupted
            sessdict = _load_sessions_corrupted()
            value = (int(self.Date), self.RecSession)
            
            if self.Animal in sessdict.keys() and value in sessdict[self.Animal]:
                # then ok, expect to fail
                print("_beh_validate_trial_number failed, but OK becuase is expected!!")
                print("**&*&**")
                print("trials in neural data:", trials_all)
                print("trials_exist_in_ml2:", trials_exist_in_ml2)
                print("neural trials that miss beh data:", [t for t in trials_all if t not in trials_exist_in_ml2])
                print("beh trials that miss neural data:", [t for t in trials_exist_in_ml2 if t not in trials_all])
            else:
                print("**&*&**")
                print("trials in neural data:", trials_all)
                print("trials_exist_in_ml2:", trials_exist_in_ml2)
                print("neural trials that miss beh data:", [t for t in trials_all if t not in trials_exist_in_ml2])
                print("beh trials that miss neural data:", [t for t in trials_exist_in_ml2 if t not in trials_all])
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
        
        # all ml2 trials
        _trials = mkl.getIndsTrials(fd)
        _durs = []
        for t in _trials:
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
            _durs.append(dur)


        # do running subtractio
        if False: # was debuggin with this, but not needed and it didnt work anyway
            print("SAVING LAGS!!")
            print(f"/tmp/lags-sess{self.RecSession}.pdf")
            list_std = []
            list_lags = []
            for i in range(len(_durs)-len(durs_exist)):
                print("testing lag ", i)
                durs_ml2_tmp = _durs[i:i+len(durs_exist)]
                list_std.append(np.std(np.array(durs_exist) - np.array(durs_ml2_tmp)))
                list_lags.append(i)
            fig, ax = plt.subplots(1,1)
            ax.plot(list_lags, list_std, '-x')
            print("lag - std")
            fig.savefig(f"/tmp/lags-sess{self.RecSession}.pdf")
            for a, b in zip(list_lags, list_std):
                print(a, b)


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
            print("____PLOTTING!")
            """ Quick plots for diagnostic. plots overlays trial durations vs. trials"""
            fig, axes = plt.subplots(2,2,figsize=(15,8))

            ax = axes.flatten()[0]
            ax.plot(trials_exist, durs_exist, '-xk', label="tdt")
            ax.plot(trials_exist, durs_exist_ml2, '-xr', label="ml2(mapped from tdt trials)")
            ax.set_title('Goal: k and r should overlap')
            ax.set_ylabel('trial durations')
            ax.set_xlabel('trials(tdt)')
            ax.legend()   

            # plot all beh trials (not just those matching neural.
            ax = axes.flatten()[1]
            ax.plot(_trials, _durs, '-xk', label="ml2_all")
            ax.plot(trials_exist_ml2, durs_exist_ml2, '-gx', alpha=0.5, label="ml2(ml2 trials)")
            ax.legend()

            # DEBUG, hand plot specific items to find what works
            if False:
                ax = axes.flatten()[2]
                ax.plot(_trials[600:], _durs[600:], '-xk', label="ml2_all")
                # ax.plot(trials_exist_ml2, durs_exist_ml2, '-gx', alpha=0.5, label="ml2(ml2 trials)")
                ax.legend()

            return fig

        def _summarize():
            fd, _ = self.beh_get_fd_trial(0)
            print("* n trials in ml2 fd: ", len(self.get_trials_list()), len(mkl.getIndsTrials(fd)))
            print("self.BehTrialMapList", self.BehTrialMapList)
            print("self.BehTrialMapListGood", self.BehTrialMapListGood)
            print(lagshift)
            print(variation)
            print(corr[np.argmax(corr)-3:np.argmax(corr)+4])
            _doplot()

        if ploton:
            fig=_doplot()
            os.makedirs(self.Paths['figs_local'], exist_ok=True)
            fig.savefig(f"{self.Paths['figs_local']}/tdt_ml2_lags.pdf")

        if variation<ACCEPTABLE_VARIATION:
            if lagshift!=0:
                self.print_summarize_expt_params()
                _summarize()
                fig = _doplot()
                fig.savefig(f"{self.Paths['figs_local']}/DEBUG.pdf")
                print("** SAVED DEBUG FIGURE  to here:")
                print(f"{self.Paths['figs_local']}/DEBUG.pdf")
                self.print_summarize_expt_params()
                assert False, "variation says they are aligned... (but lagshift doesnt)"
        else:
            if lagshift==0:
                fig = _doplot()
                _summarize()
                fig.savefig(f"{self.Paths['figs_local']}/DEBUG.pdf")
                print("** SAVED DEBUG FIGURE  to /self.Paths['figs_local']/DEBUG.pdf")
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
                    # for x in self.BehTrialMapList:
                    #     if x[0]==0:
                    #         print(self.BehTrialMapList)
                    #         assert False, "This means it iwll look for ml trial 0, which doesnt eixst. This means you do not have beh for the first neural data. Solution: skip the first n neural trials. see what I did fro Pancho 220614"

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
        if False:
            # ntrials = len(self.get_trials_list(only_if_ml2_fixation_success=False, only_if_has_valid_ml2_trial=True))
            ntrials = len(self.get_trials_list(only_if_ml2_fixation_success=False, only_if_has_valid_ml2_trial=False))

            # print(self.get_trials_list(only_if_ml2_fixation_success=False, only_if_has_valid_ml2_trial=False))
            # assert False

            # ntrials = len(self.TrialsOnset) 
        else:
            # This is better. If tdt trials start at 1 (instead of 0) then the length will be short. This leads to bug.
            # Instead, you want the max tdt trial. Is generalyl equivalent.
            _trials = self.get_trials_list(only_if_ml2_fixation_success=False, only_if_has_valid_ml2_trial=False)
            max_trial_index_beh = max(_trials)+1
            ntrials = max_trial_index_beh
            
        self.BehTrialMapListGood = get_map_trial_and_set(self.BehTrialMapList, ntrials)

        print("... Generated these...")
        print("self.BehTrialMapList", self.BehTrialMapList)
        print("self.BehTrialMapListGood", self.BehTrialMapListGood)
        print("ntrials:", ntrials)
        # assert False

    def _beh_get_fdnum_trial(self, trialtdt):
        """ Get the filedata indices and trial indices (beh) for
        this neural trial (trialtdt).
        PARAMS:
        - doreset, then resets self.BehTrialMapListGood
        """

        if self.BehTrialMapListGood is None:
            self._beh_get_fdnum_trial_generate_mapper()

        # assert trialtdt < ntrials, "This tdt trial doesnt exist, too large..."
        try:
            [fd_setnum, fd_trialnum] = self.BehTrialMapListGood[trialtdt]
        except KeyError as err:
            for k, v in self.BehTrialMapListGood.items():
                print(k, " -- ", v)
            raise err
        return fd_setnum, fd_trialnum
        
    def beh_get_fd_trial(self, trialtdt):
        """ Return the fd and trial linked to this tdt trial
        """
        fd_setnum, fd_trialnum = self._beh_get_fdnum_trial(trialtdt)
        if fd_trialnum == 0:
            assert False, "need to skip the first neural trial (it lacks beh data). See what I did for pancho 220614"
        # print("-----------------")
        # print("self.BehFdList", self.BehFdList)
        # print("fd_setnum", fd_setnum)
        # print("fd_trialnum", fd_trialnum)
        # print("trialtdt", trialtdt)
        assert len(self.BehFdList)>fd_setnum, "probably didnt load all beh data. see beh_trial_map_list.py"
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
        reltiave toe onset of the neural data, for alignment of those data.
        """

        if trialtdt in self._CachedTrialOnset.keys():
            return self._CachedTrialOnset[trialtdt]
        else:
            from ..utils.monkeylogic import ml2_get_trial_onset as ml2gto
            # convert to trialml
            fd, trialml = self.beh_get_fd_trial(trialtdt)
            return ml2gto(fd, trialml)


    ######################## BRAIN STUFF
    def sitegetter_print_summarytext_each_unit(self):
        """ for each site, print a line (string) that summarizes
        it.
        """
        for i, site in enumerate(self.sitegetterKS_map_region_to_sites_MULTREG()):
            print(f"idx {i}, site {site}, -- ", self.sitegetter_summarytext(site))

    def sitegetter_print_summary_nunits_by_region(self):
        """ Prints num units (clean) per region
        and total units etc
        """
        sites_all =[]
        for area, sites in self.sitegetterKS_generate_mapper_region_to_sites_BASE(clean=True).items():
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
        regions_summary = self.sitegetter_get_brainregion_list_BASE()
        max_prev = 0
        print("region, nunits, --, min(sitenum), max(sitenum)")
        for regsum in regions_summary:
            sites = self.sitegetterKS_map_region_to_sites(regsum)
            if len(sites)>0:
                print(regsum, len(sites), "----", min(sites), max(sites))
                min_this = min(sites)
                # assert min_this > max_prev
                max_prev = min_this
            else:
                print(regsum, len(sites))


    def sitegetter_summarytext(self, site):
        """ Return a string that useful for labeling
        """

        info = self.sitegetterKS_thissite_info(site)
        clust = info["clust"]
        site_tdt = info["site_tdt"]
        bregion = info["region"]
        rs = info["rs"]
        chan = info["chan"]
        if self.SPIKES_VERSION=="tdt":
            assert site==site_tdt
            return f"{site_tdt}|{bregion}|{rs}-{chan}"
        elif self.SPIKES_VERSION=="kilosort":
            return f"s{site_tdt}|c{clust}|{bregion}|{rs}-{chan}"
        else:
            assert False


    # def sitegetter_brainregion_chan(self, region, chan):
    #     """ Given a regin (e.g., M1_m) and chan (1-256) return its site (1-512)
    #     """ 
    #     assert False, "rewrite using sitegetter_map_region_to_site"
    #     # which rs?
    #     sites = self.sitegetter_brainregion(region)
    #     if all([s<257 for s in sites]):
    #         rs = 2
    #     elif all([s<513 for s in sites]):
    #         rs = 3
    #     else:
    #         print(sites, region)
    #         assert False

    #     site = self.convert_rschan_to_site(rs, chan)
    #     assert site in sites, "this site not in this brain region!!"
    #     return site

    def _sitegetter_map_site_to_physical_location_electrode(self, site):
        """ Get the location
        RETURNS:
        - area, str
        - location, (x, y)
        """
        assert False, "code it"
        assert False, "6/22/22 4B inserted upside down"


    # def _sitegettertdt_get_map_brainregion_to_site(self):
    #     """ [TDT SITES] Retgurn dict mapping from regions to sites.
    #     Hard coded.
    #     RETURNS:
    #     - dict[region] = list of sites
    #     """
    #     # regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
    #     #         "PMd_p", "PMd_a", "SMA_p", "SMA_a",
    #     #         "dlPFC_p", "dlPFC_a", "vlPFC_p", "vlPFC_a",
    #     #         "preSMA_p", "preSMA_a", "FP_p", "FP_a"]
    #
    #     regions_in_order = self.sitegetter_get_brainregion_list()
    #
    #     # EXCEPTIONS
    #     # if self.Animal=="Pancho" and int(self.Date)==221002:
    #     #     # Then 6A (PMvl) and 6B (PMvm) were switched
    #     #     regions_in_order = [x for x in regions_in_order]
    #     #     regions_in_order[2] = "PMv_m"
    #     #     regions_in_order[3] = "PMv_l"
    #     # else:
    #     #     regions_in_order = regions_in_order
    #
    #     # Convert to dict
    #     dict_sites ={}
    #     for i, name in enumerate(regions_in_order):
    #         dict_sites[name] = list(range(1+32*i, 1+32*(i+1)))
    #     return dict_sites

    # _sitegetterKS_generate_mapper_region_to_sites
    def sitegetterKS_generate_mapper_region_to_sites_BASE(self, clean=True,
                                                          combine_into_larger_areas=False,
                                                          force_tdt_sites=False,
                                                          exclude_bad_areas=False):
        """
        [THE ONLY PLACE finalized sites are determined]
        [KILISORT WORKS] Generate dict mapping from region to sites, with added flexiblity of paras
        PARAMS:
        - clean, bool, whether to remove bad sites
        - combine_into_larger_areas, bool,
        RETURNS:
        - dict_sites[sitename] = list of ints (either sites or clusts, if using ks)
        """

        ################################## (1) Get default sites
        # dict_sites = self._sitegettertdt_get_map_brainregion_to_site()
        regions_in_order = self.sitegetter_get_brainregion_list_BASE()
        assert len(regions_in_order)==16
        dict_sites_TDT = {}
        for i, name in enumerate(regions_in_order):
            dict_sites_TDT[name] = list(range(1+32*i, 1+32*(i+1)))

        ########################### ANIMAL-SPECIFIC THINGS
        if self.Animal=="Diego":    
            # dlPFCp is severed.
            dict_sites_TDT["dlPFC_p"] = []

        if exclude_bad_areas:
            # Optional excluding, might be useful to keep for some analtyses, but fuincationlly
            # not good.
            if self.Animal=="Pancho":
                dict_sites_TDT["PMv_l"] = [] # Not drawing-related.. face area


        ################## COMBINE INTO LARGER AREAS?
        if combine_into_larger_areas:
            regions_specific = dict_sites_TDT.keys()
            # regions_in_order = ["M1", "PMv", "PMd", "SMA", "dlPFC", "vlPFC",  "preSMA", "FP"]
            # regions_in_order_combined = ["M1", "PMv", "PMd", "dlPFC", "vlPFC", "FP",  "SMA", "preSMA"]
            def _regions_in(summary_region):
                """ get list of regions (e.g, ["dlPFC_a", 'dlPFC_p']) that are in this summary region (e.g., dlPFC)
                """

                return MAP_COMBINED_REGION_TO_REGION[summary_region]

                # assert False, "replace this with more guaranteed to work."
                # return [reg for reg in regions_specific if reg.find(summary_region)==0]

            dict_sites_new = {}
            regions_in_order_combined = self.sitegetter_get_brainregion_list_BASE(True)
            for reg_comb in regions_in_order_combined:
                regions_specific_this = _regions_in(reg_comb)
                sites_this = [s for _reg in regions_specific_this for s in dict_sites_TDT[_reg]]
                dict_sites_new[reg_comb] = sites_this
            dict_sites_TDT = dict_sites_new



        ################################ (2) Get either TDT or KS sites.
        if self.SPIKES_VERSION=="tdt" or force_tdt_sites:
            # Remove bad sites?
            if clean:
                assert self.SitesDirty is not None, "you need to enter which are bad sites in SitesDirty"
                for k, v in dict_sites_TDT.items():
                    # remove any sites that are bad
                    dict_sites_TDT[k] = [vv for vv in v if vv not in self.SitesDirty]
            return dict_sites_TDT

        elif self.SPIKES_VERSION=="kilosort":
            ############ for KILOSORT? GET CLSUTERS
            # Then you want map to cluster_id

            mapper_region_to_clustids = {}
            for br, sites in dict_sites_TDT.items():
                # Collect clust ids for this region
                clusts = self.ks_convert_sites_to_clusts(sites)
                mapper_region_to_clustids[br] = clusts

            #     print("---", br)
            #     print(dict_sites[br])
            #     print(mapper_region_to_clustids[br])
            # assert False
            return mapper_region_to_clustids
        else:
            print(self.SPIKES_VERSION)
            assert False, "code it"

    def sitegetter_get_brainregion_list_BASE(self, combine_into_larger_areas=False):
        """ Get list of str, names of all brain regions, doesnt care what sites actually
        exist, just in theory what it would be. e.g, Diego, an array dead, but still return
        its area.
        [The only place brain region list is generated]
        """

        if combine_into_larger_areas:
            return list(_REGIONS_IN_ORDER_COMBINED)
        else:
            regions_in_order = list(_REGIONS_IN_ORDER)

            if self.Animal=="Pancho" and int(self.Date)==221002:
                # Then 6A (PMvl) and 6B (PMvm) were switched
                regions_in_order[2] = "PMv_m"
                regions_in_order[3] = "PMv_l"

            assert len(regions_in_order)==16
            return regions_in_order

        # if combine_into_larger_areas:
        #     # dict_sites = sorted(self._sitegetterKS_generate_mapper_region_to_sites(clean=False,
        #     #     combine_into_larger_areas=True))
        #     dict_sites = self._sitegetterKS_generate_mapper_region_to_sites(clean=False,
        #                                                                     combine_into_larger_areas=True)
        #     return sorted(list(dict_sites.keys()))
        # else:
        #     return REGIONS_IN_ORDER

    # def sitegetter_map_multregions_to_sites(self, list_region, clean=True):
    #     """ Return lsit of sites,concatenated across regions in lisT_region
    #     """
    #     return self.sitegetter_all(list_region, clean=clean)

    def sitegetter_map_site_to_array_physical_location(self):
        """
        Map electrode to a location on the physical array.
        Also holds exceptions, such as mistakes in plugging.
        """

        if self.Animal=="Pancho" and int(self.Date)==220621:
            assert False, "preSMAa was flipped"
        if self.Animal=="Pancho" and int(self.Date)==220714:
            assert False, "FPa was flipped"
        if self.Animal=="Pancho" and int(self.Date)==240730:
            assert False, "SMAp was flipped"
        
        assert False, "code it"

    def sitegetterKS_map_region_to_sites(self, region, clean=True,
            force_tdt_sites=False,
                                         exclude_bad_areas=False):
        """ [KILOSORT works] Given a region (string) map to a list of ints (sites).
        RETURNS;
        - list of ints, either sites or clusters (if using ks)
        """

        if region in _REGIONS_IN_ORDER:
            combine_into_larger_areas = False
        elif region in _REGIONS_IN_ORDER_COMBINED:
            combine_into_larger_areas = True
        else:
            assert False

        mapper = self.sitegetterKS_generate_mapper_region_to_sites_BASE(clean,
                                                                        combine_into_larger_areas=combine_into_larger_areas,
                                                                        force_tdt_sites=force_tdt_sites,
                                                                        exclude_bad_areas=exclude_bad_areas)
        # if region not in mapper.keys():
        #     mapper = self._sitegetterKS_generate_mapper_region_to_sites(clean,
        #         combine_into_larger_areas=True, force_tdt_sites=force_tdt_sites)

        sites_or_clusts = mapper[region]
        if self._DEBUG_PRUNE_SITES and len(sites_or_clusts)>0:
            sites_or_clusts = [sites_or_clusts[0]]

        return sites_or_clusts

    def sitegetterKS_map_site_to_region(self, site, region_combined=False):
        """ [KILOSORT WORKS] REturn the regino (str) for this site (int, 1-512)
        PARAMS:
        - region_combined, bool, if true, then uses gross areas (e.g, M1) but
        if False, then uses specific area for each array (e.g., M1_l)
        """

        if False:
            # This is dangerous, since site can be clust now (kilosort)
            if region_combined:
                Mapper = self.MapSiteToRegionCombined
            else:
                Mapper = self.MapSiteToRegion
        else:
            Mapper = []

        if len(Mapper)==0:
            # Generate it
            dict_sites = self.sitegetterKS_generate_mapper_region_to_sites_BASE(
                clean=False, combine_into_larger_areas=region_combined) # clean=False, since maping from sites to reg.
            for bregion, slist in dict_sites.items():
                for s in slist:
                    if s==site:
                        return bregion
            print(site)
            assert False, "site doesnt exist"
        #             Mapper[s] = bregion
        # return Mapper[site]

    def sitegetterKS_thissite_info(self, site, clean=False, ks_get_extra_info=False):
        """ returns info for this site in a dict
        INCLUDES even dirty sites
        """

        # Get the brain region
        regionthis = self.sitegetterKS_map_site_to_region(site)
        # dict_sites = self._sitegetterKS_generate_mapper_region_to_sites(clean=clean) 
        # regionthis = None
        # for bregion, sites in dict_sites.items():
        #     if site in sites:
        #         regionthis = bregion
        #         break
        # assert regionthis is not None

        # Get the rs and chan
        if self.SPIKES_VERSION=="tdt":
            rs, chan = self.convert_site_to_rschan(site)
            site_tdt = site
            clust = None

            info = {
                "site_tdt":site_tdt,
                "clust":clust,
                "region":regionthis,
                "rs":rs,
                "chan":chan}
        elif self.SPIKES_VERSION=="kilosort":
            clust = site
            site_tdt, rs, chan = self.ks_convert_clust_to_site_rschan(clust)
            
            info = {
                "site_tdt":site_tdt,
                "clust":clust,
                "region":regionthis,
                "rs":rs,
                "chan":chan}

            if ks_get_extra_info:
                tmp = self.datall_TDT_KS_slice_single_bysite(site, self.get_trials_list()[0])
                info["label_final"] = tmp["label_final"]
        else:
            assert False

        return info

    def _sitegetter_sort_sites_by(self, sites, by, take_top_n=None):
        """ Sort sites by some method and optionally return top n
        PARAMS:
        - sites, list of ints
        - by, str, method for sorting sites
        - take_top_n, eitehr None (ignore) or int, take top N after sorting.
        RETURNS:
        - sites_sorted, list of ints, sites sorted and (optiaolly) pruned to top n
        """

        assert self.SPIKES_VERSION=="tdt", "confirm that this works evne for kilosort"
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

    def sitegetterKS_all_sites(self, clean=True):
        """
        Return list of all sites
        """
        return self.sitegetterKS_map_region_to_sites_MULTREG(clean=clean)

    def sitegetterKS_map_region_to_sites_MULTREG(self, list_regions=None, clean=True,
                                                 force_tdt_sites=False):
        """ [KILOSORT_WORKS] Get all sites, in order
        MNOTE: will be in order of list_regions
        PARAMS:
        - list_regions, get only these regions. leave None to get all. if None,
        then returns all sites.
        RETURNS:
        - list of sites
        """
        
        if list_regions is None:
            list_regions = self.sitegetter_get_brainregion_list_BASE(combine_into_larger_areas=False)

        sites = []
        for region in list_regions:
            sites_this = self.sitegetterKS_map_region_to_sites(region, clean=clean, force_tdt_sites=force_tdt_sites)
            sites.extend(sites_this)

        return sites

    # def sitegetter_brainregion(self, region=None, clean=True):
    #     """ Flexible mapping from region to site
    #     PARAMS:
    #     - region, Either string (specific region) or list of strings (concats the sites)
    #     - clean, whether to remove garbage chanels
    #     RETURNS:
    #     - 
    #     - out, depends on type of region
    #     """
    #     # Hard coded
    #     # regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
    #     #                     "PMd_p", "PMd_a", "dlPFC_p", "dlPFC_a", 
    #     #                     "vlPFC_p", "vlPFC_a", "FP_p", "FP_a",
    #     #                     "SMA_p", "SMA_a", "preSMA_p", "preSMA_a"]
    #     # regions_in_order = ["M1_m", "M1_l", "PMv_l", "PMv_m",
    #     #                     "PMd_p", "PMd_a", "SMA_p", "SMA_a", 
    #     #                     "dlPFC_p", "dlPFC_a", "vlPFC_p", "vlPFC_a", 
    #     #                     "preSMA_p", "preSMA_a", "FP_p", "FP_a"]

    #     assert False, "new version"

    #     # do clean
    #     if clean:
    #         assert self.SitesDirty is not None, "you need to enter which are bad sites in SitesDirty"
    #         for k, v in dict_sites.items():
    #             # remove any sites that are bad
    #             dict_sites[k] = [vv for vv in v if vv not in self.SitesDirty]

    #     # Get sites
    #     if region=="list_regions":
    #         assert False, "_sitegetter_generate_mapper_region_to_sites"
    #         return regions_in_order
    #     elif region=="mapper" or region is None:
    #         assert False, "_sitegetter_generate_mapper_region_to_sites"
    #         return dict_sites
    #     elif isinstance(region, int):
    #         assert False, "why do this"
    #         return dict_sites[regions_in_order[region]]
    #     elif isinstance(region, list):
    #         # then this is list of str
    #         list_chans = []
    #         for reg in region:
    #             sites = self.sitegetterKS_map_region_to_sites(reg, clean=clean)
    #             list_chans.extend(sites)
    #         return list_chans
    #     elif isinstance(region, str):
    #         return self.sitegetterKS_map_region_to_sites(region)
    #         # if region in dict_sites.keys():
    #         #     # Then is one of the main regions
    #         #     return dict_sites[region]
    #         # else:
    #         #     # Then could be a summary region
    #         #     def _regions_in(summary_region):
    #         #         """ get list of regions (e.g, ["dlPFC_a", 'dlPFC_p']) that are in this summary region (e.g., dlPFC)
    #         #         """
    #         #         return [reg for reg in regions_in_order if reg.find(summary_region)==0]
    #         #     return self.sitegetter_all(list_regions=_regions_in(region), clean=clean)
    #     else:
    #         assert False

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

    def ks_convert_clust_to_site_rschan(self, clust):
        """
        Kilosort - convert clustid to tdt information
        RETURNS:
        - site_tdt, rs, chan, all ints.
        """
        # map from clust to site
        site_tdt = self.DatSpikesSessByClust[clust]["site"]
        # site_tdt = self._MapperKsortClustToSite[clust]
        rs, chan = self._convert_sitetdt_to_rschan(site_tdt)
        return site_tdt, rs, chan

    def _convert_sitetdt_to_rschan(self, sitetdt):
        if sitetdt>256:
            rs = 3
            chan = sitetdt - 256
        else:
            rs = 2
            chan = sitetdt
        return rs, chan

    def convert_site_to_rschan(self, site_or_clust):
        """ [KILOSORT WORKS] Cnvert either:
        TDT sites (1-512) -or-
        Kilosort clusters
        to rs and chan
        e.g., 512 --> (3, 256)
        """
        if self.SPIKES_VERSION=="tdt":
            return self._convert_sitetdt_to_rschan(site_or_clust)
        else:
            site_tdt, rs, chan = self.ks_convert_clust_to_site_rschan(clust=site_or_clust)
            return rs, chan


    ########################### Stats for each site
    def sitestats_fr_extract_good(self, sitenum, trials=None, 
                                  extract_using_smoothed_fr=False, keep_within_events_flanking_trial=False):
        """
        Good, wrapper to extract firing rate across trials (one scalar each trial), along wtih the time of the trial.
        RETURNS:
        - frvals, trials, times_frac, each a (ntrials,) array.
        - trialcodes, list of trialcodes
        """
        from pythonlib.tools.datetools import standardize_time_helper

        if trials is None:
            trials = self.get_trials_list(True)

        list_fr = []
        for t in trials:
            list_fr.append(self.sitestats_fr_single(sitenum, t, extract_using_smoothed_fr, keep_within_events_flanking_trial))
        frvals = np.array(list_fr)

        # (2) times (frac of day)
        trials = np.array(trials)
        dfthis = self.datasetbeh_extract_dataframe(trials)
        times_frac = np.array([standardize_time_helper(dt) for dt in dfthis["datetime"].tolist()])

        trialcodes = dfthis["trialcode"].tolist()

        
        return frvals, trials, times_frac, trialcodes

    def sitestats_fr_single(self, sitenum, trial, extract_using_smoothed_fr=False,
                            keep_within_events_flanking_trial=False):
        """ get fr (sp/s)
        - if fr doesnt exist in self.DatAll, then will add it to the dict.
        """

        if extract_using_smoothed_fr:
            # Then use smoothed fr, which is already extracted. Might be quicker.
            # This gets within two events that flank trial. So may be different fro below.
            pa = self.smoothedfr_extract_timewindow_during_trial(sitenum, trial, keep_within_events_flanking_trial)
            return np.mean(pa.X)
        else:
            # Use spike count.
            # assert self.SPIKES_VERSION=="tdt"

            if keep_within_events_flanking_trial:
                fr_key = "fr_within_trial"
            else:
                fr_key = "fr"

            dat = self.datall_TDT_KS_slice_single_bysite(sitenum, trial)
            if fr_key not in dat.keys():

                if dat["spike_times"] is None:
                    print("****")
                    print(sitenum, trial)
                    print(self.print_summarize_expt_params())
                    assert False, "figure out why this is None. probably in loading (scipy error) and I used to replace errors with None. I should re-extract the spikes."
                
                st = dat["spike_times"]
                dur = dat["time_dur"]
                if keep_within_events_flanking_trial:
                    # print(st)
                    # print(dur)
                    t1, t2 = self.events_get_time_flanking_trial_helper(trial)
                    st = st[(st>=t1) & (st<=t2)]
                    dur = t2-t1
                    # print(st)
                    # print(dur)
                    fr_key = "fr_within_trial"
                    # assert False
                else:
                    fr_key = "fr"
                nspikes = len(st) 
                dat[fr_key] = nspikes/dur
            return dat[fr_key]

    def sitestats_fr(self, sitenum):
        """ gets fr across all trials. Only works if you have already extracted fr 
        into DatAll (and its dataframe)
        """ 
        assert self._LOAD_VERSION == "FULL_LOADING", "This required self.DatAll, which is not gotten in MINIMAL LOADING"
        assert self.SPIKES_VERSION=="tdt", "not codeede for anything else."
        list_fr = []
        
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
        if not hasattr(self, "DatAllDf"):
            self.datall_cleanup_add_things(only_generate_dataframe=True)

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
        to disk (if save==True). This is more for computation than for returning anything useufl. 
        Run this once.
        NOTE: Takes a while, like minutes, because has to load the cached spikes data for each (chan, trial)
        I checked that this is optimized, assumimng that it has to go thru process of loading spikes for
        each chan.
        """
        list_trial = self.get_trials_list(True)
        for site in self.sitegetterKS_map_region_to_sites_MULTREG(clean=False):
            if site%10==0:
                print(site)
            
            for trial in list_trial:
                self.sitestats_fr_single(site, trial)

            # self.sitestats_fr(site) # run this to iterate over all trials, and save to datall
        if save:
            self._savelocal_datall()

    def sitestats_get_low_fr_sites(self, low_fr_thresh=2, savedir=None):
        """ FInds sites with mean fr less than threshold; 
        useful for pruning neurons, etc.
        """
        frmeans = []
        sites =[]
        for site in self.sitegetterKS_map_region_to_sites_MULTREG(clean=False):
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
            first_instance_only=False, shorthand=False, refrac_period=0.01,
            exclude_times_before_trial_onset=True):
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
        - exclude_times_before_trial_onset, bool, if True, keeps only times >0, i.e,
        ignroes the 1sec (default) padding before onset of each trial.
        RETURNS:
        - np array of times. empty array if doesnt exist.
        """    

        if isinstance(code, str):
            codenum = self.behcode_convert(codename=code, shorthand=shorthand)
        else:
            codenum = code

        timesall, codes = self.extract_data_tank_epocs("behcode") 
        if timesall is None:
            return np.array([])

        inds = codes==codenum
        times = timesall[inds]

        if trial0 is not None:
            times = self.extract_windowed_data_bytrial(times, trial0)[0] 
        
        if exclude_times_before_trial_onset:
            times = times[times>0]

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
        allow_no_crossing_per_behcode_instance_if=None,
        exclude_times_before_trial_onset=True):
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
        - exclude_times_before_trial_onset, bool, if True, keeps only times >0, i.e,
        ignroes the 1sec (default) padding before onset of each trial.
        RETURNS:
        - list of dict, where each dict holds array of times for a single instance of this behcode.
        """
        from pythonlib.tools.timeseriestools import get_threshold_crossings

        assert t_post > -t_pre, "window is negative length..."

        # 1) Extract the stream siganl
        if whichstream=="touch_binary":
            times, vals = self.beh_extract_touching_binary(trial)
            vals_sm = vals.copy()
        elif whichstream=="touch_in_fixsquare_binary":
            times, vals = self.beh_extract_touch_in_fixation_square(trial)
            vals_sm = vals.copy()
        elif whichstream=="touch_done_button":
            times, vals = self.beh_extract_touch_in_done_button(trial)
            vals_sm = vals.copy()
        elif whichstream=="reward":
            times, vals = self.extract_reward_stream(trial)
            vals_sm = vals.copy()
        else:
            times, vals, fs = self.extract_data_tank_streams(whichstream, trial)
            # 2) Smooth if desired
            n = int(smooth_win * fs)
            vals_sm = np.convolve(vals, np.ones(n)/n, mode="same")
            if False:
                fig, ax = plt.subplots(1,1, figsize=(15,10))
                ax.plot(times, vals, '-k');
                ax.plot(times, vals_sm, '-r');


        # Exclude times <0
        if exclude_times_before_trial_onset:
            _indskeep = times>=0
            times = times[_indskeep]
            vals = vals[_indskeep]
            vals_sm = vals_sm[_indskeep]

        # print("HERE!!!")
        # print(times)
        # print(vals)
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
                out.append({
                    "timecrosses":np.array([]),
                    "valcrosses":np.array([]),
                    "time_of_behcode":timethis,
                    "valminmax":valminmax,
                    "threshold":np.nan,
                    })
            else:
                # does this local window have sufficient difference between min and max (compared to the 
                # global range of this signal) if not, then
                # assume there is no crossing.

                MIN_RATIO = 0.125
                if True:
                    # This means that if ratio is less than MIN_RATIO, fail.
                    MIN_FRAC_PTS_CLOSE_TO_EXTREMES = 1.01
                else:
                    # this means only fail if frac_pts close to extrems is is less than this
                    # Allow small window, assuming values are not close to the cetner...
                    MIN_FRAC_PTS_CLOSE_TO_EXTREMES = 0.4
                ratio_minmax = (valminmax[1] - valminmax[0])/(valminmax_global[1] - valminmax_global[0]) 

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
                    # n_high = np.sum(vals_frac_of_range > 0.75)
                    # n_low = np.sum(vals_frac_of_range < 0.25)
                    # frac_high_or_low = (n_high + n_low)/len(vals_frac_of_range)

                    vals_frac_of_range_abs = np.abs(vals_frac_of_range - 0.5)
                    frac_pts_close_to_extremes = np.sum(vals_frac_of_range_abs>0.25)/len(vals_frac_of_range_abs)
                    
                    # print(ratio_minmax, valminmax_global, valminmax, frac_pts_close_to_extremes, n_high, n_low, len(vals_frac_of_range), frac_high_or_low, ratio_minmax, "ASDASDASDSA")
                    if frac_pts_close_to_extremes<MIN_FRAC_PTS_CLOSE_TO_EXTREMES:
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
                # self.datasetbeh_load(version="daily", FORCE_AFTER_MINIMAL=FORCE_AFTER_MINIMAL) # daily
                self.datasetbeh_load(version="daily") # daily
                print("**Loaded dataset! daily")
            except DataMisalignError as err:
                raise err
            except Exception as err:
                # Try loading using "null" rule, which is common
                print("This err, when try to load datset:",  err)
                assert dataset_beh_expt is not None, "assuming you wanted to get daily, but somehow failed and got to here... check that daily dataset actually exists."
                self.datasetbeh_load(dataset_beh_expt=dataset_beh_expt, 
                    version="main")
                print("**Loaded dataset! using rule: null")
        except DataMisalignError as err:
            raise err
        except Exception as err:
            print("probably need to pass in correct rule to load dataset.")
            raise err

        # Final stuff for dataset

        # self._generate_mappers_quickly_datasetbeh()
        self._generate_mappers_quickly_datasetbeh()
        # (1) Remove all neural trials which are missing from datasetbeh.
        self._datasetbeh_remove_neural_trials_missing_beh()
        self.Datasetbeh.LockPreprocess = True

    def datasetbeh_load(self, dataset_beh_expt=None, 
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
        - FORCE_AFTER_MINIMAL, if True, then alloows to run this even after you are in MINIMAL_LOADING mode
        i.e, you don't want to load the cached dataset.
        RETURNS:
        - self.DatasetBeh, and returns

        NOTE: GOOD sanityc check and matchign of nerual and beh data... if pasees this, confident that aligned betwene them
        # and mnot throwing out inadvertantly.
        """
        from pythonlib.dataset.dataset import Dataset
        from pythonlib.dataset.dataset import load_dataset_notdaily_helper, load_dataset_daily_helper

        # if not FORCE_AFTER_MINIMAL:
        #     assert self.DatAll is not None, "need to load first, run SN.extract_raw_and_spikes_helper()"

        # 1) Load Dataset
        if version=="daily":
            D = load_dataset_daily_helper(self.Animal, self.Date)
        elif version=="main":
            if self.DatasetbehExptname is None:
                # you must enter it
                expt = dataset_beh_expt
                assert dataset_beh_expt is not None, "huh?"
            else:
                # load saved
                expt = self.DatasetbehExptname
            D = load_dataset_notdaily_helper(self.Animal, expt)
        else:
            print(version)
            assert False
        self.Datasetbeh = D

        # 2) keep only the dataset trials that are included in recordings
        if self._LOAD_VERSION == "MINIMAL_LOADING": # will break for full loading, no mapper yet.
            #TODO: should instead regenerate all trials in neural, since cached trials MAY have thrown out trails
            # based on older beh dataset.
            # Prune dataset to include jsut the trials that exist also in nerual data. Just to reduce size of Dataset
            trialcodes = self.Datasetbeh.Dat["trialcode"].tolist()
            trialcodes_neural_exist = self.datasetbeh_trialcode_prune_within_session(trialcodes)
            print("- Keeping only dataset trials that exist in neural")
            print("Starting length: ", len(self.Datasetbeh.Dat))
            self.Datasetbeh.Dat = self.Datasetbeh.Dat[self.Datasetbeh.Dat["trialcode"].isin(trialcodes_neural_exist)].reset_index(drop=True)
            print("Ending length: ", len(self.Datasetbeh.Dat))

            # Count how many neural trials are missing dataset beh trials
            misses = 0
            gottens = 0
            for t in self.get_trials_list(True):
                if self.datasetbeh_trial_to_datidx(t) is None:
                    misses += 1
                else: 
                    gottens += 1
            if (gottens==0) or (misses>3):
                print("misses: ", misses)
                print("gottens: ", gottens)
                # assert False, "why so many neural trials are missing from dataset? "
                print("This is often beucase there are two beh sessions and one neural session, and so this thinks there is neural trial like sess 1 trial 500, when it should be sess 2, trial 200, this is beucase beh_trial_map_list is wrong. So now this leads to auto computing of beh_trial_map_list")
                print("Trialcodes in beh dataset (before pruning):", len(trialcodes), trialcodes)
                trialcodes_neural_all = [self.datasetbeh_trial_to_trialcode(t) for t in self.get_trials_list()]
                print("Trialcodes in neural data (all):", len(trialcodes_neural_all), trialcodes_neural_all)
                trialcodes_neural_pass_fix = [self.datasetbeh_trial_to_trialcode(t) for t in self.get_trials_list(True)]
                print("Trialcodes in neural data (good, passing fixation):", len(trialcodes_neural_pass_fix), trialcodes_neural_pass_fix)
                trialcodes_missing = [tc for tc in trialcodes_neural_pass_fix if tc not in trialcodes]
                print("Trialcodes expected to exist, but which did not find in beh dataset: ", len(trialcodes_missing), trialcodes_missing)
                print("--> Why are neural trialcodes missing from beh data?")
                print("--> Sometimes, it is because many of the neural trials are actually not passing fixation. In that case, is fine to remove them")
                print("--> Check that the trialcodes missing are legit misses in beh dataset (often: single stroke that was aborted)")

                raise DataMisalignError

            # If got this far, then is fine to just take trials that exist in datasetbeh, as doing so will not throw out much
            # data. If this is False, then some dates will fail, because idx1 or idx2 will be None
            only_if_in_dataset = True

            # (2) Sanity check --> within trial 1 and trial last, all the dataset trials should have a matching
            # neural trial (or at least all). This important beucase later only gets nerual trials that exist in dataset.
            # I dont want to inadvertangly throw out many neural trails due to bug...
            t1 = self.get_trials_list(True, only_if_in_dataset=only_if_in_dataset)[0]
            t2 = self.get_trials_list(True, only_if_in_dataset=only_if_in_dataset)[-1]
            idx1 = self.datasetbeh_trial_to_datidx(t1)
            idx2 = self.datasetbeh_trial_to_datidx(t2)
            if idx1 is None or idx2 is None:
                print(t1, t2)
                print(idx1)
                print(idx2)
                assert False, "this means that one of these two trials doesnt exist in datasetbeh. But this should not be possible if only_if_in_dataset=True"
            failures = 0
            total = 0

            for idx in range(idx1, idx2+1):
                try:
                    # Check that neural data exist.
                    self.datasetbeh_datidx_to_trial(idx)
                except Exception as err:
                    # print(idx, idx1, idx2, t1, t2)
                    # assert False, "found a dataset index that is missing neural trial. This shoudl not be possible. Error in parsing neural dta?"
                    failures += 1
                total += 1
            if failures/total > 0.05:
                # Print the culprits
                for idx in range(idx1, idx2+1):
                    try:
                        # Check that neural data exist.
                        self.datasetbeh_datidx_to_trial(idx)
                    except Exception as err:
                        print(idx, idx1, idx2, t1, t2)
                        # assert False, "found a dataset index that is missing neural trial. This shoudl not be possible. Error in parsing neural dta?"
                print(idx1, idx2, t1, t2)
                assert False, "found too many dataset index that are missing neural trial. This shoudl not be possible. Error in parsing neural dta? Only misses would be due to udpate of dataset extraction..."


        # print("Loading this dataset: ", self.Animal, expt, dataset_beh_rule)
        # D = Dataset([], remove_online_abort=remove_online_abort)
        # D.load_dataset_helper(self.Animal, expt, rule=dataset_beh_rule)
        # D.load_tasks_helper()

        # 10/25/22 - done automaticlaly.
        # if not D._analy_preprocess_done:
        #     D, GROUPING, GROUPING_LEVELS, FEATURE_NAMES, SCORE_COL_NAMES = preprocessDat(D, expt)
        if False: # NOT NEEDED ANYMORE
            if False:
                # old
                trials = self.get_all_existing_site_trial_in_datall("trial")
            elif False:
                trials = self.get_trials_list(True, True)
            else:
                # Cannot be True, since that looks for datasetbeh will fail.
                trials = self.get_trials_list(False, True)
                trials = [t for t in trials if self.datasetbeh_trial_to_datidx(t) is not None]
                list_trialcodes = [self.datasetbeh_trial_to_trialcode(t) for t in trials]
            try:
                assert len(list_trialcodes)==len(trials) and isinstance(list_trialcodes[0], str)
            except Exception as err:
                print(list_trialcodes)
                print(trials)
                raise err

            print("- Keeping only dataset trials that exist in self.Dat")
            print("Starting length: ", len(self.Datasetbeh.Dat))
            self.Datasetbeh.Dat = self.Datasetbeh.Dat[self.Datasetbeh.Dat["trialcode"].isin(list_trialcodes)].reset_index(drop=True)
            # self.Datasetbeh.filterPandas({"trialcode":list_trialcodes}, "modify")
            print("Ending length: ", len(self.Datasetbeh.Dat))


            # 3) Prune trials in neural to remove trials that dont have succesfuly fix and touch.
            # THIS IS NOT NEEDED Anymore, becuase of pruning using datasetbeh above. but run jsut as sanity check.
            trials_neural_to_remove = []
            # for trial_neural, trialcode in (list_trialcodes):
            #     fd, t = self.beh_get_fd_trial(trial_neural)
            #     outcome = mkl.getTrialsOutcomesWrapper(fd, t)
            #     t2 = trials[trial_neural]
            #     print(outcome.keys())
            #     print(trial_neural, trialcode, t, t2)
            # assert False
            for trial_neural, trialcode in zip(trials, list_trialcodes):
                n_failures = 0
                if trialcode not in self.Datasetbeh.Dat["trialcode"].tolist():
                    # then this is only acceptable if this trial is not succesful fix or touch
                    fd, t = self.beh_get_fd_trial(trial_neural)
                    suc = mkl.getTrialsFixationSuccess(fd, t)

                    # NOTE: this make some trials called "touched" even though no pnut touch
                    # error, since these are excluded from Dataset
                    touched = mkl.getTrialsTouched(fd, t)
                    # tem = mkl.getTrialsOutcomesWrapper(fd,t)["trial_end_method"]
                    if touched and suc:
                        n_failures += 1
                        if False: # Dont' fail, since sometimes dataset can be missing trials (e..g,, pruned strokes).
                            outcome = mkl.getTrialsOutcomesWrapper(fd, t)
                            print(outcome)
                            pnuts = mkl.getTrialsStrokesByPeanuts(fd, t)
                            print(pnuts)
                            print(list_trialcodes)
                            print(self.Datasetbeh.Dat["trialcode"].tolist())
                            print(trialcode)
                            assert False, "some neural data not found in beh Dataset..."
                    else:
                        outcome = mkl.getTrialsOutcomesWrapper(fd, t)
                        print("Removing this trial because it is not in Dataset:", trial_neural, trialcode, t, outcome["beh_evaluation"]["trialnum"])
                        # remove this trial from self.Dat, since it has no parallele in dataset
                        trials_neural_to_remove.append(trial_neural)
                assert n_failures == 0, "jhave already included only tirals that are in dataset. how could it possibly fail here?"
                # assert n_failures/len(trials)<0.05, "why so many missing from dataset? figure this out"

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
        in order, etc, and with indices resetted.
        """

        # assert False, 'in progress'
        # trialcodes = [sn.dataset_beh_trial_to_trialcode(t) for t in list_trials]

        D = self.Datasetbeh
        inds = [self.datasetbeh_trial_to_datidx(t) for t in list_trials]
        dfthis = D.Dat.iloc[inds].reset_index(drop=True)
        return dfthis

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
            "rulecue2":[-0.6, 0.6], # 
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


    def events_get_time_using_photodiode_and_save(self, list_trial=None, do_save=True):
        """ Extract and save for all trials. using all events. GOod for preprocessing.
        """

        list_events = self.events_default_list_events(include_events_from_dict=True)

        if list_trial is None:
            list_trial = self.get_trials_list(True, True)
        
        for trial in list_trial:
            # Extract (it skips extraction if already exists)
            tmp = self.events_get_time_using_photodiode(trial, list_events, 
                do_reextract_even_if_saved=True)
 
        # save
        if do_save:
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
            
            # Sanity check that you havent cached any events that use touchscreen. This important
            # beucase touchscreen lag is solved by shifting times. THis shift wont occur
            # if already cached
            # TODO: this only should apply to preprocessing that occured before 11/2/24, otherwise is 
            # more time (for reextracting these FD) but not a real problem.
            events_not_involving_touch = ["fixcue", "rulecue2", "samp", "go", "seqon", 
                            "post", "reward_all", "seqon"]
            self.EventsTimeUsingPhd = {x:vals for x, vals in self.EventsTimeUsingPhd.items() if x[1] in events_not_involving_touch}

            return True
        else:
            print("_loadlocal_events DOESNT EXIST")
            return False

    def _events_prune_cached_which_use_datasetbeh(self):
        """
        Useeful if you reload datsetbeh --> exclude cvachced events that are based on strokes which might now
        be innacurate.
        :return:
        """
        # Keep only events that use photodiode. Events that use beh strokes shold be
        # regenreated each time.
        # See this for newnames: map_event_newname_to_oldname
        print("Pruning events in EventsTimeUsingPhd to exclude those using datasetbeh strokes...")
        EVENTS_PHOTODIODE_KEEP = ["fixcue", "rulecue2", "samp", "go", "post", "reward_all", "go_cue",
                                  "stim_onset", "post_screen_onset"]
        if DATASETBEH_CACHED_USE_BEHTOUCH:
            EVENTS_PHOTODIODE_KEEP.append(["done_button", "doneb", "fixtch", "first_raisedoneb", "fix_touch"])

        # Excludes:
        # on_strokeidx_0 first_raise on_stroke_1 on_strokeidx_0 on_stroke_1 seqon off_stroke_last
        self.EventsTimeUsingPhd = {k:v for k, v in self.EventsTimeUsingPhd.items() if k[1] in EVENTS_PHOTODIODE_KEEP}

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

    def hack_adjust_touch_times_touchscreen_lag(self, times):
        """
        ########### HACKY: the touchscreen has a slight lag of ~1.5-2 videocam frames (50hz).
        # This means that if stroke onset is here T, then it actualyl was T-delta.
        PARAMS:
        - times, list of scalar times.
        RETURNS:
        - times_shifted
        NOTE: added 11/4/24
        """
        assert HACK_TOUCHSCREEN_LAG==True, "fix this, you are not allowed to run it."
        delta = 1.5 * 0.02 # n frames x 20ms per frame
        times = [t-delta for t in times]
        return times

    def events_get_time_using_photodiode(self, trial, 
        list_events = ("stim_onset", "go", "first_raise", "on_stroke_1"),
        do_reextract_even_if_saved=False, plot_beh_code_stream = False,
        do_reextract_if_not_in_saved= False):
        """
        [GOOD] Get dict of times of important events. Uses variety of methods, including
        (i) photodiode (ii) motor behavior, (iii) beh codes, wherever appropriate.
        - All times relative to behcode 9 (trial onset) by convention.
        PARAMS:
        - list_events, list of str, each a label for an event. only gets those in this list.
        - force_single_output, if True, then asserts that ther eis one and only one crossing.
        - do_reextract_even_if_saved, then is like do_reextract_if_not_in_saved, but always do_extract, even if
        it already exists.
        - do_reextract_if_not_in_saved, bool, if True, then tries to reextract if doesnt find this event..useful 
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


            if event in ["fixtch", "fix_touch", "first_raise"]:
                if len(self.BehFdList)==0: # MINIMAIL loading...
                    self.load_behavior()

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
                    except AssertionError as err:
                        if trial<20:
                            # then is early trial, when I know this sometimes fails.
                            times = _resort_to_behcode_time(event, 0.)
                        else:
                            raise err

                elif event=="fixcue":
                    # fixation cue visible.
                    try:
                        out = self.behcode_get_stream_crossings_in_window(trial, event, t_pre=0.015, t_post = 0.16, whichstream="pd2", 
                                                ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=True,
                                                assert_single_crossing_this_trial = True,
                                                assert_expected_direction_first_crossing = None)
                    except AssertionError as err:
                        # shorter post dur, because overlaps the subsequent cue2 (i.e,, fixtouch). This happens even if not using cue2, by 
                        # defualt it is constructed. So, error is when subject presses fixation quickly
                        try:
                            out = self.behcode_get_stream_crossings_in_window(trial, event, t_pre=0.015, t_post = 0.115, whichstream="pd2", 
                                                    ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=True,
                                                    assert_single_crossing_this_trial = True,
                                                    assert_expected_direction_first_crossing = None) 
                        except AssertionError as err:
                            # Last resort, make the window small and shift it gradually.
    
                            # start forward in time, then go backwards this way get the entire
                            # ie.. a 100ms window starting at (0, 0.1), then sliding back until (-0.1, 0), where neg means
                            # before the code.
                            LIST_TPRE = list(np.linspace(0, 0.1, 50)) 
                            WINDSIZE = 0.1
                            # trace (i.e. shift back until you lost the later dip in pd)
                            SM_WIN = 0.005 # if fix cue and rule2 are too close, then smoothing makes them hard to separate...
                            for t_pre in LIST_TPRE:
                                try:
                                    t_post = -t_pre + WINDSIZE
                                    out = self.behcode_get_stream_crossings_in_window(trial, event, t_pre=t_pre, t_post = t_post, whichstream="pd2", 
                                                            ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=True,
                                                            assert_single_crossing_this_trial = True,
                                                            assert_expected_direction_first_crossing = None,
                                                            smooth_win = SM_WIN)
                                    # Got here, this means success!
                                    break  
                                except AssertionError as err:
                                    if t_pre == LIST_TPRE[-1]:
                                        # Then you exhaustred all t_pre. fail
                                        raise err
                                    else:
                                        # Keep trying
                                        continue
                    times = _extract_times(out)

                elif event in ["fixtch", "fix_touch"]:
                    behcode = "fixtch" 

                    try:
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

                            # out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.5, t_post = 1, whichstream="touch_in_fixsquare_binary", 
                            out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.6, t_post = 1, whichstream="touch_in_fixsquare_binary", 
                                                                      ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=True,
                                                                        assert_single_crossing_this_trial = True,
                                                                         assert_expected_direction_first_crossing = "up",
                                                                         refrac_period_between_events=0.05)              

                        times = _extract_times(out)

                        # take the first time
                        times = times[:1]

                        # Touchscreen lag adjust.
                        times = self.hack_adjust_touch_times_touchscreen_lag(times)
                        
                        # sometimes fixtch will be before presntation of fixcue (e.g., if subject touchees in anticiation).
                        # Make fixtch same (slightly later) than fixcue, or else this might lead to errors later.
                        time_fixcue = self.events_get_time_helper("fixcue", trial, assert_one=True)
                        if len(times)>0:
                            assert len(time_fixcue)==1, "weird..."
                            if times[0]<time_fixcue:
                                times[0] = time_fixcue[0] + 0.001 # 1 ms after

                    except Exception as err:
                        # Special case: already touching fixation before it even shows up. this happens for pancho around 7/2023 
                        # onwards. Deal by defining the fixtch to be time of fixcue onset, with idea that this is when he sees
                        # his touch hit the cue...
                        if self.beh_check_touching_fixation_square_during_onset(trial):
                            times = self.events_get_time_helper("fixcue", trial, assert_one=True)
                            times[0] = times[0]+0.001 # see above.
                        else:
                            assert False, "not sure why..."

                elif event in ["rulecue2"]:
                    # rule cue that switches on between fix touch and samp. Exists only on some days, but it one-to-one linked
                    # to behcode 132. 
                    # includes both those overlapping fix, and those separate from fix.
                    try:
                        out = self.behcode_get_stream_crossings_in_window(trial, 132, t_pre=0.05, t_post = 0.2, whichstream="pd2", 
                                              ploton=plot_beh_code_stream, cross_dir_to_take="down", 
                                              assert_single_crossing_per_behcode_instance=True,
                                                assert_single_crossing_this_trial = False,
                                                  assert_expected_direction_first_crossing = "down",
                                                  refrac_period_between_events=0.05)              
                    except AssertionError as err:
                        # USUALLY: rulecue/fixcue(samething) --> fixtouch.
                        # SOMETIMES: fixtouch --> rulecue/fixcue [i.e., he touches in anticipation]. THis is prolbem beucase
                        # fixtouch also triggers pd2. 
                        # SOLUTION: shorten t_pre until fixtouch influence is gone. This is fine, since response to rulecue occurs
                        # after rulecue.

                        # OLDER NOTES:
                        # sometimes rulecue is shown too quickly relative to fixation cue, e.g.,, becuase subject
                        # presses quickly in anticiation. Then the predur might have a contamination. solve this by
                        # shortening predur

                        from itertools import product
                        LIST_TPRE = list(np.linspace(0.045, -0.17, 40))
                        LIST_TPOST = list(np.linspace(0.2, 0.4, 20)) # Sometimes PD drops too far ahead
                        SM_WIN = 0.005 # if fix cue and rule2 are too close, then smoothing makes them hard to separate...
                        for t_pre, t_post in product(LIST_TPRE, LIST_TPOST):
                            try:
                                # out = self.behcode_get_stream_crossings_in_window(trial, 132, t_pre=t_pre, t_post = 0.22, whichstream="pd2", 
                                out = self.behcode_get_stream_crossings_in_window(trial, 132, t_pre=t_pre, t_post = t_post, whichstream="pd2", 
                                                    ploton=plot_beh_code_stream, cross_dir_to_take="down", 
                                                    assert_single_crossing_per_behcode_instance=True,
                                                        assert_single_crossing_this_trial = False,
                                                        assert_expected_direction_first_crossing = "down",
                                                        refrac_period_between_events=0.05,
                                                        smooth_win = SM_WIN)        
                                # Got here, this means success!
                                break  
                            except AssertionError as err:
                                if t_pre == LIST_TPRE[-1] and t_post == LIST_TPOST[-1]:
                                    # Then you exhaustred all t_pre. fail
                                    raise err
                                else:
                                    # Keep trying
                                    continue

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

                    try:
                        VER = 1
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                                  cross_dir_to_take=cross_dir, t_pre=t_pre,
                                                                  t_post=t_post,
                                                                  ploton=plot_beh_code_stream, assert_single_crossing_per_behcode_instance=True, 
                                                                  assert_single_crossing_this_trial = assert_single_crossing_this_trial) 
                        times = _extract_times(out)
                        times_behcode = self.behcode_extract_times_semantic(behcode, trial) 

                    except AssertionError as err:

                        try:
                            # Sometimes there is long delay from behcode to PD. 
                            t_post = 0.5                    
                            out = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                                    cross_dir_to_take=cross_dir, t_pre=t_pre,
                                                                    t_post=t_post,
                                                                    ploton=plot_beh_code_stream, assert_single_crossing_per_behcode_instance=True, 
                                                                    assert_single_crossing_this_trial = assert_single_crossing_this_trial) 
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
                    # t_post = 0.16 # missed. too short.
                    # t_post = 0.33 # is ok to be kind of long, nothing following can contaminate.
                    t_post = 0.45 # is ok to be kind of long, nothing following can contaminate.
                    out = self.behcode_get_stream_crossings_in_window(trial, behcode, whichstream=stream, 
                                                              cross_dir_to_take=cross_dir, t_pre=t_pre,
                                                              t_post=t_post,
                                                              ploton=plot_beh_code_stream, assert_single_crossing_per_behcode_instance=True, 
                                                              assert_single_crossing_this_trial = True, assert_expected_direction_first_crossing="down",
                                                              take_first_behcode_instance=True)
                    times = _extract_times(out)

                    # Touchscreen lag adjust.
                    times =self.hack_adjust_touch_times_touchscreen_lag(times)

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

                    # Touchscreen lag adjust.
                    times =self.hack_adjust_touch_times_touchscreen_lag(times)

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
                    return compute_times_from_scratch("on_strokeidx_0")

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
                    
                    # Touchscreen lag adjust.
                    times =self.hack_adjust_touch_times_touchscreen_lag(times)

                elif "off_strokeidx_" in event:
                    # e.g., on_strokeidx_2 means onset of 3rd stroke
                    # Returns empty, if this idx doesnt exist.
                    assert event.find("off_strokeidx_")==0

                    # - which stroke id
                    idx = int(event[14:])

                    # onset of idx stroke (touch)
                    _, offs = self.strokes_extract_ons_offs(trial)
                    if len(offs)<idx+1:
                        times = []
                    else:
                        times = [offs[idx]]

                    # Touchscreen lag adjust.
                    times =self.hack_adjust_touch_times_touchscreen_lag(times)

                elif event=="off_stroke_last":
                    # offset of the last stroke (touch)
                    _, offs = self.strokes_extract_ons_offs(trial)
                    if len(offs)==0:
                        times = []
                    else:
                        times = [offs[-1]]

                    # Touchscreen lag adjust.
                    times =self.hack_adjust_touch_times_touchscreen_lag(times)

                elif event in ["done_button", "doneb"]:
                    # Onset of touch of done button, based on touchgin within square
                    # NOTE: This can sometimes (rarely) be very diff (many sec) from the fb assocaited
                    # with done. this is when he holds finger by done button, but doesnt trigger...

                    # The t_pre is long becuase seomtimes he succesfuly touches, but not registered. Should coutn those.
                    # force_must_find_crossings = False # because rarely will touch, but finger not in there (w.g. water)

                    # behcode = 62
                    behcode = "doneb"
                    t_post = 0.05 # touch is always before the code, so this can be low ...
 
                    # LIST_TPRE = [0.5, 1.5, ]
                    # First try with really short pre window. This useful espeicalytl if this is short trial and fixation is close
                    # in space to done button. then will get offset of fixation... and will fail.
                    try:
                        out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.5, t_post = t_post, 
                                                whichstream="touch_done_button", 
                                                ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
                                                assert_single_crossing_this_trial = False,
                                                assert_expected_direction_first_crossing = "up",
                                                take_first_behcode_instance=True,
                                                take_first_crossing_for_each_behcode=True,
                                                refrac_period_between_events=0.05)  
                        times = _extract_times(out)
                    except AssertionError as err:
                        try:
                            # NOte: tpre very large because seomtimes he touches but isnt correctly registered.
                            # out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=2.5, t_post = 1, 

                            # ade the t_pre shoerter (was 2.5) so that does not coincide with fixation, which is now at same location.
                            out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=1.5, t_post = t_post, 
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
                            # NOte: tpre very large because seomtimes he touches but isnt correctly registered.

                            assert len(self._behcode_extract_times("doneb", trial, shorthand=True))<2, "then doent want to expand window"

                            # keep tpost constant. expand back in time tpre back until the end of the last stroke.
                            LIST_TPRE = list(np.linspace(1.5, 12, 50)) 
                            for t_pre in LIST_TPRE:
                                try:
                                    out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=t_pre, t_post = t_post, 
                                                            whichstream="touch_done_button", 
                                                            ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
                                                            assert_single_crossing_this_trial = False,
                                                            assert_expected_direction_first_crossing = "up",
                                                            take_first_behcode_instance=True,
                                                            take_first_crossing_for_each_behcode=False,
                                                            refrac_period_between_events=0.05) 
                                    # Got here, this means success!
                                    break  
                                except AssertionError as err:
                                    if t_pre == LIST_TPRE[-1]:
                                        # Then you exhaustred all t_pre. fail
                                        raise err
                                    else:
                                        # Keep trying
                                        continue
                            # out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=12, t_post = 2, 
                            #                         whichstream="touch_done_button", 
                            #                         ploton=plot_beh_code_stream, cross_dir_to_take="up", assert_single_crossing_per_behcode_instance=False,
                            #                         assert_single_crossing_this_trial = False,
                            #                         assert_expected_direction_first_crossing = "up",
                            #                         take_first_behcode_instance=True,
                            #                         take_first_crossing_for_each_behcode=False,
                            #                         refrac_period_between_events=0.05) 
                            times = _extract_times(out)

                            ### take the first time that is following the last stroke offset, or go cue
                            offs = self.strokes_extract_ons_offs(trial)[1]
                            if len(offs)>0:
                                tmin = offs[-1]
                            else:
                                # no strokes. take go cue
                                tmin = self.events_get_time_helper("go", trial)[0]
                                # times_go = self.events_get_time_using_photodiode(trial, list_events=["go"]) # {'go': [3.3162100494546394]}
                                # print(tmin, times_go)
                                # assert False

                            # # if len(times_go["go"])>0:
                            #     tgo = times_go["go"][0]
                            # print(type(tgo))
                            # print(times)
                            # print(times_go)
                            times = [t for t in times if t > tmin]
                            # print(times)
                            # assert False
                            # else:

                    times = times[0:1] # take the first time, sometimes can go in and out of squiare
                    
                    # Touchscreen lag adjust.
                    times = self.hack_adjust_touch_times_touchscreen_lag(times)

                elif event in ["post_screen_onset", "post"]:
                    # onset of the post-screen, which is offset of the "pause" after you report done
                    # Avoid, since sometimes pd1 fails.. 
                    # (e..g, if doesnt make any touches postfix...)
                        # then try other method. this fails if dont find.

                    # First, try very loose with both
                    # behcode = 73
                    behcode = "post"
                    try:
                        out1 = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.2, t_post = 0.4, whichstream="pd2", 
                          ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=False,
                            assert_single_crossing_this_trial = False,
                             assert_expected_direction_first_crossing = "down",
                             take_first_crossing_for_each_behcode=True)
                    except AssertionError:
                        # This can happen if the trial ends early, which make the pd go up too early, in which case the
                        # first crossing may ne up.
                        out1 = []
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
                        try:
                            out = self.behcode_get_stream_crossings_in_window(trial, behcode, t_pre=0.2, t_post = 0.35, whichstream="pd2", 
                              ploton=plot_beh_code_stream, cross_dir_to_take="down", assert_single_crossing_per_behcode_instance=False,
                                assert_single_crossing_this_trial = False,
                                 assert_expected_direction_first_crossing = "down",
                                 take_first_crossing_for_each_behcode=True)
                        except AssertionError as err:
                            # See note above. this is same logic
                            out = []
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
                
                elif event=="reward_first_post":
                    # Get the first reward that is triggered after done (actually, detected using end of last stroke) -- i.e,., this is evaluation of trial. This excludes
                    # rewards that at other times, which are hand triggered. Might (rarely) include rew post done, which is what I 
                    # would want anyway.
                    # NOTE: this is a bit slow!
                    
                    # time_done = self.events_get_time_helper("post", trial, True)[0] # PROLBEM, this can sometimes occur after last rew
                    # time_done = self.events_get_time_helper("doneb", trial, True)[0]
                    # if len(time_done)==0:

                    # if self.beh_this_day_uses_done_button(): # Actulaly dont do this, since some days have blocks not using done button.
                    #     time_post = self.events_get_time_helper("doneb", trial, False)
                    # else:
                    # OPTION 1 - go back 0.5, to acocunt for times that tpost is after rew. rare, and usualyl very low, like 0.01
                    time_post = self.events_get_time_helper("post", trial, False)
                    time_post = [t - 0.5 for t in time_post]

                    if len(time_post)>0:
                        times_rew = self.events_get_time_helper("reward_all", trial, False)
                        times = [t for t in times_rew if t>time_post]
                    else:
                        # Trial didnt
                        times = []

                    # Alternative, deals with issue that post sometimes occurs after rew. But above, putting -0.5, shold solve this probe
                    # times_stroke_off = self.events_get_time_helper("off_stroke_last", trial, False) # list of times
                    # time_doneb = self.events_get_time_helper("doneb", trial, False)
                    # if len(times_stroke_off)>=1:
                    #     times_stroke_off = times_stroke_off[0]
                    #     times_rew = self.events_get_time_helper("reward_all", trial, False)
                    #     times = [t for t in times_rew if (t>times_stroke_off)]

                    #     if len(time_doneb)>0:
                    #         times = [t for t in times if t>min(time_doneb)]

                    #     times = [t for t in times if t>time_post-1]

                    #     if len(times)>1:
                    #         times = [min(times)]

                    # else:
                    #     # no stroke onthis trial.. so there cant be an evaluative reward
                    #     times = []

                    if len(times)>1:
                        times = [min(times)]
                else:
                    print(event, "This event doesnt exist!!")
                    raise NotEnoughDataException
                    
                assert times is not None
            return times

        ###############################
        dict_events = {}
        for event in list_events:

            if do_reextract_even_if_saved:
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
                    if len(times)==0 and do_reextract_if_not_in_saved:
                        print("Trying to reextract (trial, event):", trial, event)
                        times = self.events_get_time_using_photodiode(trial, 
                            list_events=[event], do_reextract_even_if_saved=True)[event]
                    RECOMPUTE = False

            if RECOMPUTE:
                try:
                    times = compute_times_from_scratch(event) 
                except Exception as err:
                    print(">>>>>>>>>>>>>>>")
                    print("trial, event:", trial, event)
                    self.print_summarize_expt_params()
                    raise err
                # Always save if do recompute
                self.EventsTimeUsingPhd[(trial, event)] = times

            # Store for output
            dict_events[event] = times

        return dict_events

    # load fixation onset times (231027_eyetracking_neural_exploration.ipynb)
    def events_get_clusterfix_fixation_times_and_centroids(self, trial, on_or_off=True, event_endpoints=None,
                                                           plot_overlay_on_raw=False):
        """
        Load fixation times and locations, as derived using clusterfix.
        """
        # load this trial's saccade times
        trialcode = self.datasetbeh_trial_to_trialcode(trial)
        directory_for_clusterfix = PATH_SAVE_CLUSTERFIX

        # load the .csv file for this trialcode, containing fixation times
        if on_or_off:
            fname_times = f"{directory_for_clusterfix}/{self.Animal}-{self.Date}-{self.RecSession}/clusterfix_result_csvs/{trialcode}-fixation-onsets.csv"
        else:
            fname_times = f"{directory_for_clusterfix}/{self.Animal}-{self.Date}-{self.RecSession}/clusterfix_result_csvs/{trialcode}-fixation-offsets.csv"

        data_times = pd.read_csv(fname_times, sep=',').values
        data_times = [item[0] for item in data_times]

        # load the .csv file for this trialcode, containing fixation centroids
        fname_centroids = f"{directory_for_clusterfix}/{self.Animal}-{self.Date}-{self.RecSession}/clusterfix_result_csvs/{trialcode}-fixation-centroids.csv"
        data_centroids = pd.read_csv(fname_centroids, sep=',').values
        data_centroids = [item for item in data_centroids]

        if event_endpoints is not None:
            if event_endpoints==["stim_onset", "go"]:
                # get start, end indices
                start_time, end_time = self.get_time_window_of_events(trial, event_endpoints[0], event_endpoints[1])
                valid_inds = [i for i, t in enumerate(data_times) if end_time >= t >= start_time]
                data_times = [data_times[i] for i in valid_inds]
                data_centroids = [data_centroids[i] for i in valid_inds]


            else:
                assert False, "code this"

        if plot_overlay_on_raw:
            times_tdt, vals_tdt_calibrated, fs_tdt, vals_tdt_calibrated_sm = self.beh_extract_eye_good(trial, CHECK_TDT_ML2_MATCH=True, return_all=True,
                                                                                                    PLOT=False)

            if event_endpoints==["stim_onset", "go"]:
                _inds = (times_tdt>=start_time) & (times_tdt<=end_time)
                times_tdt = times_tdt[_inds]
                vals_tdt_calibrated = vals_tdt_calibrated[_inds, :]
                vals_tdt_calibrated_sm = vals_tdt_calibrated_sm[_inds, :]
            else:
                assert event_endpoints is None

            fig1d, axes1d = plt.subplots(4,1, figsize=(20,10), squeeze=False, sharex=True)

            ax = axes1d.flatten()[0]
            ax.plot(times_tdt, vals_tdt_calibrated[:,0], label="tdt_x", color="b")
            ax.plot(data_times, np.stack(data_centroids)[:,0],  "ob")
            for t in data_times:
                ax.axvline(t, linestyle="-", alpha=0.5, color="k")
            ax.set_title("fixation events overlaid on raw voltage (calibrated)")
            ax.legend()
            # - overlay trial events
            self.plotmod_overlay_trial_events(ax, trial)

            ax = axes1d.flatten()[1]
            ax.plot(times_tdt, vals_tdt_calibrated[:,1], label="tdt_y", color="r")
            ax.plot(data_times, np.stack(data_centroids)[:,1],  "or")
            for t in data_times:
                ax.axvline(t, linestyle="-", alpha=0.5, color="k")
            ax.legend()
            ax.set_title("fixation events overlaid on raw voltage (calibrated)")
            # - overlay trial events
            self.plotmod_overlay_trial_events(ax, trial)

            ax = axes1d.flatten()[2]
            ax.plot(times_tdt, vals_tdt_calibrated[:,0], label="tdt_x", color="b")
            ax.plot(times_tdt, vals_tdt_calibrated[:,1], label="tdt_y", color="r")
            ax.legend()

            ax.plot(data_times, np.stack(data_centroids)[:,0],  "ob")
            ax.plot(data_times, np.stack(data_centroids)[:,1],  "or")
            for t in data_times:
                ax.axvline(t, linestyle="-", alpha=0.5, color="k")
            ax.set_title("fixation events overlaid on raw voltage (calibrated)")
            # - overlay trial events
            self.plotmod_overlay_trial_events(ax, trial)

            # Also put line underneath coloring the shapes
            ax = axes1d.flatten()[3]
            self.beh_eye_fixation_task_shape_overlay_plot(trial, ax)
            
            # overlay on task image
            fig2d, axes2d = plt.subplots(2, 2, figsize=(10,10))

            def _plot_eye_trace(times, vals, ax, alpha=0.15, overlay_idx_text=False):
                ax.plot(vals[:,0], vals[:,1], "-k", alpha=alpha)
                ax.scatter(vals[:,0], vals[:,1], c=times, alpha=alpha, marker="x")

                if overlay_idx_text:
                    for i, v in enumerate(vals):
                        ax.text(v[0], v[1], i, fontsize=10)

            ax = axes2d.flatten()[0]
            self.plot_taskimage(ax, trial)
            _plot_eye_trace(times_tdt, vals_tdt_calibrated, ax)
            ax.set_title("raw, not smoothed")

            ax = axes2d.flatten()[1]
            self.plot_taskimage(ax, trial)
            _plot_eye_trace(times_tdt, vals_tdt_calibrated_sm, ax)
            ax.set_title("raw, smoothed")

            ax = axes2d.flatten()[2]
            self.plot_taskimage(ax, trial)
            _plot_eye_trace(np.array(data_times), np.stack(data_centroids), ax, 
                            alpha=1, overlay_idx_text=True)
            ax.set_title("extracted fixations")

            return data_times, data_centroids, fig1d, axes1d, fig2d, axes2d
        else:
            return data_times, data_centroids

    # load saccade onset times (231027_eyetracking_neural_exploration.ipynb)
    def events_get_clusterfix_saccade_times(self, trial, on_or_off=True, event_endpoints=None, plot_overlay_on_raw=False):
        # load this trial's saccade times
        trialcode = self.datasetbeh_trial_to_trialcode(trial)
        directory_for_clusterfix = PATH_SAVE_CLUSTERFIX

        # load the .csv file for this trialcode, containing saccade times
        if on_or_off: #TODO change
            fname = f"{directory_for_clusterfix}/{self.Animal}-{self.Date}-{self.RecSession}/clusterfix_result_csvs/{trialcode}-saccade-onsets.csv"
        else:
            fname = f"{directory_for_clusterfix}/{self.Animal}-{self.Date}-{self.RecSession}/clusterfix_result_csvs/{trialcode}-saccade-offsets.csv"

        data = pd.read_csv(fname, sep=',').values
        data = [item[0] for item in data]

        if event_endpoints is not None:
            if event_endpoints==["stim_onset", "go"]:
                # get start, end indices
                start_time, end_time = self.get_time_window_of_events(trial, event_endpoints[0], event_endpoints[1])
                valid_inds = [i for i, t in enumerate(data) if end_time >= t >= start_time]
                data = [data[i] for i in valid_inds]
            else:
                assert False, "code this"

        if plot_overlay_on_raw:
            times_tdt, vals_tdt_calibrated, fs_tdt, vals_tdt_calibrated_sm = self.beh_extract_eye_good(trial, CHECK_TDT_ML2_MATCH=True, return_all=True,
                                                                                                    PLOT=False)

            if event_endpoints==["stim_onset", "go"]:
                _inds = (times_tdt>=start_time) & (times_tdt<=end_time)
                times_tdt = times_tdt[_inds]
                vals_tdt_calibrated = vals_tdt_calibrated[_inds, :]
                vals_tdt_calibrated_sm = vals_tdt_calibrated_sm[_inds, :]
            else:
                assert event_endpoints is None

            fig, axes = plt.subplots(3,1, figsize=(15,10), squeeze=False)

            ax = axes.flatten()[0]
            ax.plot(times_tdt, vals_tdt_calibrated[:,0], label="tdt_x", color="b")
            for t in data:
                ax.axvline(t, linestyle="-", alpha=0.5, color="k")
            ax.set_title("fixation events overlaid on raw voltage (calibrated)")
            ax.legend()
            # - overlay trial events
            self.plotmod_overlay_trial_events(ax, trial)

            ax = axes.flatten()[1]
            ax.plot(times_tdt, vals_tdt_calibrated[:,1], label="tdt_y", color="r")
            for t in data:
                ax.axvline(t, linestyle="-", alpha=0.5, color="k")
            ax.legend()
            ax.set_title("fixation events overlaid on raw voltage (calibrated)")
            # - overlay trial events
            self.plotmod_overlay_trial_events(ax, trial)

            ax = axes.flatten()[2]
            ax.plot(times_tdt, vals_tdt_calibrated[:,0], label="tdt_x", color="b")
            ax.plot(times_tdt, vals_tdt_calibrated[:,1], label="tdt_y", color="r")
            ax.legend()
            for t in data:
                ax.axvline(t, linestyle="-", alpha=0.5, color="k")
            ax.set_title("fixation events overlaid on raw voltage (calibrated)")
            # - overlay trial events
            self.plotmod_overlay_trial_events(ax, trial)

            # # overlay on task image
            # fig2d, axes2d = plt.subplots(2, 2, figsize=(10,10))

            # def _plot_eye_trace(times, vals, ax, alpha=0.15, overlay_idx_text=False):
            #     ax.plot(vals[:,0], vals[:,1], "-k", alpha=alpha)
            #     ax.scatter(vals[:,0], vals[:,1], c=times, alpha=alpha, marker="x")

            #     if overlay_idx_text:
            #         for i, v in enumerate(vals):
            #             ax.text(v[0], v[1], i, fontsize=10)

            # ax = axes2d.flatten()[0]
            # self.plot_taskimage(ax, trial)
            # _plot_eye_trace(times_tdt, vals_tdt_calibrated, ax)
            # ax.set_title("raw, not smoothed")

            # ax = axes2d.flatten()[1]
            # self.plot_taskimage(ax, trial)
            # _plot_eye_trace(times_tdt, vals_tdt_calibrated_sm, ax)
            # ax.set_title("raw, smoothed")

            # ax = axes2d.flatten()[2]
            # self.plot_taskimage(ax, trial)
            # _plot_eye_trace(np.array(data_times), np.stack(data_centroids), ax, 
            #                 alpha=1, overlay_idx_text=True)
            # ax.set_title("extracted fixations")

        return data

    def events_get_time_helper(self, event, trial, assert_one=False):
        """ [GOOD] Return the time in trial for this event. Tries to use
        photodiode... Only tries other moethods if this event doesnt exist for pd.
        PARAMS:
        - event, either string or tuple.
        - eventkind, string, if None, then tries to find it automatically.
        - assert_one, then asserts exactly one time found.
        RETURNS:
        - list of numbers, one for each detection of this even in this trial, sorted.
        """

        if event in ["saccon_preparation", "saccoff_preparation"]:
            times = self.events_get_clusterfix_saccade_times(trial, event_endpoints=["stim_onset", "go"])
        elif event in ["fixon_preparation"]:
            times, _ = self.events_get_clusterfix_fixation_times_and_centroids(trial, event_endpoints=["stim_onset", "go"])
        else:
            try:
                # Better version using photodiode or motor
                times = self.events_get_time_using_photodiode(trial, [event])[event] 
            except NotEnoughDataException as err:
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
                    times = [alignto_time]
                elif isinstance(event, str):
                    # THis is behcode shorthand
                    code = self.behcode_convert(codename=event, shorthand=True)
                    # try pd again
                    try:
                        times = self.events_get_time_using_photodiode(trial, [code])[code] 
                    except NotEnoughDataException as err:
                        # Get the behcode time.
                        alignto_time = self._behcode_extract_times(code, trial, first_instance_only=True)
                        times = [alignto_time]
                elif isinstance(event, int):
                    # Then is behcode
                    alignto_time = self._behcode_extract_times(event, trial, first_instance_only=True)
                    times = [alignto_time]
                else:
                    assert False

            if assert_one:
                assert len(times)==1
                
        return times

    def events_get_time_flanking_trial_helper(self, trial):
        """
        Get times for onset and offset of meat of trial (fixcue --> post/reward)
        """

        t1 = self.events_get_time_helper("fixcue", trial, assert_one=True)[0]
        t2 = self.events_get_time_helper("post", trial, assert_one=True)[0]
        tmp = self.events_get_time_helper("reward_all", trial, assert_one=False)
        if len(tmp)>0:
            t3 = max(tmp)
        else:
            t3 = 0.
        t_start = t1
        t_end = max([t2, t3]) # Ends either at start of post, or when get reward
        return t_start, t_end

    def events_get_feature_helper(self, event, trial):
        """ [GOOD] Return the name of a feature, plus a list of feature values, in trial for this event.
        PARAMS:
        - event, either string or tuple.
        - trial, number
        RETURNS:
        - feat_name, name of feature column
        - list_featvals, list of feature values within column titled {feat_name}
        """

        if event in ["fixon_preparation"]:
            feat_name = "fixation-centroid"
            _, list_featvals = self.events_get_clusterfix_fixation_times_and_centroids(trial, event_endpoints=["stim_onset", "go"])
        else:
            assert False
                
        return feat_name, list_featvals

    def events_default_list_events(self, include_stroke_endpoints=True,
            include_events_from_dict=False):
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
            list_events = ["fixcue", "fixtch", "rulecue2", "samp", "go", 
                "first_raise", "on_stroke_1", "seqon", 
                "off_stroke_last", "doneb", 
                "post", "reward_all"]
        else:
            list_events = ["fixcue", "fixtch", "rulecue2", "samp", "go", 
                "first_raise", "seqon", "doneb", 
                "post", "reward_all"]

        # # Using full name, so is compatible with eventsanaly_helper_pre_postdur_for_analysis
        # if include_stroke_endpoints:
        #     list_events = ["fixcue", "fix_touch", "samp", "go_cue", 
        #         "first_raise", "on_stroke_1", "seqon", 
        #         "off_stroke_last", "doneb", 
        #         "post", "reward_all"]
        # else:
        #     list_events = ["fixcue", "fix_touch", "samp", "go_cue", 
        #         "first_raise", "seqon", "doneb", 
        #         "post", "reward_all"]
 
        #  dict_events_bounds = {
        #     "fixcue":[-0.6, 0.6], # onset of fixation cue
        #     "fix_touch":[-0.6, 0.6], # button-touch
        #     "rulecue2":[-0.6, 0.6], # 
        #     "samp":[-0.6, 0.6], # image response
        #     "go_cue":[-0.6, 0.6], # movement onset.
        #     "first_raise":[-0.6, 0.6], # image response
        #     "on_strokeidx_0":[-0.6, 0.6], # image response
        #     "off_stroke_last":[-0.6, 0.6], # image response
        #     "doneb":[-0.6, 0.6], # image response    
        #     "post":[-0.6, 0.6], # image response    
        #     "reward_all":[-0.6, 0.6], # image response    
        # }


        # include anything in dict
        if include_events_from_dict:
            dict_events_bounds, _ = self.eventsanaly_helper_pre_postdur_for_analysis(
                just_get_list_events=True)
            for x in dict_events_bounds.keys():
                if x not in list_events:
                    list_events.append(x)

        return list_events

    def events_rename_with_ordered_index(self, list_events):
        """
        ev --> 01_ev... returned in input order
        """

        list_events_uniqnames = []
        for i, event in enumerate(list_events):
            # Better, so that names are stable
            if event in MAP_EVENT_TO_PREFIX:
                pref = MAP_EVENT_TO_PREFIX[event] # "00"
                event_unique_name = f"{pref}_{event}"
            else:
                # give event a unique name
                # Version 2, with indices
        #         event_unique_name = f"{i}_{event[:3]}_{event[-1]}"
                if i<10:
                    idx_str = f"0{i}"
                else:
                    idx_str = f"{i}"
                event_unique_name = f"{idx_str}_{event}"
            list_events_uniqnames.append(event_unique_name)

        return list_events_uniqnames



    def eventsdataframe_sanity_check(self, DEBUG=False, PLOT=True):
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
                    elif ev=="rulecue2":
                        # is ok if not find, since if the behcode (132) does exist, then
                        # is is asserted that this pd blip is found. so lack of finding 
                        # means that 132 didnt exist in this trial..
                        # TODO: confirm that 132 doesnt exist.
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

        if PLOT:
            list_categories = list(log_events_in_chron_order_allinstances.keys())
            for category in list_categories:
                print("Saving figure for:", category, " at ", savedir)
                trials = log_events_in_chron_order_allinstances[category]
                fig, axes, _, _, = self.plotwrapper_raster_multrials_onesite(trials, plot_beh=False, 
                                                        plot_rasters=False, SIZE=0.15, alignto="go")
                
                for ax in axes.flatten():
                    # Extract the current axis limits
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()

                    # Set x and y ticks with intervals of 1 based on the extracted limits
                    from pythonlib.tools.plottools import axis_xlim_ylim_intervals_modify
                    axis_xlim_ylim_intervals_modify(ax, 0.5, "x")
                    axis_xlim_ylim_intervals_modify(ax, 0.5, "y")
                    
                # save
                fig.savefig(f"{savedir}/events_raster-eventscat_{category}.pdf")
                plt.close("all")
            
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



    def eventsdataframe_extract_timings(self, list_events=None, trials=None,
            DEBUG=False):
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

        if trials is None:
            trials = self.get_trials_list(True)

        if DEBUG:
            trials = trials[:20]
            
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
    def _spiketrain_as_elephant(self, site, trial, cache = True):
        """ Get this site and trial as elephant (Neo) SpikeTrain object
        RETURNS:
        - st, a SpikeTrain object
        (Can be None, if the spiketimes is None)
        """
        from neo.core import SpikeTrain
        from quantities import s

        # extract dat
        dat = self.datall_TDT_KS_slice_single_bysite(site, trial)
        assert dat is not None, "doesnt exist..."

        stimes = dat["spike_times"]
        if stimes is None:
            st = None
        else:
            # Convert to spike train
            st = self.elephant_spiketrain_from_values(dat["spike_times"], dat["time_on"], dat["time_off"])
            # st = SpikeTrain(dat["spike_times"]*s, t_stop=dat["time_off"], t_start=dat["time_on"])

        if cache:
            dat = self.datall_TDT_KS_slice_single_bysite(site, trial)
            dat["spiketrain"] = st

        return st

    def spiketrain_as_elephant_batch(self, save=True):
        """ Generate and save SpikeTrain for all site and trial
        RETURNS:
        - adds "spiketrain" as key in self.DatAll
        """

        ADDED = False # track whether datall is updated.

        if hasattr(self, "DatAllDf") and "spiketrain" in self.DatAllDf.columns and not np.any(self.DatAllDf["spiketrain"].isna()):
            # then already gotten. skip
            pass
        else:
            # Extract
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


        # if not hasattr(self, "DatAllDf"):
        #     assert False, "need to first run extract_raw_and_spikes_helper to extract DatAllDf"
        
        # if "spiketrain" in self.DatAllDf.columns and not np.any(self.DatAllDf["spiketrain"].isna()):
        #     # then already gotten. skip
        #     pass
        # else:
        #     for i, Dat in enumerate(self.DatAll):
        #         if "spiketrain" not in Dat.keys():
        #             if i%500==0:
        #                 print("spiketrain_as_elephant_batch, datall index: ", i)
        #             # if Dat["trial0"]%50==0:
        #             #     print(Dat["trial0"])
        #             if "site" in Dat.keys():
        #                 site = Dat["site"]
        #             else:
        #                 site = self.convert_rschan_to_site(Dat["rs"], Dat["chan"])
        #             st = self._spiketrain_as_elephant(site, Dat["trial0"])
        #             Dat["spiketrain"] = st
        #             ADDED = True

        # print("FINISHED - extracting spiketrain for all trials in self.DatAll")
        # if ADDED and save:
        #     self._savelocal_datall()

    ####################### GENERATE POPANAL for a trial
    def _popanal_generate_from_raw(self, frate_mat, times, chans, df_label_trials=None,
        list_df_label_cols_get=None):
        """ Low level code to generate PopAnal from inputed raw fr data
        THE ONLY place where PA are generated in Session or Snippets.
        PARAMS:
        - frate_mat, shape (chans, trials, times)
        - times, shape (times,)
        - df_label_trials, either None (ignroes) or df labeling each trial, 
        whwere the rows match frmat[-, :, -]
        - df_label_cols_get, which cols of df_label_trials to take. if NOne, then
        will take at least "trialcode".
        RETURNS:
        - PopAnal object
        """
        from neuralmonkey.classes.population import PopAnal

        PA = PopAnal(frate_mat, times, chans = chans, print_shape_confirmation=False)

        # Input labels
        if df_label_trials is not None:
            if len(list_df_label_cols_get)==0:
                assert "trialcode" in df_label_trials.columns, "either pass in at least one feature, or have trialcode column"
                list_df_label_cols_get = ["trialcode"] # need to be not empyt or else doswnsdtream daifls.
            assert list_df_label_cols_get is not None and len(list_df_label_cols_get)>0
            PA.labels_features_input_from_dataframe(df_label_trials, list_df_label_cols_get, dim="trials")

        return PA


    def _popanal_generate_alldata_bystroke(self, DS, sites, 
        pre_dur, post_dur, fail_if_times_outside_existing,
        use_combined_region, features_to_get_extra=None):
        """ Low level. Get a single PA, with one row for each stroke in DS.
        
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
        regions = [self.sitegetterKS_map_site_to_region(s, region_combined=use_combined_region) for s in sites]
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
        assert align_to_stroke==True, "legacy to k eep this param..."

        # 2) generate PA
        # if align_to_stroke:
        # Return the single pa, aligned to each stroke in
        pa = self._popanal_generate_alldata_bystroke(DS, sites, 
            pre_dur, post_dur, fail_if_times_outside_existing,
            use_combined_region, features_to_get_extra=features_to_get_extra)
        ListPA = [pa]
        # else:
        #     assert False, "this is HACKY. this only uses DS.Dat to collect one datapt for each stroke in DS.Dat. This should either be trial level or stroke level"

        #     # just use trials, align to specific item in each trial
        #     # (note: would use this over popanal_generate_alldata because here 
        #     # saves the stroke-level features)
        #     assert len(align_to_alternative)>0, "need to pass in list of str, events to align to"

        #     # Which trials?
        #     trials = []
        #     trialcodes = []
        #     for ind in range(len(DS.Dat)):
        #         tc = DS.Dat.iloc[ind]["dataset_trialcode"]
        #         trial_neural = self.datasetbeh_trialcode_to_trial(tc)
        #         trials.append(trial_neural)
        #         trialcodes.append(tc)

        #     # Collect
        #     ListPA = self.popanal_generate_alldata(trials, sites, align_to_alternative, 
        #         pre_dur, post_dur)

        #     # Assign stroke-level features
        #     print("Extracting dataset features into pa.Xlabel [trials]")
        #     # list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
        #     # list_cols = ['task_kind', 'gridsize', 'dataset_trialcode', 
        #     #     'stroke_index', 'stroke_index_fromlast', 'shape_oriented', 'ind_taskstroke_orig', 'gridloc',
        #     #     'gridloc_x', 'gridloc_y', 'h_v_move_from_prev']
        #     list_cols = []
        #     if features_to_get_extra is not None:
        #         assert isinstance(features_to_get_extra, list)
        #         list_cols = list(set(list_cols + features_to_get_extra))
        #     for pa in ListPA:
        #         pa.labels_features_input_from_dataframe(DS.Dat, list_cols, dim="trials")
        #         # Sanity check, input order matches output order
        #         # assert pa.Xlabels["trials"]["dataset_trialcode"].tolist() == trialcodes
        #         assert pa.Xlabels["trials"]["trialcode"].tolist() == trialcodes

        #         assert all([c in pa.Xlabels["trials"].columns for c in list_cols])

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
        regions = [self.sitegetterKS_map_site_to_region(s, region_combined=use_combined_region) for s in sites]
        for pa in ListPA:
            pa.labels_input("regions_combined", regions, dim="chans")

        return ListPA

    def _popanal_generate_timewarped_rel_events_extract_raw(self,  events=None, sites=None, trials=None,
                                                            predur_rel_first_event = -1, postdur_rel_last_event = 1,
                                                            min_interval_dur=0.1, get_dfspikes=True):
        """
        Do this separateyl for each session before concatenating.
        PARAMS:
        - min_interval_dur, min dur between events. throws out trials that fail this.
        """
        from elephant.kernels import GaussianKernel        
        from elephant.statistics import time_histogram, instantaneous_rate,mean_firing_rate
        from quantities import s
        from neo.core import SpikeTrain 

        ### Prep params
        if events is None:
            # NOTE: exclude fixcue, beucase there is sometimes very short gap between fixcue and fixtch, and also high variability. This
            # leads to weird things.
            if self.beh_this_day_uses_done_button():
                events = ['fixtch', 'samp', 'go_cue', 'first_raise', 'on_strokeidx_0', 'off_strokeidx_0', 'doneb', 'reward_first_post']
                # events = ['fixcue', 'fixtch', 'samp', 'go_cue', 'first_raise', 'on_strokeidx_0', 'off_strokeidx_0', 'doneb', 'reward_first_post']
            else:
                events = ['fixtch', 'samp', 'go_cue', 'first_raise', 'on_strokeidx_0', 'off_strokeidx_0', 'reward_first_post']
                # events = ['fixcue', 'fixtch', 'samp', 'go_cue', 'first_raise', 'on_strokeidx_0', 'off_strokeidx_0', 'reward_first_post']

        # Get trials that hold a set of events
        if trials is None:
            trials = self.get_trials_list(True, events_that_must_include_in_order = events, min_interval_dur=min_interval_dur)
        else:
            tmp = self._get_trials_list_if_include_these_events_in_order(trials, events, min_interval_dur=min_interval_dur)
            assert tmp==trials, "some trials are mssing events"
        if sites is None:
            sites = self.sitegetterKS_all_sites()

        ### Get event time distributions across trials
        dfeventsn, _ = self.eventsdataframe_extract_timings(events, trials)
        times_array = np.array(dfeventsn["time_events_flat_first_unsorted"].tolist()) # convert to array (ntrials, nevents)
        # Check that is good array
        assert times_array.shape == (len(trials), len(events))
        assert np.all(np.isnan(times_array)==False)

        # Append pre and post times to first and last times of array
        # -- makes rest of analysis easier.
        times_array = np.concatenate([(times_array[:,0] + predur_rel_first_event)[:, None], times_array, (times_array[:, -1] + postdur_rel_last_event)[:, None]], axis=1)

        ### Get spiketimes
        if get_dfspikes:
            res = []
            for site in sites:
                for ind_trial in range(len(trials)):
                    print("Getting spiketimes, site:", site)
                    event_time = times_array[ind_trial][0]
                    post_dur = times_array[ind_trial][-1] - event_time
                    pre_dur = 0
                    trial_neural = trials[ind_trial]

                    spike_times = self._snippets_extract_single_snip_spiketimes(site, trial_neural,
                        event_time, pre_dur, post_dur, subtract_event_time=False)
                    res.append({
                        "site":site,
                        "ind_trial":ind_trial,
                        "trial":trial_neural,
                        "spike_times_raw":spike_times,
                    })
            dfspikes = pd.DataFrame(res)
        else:
            dfspikes = None

        ### Get pa for each trial
        print("Getting pa for each trial")
        list_pa = []
        for ind_trial in range(len(trials)):
            t_on = times_array[ind_trial][0]
            t_off = times_array[ind_trial][-1]
            trial_neural = trials[ind_trial]
            sampling_period = 0.001 # to make interpolation more accurate
            pa = self.popanal_generate_save_trial(trial_neural, sampling_period=sampling_period)
            pa.Times = np.array(pa.Times)
            pa = pa.slice_by_dim_values_wrapper("chans", sites)
            pa = pa.slice_by_dim_values_wrapper("times", (t_on, t_off))
            # if not pa.Chans == sites:
            #     print(pa.Chans)
            #     print(sites)
            #     assert False
            if False: # doing this above now
                pa = pa.slice_by_dim_values_wrapper("times", [t_on, t_off])
            list_pa.append(pa)

        ### Get beh data
        dflab = self.datasetbeh_extract_dataframe(trials)

        # Sanity check:
        # assert times_array.shape[0]==len()
        return events, sites, times_array, dfspikes, dflab, list_pa


    # def popanal_generate_timewarped_rel_events(self, events=None, sites=None, trials=None, PLOT=False,
    #                                            predur_rel_first_event = -1, postdur_rel_last_event = 1):
    #     """
    #     Wrapper to generate PA that has fr time-warped to fit the "median" trial, based on linear
    #     interpolation between events. Each trial must have all events. 
    #     PARAMS:
    #     - events, list of str. if give 4 events, then will actuayl ahve 6, including 2 flankers, whose timing
    #     is given by predur_rel_first_event, and postdur_rel_last_event.
    #     RETURNS:
    #     - PA, with times in actual raw times. Event-info is in PA.Params
    #     """
    #     from elephant.kernels import GaussianKernel        
    #     from elephant.statistics import time_histogram, instantaneous_rate,mean_firing_rate
    #     from quantities import s
    #     from neo.core import SpikeTrain

    #     ### Prep params
    #     if events is None:
    #         events = ['samp', 'go_cue', 'first_raise', 'on_strokeidx_0']
    #     # Get trials that hold a set of events
    #     if trials is None:
    #         trials = self.get_trials_list(events_that_must_include = events)
    #     else:
    #         tmp = self._get_trials_list_if_include_these_events(trials, events)
    #         assert tmp==trials, "some trials are mssing events"
    #     if sites is None:
    #         sites = self.sitegetterKS_all_sites()

    #     ### Get event time distributions across trials
    #     dfeventsn, _ = self.eventsdataframe_extract_timings(events, trials)
    #     times_array = np.array(dfeventsn["time_events_flat_first_unsorted"].tolist()) # convert to array (ntrials, nevents)
    #     # Check that is good array
    #     assert times_array.shape == (len(trials), len(events))
    #     assert np.all(np.isnan(times_array)==False)

    #     # Append pre and post times to first and last times of array
    #     # -- makes rest of analysis easier.
    #     times_array = np.concatenate([(times_array[:,0] + predur_rel_first_event)[:, None], times_array, (times_array[:, -1] + postdur_rel_last_event)[:, None]], axis=1)

    #     ### Get median times
    #     times_median = np.median(times_array, axis=0)
        
    #     def convert_spiketime_to_timecanonical(st, ind_trial, DEBUG=False):
    #         """
    #         Given a time and a trial, convert it to common coordinate system, 0...1...2, where these are the evenst, 
    #         and fraction is fraction with nthe window
    #         """
    #         times = times_array[ind_trial]
    #         event_ind_this_time_occurs_after = np.max(np.argwhere((st-times)>0))
    #         time_delta = st - times[event_ind_this_time_occurs_after]
    #         time_interval = times[event_ind_this_time_occurs_after+1] - times[event_ind_this_time_occurs_after]
    #         time_delta_frac = time_delta/time_interval
    #         time_canonical = event_ind_this_time_occurs_after + time_delta_frac

    #         if DEBUG:
    #             print(st, times, event_ind_this_time_occurs_after, time_delta_frac)
            
    #         assert time_delta_frac>=0
    #         assert time_delta_frac<=1
    #         return time_canonical
        
    #     ### Get spiketimes
    #     res = []
    #     for site in sites:
    #         for ind_trial in range(len(trials)):
    #             print("Getting spiketimes, site:", site)
    #             event_time = times_array[ind_trial][0]
    #             post_dur = times_array[ind_trial][-1] - event_time
    #             pre_dur = 0
    #             trial_neural = trials[ind_trial]

    #             spike_times = self._snippets_extract_single_snip_spiketimes(site, trial_neural,
    #                 event_time, pre_dur, post_dur, subtract_event_time=False)
    #             res.append({
    #                 "site":site,
    #                 "ind_trial":ind_trial,
    #                 "trial":trial_neural,
    #                 "spike_times_raw":spike_times,
    #             })
    #     dfspikes = pd.DataFrame(res)
        
    #     def _get_spike_times(site, trial, spikes_version="spike_times_warped"):
    #         """ get spike times for this (site, trial)"""
    #         tmp = dfspikes[(dfspikes["site"] == site) & (dfspikes["trial"] == trial)]
    #         assert len(tmp)==1
    #         return tmp[spikes_version].values[0]

    #     ### METHODS -- Convert from canonical time to projected onto median trial event times
    #     times_median_canonical = np.arange(len(times_median)) # (0, 1, 2, ...)
    #     map_segment_to_segmentdur = {} # each segment (0,1 ,2..) to dur in sec
    #     for i, dur in enumerate(np.diff(times_median)):
    #         map_segment_to_segmentdur[i] = dur

    #     def project_spiketimescanonical_to_median(spike_times_canonical):
    #         """
    #         e.g., convert 0.1 (first segment, 0.1 frac within) to the actual raw time relative to the
    #         median trial.
    #         """
    #         spike_times_warped = np.zeros(spike_times_canonical.shape)-999

    #         for segment_idx in times_median_canonical[:-1]:

    #             # get all inds in this segment
    #             mask = ((spike_times_canonical-segment_idx)>=0) & ((spike_times_canonical-segment_idx)<1)

    #             # get their fraction within the segment, and add that to the median time onset
    #             fracs = (spike_times_canonical[mask] - segment_idx) # frac  
    #             segment_dur = map_segment_to_segmentdur[segment_idx]
    #             segment_onset = times_median[segment_idx]

    #             assert np.all(spike_times_warped[mask] == -999), "already filled..."

    #             spike_times_warped[mask] = segment_onset + fracs*segment_dur
    #         assert np.all(spike_times_warped>-999), "missed items"
    #         return spike_times_warped

    #     ### Now time-warp each spike time to canonical coordiantes then to warped.
    #     list_spike_times_canonical = []
    #     list_spike_times_warped = []
    #     for i, row in dfspikes.iterrows():
    #         ind_trial = row["ind_trial"]
    #         site = row["site"]
    #         spike_times_raw = row["spike_times_raw"]

    #         print("Times --> canonical, site:", site)
    #         spike_times_canonical = np.array([convert_spiketime_to_timecanonical(st, ind_trial) for st in spike_times_raw])

    #         # Project to median (final warp)
    #         print("Canonical --> warped, site:", site)
    #         spike_times_warped = project_spiketimescanonical_to_median(spike_times_canonical)

    #         list_spike_times_canonical.append(spike_times_canonical)
    #         list_spike_times_warped.append(spike_times_warped)
    #     dfspikes["spike_times_canonical"] = list_spike_times_canonical
    #     dfspikes["spike_times_warped"] = list_spike_times_warped

    #     # PLOT EXAMPLE
    #     if PLOT:
    #         site = sites[0]
    #         trials_plot = trials[:20]

    #         fig, axes = plt.subplots(1, 4, figsize=(40, 8))

    #         ax = axes.flatten()[0]
    #         ax.set_title("original times")
    #         for ind_trial in range(len(trials_plot)):
    #             times = times_array[ind_trial, :]
    #             ax.plot(times, np.ones(len(times))*ind_trial, "ok")

    #             # overlay spikes
    #             spike_times = _get_spike_times(site, trials[ind_trial], "spike_times_raw")
    #             ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

    #         ax = axes.flatten()[1]
    #         ax.set_title("original times, subtract first time")
    #         for ind_trial in range(len(trials_plot)):
    #             times = times_array[ind_trial, :]
    #             times = times_array[ind_trial, :]
    #             time_start = times[0]
    #             times = times - time_start
    #             ax.plot(times, np.ones(len(times))*ind_trial, "ok")

    #             # overlay spikes
    #             spike_times = _get_spike_times(site, trials[ind_trial], "spike_times_raw")
    #             spike_times = spike_times - time_start
    #             ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

    #         ax = axes.flatten()[2]
    #         ax.set_title("canonical times")
    #         for ind_trial in range(len(trials_plot)):
    #             times = times_array[ind_trial, :]

    #             # times = times_array[ind_trial, :]
    #             times = times_median_canonical
    #             ax.plot(times, np.ones(len(times))*ind_trial, "ok")

    #             # overlay spikes
    #             spike_times = _get_spike_times(site, trials[ind_trial], "spike_times_canonical")
    #             ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

    #         ax = axes.flatten()[3]
    #         ax.set_title("warped times (final)")
    #         for ind_trial in range(len(trials_plot)):
    #             times = times_array[ind_trial, :]

    #             # times = times_array[ind_trial, :]
    #             times = times_median
    #             ax.plot(times, np.ones(len(times))*ind_trial, "ok")

    #             # overlay spikes
    #             spike_times = _get_spike_times(site, trials[ind_trial], "spike_times_warped")
    #             ax.plot(spike_times, np.ones(len(spike_times))*ind_trial, "xr")

    #     ### Generate PA -- using warped aligned times, generate a single PA
    #     t_on = times_median[0]
    #     t_off = times_median[-1]
    #     list_pa = []
    #     for trial in trials:
            
    #         print("Generating PA, trial: ", trial)
            
    #         # Collect spike trains over all sites
    #         list_spiketrain = []
    #         for site in sites:
    #             # dat = self.datall_TDT_KS_slice_single_bysite(site, trial)
    #             st = _get_spike_times(site, trial, spikes_version="spike_times_warped")
    #             list_spiketrain.append(self.elephant_spiketrain_from_values(st, t_on, t_off))
                
    #         # Convert spike train to smoothed FR
    #         frate = instantaneous_rate(list_spiketrain, sampling_period=SMFR_TIMEBIN*s, 
    #             kernel=GaussianKernel(self.SMFR_SIGMA*s))

    #         # Convert to popanal
    #         times = np.array(frate.times)
    #         pa = self._popanal_generate_from_raw(frate.T.magnitude, times, sites, df_label_trials=None)
    #         list_pa.append(pa)

    #     from neuralmonkey.classes.population import concatenate_popanals
    #     PA = concatenate_popanals(list_pa, "trials", 
    #                             values_for_concatted_dim = trials,
    #                             # assert_otherdims_have_same_values=True, 
    #                             # assert_otherdims_restrict_to_these=("chans", "times"),
    #                             assert_otherdims_have_same_values=False,   # no need, it must be by design
    #                             assert_otherdims_restrict_to_these=("chans", "times"),
    #                             all_pa_inherit_times_of_pa_at_this_index=0)

    #     # Get beh data
    #     PA.Xlabels["trials"] = self.datasetbeh_extract_dataframe(trials)

    #     # Store params related to the events
    #     PA.Params["version"] = "time_warped_to_events"
    #     PA.Params["event_times_array"] = times_array
    #     PA.Params["event_times_median"] = times_median
    #     PA.Params["event_times_median_canonical"] = times_median_canonical
    #     PA.Params["events_inner"] = events
    #     PA.Params["events_all"] = ["ONSET"] + events + ["OFFSET"]
    #     PA.Params["ONSET_predur_rel_first_event"] = predur_rel_first_event
    #     PA.Params["OFFSET_postdur_rel_lst_event"] = postdur_rel_last_event

    #     # Store bregions
    #     # -- NOTE: This works.. list_pa, bregions = PA.split_by_label("chans", "bregion_combined")
    #     res =[]
    #     for site in PA.Chans:
    #         res.append(
    #             {"bregion_combined": self.sitegetterKS_map_site_to_region(site, region_combined=True),
    #             "bregion":self.sitegetterKS_map_site_to_region(site, region_combined=False),
    #             "chan":site
    #         })

    #     # save
    #     PA.Xlabels["chans"] = pd.DataFrame(res)        

    #     return PA


    def elephant_spiketrain_from_values(self, spike_times, time_on, time_off):
        """ Convert array of times to a SpikeTrain instance
        """
        from quantities import s
        from neo.core import SpikeTrain
        spiketrain = SpikeTrain(spike_times*s, t_stop=time_off, t_start=time_on)
        return spiketrain

    def elephant_spiketrain_to_smoothedfr(self, spike_times, 
        time_on, time_off, 
        gaussian_sigma = None, # changed to 0.025 on 4/3/23. ,
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

        if gaussian_sigma is None:
            gaussian_sigma = self.SMFR_SIGMA

        spiketrain = self.elephant_spiketrain_from_values(spike_times, time_on, time_off)
        # spiketrain = SpikeTrain(spike_times*s, t_stop=time_off, t_start=time_on)

        frate = instantaneous_rate(spiketrain, sampling_period=sampling_period*s, 
            kernel=GaussianKernel(gaussian_sigma*s))
        
        return frate.times[None, :], frate.T.magnitude

    def popanal_generate_save_trial(self, trial, 
            # gaussian_sigma = 0.1, 
            # gaussian_sigma = 0.025, # changed to 0.025 on 4/3/23. 
            # sampling_period=0.005, # changed to 0.005 from 0.01 on 4/18/23.
            gaussian_sigma = None, # made global on 4/23
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

        if gaussian_sigma is None:
            gaussian_sigma = self.SMFR_SIGMA

        if trial not in self.PopAnalDict.keys() or overwrite==True:
            # Get all spike trains for a trial
            list_sites = self.sitegetterKS_map_region_to_sites_MULTREG(clean=clean_chans)
            list_spiketrain = []
            for site in list_sites:
                dat = self.datall_TDT_KS_slice_single_bysite(site, trial)
                if "spiketrain" not in dat.keys() or dat["spiketrain"] is None:
                    # print("Generating spike train! (site, trial): ", site, trial)
                    self._spiketrain_as_elephant(site, trial, cache=True)
                st = dat["spiketrain"]

                # CANNOT do mods here, becuase is cached version
                # if twind is not None:
                #     start_time = twind[0] * s
                #     end_time = twind[-1] * s
                #     # print(st)
                #     st = st.time_slice(start_time, end_time)
                #     # st = st[(st>=twind[0]) & (st<=twind[1])]
                #     # print(st)
                #     # assert False

                if st is None:
                    print("Trial, site:", trial, site)
                    assert False, "first generate spike trains.."
                list_spiketrain.append(st)
                
            # Convert spike train to smoothed FR
            frate = instantaneous_rate(list_spiketrain, sampling_period=sampling_period*s, 
                kernel=GaussianKernel(gaussian_sigma*s))

            # Convert to popanal
            PA = self._popanal_generate_from_raw(frate.T.magnitude, frate.times, list_sites, df_label_trials=None)
            # PA = PopAnal(frate.T.magnitude, frate.times, chans = list_sites,
            #     spike_trains = [list_spiketrain], print_shape_confirmation=print_shape_confirmation)
            # PA.Params["frate_sampling_period"] = sampling_period
            self.PopAnalDict[trial] = PA

        # Return
        if return_sampling_period:
            return self.PopAnalDict[trial], sampling_period
        else:
            return self.PopAnalDict[trial]


    ###################### SMOOTHED FR
    def smoothedfr_extract_timewindow_during_trial(self, sitenum, trial):
        # Get for a single trial and site, the smoothed fr between two events that flank the main part of the
        # trial when subject is doing things.

        # Get the times for the events that flank the action part of trial.
        t_start, t_end = self.events_get_time_flanking_trial_helper(trial)
        pa, _, _, _ = self.smoothedfr_extract_timewindow_bytimes([trial], [t_start], [sitenum], pre_dur=0., post_dur=t_end-t_start)

        return pa

    def smoothedfr_extract_timewindow_bytimes(self, trials, times, 
        sites, pre_dur=-0.1, post_dur=0.1,
        fail_if_times_outside_existing=True,
        idx_trialtime_all=None,
        method_if_not_enough_time="keep_and_prune_time"):
        """ [GOOD, FLEXIBLE] Extract smoothed fr dataset for these trials and times.
        PARAMS:
        - trials, list of trials in sn.
        - times, list of times(in sec), one for each trial in trials. will extract aligned
        to these times.
        - fail_if_times_outside_existing, bool, if False, then skips trials that are too close to 
        trial edge to extract complete data given the pre_dur andpost_dur you want.
        RETURNS:
        - PopAnal, where PA.X[:, i, :] is the dim of len(trials):
        """
        from quantities import s
        from .population import PopAnal
        from neuralmonkey.classes.population import concatenate_popanals
        
        assert isinstance(pre_dur, (float, int))
        assert isinstance(pre_dur, (float, int))
        assert len(trials)==len(times)

        if idx_trialtime_all is None:
            idx_trialtime_all = range(len(trials)) # useful for tracking if throw out data.

        list_xslices = []
        # 1) extract each trials' PA. Use the slicing tool in PA to extract snippet
        list_trials_actual = []
        list_times_actual = []
        list_idx_trialtime_actual = []
        for tr, time_align, idx in zip(trials, times, idx_trialtime_all):

            # extract popanal
            pa, sampling_period = self.popanal_generate_save_trial(tr, return_sampling_period=True)

            # slice to desired channels
            pa = pa._slice_by_chan(sites) 

            # assert fail_if_times_outside_existing==True, "toehrwise deal with possible change in size of output."
            # Extract snip
            t1 = time_align + pre_dur
            t2 = time_align + post_dur

            # --- This does nothing!! see below.
            # if realign_to_time:
            #     # then make time_align the new 0
            #     subtract_this_from_times = time_align
            # else:
            #     subtract_this_from_times = None

            pa = pa._slice_by_time_window(t1, t2, return_as_popanal=True,
                fail_if_times_outside_existing=fail_if_times_outside_existing,
                subtract_this_from_times=time_align,
                method_if_not_enough_time=method_if_not_enough_time)

            if pa is None:
                # Then skip this
                assert fail_if_times_outside_existing==False, "how else coudl you have gotten None?"
                assert method_if_not_enough_time=="return_none"
                continue
            
            # save this slice
            list_xslices.append(pa)
            list_trials_actual.append(tr)
            list_times_actual.append(time_align)
            list_idx_trialtime_actual.append(idx)

        # 2) Concatenate all PA into a single PA
        # This error is obsolete. Now these datapts are just thrown out entirely.
        # if not fail_if_times_outside_existing:
        #     assert False, "fix this!! if pre_dur extends before first time, then this is incorrect. Should do what?"
        
        # Replace all times with this time relative to alignement.
        for pa in list_xslices:
            # sampling period, to acocunt for random variation in alignment across snips.
            TIMES = (pa.Times - pa.Times[0]) + pre_dur + sampling_period/2 # times all as [-predur, ..., postdur]
            pa.Times = TIMES

        # then concat
        PAall = concatenate_popanals(list_xslices, "trials", 
            assert_otherdims_have_same_values=True, 
            assert_otherdims_restrict_to_these=("chans", "times"),
            all_pa_inherit_times_of_pa_at_this_index=0)

        if PAall is not None:
            # Sanity checks
            assert PAall.Chans ==sites
            assert PAall.X.shape[1] == len(list_trials_actual)

        return PAall, list_trials_actual, list_times_actual, list_idx_trialtime_actual

    def smoothedfr_extract_timewindow_bystroke(self, trials, strokeids, 
        sites, pre_dur=-0.1, post_dur=0.1,
        fail_if_times_outside_existing=True, align_to="onset"):
        """ Extract smoothed fr dataset for these trials and strokeids
        """
        from quantities import s
        from .population import PopAnal

        assert len(trials)==len(strokeids)

        # Collect all event times.
        event_times = []
        for tr, indstrok in zip(trials, strokeids):
            if align_to=="onset":
                _alignto = f"on_strokeidx_{indstrok}"
            elif align_to=="offset":
                _alignto = f"off_strokeidx_{indstrok}"
            else:
                assert False
            time_align = self.events_get_time_using_photodiode(tr,
                list_events=[_alignto])[_alignto]
            time_align = time_align[0] # take first time in list of times.
            event_times.append(time_align)


        PAall, _, _, _ = self.smoothedfr_extract_timewindow_bytimes(trials, event_times,
                sites, pre_dur, post_dur, fail_if_times_outside_existing,
                                                   idx_trialtime_all=None)

        # assert isinstance(pre_dur, (float, int))
        # assert isinstance(pre_dur, (float, int))
        #
        # list_xslices = []
        # # 1) extract each trials' PA. Use the slicing tool in PA to extract snippet
        # for tr, indstrok in zip(trials, strokeids):
        #     # extract popanal
        #     pa, sampling_period = self.popanal_generate_save_trial(tr, return_sampling_period=True)
        #
        #     # slice to desired channels
        #     pa = pa._slice_by_chan(sites)
        #
        #     # slice to time window
        #     if False:
        #         # Then align to onset of stroke that is in DS
        #         # Sanity check (confirm that timing for neural is same as timing saved in dataset)
        #         ons, offs = SNthis.strokes_extract_ons_offs(trial_neural)
        #         timeon_neural = ons[indstrok]
        #         timeoff_neural = offs[indstrok]
        #         timeon = DS.Dat.iloc[ind]["time_onset"]
        #         timeoff = DS.Dat.iloc[ind]["time_offset"]
        #         assert np.isclose(timeon, timeon_neural)
        #         assert np.isclose(timeoff, timeoff_neural)
        #         time_align = timeon
        #     else:
        #         if align_to=="onset":
        #             _alignto = f"on_strokeidx_{indstrok}"
        #         elif align_to=="offset":
        #             _alignto = f"off_strokeidx_{indstrok}"
        #         else:
        #             assert False
        #         time_align = self.events_get_time_using_photodiode(tr,
        #             list_events=[_alignto])[_alignto]
        #         time_align = time_align[0] # take first time in list of times.
        #     t1 = time_align + pre_dur
        #     t2 = time_align + post_dur
        #
        #     assert fail_if_times_outside_existing==True, "toehrwise deal with possible change in size of output."
        #
        #     pa = pa._slice_by_time_window(t1, t2, return_as_popanal=True,
        #         fail_if_times_outside_existing=fail_if_times_outside_existing,
        #         subtract_this_from_times=time_align)
        #
        #     # save this slice
        #     list_xslices.append(pa)
        #
        # # 2) Concatenate all PA into a single PA
        # if not fail_if_times_outside_existing:
        #     assert False, "fix this!! if pre_dur extends before first time, then this is incorrect. Should do what?"
        #
        # # Replace all times with this time relative to alignement.
        # for pa in list_xslices:
        #     # sampling period, to acocunt for random variation in alignment across snips.
        #     TIMES = (pa.Times - pa.Times[0]) + pre_dur + sampling_period/2 # times all as [-predur, ..., postdur]
        #     pa.Times = TIMES
        #
        # # get list of np arrays
        # if False:
        #     TIMES = (list_xslices[0].Times - list_xslices[0].Times[0]) + pre_dur*s # times all as [-predur, ..., postdur]
        #     Xall = np.concatenate([pa.X for pa in list_xslices], axis=1) # concat along trials axis. each one is (nchans, 1, times)
        #     PAall = PopAnal(Xall, TIMES, sites, trials=trials)
        # else:
        #     from neuralmonkey.classes.population import concatenate_popanals
        #
        #     # then concat
        #     PAall = concatenate_popanals(list_xslices, "trials",
        #         assert_otherdims_have_same_values=True,
        #         assert_otherdims_restrict_to_these=("chans", "times"),
        #         all_pa_inherit_times_of_pa_at_this_index=0)

        return PAall

    def smoothedfr_extract_timewindow(self, trials, sites, alignto, 
        pre_dur=-0.1, post_dur=0.1,
        fail_if_times_outside_existing = True,
        method_if_not_enough_time: str = "return_none"):
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


        # Get times
        # Collect all event times.
        event_times = []
        trials_gotten = []
        for tr in trials:

            # Skip trial if doesnt have this event
            has_event = self.events_does_trial_include_all_events(tr, [alignto])
            if has_event:
                time_align = self.events_get_time_helper(alignto, tr, assert_one=False)[0]
                # time_align = self.events_get_time_using_photodiode(tr, list_events=[alignto])[alignto]
                trials_gotten.append(tr)
                event_times.append(time_align)
                if not isinstance(time_align, (float, int)):
                    print(tr, alignto)
                    print(time_align)
                    assert False, "why is not numerical?"

        PAall, _, _, _ = self.smoothedfr_extract_timewindow_bytimes(trials_gotten, event_times,
                sites, pre_dur, post_dur, fail_if_times_outside_existing,
                                                   idx_trialtime_all=None,
                                                    method_if_not_enough_time=method_if_not_enough_time)

        return PAall, trials_gotten, event_times

    def smoothedfr_extract_trials(self, trials, sites=None):
        """ Extract smoothed fr, where each datapt is a single trial (matrix over all sites).
        Ignoreing trying to clip all trials to same length,
        in a dataframe. First gets the PopAnal represntation, if not already gotten, so
        might take a while first time its run
        PARAMS:
        - trials, list of ints,
        - sites, list of ints
        RETURNS:
        - df, pandas dataframe with trial, fr (sites x times), times, as columns
        """
        import pandas as pd

        if sites is None:
            sites = self.sitegetterKS_map_region_to_sites_MULTREG()

        out = []
        for t in trials:
            pa = self.popanal_generate_save_trial(t) # pa.X --> (chans, 1, time)
            # pathis = pa._slice_by_chan(sites) # slice to these sites
            pathis = pa.slice_by_dim_values_wrapper("chans", sites)
            out.append({
                "trial":t,
                "frmat":pathis.X.squeeze(), # (sites, times)
                "times":pathis.Times # (times, )
                })
        return pd.DataFrame(out), sites


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

        if self.Datasetbeh is None or self.datasetbeh_trial_to_datidx(trial) is None:
            # Eitehr dataset doesnt eixst, or this trial doesnt exist
            use_stroke_as_proxy=False

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

    def _get_trials_list_skipped_trials(self, trials):
        """
        Decide if this session has any hard coded trials to skip --- most of the time this is due to 
        losing power in amplifier, or other recording mishaps.
        RETURNS:
        - neural_trials_missing_beh, list of ints, neural trials that should be skipped (else None, if no skipping)
        - do_skip_trials, bool, if True, then this date has trials that are skipped.
        """

        neural_trials_missing_beh = None
        do_skip_trials = True

        # Not sure if this works. It was commented out and then I turned it back on.
        if (int(self.Date))==220609 and self.RecSession==0 and self.Animal=="Pancho":
            neural_trials_missing_beh = [858, 859, 860, 861, 862, 863, 864, 865, 866, 867]
            trials = [t for t in trials if t not in neural_trials_missing_beh]

        elif (int(self.Date))==231206 and self.RecSession==0 and self.Animal=="Diego":
            neural_trials_missing_beh = [551, 552, 553, 554, 555]
            trials = [t for t in trials if t not in neural_trials_missing_beh]

        elif (int(self.Date))==250319 and self.RecSession==1 and self.Animal=="Diego":
            # Lkast good trial was beh sess 2, beh trial 531 (this only applies for chans 1-256)

            if self.Datasetbeh is not None: # Otherwise runs into recursion error. This is totally fine here
                tc_start = (250319, 2, 532)
                tc_end = (250319, 9, 999)
                if True:
                    # Doesnt work all the time, ie.. whend ataset not yet extracted
                    _, trialcodes_bad = self.Datasetbeh.trialcode_extract_rows_within_range(tc_start, tc_end, input_tuple_directly=True)
                else:
                    from pythonlib.tools.stringtools import trialcode_extract_rows_within_range
                    list_trialcode = [self._datasetbeh_trial_to_trialcode_from_raw(t) for t in trials]
                    _, trialcodes_bad = trialcode_extract_rows_within_range(list_trialcode, tc_start, tc_end, input_tuple_directly=True)

                # Convert from trialcodes to neural trial.
                neural_trials_missing_beh = self.datasetbeh_trialcode_to_trial_batch(trialcodes_bad)

                trials = [t for t in trials if t not in neural_trials_missing_beh]
                # print("trialcodes_bad (orig): ", trialcodes_bad)
                # print("trials bad:", neural_trials_missing_beh)
                # print("trials kept: ", trials)
                # # print("trials bad:", [self._datasetbeh_trial_to_trialcode_from_raw(t) for t in neural_trials_missing_beh])
                # # print("trials kept: ", [self._datasetbeh_trial_to_trialcode_from_raw(t) for t in trials])
                # assert False

        elif (int(self.Date))==220614 and self.RecSession==0 and self.Animal=="Pancho":
            # Skip the first neural trial
            neural_trials_missing_beh = [0]
            trials = [t for t in trials if t not in neural_trials_missing_beh]

        elif (int(self.Date))==220621 and self.RecSession==0 and self.Animal=="Pancho":
            # Skip the first neural trial
            neural_trials_missing_beh = [0]
            trials = [t for t in trials if t not in neural_trials_missing_beh]

        elif (int(self.Date))==220827 and self.Animal=="Pancho":
            assert False, "fill this in! See recording log for what trials lost."

        elif (int(self.Date))==230606 and self.Animal=="Pancho":
            assert False, "fill this in! See recording log for what trials lost."

        elif (int(self.Date))==250324 and self.Animal=="Pancho":
            # Batt died at very end of 793. 794 is good.
            assert False, "fill this in! See recording log for what trials lost."

        elif (int(self.Date))==250325 and self.Animal=="Pancho":
            # Batt died (throw out trials 761 to 790, inclusive), and then plugged in.
            assert False, "fill this in! See recording log for what trials lost."

        elif (int(self.Date))==250417 and self.Animal=="Diego":
            # He bit the cable. Also, bad peformance after this. Throw out the trials lost.
            assert False, "fill this in! See recording log for what trials lost."
            
        else:
            # This date is good!
            do_skip_trials = False
        
        return neural_trials_missing_beh, do_skip_trials
    
    def get_trials_list(self, only_if_ml2_fixation_success=False,
        only_if_has_valid_ml2_trial=True, only_if_in_dataset=False,
        events_that_must_include=None, events_that_must_include_in_order=None,
        dataset_input = None, nrand=None, nsub_uniform=None,
                        must_use_cached_trials=False, min_interval_dur=0.,
                        skip_cached_trials_even_if_exist=False):
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
        - must_use_cached_trials, bool

        UPDATE (3/28/24) - Does not load cached trials, but instad recomputes
        """
        if events_that_must_include is None:
            events_that_must_include = []
        if events_that_must_include_in_order is None:
            events_that_must_include_in_order = []

        if not hasattr(self, "_FORCE_GET_TRIALS_ONLY_IN_DATASET"):
            self._FORCE_GET_TRIALS_ONLY_IN_DATASET = False
            self._FORCE_GET_TRIALS_ONLY_IN_DATASET_NTRIALS = len(self.Datasetbeh.Dat)

        if self._FORCE_GET_TRIALS_ONLY_IN_DATASET:
            only_if_in_dataset = True
            if self._FORCE_GET_TRIALS_ONLY_IN_DATASET_NTRIALS > 1.1*len(self.Datasetbeh.Dat):
                print(self._FORCE_GET_TRIALS_ONLY_IN_DATASET_NTRIALS)
                print(len(self.Datasetbeh.Dat))
                print("You have pruned self.Dataset, which means turning on only_if_in_dataset may inadvertently reduce n trials more than you expect")
                print("Figure out why self.Dataset has been pruned.")
                assert False

        if self.Datasetbeh is None or not hasattr(self, "_MapperTrialcode2TrialToTrial") or len(self._MapperTrialcode2TrialToTrial)==0:
            only_if_in_dataset = False

        # print(only_if_in_dataset, self._FORCE_GET_TRIALS_ONLY_IN_DATASET)
        # assert False

        assert not isinstance(only_if_in_dataset, list), "sanity check, becasue I moved order of args..."

        # if only_if_ml2_fixation_success:
        #     # SInce I use dataset to determine if this trial was initiated (has stroke), can only include trials that have data.
        #     only_if_in_dataset = True

        # if only_if_in_dataset:
        #     # Then no need to check the foloowing, they wil be true
        #     only_if_ml2_fixation_success = False
        #     only_if_has_valid_ml2_trial = False

        # A new key, which is fine that this is not in cached, since you are relopading new dataset anyways, so dont want
        # this cached
        # key = (only_if_in_dataset, only_if_ml2_fixation_success, only_if_has_valid_ml2_trial)
        key = (only_if_ml2_fixation_success, only_if_has_valid_ml2_trial)
        if key in self._CachedTrialsList.keys() and not skip_cached_trials_even_if_exist:
            trials = self._CachedTrialsList[key]

            if only_if_in_dataset:
                trials = [t for t in trials if self.datasetbeh_trial_to_datidx(t) is not None]
        else:
            assert must_use_cached_trials==False

            trials = list(range(len(self.TrialsOffset)))

            ############# VERY HACKY,
            neural_trials_missing_beh, do_skip_trials = self._get_trials_list_skipped_trials(trials)
            if neural_trials_missing_beh is not None:
                trials = [t for t in trials if t not in neural_trials_missing_beh]

            # # Not sure if this works. It was commented out and then I turned it back on.
            # if (int(self.Date))==220609 and self.RecSession==0 and self.Animal=="Pancho":
            #     neural_trials_missing_beh = [858, 859, 860, 861, 862, 863, 864, 865, 866, 867]
            #     trials = [t for t in trials if t not in neural_trials_missing_beh]

            # if (int(self.Date))==231206 and self.RecSession==0 and self.Animal=="Diego":
            #     neural_trials_missing_beh = [551, 552, 553, 554, 555]
            #     trials = [t for t in trials if t not in neural_trials_missing_beh]

            # if (int(self.Date))==250319 and self.RecSession==1 and self.Animal=="Diego":
            #     # Lkast good trial was beh sess 2, beh trial 531 (this only applies for chans 1-256)

            #     if self.Datasetbeh is not None: # Otherwise runs into recursion error. This is totally fine here
            #         tc_start = (250319, 2, 532)
            #         tc_end = (250319, 9, 999)
            #         if True:
            #             # Doesnt work all the time, ie.. whend ataset not yet extracted
            #             _, trialcodes_bad = self.Datasetbeh.trialcode_extract_rows_within_range(tc_start, tc_end, input_tuple_directly=True)
            #         else:
            #             from pythonlib.tools.stringtools import trialcode_extract_rows_within_range
            #             list_trialcode = [self._datasetbeh_trial_to_trialcode_from_raw(t) for t in trials]
            #             _, trialcodes_bad = trialcode_extract_rows_within_range(list_trialcode, tc_start, tc_end, input_tuple_directly=True)

            #         # Convert from trialcodes to neural trial.
            #         neural_trials_missing_beh = self.datasetbeh_trialcode_to_trial_batch(trialcodes_bad)

            #         trials = [t for t in trials if t not in neural_trials_missing_beh]
            #         # print("trialcodes_bad (orig): ", trialcodes_bad)
            #         # print("trials bad:", neural_trials_missing_beh)
            #         # print("trials kept: ", trials)
            #         # # print("trials bad:", [self._datasetbeh_trial_to_trialcode_from_raw(t) for t in neural_trials_missing_beh])
            #         # # print("trials kept: ", [self._datasetbeh_trial_to_trialcode_from_raw(t) for t in trials])
            #         # assert False

            # if (int(self.Date))==220614 and self.RecSession==0 and self.Animal=="Pancho":
            #     # Skip the first neural trial
            #     neural_trials_missing_beh = [0]
            #     trials = [t for t in trials if t not in neural_trials_missing_beh]

            # if (int(self.Date))==220621 and self.RecSession==0 and self.Animal=="Pancho":
            #     # Skip the first neural trial
            #     neural_trials_missing_beh = [0]
            #     trials = [t for t in trials if t not in neural_trials_missing_beh]

            # if (int(self.Date))==220827 and self.Animal=="Pancho":
            #     assert False, "fill this in! See recording log for what trials lost."

            # if (int(self.Date))==230606 and self.Animal=="Pancho":
            #     assert False, "fill this in! See recording log for what trials lost."

            # if (int(self.Date))==250324 and self.Animal=="Pancho":
            #     # Batt died at very end of 793. 794 is good.
            #     assert False, "fill this in! See recording log for what trials lost."

            # if (int(self.Date))==250325 and self.Animal=="Pancho":
            #     # Batt died (throw out trials 761 to 790, inclusive), and then plugged in.
            #     assert False, "fill this in! See recording log for what trials lost."

            if only_if_in_dataset:
                # SHould do this first, since if this trial is not in dataset then it will fail only_if_ml2_fixation_success
                trials = [t for t in trials if self.datasetbeh_trial_to_datidx(t) is not None]

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
            if not skip_cached_trials_even_if_exist:
                self._CachedTrialsList[key] = trials

        # if only_if_in_dataset:
        #     trials = [t for t in trials if self.datasetbeh_trial_to_datidx(t) is not None]
        #     # trials_keep = []
        #     # for t in trials:
        #     #     if self.datasetbeh_trial_to_datidx(t, dataset_input=dataset_input) is None:
        #     #         # exclud
        #     #         pass
        #     #     else:
        #     #         trials_keep.append(t)
        #     # trials = trials_keep

        if len(events_that_must_include)>0:
            trials = self._get_trials_list_if_include_these_events(trials, events_that_must_include)
        if len(events_that_must_include_in_order)>0:
            trials = self._get_trials_list_if_include_these_events_in_order(trials, events_that_must_include_in_order, min_interval_dur=min_interval_dur)

        if nrand is not None:
            # take randmo subset, ordered.
            if nrand < len(trials):
                import random
                trials = sorted(random.sample(trials, nrand))

        if nsub_uniform is not None:
            if nsub_uniform < len(trials):
                from pythonlib.tools.listtools import random_inds_uniformly_distributed
                trials = random_inds_uniformly_distributed(trials, nsub_uniform, return_original_values=True)

        if self._DEBUG_PRUNE_TRIALS:
            # Return 20
            n = 20
            from pythonlib.tools.listtools import random_inds_uniformly_distributed
            trials = random_inds_uniformly_distributed(trials, n, return_original_values=True)
            # trials = trials[:10]

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
    
    def _get_trials_list_if_include_these_events_in_order(self, trials, events_that_must_include, min_interval_dur=0.):
        """ only inclues trials that have one and only one of each event, and they are present in chron order
        matching the order in <events_that_must_include>
        RETURNS:
        - trials, those that pass this test.
        """
        dfeventsn, _ = self.eventsdataframe_extract_timings(events_that_must_include, trials)

        # eventinds_in_chron_order
        inds_get = list(range(len(events_that_must_include)))
        inds_bool_keep = [inds==inds_get for inds in dfeventsn["eventinds_in_chron_order"]]
        # display(dfeventsn)
        # print(inds_bool_keep)
        # print(sum(inds_bool_keep), len(inds_bool_keep))
        # assert False

        if min_interval_dur >0:
            list_too_short = []
            for times in dfeventsn["times_ordered_flat"]:
                has_short_interval = np.any(np.diff(times)<min_interval_dur)
                list_too_short.append(has_short_interval)
                # if has_short_interval:
                #     # print(times)
                #     print(np.diff(times))

            inds_bool_keep = [a and not b for a,b in zip(inds_bool_keep, list_too_short)]
        
        # Do pruning
        trials_keep = dfeventsn[inds_bool_keep]["trial"].tolist()

        return trials_keep

    ####################### PLOTS (generic)
    def plot_spike_waveform_site(self, site):

        rs, chan = self.convert_site_to_rschan(site)
        spk = self.load_spike_waveforms_(rs, chan) # (nspk, ntimebins), e.g., (1000,30)

        fig, ax = plt.subplots(1,1)
        self.plot_spike_waveform(ax, spk)
        return fig

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
        list_sites = self.sitegetterKS_map_region_to_sites_MULTREG(clean=False)

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
        list_sites = self.sitegetterKS_map_region_to_sites_MULTREG(clean=clean)

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

    def plot_raster_sites(self, ax, trial, list_sites=None, site_to_highlight=None,
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
            list_sites = self.sitegetterKS_map_region_to_sites_MULTREG()

        for i, site in enumerate(list_sites):
            d = self.datall_TDT_KS_slice_single_bysite(site, trial)
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

    def plot_raster_trials(self, ax, list_trials, site, alignto=None,
        raster_linelengths=0.9, alpha_raster = 0.9, overlay_trial_events=True,
        ylabel_trials=True, plot_rasters=True, xmin = None, xmax = None,
        overlay_strokes=True):
        """ Plot raster, for these trials, on this axis.
        PARAMS:
        - list_trials, list of indices into self. will plot them in order, from bottom to top
        """

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
                
            if plot_rasters: # THis hould be independento f overlay_trial_egents, since latter I use for preprocessing, to make plots of events.
                # Rasters
                # rs, chan = self.convert_site_to_rschan(site)
                D = self.datall_TDT_KS_slice_single_bysite(site, trial)
                spikes = D["spike_times"]
                # print(spikes)
                self._plot_raster_line(ax, spikes, i, alignto_time=alignto_time, 
                    linelengths=raster_linelengths, alpha=alpha_raster)

        if overlay_trial_events:
            self.plotmod_overlay_trial_events_mult(ax, list_trials, list_align_time,
                ylabel_trials, xmin=xmin, xmax=xmax, overlay_strokes=overlay_strokes)
        
        if site is not None:
            ax.set_title(self.sitegetter_summarytext(site)) 


    def _plot_raster_create_figure_blank(self, duration, n_raster_lines, n_subplot_rows=1,
            nsubplot_cols=1, reduce_height_for_sm_fr=False, sharex=True, sharey=False,
                                         force_scale_height=None):
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

        if force_scale_height:
            height = force_scale_height*height

        if False:
            print(n_raster_lines, height_cell, n_subplot_rows)
            print(duration, width_cell)
            print(width, height, aspect)
            # assert False
        fig, axes = plt.subplots(n_subplot_rows, nsubplot_cols, sharex=sharex, sharey=sharey,
            figsize = (width, height), squeeze=False)

        kwargs = {
            "alpha_raster":0.7
        }
        return fig, axes, kwargs


    def _plot_raster_line(self, ax, times, yval, color='k', alignto_time=None,
        linelengths = 0.85, alpha=0.4, linewidths=None):
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
        ax.eventplot([t], lineoffsets=yval, color=color, alpha=alpha, linelengths=linelengths,
                     linewidths=linewidths)
        
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
        list_regions = [self.sitegetterKS_thissite_info(site)["region"] for site in list_sites]
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

        # subsample trials to label
        if ylabel_trials is None or ylabel_trials==True:
            ylabel_trials = list_trials
        else:
            assert len(ylabel_trials)==len(list_trials)

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
            # @KGG hack 231026 - appears to be a bug with plotting strokes, so turned off.
            # overlay_strokes = True
            if overlay_strokes:
                ALPHA_STROKES = 0.8*ALPHA_MARKERS
                self.plotmod_overlay_trial_events(ax, trial, alignto_time=alignto_time, 
                                                YLIM=[yval-0.4, yval-0.3], which_events=["strokes"], alpha=ALPHA_STROKES,
                                                xmin = xmin, xmax =xmax)

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
                    if ev in color_map:
                        colors_codes.append(color_map[ev])
                    else:
                        # use a default color
                        colors_codes.append("r")            


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
            # print("--------")
            # print("ons", ons)
            # print("offs", offs)
            if alignto_time:
                ons = [o - alignto_time for o in ons]
                offs = [o - alignto_time for o in offs]
            # print("alignto_time", alignto_time)
            # print("ons", ons)
            # print("offs", offs)
            # print(YLIM)

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

    def beh_check_touching_fixation_square_during_onset(self, trial, PRE=0.1, POST=0.1):
        """ Returns True if finger was on fixation square when it turned on,
        with pre and post time considered. Uses pd time for fixcue onset.
        PARAMS:
        - PRE, POST, finger must be on cue before and after its onset.
        (PRE 0.1 means before cue onset).
        """

        # Get time of fixation cue onset
        tmp = self.events_get_time_helper("fixcue", trial)
        assert len(tmp)==1
        time_fixcue = tmp[0]

        # Get times of touch of fixation
        times, touch = self.beh_extract_touch_in_fixation_square(trial, ploton=False)

        # if touch within fixation spans this time, then he was touching before onset of fixation cue
        # idx_mid = np.argmin(np.abs(times - time_fixcue)) 
        idx_first = np.argmin(np.abs(times - (time_fixcue-PRE))) 
        idx_last = np.argmin(np.abs(times - (time_fixcue+POST))) 

        touching_fixcue_during_fixcue_on = np.all(touch[idx_first:idx_last])

        return touching_fixcue_during_fixcue_on

    def beh_extract_touch_in_fixation_square(self, trial, window_delta_pixels = None,
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

        if window_delta_pixels is None:
            if self.Animal=="Pancho":
                # window_delta_pixels = 45 # changed to 45 on 6/15/23, since was missing some.
                window_delta_pixels = 52 # changed to 52 on 9/22/23, because now allowing him to hold finger on screen
            elif self.Animal == "Diego":
                window_delta_pixels = 52

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

    def beh_this_day_uses_done_button(self):
        """
        REturn bool if any trial on this dayuses done button (based on checking params)
        """
        for trial in self.get_trials_list():    
            fd, t = self.beh_get_fd_trial(trial)
            if mkl.getTrialsDoneButtonMethod(fd, t)=="skip":
                # Then no done button, keep chekcing
                continue
            else:
                # Found a trial with done button
                return True
        # No trial uses done button
        return False

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
        if DATASETBEH_CACHED_USE_BEHTOUCH: # Then use cached data. Doing this since it is not lile;ly to change
            # in datset. but this is not ideal
            if trial in self._CachedTouchData.keys():
                return self._CachedTouchData[trial]
            else:
                print("WARNING 1 - the touch times might have gaps, esp during hold at fixation for Diego...")
                # Should change base code in drawmonkey, updating how extract touch data (concat touches that are close).
                fd, trialml = self.beh_get_fd_trial(trial)
                xyt = mkl.getTrialsTouchData(fd, trialml)
                # times, touching = mkl.getTrialsTouchingBinary(fd, trialml)
                return xyt
                # return xyt[:,2], xyt[:,0], xyt[:,1]
        else:
            print("WARNING 2    - the touch times might have gaps, esp during hold at fixation for Diego...")
            # Should change base code in drawmonkey, updating how extract touch data (concat touches that are close).
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

    def _beh_extract_eye_raw(self, trial):
        """ Extract just raw eye track signal
        RETURNS:
            - times_tdt, vals_tdt, fs_tdt, times_ml2, vals_ml2, fs_ml2
        """
        if len(self.BehFdList)==0:
            self.load_behavior()

        # Load ml2 analog data
        # Load calibrated from monkeylogic
        fd, fd_trialnum = self.beh_get_fd_trial(trial)
        # mkl.getTrialsAnalogData(fd, fd_trialnum, "Touch")
        dat_ml2 = mkl.getTrialsEyeData(fd, fd_trialnum, return_as_pixels=True)
        times_ml2 = dat_ml2[:,2]
        vals_ml2 = dat_ml2[:,:2]
        fs_ml2 = fd["params"]["sample_rate"]

        # Load voltage from tdt
        tx, vals_x, fs_tdt = self.extract_data_tank_streams("eyex", trial)
        ty, vals_y, fs_tdt = self.extract_data_tank_streams("eyey", trial)
        assert np.all(tx==ty)
        vals_tdt = np.stack([vals_x, vals_y], axis=0).T # (times, 2)
        times_tdt = tx

        return times_tdt, vals_tdt, fs_tdt, times_ml2, vals_ml2, fs_ml2, fd

    def _beh_extract_eye_calibrate_raw(self, vals_tdt, fd):
        """ Given raw TDT signal, calibrate to ml2, return as pixels.
        RETURNS:
            - vals_tdt_calibrated, (N,2)
        """
        T = fd["MLConfig"]["EyeTransform"]["5"]["tdata"]["T"]
        tmp = np.ones((vals_tdt.shape[0], 1))
        vals_tdt_ones = np.concatenate([vals_tdt, tmp], axis=1)
        vals_tdt_calibrated = (T@vals_tdt_ones.T).T
        vals_tdt_calibrated = vals_tdt_calibrated/vals_tdt_calibrated[:, 2][:, None]
        vals_tdt_calibrated = vals_tdt_calibrated[:,:2] # remove last column of ones.
        vals_tdt_calibrated = mkl.convertDeg2PixArray(fd, vals_tdt_calibrated)
        return vals_tdt_calibrated

    def _beh_extract_eye_return_alignment_offset(self):
        """
        Compute offset that can use to align ey etracking data between ml2 and tdt. Offset
        applies to raw voltage trace. Optimizes by minimizing distance in pixel space.
        Does this over some n trials, then takes average.
        Thjen does sanity check that this offset works, thru n random subset of trials.
        :return:
        - mean offset, (2,). Caches in self._BehEyeAlignOffset
        """

        if self._BehEyeAlignOffset is None:
            from pythonlib.tools.stroketools import strokesInterpolate2
            n = 30
            trials = self.get_trials_list(True, nsub_uniform=n+1)[1:] # skip first sometimes is weird
            print("trials, for computing a global offset to align tdt and ml2 eye tracking data:", trials)
            list_x =[]
            for t in trials:
                times_tdt, vals_tdt, fs_tdt, times_ml2, vals_ml2, fs_ml2, fd = self._beh_extract_eye_raw(t)
                times_tdt_raw = times_tdt[:,None]

                def F(xs):
                    assert len(xs)==2
                    vals_tdt_offset = vals_tdt+xs

                    # calibrate to ml2 (pix)
                    vals_tdt_calibrated = self._beh_extract_eye_calibrate_raw(vals_tdt_offset, fd)

                    # interpolate to times of ml2
                    strokes_tdt_raw = [np.concatenate([vals_tdt_calibrated, times_tdt_raw], axis=1)]
                    strokes_tdt = strokesInterpolate2(strokes_tdt_raw, ["input_times", times_ml2], plot_outcome=False)
                    vals_tdt_calibrated_atml2times = strokes_tdt[0][:,:2]

                    # distance from actual ml2.
                    d1 = np.mean((vals_ml2[:,0] - vals_tdt_calibrated_atml2times[:,0])**2)
                    d2 = np.mean((vals_ml2[:,1] - vals_tdt_calibrated_atml2times[:,1])**2)

                    return d1 + d2

                from scipy.optimize import minimize
                res = minimize(F, [0,0])
                # print(res)
                list_x.append(res.x)

            # Take average across trials
            offsets = np.stack(list_x, axis=0) # (n,2)
            self._BehEyeAlignOffset = np.median(offsets, axis=0)
            assert np.all(np.abs(self._BehEyeAlignOffset)<0.08), "this seems high..."

            # Quickly run sanity check over subset of trials, to check this is good.
            apply_empirical_offset = True
            PLOT = False
            CHECK_TDT_ML2_MATCH = True
            trials_test = self.get_trials_list(True, nsub_uniform=50)[1:]
            for trial in trials_test:
                times_tdt, vals_tdt_calibrated = self.beh_extract_eye_good(trial,
                                                                           CHECK_TDT_ML2_MATCH=True,
                                                                           apply_empirical_offset=True)
            print("GOOD! passed sanity check")

        return self._BehEyeAlignOffset

    def beh_eye_fixation_extract_and_assign_task_shape(self, trial, PLOT=False, event_endpoints=None, 
                                                       return_fig=False, fig_savedir=None):
        """
        Good helper to extract fixations for a trial, optionally within a range of events, and assigning the shape he is looking at, with
        constraints on closeness in distance, and in time from onset of first event (i.e, in getting clean fixations).
        
        """
        import pandas as pd
        from pythonlib.tools.distfunctools import closest_pt_twotrajs

        if fig_savedir is not None:
            return_fig = True

        # Params - criteria for assigining a shape to fiaation, based on distance
        MIN_DIST_TASK_TO_FIX = 70 # radius (from closest point along stroke stroke task image), fixation must be within this to assign to this shape (L2)

        # Get data for this trial
        tmp = self.events_get_clusterfix_fixation_times_and_centroids(trial, plot_overlay_on_raw=PLOT,
                                                                                                    event_endpoints =event_endpoints)
        ind = self.datasetbeh_trial_to_datidx(trial)
        Tk = self.Datasetbeh.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True)

        if PLOT:
            data_times, data_centroids, fig1d, axes1d, fig2d, axes2d = tmp
        else:
            data_times, data_centroids = tmp

        if PLOT:
            # overlay locations of each shape
            ax = axes1d.flatten()[0]
            centers = [tk["center"] for tk in Tk.Tokens]
            if False: # This is not accurate anymore, since useing al;l pts, not just centroid
                for cen in centers:
                    ax.axhline(cen[0], color="b")
                    ax.axhline(cen[0]-MIN_DIST_TASK_TO_FIX, color="b", linestyle="--")
                    ax.axhline(cen[0]+MIN_DIST_TASK_TO_FIX, color="b", linestyle="--")
                
            ax = axes1d.flatten()[1]
            centers = [tk["center"] for tk in Tk.Tokens]
            if False:
                for cen in centers:
                    ax.axhline(cen[1], color="r")
                    ax.axhline(cen[1]-MIN_DIST_TASK_TO_FIX, color="r", linestyle="--")
                    ax.axhline(cen[1]+MIN_DIST_TASK_TO_FIX, color="r", linestyle="--")

        if fig_savedir is not None:
            savefig(fig1d, f"{fig_savedir}/timecourse.pdf")
            savefig(fig2d, f"{fig_savedir}/overlaid_image.pdf")
            
        # For each fixation, get its distance to task shapes.
        list_dist_to_closest_token = []
        list_idx_closest_token = []
        list_distance_to_each_token = []
        for fix_cen in data_centroids:

            # get distance to each token
            dist_to_each_token = []
            dist_to_each_token_gridloc = {}
            for tk in Tk.Tokens:
                pts = tk["Prim"].Stroke()[:,:2]
                #  - also inclue to centroid, since he may be looking in center of circle...
                pts = np.concatenate([pts, np.array(tk["center"])[None, :]],axis=0)
                mindist, _, _ = closest_pt_twotrajs(pts, fix_cen[None, :])
                dist_to_each_token.append(mindist)
                dist_to_each_token_gridloc[tk["gridloc"]] = mindist

            # find the closest token
            idx_min = np.argmin(dist_to_each_token)
            val_min = np.min(dist_to_each_token)

            list_dist_to_closest_token.append(val_min)
            list_idx_closest_token.append(idx_min)
            list_distance_to_each_token.append(dist_to_each_token_gridloc)
        
        if PLOT:
            # Overlay results
            fig, ax = plt.subplots()
            self.plot_taskimage(ax=ax, trialtdt=trial)

            # ax.plot(np.stack(data_centroids)[:, 0], np.stack(data_centroids)[:, 1], "-k", alpha=0.4)
            ax.plot(np.stack(data_centroids)[:, 0], np.stack(data_centroids)[:, 1], "-k", alpha=0.4)
            ax.scatter(np.stack(data_centroids)[:, 0], np.stack(data_centroids)[:, 1], c=list_idx_closest_token)
            for i, (fix_cen, idx_closest, dist_closest) in enumerate(zip(data_centroids, list_idx_closest_token, list_dist_to_closest_token)):
                ax.text(fix_cen[0], fix_cen[1], f"#{i}-dist={dist_closest:.1f}")
                if dist_closest>MIN_DIST_TASK_TO_FIX:
                    ax.plot(fix_cen[0], fix_cen[1], "xr")

            if fig_savedir:
                savefig(fig, f"{fig_savedir}/overlaid_image_clean.pdf")
            
        # Get final list of good fixations
        res = []
        for i in range(len(data_centroids)):
            if (list_dist_to_closest_token[i]<MIN_DIST_TASK_TO_FIX):
                # then assign the shape it was lookinga t
                tk = Tk.Tokens[list_idx_closest_token[i]]
                # - pull out useful things
                shape = tk["shape"]
                gridloc = tk["gridloc"]
                idx_task_orig = tk["ind_taskstroke_orig"]
            else:
                shape = "FAR_FROM_ALL_SHAPES"
                gridloc = "FAR_FROM_ALL_SHAPES"
                idx_task_orig = "FAR_FROM_ALL_SHAPES"
            
            res.append({
                "idx_fixation":i,
                "time_global":data_times[i],
                "fix_cen":data_centroids[i],
                "closest_task_token_dist":list_dist_to_closest_token[i],
                "closest_task_token_idx":list_idx_closest_token[i],
                "assigned_task_shape":shape,
                "assigned_task_gridloc":gridloc,
                "assigned_task_idx_task_orig":idx_task_orig,
                "assigned_task_token":tk,
                "distance_to_each_token":list_distance_to_each_token[i],
            })

        dffix = pd.DataFrame(res)

        if return_fig:
            return dffix, fig, ax
        else:
            return dffix


    # def beh_eye_fixation_extract_and_assign_task_shape(self, trial, PLOT=False, event_endpoints=None):
    #     """
    #     Good helper to extract fixations for a trial, optionally within a range of events, and assigning the shape he is looking at, with
    #     constraints on closeness in distance, and in time from onset of first event (i.e, in getting clean fixations).
        
    #     """
    #     import pandas as pd
    #     from pythonlib.tools.distfunctools import closest_pt_twotrajs

    #     assert event_endpoints == ["stim_onset", "go"], "only coded for this, since it uses the time from stim onset to fixation to decide whether to throw it out.."

    #     # Params - criteria for assigining a shape to fiaation, based on distance
    #     MIN_DIST_TASK_TO_FIX = 70 # radius (from closest point along stroke stroke task image), fixation must be within this to assign to this shape (L2)
    #     MIN_TIME_REL_STIM_ONSET = 0.15 # (saccades take ~0.05-0.1 sec. Account for 0.1 reaction time)

    #     # Get data for this trial
    #     t_stim_onset = self.events_get_time_helper("stim_onset", trial, assert_one=True)[0]
    #     tmp = self.events_get_clusterfix_fixation_times_and_centroids(trial, plot_overlay_on_raw=PLOT,
    #                                                                                                 event_endpoints =event_endpoints)
    #     ind = self.datasetbeh_trial_to_datidx(trial)
    #     Tk = self.Datasetbeh.taskclass_tokens_extract_wrapper(ind, "task", return_as_tokensclass=True)

    #     if PLOT:
    #         data_times, data_centroids, fig1d, axes1d, fig2d, axes2d = tmp
    #     else:
    #         data_times, data_centroids = tmp

    #     if PLOT:
    #         # overlay locations of each shape
    #         ax = axes1d.flatten()[0]
    #         centers = [tk["center"] for tk in Tk.Tokens]
    #         if False: # This is not accurate anymore, since useing al;l pts, not just centroid
    #             for cen in centers:
    #                 ax.axhline(cen[0], color="b")
    #                 ax.axhline(cen[0]-MIN_DIST_TASK_TO_FIX, color="b", linestyle="--")
    #                 ax.axhline(cen[0]+MIN_DIST_TASK_TO_FIX, color="b", linestyle="--")
                
    #         ax = axes1d.flatten()[1]
    #         centers = [tk["center"] for tk in Tk.Tokens]
    #         if False:
    #             for cen in centers:
    #                 ax.axhline(cen[1], color="r")
    #                 ax.axhline(cen[1]-MIN_DIST_TASK_TO_FIX, color="r", linestyle="--")
    #                 ax.axhline(cen[1]+MIN_DIST_TASK_TO_FIX, color="r", linestyle="--")

    #     # For each fixation, get its distance to task shapes.
    #     list_dist_to_closest_token = []
    #     list_idx_closest_token = []
    #     list_distance_to_each_token = []
    #     for fix_cen in data_centroids:

    #         # get distance to each token
    #         dist_to_each_token = []
    #         dist_to_each_token_gridloc = {}
    #         for tk in Tk.Tokens:
    #             pts = tk["Prim"].Stroke()[:,:2]
    #             #  - also inclue to centroid, since he may be looking in center of circle...
    #             pts = np.concatenate([pts, np.array(tk["center"])[None, :]],axis=0)
    #             mindist, _, _ = closest_pt_twotrajs(pts, fix_cen[None, :])
    #             dist_to_each_token.append(mindist)
    #             dist_to_each_token_gridloc[tk["gridloc"]] = mindist

    #         # find the closest token
    #         idx_min = np.argmin(dist_to_each_token)
    #         val_min = np.min(dist_to_each_token)

    #         list_dist_to_closest_token.append(val_min)
    #         list_idx_closest_token.append(idx_min)
    #         list_distance_to_each_token.append(dist_to_each_token_gridloc)

    #     # For each fixatoin, get its time relative to stim onset.
    #     assert event_endpoints == ["stim_onset", "go"], "only coded for this, since it uses the time from stim onset to fixation to decide whether to throw it out.."
    #     list_time_relative_stim_onset = [t - t_stim_onset for t in data_times]
        
    #     if PLOT:
    #         # Overlay results
    #         fig, ax = plt.subplots()
    #         self.plot_taskimage(ax=ax, trialtdt=trial)

    #         # ax.plot(np.stack(data_centroids)[:, 0], np.stack(data_centroids)[:, 1], "-k", alpha=0.4)
    #         ax.plot(np.stack(data_centroids)[:, 0], np.stack(data_centroids)[:, 1], "-k", alpha=0.4)
    #         ax.scatter(np.stack(data_centroids)[:, 0], np.stack(data_centroids)[:, 1], c=list_idx_closest_token)
    #         for i, (fix_cen, idx_closest, dist_closest, t_rel_stim) in enumerate(zip(data_centroids, list_idx_closest_token, list_dist_to_closest_token, list_time_relative_stim_onset)):
    #             ax.text(fix_cen[0], fix_cen[1], f"#{i}-dist={dist_closest:.1f}-trelstim={t_rel_stim:.2f}")
    #             if dist_closest>MIN_DIST_TASK_TO_FIX:
    #                 ax.plot(fix_cen[0], fix_cen[1], "xr")
    #             if t_rel_stim<MIN_TIME_REL_STIM_ONSET:
    #                 ax.plot(fix_cen[0], fix_cen[1], "dr")
                        
    #     # Get final list of good fixations
    #     res = []
    #     for i in range(len(data_centroids)):
    #         if (list_time_relative_stim_onset[i]>MIN_TIME_REL_STIM_ONSET) & (list_dist_to_closest_token[i]<MIN_DIST_TASK_TO_FIX):
    #             # then assign the shape it was lookinga t
    #             tk = Tk.Tokens[list_idx_closest_token[i]]
    #             # - pull out useful things
    #             shape = tk["shape"]
    #             gridloc = tk["gridloc"]
    #             idx_task_orig = tk["ind_taskstroke_orig"]
    #         else:
    #             shape = "FAR_FROM_ALL_SHAPES"
    #             gridloc = "FAR_FROM_ALL_SHAPES"
    #             idx_task_orig = "FAR_FROM_ALL_SHAPES"
            
    #         res.append({
    #             "idx_fixation":i,
    #             "time_global":data_times[i],
    #             "fix_cen":data_centroids[i],
    #             "time_rel_stim_onset":list_time_relative_stim_onset[i],
    #             "closest_task_token_dist":list_dist_to_closest_token[i],
    #             "closest_task_token_idx":list_idx_closest_token[i],
    #             "assigned_task_shape":shape,
    #             "assigned_task_gridloc":gridloc,
    #             "assigned_task_idx_task_orig":idx_task_orig,
    #             "assigned_task_token":tk,
    #             "distance_to_each_token":list_distance_to_each_token[i],
    #         })

    #     dffix = pd.DataFrame(res)

    #     return dffix


    def beh_eye_fixation_task_shape_overlay_plot(self, trial, ax, event_endpoints=None, 
                                                 map_shape_to_y=None, 
                                                 map_shape_to_col=None,
                                                 yplot=0, plot_vlines=True, vlines_alpha=0.5):
        """
        Extract each fixation's assigned shape, and overlay this on any plot (ax) as horiz lines at bottom of
        plot, whos y-coord and color indicates the matched shape, and where x marks fixations, and lines mark the times 
        of ongoing fixation. 
        I used this for moment by mmoment decoding.
        PARAMS:
        - map_shape_to_y, dict mappiong from shape label to y location.
        - map_shape_to_col, dict mappiong from shape label to color (4-d array)
        """

        # Extract fixation data
        dffix = self.beh_eye_fixation_extract_and_assign_task_shape(trial, PLOT=False, event_endpoints=event_endpoints)
        shapes_exist = dffix["assigned_task_shape"].unique().tolist()

        # Decode how to map shape labels to y and to color
        if map_shape_to_y is None:
            if False: # Each shape is diff y coordinate
                map_shape_to_y = {sh:i for i, sh in enumerate(shapes_exist)}
            else:
                # Move all to same Y
                map_shape_to_y = {sh:yplot for sh in shapes_exist}

        if map_shape_to_col is None:
            from pythonlib.tools.plottools import color_make_map_discrete_labels
            map_shape_to_col = color_make_map_discrete_labels(shapes_exist)[0]
            # map_shape_to_col["FAR_FROM_ALL_SHAPES"] = np.array([0.8, 0.8, 0.8, 1.])

        # Plot lines between each successive fixation
        dffix = dffix.sort_values("time_global")
        for i in range(len(dffix)):
            if i+1<len(dffix):    
                t1 = dffix.iloc[i]["time_global"]
                shape = dffix.iloc[i]["assigned_task_shape"]
                t2 = dffix.iloc[i+1]["time_global"]
                y = map_shape_to_y[shape]
                col = map_shape_to_col[shape]
                ax.plot([t1, t2], [y, y], "-x", color=col, alpha=0.8)

                if plot_vlines:
                    ax.axvline(t1, color=col, alpha=vlines_alpha, linestyle=":")
        from pythonlib.tools.plottools import legend_add_manual
        legend_add_manual(ax, map_shape_to_col.keys(), map_shape_to_col.values())
        
        return dffix, map_shape_to_y, map_shape_to_col

    def beh_plot_eye_raw_overlay_good(self, trial, ax):
        """
        QUick helper, to load raw fixation data, and plot overlaid on any axis.
        """
        times_tdt, vals_tdt_calibrated, fs_tdt, vals_tdt_calibrated_sm = self.beh_extract_eye_good(trial, 
                                                                                            CHECK_TDT_ML2_MATCH=True, 
                                                                                            return_all=True, PLOT=False)
        ax.plot(times_tdt, vals_tdt_calibrated[:,0], label="tdt_x", color="b")
        ax.plot(times_tdt, vals_tdt_calibrated[:,1], label="tdt_y", color="r")
        ax.legend()

    def beh_extract_eye_good(self, trial, apply_empirical_offset=True,
        CHECK_TDT_ML2_MATCH=False, THRESH=5, PLOT=False, return_all=False,
        SM_WIN = 0.01, return_fs_tdt=False):
        """
        Get eye track data in units of pixels (matching strokes), by using
        voltage saved in TDT, but doing projective transform using the T
        matrix saved in monkeylogic. Do this instaed of using saved ml2 values, 
        becuase the latter does not extend over the entire trial.
        PARAMS:
        - apply_empirical_offset, bool, if True, then this helps the match between
        ml2 and tdt. not sure why...
        - CHECK_TDT_ML2_MATCH, bool, if true, then asserts that the diff between tdt and 
        ml2 (after calibrating tdt) are below a threshold (pixel rms). This takes more time,
        so I only run once when checking the extracted offset.
        RETURNS:
        - times_tdt, (ntimes, ) array of times
        - vals_tdt_calibrated, (ntimes, 2) array of x and y coords, pixels
        NOTES:
        - First trial (0) can have slight offset between ml2 and neural. I measured 10ms (earlier for neural
        on 230603, Diego.
        """

        times_tdt, vals_tdt, fs_tdt, times_ml2, vals_ml2, fs_ml2, fd = self._beh_extract_eye_raw(trial)

        # Apply empriical offset, which leads to better alignement between calibrated
        # tdt data vs. saved ml2.
        if apply_empirical_offset:
            if False:
                # assert False, "not ready. this requires a  different offset for each session. should compute the offset by converting ml2 back to voltage"
                # vals_tdt = vals_tdt- [0.04,0.04] # Emprical offset...
                offset = [-0.04,-0.04] # Emprical offset...
            else:
                # Find it by optimization
                offset = self._beh_extract_eye_return_alignment_offset()
            vals_tdt = vals_tdt + offset

        # Try to transform using calibration matrix
        vals_tdt_calibrated = self._beh_extract_eye_calibrate_raw(vals_tdt, fd)

        # T = fd["MLConfig"]["EyeTransform"]["5"]["tdata"]["T"]
        # tmp = np.ones((vals_tdt.shape[0], 1))
        # vals_tdt_ones = np.concatenate([vals_tdt, tmp], axis=1)
        # vals_tdt_calibrated = (T@vals_tdt_ones.T).T
        # # normalize by last column
        # vals_tdt_calibrated = vals_tdt_calibrated/vals_tdt_calibrated[:, 2][:, None]
        # vals_tdt_calibrated = vals_tdt_calibrated[:,:2] # remove last column of ones.
        # # print("RESULT:")
        # # print(vals_tdt_calibrated[:5])
        # # print(vals_tdt_calibrated.shape)
        #
        # # COnvert from degress to pixels
        # vals_tdt_calibrated = mkl.convertDeg2PixArray(fd, vals_tdt_calibrated)

        # plot
        if PLOT:
            # Plot and overlay
            fig, axes = plt.subplots(4,1, figsize=(15,20))

            ax = axes.flatten()[0]
            ax.plot(times_ml2, vals_ml2, label="ml2")
            ax.plot(times_tdt, vals_tdt, label="tdt")
            ax.plot(times_tdt, vals_tdt_calibrated, label="tdt_calib")
            ax.legend()

            ax = axes.flatten()[1]
            ax.plot(times_ml2, vals_ml2, label="ml2")
            # ax.plot(times_tdt, vals_tdt, label="tdt")
            ax.plot(times_tdt, vals_tdt_calibrated, label="tdt_calib")
            ax.legend()

            ax = axes.flatten()[2]
            # ax.plot(times_ml2, vals_ml2, label="ml2")
            ax.plot(times_tdt, vals_tdt, label="tdt")
            ax.plot(times_tdt, vals_tdt_calibrated, label="tdt_calib")
            ax.legend()

        if CHECK_TDT_ML2_MATCH: 
            # assert False, "not ready. this requires a  different offset for each session. should compute the offset by converting ml2 back to voltage"
            # Sanity check that (i) tdt voltage transformed --> pix == (ii) monkeylogic saved.
            # - interpolate to same time base
            from pythonlib.tools.stroketools import strokesInterpolate2, smoothStrokes

            times_tdt_raw = times_tdt[:,None]
            strokes_tdt_raw = [np.concatenate([vals_tdt_calibrated, times_tdt_raw], axis=1)]
            strokes_tdt = smoothStrokes(strokes_tdt_raw, fs_tdt, 0.05, sanity_check_endpoint_not_different=False)
            strokes_tdt = strokesInterpolate2(strokes_tdt, ["input_times", times_ml2], plot_outcome=PLOT)
            vals_tdt_calibrated_atml2times_sm = strokes_tdt[0][:,:2]

            # smooth the ml2 version
            strokes_ml2_raw = [np.concatenate([vals_ml2, times_ml2[:,None]], axis=1)]
            strokes_ml2 = smoothStrokes(strokes_ml2_raw, fs_ml2, 0.05, sanity_check_endpoint_not_different=False)
            vals_ml2_sm = strokes_ml2[0][:,:2]

            diff_ml2_tdt_rms = np.sum((vals_tdt_calibrated_atml2times_sm - vals_ml2_sm)**2, axis=1)**0.5
            try:
                diff_floor = np.percentile(diff_ml2_tdt_rms[int(fs_ml2/2):-int(fs_ml2/2)], [10]) # skip the firs tand last 500 samp, which is usualyl one sec
            except Exception as err:
                diff_floor = np.percentile(diff_ml2_tdt_rms, [10])[0] # skip the firs tand last 500 samp, which is usualyl one sec
            print("This is 10th percentile of rms diff between ml2 and tdt pix values.. lower the better")
            print(diff_floor)

            # Compute offest
            # from pythonlib.tools.nptools import optimize_offset_to_align_tdt_ml2
            # print("offsets")
            # for i in [0,1]:
            #     offset = optimize_offset_to_align_tdt_ml2(vals_tdt_calibrated_atml2times_sm[:,i],
            #                                      vals_ml2_sm[:,i])
            #     print(i, offset)
            # # add the offsets

            if PLOT:
                fig, axes = plt.subplots(4,1, figsize=(20,12))

                ax = axes.flatten()[0]
                ax.plot(times_ml2, vals_ml2_sm, label="ml2")
                ax.plot(times_ml2, vals_tdt_calibrated_atml2times_sm, label="tdt_calib_interp_to_ml2_times")
                ax.legend()


                ax = axes.flatten()[1]
                ax.plot(times_ml2, diff_ml2_tdt_rms, label="diff_ml2_tdt_rms")
                ax.legend()

                # ZOOM IN
                ax = axes.flatten()[2]
                inds_ = range(1800, 2000)
                ax.plot(times_ml2[inds_], vals_ml2_sm[inds_], label="ml2")
                ax.plot(times_ml2[inds_], vals_tdt_calibrated_atml2times_sm[inds_], label="tdt_calib_interp_to_ml2_times")
                ax.legend()

                ax = axes.flatten()[3]
                ax.plot(times_ml2, vals_ml2, label="ml2_nosm")
                ax.plot(times_tdt_raw, vals_tdt_calibrated, label="tdt_calib_nosm_nointerp")
                ax.legend()

            if diff_floor>THRESH:
                print(diff_floor, THRESH)
                assert False, "empriclaly, this is too high..."

        if return_all:
            # smooth the trace
            from pythonlib.tools.stroketools import strokesInterpolate2, smoothStrokes            
            strokes_tdt = [np.concatenate([vals_tdt_calibrated, times_tdt[:,None]], axis=1)]
            strokes_tdt = smoothStrokes(strokes_tdt, fs_tdt, SM_WIN, sanity_check_endpoint_not_different=False)
            vals_tdt_calibrated_sm = strokes_tdt[0][:,:2]
            times_tdt_sm = strokes_tdt[0][:,2]
            from pythonlib.tools.nptools import isnear
            assert isnear(times_tdt_sm, times_tdt)
            return times_tdt, vals_tdt_calibrated, fs_tdt, vals_tdt_calibrated_sm
        elif return_fs_tdt:
            return times_tdt, vals_tdt_calibrated, fs_tdt
        else:
            return times_tdt, vals_tdt_calibrated


    def strokes_task_extract(self, trial):
        """ Extract the strokes for this task
        """
        if True:
            # New, always get from dataset, this removes any source of misalignmenet between beh and neural
            idx = self.datasetbeh_trial_to_datidx(trial)
            return self.Datasetbeh.Dat.iloc[idx]["strokes_task"]
        else:
            # This is not saved in Datasetbeh. Therefore resort to old methods.
            # This is not ideal, but it is never used for real analyses..
            if trial in self._CachedStrokesTask.keys():
                return self._CachedStrokesTask[trial]
            else:
                from ..utils.monkeylogic import getTrialsTaskAsStrokes
                fd, trialml = self.beh_get_fd_trial(trial)
                strokestask = getTrialsTaskAsStrokes(fd, trialml)
                return strokestask

    def strokes_extract(self, trialtdt, peanuts_only=False):
        """ [GOOD] The only place where beh strokes are extracted
        """
        if peanuts_only:
            if True:
                # New, always get from dataset, this removes any source of misalignmenet between beh and neural
                idx = self.datasetbeh_trial_to_datidx(trialtdt)
                if idx is None:
                    print(trialtdt)
                    assert False, "this trial not in dataset.. why this trial has not been pruned?"
                strokes = self.Datasetbeh.Dat.iloc[idx]["strokes_beh"]
            else:
                # This is not saved in Datasetbeh. Therefore resort to old methods.
                # This is not ideal, but it is never used for real analyses..
                print("WARNING: using non-peanuts strokes, which may be misaligned with strokes in updated Datasetbeh...")
                # To fix: should force to use getTrialsStrokesByPeanuts, BUT should make sure it applies the preprocessing
                # that goes into extract strokes to construct datasetbeh
                from ..utils.monkeylogic import getTrialsStrokes, getTrialsStrokesByPeanuts
                if trialtdt in self._CachedStrokesPeanutsOnly.keys():
                    strokes = self._CachedStrokesPeanutsOnly[trialtdt]
                else:
                    fd, trialml = self.beh_get_fd_trial(trialtdt)
                    strokes = getTrialsStrokesByPeanuts(fd, trialml)
        else:
            if trialtdt in self._CachedStrokes.keys():
                strokes = self._CachedStrokes[trialtdt]
            else:
                from ..utils.monkeylogic import getTrialsStrokes, getTrialsStrokesByPeanuts
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
        
    def plot_epocs(self, ax, trial, list_epocs=("camframe", "camtrialon", "camtrialoff", 
        "rewon", "rewoff", "behcode"), overlay_trial_events=True, 
        overlay_trial_events_notpd=False, label_cam_frames=False):
        """ Plot discrete events onto axes, for this trial
        """
        
        # -- Epochs
        ax.set_title("epocs")
        for i, pl in enumerate(list_epocs):
            times, vals = self.extract_data_tank_epocs(pl, trial0=trial)
            if times is not None:
                ax.plot(times, np.ones(times.shape)+i, 'x', label=pl)
                if pl == "camframe" and label_cam_frames:
                    time_diffs = list(np.diff(np.array(times)))
                    jump_ind = time_diffs.index(max(time_diffs))+1
                    for x in range(jump_ind,len(times)):
                        offset = -10 if (x & 1) else -15
                        ax.annotate(f'{x-jump_ind}', (times[x],1), textcoords="offset points", xytext=(0,offset), ha='center')
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
        else:
            assert isinstance(site, int)

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
            overlay_trial_events_notpd=False,
            only_cam_stuff=False):
        """ Plot a single raster for this trial, across these sites
        PARAMS:
        - site_to_highlight, bool, if True, colors it diff
        """
        
        # fig, axes = plt.subplots(2,2, figsize=(WIDTH, HEIGHT), sharex=False, 
        #                          gridspec_kw={'height_ratios': [1,8], 'width_ratios':[8,1]})
        #Juts do plots relevant for cam stuff
        if only_cam_stuff:
            fig1, axes = plt.subplots(2, 1, figsize=(15, 14), sharex=True)
            ax = axes.flatten()[0]
            self.plot_epocs(ax, trialtdt, overlay_trial_events=overlay_trial_events,
            overlay_trial_events_notpd=overlay_trial_events_notpd,label_cam_frames=True)
            ax = axes.flatten()[1]
            ax.set_title("beh strokes")
            self.plot_trial_timecourse_summary(ax, trialtdt, overlay_trial_events=overlay_trial_events)
            return fig1, None
        
        
        fig1, axes = plt.subplots(10, 1, figsize=(15, 28), sharex=True, 
            gridspec_kw={'height_ratios': [1, 1, 1, 1,1,1,1,12,1, 1]})

        # -- Epochs (events)
        ax = axes.flatten()[0]
        self.plot_epocs(ax, trialtdt, overlay_trial_events=overlay_trial_events,
            overlay_trial_events_notpd=overlay_trial_events_notpd)
        # XLIM = ax.get_xlim()
            
        # ax.set_xlim(XLIM)
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
        site = random.choice(self.sitegetterKS_map_region_to_sites_MULTREG())
        ax.set_title(f"ranbdom raw data: site{site}")
        D = self.datall_TDT_KS_slice_single_bysite(site, trialtdt)
        if D is not None:
            if "raw" in D.keys():
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
        self.plot_raster_sites(ax, trialtdt, list_sites, overlay_trial_events=overlay_trial_events)
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
        pa, _, _ = self.smoothedfr_extract_timewindow(trials, sites, alignto, pre_dur, post_dur)

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
        sites = self.sitegetterKS_map_region_to_sites_MULTREG()
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
        sites = self.sitegetterKS_map_region_to_sites_MULTREG()
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
        sites = self.sitegetterKS_map_region_to_sites_MULTREG()

        # LIST_ALIGN_TO = ["go", "samp"]
        LIST_ALIGN_TO = ["go"]
        for s in sites:
            bregion = self.sitegetterKS_thissite_info(s)["region"]
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

        trials_list = self.get_trials_list(True)
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
                
            for trial in trials_list:
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

    def ks_plot_compare_overlaying_spikes_on_raw_filtered(self, site, trial, twind_plot=None):
        """
        Make two debugging plots, both overlaying spikes on raw filt data, but one KS other TDT.
        :param site:
        :param trial:
        :param twind_plot:
        :return:
        """
        assert self.SPIKES_VERSION == "kilosort"

        # 1. Plot current, e.g., kilosort.
        self.plot_raw_overlay_spikes_on_raw_filtered(site, trial, twind_plot)

        # 2. Plot tdt.
        try:
            site_tdt = self.sitegetterKS_thissite_info(site)["site_tdt"]
            self.SPIKES_VERSION = "tdt"
            self._SPIKES_VERSION_INPUTED = "tdt"
            self.plot_raw_overlay_spikes_on_raw_filtered(site_tdt, trial, twind_plot)
            self.SPIKES_VERSION = "kilosort"
            self._SPIKES_VERSION_INPUTED = "kilosort"
        except Exception as err:
            self.SPIKES_VERSION = "kilosort"
            self._SPIKES_VERSION_INPUTED = "kilosort"

    def plot_raw_overlay_spikes_on_raw_filtered(self, site, trial, twind_plot=None):
        """
        Plot raw neural data (filtered) and overlying rasters on that.
        Useful for debugging, seeing results of spike clustering.
        PARAMS:
        - twind_plot, window relative to trial onset, to plot.
        :return:
        """
        import elephant as el

        if twind_plot is None:
            twind_plot = [0, 0.5]

        # 1. Extract spikes data for this trial, site
        datspikes = self.datall_TDT_KS_slice_single_bysite(site, trial)
        st = datspikes["spike_times"]

        # 2. Extract raw data
        chan = self.sitegetterKS_thissite_info(site)["chan"]
        rs = self.sitegetterKS_thissite_info(site)["rs"]
        tmp = self.load_raw([rs], [chan], trial, get_raw=True)
        assert len(tmp)==1
        datraw = tmp[0]
        raw = datraw["raw"]
        times = datraw["tbins0"]
        fs = datraw["fs"]

        # 3. Filter the neural singnal
        raw_filt = el.signal_processing.butter(raw, highpass_frequency=300, lowpass_frequency=3000,  sampling_frequency=fs)

        # 4. Make sure signal is relative to trial
        # print(times)
        # times, raw_filt, _, _, _ = self.extract_windowed_data_bytrial(times, trial, raw_filt)
        # print(times)
        # assert False

        # 4. Plot a specific time window
        times_this, raw_filt_this, _, _, _ = self._timewindow_window_this_data(times, twind_plot, raw_filt)
        st_this, _, _, _, _ = self._timewindow_window_this_data(st, twind_plot)

        fig, ax = plt.subplots(figsize=(25,5))
        ax.plot(times_this, raw_filt_this)
        self._plot_raster_line(ax, st_this, -50, alpha=1, linelengths=10, linewidths=2, color="r")

        return fig, ax

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
                raw = self._datallTDT_slice_single(rs, chan, trial)
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

    def _datasetbeh_trial_to_trialcode_from_raw(self, trial):
        """ Recompute trialcode for this trial(tdt), not from cached.
        This should only be called when generating _MapperTrialcode2TrialToTrial
        RETURNS:
        - trialcode, a string
        """
        date = self.Date
        index_sess, trial_ml = self._beh_get_fdnum_trial(trial)
        session_ml = self.BehSessList[index_sess]

        trialcode = f"{date}-{session_ml}-{trial_ml}"
        return trialcode

    def datasetbeh_trial_to_trialcode(self, trial):
        """ Try to load from cached, get trialcode for this trial(tdt)
        RETURNS:
        - trialcode, a string
        """

        if (not hasattr(self, "_MapperTrialcode2TrialToTrial")) or (len(self._MapperTrialcode2TrialToTrial) == 0):
            # generate mappers from trialcode to trial
            self._generate_mappers_quickly_datasetbeh()

        # Return the trialcode
        try:
            idx = list(self._MapperTrialcode2TrialToTrial.values()).index(trial)
        except Exception as err:
            print("trials in mapper: ", self._MapperTrialcode2TrialToTrial.values())
            print("Looking for this trial", trial)
            raise err
        
        return list(self._MapperTrialcode2TrialToTrial.keys())[idx]

        # trialcode = self.datasetbeh_trial_to_trialcode_from_raw(trial)
        #
        # date = self.Date
        # index_sess, trial_ml = self._beh_get_fdnum_trial(trial)
        # session_ml = self.BehSessList[index_sess]
        #
        # trialcode = f"{date}-{session_ml}-{trial_ml}"
        # return trialcode

    def datasetbeh_trial_to_datidx(self, trial, dataset_input=None):
        """ returns the index in self.Datasetbeh correspodning to
        this trial. If doesnt exist, then returns None. 
        This is accurate even if self.Datasetbeh is changed.
        - dataset_input, which datsaet to query. useful to pass in a pruned
        datsaet if you want to check whether this trial exists (i.e., returns None)
        """

        if dataset_input is None:
            dfcheck = self.Datasetbeh
        else:
            dfcheck = dataset_input
        assert dfcheck is not None

        # key = tuple(dfcheck["trialcode"].tolist())

        tc = self.datasetbeh_trial_to_trialcode(trial)
        # print("get_trials_list - datasetbeh_trial_to_datidx", trial, tc)

        dfthis = dfcheck.Dat[dfcheck.Dat["trialcode"]==tc]

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
        if len(self._MapperTrialcode2TrialToTrial)==0:
            self._generate_mappers_quickly_datasetbeh()
        # assert len(self._MapperTrialcode2TrialToTrial)>0, "cannot check without this..."
        return [trialcode for trialcode in list_trialcodes if trialcode in self._MapperTrialcode2TrialToTrial.keys()]
        

    ##################### SNIPPETS
    def snippets_extract_by_event_flexible(self, sites, trials,
        list_events, 
        pre_dur= -0.4, post_dur= 0.4, features_to_get_extra=None, 
        fr_which_version="sqrt", DEBUG=False,
        fail_if_times_outside_existing=True,
        method_if_not_enough_time="return_none"):
        """ Helper to extract snippets in flexible way, saligend to each event.
        PARAMS:
        - sites, list of ints to extract.
        - list_events, list of string code names of events. if mult per trials, gets all
        of them
        - features_to_get_extra, list of str, features to extract from DS. fails if these dont
        already exist in DS.
        - SANITY_CHECK, if True, checks alignment across diff columns of df during extraction, no mistakes.
        - method_if_not_enough_time, str, note: Make it method_if_not_enough_time="return_none" and fail_if_times_outside_existing==None,
        # as Kedar was using this for fixations... So this throws out fixations that are
        # too close to edge of trial. In general should determine this outside the funcgtion.

        RETURNS:
        - dataframe, each row a (chan, event).
        """

        import pandas as pd
        from pythonlib.tools.pandastools import applyFunctionToAllRows

        OUT = []
        trials_all = []
        times_all = []
        idx_trialtime_all = []
        idx_trialtime=0
        if DEBUG:
            print("  trial - event - event_time")

        for i_e, event in enumerate(list_events):
            event_unique_name = self.events_rename_with_ordered_index([event])[0]
            # if i_e<10:
            #     idx_str = f"0{i_e}"
            # else:
            #     idx_str = f"{i_e}"
            # event_unique_name = f"{idx_str}_{event}"
            for trial_neural in trials:
                if trial_neural%50==0:
                    print(event_unique_name, " - ", trial_neural)
                list_times = sorted(self.events_get_time_helper(event, trial_neural))

                # For some events, you want to extract features associated with each event.
                if event in ["fixon_preparation"]:
                    #### extracts featurename and featurevals FOR certain custom events (e.g. saccades, fixations)
                    # get feature name and list of values
                    feat_name, list_featvals = self.events_get_feature_helper(event, trial_neural)
                else:
                    feat_name = None
                    list_featvals = [None for _ in range(len(list_times))]

                # make sure there is one value per event time
                assert len(list_times)==len(list_featvals)

                # add entry to dataframe
                for trial_ind, (event_time, featval) in enumerate(zip(list_times, list_featvals)):

                    if DEBUG:
                        print(trial_neural, ' - ', event, ' - ' , event_time)

                    trials_all.append(trial_neural)
                    times_all.append(event_time)
                    idx_trialtime_all.append(idx_trialtime)

                    for s in sites:

                        # get spiketimes
                        spike_times = self._snippets_extract_single_snip_spiketimes(s, trial_neural,
                            event_time, pre_dur, post_dur)

                        # save it
                        tc = self.datasetbeh_trial_to_trialcode(trial_neural)
                        OUT.append({
                            "trialcode":tc,
                            "chan":s,
                            "event_unique_name":event_unique_name,
                            "event_aligned":event,
                            "spike_times":spike_times,
                            "trial_neural":trial_neural,
                            "event_time":event_time,
                            # feat_name:featval,
                            "idx_trialtime":idx_trialtime,
                            "event_idx_within_trial":trial_ind
                        })
                        if feat_name is not None:
                            OUT[-1][feat_name] = featval
                    # increment index
                    idx_trialtime+=1

        # Get smoothed fr. this is MUCH faster than computing above.
        print("Extracting smoothed FR for all data...")
        # fail_if_times_outside_existing = True
        pa, trials_all, times_all, idx_trialtime_all = self.smoothedfr_extract_timewindow_bytimes(trials_all, times_all, sites, 
            pre_dur=pre_dur, post_dur=post_dur, 
            fail_if_times_outside_existing=fail_if_times_outside_existing,
            idx_trialtime_all=idx_trialtime_all,
            method_if_not_enough_time=method_if_not_enough_time)

        # Prune OUT to just those trials that have full data...
        # assert sorted(set([O["idx_trialtime"] for O in OUT])) == list(range(len(OUT))), "just checking that I undersatnd what is going on."
        OUT = [O for O in OUT if O["idx_trialtime"] in idx_trialtime_all]

        if DEBUG:
            print(pa.X.shape) # (chans, trials, tbins)
            print(pa.Trials)
            print(pa.Chans) 
            print(len(trials_all))
            assert False

        # deal out time to each site and trial.
        assert len(trials_all)==pa.X.shape[1]
        assert len(sites)==pa.X.shape[0]

        print("Inserting smoothed FR into dataset...")
        ct = 0
        for i in range(len(trials_all)):
            for j in range(len(sites)):
                fr_sm = pa.X[j, i, :]
                fr_sm_times = pa.Times
                OUT[ct]["fr_sm"] = fr_sm[None, :]
                OUT[ct]["fr_sm_times"] = fr_sm_times[None, :]

                # sanity check
                assert OUT[ct]["chan"] == sites[j]
                assert OUT[ct]["trial_neural"] == trials_all[i]
                assert OUT[ct]["event_time"] == times_all[i]
                assert OUT[ct]["idx_trialtime"] == idx_trialtime_all[i]

                ct+=1
        # ----
        df = pd.DataFrame(OUT)

        return df

    def snippets_extract_bysubstroke(self, sites, DS, pre_dur= -0.4, post_dur= 0.4,
        features_to_get_extra=None, SANITY_CHECK=False, align_to="onset"):
        """ Wrapper of snippets_extract_bystroke"""

        fr_which_version = None # not used.
        return self.snippets_extract_bystroke(sites, DS, pre_dur, post_dur,
                    features_to_get_extra, fr_which_version, SANITY_CHECK,
                    align_to, use_time_within_DS=True)

    def snippets_extract_bystroke(self, sites, DS, pre_dur= -0.4, post_dur= 0.4,
        features_to_get_extra=None, fr_which_version="sqrt", SANITY_CHECK=True,
                                  align_to="onset", use_time_within_DS=False):
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
        event_times = []
        for ind in range(len(DS.Dat)):

            # print(ind)
            if ind%50==0:
                print("index strokes: ", ind)

            tc = DS.Dat.iloc[ind]["dataset_trialcode"]
            si = DS.Dat.iloc[ind]["stroke_index"]
            trial_neural = self.datasetbeh_trialcode_to_trial(tc)
            if align_to=="onset":
                event = f"on_strokeidx_{si}"
            elif align_to=="offset":
                event = f"off_strokeidx_{si}"
            else:
                assert False

            if use_time_within_DS:
                assert False, "never use this, it has touchscreen lag.. see below"
                if align_to=="onset":
                    event_time = DS.Dat.iloc[ind]["time_onset"]
                elif align_to=="offset":
                    event_time = DS.Dat.iloc[ind]["time_offset"]
                else:
                    assert False
            else:
                event_time = self.events_get_time_helper(event, trial_neural)[0]
                # Sanity check
                try:
                    if align_to=="onset":
                        assert [event_time]==self.hack_adjust_touch_times_touchscreen_lag([DS.Dat.iloc[ind]["time_onset"]]), "bug somewjhere"
                    elif align_to=="offset":
                        assert [event_time]==self.hack_adjust_touch_times_touchscreen_lag([DS.Dat.iloc[ind]["time_offset"]]), "bug somewjhere"
                except Exception as err:
                    print(event_time)
                    print(DS.Dat.iloc[ind]["time_onset"])
                    print(DS.Dat.iloc[ind]["time_offset"])
                    print(DS.Dat.iloc[ind])
                    raise err

            trials.append(trial_neural)
            strokeids.append(si)
            event_times.append(event_time)

            for s in sites:

                # get spiketimes
                spike_times = self._snippets_extract_single_snip_spiketimes(s, trial_neural, 
                    event_time, pre_dur, post_dur)

                # save it
                OUT.append({
                    "index_DS":ind,
                    "trialcode":tc,
                    "chan":s,
                    "event_aligned":event_name,
                    "stroke_index":si,
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
        if True:
            # Works for substrokes, doesnt have to get stroke events using SN.
            pa = self.smoothedfr_extract_timewindow_bytimes(trials, event_times,
                sites, pre_dur=pre_dur, post_dur=post_dur,
                fail_if_times_outside_existing=fail_if_times_outside_existing)[0]
            assert pa.X.shape[1]==len(trials)
        else:
            pa = self.smoothedfr_extract_timewindow_bystroke(trials, strokeids, sites,
                pre_dur=pre_dur, post_dur=post_dur,
                fail_if_times_outside_existing=fail_if_times_outside_existing,
                align_to=align_to)

        # print(pa.X.shape) # (chans, trials, tbins)
        # print(pa.Trials)
        # print(pa.Chans) 
        # deal out time to each site and trial.
        print("Inserting smoothed FR into dataset...")
        ct = 0
        # for i in range(len(DS.Dat)):
        for i in range(len(trials)):
            for j in range(len(sites)):
                fr_sm = pa.X[j, i, :]
                fr_sm_times = pa.Times
                OUT[ct]["fr_sm"] = fr_sm[None, :]
                OUT[ct]["fr_sm_times"] = fr_sm_times[None, :]
                if SANITY_CHECK:
                    assert pa.Chans[j] == sites[j]
                    assert OUT[ct]["chan"] == sites[j]
                    assert OUT[ct]["index_DS"] == i
                    assert OUT[ct]["trialcode"] == DS.Dat.iloc[i]["trialcode"]
                    # assert OUT[ct]["event_time"] ==
                    # event_time":event_time
                ct+=1
        # ----
        df = pd.DataFrame(OUT)

        # get every column in DS
        print("Appending other columns into dataset...")
        if True:
            # Much faster
            # for col in DS.Dat.columns:
            from pythonlib.tools.pandastools import _check_index_reseted, join_dataframes_appending_columns
            _check_index_reseted(DS.Dat)
            if len(list_cols)>0:
                df2 = DS.Dat.loc[list_ind, list_cols].reset_index(drop=True)
                df = join_dataframes_appending_columns(df, df2, list_cols)

            if len(features_to_get_extra)>0:
                df2 = DS.Dat.loc[list_ind, features_to_get_extra].reset_index(drop=True)
                df = join_dataframes_appending_columns(df, df2, features_to_get_extra)

        else:
            for col in list_cols:
                df[col] = DS.Dat.iloc[list_ind][col].tolist()
                # OUT[-1][col] = DS.Dat.iloc[ind][col]
            for col in features_to_get_extra:
                df[col] = DS.Dat.iloc[list_ind][col].tolist()
                # OUT[-1][col] = DS.Dat.iloc[ind][col]

        if SANITY_CHECK:
            assert df["trialcode"].tolist() == DS.Dat.iloc[list_ind]["dataset_trialcode"].tolist()
            assert df["dataset_trialcode"].tolist() == DS.Dat.iloc[list_ind]["dataset_trialcode"].tolist()

        return df

    def snippets_extract_bytrial(self, sites, trials, events,
        list_pre_dur, list_post_dur, features_to_get_extra=None):
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

        assert len(list_pre_dur)==len(events)
        assert len(list_post_dur)==len(events)

        # Extract snippets across all trials, sites, and events.
        # list_cols = ['task_kind']
        list_cols = []

        OUT = []
        list_events_uniqnames = []
        for i_e, (e, pre_dur, post_dur) in enumerate(zip(events, list_pre_dur, list_post_dur)):
            event_unique_name = self.events_rename_with_ordered_index([e])[0]
            # if i_e<10:
            #     idx_str = f"0{i_e}"
            # else:
            #     idx_str = f"{i_e}"
            # event_unique_name = f"{idx_str}_{e}"
            list_events_uniqnames.append(event_unique_name)

            # Skip if this trial doesnt have this event
            print("Extraction pa for this event... ", e)
            method_if_not_enough_time = "return_none"
            fail_if_times_outside_existing = False
            pa, trials_this_event, event_times_gotten = self.smoothedfr_extract_timewindow(trials, sites,
                                                                       e, pre_dur, post_dur,
                                                                       fail_if_times_outside_existing=fail_if_times_outside_existing,
                                                                       method_if_not_enough_time=method_if_not_enough_time)

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

            print("... extracting data for each trial for event ", e)
            for i_t, t in enumerate(trials_this_event):

                if t%100==0:
                    print(t)
                # print(t)

                tc = self.datasetbeh_trial_to_trialcode(t)
                ind_dataset = self.datasetbeh_trial_to_datidx(t)

                # get eventtime
                list_event_time = self.events_get_time_helper(e, t)
                # take the first
                if len(list_event_time)>0:
                    event_time = list_event_time[0]
                else:
                    print(t, e)
                    assert False, "no event"

                assert np.isclose(event_times_gotten[i_t], event_time), f"bug?? {event_times_gotten[i_t]}, {event_time}"

                for i_s, s in enumerate(sites):

                    # get spiketimes
                    spike_times = self._snippets_extract_single_snip_spiketimes(s, t,
                        event_time, pre_dur, post_dur)

                    # get smoothed fr
                    fr_sm = pa.X[i_s, i_t, :]
                    fr_sm_times = pa.Times

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

    def _snippets_extract_single_snip_spiketimes(self, site, trial, event_time,
            pre_dur, post_dur, subtract_event_time=True):
        """ Extract a single snippet's spike times. aligned to event_time (i.e,
        subtracts that time, setting it to 0).
        PARAMS:
        - pre_dur, input a negative number, in sec, relative to 0.
        - post_dur, postiive number, in sec, rleative to 0
        """
        dat = self.datall_TDT_KS_slice_single_bysite(site, trial)
        spike_times = dat["spike_times"]
        # time_on = dat["time_on"]
        # time_off = dat["time_off"]
        # spike_times = dat["spiketrain"]
        
        # recenter s times to event
        spike_times = spike_times - event_time
        # time_on = time_on - event_time
        # time_off = time_off - event_time

        # get windowed spike times
        if True:
            # use popanal
            spike_times = spike_times[(spike_times >= pre_dur) & (spike_times <= post_dur)]
            if not subtract_event_time:
                # Then un-subtract event_time
                spike_times = spike_times + event_time
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
        assert False, "use datall_TDT_KS_slice_single_bysite so works for kilosort"
        dat = self._datallTDT_slice_single(rs, chan, trial)

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


    ########################### 
    def export_to_dataframe(self, savedir, suffix):
        """
        """

        sdir = f"{savedir}/{self.Animal}-{self.Date}"
        os.makedirs(sdir, exist_ok=True)

        trials = self.get_trials_list(True)

        ## Extract smoothed fr for each trial
        dfextract, sites = self.smoothedfr_extract_trials(trials)
        assert dfextract["trial"].tolist()==trials
        # sn.smoothedfr_extract(trials[:10], sites[:10])
        # sn.smoothedfr_extract_timewindow
        # sn.smoothedfr_extract_timewindow(trials[:10], sites[:10], "go", )

        ## Channels/sites data
        list_regions = []
        for s in sites:
            list_regions.append(self.sitegetterKS_thissite_info(s)["region"])
        dfchannels = pd.DataFrame({
            "frmat_row":range(len(sites)),
            "channel_idxs":sites,
            "channel_region":list_regions
        })

        ## Extract task events
        events = self.events_default_list_events(include_stroke_endpoints=True, include_events_from_dict=True)
        dfevents, list_events = self.eventsdataframe_extract_timings(events, trials=trials)
        assert dfevents["trial"].tolist()==trials

        if False:
            # dont do. leads to variable column names...
            dfextract = pd.merge(dfextract, dfevents, on="trial")

        ## Strokes (beh and task image).
        list_strokes_beh = []
        for t in trials:
            strokes = self.strokes_extract(t, peanuts_only=True)
            list_strokes_beh.append(strokes)
        dfextract["strokes_beh"] = list_strokes_beh

        list_strokes_task = []
        for t in trials:
            strokes = self.strokes_task_extract(t)
            list_strokes_task.append(strokes)
        dfextract["strokes_task"] = list_strokes_task

        ## Ad-hoc trial-level dat.
        D = self.Datasetbeh

        # - epoch
        list_epoch = []
        list_sess = []
        # list_date = []
        for t in trials:
            idx = self.datasetbeh_trial_to_datidx(t)
            list_epoch.append(D.Dat.iloc[idx]["epoch"])
            list_sess.append(D.Dat.iloc[idx]["session"])
            # list_date.append(D.Dat.iloc[idx]["date"])
        dfextract["epoch"] = list_epoch
        dfextract["session"] = list_sess

        # - shape
        # list_tok = []
        list_shape = []
        list_loc = []
        for t in trials:
            idx = self.datasetbeh_trial_to_datidx(t)
            D.taskclass_tokens_extract_wrapper(idx)
            tokens = D.taskclass_tokens_extract_wrapper(idx)
            assert len(tokens)==1, "single prim, for Dan Dolnik"
            tok = tokens[0]
            # keys_keep = ["shape", "gridloc"]
            # tok = {k:v for k,v in tok.items() if k in keys_keep}
            # list_tok.append(tok)
            list_shape.append(tok["shape"])
            list_loc.append(tok["gridloc"])
        dfextract["shape"] = list_shape
        dfextract["gridloc"] = list_loc

        ## Save
        path = f"{sdir}/dfextract-{suffix}.pkl"
        print("SAVING: ", path)
        dfextract.to_pickle(path)

        path = f"{sdir}/dfevents-{suffix}.pkl"
        print("SAVING: ", path)
        dfevents.to_pickle(path)

        path = f"{sdir}/dfchannels-{suffix}.pkl"
        print("SAVING: ", path)
        dfchannels.to_pickle(path)

    def debug_mode_switch_to(self, sites=True, trials=True):
        """ Seitch debug mode to variable passed in.
        Takes care of clearing variables that have state (memoized)
        Debug mode this prunes sites and trials, so analyses runs faster
    """
    
        if trials==True and self._DEBUG_PRUNE_TRIALS==False:
            # Turn on debug mode
            self._DEBUG_PRUNE_TRIALS=True
        elif trials==False and self._DEBUG_PRUNE_TRIALS==True:
            self._DEBUG_PRUNE_TRIALS=False
            # clear cached population data, since this is using subset of trials.
            self.PopAnalDict = {}

        if sites==True and self._DEBUG_PRUNE_SITES==False:
            self._DEBUG_PRUNE_SITES=True
        elif sites==True and self._DEBUG_PRUNE_SITES==False:
            self._DEBUG_PRUNE_SITES=True

    #######################
    def datamod_sitegetter_reorder_by_bregion(self, df, col="bregion"):
        """ reorder rows of dataframe based on bregion, from top to bottom
        RETURNS:
            - df, sorted. DOes not modify input
        """
        from neuralmonkey.neuralplots.brainschematic import datamod_reorder_by_bregion
        return datamod_reorder_by_bregion(df)
        # if col not in df.columns:
        #     if "region" in df.columns:
        #         col = "region"
        #     else:
        #         print(df.columns)
        #         assert False, "whichc olumn holds regions?"
        # map_region_to_index = {region:i for i, region in enumerate(REGIONS_IN_ORDER)}
        # def F(x):
        #     return [map_region_to_index[xx] for xx in x] # list of ints
        # return df.sort_values(by=col, key=lambda x:F(x))

    ####################################
    #### CLUSTERFIX EXTRACTION CODE ####
    ####################################

    ######################
    ## helper functions ##
    ######################
    def datasetbeh_get_all_trialcodes(self):
        assert False, "do not use this. Datasetbeh.Dat is not guaranteed to map onto neural trials one to one."
        return self.Datasetbeh.Dat['trialcode'].tolist()

    def datasetbeh_get_all_trialnums(self):
        return self.datasetbeh_trialcode_to_trial_batch(self.datasetbeh_get_all_trialcodes())

    # returns smoothed and transformed x,y data for a session/trialnum
    def get_eye_xy_smoothed_and_transformed(self, trialnum, PLOT=True):
        # get TRANSFORMED xy-coordinates (used calibration matrix to map to screen)
        st = self.beh_extract_eye_good(trialnum, CHECK_TDT_ML2_MATCH=True, return_fs_tdt=True)
        times = st[0]
        x_aff = st[1][:,0]
        y_aff = st[1][:,1]
        fs_tdt = st[2]

        # SMOOTH DATA
        from pythonlib.tools.timeseriestools import smoothDat
        x_aff_sm = smoothDat(x_aff, window_len=10)
        y_aff_sm = smoothDat(y_aff, window_len=10)
        
        return x_aff_sm, y_aff_sm, times, fs_tdt

    # get the start, end times for the window spanned by start_event, end_event
    def get_time_window_of_events(self, trial, start_event, end_event):
        # keep just times between [start_event, end_event]
        dict_event_times = self.events_get_time_sorted(trial, list_events=(start_event, end_event))[0]
        start_time = dict_event_times[start_event][0]
        end_time = dict_event_times[end_event][0]
        
        return start_time, end_time

    #########################
    ## exporting functions ##
    #########################
    def extract_and_save_clusterfix_trial_xy_mats(self, fs_new=200):    
        # export .mat files with raw x,y to be used in MATLAB
        # LT updated 5/26/24, confirmed all steps good.
        animal = self.Animal 
        date = self.Date 
        session_no = self.RecSession

        SAVEDIR = f"{PATH_SAVE_CLUSTERFIX}/{animal}-{date}-{session_no}/raw_xy_mats"
        os.makedirs(SAVEDIR, exist_ok=True)

        # This may run into error in getting dataset trialcodes that dont exist as neural trials..
        # trialcodes = self.datasetbeh_get_all_trialcodes()
        # trialnums = self.datasetbeh_get_all_trialnums()
        # scipy.io.savemat(f"{SAVEDIR}/all_ntrialnums.mat", dict(neuraltrialnums=trialnums))
        # scipy.io.savemat(f"{SAVEDIR}/all_trialcodes.mat", dict(trialcodes=trialcodes))

        # loop thru trials and save xy data
        trialnums = self.get_trials_list()
        trialcodes = []
        for tnum in trialnums:

            tc = self.datasetbeh_trial_to_trialcode(tnum)
            trialcodes.append(tc)

            # get sampling rate
            # _, _, fs_raw = self.extract_data_tank_streams("eyex", tnum, ploton=False)

            # get XY smoothed / transformed on eye calibration matrix
            x_raw,y_raw,times_raw, fs_raw = self.get_eye_xy_smoothed_and_transformed(tnum, False)

            # resample x, y, times using integer sampling rate
            stroke_raw = [np.array([x_raw, y_raw, times_raw]).T] # dummy stroke list
            stroke_intp = strokesInterpolate2(stroke_raw, ["fsnew", fs_new, fs_raw])
            stroke_resampled = stroke_intp[0]
            x_rs = stroke_resampled[:,0]
            y_rs = stroke_resampled[:,1]
            times_rs = stroke_resampled[:,2]

            # save data to be loaded into MATLAB
            fname = f"{SAVEDIR}/ntrial{str(tnum)}.mat"
            scipy.io.savemat(fname, dict(x=x_rs, y=y_rs, times_xy=times_rs, fs_hz=fs_new))

        # Save mappings between trials and trialcodes.
        scipy.io.savemat(f"{SAVEDIR}/all_ntrialnums.mat", dict(neuraltrialnums=trialnums))
        scipy.io.savemat(f"{SAVEDIR}/all_trialcodes.mat", dict(trialcodes=trialcodes))

    def extract_and_save_clusterfix_results_mat(self):
        # run MATLAB to extract clusterfix_results.mat
        animal = self.Animal 
        date = self.Date 
        session_no = self.RecSession
        base_dir = PATH_SAVE_CLUSTERFIX
        
        PATH_TO_CLUSTERFIX_CODE = f"{PATH_NEURALMONKEY}/eyetracking_analyses"
        os.system(f"{PATH_MATLAB} -nodisplay -nosplash -nodesktop -r \"addpath(genpath('{PATH_TO_CLUSTERFIX_CODE}')); get_clusterfix_results_mat('{animal}', '{date}', '{session_no}', '{base_dir}'); quit\"")

    def extract_clusterfix_results_dataframe(self):
        # creating a dataframe of clusterfix_results from the original .mat
        # note: clusterfix_results.mat has a unique structure, similar to JSON...

        animal = self.Animal 
        date = self.Date 
        session_no = self.RecSession

        # load in results
        mat = scipy.io.loadmat(f"{PATH_SAVE_CLUSTERFIX}/{animal}-{date}-{session_no}/clusterfix_results.mat")

        # add results to dataframe
        mat_vars = ['neuraltrialnum', 'trialcode', 'fs', 'x', 'y', 'times', 'fixation_start_inds',
                    'fixation_end_inds', 'fixation_centroids_x', 'fixation_centroids_y', 
                    'saccade_start_inds', 'saccade_end_inds']
        tmp = []
        for i in range(len(mat['RESULTS'][0])):
            neuraltrialnum = mat['RESULTS'][0]['neuraltrialnum'][i][0,0]
            tcode = mat['RESULTS'][0]['trialcode'][i][0]
            fs = mat['RESULTS'][0]['fs'][i][0,0]
            x = mat['RESULTS'][0]['x'][i][0]
            y = mat['RESULTS'][0]['y'][i][0]
            times = mat['RESULTS'][0]['times'][i][0]

            # get start, end inds for fixations/saccades
            fixation_start_inds = mat['RESULTS'][0]['fixation_inds'][i][0]
            fixation_end_inds = mat['RESULTS'][0]['fixation_inds'][i][1]
            saccade_start_inds = mat['RESULTS'][0]['saccade_inds'][i][0]
            saccade_end_inds = mat['RESULTS'][0]['saccade_inds'][i][1]

            # get centroids x,y
            fixation_centroids_x = mat['RESULTS'][0]['fixation_centroids'][i][0]
            fixation_centroids_y = mat['RESULTS'][0]['fixation_centroids'][i][1]

            dat = [neuraltrialnum, tcode, fs, x, y, times, fixation_start_inds, fixation_end_inds,
                    fixation_centroids_x, fixation_centroids_y, saccade_start_inds, saccade_end_inds]
            tmp.append({})
            for v, d in zip(mat_vars, dat):
                tmp[-1][v]=d

        return pd.DataFrame(tmp, columns=mat_vars)

    def extract_and_save_clusterfix_trial_fixsacc_csvs(self):
        # export .csv files with saccade times and fixation times/centroids

        animal = self.Animal 
        date = self.Date 
        session_no = self.RecSession

        SAVEDIR = f"{PATH_SAVE_CLUSTERFIX}/{animal}-{date}-{session_no}/clusterfix_result_csvs"
        os.makedirs(SAVEDIR, exist_ok=True)

        clusterfix_results = self.extract_clusterfix_results_dataframe()
        for index, row in clusterfix_results.iterrows():
            tnum = row['neuraltrialnum']
            tcode = row['trialcode']

            x_t = row['x']
            y_t = row['y']
            times_t = row['times']

            # get the FIXATIONS belonging to this trial
            fixation_start_inds = row['fixation_start_inds']
            fixation_end_inds = row['fixation_end_inds']
            fixation_centroids_x = row['fixation_centroids_x']
            fixation_centroids_y = row['fixation_centroids_y']
            centroid_pairs = [[x,y] for x,y in zip(fixation_centroids_x, fixation_centroids_y)]

            # get the times of the FIXATIONS
            fixation_start_times = times_t[fixation_start_inds]
            fixation_end_times = times_t[fixation_end_inds]

            # save fixation start times using TRIALCODE, to load into session.py
            fname = f"{SAVEDIR}/{tcode}-fixation-onsets.csv"
            np.savetxt(fname, fixation_start_times, delimiter=',')

            # save fixation centroids using TRIALCODE, to load into session.py
            fname = f"{SAVEDIR}/{tcode}-fixation-centroids.csv"
            np.savetxt(fname, centroid_pairs, delimiter=",")

            # get the times of the SACCADES
            saccade_start_inds = row['saccade_start_inds']
            saccade_end_inds = row['saccade_end_inds']
            saccade_start_times = times_t[saccade_start_inds]
            saccade_end_times = times_t[saccade_end_inds]

            # save saccade start times using TRIALCODE, to load into session.py
            fname = f"{SAVEDIR}/{tcode}-saccade-onsets.csv"
            np.savetxt(fname, saccade_start_times, delimiter=',')

    # main function to extract clusterfix results
    # end result: this generates .csv files in PATH_SAVE_CLUSTERFIX which are then used by events_get_time_helper
    def extract_and_save_clusterfix_results(self, fs_new=200):
        print("exporting xy mats...")
        self.extract_and_save_clusterfix_trial_xy_mats(fs_new)
        print("running clusterfix in matlab...")
        self.extract_and_save_clusterfix_results_mat()
        print("exporting fixation/saccade csvs...")
        self.extract_and_save_clusterfix_trial_fixsacc_csvs()

    def clusterfix_check_if_preprocessing_complete(self):
        """
        Return True if clusterfix has been done on this day, to extract
        fixation events.
        """
        animal = self.Animal 
        date = self.Date 
        session_no = self.RecSession
        from neuralmonkey.utils.directory import clusterfix_check_if_preprocessing_complete
        return clusterfix_check_if_preprocessing_complete(animal, date, session_no)

        # SAVEDIR = f"{PATH_SAVE_CLUSTERFIX}/{animal}-{date}-{session_no}/clusterfix_result_csvs"
        # import os
        # return os.path.exists(SAVEDIR)

    ##################### SANITY CHECKS
    def sanity_waveforms_all_arrays_extract(self):
        """
        Extract data across arrays (bregion, site_within_region). The data will be
        peak_minus_trough of average waveform, althgou code culd be modiifed to 
        get different metrics.

        The output can be used for computation of changes in arrays across days.
        """

        ### Get mappers between global site and (region, site_within_region)
        map_region_site = self.sitegetterKS_generate_mapper_region_to_sites_BASE(clean=False)

        # Get global site that is index of first site in this region
        map_region_first_site = {} 
        for region, sites_global in map_region_site.items():
            if len(sites_global)>0:
                map_region_first_site[region] = min(sites_global)
                print(region, sites_global)
                # print(region, min(sites_global))

        # Finally, map from global to (region, site_within)
        map_site_to_region_sitewithin = {}
        for region, sites_global in map_region_site.items():
            
            if len(sites_global)>0:

                site_first = map_region_first_site[region]

                for s in sites_global:
                    map_site_to_region_sitewithin[s] = (region, s-site_first)

        ### Collect data for each site
        res = []
        sites = self.sitegetterKS_all_sites(clean=False) # get all sites, except missing arrays
        for s in sites:
            if s in self.DatSpikeWaveforms:
                waveforms = self.DatSpikeWaveforms[s]
            else:
                self.load_spike_waveforms(s)
                waveforms = self.DatSpikeWaveforms[s]

            # Compute stats
            wf_mean = np.mean(waveforms, axis=0)
            peak_minus_trough = np.max(wf_mean) - np.min(wf_mean)

            # Store
            region = self.sitegetterKS_map_site_to_region(s)
            _region, site_within = map_site_to_region_sitewithin[s]
            assert region==_region

            res.append({
                "region":region,
                "site_within":site_within,
                "site_global":s,
                "peak_minus_trough":peak_minus_trough,
                "wf_mean":wf_mean,
            })        

        dfres = pd.DataFrame(res)
        return dfres        
    


    def _sanity_waveforms_concat_waveforms(self, dfres):
        """
        Get data (bregions, sites x time bins), which is a more fine-grained represntation of entire array, to help in comparing
        across days. 
        PARAMS:
        - dfres, output of sanity_waveforms_all_arrays_extract
        RETURNS:
        - dataframe, index is nregions, and columns are ntimes after concatting across all 32 sites for this array (960 usually)
        """

        list_sites_within = range(32)
        list_regions = dfres["region"].unique().tolist()

        out = []
        for br in list_regions:
            
            # Collect and concat all sites
            wf_all = []
            for s in list_sites_within:
                dfthis = dfres[(dfres["region"] == br) & (dfres["site_within"] == s)]
                assert len(dfthis)==1

                wf_mean = dfthis["wf_mean"].item()
                wf_all.append(wf_mean)

            # concat into a single vector
            wf_all_concat = np.concatenate(wf_all, axis=0)
            out.append(wf_all_concat)
        mat_region_wfall = np.stack(out, axis=0) # (bregions, concatted times)    

        df = pd.DataFrame(mat_region_wfall, index=list_regions)

        return df
    
    def _sanity_waveforms_verify_finally(self, dat1, dat2, savedir, suffix,
                                         corr_method, heatmap_zlims):
        """
        Final quantification to verify array alignement across days.
        PARAMS:
        - dat1, dataframe data for day1, (nregions, ndimensions) where ndimensions could be nsites (each with a scalar statstic), or
        sites x time bins (if concatenate waveforms)
        - dat2, dataframe data for day2, similar structure.
        - regions, list of regions, assumed that these are the rows for both dat1, and dat2
        RETURNS:
        - passed, bool, True iff all bregions passed
        """
        from pythonlib.tools.pandastools import plot_subplots_heatmap
        from pythonlib.tools.snstools import heatmap

        # Check regions (rows) and confirm match across dataframes
        assert np.all(dat1.index==dat2.index)
        regions = dat1.index.tolist()

        # Get correlation between each pair of arrays (pairs are across datasets)
        res_cross = []
        for br1 in dat1.index:
            vals1 = dat1.loc[br1, :]

            for br2 in dat2.index:
                vals2 = dat2.loc[br2, :]

                if corr_method == "pearson":
                    corr = np.corrcoef(vals1, vals2)[0,1]
                elif corr_method == "spearman":
                    from scipy import stats
                    res = stats.spearmanr(vals1, vals2)
                    corr = res.statistic
                else:
                    assert False
                    
                res_cross.append({
                    "region1":br1,
                    "region2":br2,
                    "corr":corr
                })
        dfcross = pd.DataFrame(res_cross)

        ### Plots
        # Plot heatmaps for each dataset
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax = axes.flatten()[0]
        heatmap(dat1, ax, False, diverge=False, labels_row=regions, zlims=heatmap_zlims)#     
        ax = axes.flatten()[1]
        heatmap(dat2, ax, False, diverge=False, labels_row=regions, zlims=heatmap_zlims)#     
        savefig(fig, f"{savedir}/heatmaps_each_both_datasets-suffix={suffix}.pdf")
        
        # Plot heatmap of cross-corrleation
        # dfcross2d = plot_subplots_heatmap(dfcross, "region1", "region2", "corr", None, return_dfs=True)[2]["dummy"]
        fig, axes, tmp = plot_subplots_heatmap(dfcross, "region1", "region2", "corr", None, return_dfs=True, ZLIMS=[-1, 1], diverge=True)
        dfcross2d = tmp["dummy"]
        savefig(fig, f"{savedir}/correlation_across_sessions-suffix={suffix}.pdf")

        # Check that diagonals are much higher than offs
        passed_cross = True
        failed_regions = []
        for i, (br, row) in enumerate(dfcross2d.iterrows()):
            corr_same_array = row[i]
            corrs_other_arrays = np.r_[row[:i], row[i+1:]]

            if not corr_same_array > max(corrs_other_arrays):
                print(i, br, row)
                # assert False, "why array on day 1 mathces a different arrya on day 2?"
                passed_cross = False
                failed_regions.append((i, br, row))
        if passed_cross:
            print("Good!! passed sanity check. Each array on day 1, tested on day 2, matches best to itself.")        
        else:
            print("Failed! These were the failed regions:")
            for x in failed_regions:
                print(x)

        fig, ax = plt.subplots()
        savefig(fig, f"{savedir}/passed_cross={passed_cross}-suffix={suffix}")

        ##################################################
        ### Also, check whether array was flipped. i..e, if corr across days improves after flipping one array, then that means is
        # was flipped.
        dat2_flipped = pd.DataFrame(np.fliplr(dat2), index=dat2.index)

        # Get correlation between each pair of arrays (pairs are across datasets)
        res_cross = []
        for br1 in dat1.index:
            vals1 = dat1.loc[br1, :]

            for br2 in dat2_flipped.index:
                vals2 = dat2_flipped.loc[br2, :]

                if corr_method == "pearson":
                    corr = np.corrcoef(vals1, vals2)[0,1]
                elif corr_method == "spearman":
                    from scipy import stats
                    res = stats.spearmanr(vals1, vals2)
                    corr = res.statistic
                else:
                    assert False

                res_cross.append({
                    "region1":br1,
                    "region2":br2,
                    "corr":corr
                })
        dfcross_rev = pd.DataFrame(res_cross)

        # Plot heatmaps for each dataset
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax = axes.flatten()[0]
        heatmap(dat1, ax, False, diverge=False, labels_row=regions, zlims=heatmap_zlims) #     
        ax = axes.flatten()[1]
        heatmap(dat2_flipped, ax, False, diverge=False, labels_row=regions, zlims=heatmap_zlims) #     
        savefig(fig, f"{savedir}/heatmaps_each_both_datasets-FLIPPED-suffix={suffix}.pdf")
        
        # Plot heatmap of cross-corrleation
        fig, axes, tmp = plot_subplots_heatmap(dfcross_rev, "region1", "region2", "corr", None, return_dfs=True, ZLIMS=[-1, 1], diverge=True)
        savefig(fig, f"{savedir}/correlation_across_sessions-FLIPPED-suffix={suffix}.pdf")

        # Get the diagonals
        dfcorr = dfcross[dfcross["region1"] == dfcross["region2"]]
        dfcorr_rev = dfcross_rev[dfcross_rev["region1"] == dfcross_rev["region2"]]
        assert np.all(dfcorr["region1"] == dfcorr_rev["region1"])
        assert np.all(dfcorr["region2"] == dfcorr_rev["region2"])

        passed_flip = all(dfcorr["corr"] > dfcorr_rev["corr"])

        fig, ax = plt.subplots()
        savefig(fig, f"{savedir}/passed_flip={passed_flip}-suffix={suffix}")

        return passed_cross, passed_flip, dfcross
    
#####################################################################
assert _REGIONS_IN_ORDER == ("M1_m", "M1_l", "PMv_l", "PMv_m",
                "PMd_p", "PMd_a", "dlPFC_p", "dlPFC_a", 
                "vlPFC_p", "vlPFC_a", "FP_p", "FP_a", 
                "SMA_p", "SMA_a", "preSMA_p", "preSMA_a") # to avoid accidental changes to REGIONS_IN_ORDER
assert _REGIONS_IN_ORDER_COMBINED == ("M1", "PMv", "PMd", "dlPFC", "vlPFC", "FP",  "SMA", "preSMA")
