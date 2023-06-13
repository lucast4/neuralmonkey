""" for things for monkeylogic. uses the code there but here are wrappers
"""

from pythonlib.globals import PATH_DRAWMONKEY_DIR
# PATH_DRAWMONKEY_DIR = "/data1/code/python/drawmonkey"
import sys
sys.path.append(PATH_DRAWMONKEY_DIR)

from tools.preprocess import loadSingleDataQuick, getSessionsList
from tools.utils import * 
from tools.plots import plotDatStrokes


def ml2_get_trial_onset(fd, trialml):
    """ Return time of behcode 9. This is always close to 0,
    but not quite 
    """
    bc = getTrialsBehCodes(fd, trialml)
    if bc["num"] is None:
        print(trialml)
        print(fd["params"])
        assert False
    for num, time in zip(bc["num"], bc["time"]):
        if num==9:
            return time
    assert False

def loadSingleDataQuick(a, d, e, s):
    from tools.preprocess import loadSingleDataQuick
    fd = loadSingleDataQuick(a,d,e,s)
    if fd is None:
        print("++++ (params attempted for loading beh eval. [beh-specific parmas]", a, d, e, s)
        assert False
    return fd

def _load_sessions_corrupted():
    """
    Load metadat indicating which beh sessions were corrputed.
    RETURNS:
    - sessdict, keys are animals, values are (date, sess), e..g,m
    {'Pancho': [(230320, 0)]}
    """
    from pythonlib.tools.expttools import load_yaml_config
    from pythonlib.globals import PATH_NEURALMONKEY
    path = f"{PATH_NEURALMONKEY}/metadat/misc/corrupted_ml2_sessions.yaml"
    return load_yaml_config(path)

def _load_session_mapper(animal, date = None):
    """ Load a dict (metadat) mapping from session (neural) to beh.
    PARAMS:
    - date, if None, returns the dict. if date is int (YYMMDD), returns its
    map (if doesnt exist, then returns None)
    RETURNS:
    - either:
    --- dict, all mappings, with date:list_of_ints (1-indexed beh sessions, in oprder)
    --- list of ints (see above)
    --- None, gave date, but didnt find
    """    
    from .directory import find_rec_session_paths
    from pythonlib.tools.expttools import load_yaml_config
    from pythonlib.globals import PATH_NEURALMONKEY
    path = f"{PATH_NEURALMONKEY}/metadat/session_mappings/rec_to_beh_{animal}.yaml"
    pathdict = load_yaml_config(path)
    if date is None:
        # Return dict
        return pathdict
    elif isinstance(date, int):
        if date in pathdict.keys():
            # SANITY CHECK
            
            # First, confirm that the mapping includes all and only all the sessions
            sessionslist = find_rec_session_paths(animal, date)
            # print(pathdict, date)
            # print(pathdict[date])
            # print(len(sessionslist))
            # for x in sessionslist:
            #     print(x["sessnum"])

            # check that the beh sessions exist
            # for beh_sess in pathdict[date]:
            #     assert beh_sess in 
            # assert len(pathdict[date]) == len(sessionslist)
            return pathdict[date]
        else:
            return None
    else:
        print(date)
        assert False

def session_map_from_rec_to_ml2_ntrials_mapping(animal, date, rec_session, DEBUG=False):
    """ Help to identify ampping between nerual and beh, if beh was split into
    multiple sessions, with fierst trial of first session matching firsr trial of
    neural. 
    Loads the fd for each beh session, extracts its n trials, and increments the 
    output data.
    NOTE: for different scenario when one beh session across multiple rec sessions,
    should write something similar.
    PARAMS;
    - animal, date, rec_session, see session_map_from_rec_to_ml2
    RETURNS:
    - beh_trial_map_list, input into Session.
    e.g., [(20,0), (1,200)] means that the first fd's trial 20 maps onto trial 0 neural and
    the second fd starts (trial 1) from trial 200 neural. Pass in None to try to autoamtically 
    figure out. does so by assuming that neural and beh recording starting on the same trial.
    """

    _, _, sessdict, _ = session_map_from_rec_to_ml2(animal, date, 0)
    # sessdict =
    # (2, 'priminvar3e', {'220719': [(1, 'priminvar3e'), (2, 'priminvar3e')]})

    # beh_session = beh_session[0]
    # exptname = exptname[0]    

    if DEBUG:
        print("HERE")
        print(animal, date, rec_session)
        # print(beh_session, exptname, sessdict)
        print(sessdict)

    # Iterate over each beh session
    n_beh_sessions = len(sessdict[date])
    beh_trial_map_list = []
    tdt_trial_current = 0
    for i in range(n_beh_sessions):
        e = sessdict[date][i][1]
        s = sessdict[date][i][0]
        fd = loadSingleDataQuick(animal, date, e, s)
        ntrials = fd["params"]["n_trials"]
    #     'n_trialoutcomes': 653,]
        beh_trial_map_list.append(tuple([1, tdt_trial_current])) # update mapping.
        tdt_trial_current = tdt_trial_current + ntrials # increment n beh trails

    # Get list of beh sessions and indices
    beh_expt_list = [sess_expt[1] for sess_expt in sessdict[date]]
    beh_sess_list = [sess_expt[0] for sess_expt in sessdict[date]]

    return beh_expt_list, beh_sess_list, beh_trial_map_list

def session_map_from_rec_to_ml2(animal, date, rec_session):
    """ [Wrapper] Helper to map from rec_session to beh data session (ml2). Takes care
    of all mistakes/variations, misalignemnets between rec and beh.
    Looks in metadata for hand-entered mapping. if doesnt find that,
    then uses rec_session+1.
    PARAMS:
    - date, int or str, YYMMDD.
    - rec_session, int, (0-indexed)
    RETURNS:
    - if rec/beh session both exist:
    - beh_sess_list, list of ints
    - beh_expt_list, list of ml22 expt names.
    --- sessdict, dict for this day holding all sessions (beh), including those that 
    you might not acutally use (ignroed)
    - if doesnt exist:
    --- None
    """
    from neuralmonkey.metadat.session_mappings.beh_trial_map_list import load_beh_trial_map_list

    # assume that beh sessions are indexed by neural rec sessions
    # beh_session = rec_session+1    

    # Which beh session maps to this neural session?
    session_map = _load_session_mapper(animal, int(date)) # e..g [3,4,5] for beh sessions
    sessdict = getSessionsList(animal, datelist=[date]) 
    
    # confirm that each sess num (beh) only occurs once
    tmp = [x[0] for x in sessdict[date]]
    assert len(tmp)==len(list(set(tmp))), f"PRoblem: a sess index (beh) occurs multiple times {sessdict}"
    beh_trial_map_list = load_beh_trial_map_list(animal=animal, DATE=date, rec_session=rec_session)

    if beh_trial_map_list == [(1, 0)]: # Default, no hand coded mods.
        #### ONE REC SESSION - ONE BEH SESSION
        # Prune sessdict that that it only keeps the beh sessions that are used.
        assert list(sessdict.keys()) == [date]
        if session_map is not None:
            sessdict[date] = [sess_expt for sess_expt in sessdict[date] if sess_expt[0] in session_map]

            # then use hand-entered session
            if rec_session+1 > len(session_map):
                return None
            beh_session = session_map[rec_session]
            print("Beh Sessions hand netered (mapping: rec sess --> beh sess): ", session_map)
        else:

            # Then use rec_session+1 (indexing into the beh sessions that exist.)
            if rec_session+1 > len(sessdict[date]):
                return None

            # Second, pull out this session.
            beh_session = sessdict[date][rec_session][0]
            print("Beh Sessions that exist on this date: ", sessdict)

        # Convert to list
        beh_sess_list = [beh_session]
        beh_expt_list = [sessdict[date][rec_session][1]]

    else:
        #### ONE REC SESSION - MULTIPLE BEH SESSIONS
        # then splitting neural sess to mult beh sessions.
        # e.g.,:
            # session_map: None
            # sessdict: {'220719': [(1, 'priminvar3e'), (2, 'priminvar3e')]}
            # beh_trial_map_list: [(1, 0), (1, 45)] # two beh sessions, neurla trial 0-44 and 45+.

        # Confirm that the num beh sessions matches how many you indicate in map list.
        if not len(beh_trial_map_list)==len(sessdict[date]):
            print("session_map:", session_map)
            print("sessdict:", sessdict)
            print("beh_trial_map_list:", beh_trial_map_list)
            assert False, "prob too many beh sessions exist. I don't have a solution yet other than to delete the mkl .pkl files. Really what you want is database of rec-->(mult ml2 sessions), and then use that in session_map_from_rec_to_ml2_ntrials_mapping to auto get it"

        # Take the number of desired beh sessions.
        beh_sess_list = [x[0] for x in sessdict[date]]
        beh_expt_list = [x[1] for x in sessdict[date]]
    # print("HERE:")
    # print(beh_sess_list)
    # print(beh_expt_list)
    # assert False

    # print("taking this beh session:", beh_session)
    # beh_expt_list = [sess_expt[1] for sess_expt in sessdict[date] if sess_expt[0]==beh_session]
    # if len(beh_expt_list)!=1:
    #     print(beh_expt_list)
    #     print(sessdict, date)
    #     assert False, "multiple expts in this folder?"
    # exptname = beh_expt_list[0]

    # return beh_session, exptname, sessdict
    return beh_sess_list, beh_expt_list, sessdict, beh_trial_map_list



