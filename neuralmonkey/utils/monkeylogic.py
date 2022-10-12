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
        print("++++", a, d, e, s)
        assert False
    return fd


def _load_session_mapper(animal, date = None):
    """ Load a dict (metadat) mapping from session (neural) to beh
    PARAMS:
    - date, if None, returns the dict. if date is int (YYMMDD), returns its
    map (if doesnt exist, then returns None)
    RETURNS:
    - either:
    --- dict, all mappings, with date:list_of_ints (1-indexed beh sessions, in oprder)
    --- list of ints (see above)
    --- None, gave date, but didnt find
    """

    from pythonlib.tools.expttools import load_yaml_config
    from pythonlib.globals import PATH_NEURALMONKEY
    path = f"{PATH_NEURALMONKEY}/metadat/session_mappings/rec_to_beh_{animal}.yaml"
    pathdict = load_yaml_config(path)
    if date is None:
        # Return dict
        return pathdict
    elif isinstance(date, int):
        if date in pathdict.keys():
            return pathdict[date]
        else:
            return None
    else:
        print(date)
        assert False

def session_map_from_rec_to_ml2(animal, date, rec_session):
    """ Helper to map from rec_session to beh data session (ml2).
    Looks in metadata for hand-entered mapping. if doesnt find that,
    then uses rec_session+1.
    PARAMS:
    - date, int or str, YYMMDD.
    - rec_session, int, (0-indexed)
    RETURNS:
    - if rec/beh session both exist:
    --- beh_session, int index
    --- exptname, name of expt for this session
    --- sessdict, dict for this day holding all sessions (beh).
    - if doesnt exist:
    --- None
    """


    # assume that beh sessions are indexed by neural rec sessions
    # beh_session = rec_session+1    

    # Which beh session maps to this neural session?
    session_map = _load_session_mapper(animal, int(date))
    sessdict = getSessionsList(animal, datelist=[date])
    if session_map is not None:
        # then use hand-entered session
        if rec_session+1 > len(session_map):
            return None
        beh_session = session_map[rec_session]
        print("Beh Sessions hand netered (mapping: rec sess --> beh sess): ", session_map)
    else:
        # Then use rec_session+1 (indexing into the beh sessions that exist.)
        if rec_session+1 > len(sessdict[date]):
            return None
        beh_session = sessdict[date][rec_session][0]
        print("Beh Sessions that exist on this date: ", sessdict)

    print("taking this beh session:", beh_session)
    beh_expt_list = [sess_expt[1] for sess_expt in sessdict[date] if sess_expt[0]==beh_session]
    assert len(beh_expt_list)==1, "multiple expts in this folder?"
    exptname = beh_expt_list[0]

    return beh_session, exptname, sessdict



