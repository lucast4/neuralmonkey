""" for things for monkeylogic. uses the code there but here are wrappers
"""


PATH_TO_DRAWMONKEY = "/data1/code/python/drawmonkey"
import sys
sys.path.append(PATH_TO_DRAWMONKEY)

from tools.preprocess import loadSingleDataQuick, getSessionsList
from tools.utils import *
from tools.plots import plotDatStrokes


def ml2_get_trial_onset(fd, trialml):
    """ Return time of behcode 9. This is always close to 0,
    but not quite 
    """
    bc = getTrialsBehCodes(fd, trialml)
    for num, time in zip(bc["num"], bc["time"]):
        if num==9:
            return time
    assert False

def loadSingleDataQuick(a, d, e, s):
    from tools.preprocess import loadSingleDataQuick
    fd = loadSingleDataQuick(a,d,e,s)
    return fd
