""" Holds generic preprocess plots, ideally should run for all sessions
4/2023
"""

from neuralmonkey.classes.session import Session
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.multsessions import MultSessions

import matplotlib.pyplot as plt

from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED


if __name__=="__main__":
    import sys
    import os 
    DATE = sys.argv[1]

    PLOT_ALLTRIALS = sys.argv[2]=="y"
    PLOT_EVENTS = sys.argv[3]=="y"
    PLOT_EVENTS_TRIALBLOCKED = sys.argv[4]=="y"

    ANIMAL = "Pancho"
    MS = load_mult_session_helper(DATE, ANIMAL)

    for sn in MS.SessionsList:
        
        # SDIR = f"{PATH_DATA_NEURAL_PREPROCESSED}/plots/{ANIMAL}/{DATE}/{sn.RecSession}"
        # SDIR = f"{PATH_DATA_NEURAL_PREPROCESSED}/plots/{ANIMAL}/eachsite_alltrials/{DATE}/{sn.RecSession}"

        if PLOT_ALLTRIALS:
            # sdir = f"{SDIR}/eachsite_alltrials"
            sdir = f"{PATH_DATA_NEURAL_PREPROCESSED}/plots/{ANIMAL}/eachsite_alltrials/{DATE}/{sn.RecSession}"
            os.makedirs(sdir, exist_ok=True)
            sn.plotbatch_alltrails_for_each_site(sdir=sdir)

        if PLOT_EVENTS:
            sdir = f"{PATH_DATA_NEURAL_PREPROCESSED}/plots/{ANIMAL}/events_subplots/{DATE}/{sn.RecSession}"
            os.makedirs(sdir, exist_ok=True)
            sn.plotbatch_rastersevents_each_site(sdir=sdir)

        if PLOT_EVENTS_TRIALBLOCKED:
            sdir = f"{PATH_DATA_NEURAL_PREPROCESSED}/plots/{ANIMAL}/events_subplots_trialsblocked/{DATE}/{sn.RecSession}"
            os.makedirs(sdir, exist_ok=True)
            sn.plotbatch_rastersevents_blocked(sdir=sdir)



        


