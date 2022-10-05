

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
    ANIMAL = "Pancho"
    MS = load_mult_session_helper(DATE, ANIMAL)

    for sn in MS.SessionsList:
        
        
        SDIR = f"{PATH_DATA_NEURAL_PREPROCESSED}/plots/{ANIMAL}/{DATE}/{sn.RecSession}"
        sdir = f"{SDIR}/eachsite_alltrials"
        os.makedirs(sdir, exist_ok=True)

        sn.plotbatch_alltrails_for_each_site(sdir=sdir)

