""" One goal - extraction of snippets, which can then be used
for any analyses.
"""

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
import sys
from neuralmonkey.classes.snippets import load_snippet_single
from pythonlib.tools.expttools import writeDictToYaml
from neuralmonkey.classes.snippets import Snippets, extraction_helper
from neuralmonkey.classes.snippets import _dataset_extract_prune_general
from pythonlib.tools.exceptions import NotEnoughDataException

#### USER PARAMS
# SPIKES_VERSION = "tdt" # Still improving ks curation.
# LIST_WHICH_LEVEL = ["trial", "stroke", "stroke_off"]
# FORCE_EXTRACT = 0
# # PRE_DUR = -0.8
# # POST_DUR = 0.8
# SAVEDIR_BASE = "/gorilla1/analyses/recordings/main/anova"


if __name__=="__main__":
    # To extract SP and save.

    #### USUALLY NOT CHANGE PARAMS
    # LIST_SESSIONS = None
    # DEBUG = False # runs fast

    animal = sys.argv[1]
    DATE = int(sys.argv[2])
    if len(sys.argv)>3:
        PLOT_EACH_TRIAL = sys.argv[3]==1
    else:
        PLOT_EACH_TRIAL = True
        
    # to help debug if times are misaligned.
    MS = load_mult_session_helper(DATE, animal)

    if False:
        # Old code session by session
        for sn in MS.SessionsList:
            sn.sitesdirtygood_preprocess_wrapper()
    else:
        # New, combines sessions before doing.
        MS.sitesdirtygood_preprocess_wrapper(PLOT_EACH_TRIAL=PLOT_EACH_TRIAL)