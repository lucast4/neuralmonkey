from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper


# DATE = "221020"
# animal = "Pancho"
# dataset_beh_expt = None

DATE = "220616"
animal = "Pancho"
dataset_beh_expt = None

MS = load_mult_session_helper(DATE, animal)
for sn in MS.SessionsList:
    sn.datasetbeh_load_helper(dataset_beh_expt)


from neuralmonkey.analyses.event_temporal_modulation import preprocess_and_plot, plot_overview
import os


# GOOD
for session in range(len(MS.SessionsList)):
    
    SAVEDIR = f"/gorilla1/analyses/recordings/main/events_modulation/{animal}-{DATE}-sess_{session}"
    os.makedirs(SAVEDIR, exist_ok=True)
    
    SP_trial, SP_stroke, Mscal, df_modtime = preprocess_and_plot(MS, SAVEDIR, session=session, DEBUG=False)
