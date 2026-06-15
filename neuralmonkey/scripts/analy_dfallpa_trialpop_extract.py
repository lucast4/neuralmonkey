""" One goal - extraction and saving of a DFallpa object, which can then be used
for any analyses.
"""

from neuralmonkey.classes.session import load_mult_session_helper
import os
from pythonlib.globals import PATH_DATA_NEURAL_PREPROCESSED
from neuralmonkey.classes.session import load_mult_session_helper
from pythonlib.tools.plottools import savefig
import pickle 
import sys

if __name__=="__main__":
    # e..g, python analy_dfallpa_extract.py Pancho 240619 PIG_BASE_trial 0
    animal = sys.argv[1]
    date = int(sys.argv[2])

    savedir = f"{PATH_DATA_NEURAL_PREPROCESSED}/PA_trialpop/{animal}-{date}"
    os.makedirs(savedir, exist_ok=True)

    MS = load_mult_session_helper(date, animal)   

    if False:
        # Not good -- this shifts spike times.
        PA, fig = MS.popanal_timewarp_rel_events_SHIFTSPIKES(PLOT=True)
        savefig(fig, f"{savedir}/example_warping.pdf")
    else:
        # Better, linear interpolation of fr.
        PA, fig1, fig2, fig3 = MS.popanal_timewarp_rel_events_INTERPRATES(PLOT=True)
        savefig(fig1, f"{savedir}/example_warping-1.pdf")
        savefig(fig2, f"{savedir}/example_warping-2.pdf")
        savefig(fig3, f"{savedir}/times_scatter-3.pdf")

    path = f"{savedir}/PA.pkl"
    with open(path, "wb") as f:
        pickle.dump(PA, f)
    
    from pythonlib.tools.expttools import writeDictToTxt

    keys = ["version", "event_times_median", "events_inner", "events_all", "ONSET_predur_rel_first_event", "OFFSET_postdur_rel_lst_event"]
    params = {k:PA.Params[k] for k in keys}
    writeDictToTxt(params, f"{savedir}/params.txt")