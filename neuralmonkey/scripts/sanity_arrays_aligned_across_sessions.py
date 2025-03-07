"""
250307, Sanity check, each time a pair of dates, comfirm that arrays are aligned, by using spike waveform statistics.

See notebook: 220607_debug_eventcode.. Cell called: "Sanity..."
"""

import os
import sys

from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    ############# USER PARAMS
    animal = sys.argv[1]
    # The two dates that will be compared.
    date1 = int(sys.argv[2]) 
    date2 = int(sys.argv[3])

    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/sanity_checks/arrays_aligned_across_days"
    savedir = f"{SAVEDIR}/{animal}-{date1}-{date2}"
    os.makedirs(savedir, exist_ok=True)

    # To get two days, 
    MS1 = load_mult_session_helper(date1, animal, MINIMAL_LOADING=True, spikes_version="tdt")   
    MS2 = load_mult_session_helper(date2, animal, MINIMAL_LOADING=True, spikes_version="tdt")   

    # Sanity check: arrays match across dates?
    sn1 = MS1.SessionsList[0]
    sn2 = MS2.SessionsList[0]

    ### v1 -- using peak-to-trough of waveform.
    dfres1 = sn1.sanity_waveforms_all_arrays_extract()
    dfres2 = sn2.sanity_waveforms_all_arrays_extract()

    # Plot -- make a heatmap showing all the channels
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    _, _, DictSubplotsDf1 = plot_subplots_heatmap(dfres1, "region", "site_within", "peak_minus_trough", None, False, 
                                                True, annotate_heatmap=False, W = 6, return_dfs=True)
    _, _, DictSubplotsDf2 = plot_subplots_heatmap(dfres2, "region", "site_within", "peak_minus_trough", None, False, 
                                                True, annotate_heatmap=False, W = 6, return_dfs=True)
    dat1 = DictSubplotsDf1["dummy"] # (region, electrode)
    dat2 = DictSubplotsDf2["dummy"] # (region, electrode)

    # Compute and verify match acrsss dates
    passed1 = sn1._sanity_waveforms_verify_finally(dat1, dat2, savedir, suffix="peakmintrough")

    ### v2: Another version, using entire waveform, not just the statistics
    dat1time = sn1._sanity_waveforms_concat_waveforms(dfres1)
    dat2time = sn1._sanity_waveforms_concat_waveforms(dfres2)
    
    # Compute and verify match acrsss dates
    passed2 = sn1._sanity_waveforms_verify_finally(dat1time, dat2time, savedir, suffix="waveform")

    # Make a note whether passed.
    from pythonlib.tools.expttools import writeStringsToFile
    passed = passed1 and passed2
    writeStringsToFile(f"{SAVEDIR}/{animal}-{date1}-{date2}-passed={passed}.txt", [])