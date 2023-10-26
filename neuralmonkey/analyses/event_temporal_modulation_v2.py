""" Meant to replace 
event_temporal_modulation
becuase SP is updated, and much there doesnt work.
10/25/23 - ALL DONE, except kernel stuff.
See below, where PLOT_KERNELS==False. If can turn this to True, the you have completed
the port
"""

import pickle
import seaborn as sns
import pandas as pd
import os

def preprocess_and_extract(SP, SAVEDIR=None):
    """
    Compute modulation by each event, for each site, and return
    results in a new dataramme where each row is (site x event).
    """
    from neuralmonkey.metrics.scalar import MetricsScalar
    from pythonlib.tools.pandastools import applyFunctionToAllRows

    Mscal = MetricsScalar(SP.DfScalar)

    ##### COMPUTE EVENT MODULATION
    res_all = []
    list_event_get = Mscal.ListEventsUniqname
    list_site_good = SP.Sites

    print("Running Mscal.modulationbytime_calc_this using these events:")
    for ev in list_event_get:
        print(ev)

    for site in list_site_good:
        if site%20==0:
            print(site)

        # 1) Compute matrix for trial-level data
        res_this = Mscal.modulationbytime_calc_this(site, list_event=list_event_get)
        res_all.extend(res_this)
    df_modtime = pd.DataFrame(res_all)
    df_modtime["event_var_level"] = df_modtime["event"] # Legacy

    # give site_area names
    def F(x):
        return SP.SN.sitegetter_summarytext(x["site"])
    df_modtime = applyFunctionToAllRows(df_modtime, F, newcolname="site_region")

    def F(x):
        return SP.SN.sitegetter_map_site_to_region(x["site"])
    df_modtime = applyFunctionToAllRows(df_modtime, F, newcolname="region")

    # SAVE
    if SAVEDIR:
        path = f"{SAVEDIR}/df_modtime.pkl"
        with open(path, "wb") as f:
            pickle.dump(df_modtime, f)

        path = f"{SAVEDIR}/Mscal.pkl"
        with open(path, "wb") as f:
            pickle.dump(Mscal, f)    

    return df_modtime


def plot_overview(df_modtime, SP, SAVEDIR, response = "r2_time_minusmean"):
    """ Helkper for the overall plots to look at event encoding
    """
    from neuralmonkey.neuralplots.brainschematic import plot_df

    # 1) old plots
    from neuralmonkey.analyses.event_temporal_modulation import plot_overview as po
    po(df_modtime, SAVEDIR=SAVEDIR, PLOT_KERNELS=False)

    # 2) Brain schematic.
    sdir = f"{SAVEDIR}/brain_schematic"
    os.makedirs(sdir, exist_ok=True)
    plot_df(df_modtime, "r2_time_minusmean", None, savedir=sdir)

    # 3) z-scored heatmaps of activity
    sdir = f"{SAVEDIR}/heatmaps_smfr"
    os.makedirs(sdir, exist_ok=True)
    for ZSCORE in [True, False]:
        SP.plotgood_heatmap_smfr(sdir=sdir, ZSCORE=ZSCORE)
