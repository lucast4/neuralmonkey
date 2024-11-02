"""
Plot multiple days and bregions, for results of euclidian_time_resolved_wrapper in 
analy_euclidian_chars_sp.py

I.e, the results of analysis #2

"""
from neuralmonkey.classes.session import _REGIONS_IN_ORDER, _REGIONS_IN_ORDER_COMBINED

import pandas as pd

# Load all data (all bregions and dates)

# n_min_trials_per_shape = 5
# raw_subtract_mean_each_timepoint = False

# NPCS_KEEP = 6

# twind_analy = (-0.35, 0.5)
# tbin_dur = 0.1
# tbin_slide = 0.02

SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/MULT"
 
if False:
    # Pancho (good dates, first run)
    LIST_ANIMAL_DATE_COMB = [
        ("Pancho", 230119, False),
        ("Pancho", 230120, False),
        ("Pancho", 230122, False),
        ("Pancho", 230125, False),
        ("Pancho", 230126, False),
        ("Pancho", 230127, False),
    ]
elif False:
    # Pancho (older dates, second run)
    LIST_ANIMAL_DATE_COMB = [
        # ("Pancho", 220531, False),
        # ("Pancho", 220602, False),
        ("Pancho", 220603, False),
        ("Pancho", 220618, False),
        ("Pancho", 220626, False),
        ("Pancho", 220628, False),
        ("Pancho", 220630, False),
    ]
else:
    # Diego (all)
    LIST_ANIMAL_DATE_COMB = [
        # ("Diego", 231130, True),
        ("Diego", 231205, True),
        ("Diego", 231211, True),
        ("Diego", 231122, True),
        ("Diego", 231128, True),
        ("Diego", 231129, True),
        ("Diego", 231201, True),
        ("Diego", 231213, True),
        ("Diego", 231204, True),
    ]


list_df = []

for animal, date, combine in LIST_ANIMAL_DATE_COMB:
    SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/{animal}-{date}-combine={combine}"

    if combine:
        REGIONS = _REGIONS_IN_ORDER_COMBINED
    else:
        REGIONS = _REGIONS_IN_ORDER
        REGIONS = [r for r in REGIONS if not r=="PMv_l"]

    for bregion in REGIONS:
        for prune_version in ["sp_char_0", "pig_char_0", "sp_char", "pig_char"]:
            if prune_version in ["sp_char_0"]:
                subspace_projection_extra = "shape_prims_single"
            elif prune_version in ["pig_char_0", "pig_char", "sp_char"]:
                subspace_projection_extra = "shape_all"
            else:
                print(prune_version)
                assert False
                
            for subspace_projection in [None, "pca", subspace_projection_extra]:
                for remove_drift in [False]:
                    for raw_subtract_mean_each_timepoint in [False]:
                        for remove_singleprims_unstable in [False, True]:
                            SAVEDIR = f"{SAVEDIR_ANALYSIS}/{bregion}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-subtrmean={raw_subtract_mean_each_timepoint}"

                            print(SAVEDIR)
                            try:
                                path = f"{SAVEDIR}/DFDIST.pkl"
                                dfdist = pd.read_pickle(path)
                            except FileNotFoundError as err:
                                if animal=="Pancho" and bregion in ["FP_a", "FP_p", "FP", "PMv_l"]:
                                    print("Skipping this region:", bregion, animal)
                                elif animal=="Diego" and bregion in ["dlPFC_p", "FP_a", "FP_p", "FP"]:
                                    print("Skipping this region:", bregion, animal)
                                else:
                                    raise err

                            dfdist["animal"] = animal
                            dfdist["date"] = date
                            dfdist["bregion"] = bregion
                            dfdist["prune_version"] = prune_version
                            dfdist["subspace_projection"] = subspace_projection
                            dfdist["remove_drift"] = remove_drift
                            dfdist["raw_subtract_mean_each_timepoint"] = raw_subtract_mean_each_timepoint
                            dfdist["remove_singleprims_unstable"] = remove_singleprims_unstable

                            list_df.append(dfdist)

DFDIST = pd.concat(list_df).reset_index(drop=True)

from pythonlib.tools.pandastools import append_col_with_grp_index

DFDIST = append_col_with_grp_index(DFDIST, ["prune_version", "subspace_projection", "remove_drift", "raw_subtract_mean_each_timepoint", 
                                   "remove_singleprims_unstable"], "metaparams")
DFDIST["metaparams"].value_counts()
a = "_".join(set([x[0] for x in LIST_ANIMAL_DATE_COMB]))
b = min([x[1] for x in LIST_ANIMAL_DATE_COMB])
c = max([x[1] for x in LIST_ANIMAL_DATE_COMB])


savedir = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}"
print(savedir)
import os
os.makedirs(savedir, exist_ok=True)


################################################ 
### Combine across days
list_metaparam = DFDIST["metaparams"].unique().tolist()
list_metaparam
savedir = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}/agg"
print(savedir)
import os
os.makedirs(savedir, exist_ok=True)
from pythonlib.tools.pandastools import aggregGeneral
import seaborn as sns
from pythonlib.tools.plottools import savefig
import matplotlib.pyplot as plt

list_y = ["dist_mean", "dist_norm", "dist_yue_diff"]

for metaparam in list_metaparam:
    DFTHISthis  = DFDIST[(DFDIST["metaparams"] == metaparam)].reset_index(drop=True)
    DFTHISthisAgg = aggregGeneral(DFTHISthis, ["animal", "date", "bregion", "metaparams", "time_bin", "time_bin_idx", "same-task|shape"], values=["dist_mean", "dist_norm", "dist_yue_diff"])

    # print(len(DFTHISthis))
    # print(len(DFTHISthisAgg))

    # Eacg dat gest 
    for y in list_y:

        # Grand mean
        fig = sns.relplot(data=DFTHISthisAgg, x="time_bin", y=y, hue="same-task|shape", kind="line", col="bregion", col_wrap=6,
                        errorbar=("ci", 68))
        for ax in fig.axes.flatten():
            ax.axvline(0, color="k", alpha=0.5)
            ax.axhline(0, color="k", alpha=0.5)
        savefig(fig, f"{savedir}/grand_mean-{metaparam}-{y}.pdf")

        # All expts single plot
        fig = sns.relplot(data=DFTHISthis, x="time_bin", y=y, hue="same-task|shape", kind="line", 
                        col="bregion", row = "date", errorbar=("ci", 68))

        for ax in fig.axes.flatten():
            ax.axvline(0, color="k", alpha=0.5)
            ax.axhline(0, color="k", alpha=0.5)
        savefig(fig, f"{savedir}/each_expt-{metaparam}-{y}.pdf")

        plt.close("all")
        
################################################ 
##### One plot per day and metaparams set [takes longer]
import seaborn as sns
from pythonlib.tools.plottools import savefig
import matplotlib.pyplot as plt


list_prune_version = ["sp_char_0", "sp_char"]
# list_y = ["dist_mean", "dist_norm", "dist_yue_diff"]
list_y = ["dist_yue_diff"]

for date in DFDIST["date"].unique().tolist():
    for metaparam in DFDIST["metaparams"].unique().tolist():
        DFDISTthis = DFDIST[(DFDIST["date"] == date) & (DFDIST["metaparams"] == metaparam) & (DFDIST["prune_version"].isin(list_prune_version))].reset_index(drop=True)
        if len(DFDISTthis)>0:
            for y in list_y:
                fig = sns.relplot(data=DFDISTthis, x="time_bin", y=y, hue="same-task|shape", kind="line", col="bregion", col_wrap=6,
                                errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axvline(0, color="k", alpha=0.5)
                    ax.axhline(0, color="k", alpha=0.5)
                path = f"{savedir}/{date}-catplot-prune_subspace_rmvdrift_subtrmean_rmvunstable={metaparam}-y={y}.pdf"
                print(path)
                savefig(fig, f"{savedir}/{date}-catplot-prune_subspace_rmvdrift_subtrmean_rmvunstable={metaparam}-y={y}.pdf")

                plt.close("all")
