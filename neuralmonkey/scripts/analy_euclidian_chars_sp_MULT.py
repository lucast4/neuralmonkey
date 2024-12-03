"""
Plot multiple days and bregions, for results of euclidian_time_resolved_wrapper in 
analy_euclidian_chars_sp.py

I.e, the results of analysis #2

"""
from neuralmonkey.classes.session import _REGIONS_IN_ORDER, _REGIONS_IN_ORDER_COMBINED

import pandas as pd
import os
from pythonlib.tools.pandastools import aggregGeneral
import seaborn as sns
from pythonlib.tools.plottools import savefig
import matplotlib.pyplot as plt
from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.tools.pandastools import replace_values_with_this
from pythonlib.tools.pandastools import grouping_print_n_samples
from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

# Load all data (all bregions and dates)

# n_min_trials_per_shape = 5
# raw_subtract_mean_each_timepoint = False

# NPCS_KEEP = 6

# twind_analy = (-0.35, 0.5)
# tbin_dur = 0.1
# tbin_slide = 0.02
SAVEDIR_MULT = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/MULT"

def load_all_dates(LIST_ANIMAL_DATE_COMB, which_level, savedir_method_old):
    """
    Load and collect all dates into a single dataframe.
    LIST_ANIMAL_DATE_COMB, should have just one animal, date
    PARAMS:
    - savedir_method_old, this is for stroke.
    """

    if which_level == "stroke":
        events = ["00_stroke"]
    elif which_level == "trial":
        events = ["03_samp", "05_first_raise"]
    else:
        assert False


    list_df = []
    paths_loaded = []
    for animal, date, combine in LIST_ANIMAL_DATE_COMB:
        if savedir_method_old:
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/{animal}-{date}-combine={combine}"
        else:
            SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/euclidian_char_sp/EUCL_TIME_RESOLV/{animal}-{date}-combine={combine}-wl={which_level}"

        if combine:
            REGIONS = _REGIONS_IN_ORDER_COMBINED
        else:
            REGIONS = _REGIONS_IN_ORDER
            # REGIONS = [r for r in REGIONS if not r=="PMv_l"]

        for bregion in REGIONS:
            for ev in events:
                for prune_version in ["sp_char_0", "pig_char_0", "sp_char", "pig_char", "pig_char_1plus"]:
                    for subspace_projection in [None, "pca", "shape_all", "shape_prims_single", "task_shape_si"]:
                        for remove_drift in [False, True]:
                            for raw_subtract_mean_each_timepoint in [False, True]:
                                for remove_singleprims_unstable in [False, True]:
                                    if savedir_method_old:
                                        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{bregion}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-subtrmean={raw_subtract_mean_each_timepoint}"
                                    else:
                                        SAVEDIR = f"{SAVEDIR_ANALYSIS}/{which_level}-{bregion}-{ev}-prune={prune_version}-ss={subspace_projection}-nodrift={remove_drift}-SpUnstable={remove_singleprims_unstable}-subtrmean={raw_subtract_mean_each_timepoint}"

                                    print(SAVEDIR)
                                    try:
                                        path = f"{SAVEDIR}/DFDIST.pkl"
                                        dfdist = pd.read_pickle(path)
                                        paths_loaded.append(path)
                                    except FileNotFoundError as err:
                                        print("Skipping this region:", bregion, animal)
                                        continue
                                        # if animal=="Pancho" and bregion in ["FP_a", "FP_p", "FP", "PMv_l"]:
                                        #     print("Skipping this region:", bregion, animal)
                                        # elif animal=="Diego" and bregion in ["dlPFC_p", "FP_a", "FP_p", "FP"]:
                                        #     print("Skipping this region:", bregion, animal)
                                        # else:
                                        #     raise err

                                    dfdist["animal"] = animal
                                    dfdist["date"] = date
                                    dfdist["bregion"] = bregion
                                    dfdist["combine_areas"] = combine
                                    dfdist["event"] = ev

                                    # Metaparams
                                    dfdist["prune_version"] = prune_version
                                    dfdist["subspace_projection"] = subspace_projection
                                    dfdist["subspace_twind"] = "unknown"
                                    dfdist["remove_drift"] = remove_drift
                                    dfdist["raw_subtract_mean_each_timepoint"] = raw_subtract_mean_each_timepoint
                                    dfdist["remove_singleprims_unstable"] = remove_singleprims_unstable

                                    list_df.append(dfdist)

    print("... Loaded these paths:")
    for p in paths_loaded:
        print(p)

    ### Collect across days
    DFDIST = pd.concat(list_df).reset_index(drop=True)
    DFDIST = append_col_with_grp_index(DFDIST, ["prune_version", "subspace_projection", "subspace_twind", "remove_drift", "raw_subtract_mean_each_timepoint", 
                                    "remove_singleprims_unstable"], "metaparams")
    replace_values_with_this(DFDIST, "subspace_projection", None, "none")

    # Some plots use this...
    if "prune_version" not in DFDIST:
        DFDIST["prune_version"] = "dummy"
    if "remove_trials_with_bad_strokes" not in DFDIST:
        DFDIST["remove_trials_with_bad_strokes"] = "dummy"

    # Check there are no Nones
    assert DFDIST.isnull().values.any() == False, "replace Nones using replace_values_with_this"

    return DFDIST


def plot_timecourse_all(DFDIST, SAVEDIR_MULT, var1="task", var2="shape"):
    """
    """
    from pythonlib.tools.pandastools import stringify_values
    
    DFDIST = stringify_values(DFDIST)

    # Sanity check, the following plot code assumes this is just a single (animal, combine_areas)
    tmp = DFDIST["combine_areas"].unique().tolist()
    assert len(tmp)==1
    
    tmp = DFDIST["animal"].unique().tolist()
    assert len(tmp)==1

    # Get a param
    tmp = DFDIST["combine_areas"].unique().tolist()
    combine = tmp[0]
    
    var_same = f"same-{var1}|{var2}"

    ### SAVING
    if True:
        ### Summarize extracted cases
        savepath = f"{SAVEDIR_MULT}/extracted_cases.txt"
        grouping_print_n_samples(DFDIST, ["animal", "date", "combine_areas", "prune_version", "event", "subspace_projection", "subspace_twind", 
                                          "remove_trials_with_bad_strokes", "remove_drift", "raw_subtract_mean_each_timepoint", "remove_singleprims_unstable", 
                                          "bregion"], savepath=savepath)

        ############################################# 
        ### Plots

        ################################################ 
        ### Combine across days
        savedir = f"{SAVEDIR_MULT}/agg"
        print(savedir)
        os.makedirs(savedir, exist_ok=True)

        # list_y = ["dist_mean", "dist_norm", "dist_yue_diff"]
        list_y = ["dist_yue_diff"]
        list_metaparam = DFDIST["metaparams"].unique().tolist()
        for metaparam in list_metaparam:
            DFTHISthis  = DFDIST[(DFDIST["metaparams"] == metaparam)].reset_index(drop=True)
            DFTHISthisAgg = aggregGeneral(DFTHISthis, ["animal", "date", "combine_areas", "event", "bregion", "metaparams", "time_bin", "time_bin_idx", var_same], values=["dist_mean", "dist_norm", "dist_yue_diff"])

            # Eacg dat gest 
            for y in list_y:

                # Grand mean
                fig = sns.relplot(data=DFTHISthisAgg, x="time_bin", y=y, hue=var_same, kind="line", col="bregion", row="event",
                                errorbar=("ci", 68))
                for ax in fig.axes.flatten():
                    ax.axvline(0, color="k", alpha=0.5)
                    ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/grand_mean-{metaparam}-{y}.pdf")

                # All expts single plot
                for event in DFTHISthis["event"].unique().tolist():
                    dfthis = DFTHISthis[DFTHISthis["event"] == event]
                    fig = sns.relplot(data=dfthis, x="time_bin", y=y, hue=var_same, kind="line", 
                                    col="bregion", row = "date", errorbar=("ci", 68))

                    for ax in fig.axes.flatten():
                        ax.axvline(0, color="k", alpha=0.5)
                        ax.axhline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/event={event}-each_expt-{metaparam}-{y}.pdf")

                    plt.close("all")
                        

        ################################################ 
        ##### One plot per day and metaparams set [takes longer]
        if False: # Not needed -- can use the plot that has all dates
            assert False, "jhave tp also split by event"
            grpdict = grouping_append_and_return_inner_items_good(DFDIST, ["animal", "date", "combine_areas", "event", "metaparams"])
            for (animal, date, combine_areas, event, metaparams), inds in grpdict.items():
                DFDISTthis = DFDIST.iloc[inds].reset_index(drop=True)
                for y in list_y:
                    fig = sns.relplot(data=DFDISTthis, x="time_bin", y=y, hue=var_same, kind="line", col="bregion", col_wrap=6,
                                    errorbar=("ci", 68))
                    for ax in fig.axes.flatten():
                        ax.axvline(0, color="k", alpha=0.5)
                        ax.axhline(0, color="k", alpha=0.5)
                    path = f"{savedir}/{animal}-{date}-{combine_areas}-{event}-{metaparams}-catplot-y={y}.pdf"
                    print(path)
                    savefig(fig, path)
                    plt.close("all")


    # list_y = ["dist_yue_diff"]
    # for date in DFDIST["date"].unique().tolist():
    #     for event in DFDIST["event"].unique().tolist():
    #         for metaparam in DFDIST["metaparams"].unique().tolist():
    #             DFDISTthis = DFDIST[(DFDIST["date"] == date) & (DFDIST["event"] == event) & (DFDIST["metaparams"] == metaparam) & (DFDIST["prune_version"].isin(list_prune_version))].reset_index(drop=True)
    #             if len(DFDISTthis)>0:
    #                 for y in list_y:
    #                     fig = sns.relplot(data=DFDISTthis, x="time_bin", y=y, hue="same-task|shape", kind="line", col="bregion", col_wrap=6,
    #                                     errorbar=("ci", 68))
    #                     for ax in fig.axes.flatten():
    #                         ax.axvline(0, color="k", alpha=0.5)
    #                         ax.axhline(0, color="k", alpha=0.5)
    #                     path = f"{savedir}/{date}-catplot-prune_subspace_rmvdrift_subtrmean_rmvunstable={metaparam}-y={y}.pdf"
    #                     print(path)
    #                     savefig(fig, f"{savedir}/{date}-catplot-prune_subspace_rmvdrift_subtrmean_rmvunstable={metaparam}-y={y}.pdf")

    #                     plt.close("all")

    


def plot_scalar_all(DFDIST, savedir, map_event_to_listtwind, var1="task", var2="shape", reverse_axis_order=False):
    """
    All preprocessing and plots for scalar plots, using the time-resolved
    distances in DFDIST
    PARAMS:
    - reverse_axis_order, flips x and y axes in scatter plots
    """

    # stroke_twind = map_event_to_twind["00_stroke"]
    # savedir = f"{SAVEDIR}/scalar-twind={stroke_twind}"
    # os.makedirs(savedir, exist_ok=True)

    # ########################### SCALAR
    # savedir = f"{SAVEDIR}/scalar-twind={twind_scalar}"
    # os.makedirs(savedir, exist_ok=True)

    # # Prune to time window
    # DFDISTthis = DFDIST[(DFDIST["time_bin"]>=twind_scalar[0]) & (DFDIST["time_bin"]<=twind_scalar[1])].reset_index(drop=True)

    # # Agg, averaging over time
    # DFTHISscal = aggregGeneral(DFDISTthis, ["animal", "date", "combine_areas", "event", "bregion", "metaparams", "same-task|shape", "prune_version", "subspace_projection", "remove_drift", "raw_subtract_mean_each_timepoint", 
    #                                 "remove_singleprims_unstable"], values=["dist_mean", "dist_norm", "dist_yue_diff"])


    # map_event_to_twind = {
    #     # "00_stroke":[-0.3, 0.1],
    #     "00_stroke":stroke_twind,
    #     "03_samp":[0.2, 1.0],
    #     "05_first_raise":[-0.5,  0],
    # } 

    for v in map_event_to_listtwind.values():
        assert isinstance(v, list)
        assert isinstance(v[0], tuple), "Updated code so this is list_twind, not twind"

    var_same = f"same-{var1}|{var2}"

    for ev in DFDIST["event"].unique():
        assert ev in map_event_to_listtwind.keys(), "add this to map_ev..."

    print("Collecting...")
    list_df = []
    for event, list_twind in map_event_to_listtwind.items():
        for twind_scalar in list_twind:
            dfthis = DFDIST[DFDIST["event"] == event].reset_index(drop=True)
            dfthis_sub = dfthis[(dfthis["time_bin"]>=twind_scalar[0]-0.001) & (dfthis["time_bin"]<=twind_scalar[1]+0.001)].reset_index(drop=True)
            dfthis_sub["twind_scalar"] = [twind_scalar for _ in range(len(dfthis_sub))]
            list_df.append(dfthis_sub)
    DFDISTthis = pd.concat(list_df).reset_index(drop=True)
    
    # Agg, averaging over time
    print("Agging...")
    DFTHISscal = aggregGeneral(DFDISTthis, ["animal", "date", "combine_areas", "event", "twind_scalar", "bregion", "metaparams", var_same, 
                                            "prune_version", "remove_trials_with_bad_strokes", "subspace_projection", "subspace_twind", "remove_drift", "raw_subtract_mean_each_timepoint", 
                                            "remove_singleprims_unstable"], values=["dist_mean", "dist_norm", "dist_yue_diff"])

    from pythonlib.tools.expttools import writeDictToTxt
    writeDictToTxt(map_event_to_listtwind, f"{savedir}/twind_params.txt")

    from pythonlib.tools.pandastools import append_col_with_grp_index
    DFTHISscal = append_col_with_grp_index(DFTHISscal, ["event", "twind_scalar"], "event_twindscal")

    ### Combine all kinds of pairwise comparisons in the same plot
    # i.e., subplot = "prune_version"
    if reverse_axis_order:
        x_lev_manip = "1|0"
        y_lev_manip = "0|1"
    else:
        x_lev_manip = "0|1"
        y_lev_manip = "1|0"
        
    #################### PLOTS
    grpdict = grouping_append_and_return_inner_items_good(DFTHISscal, ["event_twindscal", "subspace_projection", "subspace_twind", "remove_drift", "raw_subtract_mean_each_timepoint", 
                                                                    "remove_singleprims_unstable"])
    for (event_twindscal, subspace_projection, subspace_twind, remove_drift, raw_subtract_mean_each_timepoint, remove_singleprims_unstable), inds in grpdict.items():
        dfthis = DFTHISscal.iloc[inds].reset_index(drop=True)

        _, fig = plot_45scatter_means_flexible_grouping(dfthis, var_same, x_lev_manip, y_lev_manip, 
                                            "prune_version", "dist_yue_diff", "bregion", 
                                            True, shareaxes=True, SIZE=4);
        savefig(fig, f"{savedir}/EVENT={event_twindscal}-ss={subspace_projection}_{subspace_twind}-rmvdrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-rmvunstable={remove_singleprims_unstable}.pdf")
        plt.close("all")


    ### Combine all kinds of pairwise comparisons in the same plot
    grpdict = grouping_append_and_return_inner_items_good(DFTHISscal, ["prune_version", "subspace_projection", "subspace_twind", "remove_drift", "raw_subtract_mean_each_timepoint", 
                                                                    "remove_singleprims_unstable"])
    for (prune_version, subspace_projection, subspace_twind, remove_drift, raw_subtract_mean_each_timepoint, remove_singleprims_unstable), inds in grpdict.items():
        dfthis = DFTHISscal.iloc[inds].reset_index(drop=True)

        _, fig = plot_45scatter_means_flexible_grouping(dfthis, var_same, x_lev_manip, y_lev_manip, 
                                            "event_twindscal", "dist_yue_diff", "bregion", 
                                            True, shareaxes=True, SIZE=4);    
        savefig(fig, f"{savedir}/PRUNE={prune_version}-ss={subspace_projection}_{subspace_twind}-rmvdrift={remove_drift}-subtrmean={raw_subtract_mean_each_timepoint}-rmvunstable={remove_singleprims_unstable}.pdf")
        plt.close("all")
 

# # Pancho (good dates, first run)
# LIST_ANIMAL_DATE_COMB = [
#     ("Pancho", 230119, False),
#     ("Pancho", 230120, False),
#     ("Pancho", 230122, False),
#     ("Pancho", 230125, False),
#     ("Pancho", 230126, False),
#     ("Pancho", 230127, False),
# ]

# # Pancho (older dates, second run)
# LIST_ANIMAL_DATE_COMB = [
#     # ("Pancho", 220531, False),
#     # ("Pancho", 220602, False),
#     ("Pancho", 220603, False),
#     ("Pancho", 220618, False),
#     ("Pancho", 220626, False),
#     ("Pancho", 220628, False),
#     ("Pancho", 220630, False),
# ]

# # Pancho, trial
# LIST_ANIMAL_DATE_COMB = [
#     ("Pancho", 220618, True),
#     ("Pancho", 220626, True),
#     ("Pancho", 220628, True),
#     ("Pancho", 220630, True),
#     ("Pancho", 230119, True),
#     ("Pancho", 230120, True),
#     ("Pancho", 230126, True),
#     ("Pancho", 230127, True),
# ]
# which_level = "trial"
# savedir_method_old = False
# savesuff = "trial"

# # Pancho, combined areas
# LIST_ANIMAL_DATE_COMB = [
#     ("Pancho", 220618, True),
#     ("Pancho", 220626, True),
#     ("Pancho", 220628, True),
#     ("Pancho", 220630, True),
#     ("Pancho", 230119, True),
#     ("Pancho", 230120, True),
#     ("Pancho", 230126, True),
#     ("Pancho", 230127, True),
# ]
# which_level = "stroke"
# savedir_method_old = False
# savesuff = "stroke"

# # [0] Diego (all, first run)
# LIST_ANIMAL_DATE_COMB = [
#     # ("Diego", 231130, True),
#     ("Diego", 231211, True),
#     ("Diego", 231122, True),
#     ("Diego", 231128, True),
#     ("Diego", 231129, True),
#     ("Diego", 231201, True),
#     ("Diego", 231205, True),
#     ("Diego", 231213, True),
#     ("Diego", 231204, True),
# ]
# which_level = "stroke"
# savedir_method_old = True
# savesuff = "stroke_good_no_sp_drift"

# # [1] Diego, just dates where have SP multiple bloques (subset of first run)
# LIST_ANIMAL_DATE_COMB = [
#     # ("Diego", 231130, True),
#     ("Diego", 231122, True),
#     ("Diego", 231128, True),
#     ("Diego", 231129, True),
#     ("Diego", 231201, True),
#     ("Diego", 231205, True),
# ]
# which_level = "stroke"
# savedir_method_old = False
# savesuff = "stroke_good_no_sp_drift"

# [2] Diego, [1] plus more dates without SP across multiple bloques.
# NOTE still a few more exist
LIST_ANIMAL_DATE_COMB = [
    # ("Diego", 231130, True),
    ("Diego", 231122, True), # Old [1]
    ("Diego", 231128, True), 
    ("Diego", 231129, True),
    ("Diego", 231201, True),
    ("Diego", 231205, True),
    ("Diego", 231120, True), # Added
    ("Diego", 231121, True),
    ("Diego", 231206, True),
    ("Diego", 231218, True),
    ("Diego", 231220, True),
]
which_level = "stroke"
savedir_method_old = False
savesuff = "stroke_all"


# # Diego (trial, all)
# LIST_ANIMAL_DATE_COMB = [
#     ("Diego", 231122, True),
#     ("Diego", 231128, True),
#     ("Diego", 231129, True),
#     ("Diego", 231201, True),
#     ("Diego", 231204, True),
#     ("Diego", 231205, True),
#     ("Diego", 231211, True),
#     ("Diego", 231213, True),
# ]
# which_level = "trial"
# savedir_method_old = False
# savesuff = "trial"


if __name__=="__main__":

    assert savesuff is not None

    # Load all data
    DFDIST = load_all_dates(LIST_ANIMAL_DATE_COMB, which_level, savedir_method_old)

    # Make save directory
    a = "_".join(set([x[0] for x in LIST_ANIMAL_DATE_COMB]))
    b = min([x[1] for x in LIST_ANIMAL_DATE_COMB])
    c = max([x[1] for x in LIST_ANIMAL_DATE_COMB])

    ### Summarize extracted cases
    SAVEDIR = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}-suff={savesuff}"
    plot_timecourse_all(DFDIST, SAVEDIR, var1="task", var2="shape")


    # # Sanity check, the following plot code assumes this is just a single (animal, combine_areas)
    # tmp = DFDIST["combine_areas"].unique().tolist()
    # assert len(tmp)==1
    
    # tmp = DFDIST["animal"].unique().tolist()
    # assert len(tmp)==1

    # # Get a param
    # tmp = DFDIST["combine_areas"].unique().tolist()
    # combine = tmp[0]
    
    # ### SAVING
    # a = "_".join(set([x[0] for x in LIST_ANIMAL_DATE_COMB]))
    # b = min([x[1] for x in LIST_ANIMAL_DATE_COMB])
    # c = max([x[1] for x in LIST_ANIMAL_DATE_COMB])

    # if True:
    #     ### Summarize extracted cases
    #     savedir = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}-combine={combine}-suff={savesuff}"
    #     print(savedir)
    #     os.makedirs(savedir, exist_ok=True)
    #     savepath = f"{savedir}/extracted_cases.txt"
    #     grouping_print_n_samples(DFDIST, ["animal", "date", "combine_areas", "prune_version", "subspace_projection", "remove_drift", "raw_subtract_mean_each_timepoint", 
    #                                     "remove_singleprims_unstable", "event", "bregion"], savepath=savepath)

    #     ############################################# 
    #     ### Plots

    #     ################################################ 
    #     ### Combine across days
    #     savedir = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}-combine={combine}-suff={savesuff}/agg"
    #     print(savedir)
    #     os.makedirs(savedir, exist_ok=True)

    #     # list_y = ["dist_mean", "dist_norm", "dist_yue_diff"]
    #     list_y = ["dist_yue_diff"]
    #     list_metaparam = DFDIST["metaparams"].unique().tolist()
    #     for metaparam in list_metaparam:
    #         DFTHISthis  = DFDIST[(DFDIST["metaparams"] == metaparam)].reset_index(drop=True)
    #         DFTHISthisAgg = aggregGeneral(DFTHISthis, ["animal", "date", "combine_areas", "event", "bregion", "metaparams", "time_bin", "time_bin_idx", "same-task|shape"], values=["dist_mean", "dist_norm", "dist_yue_diff"])

    #         # Eacg dat gest 
    #         for y in list_y:

    #             # Grand mean
    #             fig = sns.relplot(data=DFTHISthisAgg, x="time_bin", y=y, hue="same-task|shape", kind="line", col="bregion", row="event",
    #                             errorbar=("ci", 68))
    #             for ax in fig.axes.flatten():
    #                 ax.axvline(0, color="k", alpha=0.5)
    #                 ax.axhline(0, color="k", alpha=0.5)
    #             savefig(fig, f"{savedir}/grand_mean-{metaparam}-{y}.pdf")

    #             # All expts single plot
    #             for event in DFTHISthis["event"].unique().tolist():
    #                 dfthis = DFTHISthis[DFTHISthis["event"] == event]
    #                 fig = sns.relplot(data=dfthis, x="time_bin", y=y, hue="same-task|shape", kind="line", 
    #                                 col="bregion", row = "date", errorbar=("ci", 68))

    #                 for ax in fig.axes.flatten():
    #                     ax.axvline(0, color="k", alpha=0.5)
    #                     ax.axhline(0, color="k", alpha=0.5)
    #                 savefig(fig, f"{savedir}/event={event}-each_expt-{metaparam}-{y}.pdf")

    #                 plt.close("all")
                        

    #     ################################################ 
    #     ##### One plot per day and metaparams set [takes longer]
    #     if False: # Not needed -- can use the plot that has all dates
    #         assert False, "jhave tp also split by event"
    #         grpdict = grouping_append_and_return_inner_items_good(DFDIST, ["animal", "date", "combine_areas", "event", "metaparams"])
    #         for (animal, date, combine_areas, event, metaparams), inds in grpdict.items():
    #             DFDISTthis = DFDIST.iloc[inds].reset_index(drop=True)
    #             for y in list_y:
    #                 fig = sns.relplot(data=DFDISTthis, x="time_bin", y=y, hue="same-task|shape", kind="line", col="bregion", col_wrap=6,
    #                                 errorbar=("ci", 68))
    #                 for ax in fig.axes.flatten():
    #                     ax.axvline(0, color="k", alpha=0.5)
    #                     ax.axhline(0, color="k", alpha=0.5)
    #                 path = f"{savedir}/{animal}-{date}-{combine_areas}-{event}-{metaparams}-catplot-y={y}.pdf"
    #                 print(path)
    #                 savefig(fig, path)
    #                 plt.close("all")


    # # list_y = ["dist_yue_diff"]
    # # for date in DFDIST["date"].unique().tolist():
    # #     for event in DFDIST["event"].unique().tolist():
    # #         for metaparam in DFDIST["metaparams"].unique().tolist():
    # #             DFDISTthis = DFDIST[(DFDIST["date"] == date) & (DFDIST["event"] == event) & (DFDIST["metaparams"] == metaparam) & (DFDIST["prune_version"].isin(list_prune_version))].reset_index(drop=True)
    # #             if len(DFDISTthis)>0:
    # #                 for y in list_y:
    # #                     fig = sns.relplot(data=DFDISTthis, x="time_bin", y=y, hue="same-task|shape", kind="line", col="bregion", col_wrap=6,
    # #                                     errorbar=("ci", 68))
    # #                     for ax in fig.axes.flatten():
    # #                         ax.axvline(0, color="k", alpha=0.5)
    # #                         ax.axhline(0, color="k", alpha=0.5)
    # #                     path = f"{savedir}/{date}-catplot-prune_subspace_rmvdrift_subtrmean_rmvunstable={metaparam}-y={y}.pdf"
    # #                     print(path)
    # #                     savefig(fig, f"{savedir}/{date}-catplot-prune_subspace_rmvdrift_subtrmean_rmvunstable={metaparam}-y={y}.pdf")

    # #                     plt.close("all")

    
    ########################### SCALAR
    # twind_scalar = [-0.3, 0.1]
    SAVEDIR = f"{SAVEDIR_MULT}/{a}-{b}-to-{c}-combine={combine}-suff={savesuff}-stroke"
    os.makedirs(SAVEDIR, exist_ok=True)
    for stroke_twind in [
        (-0.4, 0.),
        (-0.3, 0.),
        (-0.2, 0.),
        (-0.4, 0.1),
        (-0.3, 0.1),
        (-0.2, 0.1),
        ]:

        map_event_to_twind = {
                # "00_stroke":[-0.3, 0.1],
                "00_stroke":stroke_twind,
                "03_samp":[0.2, 1.0],
                "05_first_raise":[-0.5,  0],
            }

    #         stroke_twind = map_event_to_twind["00_stroke"]
    # savedir = f"{SAVEDIR}/scalar-twind={stroke_twind}"
    # os.makedirs(savedir, exist_ok=True)

        savedir = f"{SAVEDIR}/scalar-twind={stroke_twind}"
        os.makedirs(savedir, exist_ok=True)
    
        # plot_scalar_all(DFDIST, SAVEDIR, twind_scalar)
        plot_scalar_all(DFDIST, savedir, map_event_to_twind)

