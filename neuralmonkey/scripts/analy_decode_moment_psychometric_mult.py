"""
Collect results and plot psychometric -- things trying to test for sigmoidal effect between base 1 and base 2.

"""

import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys

import pandas as pd
import matplotlib.pyplot as plt

from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa

from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_good_morphsets_no_switching
import numpy as np
from pythonlib.tools.vectools import projection_onto_axis_subspace
import seaborn as sns
import pickle
from pythonlib.tools.plottools import savefig
from pythonlib.tools.pandastools import plot_subplots_heatmap

from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED
from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single


def plot_with_preprocess(DFPROJ, DFPROJ_INDEX, DFDIFFS, DFDIFFSPROJ, DFDIFFSINDEX, DFDIFFS_CATEG, savedir, 
                         list_thresh_separation=None):
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.snstools import rotateLabel

    # DFPROJ = pd.concat(list_dfproj).reset_index(drop=True)
    # DFPROJ_INDEX = pd.concat(list_dfproj_index).reset_index(drop=True)

    # DFDIFFS = pd.concat(list_dfdiffs).reset_index(drop=True)
    # DFDIFFSPROJ = pd.concat(list_dfdiffsproj).reset_index(drop=True)
    # DFDIFFSINDEX = pd.concat(list_dfdiffsindex).reset_index(drop=True)
    
    DFDIFFSPROJ = append_col_with_grp_index(DFDIFFSPROJ, ["animal", "date", "morphset"], "ani_date_mrp")
    DFDIFFS = append_col_with_grp_index(DFDIFFS, ["animal", "date", "morphset"], "ani_date_mrp")
    DFPROJ = append_col_with_grp_index(DFPROJ, ["animal", "date", "morphset"], "ani_date_mrp")
    DFPROJ_INDEX = append_col_with_grp_index(DFPROJ_INDEX, ["animal", "date", "morphset"], "ani_date_mrp")
    DFDIFFSINDEX = append_col_with_grp_index(DFDIFFSINDEX, ["animal", "date", "morphset"], "ani_date_mrp")
    DFDIFFS_CATEG = append_col_with_grp_index(DFDIFFS_CATEG, ["animal", "date", "morphset"], "ani_date_mrp")
    
    DFDIFFSPROJ["dist_norm"] = DFDIFFSPROJ["dist"] * DFDIFFSPROJ["dist_between_bases"]

    # Assign generic names for plotting
    DFDIFFS_CATEG["dist"] =     DFDIFFS_CATEG["dist_yue_diff"]
    DFDIFFS_CATEG["idx_along_morph"] = DFDIFFS_CATEG["idx_boundary_left"]+0.5

    ### Label if morphset has intermediate shapes (e..g, line1-line2-line3)
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_has_intermediate_shape
    cachedict = {}
    for dfthis in [DFDIFFSPROJ, DFDIFFS, DFPROJ, DFPROJ_INDEX, DFDIFFSINDEX, DFDIFFS_CATEG]:
        list_has_intermediate = []
        for i, row in dfthis.iterrows():
            animal = row["animal"]
            date = row["date"]

            if (animal, date) not in cachedict:
                cachedict[(animal, date)] = params_has_intermediate_shape(animal, date)
            
            # Check if this has intermediate
            list_has_intermediate.append(row["morphset"] in cachedict[(animal, date)])
        dfthis["morphset_has_intermediate"] = list_has_intermediate

    ### Label whetther base prims are separated (for PMv)
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_base_prims_not_separated
    for dfthis in [DFDIFFSPROJ, DFDIFFS, DFPROJ, DFPROJ_INDEX, DFDIFFSINDEX, DFDIFFS_CATEG]:
        dfthis["keep_because_base_prims_separated"] = [row["morphset"] not in params_base_prims_not_separated(row["animal"], row["date"]) for _, row in dfthis.iterrows()]

    # all projection distance, rescale by the distance between base1 and base2, so that the aggregated data is not super noisy.
    # - i.e., 
    # DFPROJ["x_proj_norm"] = [row["x_proj"] * row["dist_norm_between_bases"] for i, row in DFPROJ.iterrows()]
    # DFPROJ["x_proj_norm"] = [row["x_proj"] * row["dist_between_bases"] for i, row in DFPROJ.iterrows()]
    # Plot results, using differences between adjacent indices

    # SCore for how well base prims are separated
    fig = sns.catplot(data=DFPROJ, x="ani_date_mrp", y="base_prim_separation_score", hue="bregion", kind="point", aspect=2)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0)
    savefig(fig, f"{savedir}/base_prim_separation_score-1.pdf")

    fig = sns.catplot(data=DFPROJ, x="ani_date_mrp", y="base_prim_separation_score", hue="nmorphs", 
                    col="bregion", col_wrap=6, kind="point", aspect=1.5)
    rotateLabel(fig)
    for ax in fig.axes.flatten():
        ax.axhline(0)
    savefig(fig, f"{savedir}/base_prim_separation_score-2.pdf")
    
    ### SAVE DATAFRAMES
    for _df, _name in [
        (DFPROJ, "DFPROJ"), 
        (DFPROJ_INDEX, "DFPROJ_INDEX"), 
        (DFDIFFS, "DFDIFFS"), 
        (DFDIFFSPROJ, "DFDIFFSPROJ"), 
        (DFDIFFSINDEX, "DFDIFFSINDEX"),
        (DFDIFFS_CATEG, "DFDIFFS_CATEG"),
        ]:
        path = f"{savedir}/{_name}.pkl"
        pd.to_pickle(_df, path)

    print("Saving at ...: ", savedir)
    if list_thresh_separation is None:
        list_thresh_separation = [0.3, 0.5, 0.75, 0.]
    for thresh_separation in list_thresh_separation:
        for keep_because_base_prims_separated in [True, False]:
            sdir = f"{savedir}/thresh-base_prim_separation_score={thresh_separation}-base_prims_separated={keep_because_base_prims_separated}"
            os.makedirs(sdir, exist_ok=True)
            _plot(DFPROJ, DFPROJ_INDEX, DFDIFFS, DFDIFFSPROJ, DFDIFFSINDEX, DFDIFFS_CATEG, thresh_separation, sdir,
                  keep_because_base_prims_separated)


def _plot(DFPROJ, DFPROJ_INDEX, DFDIFFS, DFDIFFSPROJ, DFDIFFSINDEX, DFDIFFS_CATEG, thresh_separation, savedir,
          keep_because_base_prims_separated):
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    DFDIFFSPROJ = DFDIFFSPROJ[DFDIFFSPROJ["keep_because_base_prims_separated"]==keep_because_base_prims_separated].reset_index(drop=True)
    DFDIFFS = DFDIFFS[DFDIFFS["keep_because_base_prims_separated"]==keep_because_base_prims_separated].reset_index(drop=True)
    DFPROJ = DFPROJ[DFPROJ["keep_because_base_prims_separated"]==keep_because_base_prims_separated].reset_index(drop=True)
    DFPROJ_INDEX = DFPROJ_INDEX[DFPROJ_INDEX["keep_because_base_prims_separated"]==keep_because_base_prims_separated].reset_index(drop=True)
    DFDIFFSINDEX = DFDIFFSINDEX[DFDIFFSINDEX["keep_because_base_prims_separated"]==keep_because_base_prims_separated].reset_index(drop=True)
    DFDIFFS_CATEG = DFDIFFS_CATEG[DFDIFFS_CATEG["keep_because_base_prims_separated"]==keep_because_base_prims_separated].reset_index(drop=True)

    DFDIFFSPROJ = DFDIFFSPROJ[DFDIFFSPROJ["base_prim_separation_score"]>=thresh_separation].reset_index(drop=True)
    DFDIFFS = DFDIFFS[DFDIFFS["base_prim_separation_score"]>=thresh_separation].reset_index(drop=True)
    DFPROJ = DFPROJ[DFPROJ["base_prim_separation_score"]>=thresh_separation].reset_index(drop=True)
    DFPROJ_INDEX = DFPROJ_INDEX[DFPROJ_INDEX["base_prim_separation_score"]>=thresh_separation].reset_index(drop=True)
    DFDIFFSINDEX = DFDIFFSINDEX[DFDIFFSINDEX["base_prim_separation_score"]>=thresh_separation].reset_index(drop=True)
    DFDIFFS_CATEG = DFDIFFS_CATEG[DFDIFFS_CATEG["base_prim_separation_score"]>=thresh_separation].reset_index(drop=True)

    ### (0) Plots, aligned to middle index
    dfthis = DFDIFFSINDEX
    ydist = "dist"
    xvar = "idx_along_morph_centered"

    # dfthis = DFPROJ_INDEX
    # ydist = "dist_index"
    # xvar = "idx_morph_temp_rank_centered"

    for dfthis, ydist, xvar, suff in [
        (DFDIFFSINDEX, "dist", "idx_along_morph_centered", "diffidx"),
        (DFPROJ_INDEX, "dist_index", "idx_morph_temp_rank_centered", "distidx"),
        (DFDIFFS_CATEG, "dist_yue_diff", "idx_boundary_left_centered", "diffcateg"),
        ]:
        for morphset_has_intermediate in [False, True]:
            dfthisthis = dfthis[dfthis["morphset_has_intermediate"] == morphset_has_intermediate].reset_index(drop=True)
            if len(dfthisthis)>0:

                fig = sns.catplot(data=dfthisthis, x=xvar, y=ydist, hue="ani_date_mrp", kind="point", col="bregion", row="animal")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0)
                savefig(fig, f"{savedir}/ALIGNED_INDEX-catplot-{suff}-xvar={xvar}-ydist={ydist}-mshasinterm={morphset_has_intermediate}-1.pdf")
                
                fig = sns.catplot(data=dfthisthis, x=xvar, y=ydist, kind="point", col="bregion", row="animal")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0)
                savefig(fig, f"{savedir}/ALIGNED_INDEX-catplot-{suff}-xvar={xvar}-ydist={ydist}-mshasinterm={morphset_has_intermediate}-2.pdf")

                plt.close("all")

                fig = sns.catplot(data=dfthisthis, x=xvar, y=ydist, kind="point", hue="nmorphs", col="bregion", row="animal")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0)
                savefig(fig, f"{savedir}/ALIGNED_INDEX-catplot-{suff}-xvar={xvar}-ydist={ydist}-mshasinterm={morphset_has_intermediate}-3.pdf")

                fig = sns.catplot(data=dfthisthis, x=xvar, y=ydist, hue="bregion", kind="point", col="ani_date_mrp", 
                                col_wrap=6)
                # for ax in fig.axes.flatten():
                #     ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/ALIGNED_INDEX-catplot-{suff}-xvar={xvar}-ydist={ydist}-mshasinterm={morphset_has_intermediate}-4.pdf")

                fig, _ = plot_subplots_heatmap(dfthisthis, "ani_date_mrp", xvar, ydist, "bregion", share_zlim=True, ZLIMS=None)
                savefig(fig, f"{savedir}/ALIGNED_INDEX-heatmaps_all-{suff}-xvar={xvar}-ydist={ydist}-mshasinterm={morphset_has_intermediate}.pdf")

                plt.close("all")

    ### (1) Plot differences between adjacent indices.
    for dfthis, savesuff in [
        (DFDIFFS, "eucl"), 
        (DFDIFFSPROJ, "axis_proj"),
        (DFDIFFSINDEX, "dist_index"),
        (DFDIFFS_CATEG, "diff_categ")
        ]:

        if len(dfthis)>0:
            if savesuff == "proj_diffs":
                list_dist = ["dist", "dist_norm"]
            else:
                list_dist = ["dist"]
                
            for ydist in list_dist:
                fig = sns.catplot(data=dfthis, x="idx_along_morph", y=ydist, hue="bregion", kind="point", col="ani_date_mrp", 
                                col_wrap=6)
                # for ax in fig.axes.flatten():
                #     ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/DIST_BETWEEN_ADJ_IDXS-{savesuff}-catplot-ydist={ydist}-1.pdf")

                fig = sns.catplot(data=dfthis, x="idx_along_morph", y=ydist, hue="ani_date_mrp", kind="point", col="bregion", 
                                row="nmorphs")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/DIST_BETWEEN_ADJ_IDXS-{savesuff}-catplot-ydist={ydist}-2.pdf")

                plt.close("all")

                fig = sns.catplot(data=dfthis, x="idx_along_morph", y=ydist, hue="ani_date_mrp", kind="point", col="bregion", 
                                row="morphset_has_intermediate")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/DIST_BETWEEN_ADJ_IDXS-{savesuff}-catplot-ydist={ydist}-3.pdf")

                fig = sns.catplot(data=dfthis, x="idx_along_morph", y=ydist, hue="nmorphs", kind="point", 
                                col="bregion", row="morphset_has_intermediate")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/DIST_BETWEEN_ADJ_IDXS-{savesuff}-catplot-ydist={ydist}-4.pdf")


                fig = sns.catplot(data=dfthis, x="idx_along_morph", y=ydist, kind="point", col="bregion", row="morphset_has_intermediate")
                # for ax in fig.axes.flatten():
                #     ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/DIST_BETWEEN_ADJ_IDXS-{savesuff}-catplot-ydist={ydist}-5.pdf")

                plt.close("all")

                # Heatmaps
                for morphset_has_intermediate in [False, True]:
                    dfthisthis = dfthis[dfthis["morphset_has_intermediate"] == morphset_has_intermediate].reset_index(drop=True)
                    if len(dfthisthis)>0:
                        fig, _ = plot_subplots_heatmap(dfthisthis, "ani_date_mrp", "idx_along_morph", ydist, "bregion", share_zlim=True, ZLIMS=None)
                        savefig(fig, f"{savedir}/DIST_BETWEEN_ADJ_IDXS-heatmaps_all-{savesuff}-mshasinterm={morphset_has_intermediate}.pdf")
                        plt.close("all")

    ### (2) Plot distances for each index from base prims
    for dfthis, savesuff, ydist in [
        (DFPROJ, "axis_proj", "x_proj"), 
        (DFPROJ_INDEX, "dist_index", "dist_index"), 
        ]:

        if len(dfthis)>0:

            fig = sns.catplot(data=dfthis, x="idx_morph_temp", y=ydist, hue="bregion", kind="point", col="ani_date_mrp", 
                                col_wrap=6)
            # for ax in fig.axes.flatten():
            #     ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/DIST_FROM_BASE_PRIMS-{savesuff}-catplot-ydist={ydist}-1.pdf")

            fig = sns.catplot(data=dfthis, x="idx_morph_temp", y=ydist, hue="ani_date_mrp", kind="point", col="bregion", 
                                row="nmorphs")
            # for ax in fig.axes.flatten():
            #     ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/DIST_FROM_BASE_PRIMS-{savesuff}-catplot-ydist={ydist}-2.pdf")

            fig = sns.catplot(data=dfthis, x="idx_morph_temp", y=ydist, hue="ani_date_mrp", kind="point", col="bregion", 
                                row="morphset_has_intermediate")
            # for ax in fig.axes.flatten():
            #     ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/DIST_FROM_BASE_PRIMS-{savesuff}-catplot-ydist={ydist}-3.pdf")

            plt.close("all")

            fig = sns.catplot(data=dfthis, x="idx_morph_temp", y=ydist, hue="nmorphs", kind="point", 
                            col="bregion", row="morphset_has_intermediate")
            # for ax in fig.axes.flatten():
            #     ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/DIST_FROM_BASE_PRIMS-{savesuff}-catplot-ydist={ydist}-4.pdf")


            fig = sns.catplot(data=dfthis, x="idx_morph_temp", y=ydist, kind="point", col="bregion", row="morphset_has_intermediate")
            # for ax in fig.axes.flatten():
                # ax.axhline(0, color="k", alpha=0.5)
            savefig(fig, f"{savedir}/DIST_FROM_BASE_PRIMS-{savesuff}-catplot-ydist={ydist}-5.pdf")

            plt.close("all")

            # Heatmaps
            for morphset_has_intermediate in [False, True]:
                dfthisthis = dfthis[dfthis["morphset_has_intermediate"] == morphset_has_intermediate].reset_index(drop=True)
                if len(dfthisthis)>0:
                    fig, _ = plot_subplots_heatmap(dfthisthis, "ani_date_mrp", "idx_morph_temp", ydist, "bregion", share_zlim=True, ZLIMS=None)
                    savefig(fig, f"{savedir}/DIST_FROM_BASE_PRIMS-heatmaps_all-{savesuff}-mshasinterm={morphset_has_intermediate}.pdf")
                    plt.close("all")    

    if False: # Old, not used anymore.
        from pythonlib.tools.snstools import rotateLabel
        THRESH = 0.5
        fig = sns.catplot(data=DFPROJ, x="ani_date_mrp", y="dist_norm_between_bases", col="bregion", col_wrap=4, kind="point", aspect=1.5)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0)
            ax.axhline(THRESH)
        savefig(fig, f"{savedir}/dist_norm_between_bases.pdf")

        fig = sns.catplot(data=DFPROJ, x="ani_date_mrp", y="dist_between_bases", col="bregion", col_wrap=4, kind="point", aspect=1.5)
        rotateLabel(fig)
        for ax in fig.axes.flatten():
            ax.axhline(0)
            ax.axhline(THRESH)
        savefig(fig, f"{savedir}/dist_between_bases.pdf")

        plt.close("all")


# def convert_dist_to_distdiff(dfthis, var_score, var_idx = "idx_morph_temp_rank"):
#     ########################## GET projection distance between adjacent indices
#     # Get differenes between adjancent morph indices, projected onto the axis.
    
#     list_idx_morph = sorted(dfthis[var_idx].unique().tolist())
#     res = []
#     for i in range(len(list_idx_morph)-1):
#         idx1 = list_idx_morph[i]
#         idx2 = list_idx_morph[i+1]
        
#         score1 = dfthis[dfthis[var_idx] == idx1][var_score]
#         score2 = dfthis[dfthis[var_idx] == idx2][var_score]

#         res.append({
#             # f"{var_score}-idx2-min-idx1":np.mean(score2) - np.mean(score1),
#             "dist":np.mean(score2) - np.mean(score1),
#             "idx_along_morph":i,
#             "idx1":idx1,
#             "idx2":idx2,
#             "var_idx":var_idx,
#         })
    
#     return pd.DataFrame(res)


# def compute_pa_to_df_dist_index_using_splits(PA, PRINT=False, var_idx = "idx_morph_temp_rank"):
    
#     # 1. Split into stratified dfatasets (2)
#     paredu1, paredu2 = PA.split_sample_stratified_by_label(["idx_morph_temp", "seqc_0_loc"], PRINT=PRINT)

#     dfproj_index_1, dfdiffsindex_1 = _compute_df_using_dist_index(paredu1)
#     dfproj_index_2, dfdiffsindex_2 = _compute_df_using_dist_index(paredu2)

#     if PRINT:
#         display(dfproj_index_1)
#         display(dfproj_index_2)
#         display(dfdiffsindex_1)
#         display(dfdiffsindex_2)

#     dfproj_index_2, dfdiffsindex_2 = _recenter_idx(dfdiffsindex_1, dfproj_index_2, dfdiffsindex_2)
#     dfproj_index_1, dfdiffsindex_1 = _recenter_idx(dfdiffsindex_2, dfproj_index_1, dfdiffsindex_1)

#     if PRINT:
#         display(dfproj_index_1)
#         display(dfproj_index_2)
#         display(dfdiffsindex_1)
#         display(dfdiffsindex_2)

#     # concatenate reuslts, trakcing the split index
#     dfdiffsindex_1["split_idx"] = 1
#     dfdiffsindex_2['split_idx'] = 2
#     dfdiffsindex = pd.concat([dfdiffsindex_1, dfdiffsindex_2], axis=0).reset_index(drop=True)

#     dfproj_index_1["split_idx"] = 1
#     dfproj_index_2['split_idx'] = 2
#     dfproj_index = pd.concat([dfproj_index_1, dfproj_index_2], axis=0).reset_index(drop=True)

#     # RETURNS
#     return dfdiffsindex, dfproj_index


# def _rank_idxs_append(df):
#     """
#     Rank idxs, so that 0-->0 and 99-->(max morph idx +1)
#     e.g. [-1, 0, 1, 2, 99, 100] --> [-1, 0, 1, 2, 3, 4]

#     Modifies df by appending column: idx_morph_temp_rank
#     """
#     # from pythonlib.tools.listtools import rank_items
#     # rank_items(dfproj_index_1["idx_morph_temp"], "dense")
#     idx_max = df[df["idx_morph_temp"] < 99]["idx_morph_temp"].max()
#     def f(x):
#         if x<=idx_max:
#             return x
#         elif x>=99:
#             return x - 99 + idx_max + 1
#         else:
#             assert False    
#     df["idx_morph_temp_rank"] = [f(x) for x in df["idx_morph_temp"]]
    
# def _compute_df_using_dist_index(pa):
#     """
#     Helper to extract dist (eucl, pairwise btw pts) between each level of idx_morph_temp
#     and then convert that into an index, 0...1, for how close each morph is to base1 and base2.
#     Also get diference of this score between adjacent indices
#     """

#     # 1. Convert to pairwise distances dist_yue_diff)
#     cldist, _ = euclidian_distance_compute_trajectories_single(pa, "idx_morph_temp", ["epoch"], 
#                                                             version_distance="euclidian", return_cldist=True, get_reverse_also=False)
#     dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)
    
#     dfproj_index = dfdist_to_dfproj_index(dfdist)
#     _rank_idxs_append(dfproj_index)
#     dfdiffsindex = convert_dist_to_distdiff(dfproj_index, "dist_index", "idx_morph_temp_rank")

#     return dfproj_index, dfdiffsindex

# def dfdist_to_dfproj_index(dfdist, var_score="dist_mean"):
#     """Convert pairwise distances (between idx) into dist
#     index
#     PARAMS:
#     - dist_mean, should use "dist_mean", the pairiwse distance betwene
#     all pts...
#     """

#     list_idx_morph = sorted(dfdist["idx_morph_temp_1"].unique().tolist())
#     res_dist_index = []
#     for idx in list_idx_morph:
        
#         tmp = dfdist[(dfdist['idx_morph_temp_1']==0) & (dfdist['idx_morph_temp_2']==idx)]
#         assert len(tmp)==1
#         d1 = tmp[var_score].values[0]
        
#         tmp = dfdist[(dfdist['idx_morph_temp_1']==idx) & (dfdist['idx_morph_temp_2']==99)]
#         assert len(tmp)==1
#         d2 = tmp[var_score].values[0]

#         dist_index = d1/(d1+d2)

#         # print(idx, d1, d2, " -- ", dist_index)
#         res_dist_index.append({
#             "dist_index":dist_index,
#             "idx_morph_temp":idx,
#         })
#     dfproj_index = pd.DataFrame(res_dist_index)

#     return dfproj_index


# def _recenter_idx(dfdiffsindex_in, dfproj_index_out, dfdiffsindex_out, var_diff_in="dist", var_idx="idx_morph_temp_rank"):
#     """
#     Append columns in dfproj_index_out, dfdiffsindex_out, recentering their indices based
#     on the index in dfdiffsindex_in that has max diff. 

#     E.g., useful for aligning in heatmap plot
#     """
#     # 3. find point of max
#     indrow_max = dfdiffsindex_in[var_diff_in].idxmax()

#     if not dfdiffsindex_in["var_idx"].unique().tolist() == [var_idx]:
#         print(dfdiffsindex_in)
#         print(var_idx)
#         assert False

#     # 4. apply that max to the other dataset
#     # - apply to both the (dist from base) and (dist between adjacent) datasets
#     dfproj_index_out[f"{var_idx}_centered"] = dfproj_index_out[var_idx] - dfdiffsindex_in.iloc[indrow_max]["idx1"]
#     dfdiffsindex_out["idx_along_morph_centered"] = dfdiffsindex_out["idx_along_morph"] - dfdiffsindex_in.iloc[indrow_max]["idx_along_morph"]

#     return dfproj_index_out, dfdiffsindex_out


# dfdiffsindex, dfproj_index = compute_pa_to_df_dist_index_using_splits(PAredu)


if __name__=="__main__":

    from pythonlib.tools.pandastools import append_col_with_grp_index

    DEBUG = False
    SAVEDIR_BASE_LOAD = "/lemur2/lucas/analyses/recordings/main/decode_moment/PSYCHO_SP";

    if DEBUG:
        SAVEDIR_BASE_SAVE = "/tmp"
        # LIST_EXPTS = [("Diego", 240515), ("Diego", 240517)]
        LIST_EXPTS = [("Diego", 240515)]
    else:
        SAVEDIR_BASE_SAVE = SAVEDIR_BASE_LOAD
        LIST_EXPTS = [("Diego", 240515), ("Diego", 240517), ("Diego", 240521), 
                                ("Diego", 240523), ("Diego", 240731), ("Diego", 240801), ("Diego", 240802), 
                                ("Pancho", 240516), ("Pancho", 240524), ("Pancho", 240521), 
                                ("Pancho", 240801), ("Pancho", 240802)]

    combine = True
    version = "dpca"
    subtrmean = False
    projtwind = (0.1, 1.2)
    n_flank_boundary = 2
    
    if False:
        scalar_or_traj = "scal"
        npcs = 8
        list_ndims_proj = [2, 3, 4]
    else:
        scalar_or_traj = "traj"
        npcs = 6
        list_ndims_proj = [8] # just use all
        agg_tbindur = 0.1
        agg_tbinslide = 0.05

    list_bregion = _REGIONS_IN_ORDER_COMBINED
    # list_bregion = ["PMv", "PMd"]

    DOPLOTS = False
    # exclude_flankers = True
    
    # Params - computing projection score
    for exclude_flankers in [True, False]:        
        for ndims_proj in list_ndims_proj:
            # ndims_proj = 3

            # if (ndims_proj, exclude_flankers) in [
            #     (2, True), (3,True)
            #     ]:
            #     continue
            ### 
            list_dfproj = []
            list_dfdiffs = []
            list_dfproj_index = []
            list_dfdiffsproj = []
            list_dfdiffsindex = []

            for animal, date in LIST_EXPTS:
                list_morphset = params_good_morphsets_no_switching(animal, date)

                for morphset in list_morphset:
                    # morphset = 7

                    for bregion in list_bregion:
                        # bregion = "PMd"

                        SAVEDIR = f"{SAVEDIR_BASE_LOAD}/{animal}-{date}-logistic-combine={combine}/statespace_euclidian/bregion={bregion}/morphset={morphset}-ver={scalar_or_traj}-{version}-subtrmean={subtrmean}-projtwind={projtwind}-npcs={npcs}"
                    
                        # Load data
                        path = f"{SAVEDIR}/Cldist.pkl"
                        print(path)
                        with open(path, "rb") as f:
                            Cldist = pickle.load(f)

                        path = f"{SAVEDIR}/PAredu.pkl"
                        print(path)
                        with open(path, "rb") as f:
                            PAredu = pickle.load(f)

                        if scalar_or_traj == "traj":
                            # Then a few cleanup things.
                            # - this reduces noise a bit.
                            PAredu = PAredu.agg_by_time_windows_binned(agg_tbindur, agg_tbinslide)

                        if ndims_proj is not None and (ndims_proj <= PAredu.X.shape[0]):
                            # X = X[:, :ndims_proj] # (trials, ndims)
                            PAredu = PAredu.slice_by_dim_indices_wrapper("chans", list(range(ndims_proj)))

                        # Dir for plotting
                        savedir = f"{SAVEDIR_BASE_SAVE}/MULT_BREGION/statespace_euclidian/{animal}-{date}-logistic-combine={combine}/statespace_euclidian/bregion={bregion}/ver={scalar_or_traj}-{version}-subtrmean={subtrmean}-projtwind={projtwind}-npcs={npcs}"
                        os.makedirs(savedir, exist_ok=True)

                        from neuralmonkey.scripts.analy_decode_moment_psychometric import analy_morphsmooth_euclidian_score
                        dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex = analy_morphsmooth_euclidian_score(PAredu, savedir, exclude_flankers, morphset, n_flank_boundary, DOPLOTS=False)

                        # if exclude_flankers:
                        #     _idxs_no_flankers = [x for x in PAredu.Xlabels["trials"]["idx_morph_temp"].unique().tolist() if x>=0 and x<=99]
                        #     PAredu = PAredu.slice_by_labels_filtdict({"idx_morph_temp":_idxs_no_flankers})

                        # ####################### Project trials onto the axis (base1 - base2)
                        # if PAredu.X.shape[2]>1:
                        #     # This is traj. Hacky solution, to avgg over time. Is ok since I prob dont care this value (proj) anyway.
                        #     X = np.mean(PAredu.X, axis=2).T # (trials, ndims)
                        # else:
                        #     X = PAredu.X.squeeze(axis=2).T # (trials, ndims)
                        # dflab = PAredu.Xlabels["trials"]

                        # # Do this in cross-validated way -- split base prims into two sets, run this 2x, then 
                        # # average over base prims.
                        # from pythonlib.tools.statstools import crossval_folds_indices

                        # inds_pool_base1 = dflab[dflab["idx_morph_temp"] == 0].index.values
                        # inds_pool_base2 = dflab[dflab["idx_morph_temp"] == 99].index.values
                        # list_inds_1, list_inds_2 = crossval_folds_indices(len(inds_pool_base1), len(inds_pool_base2), 2)

                        # # shuffle inds
                        # np.random.shuffle(inds_pool_base1)
                        # np.random.shuffle(inds_pool_base2)

                        # list_inds_1 = [inds_pool_base1[_inds] for _inds in list_inds_1]
                        # list_inds_2 = [inds_pool_base2[_inds] for _inds in list_inds_2]

                        # list_X =[]
                        # for _i, (inds_base_1, inds_base_2) in enumerate(zip(list_inds_1, list_inds_2)):
                        #     # inds_base_1 = inds_pool_base1[_inds1]
                        #     # inds_base_2 = inds_pool_base2[_inds2]
                        #     # print(inds_base_1)

                        #     # - get mean activity for base1, base2
                        #     xmean_base1 = np.mean(X[inds_base_1,:], axis=0) # (ndims,)
                        #     xmean_base2 = np.mean(X[inds_base_2,:], axis=0) # (ndims,)

                        #     # get mean projected data for each state
                        #     # xproj, fig = projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=True)
                        #     plot_color_labels = dflab["idx_morph_temp"].values
                        #     xproj, fig = projection_onto_axis_subspace(xmean_base1, xmean_base2, X, doplot=True, 
                        #                                             plot_color_labels=plot_color_labels)
                        #     savefig(fig, f"{savedir}/morphset={morphset}-projections_preprocess-iter={_i}.pdf")
                        #     # replace the train data with nan
                        #     xproj[inds_base_1] = np.nan
                        #     xproj[inds_base_2] = np.nan

                        #     list_X.append(xproj)

                        # # Get average over iterations
                        # Xproj = np.nanmean(np.stack(list_X), axis=0)

                        # dfproj = PAredu.Xlabels["trials"].copy()
                        # dfproj["x_proj"] = Xproj

                        # ####################### Get eucl distnaces between adjacent indices
                        # # Get distances between adjacent indices
                        # if True:
                        #     # Use dist_yue instead
                        #     cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp", ["epoch"], 
                        #                                                             version_distance="euclidian", return_cldist=True, get_reverse_also=False)
                        #     # get distance between 0 and 99
                        #     # res, DIST_NULL_50, DIST_NULL_95, DIST_NULL_98 = cldist.rsa_distmat_score_same_diff_by_context("idx_morph_temp", ["epoch"], None, "pts", )
                        #     # pd.DataFrame(res)
                        #     dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups(get_only_one_direction=False)
                        #     VAR_SCORE = "dist_yue_diff"
                        # else:
                        #     # use euclidian dist norm -- problem --> this is noisy.
                        #     assert False, "dist index below assumes is doing euclidian, not euclidian_unbiased"
                        #     cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp", ["epoch"], 
                        #                                                             version_distance="euclidian_unbiased", return_cldist=True, get_reverse_also=False)
                        #     # get distance between 0 and 99
                        #     dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups()
                        #     VAR_SCORE = "dist_mean"

                        # # Collect all adjacent pairs
                        # dflab = PAredu.Xlabels["trials"]
                        # list_idx_morph = sorted(dflab[(dflab["idx_morph_temp"]>=0) & (dflab["idx_morph_temp"]<=99)]["idx_morph_temp"].unique().tolist())
                        # res_diffs = []
                        # for i in range(len(list_idx_morph)-1):
                        #     tmp = dfdist[(dfdist["idx_morph_temp_1"] == list_idx_morph[i]) & (dfdist["idx_morph_temp_2"] == list_idx_morph[i+1])]
                        #     assert len(tmp)==1, f"{i}"

                        #     # Get the distance
                        #     d = tmp[VAR_SCORE].values[0]

                        #     # save
                        #     res_diffs.append({
                        #         "idx_along_morph":i,
                        #         "idx1":list_idx_morph[i],
                        #         "idx2":list_idx_morph[i+1],
                        #         "dist":d,
                        #         "animal":animal,
                        #         "date":date,
                        #         "morphset":morphset,
                        #         "bregion":bregion
                        #     })
                        # dfdiffs = pd.DataFrame(res_diffs)            


                        # ####################### EXTRA STUFF
                        # # Also figure out whether base1 and base2 are sufficiently separated to include this data
                        # # - Collect distances between adjacent morph indices.
                        # # cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, "idx_morph_temp", 
                        # #                                                                 ["epoch"], return_cldist=True, get_reverse_also=False)
                        # # get distance between 0 and 99
                        # # dfdist = cldist.rsa_distmat_score_all_pairs_of_label_groups()
                        # tmp = dfdist[(dfdist["idx_morph_temp_1"] == 0) & (dfdist["idx_morph_temp_2"] == 99)]
                        # assert len(tmp)==1
                        # dist_between_bases = tmp[VAR_SCORE].values[0]
                        # DIST_98 = tmp["DIST_98"].values[0]
                        # # dfproj["dist_between_bases"] = dist_between_bases
                        # # dfproj["dist_norm_between_bases"] = dist_between_bases/DIST_98

                        # # dfdiffs["dist_between_bases"] = dist_between_bases
                        # # dfdiffs["dist_norm_between_bases"] = dist_between_bases/DIST_98

                        # ### Base prim separation score
                        # # - ie., frac (between 0, 1), if 1 then base prim separation is greater than 100% of the other prims.
                        # _dfdist_notbase = dfdist[(~dfdist["idx_morph_temp_1"].isin([0, 99])) & (~dfdist["idx_morph_temp_2"].isin([0, 99]))]
                        # n = sum(_dfdist_notbase[VAR_SCORE]<dist_between_bases)
                        # ntot = len(_dfdist_notbase)
                        # base_prim_separation_score = n/ntot
                        # # dfproj["base_prim_separation_score"] = base_prim_separation_score
                        # # dfdiffs["base_prim_separation_score"] = base_prim_separation_score

                        # if DOPLOTS:
                        #     fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", jitter=True, alpha=0.5)
                        #     for ax in fig.axes.flatten():
                        #         ax.axhline(0, color="k", alpha=0.5)
                        #         ax.axhline(1, color="k", alpha=0.5)
                        #     savefig(fig, f"{savedir}/morphset={morphset}-catplot-1.pdf")

                        #     fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", kind="point")
                        #     for ax in fig.axes.flatten():
                        #         ax.axhline(0, color="k", alpha=0.5)
                        #         ax.axhline(1, color="k", alpha=0.5)
                        #     savefig(fig, f"{savedir}/morphset={morphset}-catplot-2.pdf")

                        #     fig = sns.catplot(data=dfproj, x="idx_morph_temp", y="x_proj", kind="violin")
                        #     for ax in fig.axes.flatten():
                        #         ax.axhline(0, color="k", alpha=0.5)
                        #         ax.axhline(1, color="k", alpha=0.5)
                        #     savefig(fig, f"{savedir}/morphset={morphset}-catplot-3.pdf")

                        # plt.close("all")

                        # ### COLLECT ALL
                        # # Rescale so that 0 <--> 99 maps onto 0 <--> 1 index, so can compare across expts
                        # idx_max = dfproj[dfproj["idx_morph_temp"] < 99]["idx_morph_temp"].max()

                        # def f(x):
                        #     if x<=idx_max:
                        #         return x
                        #     elif x>=99:
                        #         return x - 99 + idx_max + 1
                        #     else:
                        #         assert False
                        
                        # _rank_idxs_append(dfproj)
                        # dfproj["idx_morph_rescaled"] = [f(x) for x in dfproj["idx_morph_temp"]]
                        # from pythonlib.tools.listtools import rank_items
                        # # assert np.all(dfproj["idx_morph_rescaled"].tolist() == list(rank_items(dfproj["idx_morph_temp"]-1, method="dense"))), "sanity cehck."
                        # dfproj["idx_morph_rescaled"] = dfproj["idx_morph_rescaled"]/(idx_max+1) # rescale
                        # # dfproj.groupby(["idx_morph_temp", "idx_morph_rescaled"]).size()

                        # dfproj["animal"] = animal
                        # dfproj["date"] = date
                        # dfproj["morphset"] = morphset
                        # dfproj["bregion"] = bregion
                        
                        # # Track how many morphs exist
                        # nmorphs = len(dfproj[(dfproj["idx_morph_temp"]>=0) & (dfproj["idx_morph_temp"]<=99)]["idx_morph_temp"].unique())
                        # dfproj["nmorphs"] = nmorphs
                        # dfdiffs["nmorphs"] = nmorphs
                            
                        # ########################## GET projection distance between adjacent indices

                        # # Optionally remove flankers
                        # var_score = "x_proj"
                        # dfthis = dfproj
                        # # Get differenes between adjancent morph indices, projected onto the axis.
                        # dfdiffsproj = convert_dist_to_distdiff(dfthis, var_score)
                        # # dfdiffsproj["animal"] = animal
                        # # dfdiffsproj["date"] = date
                        # # dfdiffsproj["morphset"] = morphset
                        # # dfdiffsproj["bregion"] = bregion
                        # # dfdiffsproj["base_prim_separation_score"] = base_prim_separation_score
                        # # dfdiffsproj["dist_between_bases"] = dist_between_bases
                        # # dfdiffsproj["dist_norm_between_bases"] = dist_between_bases/DIST_98
                        # # dfdiffsproj["nmorphs"] = nmorphs

                        # # list_idx_morph = sorted(dfthis["idx_morph_temp"].unique().tolist())
                        # # for i in range(len(list_idx_morph)-1):
                        # #     idx1 = list_idx_morph[i]
                        # #     idx2 = list_idx_morph[i+1]
                            
                        # #     score1 = dfthis[dfthis["idx_morph_temp"] == idx1][var_score]
                        # #     score2 = dfthis[dfthis["idx_morph_temp"] == idx2][var_score]

                        # #     res_diffs_proj.append({
                        # #         # f"{var_score}-idx2-min-idx1":np.mean(score2) - np.mean(score1),
                        # #         "dist":np.mean(score2) - np.mean(score1),
                        # #         "idx_along_morph":i,
                        # #         "idx1":idx1,
                        # #         "idx2":idx2,
                        # #         "animal":animal, 
                        # #         "date":date, 
                        # #         "morphset":morphset, 
                        # #         "bregion":bregion,
                        # #         "base_prim_separation_score":base_prim_separation_score,
                        # #         "dist_between_bases":dist_between_bases,
                        # #         "dist_norm_between_bases":,
                        # #         "nmorphs":nmorphs
                        # #     })

                        # ### [Best method?] dist index
                        # if True:
                        #     # Does splits, returning 2x num rows, splitting lets you recenter indinces without overfitting.
                        #     dfdiffsindex, dfproj_index = compute_pa_to_df_dist_index_using_splits(PAredu)
                        # else:

                        #     dfproj_index = dfdist_to_dfproj_index(dfdist)
                        #     # # compute an index which is how close you are to base 1 vs. to base 2.
                        #     # # Ranging in (0,1), where 0 means is close to base 1 and far from base 2
                        #     # list_idx_morph = sorted(dfdist["idx_morph_temp_1"].unique().tolist())
                        #     # res_dist_index = []
                        #     # for idx in list_idx_morph:
                                
                        #     #     tmp = dfdist[(dfdist['idx_morph_temp_1']==0) & (dfdist['idx_morph_temp_2']==idx)]
                        #     #     assert len(tmp)==1
                        #     #     d1 = tmp["dist_mean"].values[0]
                                
                        #     #     tmp = dfdist[(dfdist['idx_morph_temp_1']==idx) & (dfdist['idx_morph_temp_2']==99)]
                        #     #     assert len(tmp)==1
                        #     #     d2 = tmp["dist_mean"].values[0]

                        #     #     dist_index = d1/(d1+d2)

                        #     #     # print(idx, d1, d2, " -- ", dist_index)
                        #     #     res_dist_index.append({
                        #     #         "dist_index":dist_index,
                        #     #         "idx_morph_temp":idx,
                        #     #         # "animal":animal, 
                        #     #         # "date":date, 
                        #     #         # "morphset":morphset, 
                        #     #         # "bregion":bregion,
                        #     #         # "base_prim_separation_score":base_prim_separation_score,
                        #     #         # "dist_between_bases":dist_between_bases,
                        #     #         # "dist_norm_between_bases":dist_between_bases/DIST_98,
                        #     #         # "nmorphs":nmorphs
                        #     #     })
                        #     # dfproj_index = pd.DataFrame(res_dist_index)

                        #     # Get diffs
                        #     dfdiffsindex = convert_dist_to_distdiff(dfproj_index, "dist_index")

                        # # Append useful stuff
                        # for _df in [dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex]:
                        #     _df["animal"] = animal
                        #     _df["date"] = date
                        #     _df["morphset"] = morphset
                        #     _df["bregion"] = bregion
                        #     _df["base_prim_separation_score"] = base_prim_separation_score
                        #     _df["dist_between_bases"] = dist_between_bases
                        #     _df["dist_norm_between_bases"] = dist_between_bases/DIST_98
                        #     _df["nmorphs"] = nmorphs



                        for _df in [dfproj, dfproj_index, dfdiffs, dfdiffsproj, dfdiffsindex]:
                            _df["animal"] = animal
                            _df["date"] = date
                            _df["morphset"] = morphset
                            _df["bregion"] = bregion

                        ############ COLLECT ALL
                        list_dfproj.append(dfproj)
                        list_dfproj_index.append(dfproj_index)

                        list_dfdiffs.append(dfdiffs)
                        list_dfdiffsproj.append(dfdiffsproj)
                        list_dfdiffsindex.append(dfdiffsindex)


            ############ PLOTS
            savedir = f"{SAVEDIR_BASE_SAVE}/MULT_BREGION/statespace_euclidian/COMBINED/USING_DIFFS_ADJACENT_INDS-{scalar_or_traj}-ndims_proj={ndims_proj}-excludeflank={exclude_flankers}"
            os.makedirs(savedir, exist_ok=True)

            from pythonlib.tools.pandastools import append_col_with_grp_index
            from pythonlib.tools.snstools import rotateLabel

            DFPROJ = pd.concat(list_dfproj).reset_index(drop=True)
            DFPROJ_INDEX = pd.concat(list_dfproj_index).reset_index(drop=True)

            DFDIFFS = pd.concat(list_dfdiffs).reset_index(drop=True)
            DFDIFFSPROJ = pd.concat(list_dfdiffsproj).reset_index(drop=True)
            DFDIFFSINDEX = pd.concat(list_dfdiffsindex).reset_index(drop=True)
            
            from neuralmonkey.scripts.analy_decode_moment_psychometric_mult import plot_with_preprocess
            plot_with_preprocess(DFPROJ, DFPROJ_INDEX, DFDIFFS, DFDIFFSPROJ, DFDIFFSINDEX, DFDIFFSCATEG, savedir)
    
            # DFDIFFSPROJ = append_col_with_grp_index(DFDIFFSPROJ, ["animal", "date", "morphset"], "ani_date_mrp")
            # DFDIFFS = append_col_with_grp_index(DFDIFFS, ["animal", "date", "morphset"], "ani_date_mrp")
            # DFPROJ = append_col_with_grp_index(DFPROJ, ["animal", "date", "morphset"], "ani_date_mrp")
            # DFPROJ_INDEX = append_col_with_grp_index(DFPROJ_INDEX, ["animal", "date", "morphset"], "ani_date_mrp")
            # DFDIFFSINDEX = append_col_with_grp_index(DFDIFFSINDEX, ["animal", "date", "morphset"], "ani_date_mrp")
            
            # DFDIFFSPROJ["dist_norm"] = DFDIFFSPROJ["dist"] * DFDIFFSPROJ["dist_between_bases"]


            # ### Label if morphset has intermediate shapes (e..g, line1-line2-line3)
            # from pythonlib.dataset.dataset_analy.psychometric_singleprims import params_has_intermediate_shape
            # cachedict = {}
            # for dfthis in [DFDIFFSPROJ, DFDIFFS, DFPROJ, DFPROJ_INDEX, DFDIFFSINDEX]:
            #     list_has_intermediate = []
            #     for i, row in dfthis.iterrows():
            #         animal = row["animal"]
            #         date = row["date"]

            #         if (animal, date) not in cachedict:
            #             cachedict[(animal, date)] = params_has_intermediate_shape(animal, date)
                    
            #         # Check if this has intermediate
            #         list_has_intermediate.append(row["morphset"] in cachedict[(animal, date)])

            #     dfthis["morphset_has_intermediate"] = list_has_intermediate

            # # all projection distance, rescale by the distance between base1 and base2, so that the aggregated data is not super noisy.
            # # - i.e., 
            # # DFPROJ["x_proj_norm"] = [row["x_proj"] * row["dist_norm_between_bases"] for i, row in DFPROJ.iterrows()]
            # # DFPROJ["x_proj_norm"] = [row["x_proj"] * row["dist_between_bases"] for i, row in DFPROJ.iterrows()]
            # # Plot results, using differences between adjacent indices

            # # SCore for how well base prims are separated
            # fig = sns.catplot(data=DFPROJ, x="ani_date_mrp", y="base_prim_separation_score", hue="bregion", kind="point", aspect=2)
            # rotateLabel(fig)
            # for ax in fig.axes.flatten():
            #     ax.axhline(0)
            # savefig(fig, f"{savedir}/base_prim_separation_score-1.pdf")

            # fig = sns.catplot(data=DFPROJ, x="ani_date_mrp", y="base_prim_separation_score", hue="nmorphs", 
            #                 col="bregion", col_wrap=6, kind="point", aspect=1.5)
            # rotateLabel(fig)
            # for ax in fig.axes.flatten():
            #     ax.axhline(0)
            # savefig(fig, f"{savedir}/base_prim_separation_score-2.pdf")
            
            # ### SAVE DATAFRAMES
            # for _df, _name in [
            #     (DFPROJ, "DFPROJ"), 
            #     (DFPROJ_INDEX, "DFPROJ_INDEX"), 
            #     (DFDIFFS, "DFDIFFS"), 
            #     (DFDIFFSPROJ, "DFDIFFSPROJ"), 
            #     (DFDIFFSINDEX, "DFDIFFSINDEX"),
            #     ]:
            #     path = f"{savedir}/{_name}.pkl"
            #     pd.to_pickle(_df, path)

            # print("Saving at ...: ", savedir)
            # for thresh_separation in [0.2, 0.4, 0.6, 0.8, 0.]:
            #     sdir = f"{savedir}/thresh-base_prim_separation_score={thresh_separation}"
            #     os.makedirs(sdir, exist_ok=True)
            #     _plot(DFPROJ, DFPROJ_INDEX, DFDIFFS, DFDIFFSPROJ, DFDIFFSINDEX, thresh_separation, sdir)

            