"""
Scores for novel prims, including consistency, stregnth of represnetation, etc.
Assumes that novelprims have "novelprims" in their names.


"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys

from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from pythonlib.tools.plottools import savefig
from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score_with_splits, pipeline_train_test_scalar_score_split_gridloc


def analy_novelprim_prepare_dataset(DFallpa, animal, date, SAVEDIR_BASE):

    # Cleanup, some days should group gridloc
    if (animal, date) == ("Diego", 240522):
        # Mistake, slight offset in locations, actually there are just two diff locaitons
        map_gridloc_old_to_new = {}
        map_gridloc_old_to_new[(-1, -1)] = (0,0)
        map_gridloc_old_to_new[(2, 1)] = (1,1)
        map_gridloc_old_to_new[(0,0)] = (0,0)
        map_gridloc_old_to_new[(1,1)] = (1,1)

        for pa in DFallpa["pa"]:
            dflab = pa.Xlabels["trials"]
            dflab["seqc_0_loc"] = [map_gridloc_old_to_new[loc] for loc in dflab["seqc_0_loc"]]

    if (animal, date) == ("Pancho", 240523):
        # a shape was given diff labels at two diff locaitons. find it and give it a single name.
        dflab = DFallpa["pa"].values[0].Xlabels["trials"]

        shapes_that_have_only_one_loc = []
        for sh in dflab[dflab["shape_is_novel_all"]==False]["seqc_0_shape"].unique():
            nloc = len(dflab[dflab["seqc_0_shape"] == sh]["seqc_0_loc"].unique())
            print(sh, " --- ", nloc)
            if nloc == 1:
                shapes_that_have_only_one_loc.append(sh)

        assert len(shapes_that_have_only_one_loc)==2
        map_shape_old_to_new = {}
        map_shape_old_to_new[shapes_that_have_only_one_loc[0]] = shapes_that_have_only_one_loc[1]
        map_shape_old_to_new[shapes_that_have_only_one_loc[1]] = shapes_that_have_only_one_loc[1]

        # add al the other shapes
        for sh in dflab["seqc_0_shape"].unique().tolist():
            if sh not in shapes_that_have_only_one_loc:
                map_shape_old_to_new[sh] = sh
        
        for pa in DFallpa["pa"]:
            dflab = pa.Xlabels["trials"]
            dflab["seqc_0_shape"] = [map_shape_old_to_new[sh] for sh in dflab["seqc_0_shape"]]

    # Rename prims with prefix indicating if is novel or learned. 

    # def F(x):
    #     if "novelprim" in x:
    #         return f"N|{x}"
    #     else:
    #         return f"L|{x}"
    def F(sh, novel):
        if novel:
            return f"N|{sh}"
        else:
            return f"L|{sh}"

    for pa in DFallpa["pa"].values:
        dflab = pa.Xlabels["trials"]
        # dflab["seqc_0_shape_pref"] = [F(sh) for sh in dflab["seqc_0_shape"]]
        dflab["seqc_0_shape_pref"] = [F(row["seqc_0_shape"], row["shape_is_novel_all"]) for i, row in dflab.iterrows()]

        # sanity
        if sum(dflab["shape_is_novel_all"]) == 0:
            print(dflab["seqc_0_shape_pref"].unique())
            assert False

    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    fig = grouping_plot_n_samples_conjunction_heatmap(dflab, "seqc_0_shape_pref", "shape_is_novel_all")
    savefig(fig, f"{SAVEDIR_BASE}/counts_novel_shapes.pdf")

def analy_novelprims_dfscores_condition(dfscores, dflab):
    """
    """
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    if "bregion" in dfscores.columns:
        assert dfscores.groupby(["bregion", "trialcode", "decoder_class"]).size().max() == 1, "mistake, maybe diff train-test splits applied to same trial? this is doublecoutning."
    else:
        assert dfscores.groupby(["trialcode", "decoder_class"]).size().max() == 1, "mistake, maybe diff train-test splits applied to same trial? this is doublecoutning."

    # Assign variables
    map_tc_to_novel = {row["trialcode"]:row["shape_is_novel_all"] for i, row in dflab.iterrows()}
    map_shape_to_novel = {row["seqc_0_shape_pref"]:row["shape_is_novel_all"] for i, row in dflab.iterrows()}

    # dfscores["shape_is_novel_all"] = [map_tc_to_novel[tc] for tc in dfscores["trialcode"]]
    # [dflab["trialcode"] == tc for tc in dfscores["trialcode"]]
    dfscores["decoder_shape_novel"] = [map_shape_to_novel[sh] for sh in dfscores["decoder_class"]]
    dfscores["pa_shape_novel"] = [map_shape_to_novel[sh] for sh in dfscores["pa_class"]]
    dfscores["pa_decoder_shape_novel"] = [(map_shape_to_novel[row["pa_class"]], map_shape_to_novel[row["decoder_class"]]) for i, row in dfscores.iterrows()]

    return map_tc_to_novel, map_shape_to_novel

def analy_novelprim_score_postsamp(dfscores, Dc, savedir):
    """
    Summary scores for a single bregion, of variosu "questions" related to novel rims.
    PARAMS:
    - dflab, labels across trials.
    """
    import seaborn as sns
    from pythonlib.tools.snstools import rotateLabel
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping


    # Clean up -- just those novel prims that have consistent behavior.
    if False:
        # This is hard coded for Diego 240516. 
        # Use the code that generated these, more genearlly.
        novel_shapes_good = [
            "novelprim290480141634-125-4.1-0.1",
            "novelprim302322405581-125-4.6-0.1",
            "novelprim307096595122-125-3.3-1.5",
            "novelprim495912188195-130-4.6-2.2",
            "novelprim625505027794-130-5.3-1.5",
            "novelprim65385831858-75-5.3-0.1",
            "novelprim675463345033-91-5.3-0.1",
            "novelprim677917180851-125-4.5-0.7",
            "novelprim902433564644-125-3.8-1.4",
            "novelprim965178957006-125-3.4-0.9",
            "novelprim987866512255-90-4.2-0.1",
        ]
        novel_shapes_good = [f"N|{sh}" for sh in novel_shapes_good]

        for sh in novel_shapes_good:
            # assert f"N|{sh}" in dfscores["pa_class"].tolist()
            assert f"{sh}" in dfscores["pa_class"].tolist()
        print("Good! ALl shapes you inputed were correctly found to exist in dataset")

        a = (dfscores["pa_shape_novel"]==True) & (dfscores["pa_class"].isin(novel_shapes_good))
        b = (dfscores["pa_shape_novel"]==False)
        dfscores = dfscores[a | b].reset_index(drop=True)

    ############# QUESTIONS
    ### (1) Novel prims --> worse accuracy (i.e., specificity to their own prim)?
    # - AND -
    ### (2) Novel prims --> better generalization to other novel prims? (i.e., diff subspace for learned vs. novel)
    yvar = "score"
    fig = sns.catplot(data=dfscores, x="decoder_class", y=yvar, hue="same_class", kind="bar", col="pa_decoder_shape_novel", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-decoder_class-1.pdf")

    fig = sns.catplot(data=dfscores, x="pa_class", y=yvar, hue="same_class", kind="bar", col="pa_decoder_shape_novel", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-pa_class-2.pdf")

    fig = sns.catplot(data=dfscores, x="pa_decoder_shape_novel", y=yvar, hue="same_class", kind="bar", aspect=1.5)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/catplot-pa_decoder_shape_novel-3.pdf")

    plt.close("all")

    # A single metric that scores accuracy 
    for version in ["decoder_class", "pa_class"]:
        _, dfsummary, yvar = Dc.scalar_score_compute_metric(dfscores, version=version)
        fig = sns.catplot(data=dfsummary, x=version, y=yvar, col="twind", aspect=1.5, kind="bar")
        rotateLabel(fig)
        savefig(fig, f"{savedir}/summarycatplot-{version}.pdf")

    # Scatterplot comparing score for same vs. diff decoder.
    var_score = "score"
    for var_datapt in ["decoder_class", "pa_class", "trialcode"]:
        _, fig = plot_45scatter_means_flexible_grouping(dfscores, "same_class", True, False, "pa_decoder_shape_novel", 
                                            var_score, var_datapt, False, shareaxes=True, alpha=0.1)
        savefig(fig, f"{savedir}/scatter-var_datapt={var_datapt}.pdf")

    plt.close("all")

    ### (2) Novel prims --> more trial by trial variation?
    from pythonlib.tools.statstools import statsmodel_ols
    # - for each pa class, calculate variance and mean of each decoder across trials.
    # dfscores["decoder_shape_novel"] = [map_shape_to_novel[sh] for sh in dfscores["decoder_class"]]
    # dfscores["pa_shape_novel"] = [map_shape_to_novel[sh] for sh in dfscores["pa_class"]]
    # dfscores["pa_decoder_shape_novel"] = [(map_shape_to_novel[row["pa_class"]], map_shape_to_novel[row["decoder_class"]]) for i, row in dfscores.iterrows()]
    dfcases_mean = dfscores.groupby(["pa_class", "decoder_class", "decoder_shape_novel", "pa_shape_novel", "pa_decoder_shape_novel", "same_class"])[var_score].mean().reset_index()
    dfcases_std = dfscores.groupby(["pa_class", "decoder_class", "decoder_shape_novel", "pa_shape_novel", "pa_decoder_shape_novel", "same_class"])[var_score].std().reset_index()
    dfcases_mean["score_mean"] = dfcases_mean["score"]
    dfcases_mean["score_std"] = dfcases_std["score"]
    dfcases_mean = dfcases_mean.drop("score", axis=1)
    dfcases_mean_orig = dfcases_mean

    # Onlye keep cases of the decoder matching pa
    for pa_decoder_same_class in ["ignore", True]:
        if pa_decoder_same_class==True:
            dfcases_mean = dfcases_mean_orig[dfcases_mean_orig["same_class"] == True].reset_index(drop=True)
        elif pa_decoder_same_class=="ignore":
            dfcases_mean = dfcases_mean_orig.copy()
        else:
            assert False

        for i, list_pa_decoder_shape_novel in enumerate([
            [(False, True), (True, False), (False, False), (True, True)],
            [(False, False), (True, True)],
            [(False, False)],
            [(True, True)]]):

            # 1. Overlay all cases
            dfcases_mean_this = dfcases_mean[dfcases_mean["pa_decoder_shape_novel"].isin(list_pa_decoder_shape_novel)]
            fig, ax = plt.subplots(1,1, figsize=(7,7))
            sns.scatterplot(data=dfcases_mean_this, x="score_mean", y="score_std", hue="pa_decoder_shape_novel", alpha=0.35, ax=ax)
            ax.set_title(f"list_pa_decoder_shape_novel = {list_pa_decoder_shape_novel}")
            savefig(fig, f"{savedir}/std_vs_mean-pa_decoder_same_class={pa_decoder_same_class}-data_subset={i}-alldata.pdf")

        # 2. Plot cases separately, and each fitting regression.
        list_pa_decoder_shape_novel = dfcases_mean["pa_decoder_shape_novel"].unique().tolist()
        fig, axes = plt.subplots(2,2, figsize=(10, 10), sharex=True, sharey=True)

        for ax, pa_decoder_shape_novel in zip(axes.flatten(), list_pa_decoder_shape_novel):
            ax.set_title(f"pa_decoder_shape_novel={pa_decoder_shape_novel}")
            dfthis = dfcases_mean[dfcases_mean["pa_decoder_shape_novel"] == pa_decoder_shape_novel].reset_index(drop=True)
            sns.regplot(data=dfthis, x="score_mean", y="score_std", ax=ax)
            results = statsmodel_ols(dfthis["score_mean"], dfthis["score_std"], overlay_on_this_ax=ax,
                        overlay_x=0, overlay_y = 0.1)
            ax.axhline(0)
            ax.axvline(0)
        savefig(fig, f"{savedir}/std_vs_mean-pa_decoder_same_class={pa_decoder_same_class}-regression.pdf")

    plt.close("all")


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    version = "split_gridloc"

    # LIST_COMBINE_AREAS = [True, False]
    # fr_normalization_method = "across_time_bins" # Slightly better

    COMBINE_AREAS = animal=="Diego"
    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    SAVEDIR_BASE = f"/lemur2/lucas/analyses/recordings/main/decode_moment/NOVEL_PRIMS/{animal}-{date}-combine={COMBINE_AREAS}"
    os.makedirs(SAVEDIR_BASE, exist_ok=True)

    #####################################
    # Method 2 - Combine two dfallpa
    DFallpa = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=COMBINE_AREAS)

    #################### PREPROCESSING
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
    dfpa_concatbregion_preprocess_wrapper(DFallpa)


    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import dfallpa_preprocess_condition
    shape_var_suff = "shape"
    loc_var_suff = "loc"
    dfallpa_preprocess_condition(DFallpa, shape_var_suff, loc_var_suff)

    analy_novelprim_prepare_dataset(DFallpa, animal, date, SAVEDIR_BASE)
    

    if version == "split_gridloc":
        # Split by gridloc
        list_loc = None
        auto_prune_locations = True
        list_downsample_trials = [False, True]
        PLOT_TEST = True
    elif version == "train_test_split":
        list_downsample_trials = [False, True]
        # Train/test splits
        do_train_splits_nsplits=10
        score_user_test_data = False
        PLOT_TEST_SPLIT = False
        DO_TRAIN_TEST_SPLIT=True
        PLOT_TEST_CONCATTED = True

    else:
        assert False

    # Just to get these, since they were left out...
    list_downsample_trials = [True]

    ################ SCORE DATA     
    # PARAMS
    for downsample_trials in list_downsample_trials:
        if downsample_trials:
            # So that the lowest N doesnt pull all other categories down.
            n_min_per_var = 10
        else:
            n_min_per_var = 7
        
        if version == "split_gridloc":
            n_min_per_var = 6 # to allow gen across loc.

        for TWIND_TEST in [(0.05, 1.2), (0.05, 0.6), (0.6, 1.2)]:
            do_upsample_balance=True
            PLOT_DECODER = False

            TWIND_TRAIN = (0.05, 1.2)

            # Subtrract baseline?
            subtract_baseline=False
            subtract_baseline_twind=None
            include_null_data = False
            prune_labels_exist_in_train_and_test = True

            # - Train params
            event_train = "03_samp"
            twind_train = TWIND_TRAIN
            var_train = "seqc_0_shape_pref"
            filterdict_train = None

            # - Test params
            var_test = "seqc_0_shape_pref"
            event_test = "03_samp"
            which_level_test = "trial"
            filterdict_test = None
            # list_twind_test = [(-0.8, -0.05), TWIND_TEST]
            list_twind_test = [TWIND_TEST]

            # Other params
            SAVEDIR = f"{SAVEDIR_BASE}/downsample_trials={downsample_trials}-TWIND_TEST={TWIND_TEST}-version={version}-nmin={n_min_per_var}"

            list_bregion = DFallpa["bregion"].unique().tolist()
            for bregion in list_bregion:
                savedir = f"{SAVEDIR}/{bregion}/decoder_training"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)

                if version == "split_gridloc":
                    # n_min_per_var = 4 # to allow gen across loc.
                    # n_min_per_var = 5 # to allow gen across loc.
                    do_train_splits_nsplits = 6
                    dfscores, decoders, list_pa_train, list_pa_test = pipeline_train_test_scalar_score_split_gridloc(list_loc, savedir,
                                                                                                                    DFallpa, 
                                                                                    bregion, var_train, event_train, 
                                                                                    twind_train, filterdict_train,
                                                        var_test, event_test, list_twind_test, filterdict_test, 
                                                        include_null_data=include_null_data, 
                                                        prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, 
                                                        PLOT=PLOT_DECODER, PLOT_TEST=PLOT_TEST,
                                                        which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                        subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                        do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials,
                                                        auto_prune_locations=auto_prune_locations, do_train_splits_nsplits=do_train_splits_nsplits)
                    Dc = decoders[0]

                else:
                    if not DO_TRAIN_TEST_SPLIT:
                        dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, 
                                                                                        twind_train, filterdict_train,
                                                            var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                            include_null_data=include_null_data, decoder_method_index=None,
                                                            prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, PLOT=PLOT_DECODER,
                                                            which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                            subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                            do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials)
                    else:
                        dfscores, dfscores_usertest, dfscores_both, decoders, trainsets, PAtest = pipeline_train_test_scalar_score_with_splits(DFallpa, 
                                                                                        bregion, var_train, event_train, 
                                                                                        twind_train, filterdict_train,
                                                            var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                            include_null_data=include_null_data, decoder_method_index=None,
                                                            prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, 
                                                            PLOT_TRAIN=PLOT_DECODER, PLOT_TEST_SPLIT=PLOT_TEST_SPLIT, PLOT_TEST_CONCATTED=PLOT_TEST_CONCATTED,
                                                            which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                            subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                            do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials,
                                                            do_train_splits_nsplits=do_train_splits_nsplits, 
                                                            score_user_test_data=score_user_test_data)
                        Dc = decoders[0]

                # Condition dfscores
                dflab = DFallpa["pa"].values[0].Xlabels["trials"]
                analy_novelprims_dfscores_condition(dfscores, dflab)

                ### Save this bregion
                import pickle
                savedir = f"{SAVEDIR}/{bregion}"
                with open(f"{savedir}/dfscores.pkl", "wb") as f:
                    pickle.dump(dfscores, f)
                with open(f"{savedir}/Dc.pkl", "wb") as f:
                    pickle.dump(Dc, f)
                with open(f"{savedir}/dflab.pkl", "wb") as f:
                    pickle.dump(dflab, f)

                ######################## PLOTS
                from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp
                savedir = f"{SAVEDIR}/{bregion}/PLOTS"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)
                analy_novelprim_score_postsamp(dfscores, Dc, savedir)