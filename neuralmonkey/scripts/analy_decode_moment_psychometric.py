

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

def analy_psychoprim_statespace_euclidian(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, list_morphset):
    """
    Plot euclidian distances, looking for categorical represntation, continuous morphts.
    """
    from neuralmonkey.analyses.state_space_good import trajgood_plot_colorby_splotby_WRAPPER, trajgood_plot_colorby_splotby_scalar_WRAPPER

    list_bregion = DFallpa["bregion"].unique().tolist()

    # Extract data for this morphset, wiht labels updated
    for bregion in list_bregion:
        for morphset in list_morphset:
            ############### PREPARE DATASET
            PA = extract_single_pa(DFallpa, bregion, None, "trial", "03_samp")

            # Given morphset, assign new column which is the trial's role in that morphset.
            dflab = PA.Xlabels["trials"]
            dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

            # keep just trials that are in this morphset
            idx_exist = list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"]))
            filtdict = {
                "idx_morph_temp":idx_exist,
            }
            PA = PA.slice_by_labels_filtdict(filtdict)

            ############### DIM REDUCTION
            # Train -- PC space.
            # -- Params
            # NPCS_KEEP = 6
            # twind_train = (0.1, 1.0)

            # -- Fixed params
            superv_dpca_var = "idx_morph_temp"
            superv_dpca_vars_group = None
            tbin_dur = 0.1
            pca_tbin_slice = 0.01

            for raw_subtract_mean_each_timepoint in [True, False]:
                for scalar_or_traj in ["scal", "traj"]:
                    if scalar_or_traj=="traj":
                        NPCS_KEEP = 6
                        # list_twind = [(-0.5, 1.2), twind_train]
                        # list_twind = [(0.1, 1.2), (0.1, 0.6), (0.6, 1.2)]
                        list_twind = [(0.1, 1.2), (0.6, 1.2)]
                    else:
                        NPCS_KEEP = 8
                        # list_twind = [(0.1, 1.2), (0.1, 0.6), (0.6, 1.2)]
                        list_twind = [(0.1, 1.2), (0.6, 1.2)]

                    for proj_twind in list_twind:
                        # for version in ["pca", "dpca"]:
                        for version in ["dpca"]:
                            
                            savedir = f"{SAVEDIR}/statespace_euclidian/bregion={bregion}/morphset={morphset}-ver={scalar_or_traj}-{version}-subtrmean={raw_subtract_mean_each_timepoint}-projtwind={proj_twind}-npcs={NPCS_KEEP}"
                            os.makedirs(savedir, exist_ok=True)

                            ### DIM REDUCTION
                            savedirpca = f"{savedir}/pca_construction"
                            os.makedirs(savedirpca, exist_ok=True)
                    
                            Xredu, PAredu = PA.dataextract_dimred_wrapper(scalar_or_traj, version, savedirpca, 
                                    proj_twind, tbin_dur, pca_tbin_slice, NPCS_KEEP = NPCS_KEEP,
                                    dpca_var = superv_dpca_var, dpca_vars_group = superv_dpca_vars_group, dpca_proj_twind = None, 
                                    raw_subtract_mean_each_timepoint=raw_subtract_mean_each_timepoint)
                            

                            ### STATE SPACE PLOTS
                            dflab = PAredu.Xlabels["trials"]
                            times = PAredu.Times

                            if scalar_or_traj == "traj":
                                ### Trajectory plots 
                                time_bin_size = 0.05
                                list_var_color_var_subplot = [
                                    [superv_dpca_var, None],
                                ]
                                LIST_DIMS = [(0,1), (2,3)]
                                for var_color, var_subplot in list_var_color_var_subplot:
                                    trajgood_plot_colorby_splotby_WRAPPER(Xredu, times, dflab, var_color, savedir,
                                                                vars_subplot=var_subplot, list_dims=LIST_DIMS, time_bin_size=time_bin_size)
                                plt.close("all")

                            elif scalar_or_traj == "scal":
                                ### Plot scalars
                                list_var_color_var_subplot = [
                                    [superv_dpca_var, None],
                                ]
                                LIST_DIMS = [(0,1), (2,3)]
                                for var_color, var_subplot in list_var_color_var_subplot:

                                    trajgood_plot_colorby_splotby_scalar_WRAPPER(Xredu, dflab, var_color, savedir,
                                                                                    vars_subplot=var_subplot, list_dims=LIST_DIMS,
                                                                                    overlay_mean_orig=True
                                                                                    )
                            else:
                                assert False
                            
                            ############################## EUCLIDIAN DISTANCE
                            # Euclidian distance between adjacent steps in morphset
                            from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single
                            var_effect = "idx_morph_temp"
                            vars_others = None
                            PLOT_HEATMAPS = False
                            Cldist, _ = euclidian_distance_compute_trajectories_single(PAredu, var_effect, vars_others, PLOT_HEATMAPS=PLOT_HEATMAPS, 
                                                                        savedir_heatmaps=savedir, get_reverse_also=False, return_cldist=True,
                                                                        compute_same_diff_scores=False)
                            dfdists = Cldist.rsa_dataextract_with_labels_as_flattened_df(plot_heat=False, exclude_diagonal=True)

                            import seaborn as sns
                            from pythonlib.tools.pandastools import plot_subplots_heatmap

                            fig, _ =plot_subplots_heatmap(dfdists, "idx_morph_temp_row", "idx_morph_temp_col", "dist", None)
                            savefig(fig, f"{savedir}/heatmap_distances.pdf")

                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", hue="idx_morph_temp_col", kind="point")
                            savefig(fig, f"{savedir}/catplot-1.pdf")

                            # Test hypothesis -- logistic curve
                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", col="idx_morph_temp_col", kind="point")
                            savefig(fig, f"{savedir}/catplot-2.pdf")

                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", col="idx_morph_temp_col", jitter=True, alpha=0.15)
                            savefig(fig, f"{savedir}/catplot-3.pdf")

                            fig = sns.catplot(data=dfdists, x="idx_morph_temp_row", y="dist", col="idx_morph_temp_col", kind="violin")
                            savefig(fig, f"{savedir}/catplot-4.pdf")

                            # For each index, get its distance to its adjacent indices
                            dfdists = Cldist.rsa_dataextract_with_labels_as_flattened_df(exclude_diagonal=True, plot_heat=False)
                            # Middle states -- evidence for trial by trial variation? i.e.,
                            # Scatterplot -- if similar to base1, then is diff from base 2?

                            from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
                            # subplot = trial kind 
                            # x and y are "decoders"
                            # var_datapt = same as "trialcode"
                            _, fig = plot_45scatter_means_flexible_grouping(dfdists, "idx_morph_temp_col", 0, 99, "idx_morph_temp_row", 
                                                                "dist", "idx_row", plot_text=False, plot_error_bars=True, SIZE=4.5, 
                                                                shareaxes=True)
                            savefig(fig, f"{savedir}/scatter-1.pdf")

                            plt.close("all")

                            # SAVE:
                            import pickle
                            with open(f"{savedir}/PAredu.pkl", "wb") as f:
                                pickle.dump(PAredu, f)
                            with open(f"{savedir}/Cldist.pkl", "wb") as f:
                                pickle.dump(Cldist, f)
                            
def prune_dfscores_good_morphset(dfscores, morphset_get, animal, date):
    """
    """

    print("morphset_get:", morphset_get)
    if morphset_get == "good_ones":
        # Hand modified
        if (animal, date) == ("Diego", 240515):
            # Angle rotation
            morphsets_ignore = [2] # Did most with two strokes...
        if (animal, date) == ("Diego", 240523):
            # THis is a garbage morphset, is not actually morphing.
            morphsets_ignore = [0]
        elif (animal, date) == ("Pancho", 240521):
            morphsets_ignore = [0] # one base prim is garbage
        elif (animal, date) == ("Pancho", 240524):
            morphsets_ignore = [4] # doesnt actually vaciallte across tirals
        else:
            morphsets_ignore = []
    elif morphset_get is None:
        # Get all morphsets
        morphsets_ignore = []
    elif isinstance(morphset_get, int):
        # get just this one morphset (exclude the others)
        morphsets_ignore = [ms for ms in DF_TCRES["run_morphset"].unique().tolist() if ms!=morphset_get]
    else:
        print(morphset_get)
        assert False

    # Prune dfscores
    morphsets_keep = [ms for ms in dfscores["run_morph_set_idx"].unique().tolist() if ms not in morphsets_ignore]
    print("morphsets_ignore: ", morphsets_ignore)
    print("morphsets_exist: ", sorted(dfscores["run_morph_set_idx"].unique()))
    print("morphsets_keep: ", morphsets_keep)
    dfscores = dfscores[dfscores["run_morph_set_idx"].isin(morphsets_keep)].reset_index(drop=True)
    print("morphsets_exist (after pruen): ", sorted(dfscores["run_morph_set_idx"].unique()))

    return dfscores


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    # WHICH_PLOTS = sys.argv[3] # 01, 11, ...

    classifier_ver = "logistic"
    # classifier_ver = "ensemble"

    which_level = "trial"
    # LIST_COMBINE_AREAS = [False, True]
    LIST_COMBINE_AREAS = [True, False]
    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    PLOT_EACH_IDX = False

    # PLOTS_DO = [1, 2, 0]
    PLOTS_DO = [3]

    for COMBINE_AREAS in LIST_COMBINE_AREAS:
        
        SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PSYCHO_SP/{animal}-{date}-{classifier_ver}-combine={COMBINE_AREAS}"
        os.makedirs(SAVEDIR, exist_ok=True)

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

        from neuralmonkey.analyses.decode_moment import analy_psychoprim_prepare_beh_dataset
        savedir = f"{SAVEDIR}/morphsets_drawings"
        os.makedirs(savedir, exist_ok=True)
        DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, \
            map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = analy_psychoprim_prepare_beh_dataset(animal, date, savedir)
        
        # Prune neural data to keep only those trialcodes that are in DSmorphsets
        list_trialcode = DSmorphsets.Dat["trialcode"].unique().tolist()
        DFallpa["pa"] = [pa.slice_by_labels("trials", "trialcode", list_trialcode) for pa in DFallpa["pa"].values]

        if 0 in PLOTS_DO:
            from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp_better
            savedir_base = f"{SAVEDIR}/better_seperate_for_each_morph_idx"
            os.makedirs(savedir_base, exist_ok=True)
            analy_psychoprim_score_postsamp_better(DFallpa, DSmorphsets, 
                                                map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                                map_tcmorphset_to_info,
                                                savedir_base, animal, date,
                                                PLOT_EACH_IDX=PLOT_EACH_IDX)
        
        ###########################
        if 1 in PLOTS_DO:
            from neuralmonkey.analyses.decode_moment import analy_psychoprim_score_postsamp
            analy_psychoprim_score_postsamp(DFallpa, DSmorphsets, 
                                                map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                                map_tcmorphset_to_info,
                                                SAVEDIR,
                                                animal=animal, date=date)
                
        if 2 in PLOTS_DO:
            list_morphset = DSmorphsets.Dat["morph_set_idx"].unique().tolist()
            analy_psychoprim_statespace_euclidian(DFallpa, SAVEDIR, map_tcmorphset_to_idxmorph, list_morphset)

        if 3 in PLOTS_DO:
            ### sliceTCA (testing this out..)
            
            from neuralmonkey.scripts.analy_slicetca_script import run
            if True:
                list_bregion = DFallpa["bregion"].unique().tolist()
            else:
                # list_bregion = ["PMv", "PMd", "SMA", "vlPFC", "M1"]
                list_bregion = ["PMv", "PMd", "M1"]

            list_morphset = DSmorphsets.Dat["morph_set_idx"].unique().tolist()
            # list_twind_analy = [(-0.1, 1.)]

            list_event_twindanaly = [
                ("03_samp", (-0.05, 1.1)),
                ("06_on_strokeidx_0", (-0.1, 0.4)),
            ]

            # Extract data for this morphset, wiht labels updated
            for event, twind_analy in list_event_twindanaly:
                for bregion in list_bregion:
                    for morphset in list_morphset:
                        ############### PREPARE DATASET
                        PA = extract_single_pa(DFallpa, bregion, None, "trial", event)

                        # Given morphset, assign new column which is the trial's role in that morphset.
                        dflab = PA.Xlabels["trials"]
                        dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

                        # keep just trials that are in this morphset
                        idx_exist = list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"]))
                        filtdict = {
                            "idx_morph_temp":idx_exist,
                        }
                        PA = PA.slice_by_labels_filtdict(filtdict)

                        savedir = f"{SAVEDIR}/sliceTCA/event={event}-twind={twind_analy}/bregion={bregion}/morphset={morphset}"
                        os.makedirs(savedir, exist_ok=True)

                        list_var_color = ["idx_morph_temp", "seqc_0_loc", "seqc_0_shape"]
                        run(PA, savedir, twind_analy, list_var_color)