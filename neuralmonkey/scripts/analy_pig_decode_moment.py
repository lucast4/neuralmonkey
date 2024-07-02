"""
Train and test decoders, using diff event wwindows, and plot summaries.
Mainly goal is evaluating different decoder params, for momentbymoment deocoding.



"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys

if __name__=="__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])
    combine_areas = bool(int(sys.argv[3]))

    use_final_params = True
    # animal = "Diego"
    # date = 240619
    if use_final_params:
        list_fr_normalization_method = ["across_time_bins"] # Slightly better
        list_preprocess_pca = [False] # Much better
        list_include_null_data = [False] # Better (but slightly more noise)
        list_decoder_method_index = [2] # Slightly better
    else:
        # Grid search
        list_fr_normalization_method = ["across_time_bins", "each_time_bin"]
        list_preprocess_pca = [False, True]
        list_include_null_data = [False, True]
        list_decoder_method_index = [1,2]

    for fr_normalization_method in list_fr_normalization_method:
        for preprocess_pca in list_preprocess_pca:
            for include_null_data in list_include_null_data:
                for decoder_method_index in list_decoder_method_index:
                    # fr_normalization_method = "each_time_bin"
                    # fr_normalization_method = "across_time_bins"
                    # preprocess_pca = False

                    #####################################
                    DFallpa = load_handsaved_wrapper(animal=animal, date=date, version="trial", combine_areas=combine_areas)

                    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PIG/pipeline_train_test_scalar_score/{animal}-{date}-norm={fr_normalization_method}-pca={preprocess_pca}-inclnull={include_null_data}-decoder={decoder_method_index}"
                    os.makedirs(SAVEDIR, exist_ok=True)

                    #################### PREPROCESSING
                    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels
                    dfpa_concatbregion_preprocess_clean_bad_channels(DFallpa, PLOT=False)

                    from neuralmonkey.classes.population_mult import dfpa_concat_normalize_fr_split_multbregion
                    dfpa_concat_normalize_fr_split_multbregion(DFallpa, fr_normalization_method=fr_normalization_method)

                    if preprocess_pca:
                        from neuralmonkey.classes.population_mult import dfpa_concat_pca_split_multbregion
                        dfpa_concat_pca_split_multbregion(DFallpa, npcs_keep_force=15)
                        DFallpa["pa"] = DFallpa["pa_pca"]

                    list_bregion = DFallpa["bregion"].unique().tolist()

                    RES = []
                    for train_dataset, test_dataset in [
                        ("sp_samp", "sp_samp"),
                        ("sp_samp", "pig_samp"),
                        ("sp_samp", "sp_prestroke"),
                        ("pig_samp", "pig_samp"),
                        ("pig_samp", "sp_samp"),
                        ("pig_samp", "sp_prestroke"),
                        ("sp_prestroke", "sp_samp"),
                        ("sp_prestroke", "pig_samp"),
                        ("sp_prestroke", "pig_prestroke"),
                        ("pig_prestroke", "sp_samp"),
                        ("pig_prestroke", "pig_samp"),
                        ("pig_prestroke", "sp_prestroke"),
                        ]:

                        print(train_dataset, test_dataset)

                        for bregion in list_bregion:
                            # bregion = "PMd"

                            ### Train params

                            # Train params
                            var_train = "seqc_0_shape"
                            n_min_per_var = 3
                            # include_null_data = True

                            # Test params
                            var_test = "seqc_0_shape"

                            def get_dataset_params(dataset_name):
                                if dataset_name == "sp_samp":
                                    event = "03_samp"
                                    twind = (0.1, 1.2)
                                    list_twind = [(-0.9, -0.1), twind]
                                    filterdict = {"FEAT_num_strokes_task":[1]}
                                elif dataset_name == "pig_samp":
                                    event = "03_samp"
                                    twind = (0.1, 1.2)
                                    list_twind = [(-0.9, -0.1), twind]
                                    filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8]}
                                elif dataset_name == "sp_prestroke":
                                    event = "06_on_strokeidx_0"
                                    twind = (-0.6, 0)
                                    list_twind = [twind, (0, 0.6)]
                                    filterdict = {"FEAT_num_strokes_task":[1]}
                                elif dataset_name == "pig_prestroke":
                                    event = "06_on_strokeidx_0"
                                    twind = (-0.6, 0)
                                    list_twind = [twind, (0, 0.6)]
                                    filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8]}
                                else:
                                    print(dataset_name)
                                    assert False

                                return event, twind, filterdict, list_twind

                            event_train, twind_train, filterdict_train, _ = get_dataset_params(train_dataset)

                            ### Test params
                            event_test, _, filterdict_test, list_twind_test = get_dataset_params(test_dataset)

                            from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score
                            savedir = f"{SAVEDIR}/train={train_dataset}-test={test_dataset}-bregion={bregion}"
                            os.makedirs(savedir, exist_ok=True)

                            dfscores, _, _, _ = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, twind_train, filterdict_train,
                                                                var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                                include_null_data=include_null_data, decoder_method_index=decoder_method_index)
                            
                            RES.append({
                                "dfscores":dfscores,
                                "train_dataset":train_dataset,
                                "test_dataset":test_dataset,
                                "bregion":bregion
                            })

                    ################ Summarize
                    for res in RES:
                        dfscores = res["dfscores"]
                        dfscores["train_dataset"] = res["train_dataset"]
                        dfscores["test_dataset"] = res["test_dataset"]
                        dfscores["bregion"] = res["bregion"]

                    DFSCORES = pd.concat([res["dfscores"] for res in RES]).reset_index(drop=True)

                    from pythonlib.tools.pandastools import summarize_featurediff
                    from pythonlib.tools.pandastools import append_col_with_grp_index

                    DFSUMMARY, _, _, _, _ = summarize_featurediff(DFSCORES, "same_class", [False, True], ["score"], 
                                        ["train_dataset", "test_dataset", "bregion", "decoder_class_good", "decoder_class_is_in_pa", "pa_class", "pa_class_is_in_decoder", "pa_idx", "trialcode", "twind"])

                    DFSUMMARY = append_col_with_grp_index(DFSUMMARY, ["train_dataset", "test_dataset"], "train_test_dataset")
                    a = DFSUMMARY["decoder_class_good"]==True
                    b = DFSUMMARY["decoder_class_is_in_pa"]==True
                    c = DFSUMMARY["pa_class_is_in_decoder"]==True
                    DFSUMMARY = DFSUMMARY[a & b & c].reset_index(drop=True)
                    
                    from pythonlib.tools.pandastools import plot_subplots_heatmap
                    from pythonlib.tools.pandastools import savefig

                    # plot_subplots_heatmap(DFSUMMARY, "train_test_dataset", "bregion", "score-TrueminFalse", "twind", True, True)

                    savedir = f"{SAVEDIR}/ALL_TRAIN_TEST"
                    os.makedirs(savedir, exist_ok=True)

                    from pythonlib.tools.pandastools import plot_subplots_heatmap

                    for share_zlim in [False, True]:
                        for annotate_heatmap in [False, True]:
                            fig, axes = plot_subplots_heatmap(DFSUMMARY, "bregion", "twind", "score-TrueminFalse", "train_test_dataset", 
                                                            True, share_zlim=share_zlim, annotate_heatmap=annotate_heatmap)
                            savefig(fig, f"{savedir}/score-TrueminFalse--share_zlim={share_zlim}--annotate_heatmap={annotate_heatmap}")

                    # Save all results
                    path = f"{SAVEDIR}/DFSCORES.pkl"
                    pd.to_pickle(DFSCORES, path)            