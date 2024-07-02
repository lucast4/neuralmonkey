"""
One job -- plots of decoder strenght for Shape sequence experiments (TI),
iterating over many params.

Both timecourse and scalar plots.

And then saves the dataframes

See notebook: 240617_pig_moment...

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
from neuralmonkey.classes.population_mult import extract_single_pa
from neuralmonkey.analyses.decode_moment import train_decoder_helper
import sys

def _syntax_concrete_string_to_indices(s):
    """
    PARAMS:
    - s, like 
        s = "0|0|0|1|1|0"
    RETURNS:
    - s --> list of int indices where 1's are
    """
    ys = []
    for i, _tmp in enumerate(s[0::2]):
        if _tmp=="1":
            ys.append(i)
    return ys
    
from neuralmonkey.analyses.decode_moment import train_decoder_helper, pipeline_train_test_scalar_score
import seaborn as sns


def prepare_beh_dataset(animal, date):
    # (1) Load session object
    from neuralmonkey.classes.session import load_mult_session_helper
    spikes_version = "kilosort_if_exists"
    MS = load_mult_session_helper(date, animal, MINIMAL_LOADING=True, spikes_version=spikes_version) 

    # Get ground truth shape sequence
    D = MS.datasetbeh_extract().copy()

    # For each trial, get shape sequence
    D.preprocessGood(params=["remove_baseline"])
    D.grammarparses_syntax_concrete_append_column(PRINT=False)

    # exclude last index, it is not sahpe...
    D.Dat["syntax_concrete"] = [x[:-1] for x in D.Dat["syntax_concrete"]]
    D.Dat["syntax_concrete"].value_counts()

    shape_sequence = D.grammarparses_rules_shape_AnBmCk_get_shape_order()

    if "success_binary_quick" not in D.Dat.columns:
        D.grammarparses_successbinary_score_wrapper()
    
    # map from trialcode to syntax
    list_trialcode = D.Dat["trialcode"]
    map_tc_to_syntaxconcrete = {}
    for i, row in D.Dat.iterrows():
        tc = row["trialcode"]
        if row["epoch"] in ["base", "baseline"]:
            map_tc_to_syntaxconcrete[tc] = None
        else:
            tmp = D.Dat[D.Dat["trialcode"]==tc]
            assert len(tmp)==1
            map_tc_to_syntaxconcrete[tc] = tmp["syntax_concrete"].values[0]

    # Map from location to shape on this trial
    map_trialcode_loc_to_shape = {}
    for ind in range(len(D.Dat)):
        taskconfig_shploc = D.Dat.iloc[ind]["taskconfig_shploc"]
        trialcode = D.Dat.iloc[ind]["trialcode"]

        for sh, loc in taskconfig_shploc:
            map_trialcode_loc_to_shape[(trialcode, loc)] = sh

    return MS, D, shape_sequence, map_tc_to_syntaxconcrete, map_trialcode_loc_to_shape


def get_dataset_params(dataset_name):
    which_level = "trial"

    SP_HOLD_DUR = 1.2
    # PIG_HOLD_DUR = 1.2
    PIG_HOLD_DUR = 1.6

    if dataset_name == "sp_samp":
        event = "03_samp"
        twind = (0.1, SP_HOLD_DUR)
        list_twind = [(-0.9, -0.1), twind]
        filterdict = {"FEAT_num_strokes_task":[1]}

    elif dataset_name == "pig_samp":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR)
        list_twind = [(-0.9, -0.1), twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8]}

    elif dataset_name == "pig_samp_post":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR)
        list_twind = [twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8]}

    elif dataset_name == "pig_samp_post_early":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR/2)
        list_twind = [twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8]}

    elif dataset_name == "pig_samp_post_late":
        event = "03_samp"
        twind = (PIG_HOLD_DUR/2, PIG_HOLD_DUR)
        list_twind = [twind]
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

    elif dataset_name == "pig_gaps_0_1":
        # gap between firs and 2nd storkes-- i.e,, preostroke stroke 1
        event = "00_stroke"
        twind = (-0.6, 0)
        list_twind = [twind]
        filterdict = {"stroke_index":[1]}
        which_level = "stroke"

    else:
        print(dataset_name)
        assert False

    return event, twind, filterdict, list_twind, which_level

def probs_timecourse_normalize(Dc, PAprobs, norm_method, map_tc_to_syntaxconcrete, twind_base = None):
    """
    Normalize time course of probabilties.
    RETURNS:
    - PAprobsNorm, copy of PAprobs, norming the X
    """
    if twind_base is None:
        twind_base = (-0.45, -0.05)

    times = PAprobs.Times

    if norm_method == "minus_base_twind":
        # Method 1 - subtract twind before image onset.
        
        twind_base = (-0.45, -0.05)
        inds_base = (times>=twind_base[0]) &  (times<=twind_base[1])

        PAprobs_base = PAprobs.slice_by_dim_values_wrapper("times", twind_base, time_keep_only_within_window=True).agg_wrapper("times")
        PAprobsNorm = PAprobs.copy()
        PAprobsNorm.X = PAprobs.X - PAprobs_base.X

    elif norm_method=="minus_not_visible_timecourse":
        # Better, subtract mean for when is not visible, subtract the mean timecourse of decode prob.

        # Get baseline timecourse for each decoder label
        list_prob_vec = []
        n_classes = len(PAprobs.Chans)
        for idx_decoder in range(n_classes):
            decoder_class = PAprobs.Chans[idx_decoder]
            # decoder_class = "Lcentered-4-2-0"

            # Find all the trials where the decoder shape was not drawn
            if Dc.VarDecode == "seqc_0_shape":
                tmp = [decoder_class not in taskconfig for taskconfig in PAprobs.Xlabels["trials"]["taskconfig_shp"]]
                tmp2 = [map_tc_to_syntaxconcrete[tc][idx_decoder]==0 for tc in PAprobs.Xlabels["trials"]["trialcode"]]
                assert tmp==tmp2, "bug, I expect this to be."
            elif Dc.VarDecode == "seqc_0_loc":
                tmp = [decoder_class not in taskconfig for taskconfig in PAprobs.Xlabels["trials"]["taskconfig_loc"]]
            else:
                print(Dc.VarDecode)
                assert False
            inds_base = [int(i) for i in np.argwhere(tmp)]
            
            # Get average timecoures for those trials
            pa = PAprobs.slice_by_dim_indices_wrapper("trials", inds_base)
            prob_vec = np.mean(pa.X[idx_decoder, :, :], axis=0) # (ntimes, ) prob vector

            list_prob_vec.append(prob_vec)

        prob_mat_base = np.stack(list_prob_vec, axis=0)[:, None, :] # (nclass, 1, ntimes)

        if False: # This doesnt make sense, since pre image onset monkey doesnt know image. This is identical to just subntracting
            # mean pre-image across all trials.
            # Option 2: get single scalar mean for each class, 
            # Get single scalar mean for each class
            assert len(times) == prob_mat_base.shape[2]
            inds_base = (times>=twind_base[0]) &  (times<=twind_base[1])
            prob_mat_base = np.mean(prob_mat_base[:, :, inds_base], axis=2)[:, :, None] # (nclasses, 1, 1) 

        # Subtract this from raw probs
        PAprobsNorm = PAprobs.copy()
        PAprobsNorm.X = PAprobs.X - prob_mat_base
    elif norm_method == "minus_not_visible_and_base":
        # Do both, in order.
        PAprobsNorm = probs_timecourse_normalize(Dc, PAprobs, "minus_not_visible_timecourse", map_tc_to_syntaxconcrete, twind_base)
        PAprobsNorm = probs_timecourse_normalize(Dc, PAprobsNorm, "minus_base_twind", map_tc_to_syntaxconcrete, twind_base)

    else:
        print(norm_method)
        assert False

    return PAprobsNorm


if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper

    animal = sys.argv[1]
    date = int(sys.argv[2])
    WHICH_PLOTS = sys.argv[3] # 01, 11, ...

    SAVEDIR_BASE = f"/lemur2/lucas/analyses/recordings/main/decode_moment/PIG/{animal}-{date}-ipynb_params_2"
    which_level = "trial"
    question = "PIG_BASE_trial"
    # events_keep = ["03_samp", "06_on_strokeidx_0"]
    events_keep = None
    LIST_COMBINE_AREAS = [False, True]
    fr_normalization_method = "across_time_bins" # Slightly better

    LIST_BREGION_IGNORE = ["FP", "FP_p", "FP_a"]

    for COMBINE_AREAS in LIST_COMBINE_AREAS:

        #####################################
        if True:
            # First try to load. If fails, then extract
            DFallpa = load_handsaved_wrapper(animal=animal, date=date, version=which_level, combine_areas=COMBINE_AREAS,
                                            return_none_if_no_exist=True)
            if DFallpa is None:
                DFallpa = extract_dfallpa_helper(animal, date, question, COMBINE_AREAS, events_keep=events_keep)
        else:
            # Alwyas reextract, since lots of varialbes, and longer twind.
            DFallpa = extract_dfallpa_helper(animal, date, question, COMBINE_AREAS, events_keep=events_keep)

        #################### PREPROCESSING
        from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels
        dfpa_concatbregion_preprocess_clean_bad_channels(DFallpa, PLOT=False)

        from neuralmonkey.classes.population_mult import dfpa_concat_normalize_fr_split_multbregion
        dfpa_concat_normalize_fr_split_multbregion(DFallpa, fr_normalization_method=fr_normalization_method)


        shape_var_suff = "shape"
        # shape_var_suff = "shapesemgrp"

        for i, row in DFallpa.iterrows():
            if row["which_level"] == "trial":
                PA = row["pa"]
                dflab = PA.Xlabels["trials"]

                # (1) shapes drawn (tuple of shapes)
                list_shapes_drawn = []
                list_locs_drawn = []
                for ind in range(len(dflab)):
                    shapes_drawn = tuple([dflab.iloc[ind][f"seqc_{i}_{shape_var_suff}"] for i in range(6) if dflab.iloc[ind][f"seqc_{i}_{shape_var_suff}"] != "IGN"])
                    list_shapes_drawn.append(shapes_drawn)

                    locs_drawn = tuple([dflab.iloc[ind][f"seqc_{i}_loc"] for i in range(6) if dflab.iloc[ind][f"seqc_{i}_loc"][0] != "IGN"])
                    list_locs_drawn.append(locs_drawn)

                    assert len(locs_drawn)==len(shapes_drawn)

                dflab["shapes_drawn"] = list_shapes_drawn            
                dflab["locs_drawn"] = list_locs_drawn    


        # (1) Load session object
        MS, D, shape_sequence, map_tc_to_syntaxconcrete, map_trialcode_loc_to_shape = prepare_beh_dataset(animal, date)        

        ############################################################### 
        ######################### PLOTS (TIMECOURSES)
        DEBUG = False
        if WHICH_PLOTS[0]=="1":
            list_bregion = DFallpa["bregion"].unique().tolist()
            for bregion in list_bregion:
                if bregion in LIST_BREGION_IGNORE:
                    continue
                for var_train, var_test, labels_in_order_keep in [
                    ("seqc_0_shape", "seqc_0_shape", shape_sequence),
                    ("seqc_0_loc", "seqc_0_loc", None),
                    ]:
                    include_null_data = False
                    twind_test = [-0.6, 1.8]

                    # --------------------
                    # var_train = "seqc_0_shape"
                    # var_test = "seqc_0_shape"
                    # labels_in_order_keep = shape_sequence

                    # var_train = "seqc_0_loc"
                    # var_test = "seqc_0_loc"
                    # labels_in_order_keep = None

                    # -------------------

                    train_dataset = "sp_samp"
                    n_min_per_var = 3

                    # Test params
                    # - post-samp
                    test_dataset = "pig_samp_post"

                    event_train, twind_train, filterdict_train, _, _ = get_dataset_params(train_dataset)
                    event_test, _, filterdict_test, _, which_level_test = get_dataset_params(test_dataset)
                    list_twind_test = [twind_test]

                    SAVEDIR = f"{SAVEDIR_BASE}/syntax_TI_plots/timecourse/var_train={var_train}-var_test={var_test}-combine={COMBINE_AREAS}/{bregion}"
                    os.makedirs(SAVEDIR, exist_ok=True)

                    from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score

                    sdir = f"{SAVEDIR}/decoder_train"
                    os.makedirs(sdir, exist_ok=True)
                    dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, twind_train, 
                                                                                    filterdict_train,
                                                        var_test, event_test, list_twind_test, filterdict_test, sdir,
                                                        include_null_data=include_null_data, decoder_method_index=None,
                                                        prune_labels_exist_in_train_and_test=False, PLOT=False,
                                                        which_level_test=which_level_test)

                    ### Get timecourses
                    # Get all test trials, just correct ones
                    dflab = PAtest.Xlabels["trials"]
                    tcs = D.Dat[D.Dat["success_binary_quick"]==True]["trialcode"]
                    b = dflab["trialcode"].isin(tcs)
                    indtrials = dflab[b].index.tolist()

                    if DEBUG:
                        indtrials = list(range(12))
                    PAprobsOrig, probs_mat_all, times, labels = Dc.timeseries_score_wrapper(PAtest, twind_test, indtrials, labels_in_order_keep=labels_in_order_keep)
                    
                    for NORM_METHOD in [None, "minus_base_twind", "minus_not_visible_and_base"]:
                        savedir = f"{SAVEDIR}/norm={NORM_METHOD}"
                        os.makedirs(savedir, exist_ok=True)
                        print("Saving to ...", savedir)

                        # Also get normalized.
                        if NORM_METHOD is not None:
                            from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import probs_timecourse_normalize
                            PAprobs = probs_timecourse_normalize(Dc, PAprobsOrig, NORM_METHOD, map_tc_to_syntaxconcrete)
                            YLIMS = (-0.25, 0.4)
                        else:
                            PAprobs = PAprobsOrig.copy()
                            YLIMS = (0, 0.6)

                        # Reset params
                        dflab = PAprobs.Xlabels["trials"]

                        # Update dflab with syntax stuff
                        dflab["syntax_concrete"] = [map_tc_to_syntaxconcrete[tc] for tc in dflab["trialcode"]]
                        dflab["syntax_concrete_indices"] = [tuple(np.argwhere(syntax_concrete).squeeze().tolist()) for syntax_concrete in dflab["syntax_concrete"]]
                        dflab["syntax_concrete_idx1"] = [sci[0] for sci in dflab["syntax_concrete_indices"]]
                        dflab["syntax_concrete_idxlast"] = [sci[-1] for sci in dflab["syntax_concrete_indices"]]

                        dflab["syntax_rank_dist_0_1"] = [sci[1] - sci[0] for sci in dflab["syntax_concrete_indices"]]
                        # easy vs. hard.
                        def F(x):
                            return (x["syntax_concrete"][0]==1, x["syntax_concrete"][-1]==1, x["syntax_rank_dist_0_1"]>1)

                        from pythonlib.tools.pandastools import applyFunctionToAllRows
                        dflab = applyFunctionToAllRows(dflab, F, "sytx_onset_offset_separated")
                        PAprobs.Xlabels["trials"] = dflab

                        # Save it
                        import pickle
                        path = f"{savedir}/PAprobs.pkl"
                        with open(path, "wb") as f:
                            pickle.dump(PAprobs, f)

                        ##### RUN ANALY
                        ##### Plot
                        LIST_PARAMS = []
                        LIST_SAVESUFF = []

                        list_title_filtdict = [
                            ("all", None),
                        ]
                        savesuffix = "all"
                        LIST_PARAMS.append(list_title_filtdict)
                        LIST_SAVESUFF.append(savesuffix)


                        list_title_filtdict = [
                            ("syntax_concrete_idx1=0", {"syntax_concrete_idx1":[0]}),
                            ("syntax_concrete_idx1>0", {"syntax_concrete_idx1":list(range(1, 10))}),
                        ]
                        savesuffix = "syntax_concrete_idx1"
                        LIST_PARAMS.append(list_title_filtdict)
                        LIST_SAVESUFF.append(savesuffix)


                        list_title_filtdict = [
                            ("syntax_concrete_idx1=0", {"syntax_concrete_idx1":[0]}),
                            ("syntax_concrete_idx1=1", {"syntax_concrete_idx1":[1]}),
                            ("syntax_concrete_idx1=2", {"syntax_concrete_idx1":[2]}),
                            ("syntax_concrete_idx1>2", {"syntax_concrete_idx1":list(range(3, 10))}),
                        ]
                        savesuffix = "syntax_concrete_idx1-2"
                        LIST_PARAMS.append(list_title_filtdict)
                        LIST_SAVESUFF.append(savesuffix)


                        list_title_filtdict = []
                        for a in [False, True]:
                            for b in [False, True]:
                                for c in [False, True]:
                                    list_title_filtdict. append(
                                        ((a, b, c), {"sytx_onset_offset_separated":[(a, b, c)]})
                                    )
                        savesuffix = "sytx_onset_offset_separated-1"
                        LIST_PARAMS.append(list_title_filtdict)
                        LIST_SAVESUFF.append(savesuffix)


                        list_title_filtdict = []
                        for a in [False, True]:
                            for b in [False, True]:
                                list_title_filtdict. append(
                                    ((a, b, "*"), {"sytx_onset_offset_separated":[(a, b, False), (a, b, True)]})
                                )
                                list_title_filtdict = []
                        savesuffix = "sytx_onset_offset_separated-2"
                        LIST_PARAMS.append(list_title_filtdict)
                        LIST_SAVESUFF.append(savesuffix)


                        for a in [False, True]:
                            list_title_filtdict. append(
                                ((a, "*", "*"), {"sytx_onset_offset_separated":[(a, False, False), (a, False, True), (a, True, False), (a, True, True)]})
                            )
                        savesuffix = "sytx_onset_offset_separated-3"
                        LIST_PARAMS.append(list_title_filtdict)
                        LIST_SAVESUFF.append(savesuffix)

                                
                        from pythonlib.tools.plottools import savefig
                        
                        for ylims in [None, YLIMS]:
                            list_n_strokes = [2,3]
                            SIZE = 4
                            for list_title_filtdict, savesuffix in zip(LIST_PARAMS, LIST_SAVESUFF):
                                ncols = len(list_title_filtdict)
                                nrows = len(list_n_strokes)
                                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)

                                ### FILTER
                                ct = 0
                                for nstrokes in [2,3]:
                                    for title, filtdict in list_title_filtdict:
                                        ax = axes.flatten()[ct]
                                        # ax.set_title(title)
                                        ax.set_title(f"nstrokes={nstrokes}--{title}")

                                        ct+=1

                                        filtdict_this = {"FEAT_num_strokes_task":[nstrokes], "FEAT_num_strokes_beh":[nstrokes]}
                                        if filtdict is not None:
                                            for k, v in filtdict.items():
                                                filtdict_this[k] = v

                                        pathis = PAprobs.slice_by_labels_filtdict(filtdict_this)
                                        if len(pathis.Trials)==0:
                                            print("SKIPPING!! this filtdict led to all data lost:")
                                            print(filtdict_this)
                                            continue

                                        Dc.timeseries_plot_by_shape_drawn_order(pathis, ax, ylims=ylims)

                                savefig(fig, f"{savedir}/timecourse_mean-subplot={savesuffix}-ylims={ylims}.pdf")
                                plt.close("all")


        ############################################################### 
        ######################### PLOTS (SCALAR SCORES)
        if WHICH_PLOTS[1] == "1":
            # for test_dataset in ["pig_samp_post", "pig_samp_post_early", "pig_samp_post_late", "pig_gaps_0_1"]:
            for test_dataset in ["pig_samp_post", "pig_samp_post_early", "pig_samp_post_late"]:
                for n_min_per_var in [3, 10]:
                    # Pipeline to train and test

                    include_null_data = False

                    # Train params
                    train_dataset = "sp_samp"
                    # train_dataset = "sp_prestroke"
                    var_train = "seqc_0_shape"

                    # Test params
                    # - post-samp
                    # test_dataset = "pig_samp_post_early"
                    var_test = "seqc_0_shape"

                    # - gaps (0-1)
                    # test_dataset = "pig_gaps_0_1"
                    # var_test = "shape"

                    # Subtrract baseline?
                    subtract_baseline=True
                    subtract_baseline_twind=(-0.45, -0.05)

                    SAVEDIR = f"{SAVEDIR_BASE}/syntax_TI_plots/scalar/train_data={train_dataset}-test_data={test_dataset}-combine_{COMBINE_AREAS}-include_null={include_null_data}-nmin={n_min_per_var}"
                    
                    list_bregion = DFallpa["bregion"].unique().tolist()
                    for bregion in list_bregion:
                        if bregion in LIST_BREGION_IGNORE:
                            continue

                        # Other params
                        savedir = f"{SAVEDIR}/decoder_training/{bregion}"
                        os.makedirs(savedir, exist_ok=True)
                        PLOT = False

                        event_train, twind_train, filterdict_train, _, _ = get_dataset_params(train_dataset)
                        event_test, _, filterdict_test, list_twind_test, which_level_test = get_dataset_params(test_dataset)

                        dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, twind_train, filterdict_train,
                                                            var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                            include_null_data=include_null_data, decoder_method_index=None,
                                                            prune_labels_exist_in_train_and_test=False, PLOT=PLOT,
                                                            which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                            subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind)

                        # Keep only decoder class that is in shape sequence
                        dfscores = dfscores[dfscores["decoder_class"].isin(shape_sequence)].reset_index(drop=True)
                        # make sure all test trials' labels are in decoder
                        assert np.all(dfscores["pa_class_is_in_decoder"]==True)

                        # Append useful information about the shape sequence on each trial
                        # (1) Get dflab
                        dflab = PAtest.Xlabels["trials"]
                        if which_level_test == "stroke":
                            # Hacky, update dflab so that it has the trial data too. Makes a copy of dflab, appending columns from trialcode.
                            from pythonlib.tools.pandastools import slice_by_row_label, merge_subset_indices_prioritizing_second
                            # event_test, _, filterdict_test, list_twind_test, which_level_test
                            PAtrial = extract_single_pa(DFallpa, bregion, None, "trial", "03_samp")
                            list_tc = dflab["trialcode"].tolist()
                            _dflab_trial_toappend = slice_by_row_label(PAtrial.Xlabels["trials"], "trialcode", list_tc)
                            dflab = merge_subset_indices_prioritizing_second(dflab, _dflab_trial_toappend, "trialcode")    

                        # map from trialcode to syntax
                        list_trialcode = dflab["trialcode"]
                        map_tc_to_syntaxconcrete = {}
                        for i, row in dflab.iterrows():
                            tc = row["trialcode"]
                            if row["epoch"] in ["base", "baseline"]:
                                map_tc_to_syntaxconcrete[tc] = None
                            else:
                                tmp = D.Dat[D.Dat["trialcode"]==tc]
                                assert len(tmp)==1
                                map_tc_to_syntaxconcrete[tc] = tmp["syntax_concrete"].values[0]

                        list_decoder_class_idx_in_shapes_drawn = []
                        list_decoder_class_idx_in_syntax = []
                        list_decoder_class_was_drawn = []
                        list_decoder_class_was_first_drawn = []
                        list_syntax_concrete = []
                        list_syntax_role = []
                        list_rank_dist_0_1_correct = []

                        for _i, row in dfscores.iterrows():

                            decoder_class = row["decoder_class"]
                            pa_idx = row["pa_idx"]
                            trialcode = row["trialcode"]
                            epoch = row["epoch"]

                            shapes_drawn = dflab.iloc[pa_idx]["shapes_drawn"]
                            decoder_class_idx_in_syntax = shape_sequence.index(decoder_class)

                            if decoder_class in shapes_drawn:
                                decoder_class_idx_in_shapes_drawn = shapes_drawn.index(decoder_class)
                            else:
                                decoder_class_idx_in_shapes_drawn = -1

                            syntax_concrete = map_tc_to_syntaxconcrete[trialcode]
                            syntax_concrete_indices = tuple(np.argwhere(syntax_concrete).squeeze().tolist())
                            if len(syntax_concrete_indices)>1:
                                rank_dist_0_1_correct = syntax_concrete_indices[1] - syntax_concrete_indices[0]
                            else:
                                rank_dist_0_1_correct = -1

                            if epoch in ["base", "baseline"]:
                                syntax_role = None
                            else:
                                if syntax_concrete[decoder_class_idx_in_syntax] == 1:
                                    syntax_role = "visible"
                                else:
                                    a = syntax_concrete[decoder_class_idx_in_syntax] == 0
                                    b = sum(syntax_concrete[:decoder_class_idx_in_syntax])>0
                                    c = sum(syntax_concrete[decoder_class_idx_in_syntax+1:])>0
                                    d = (decoder_class_idx_in_syntax > syntax_concrete_indices[0]) & (decoder_class_idx_in_syntax < syntax_concrete_indices[1])
                                    e = decoder_class_idx_in_syntax == syntax_concrete_indices[0]-1

                                    if a & b & c & d:
                                        syntax_role = "within_idx_0_1"
                                    elif a & b & c & ~d:
                                        syntax_role = "within_idx_others"
                                    elif e:
                                        syntax_role = "outside_preidx0"
                                    else:
                                        syntax_role = "outside"

                            # Get rank distance between first and second shapes
                            # if len(shapes_drawn)
                            # rank_distance_0_1 = shape_sequence.index(shapes_drawn[1]) - shape_sequence.index(shapes_drawn[0])
                            
                            row["decoder_class_idx_in_syntax"] = decoder_class_idx_in_syntax

                            list_decoder_class_idx_in_shapes_drawn.append(decoder_class_idx_in_shapes_drawn)
                            list_decoder_class_idx_in_syntax.append(decoder_class_idx_in_syntax)
                            list_decoder_class_was_drawn.append(decoder_class in shapes_drawn)
                            list_decoder_class_was_first_drawn.append(decoder_class == shapes_drawn[0])
                            list_syntax_concrete.append(syntax_concrete)
                            list_syntax_role.append(syntax_role)
                            list_rank_dist_0_1_correct.append(rank_dist_0_1_correct)
                            
                        dfscores["decoder_class_idx_in_shapes_drawn"] = list_decoder_class_idx_in_shapes_drawn
                        dfscores["decoder_class_idx_in_syntax"] = list_decoder_class_idx_in_syntax
                        dfscores["decoder_class_was_drawn"] = list_decoder_class_was_drawn
                        dfscores["decoder_class_was_first_drawn"] = list_decoder_class_was_first_drawn
                        dfscores["syntax_concrete"] = list_syntax_concrete
                        dfscores["syntax_role"] = list_syntax_role
                        dfscores["rank_dist_0_1_correct"] = list_rank_dist_0_1_correct

                        dfscores["syntax_concrete_indices"] = [tuple(np.argwhere(syntax_concrete).squeeze().tolist()) for syntax_concrete in dfscores["syntax_concrete"]]
                        dfscores["syntax_concrete_idx1"] = [sci[0] for sci in dfscores["syntax_concrete_indices"]]
                        dfscores["syntax_concrete_idxlast"] = [sci[-1] for sci in dfscores["syntax_concrete_indices"]]
                        dfscores["syntax_concrete_idx1_drawn"] = [sc[0]==1 for sc in dfscores["syntax_concrete"]]

                        from pythonlib.tools.pandastools import append_col_with_grp_index
                        dfscores = append_col_with_grp_index(dfscores, ["syntax_concrete_idx1", "rank_dist_0_1_correct"], "sytx_idx1_rank")

                        # More easily semantic
                        def F(row):
                            if row["decoder_class_idx_in_shapes_drawn"] == -1:
                                tmp = row["syntax_role"]
                            else:
                                if row["decoder_class_idx_in_shapes_drawn"]>1:
                                    # stroke 2+, this is ignored by him
                                    tmp = (row["decoder_class_idx_in_shapes_drawn"], "all")
                                elif row["syntax_concrete_indices"][0]>0 and row["syntax_concrete_indices"][1]+1 < len(row["syntax_concrete"]):
                                    # 0tha nd 1th stroke, hard
                                    tmp = (row["decoder_class_idx_in_shapes_drawn"], "hard")
                                else:
                                    # 0tha nd 1th stroke, easys
                                    tmp = (row["decoder_class_idx_in_shapes_drawn"], "easy")
                            return tmp

                        dfscores["syntax_role_v2"] = [F(row) for i, row in dfscores.iterrows()]

                        # Normalize decode by subtracting mean within each decoder class
                        if False: # Normalize by subtracting  mean for each decode class
                            from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping_within
                            datamod_normalize_row_after_grouping_within(dfscores, "decoder_class", "score")
                        else: # Subtract when the ddeocde class is not visible
                            from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping_return_same_len_df
                            dfscores["baseline_norm"] = [sr!="visible" for sr in dfscores["syntax_role"]]
                            dfscores, _, _ = datamod_normalize_row_after_grouping_return_same_len_df(dfscores, 
                                                                                                    "baseline_norm", 
                                                                                                    ["decoder_class_idx_in_syntax"], 
                                                                                                    "score", True, True)

                        # (1) keep only successful trials.
                        trialcodes_success = D.Dat[D.Dat["success_binary_quick"]==True]["trialcode"].tolist()
                        print(len(dfscores))
                        dfscores_success = dfscores[dfscores["trialcode"].isin(trialcodes_success)].reset_index(drop=True)
                        print(len(dfscores_success))

                        from pythonlib.tools.pandastools import stringify_values
                        dfscores_str = stringify_values(dfscores)
                        dfscores_str_success = stringify_values(dfscores_success)


                        # Save the dataframe
                        pd.to_pickle(dfscores, f"{SAVEDIR}/{bregion}-dfscores.pkl")
                        pd.to_pickle(dfscores_success, f"{SAVEDIR}/{bregion}-dfscores_success.pkl")

                        ####################################### PLOTS
                        from pythonlib.tools.pandastools import plot_subplots_heatmap
                        from pythonlib.tools.snstools import rotateLabel


                        from pythonlib.tools.plottools import savefig

                        list_df_ver = ["all", "success"]
                        for df_ver in list_df_ver:
                            if df_ver == "all":
                                dfthis = dfscores_str
                            elif df_ver == "success":
                                dfthis = dfscores_str_success
                            else:
                                assert False
                            row_values = sorted(dfthis["decoder_class_idx_in_syntax"].unique())[::-1]
                            col_values = sorted(dfthis["syntax_concrete"].unique())

                            savedir = f"{SAVEDIR}/{bregion}-df_ver={df_ver}"
                            os.makedirs(savedir, exist_ok=True)

                            for var_score_meth in [0, 1, 2]:

                                if var_score_meth==0:
                                    var_score = "score"
                                    norm_method = None
                                    zlims = [0, 0.5]
                                    diverge = False
                                elif var_score_meth==1:
                                    var_score = "score_norm"
                                    norm_method = None
                                    zlims = [-0.3, 0.3]
                                    diverge = True
                                elif var_score_meth==2:
                                    var_score = "score_min_base"
                                    norm_method = None
                                    zlims = [-0.3, 0.3]
                                    diverge = True
                                else:
                                    assert False
                                
                                fig, axes = plot_subplots_heatmap(dfthis, "decoder_class_idx_in_syntax", "syntax_concrete", var_score, "epoch",
                                                    annotate_heatmap=False, norm_method=norm_method, 
                                                    row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                                for ax in axes.flatten():
                                    for x, cat in enumerate(col_values):
                                        ys = _syntax_concrete_string_to_indices(cat)
                                        # print(cat, " -- ", ys, " -- ", x)
                                        ax.plot([x+0.5 for _ in range(len(ys))], [len(row_values) - 1 - y+0.5 for y in ys], "ok", mfc="none")
                                savefig(fig, f"{savedir}/heatmap-decode_vs_syntconcrete-var_score={var_score}.pdf")

                                for var_subplot in ["syntax_role", "syntax_role_v2", "rank_dist_0_1_correct", "sytx_idx1_rank", "syntax_concrete_idx1", "syntax_concrete_idxlast"]:
                                    fig, axes = plot_subplots_heatmap(dfthis, "decoder_class_idx_in_syntax", "syntax_concrete", var_score, var_subplot,
                                                        annotate_heatmap=False, norm_method=norm_method, 
                                                        row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                                    for ax in axes.flatten():
                                        for x, cat in enumerate(col_values):
                                            ys = _syntax_concrete_string_to_indices(cat)
                                            # print(cat, " -- ", [len(col_values) - 1 - y+0.5 for y in ys], " -- ", [x+0.5 for _ in range(len(ys))])
                                            ax.plot([x+0.5 for _ in range(len(ys))], [len(row_values) - 1 - y+0.5 for y in ys], "ok", mfc="none")
                                    savefig(fig, f"{savedir}/heatmap-decode_vs_syntconcrete-var_score={var_score}-subplot_{var_subplot}.pdf")

                                    plt.close("all")

                                # # Effect, as function of index

                                fig = sns.catplot(data=dfthis, x = "decoder_class_idx_in_syntax", y=var_score, col="sytx_idx1_rank", hue="decoder_class_was_drawn", col_wrap=6, kind="point", errorbar=("ci", 68))
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-sytx_idx1_rank-var_score={var_score}-1.pdf")


                                fig = sns.catplot(data=dfthis, x = "decoder_class_idx_in_shapes_drawn", y=var_score, col="sytx_idx1_rank", hue="decoder_class_was_drawn", col_wrap=6, kind="point", errorbar=("ci", 68))
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-sytx_idx1_rank-var_score={var_score}-2.pdf")
                                plt.close("all")


                                    
                                fig, axes = plot_subplots_heatmap(dfthis, "syntax_role_v2", "decoder_class_idx_in_syntax", var_score, "epoch", diverge=diverge, ZLIMS=zlims)
                                savefig(fig, f"{savedir}/heatmap-syntax_role_v2-var_score={var_score}.pdf")

                                from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

                                path = f"{savedir}/counts-syntax_role_v2.pdf"
                                _, _ = extract_with_levels_of_conjunction_vars_helper(dfthis, "syntax_role_v2", ("decoder_class_idx_in_syntax",), 3, path, 3)

                                plt.close("all")


                                order = sorted(dfthis["syntax_role_v2"].unique())

                                fig = sns.catplot(data=dfthis, x = "syntax_role_v2", y=var_score, kind="bar", errorbar=("ci", 68), aspect=1,
                                                order=order)
                                rotateLabel(fig)
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-syntax_role_v2-var_score={var_score}-1.pdf")

                                fig = sns.catplot(data=dfthis, x = "syntax_role_v2", y=var_score, hue="decoder_class_idx_in_syntax", kind="point", errorbar=("ci", 68), aspect=1,
                                                order=order)
                                rotateLabel(fig)
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-syntax_role_v2-var_score={var_score}-2.pdf")
                                plt.close("all")

                                order = sorted(dfthis["syntax_role"].unique())

                                fig = sns.catplot(data=dfthis, x = "syntax_role", y=var_score, kind="bar", errorbar=("ci", 68), aspect=1,
                                                order=order)
                                rotateLabel(fig)
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-syntax_role-var_score={var_score}-1.pdf")

                                fig = sns.catplot(data=dfthis, x = "syntax_role", y=var_score, hue="decoder_class_idx_in_syntax", kind="point", errorbar=("ci", 68), aspect=1,
                                                order=order)
                                rotateLabel(fig)
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-syntax_role-var_score={var_score}-2.pdf")
                                plt.close("all")


                                fig = sns.catplot(data=dfthis, x="syntax_concrete_idx1", y=var_score, errorbar=("ci", 68), hue="decoder_class_idx_in_shapes_drawn", kind="bar")
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-syntax_concrete_idx1-var_score={var_score}-1.pdf")

                                # fig = sns.catplot(data=dfthis, x="decoder_class_idx_in_shapes_drawn", y=var_score, errorbar=("ci", 68), hue="syntax_concrete_idx1", kind="point")
                                # for ax in fig.axes.flatten():
                                #     ax.axhline(0, color="k", alpha=0.5)
                                # savefig(fig, f"{savedir}/catplot-syntax_concrete_idx1-var_score={var_score}-2.pdf")

                                fig = sns.catplot(data=dfthis, x="decoder_class_idx_in_shapes_drawn", y=var_score, errorbar=("ci", 68), hue="rank_dist_0_1_correct", col="syntax_concrete_idx1", col_wrap=4, kind="point")
                                for ax in fig.axes.flatten():
                                    ax.axhline(0, color="k", alpha=0.5)
                                savefig(fig, f"{savedir}/catplot-syntax_concrete_idx1-var_score={var_score}-3.pdf")
                                plt.close("all")

                                # Related to decoder rank in shapes drawn
                                # fig = sns.catplot(data=dfthis, x="decoder_class_idx_in_shapes_drawn", y=var_score, kind="point", errorbar=("ci", 68))
                                # rotateLabel(fig)
                                # for ax in fig.axes.flatten():
                                #     ax.axhline(0, color="k", alpha=0.5)
                                # savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-var_score={var_score}-1.pdf")

                                # fig = sns.catplot(data=dfthis, x="decoder_class_idx_in_shapes_drawn", y=var_score, kind="point", errorbar=("ci", 68),
                                #             hue="decoder_class_idx_in_syntax")
                                # rotateLabel(fig)
                                # for ax in fig.axes.flatten():
                                #     ax.axhline(0, color="k", alpha=0.5)
                                # savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-var_score={var_score}-2.pdf")

                                # fig = sns.catplot(data=dfthis, x="decoder_class_idx_in_shapes_drawn", y=var_score, kind="point", errorbar=("ci", 68),
                                #             hue="syntax_concrete", col="decoder_class_idx_in_syntax")
                                # rotateLabel(fig)
                                # for ax in fig.axes.flatten():
                                #     ax.axhline(0, color="k", alpha=0.5)
                                # savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-var_score={var_score}-3.pdf")


                                # fig = sns.catplot(data=dfthis, x="decoder_class_idx_in_shapes_drawn", y=var_score, kind="point", errorbar=("ci", 68),
                                #             hue="syntax_concrete_idx1")
                                # rotateLabel(fig)
                                # for ax in fig.axes.flatten():
                                #     ax.axhline(0, color="k", alpha=0.5)
                                # savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-var_score={var_score}-4.pdf")
                                

                                for hue in [None, "decoder_class_idx_in_syntax", "syntax_concrete_idx1", "syntax_concrete_idx1_drawn"]:
                                    fig = sns.catplot(data=dfthis, x = "decoder_class_idx_in_shapes_drawn", y=var_score, 
                                                        kind="point", errorbar=("ci", 68), hue = hue)
                                    for ax in fig.axes.flatten():
                                        ax.axhline(0, color="k", alpha=0.5)
                                    savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-hue={hue}-var_score={var_score}.pdf")
                                
                                plt.close("all")
