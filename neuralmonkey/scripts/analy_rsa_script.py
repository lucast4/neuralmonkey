""" Compute RSA analyses for all QUESTIONS for this day.
Starting from pre-computed Snippets...
ie across all whicih_levels, events, breagions, and time windows.
Just savees Results of RSA< without necsariyl making plots.

SUMMARY OF STEPS:
1. Extract Snippets using ./analy_snippets_extract_script.sh
2. Update params in rsa.rsagood_questions_params
3. Update params in rsa.rsagood_questions_dict (make sure animal, date is present).
4. Run this script.

"""
import sys
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
from neuralmonkey.analyses.rsa import rsagood_questions_dict
from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
from neuralmonkey.analyses.state_space_good import snippets_extract_popanals_split_bregion_twind
import pandas as pd
from neuralmonkey.analyses.rsa import rsagood_score_wrapper, rsagood_pa_effectsize_plot_summary, rsagood_pa_vs_theor_samecontextctrl, rsagood_pa_vs_theor_plot_results, rsagood_pa_vs_theor_plot_pairwise_distmats, rsagood_score_vs_shuff_wrapper
from neuralmonkey.classes.population_mult import dfpa_slice_specific_windows, dfpa_group_and_split



assert False, "repalce with dfallpa_extraction_load_wrapper"

# list_time_windows = [
#     (-0.6, -0.4),
#     (-0.5, -0.3),
#     (-0.4, -0.2),
#     (-0.3, -0.1),
#     (-0.2, 0.),
#     (-0.1, 0.1),
#     (0., 0.2),
#     (0.1, 0.3),
#     (0.2, 0.4),
#     (0.3, 0.5),
#     (0.4, 0.6),
#     ]
list_time_windows = [
    (-0.5, -0.3),
    (-0.3, -0.1),
    (-0.1, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    ]
# list_time_windows = [
#     (-0.5, -0.25),
#     (-0.25, 0.),
#     (0., 0.25),
#     (0.25, 0.5),
#     ]
# version_distance = "pearson"
# version_distance = "euclidian"
version_distance = "euclidian_unbiased"
HACK_RENAME_SHAPES = True
SPIKES_VERSION = "tdt"

def rsa_pipeline(MS, question, q_params):
    """ Computes and plots all for this question
    """

    # Collect PA across all wls.
    outs = []
    for wl in q_params["list_which_level"]:
        # Load Snippets
        SP, _ = load_and_concat_mult_snippets(MS, which_level = wl)

        # Clean up SP and extract features
        D, list_features_extraction = SP.datasetbeh_preprocess_clean_by_expt(
            ANALY_VER=q_params["ANALY_VER"], vars_extract_append=q_params["effect_vars"])

        assert "epoch" in list_features_extraction, "sanity check"

        # Prune to just the events of interest
        # if q_params["events_keep"] is not None:
        #     events_in = SP.DfScalar["event"].unique().tolist()
        #     SP.DfScalar = SP.DfScalar[SP.DfScalar["event"].isin(q_params["events_keep"])].reset_index(drop=True)
        #     if len(SP.DfScalar)==0:
        #         print(q_params["events_keep"])
        #         print(events_in)
        #         assert False

        # Prune to just specific var:level combinations of interest
        if q_params["dict_vars_levels_prune"] is not None:
            print("Pruning data from SP.DfScalar (dict_vars_levels_prune is not None):")
            for var, levs in q_params["dict_vars_levels_prune"].items():
                # For example:
                # var = "epoch"
                # levs = ["oneshp_varyloc"]

                print(var, ", these levels keep: ", levs)
                SP.DfScalar = SP.DfScalar[SP.DfScalar[var].isin(levs)].reset_index(drop=True)
                if len(SP.DfScalar)==0:
                    print(q_params["events_keep"])
                    print(events_in)
                    assert False

        # Extract all popanals
        dfallpa = snippets_extract_popanals_split_bregion_twind(SP, list_time_windows,
                                                        list_features_extraction,
                                                        HACK_RENAME_SHAPES=HACK_RENAME_SHAPES,
                                                        events_keep=q_params["events_keep"])

        assert "epoch" in list_features_extraction, "sanity check"
        list_pa = dfallpa["pa"].tolist()
        for pa in list_pa:
            for var in list_features_extraction:
                if var not in pa.Xlabels["trials"].columns:
                    print(q_params)
                    print(pa.Xlabels["trials"])
                    print(pa.Xlabels["trials"].columns)
                    print(list_features_extraction)
                    assert False

        # Collect
        outs.append(dfallpa)

    # Concat across all
    DFallpa = pd.concat(outs).reset_index(drop=True)

    # Clear memory
    del SP

    # Optionally, slice and agg.
    if q_params["slice_agg_slices"] is not None:
        slice_agg_slices = q_params["slice_agg_slices"]
        slice_agg_vars_to_split = q_params["slice_agg_vars_to_split"]

        if slice_agg_slices is not None:
            # 1) slice
            print(" *** Before dfpa_slice_specific_windows")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print("slice_agg_slices:", slice_agg_slices)
            DFallpa = dfpa_slice_specific_windows(DFallpa, slice_agg_slices)

            # 2) agg (one pa per bregion)
            print(" *** Before dfpa_group_and_split")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print(slice_agg_vars_to_split)
            DFallpa = dfpa_group_and_split(DFallpa, vars_to_split=slice_agg_vars_to_split)

            print(" *** After dfpa_group_and_split")
            print(DFallpa["which_level"].value_counts())
            print(DFallpa["event"].value_counts())
            print(DFallpa["twind"].value_counts())
            print("Event, within pa:")

            for pa in DFallpa["pa"].tolist():
                print(pa.Xlabels["trials"]["event"].value_counts())
                print(pa.Xlabels["trials"]["wl_ev_tw"].value_counts())
                assert isinstance(pa.Xlabels["trials"]["wl_ev_tw"].values[0], str)

    # Other params for rsa computation.
    list_subtract_mean_each_level_of_var = q_params["list_subtract_mean_each_level_of_var"]
    list_vars_test_invariance_over_dict = q_params["list_vars_test_invariance_over_dict"]

    # if version_distance in "euclidian_unbiased":
    #     use_distributional_distance = True
    # else:
    #     use_distributional_distance = False

    # if False: # DEBUGGING
    list_pa = DFallpa["pa"].tolist()
    for pa in list_pa:
        for var in q_params["effect_vars"]:
            if var not in pa.Xlabels["trials"].columns:
                print(var)
                print(q_params)
                print(pa.Xlabels["trials"])
                print(list_features_extraction)
                assert False

    for subtract_mean_each_level_of_var in list_subtract_mean_each_level_of_var:
        for vars_test_invariance_over_dict in list_vars_test_invariance_over_dict:

            ############################# RSA COMPUTE
            PLOT_INDIV = False
            DO_AGG_TRIALS = True
            DFRES_THEOR, DFRES_EFFECT_MARG, DFRES_EFFECT_CONJ, DFRES_SAMEDIFF, Params = \
                rsagood_score_wrapper(DFallpa, animal, date, question, q_params, version_distance,
                                      subtract_mean_each_level_of_var,
                                      vars_test_invariance_over_dict,
                                      DO_AGG_TRIALS=DO_AGG_TRIALS, PLOT_INDIV=PLOT_INDIV)

            ########################### RSA PLOTS
            # DFRES_THEOR, DFallpa, Params, savedir = rsagood_pa_vs_theor_wrapper_loadresults(animal, date, question, version_distance, DO_AGG_TRIALS, subtract_mean_each_level_of_var)
            savedir = Params["savedir"]
            yvar = "cc"

            ### (1) Overview summary plots
            print("** SAving figures to:", savedir)
            rsagood_pa_vs_theor_plot_results(DFRES_THEOR, savedir, yvar)

            if "cc_same_context" in DFRES_THEOR.columns:
                rsagood_pa_vs_theor_samecontextctrl(DFRES_THEOR, savedir)

            ### (2) PLOT PAIRWISE DIST MATRICES, for hand-selected variables.
            # Reload the params for this question.
            # question_params = rsagood_questions_params(Params["question"])
            question_params = Params["question_params"]
            variables_plot = question_params["plot_pairwise_distmats_variables"]
            list_wl_ev_tw_plot = question_params["plot_pairwise_distmats_twinds"]
            vars_test_invariance_over_dict_THIS = None # becuase here is looking at pairwise distmats, so would fail
            rsagood_pa_vs_theor_plot_pairwise_distmats(DFallpa, Params, savedir, variables_plot,
                                                       list_wl_ev_tw_plot, vars_test_invariance_over_dict_THIS,
                                                       DO_AGG_TRIALS_PLOT=True)

            ### (3) Plot effect sizes, and relationship between RSA and effect sizes
            if False:
                # Skip, works, but havent really been checking... redudnant with some above stuff (same, diff in same context)
                rsagood_pa_effectsize_plot_summary(DFRES_THEOR, DFRES_EFFECT_MARG, DFRES_EFFECT_CONJ,
                                                       Params)


            ### (4) Get same-diff scores, and compare to shuffle controls (upper and lower bounds)
            if False:
                # This isnt the best shuffle, replaced with above
                # rsagood_pa_vs_theor_samecontextctrl
                rsagood_score_vs_shuff_wrapper(DFallpa, animal, date, question, q_params,
                                          subtract_mean_each_level_of_var, version_distance,
                                            vars_test_invariance_over_dict, savedir)

if __name__ == "__main__":

    animal = sys.argv[1]
    date = int(sys.argv[2])
    MS = load_mult_session_helper(date, animal, spikes_version=SPIKES_VERSION)
    MULTI = False

    DictParamsEachQuestion = rsagood_questions_dict(animal, date)

    questions = [q for q in DictParamsEachQuestion.keys()]
    q_params_s = [qp for qp in DictParamsEachQuestion.values()]
    MSs = [MS for _ in range(len(questions))]

    if MULTI:
        from multiprocessing import Pool
        ncores = 4
        with Pool(ncores) as pool:
            pool.starmap(rsa_pipeline, zip(MSs, questions, q_params_s))
    else:
        for question, q_params in DictParamsEachQuestion.items():
            rsa_pipeline(MS, question, q_params)


