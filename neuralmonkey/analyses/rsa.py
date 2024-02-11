""" Related to rsa analyses
1/15/24 (NOTE, was previously in state_space_good..

"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.expttools import load_yaml_config
from pythonlib.tools.plottools import makeColors, legend_add_manual, savefig, rotate_x_labels, rotate_y_labels
from pythonlib.tools.expttools import writeDictToYaml
from pythonlib.tools.pandastools import append_col_with_grp_index, convert_to_2d_dataframe
from pythonlib.tools.snstools import rotateLabel
from neuralmonkey.analyses.state_space_good import popanal_preprocess_scalar_normalization

from pythonlib.globals import PATH_ANALYSIS_OUTCOMES

SAVEDIR_ANALYSES = f"{PATH_ANALYSIS_OUTCOMES}/recordings/main/RSA"

def behavior_load_pairwise_strokes_distmat(animal, DATE, PA, distance_ver = "dtw_vels_2d"):
    """ Load pre-computed distnace matrix for this dataset, return distmat matching exactly the
    rows in PA.Xlabels["trials"], and ensuring that it is distnace (not sim) matrix
    """
    DIR = f"/gorilla1/analyses/recordings/main/EXPORTED_BEH_DATA/DS/{animal}/{DATE}/distance_matrix_all_strokes_pairwise/{distance_ver}"

    ## LOAd
    # DATA
    path = f"{DIR}/data.npy"
    with open(path, "rb") as f:
        simmat = np.load(f)
        print("... loaded pre-saved distmat data:", path, " -- shape: ", simmat.shape)

    path = f"{DIR}/stroke_indices_in_order.npy"
    with open(path, "rb") as f:
        stroke_indices_in_order = np.load(f)

    path = f"{DIR}/trialcodes_in_order.npy"
    with open(path, "rb") as f:
        trialcodes_in_order = np.load(f)

    ## Slice data matrix to match the data in PA exactly. Extract in correct shape for PA
    # - target indices
    list_tc_get = PA.Xlabels["trials"]["trialcode"].tolist()
    list_si_get = PA.Xlabels["trials"]["stroke_index"].tolist()
    list_tc_si_get = [(tc, si) for tc, si in zip(list_tc_get, list_si_get)]
    # - loaded indices
    list_tc_si_loaded = [(tc, si) for tc, si in zip(trialcodes_in_order, stroke_indices_in_order)]
    assert len(set(list_tc_si_loaded)) == len(list_tc_si_loaded), "why? bug?"

    # Slices the loaded data to match order of target
    idxs = []
    for tc_si in list_tc_si_get:
        # find the index of this tc_si in the loaded data
        idx = list_tc_si_loaded.index(tc_si)
        assert isinstance(idx, int)
        idxs.append(idx)
    assert len(set(idxs)) == len(idxs), "why? bug?"

    # sanity check
    assert [trialcodes_in_order[i] for i in idxs] == list_tc_get

    ## finally, slice data
    simmat_correct = simmat[idxs,:][:, idxs]

    # if this is sim, then flip it
    if distance_ver=="euclidian_diffs":
        assert np.max(simmat_correct)<=1.
        assert np.min(simmat_correct)>=0.
        distmat = 1-simmat_correct # Flip it, so is "distance"
    else:
        print(distance_ver)
        assert False, "hand input the tformation"

    return distmat

def OLD_pipeline_rsa_all_steps(SP, EFFECT_VARS, list_time_windows,
                               SAVEDIR, version_distance,
                               subtract_mean_each_level_of_var = None,
                               PLOT_INDIV=True, SKIP_ANALY_PLOTTING=True):
    """ Extraction of specific PopAnals for each conjunction of (twind, bregion).
    Optioanlly also computes, for each of those, their RSAs (comapred to theoretical
    dist matrices) [if SKIP_ANALY_PLOTTING==False]
    PARAMS:
    - EFFECT_VARS, list of str, vars to extract, mainly to make sure the etracted PA have all
    variables. If not SKIP_ANALY_PLOTTING, then these also determine which plots.
    - list_time_windowsm, list of timw eindow, tuples .e.g, (-0.2, 0.2), each defining a specific
    extracvted PA.
    - version_distance, for computing distance matrices (of neural activity). OInly used if plotting.
    - subtract_mean_each_level_of_var, for normalizing, befor compute dist mat (plotting_).
    RETURNS:
    - DFRES_SAMEDIFF, DFRES_THEOR, DictBregionTwindPA, \
        DictBregionTwindClraw, DictBregionTwindClsim, savedir
        (NOTE DictBregionTwindPA is the important one that does regardless whether you do plotign)_.
    """
    from pythonlib.tools.pandastools import summarize_featurediff
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from neuralmonkey.analyses.state_space_good import snippets_extract_popanals_split_bregion_twind

    DictBregionTwindPA = snippets_extract_popanals_split_bregion_twind(SP, list_time_windows, EFFECT_VARS)

    savedir = f"{SAVEDIR}/{SP.animal()}/{SP.date()}/{SP.Params['which_level']}/{version_distance}"
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    if SKIP_ANALY_PLOTTING:
        return None, None, DictBregionTwindPA, \
            None, None, None, savedir,
    else:
        return _pipeline_rsa_score_pa_all(DictBregionTwindPA, EFFECT_VARS, savedir,
                                          version_distance, subtract_mean_each_level_of_var,
                                          PLOT_INDIV)

def rsagood_pa_vs_theor_wrapper_loadresults(animal, date, question, version_distance,
                                            DO_AGG_TRIALS, subtract_mean_each_level_of_var,
                                            vars_test_invariance_over_dict):
    """ Load results of rsagood_pa_vs_theor_wrapper,
    RETURNS:
        - DFRES_THEOR, df with columns var, cc, bregion, twind, evnet, which_level
    """

    if vars_test_invariance_over_dict is None:
        suff = "invar_None"
    else:
        a = "_".join(vars_test_invariance_over_dict["same"])
        b = "_".join(vars_test_invariance_over_dict["diff"])
        suff = f"invar_{a}_{b}"

    for savedir in [
        f"{SAVEDIR_ANALYSES}/{animal}-{date}/agg_{DO_AGG_TRIALS}-subtr_{subtract_mean_each_level_of_var}-dist_{version_distance}/{question}",
        f"{SAVEDIR_ANALYSES}/{animal}-{date}/agg_{DO_AGG_TRIALS}-subtr_{subtract_mean_each_level_of_var}-dist_{version_distance}-{suff}/{question}"
        ]:

        if os.path.exists(savedir):
            path = f"{savedir}/DFRES_THEOR.pkl"
            DFRES_THEOR = pd.read_pickle(path)

            path = f"{savedir}/DFRES_EFFECT_CONJ.pkl"
            DFRES_EFFECT_CONJ = pd.read_pickle(path)

            path = f"{savedir}/DFRES_EFFECT_MARG.pkl"
            DFRES_EFFECT_MARG = pd.read_pickle(path)

            path = f"{savedir}/DFRES_SAMEDIFF.pkl"
            DFRES_SAMEDIFF = pd.read_pickle(path)

            path = f"{savedir}/DFallpa.pkl"
            DFallpa = pd.read_pickle(path)
            # Fix an old problem, linked data (all data before 1/28/24)
            for i, row in DFallpa.iterrows():
                # Fix a problem
                # The only var that was (incorrectly) linked across pa was twind.
                a = row["twind"] # The correct twind
                b = row["pa"].Xlabels["trials"]["twind"].values[0] # iuncorrect
                if not a==b:
                    row["pa"].Xlabels["trials"] = row["pa"].Xlabels["trials"].copy()
                    row["pa"].Xlabels["trials"]["twind"] = [a for _ in range(len(row["pa"].Xlabels["trials"]))]

            from pythonlib.tools.expttools import load_yaml_config
            path = f"{savedir}/Params.yaml"
            Params = load_yaml_config(path)

            Params["list_which_level"] = sorted(DFallpa["which_level"].unique().tolist())
            Params["list_event"] = sorted(DFallpa["event"].unique().tolist())
            Params["list_bregion"] = sorted(DFallpa["bregion"].unique().tolist())
            Params["list_twind"] = sorted(DFallpa["twind"].unique().tolist())
            Params["EFFECT_VARS"] = sorted(DFRES_THEOR["var"].unique().tolist())

            return DFallpa, DFRES_THEOR, DFRES_SAMEDIFF, DFRES_EFFECT_CONJ, DFRES_EFFECT_MARG, Params, savedir
    assert False, "didnt find saved data"

def rsagood_score_vs_shuff_wrapper(DFallpa, animal, date, question, q_params,
                                   subtract_mean_each_level_of_var, version_distance,
                                   vars_test_invariance_over_dict,
                                   SAVEDIR, yvar="cc"):
    """
    Does a couple things: A new wauy to score, which gets, for each var (against
    conjunction of othervars)
    - same_good: for each lev of var, dist to same lev across levels of othervar
    - diff_good: for each lev of var, dist to diff lev (same var) across lev of othervar.
    And does this after shuffling for each var, within each level of that var, shuffling
    levels of otherviar.
    Idea:
    - upper bound: shuffle.
    - data: same_good score.
    - lower bound: diff_good.

    :param DFallpa:
    :param animal:
    :param date:
    :param question:
    :param q_params:
    :param version_distance:
    :return:
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index

    effect_vars = q_params["effect_vars"]
    savedir = f"{SAVEDIR}/samediff_upperlower"
    os.makedirs(savedir, exist_ok=True)

    ##################### GET DATA
    # version_distance = "pearson"
    # version_distance = "euclidian"
    # version_distance = "angle"
    # subtract_mean_each_level_of_var = q_params["subtract_mean_each_level_of_var"]
    PLOT_INDIV = False
    DO_AGG_TRIALS = True
    DFRES_THEOR, DFRES_EFFECT_MARG, DFRES_EFFECT_CONJ, DFRES_SAMEDIFF, Params = rsagood_score_wrapper(DFallpa, animal,
                                                                                                      date, question,
                                                                                                      q_params,
                                                                                                      version_distance,
                                                                                                      subtract_mean_each_level_of_var,
                                                                                                      vars_test_invariance_over_dict,
                                                                                                      DO_AGG_TRIALS=DO_AGG_TRIALS,
                                                                                                      PLOT_INDIV=PLOT_INDIV,
                                                                                                      do_save=False)
    DFRES_SAMEDIFF = append_col_with_grp_index(DFRES_SAMEDIFF, ["which_level", "twind"], "wl_tw", use_strings=False)
    DFRES_THEOR = append_col_with_grp_index(DFRES_THEOR, ["which_level", "twind"], "wl_tw", use_strings=False)

    ##################### GET A SINGLE SHUFFLE
    from neuralmonkey.metrics.scalar import _shuffle_dataset_hier
    for var in effect_vars:
        list_pa_shuff = []
        for pa in DFallpa["pa"].tolist():
            ##########
            vars_shuffle = [v for v in effect_vars if not v==var]

            df = pa.Xlabels["trials"]
            df_shuff = _shuffle_dataset_hier(df, [var], vars_shuffle)
            assert(np.all(df[var] == df_shuff[var]))
            assert(np.all(df["index_datapt"] == df_shuff["index_datapt"]))

            # Replace
            pa_shuff = pa.copy()
            pa_shuff.Xlabels["trials"] = df_shuff

            # Save
            list_pa_shuff.append(pa_shuff)

        dfallpa_shuff = DFallpa.copy()
        dfallpa_shuff["pa"] = list_pa_shuff

        # Compute
        DFRES_THEOR_SHUFF, DFRES_EFFECT_MARG_SHUFF, DFRES_EFFECT_CONJ_SHUFF, DFRES_SAMEDIFF_SHUFF, Params_SHUFF = rsagood_score_wrapper(
            dfallpa_shuff, animal, date, question, q_params, version_distance,
            subtract_mean_each_level_of_var, vars_test_invariance_over_dict,
            DO_AGG_TRIALS=DO_AGG_TRIALS, PLOT_INDIV=False, do_save=False)
        DFRES_SAMEDIFF_SHUFF = append_col_with_grp_index(DFRES_SAMEDIFF_SHUFF, ["which_level", "twind"], "wl_tw", use_strings=False)
        DFRES_THEOR_SHUFF = append_col_with_grp_index(DFRES_THEOR_SHUFF, ["which_level", "twind"], "wl_tw", use_strings=False)

        # Append shuff to nonshuff
        # only keep the "same" for shuff
        dftmp = DFRES_SAMEDIFF_SHUFF[DFRES_SAMEDIFF_SHUFF["score_ver"]=="same_good"].reset_index(drop=True)
        dftmp["score_ver"] = f"ShfPosctrl-{var}"
        DFRES_SAMEDIFF = pd.concat([DFRES_SAMEDIFF, dftmp]).reset_index(drop=True)

        # Append
        list_var = DFRES_THEOR_SHUFF["var"].tolist()
        DFRES_THEOR_SHUFF["var"] = [f"{v}-ShfPosctrl-{var}" for v in list_var]
        DFRES_THEOR = pd.concat([DFRES_THEOR, DFRES_THEOR_SHUFF]).reset_index(drop=True)

    ###################### PLOTS
    list_ev = sorted(DFRES_SAMEDIFF["event"].unique().tolist())
    for ev in list_ev:
        dfthis = DFRES_SAMEDIFF[DFRES_SAMEDIFF["event"]==ev]
        fig = sns.catplot(data=dfthis, x="bregion", y="score", hue="score_ver", col="wl_tw", row="var", kind="point")
        plt.axhline(0)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/pointplot-same_diff-{ev}.pdf")
        plt.close("all")

    fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", col="wl_tw", row="event", kind="point")
    plt.axhline(0)
    rotateLabel(fig)
    savefig(fig, f"{savedir}/pointplot-theor.pdf")

    plt.close("all")

# _pipeline_rsa_score_pa_all
# rsagood_pa_vs_theor_wrapper
def rsagood_score_wrapper(DFallpa, animal, date, question, q_params, version_distance,
                          subtract_mean_each_level_of_var, vars_test_invariance_over_dict,
                          DO_AGG_TRIALS, PLOT_INDIV=True, do_save=True):
    """ GOOD - For each PA in DF, cOmpare distances matrices against theoretical matrices, and
    return summary rsults (and save).
    PARAMS.
    :param vars_test_invariance_over_dict:  independent vars, for rsa comparison
    to theoretical matrices, will onluy cases with diff values for each of the variables
    in varsconj_test_invariance_over. e.g., if vars_test_invariance_over = [shape, event], then
    must be diff for BOTH shape and event.
    """
    from pythonlib.tools.pandastools import summarize_featurediff
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.expttools import writeDictToYaml

    if vars_test_invariance_over_dict is None:
        suff = "invar_None"
    else:
        a = "_".join(vars_test_invariance_over_dict["same"])
        b = "_".join(vars_test_invariance_over_dict["diff"])
        suff = f"invar_{a}_{b}"

    # Sanity
    if do_save:
        savedir = f"{SAVEDIR_ANALYSES}/{animal}-{date}/agg_{DO_AGG_TRIALS}-subtr_{subtract_mean_each_level_of_var}-dist_{version_distance}-{suff}/{question}"
        os.makedirs(savedir, exist_ok=True)
    else:
        savedir = None

    # if use_distributional_distance:
    #     DO_AGG_TRIALS = False # need to keep distribution before agging.

    ## PRUNE DATA based on this question
    if False:
        # This must operate BEFORE agging dfallpa in dfpa_group_and_split.
        # Its done
        if q_params["events_keep"] is not None:
            DFallpa = DFallpa["event"].isin(q_params["events_keep"]).reset_index(drop=True)

    list_wl = sorted(DFallpa["which_level"].unique().tolist())
    list_ev = sorted(DFallpa["event"].unique().tolist())

    #################### First, quickly check that all params are doable.
    # e.g, enough data for all conjunctions.
    for wl in list_wl:
        for ev in list_ev:
            # Do for each wl, ev, since htey can have diferent data sizes.

            # Check the first PA.
            DFallpa_THIS = DFallpa[(DFallpa["which_level"]==wl) & (DFallpa["event"]==ev)].reset_index(drop=True)
            pa = DFallpa_THIS["pa"].values[0]

            # Prune PA based on this question
            PAout, res_check_tasksets, res_check_effectvars = preprocess_rsa_prepare_popanal_wrapper(pa, **q_params)

            # SAVE
            for var, dfcheck in res_check_tasksets.items():
                print("---------------")
                print("var:", var)
                # print("levels:", dfcheck["lev"].tolist())

                if do_save:
                    # Save the checks
                    path = f"{savedir}/data_check_pruned-{question}-{var}--{wl}-{ev}.csv"
                    dfcheck.to_csv(path)

            if PAout is None:
                # Summarize variables
                for k, v in res_check_tasksets.items():
                    print(k)
                for var, v in res_check_effectvars.items():
                    print(var)

                if do_save:
                    ########### SKIPPING ANALYSIS AND EXITING!!
                    path = f"{savedir}/SKIPPING_QUESTION_NOT_ENOUGH_DATA.yaml"
                    writeDictToYaml(q_params, path)

                    path = f"{savedir}/res_check_effectvars.yaml"
                    writeDictToYaml(res_check_effectvars, path)
                return None



    ################ COLLECT AND SAVE ALL RESULTS, running separately for each question.
    EFFECT_VARS = q_params["effect_vars"]
    if "Stroke" in EFFECT_VARS:
        assert q_params["distmat_distance_ver"] is not None
        assert q_params["distmat_animal"] is not None
        assert q_params["distmat_date"] is not None

    # Iterate separately for each which_level and event, since there are possible they have
    # different num rows (datapts), and these are made efficent assuming they are same.
    list_dfres_theor = []
    list_dfeffect_marg = []
    list_dfeffect_conj = []
    list_df_sd = []
    # for wl in list_wl:
    #     for ev in list_ev:
    #
    #         DFallpa_THIS = DFallpa[(DFallpa["which_level"]==wl) & (DFallpa["event"]==ev)].reset_index(drop=True)
    # DictVarToClsimtheor = None # only compute first time, rest of time pass it back in.

    # sort by wl, then ev, so that the below can easily reuse DictVarToClsimtheor
    DFallpa = DFallpa.sort_values(["which_level", "event", "twind"])

    # Collect results.
    ct = 0
    # DictVarToClsimtheor_EachLevEvent = {}
    DictVarToClsimtheor = {}
    for ind in range(len(DFallpa)):
        pa = DFallpa.iloc[ind]["pa"]
        wl = DFallpa.iloc[ind]["which_level"]
        ev = DFallpa.iloc[ind]["event"]
        br = DFallpa.iloc[ind]["bregion"]
        tw = DFallpa.iloc[ind]["twind"]

        if do_save:
            sdir = f"{savedir}/process_single_pa/{wl}-{ev}-{br}-{tw}"
            os.makedirs(sdir, exist_ok=True)
            print(sdir)
        else:
            sdir = None

        # Prune PA based on this question
        pa, res_check_tasksets, res_check_effectvars = preprocess_rsa_prepare_popanal_wrapper(pa, **q_params)

        assert len(pa.X)>0

        # if (wl, ev) in DictVarToClsimtheor_EachLevEvent.keys():
        #     DictVarToClsimtheor = DictVarToClsimtheor_EachLevEvent[(wl, ev)]
        # else:
        #     DictVarToClsimtheor = None

        ## RUN ANALYSIS
        PLOT_THEORETICAL_SIMMATS = ct==0 and PLOT_INDIV==True # Only do once. this same across bregions and twinds.
        _, dfres_theor, Clraw, _, PAagg, DictVarToClsimtheor = rsagood_pa_vs_theor_single(pa, EFFECT_VARS,
                                                                                          version_distance,
                                                                                          subtract_mean_each_level_of_var=subtract_mean_each_level_of_var,
                                                                                          vars_test_invariance_over_dict=vars_test_invariance_over_dict,
                                                                                          PLOT=PLOT_INDIV,
                                                                                          sdir=sdir,
                                                                                          PLOT_THEORETICAL_SIMMATS=PLOT_THEORETICAL_SIMMATS,
                                                                                          DO_AGG_TRIALS=DO_AGG_TRIALS,
                                                                                          distmat_animal=
                                                                                          q_params[
                                                                                              "distmat_animal"],
                                                                                          distmat_date=q_params[
                                                                                              "distmat_date"],
                                                                                          distmat_distance_ver=
                                                                                          q_params[
                                                                                              "distmat_distance_ver"],
                                                                                          DictVarToClsimtheor=DictVarToClsimtheor)

        assert len(dfres_theor)>0
        plt.close("all")
        # DictVarToClsimtheor_EachLevEvent[(wl, ev)] = DictVarToClsimtheor

        ####### save this result

        # (1) Analysis: corr between data and theoretical distmat
        # Collect results
        dfres_theor["bregion"] = br
        dfres_theor["twind"] = [tw for _ in range(len(dfres_theor))]
        dfres_theor["event"] = ev
        dfres_theor["which_level"] = wl
        list_dfres_theor.append(dfres_theor)

        ################# EFFECT SIZES
        df_effect_marg, df_effect_conj = rsagood_pa_effectsize_single(pa, EFFECT_VARS,
                                                      subtract_mean_each_level_of_var=subtract_mean_each_level_of_var,
                                                      DO_AGG_TRIALS=DO_AGG_TRIALS)
        if False: # Instead, get it straight from pa
            # (2) Effect sizes
            norms = np.linalg.norm(Clraw.Xinput, axis=1) # (nlabs, nvars)
            dfeffect = Clraw.rsa_labels_return_as_df(False)
            dfeffect["norm"] = norms

        df_effect_marg["bregion"] = br
        df_effect_marg["twind"] = [tw for _ in range(len(df_effect_marg))]
        df_effect_marg["event"] = ev
        df_effect_marg["which_level"] = wl

        df_effect_conj["bregion"] = br
        df_effect_conj["twind"] = [tw for _ in range(len(df_effect_conj))]
        df_effect_conj["event"] = ev
        df_effect_conj["which_level"] = wl

        list_dfeffect_marg.append(df_effect_marg)
        list_dfeffect_conj.append(df_effect_conj)

        ################### SAME, DIFF metrics (new)
        if False:
            # Skip this beucase this repalced by above, this is older, for use with
            # rsagood_score_vs_shuff_wrapper, which I don't run anymore
            df_sd = rsagood_pa_samediff_single(pa, EFFECT_VARS, subtract_mean_each_level_of_var,
                                               vars_test_invariance_over_dict=vars_test_invariance_over_dict,
                                               DO_AGG_TRIALS=DO_AGG_TRIALS, version_distance=version_distance,
                                               PLOT_MASKS=False)
            df_sd["bregion"] = br
            df_sd["twind"] = [tw for _ in range(len(df_sd))]
            df_sd["event"] = ev
            df_sd["which_level"] = wl
            list_df_sd.append(df_sd)


        # INCREMENT
        ct += 1



    if len(list_dfres_theor)==0:
        DFRES_THEOR = None
        DFRES_EFFECT_MARG = None
        DFRES_EFFECT_CONJ = None
        DFRES_SAMEDIFF = None
    else:
        DFRES_THEOR = pd.concat(list_dfres_theor).reset_index(drop=True)
        DFRES_EFFECT_MARG = pd.concat(list_dfeffect_marg).reset_index(drop=True)
        DFRES_EFFECT_CONJ = pd.concat(list_dfeffect_conj).reset_index(drop=True)
        if len(list_df_sd)>0:
            DFRES_SAMEDIFF = pd.concat(list_df_sd).reset_index(drop=True)
        else:
            DFRES_SAMEDIFF = None

    ###############################
    # SUMMARY PLOTS

    #################### 2. Comparing data simmat to theoretical simmats.
    if False: # Skip, since takes times, and I never look at this. just use heatmaps
        sdir = f"{savedir}/summary"
        os.makedirs(sdir, exist_ok=True)

        for yvar in ["cc", "mr_coeff"]:
            if yvar in DFRES_THEOR.columns:
                fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", kind="point", aspect=1.5,
                                row="twind")
                rotateLabel(fig)
                savefig(fig, f"{sdir}/vs_theor_simmat-pointplot-{yvar}.pdf")

                fig = sns.catplot(data=DFRES_THEOR, x="bregion", y=yvar, hue="var", alpha=0.5, aspect=1.5, row="twind")
                rotateLabel(fig)
                savefig(fig, f"{sdir}/vs_theor_simmat-scatterplot-{yvar}.pdf")

                # Summarize results in a heatmap (region x effect)
                for norm in ["col_sub", "row_sub", None]:
                    for twind in list_time_windows:
                        dfthis = DFRES_THEOR[DFRES_THEOR["twind"] == twind]
                        _, fig, _, _ = convert_to_2d_dataframe(dfthis, "bregion", "var",
                                                               True, "mean",
                                                               yvar, annotate_heatmap=False, dosort_colnames=False,
                                                               norm_method=norm)
                        savefig(fig, f"{sdir}/vs_theor_simmat-heatmap-{yvar}-norm_{norm}-twind_{twind}.pdf")

                        plt.close("all")

    Params = {}
    Params["animal"] = animal
    Params["date"] = date
    Params["version_distance"] = version_distance
    Params["subtract_mean_each_level_of_var"] = subtract_mean_each_level_of_var
    Params["DO_AGG_TRIALS"] = DO_AGG_TRIALS
    Params["question"] = question
    Params["question_params"] = q_params
    Params["savedir"] = savedir

    if do_save:
        ## SAVE RESULTS (for this question)
        path = f"{savedir}/DFRES_THEOR.pkl"
        pd.to_pickle(DFRES_THEOR, path)

        path = f"{savedir}/DFRES_EFFECT_MARG.pkl"
        pd.to_pickle(DFRES_EFFECT_MARG, path)

        path = f"{savedir}/DFRES_EFFECT_CONJ.pkl"
        pd.to_pickle(DFRES_EFFECT_CONJ, path)

        path = f"{savedir}/DFRES_SAMEDIFF.pkl"
        pd.to_pickle(DFRES_SAMEDIFF, path)

        path = f"{savedir}/DFallpa.pkl"
        pd.to_pickle(DFallpa, path)

        path = f"{savedir}/Params.yaml"
        writeDictToYaml(Params, path)

    return DFRES_THEOR, DFRES_EFFECT_MARG, DFRES_EFFECT_CONJ, DFRES_SAMEDIFF, Params

def rsagood_pa_samediff_single(PA, effect_vars, subtract_mean_each_level_of_var, vars_test_invariance_over_dict,
                               DO_AGG_TRIALS, version_distance, PLOT_MASKS=False):
    """ [good?] score distance for (i) same, which is, for a given variable, same for
    that var, but across levels of othervar, and (ii) diff, which is diff for that var,
    and also across levels of othervar
    :param vars_test_invariance_over_dict:
    """

    _, _, PAagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(PA, effect_vars,
                                                              subtract_mean_each_level_of_var = subtract_mean_each_level_of_var,
                                                              DO_AGG_TRIALS=DO_AGG_TRIALS)

    Clraw, Clsim = _rsagood_convert_PA_to_Cl(PAagg, effect_vars, version_distance,
                                             DO_AGG_TRIALS)

    # assert False, "here, should incorporate vars_test_invariance_over_dict"
    ma_ut = Clsim._rsa_matindex_generate_upper_triangular()
    res = []
    for var in effect_vars:
        other_vars = [v for v in effect_vars if not v==var]

        # Get component matrices
        ma_same_var, ma_diff_var = Clsim.rsa_matindex_same_diff_this_var([var])
        ma_same_othvar, ma_diff_othvar = Clsim.rsa_matindex_same_diff_this_var(other_vars)

        # Find cases with same lev for var, and diff level for othervars
        ma_same_good = ma_same_var & ma_diff_othvar & ma_ut

        # Find cases with diff lev for var, and diff level for othervars
        ma_diff_good = ma_diff_var & ma_diff_othvar & ma_ut

        if PLOT_MASKS:
            fig, axes = plt.subplots(2,3, figsize=(3*4, 2*4))
            for ax, ma in zip(axes.flatten(),
                              [ma_same_var, ma_diff_var, ma_same_othvar, ma_diff_othvar, ma_same_good, ma_diff_good]):
                Clsim.rsa_matindex_plot_bool_mask(ma, ax)
            Clsim.rsa_plot_heatmap()

        # Score
        score_same = Clsim.Xinput[ma_same_good].mean()
        score_diff = Clsim.Xinput[ma_diff_good].mean()

        # Save
        # res.append({
        #     "var":var,
        #     "score_same_good":score_same,
        #     "score_diff_good":score_diff
        # })
        res.append({
            "var":var,
            "score_ver":"same_good",
            "score":score_same
        })
        res.append({
            "var":var,
            "score_ver":"diff_good",
            "score":score_diff
        })

    dfres = pd.DataFrame(res)

    return dfres


def rsagood_pa_effectsize_single(PA, grouping_vars, subtract_mean_each_level_of_var,
                                 DO_AGG_TRIALS=True):
    """
    Get norms of fr vector for levels of the grouping variables. i.e., the area-wide
    "modulation" by this variable
    :param PA:
    :param grouping_vars:
    :param subtract_mean_each_level_of_var:
    :param DO_AGG_TRIALS:
    :return:
    - DF_EFFECT_MARG, df, marginals, one norm for each level of each variable.
    - DF_EFFECT_CONJ, df, conjucntions, one norm for each conjunction of levels across variables
    """

    # (1) Marginals, one value for each lev of each var
    list_df = []
    for var in grouping_vars:
        _, _, PAagg, _, _, _ = popanal_preprocess_scalar_normalization(PA, [var],
                                                                                      subtract_mean_each_level_of_var,
                                                                                      DO_AGG_TRIALS=DO_AGG_TRIALS)
        # get norm -- i.e., mean over units
        norm = np.linalg.norm(PAagg.X, axis=0).squeeze() # (nlevs, )

        # get results
        # dfthis = pd.DataFrame(PAagg.Xlabels["trials"][var].reset_index(drop=True))
        # dfthis["norm"] = norm
        # dfthis["var"] = var
        dfthis = pd.DataFrame({"lev":PAagg.Xlabels["trials"][var].tolist(), "norm":norm.squeeze(), "var":var})

        list_df.append(dfthis)
    DF_EFFECT_MARG = pd.concat(list_df)

    # (2) Conjucntions, one value for each conjucntion var
    _, _, PAagg, _, _, _ = popanal_preprocess_scalar_normalization(PA, grouping_vars,
                                                                                  subtract_mean_each_level_of_var,
                                                                                  DO_AGG_TRIALS=DO_AGG_TRIALS)
    # get norm -- i.e., mean over units
    norm = np.linalg.norm(PAagg.X, axis=0).squeeze() # (nlevs, )

    # get results
    # dfthis = pd.DataFrame(PAagg.Xlabels["trials"][var].reset_index(drop=True))
    # dfthis["norm"] = norm
    # dfthis["var"] = var
    DF_EFFECT_CONJ = PAagg.Xlabels["trials"].loc[:, grouping_vars]
    DF_EFFECT_CONJ["norm"] = norm

    return DF_EFFECT_MARG, DF_EFFECT_CONJ



def rsagood_pa_vs_theor_single(PA, grouping_vars, version_distance, subtract_mean_each_level_of_var,
                               vars_test_invariance_over_dict=None, PLOT=False, sdir=None,
                               PLOT_THEORETICAL_SIMMATS=False, COMPUTE_SAME_DIFF_DIST=False, COMPUTE_VS_THEOR_MAT=True,
                               DO_AGG_TRIALS=True, list_yvar=None, distmat_animal=None, distmat_date=None,
                               distmat_distance_ver=None, DictVarToClsimtheor=None):
    """
    Computes distance matrices, etc, for a single PA, usually a single subpopulation (e.g., bregion) and time slice.
    :param vars_test_invariance_over_dict:
    """
    from pythonlib.tools.plottools import savefig
    from itertools import permutations
    from scipy.stats import sem

    if list_yvar is None:
        list_yvar = ["cc"]
        # list_yvar = ["cc", "mr_coeff"]
    assert "cc" in list_yvar, "not yet coded... it depends on doing cc first. rewrite code."

    if PLOT:
        assert sdir is not None

    HACK_SKIP_SORTING = False
    RES = []
    assert isinstance(grouping_vars, list)

    if PLOT:
        plot_example_chan = PA.Chans[0]
    else:
        plot_example_chan = None

    plot_example_split_var="shape_oriented"
    if plot_example_split_var not in grouping_vars:
        plot_example_split_var=grouping_vars[0]

    if DO_AGG_TRIALS:
        _gv = grouping_vars
    else:
        _gv = None
    _, PAscal, PAscalagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(PA, _gv,
                                                                                  subtract_mean_each_level_of_var,
                                                                                  plot_example_chan=plot_example_chan,
                                                                                  plot_example_split_var=plot_example_split_var,
                                                                                  DO_AGG_TRIALS=DO_AGG_TRIALS)
    PAagg = PAscalagg

    if PLOT:
        path = f"{sdir}/preprocess_example_chan_{plot_example_chan}.pdf"
        savefig(fig, path)

    ######################## EXIT, IF NO DATA
    if PAagg is None:
        # Then pruning led to loss of all data
        return None, None, None, None, None

    ####################### CONTINUE
    # Two options:
    # if use_distributional_distance:
    #     # 2) COmpute using raw (distrubtion of trial) data, but resulting in one scalar dist per group.
    #     Clraw, Clsim = _rsagood_convert_PA_to_Cl(PAscal, grouping_vars, version_distance,
    #                                              use_distributional_distance=use_distributional_distance)
    # else:
    #     # 1) Compute sim mat using already-agged data (one dat per trial group)
    #     Clraw, Clsim = _rsagood_convert_PA_to_Cl(PAagg, grouping_vars, version_distance,
    #                                              use_distributional_distance=use_distributional_distance)
    Clraw, Clsim = _rsagood_convert_PA_to_Cl(PAscal, grouping_vars, version_distance, DO_AGG_TRIALS)

    assert len(Clraw.Xinput)>0
    assert len(Clsim.Xinput)>0

    ########## GENERATE THEORETEICAL DISTANCE MATRICES.
    if DictVarToClsimtheor is None:
        # Initialize empty
        DictVarToClsimtheor = {}
    else:
        # Then you passed it in. check that it matches data
        vars_remove = []
        for var in grouping_vars:
            if var in DictVarToClsimtheor:
                # Throw out if its incompatib with data.
                Cltheor = DictVarToClsimtheor[var]
                if not (Cltheor.Labels==Clsim.Labels and Cltheor.LabelsCols==Clsim.LabelsCols):
                    # Then remove it
                    vars_remove.append(var)
                    # print(len(Cltheor.Labels), len(Clsim.Labels))
                    # for l1, l2 in zip(Cltheor.Labels, Clsim.Labels):
                    #     print(l1, " --- ", l2)
                    # assert False, "inputted doesnt match..."
        DictVarToClsimtheor = {k:v for k,v in DictVarToClsimtheor.items() if k not in vars_remove}

    # Collect all the missing vars
    for var in grouping_vars:
        if var not in DictVarToClsimtheor:
            print("Constructing theoretical distmat for var:", var)

            if var=="Stroke":
                assert False, "decide if use PAagg or PAscal"
                # then load pre-computed distance matrix. otherwise takes forever
                dist_mat_manual = behavior_load_pairwise_strokes_distmat(distmat_animal,
                                                                         distmat_date,
                                                                         PAagg,
                                                                         distmat_distance_ver)
            else:
                dist_mat_manual = None

            Cltheor, _ = Clsim.rsa_distmat_construct_theoretical(var, PLOT=False, dist_mat_manual=dist_mat_manual)
            DictVarToClsimtheor[var] = Cltheor

    ################## STUFF REALTED TO SIMILARITY STRUCTURE (BETWEEN VARIABLES).
    # Plot heatmaps (raw and sim mats)
    if PLOT:
        if len(grouping_vars)<4:
            list_sort_order = permutations(range(len(grouping_vars)))
        else:
            list_sort_order = [list(range(len(grouping_vars)))]
        for sort_order in list_sort_order:
            figraw, ax = Clraw.rsa_plot_heatmap(sort_order, diverge=True)

            if version_distance in ["_pearson_raw"]:
                diverge = True
            else:
                diverge = False
            figsim, ax = Clsim.rsa_plot_heatmap(sort_order, diverge=diverge)
            # - name this sort order
            main_var = grouping_vars[sort_order[0]]
            s = "_".join([str(i) for i in sort_order])
            s+=f"_{main_var}"

            path = f"{sdir}/heat_raw-sort_order_{s}.pdf"
            savefig(figraw, path)

            path = f"{sdir}/heat_sim-sort_order_{s}.pdf"
            savefig(figsim, path)

            # PLOT raw data, agged.
            if len(Clraw.Labels) > len(Clsim.Labels):
                from pythonlib.cluster.clustclass import Clusters

                # then Clraw is not agged data.
                PAagg, _ = PAscal.slice_and_agg_wrapper("trials", grouping_vars, return_group_dict=True)
                X = PAagg.X.squeeze().T # (ndat, nchans)
                labels_rows = PAagg.Xlabels["trials"].loc[:, grouping_vars].values.tolist()
                labels_rows = [tuple(x) for x in labels_rows] # list of tuples
                labels_cols = PAagg.Chans # list of ints
                params = {
                    "label_vars":grouping_vars,
                }
                ClrawAGG = Clusters(X, labels_rows, labels_cols, ver="rsa", params=params)
                figraw, ax = ClrawAGG.rsa_plot_heatmap(sort_order, diverge=True)
                path = f"{sdir}/heat_raw_AGG-sort_order_{s}.pdf"
                savefig(figraw, path)

            # Make the same plots for all theoretical sim mats
            if PLOT_THEORETICAL_SIMMATS:
                for var in grouping_vars:
                    Cltheor = DictVarToClsimtheor[var]
                    figsim, ax = Cltheor.rsa_plot_heatmap(sort_order, diverge=False)

                    # - name this sort order
                    main_var = grouping_vars[sort_order[0]]
                    s = "_".join([str(i) for i in sort_order])
                    s+=f"_{main_var}"

                    # - save
                    path = f"{sdir}/heat_sim-THEOR_{var}-sort_order_{s}.pdf"
                    savefig(figsim, path)

                    if vars_test_invariance_over_dict is not None:
                        # then plot the mask
                        # vars_test_invariance_over = ["event", "seqc_0_loc"]
                        vars_same = vars_test_invariance_over_dict["same"]
                        vars_diff = vars_test_invariance_over_dict["diff"]
                        ma_invar = Clsim.rsa_matindex_same_diff_mult_var_flex(vars_same, vars_diff)
                        # ma_invar2 = Cltheor.rsa_matindex_same_diff_mult_var_flex([], vars_test_invariance_over)
                        # assert np.all(ma_invar==ma_invar2), "just sanity check of code"

                        figma, ax = Cltheor.rsa_plot_heatmap(sort_order, diverge=False, X=ma_invar)
                        ax.set_title(f"Invar over:{vars_test_invariance_over_dict}")

                        # - save
                        path = f"{sdir}/heat_sim-THEOR_{var}-sort_order_{s}-MASK.pdf"
                        savefig(figma, path)


    #### For each var, compare beh PA to ground-truth under each possible variable
    effect_vars = grouping_vars
    all_vars = grouping_vars
    # Remove vars that are not relevant.
    if vars_test_invariance_over_dict:
        effect_vars = [var for var in effect_vars if var not in vars_test_invariance_over_dict["same"]]
        effect_vars = [var for var in effect_vars if var not in vars_test_invariance_over_dict["diff"]]
    if subtract_mean_each_level_of_var is not None:
        effect_vars = [var for var in effect_vars if not var==subtract_mean_each_level_of_var]

    RES_VS_THEOR = []
    list_vec = []
    if len(all_vars)>1:
        for var in effect_vars:
            Cltheor = DictVarToClsimtheor[var]
            # Cltheor, fig = Clsim.rsa_distmat_construct_theoretical(var, PLOT=False)

            # plot
            # if PLOT_THEORETICAL_SIMMATS:
            #     # Plot heatmaps (raw and sim mats)
            #     sort_order = (0,) # each tuple is only len 1...
            #     figsim, ax = Cltheor.rsa_plot_heatmap(sort_order, diverge=False)
            #     s = "_".join([str(i) for i in sort_order])
            #     s+=f"_{var}"
            #     path = f"{sdir}/heat_sim-THEOR-sort_order_{s}.pdf"
            #     savefig(figsim, path)

            if COMPUTE_VS_THEOR_MAT and "cc" in list_yvar:

                # Optionally mask data before scoring
                if vars_test_invariance_over_dict is not None:
                    # generate mask
                    mask_vars_same = vars_test_invariance_over_dict["same"]
                    mask_vars_diff = vars_test_invariance_over_dict["diff"]
                else:
                    mask_vars_same, mask_vars_diff = None, None

                if sdir is not None:
                    plot_and_save_mask_path = f"{sdir}/final_mask-diff_ctxt-{var}.png"
                else:
                    plot_and_save_mask_path = None

                if False:
                    # old version
                    c = Clsim.rsa_distmat_score_vs_theor(Cltheor, mask_vars_same, mask_vars_diff,
                                                         help_context="othervars_at_least_one_diff",
                                                         plot_and_save_mask_path=plot_and_save_mask_path)

                    # Also get positive control? Usually this is done by restricting analysis
                    # to same values for a set of context vars. With one effect var of interest
                    # - This should be all the vars that are NOT the tested var
                    if sdir is not None:
                        plot_and_save_mask_path = f"{sdir}/final_mask-same_ctxt-{var}.png"
                    else:
                        plot_and_save_mask_path = None
                    c_same_context = Clsim.rsa_distmat_score_vs_theor(Cltheor, PLOT=False, exclude_diag=False,
                                                                  help_context="othervars_all_same",
                                                                  plot_and_save_mask_path=plot_and_save_mask_path)
                else:
                    c, c_same_context = Clsim.rsa_distmat_score_vs_theor(Cltheor,
                                                                         vars_test_invariance_over_dict,
                                                                         plot_and_save_mask_path=plot_and_save_mask_path)

                # # Correlation matrix between data and theoreitcal sim mats
                # # - get upper triangular
                # list_masks = None
                # if vars_test_invariance_over_dict is not None:
                #     # generate mask
                #     ma_invar = Clsim.rsa_matindex_same_diff_mult_var_flex(
                #         vars_test_invariance_over_dict["same"],
                #         vars_test_invariance_over_dict["diff"])
                #     list_masks = [ma_invar]
                #
                # assert Clsim.Labels == Cltheor.Labels
                # assert Clsim.LabelsCols == Cltheor.LabelsCols
                # vec_data = Clsim.dataextract_masked_upper_triangular_flattened(list_masks=list_masks, plot_mask=False)
                # vec_theor = Cltheor.dataextract_masked_upper_triangular_flattened(list_masks=list_masks, plot_mask=False)
                #
                # c = np.corrcoef(vec_data, vec_theor)[0,1]

                # Collect
                RES_VS_THEOR.append({
                    "var":var,
                    "cc":c,
                    "cc_same_context":c_same_context,
                    # "Clsim_theor":Cltheor
                })

                # Also score "same vs diff"
                restmp = Clsim.rsa_distmat_score_same_diff(var, all_vars, vars_test_invariance_over_dict, PLOT=False)

                # Append
                for score_name, score in restmp.items():
                    RES_VS_THEOR[-1][score_name] = score


    if COMPUTE_VS_THEOR_MAT:
        if "mr_coeff" in list_yvar:
            assert False, "confirm this is correct given the above masking, ma_invar"
            ## MULTIPLE REGRESSION
            from sklearn.linear_model import LinearRegression
            X = np.stack(list_vec) # (nvar, ndat)
            y = Clsim.dataextract_masked_upper_triangular_flattened()
            reg = LinearRegression().fit(X.T,y)

            # collect results
            RES_VS_THEOR_MULT_REGR = []
            for i, var in enumerate(effect_vars):
                _found = False
                for res in RES_VS_THEOR:
                    if res["var"]==var:
                        res["mr_coeff"] = reg.coef_[i]
                        _found=True
                assert _found==True

        dfres_theor = pd.DataFrame(RES_VS_THEOR)
    else:
        dfres_theor = None

    ##### Get mean within-and across context distances
    if COMPUTE_SAME_DIFF_DIST:
        assert False, "dont do this here naymore. use rsagood_"
        for ind_var in range(len(grouping_vars)):
            vals_same, vals_diff = Clsim.rsa_distmat_quantify_same_diff_variables(ind_var)
            # vals_same, vals_diff = rsa_distmat_quantify_same_diff_variables(Clsim, ind_var)

            # RES.append({
            #     "same_mean": np.mean(vals_same),
            #     "diff_mean": np.mean(vals_diff),
            #     "same_sem": sem(vals_same),
            #     "diff_sem": sem(vals_diff),
            #     "sort_order":sort_order,
            #     "ind_var":ind_var,
            #     "bregion":bregion,
            #     "grouping_vars":grouping_vars,
            #     "subtract_mean_each_level_of_var":subtract_mean_each_level_of_var
            # })

            for same_or_diff, vals in zip(
                    ["same", "diff"],
                    [vals_same, vals_diff]):
                RES.append({
                    "same_or_diff":same_or_diff,
                    "vals":vals,
                    "mean": np.mean(vals),
                    "sem": sem(vals),
                    "sort_order":tuple(sort_order) if sort_order is not None else "IGNORE",
                    "ind_var":ind_var,
                    "ind_var_str":grouping_vars[ind_var],
                    # "bregion":bregion,
                    "grouping_vars":tuple(grouping_vars),
                    "subtract_mean_each_level_of_var":subtract_mean_each_level_of_var,
                    "Clsim":Clsim,
                    "version_distance":version_distance
                })
            plt.close("all")
        dfres_same_diff = pd.DataFrame(RES)
    else:
        dfres_same_diff = None

    return dfres_same_diff, dfres_theor, Clraw, Clsim, PAagg, DictVarToClsimtheor

def _rsagood_convert_PA_to_Cl(PAscal, grouping_vars, version_distance,
                              DO_AGG_TRIALS):
    """ Convert from scalar popanal (chans, trials, 1),
    to raw Cl and distance matrix Cl objects.
    OPTIONS:
    (1) PAagg is trial-level and use_distributional_distance==False:
    - dist mat is each datapt (trial) vs. each datpt
    (2) PAagg is trial-level and use_distributional_distance==True:
    - dist mat is each conj level of grouping_vars
    (3) PAagg is trial-averaged and use_distributional_distance==False:
    - dist mat is each conj level of grouping_vars
    (4) PAagg is trial-averaged and use_distributional_distance==True:
    DOESNT MAKE SENSE!
    """
    from pythonlib.cluster.clustclass import Clusters

    if version_distance in ["euclidian", "pearson", "angle", "_pearson_raw"]:
        use_distributional_distance = False
    elif version_distance in ["euclidian_unbiased"]:
        use_distributional_distance = True
    else:
        assert False

    if DO_AGG_TRIALS and use_distributional_distance==False:
        PAagg, _ = PAscal.slice_and_agg_wrapper("trials", grouping_vars, return_group_dict=True)
    else:
        PAagg = PAscal

    for var in grouping_vars:
        assert var in PAagg.Xlabels["trials"].columns

    # Pull out data in correct format, and return as clusters.
    assert PAagg.X.shape[2]==1, "take mean over time first"
    X = PAagg.X.squeeze().T # (ndat, nchans)
    labels_rows = PAagg.Xlabels["trials"].loc[:, grouping_vars].values.tolist()
    labels_rows = [tuple(x) for x in labels_rows] # list of tuples
    labels_cols = PAagg.Chans # list of ints
    params = {
        "label_vars":grouping_vars,
    }
    Clraw = Clusters(X, labels_rows, labels_cols, ver="rsa", params=params)

    # Distnace matrix
    if use_distributional_distance:
        # Clsim = Clraw.distsimmat_convert_distr(grouping_vars, "euclidian_unbiased")
        Clsim = Clraw.distsimmat_convert_distr(grouping_vars, version_distance)
    else:
        Clsim = Clraw.distsimmat_convert(version_distance=version_distance)

    return Clraw, Clsim


################### CLEAN/CONDITION PA to be suitable for various RSA analyses...


def _preprocess_check_which_vars_categorical(PA, EFFECT_VARS):
    """ REturn list of vars wjhich are discrete or categorical
    IE exclude vars that are numerical and/opr have too many uquie data,
    and other exceptions, such as custom classses
    """
    from pythonlib.behavior.strokeclass import StrokeClass

    vars_exclude = []
    for var in EFFECT_VARS:
        nunique = len(PA.Xlabels["trials"][var].unique())
        ntot = len(PA.Xlabels["trials"][var])
        if nunique>50 and nunique/ntot>0.8:
            # Then is not group
            vars_exclude.append(var)
        elif all([isinstance(x, StrokeClass) for x in PA.Xlabels["trials"][var].tolist()]):
            # Then is strokeclass
            vars_exclude.append(var)
    vars_cat = [var for var in EFFECT_VARS if var not in vars_exclude]
    return vars_cat



def preprocess_rsa_prepare_popanal_wrapper(PA, effect_vars, exclude_last_stroke, exclude_first_stroke,
                                           min_taskstrokes,
                                           max_taskstrokes, keep_only_first_stroke=False,
                                           THRESH_clust_sim_max=None, **kwargs):
    """ This (trial-level) PA, prune it to be ready for input into analyses, based on slicing into
    subset of trials, etc. TAilored for sequence-related analyses (PIG), on datstroke data,
     and  RSA anaysi,
    e.g., including pruning lelvels with few trials, and ekeeping only speicifc tasksets.
    Then, does sanity check that have enough data, including n trials per level, and at least 2 levels of
    each other levels of conjuctions, per each level.
    PARAMS:
    - effect_vars, list of str, for checking that have neough data
    - exclude_last_stroke, bool, throw out strokes that are last (beh).
    - exclude_first_stroke, bool
    - min_taskstrokes, only keep if the trial has this many taskstrokes.
    - max_taskstrokes,
    - keep_only_first_stroke, bool, only keep the first strokes.

    """

    if keep_only_first_stroke:
        assert exclude_first_stroke==False
        assert exclude_last_stroke==False

    if keep_only_first_stroke:
        vals = [0]
        PA = PA.slice_by_labels("trials", "stroke_index", vals) # list(range(-10, -1)) --> [-10, ... , -2]
        assert len(PA.Xlabels["trials"])>0
    if exclude_last_stroke:
        vals = PA.Xlabels["trials"]["stroke_index_fromlast"].unique().tolist()
        vals = [v for v in vals if v <-1]
        PA = PA.slice_by_labels("trials", "stroke_index_fromlast", vals) # list(range(-10, -1)) --> [-10, ... , -2]
        assert len(PA.Xlabels["trials"])>0
    if exclude_first_stroke:
        vals = PA.Xlabels["trials"]["stroke_index"].unique().tolist()
        vals = [v for v in vals if v >0]
        PA = PA.slice_by_labels("trials", "stroke_index", vals) # list(range(-10, -1)) --> [-10, ... , -2]
        assert len(PA.Xlabels["trials"])>0
    PA = PA.slice_by_labels("trials", "FEAT_num_strokes_task", list(range(min_taskstrokes, max_taskstrokes+1))) # list(range(-10, -1)) --> [-10, ... , -2]
    assert len(PA.Xlabels["trials"])>0

    # if THRESH_clust_sim_max is not None:
    #     _max = PA.Xlabels["trials"]["clust_sim_max"].max()+1
    #     PA = PA.slice_by_labels_range("trials", "clust_sim_max", THRESH_clust_sim_max, _max)
    #     assert len(PA.Xlabels["trials"])>0

    # ONly check conjucntiosn and groupings for categorical variables
    effect_vars_categ = _preprocess_check_which_vars_categorical(PA, effect_vars)

    res_check_effectvars_before = _preprocess_pa_check_how_much_data(PA, effect_vars_categ)
    PA, res_check_before, res_check_after, vars_remove, reason_vars_remove = preprocess_prune_pa_enough_data(
        PA, effect_vars_categ)

    if PA is None:
        print(effect_vars, exclude_last_stroke, exclude_first_stroke,
                                           min_taskstrokes,
                                           max_taskstrokes, keep_only_first_stroke,
                                           THRESH_clust_sim_max)
        print(effect_vars_categ)
        print(vars_remove, reason_vars_remove)
        # show what went in
        for k, v in res_check_effectvars_before.items():
            print(" ----- ", k)
            print(v)
        assert False, "why threw out all the data?"

    # summarize variables used above for pruning taskset
    res_check_tasksets = _preprocess_pa_check_how_much_data(PA, effect_vars_categ)
    res_check_effectvars = res_check_after

    return PA, res_check_tasksets, res_check_effectvars


def preprocess_prune_pa_enough_data(PA, EFFECT_VARS):
    """ REpeatedly run this until there is no more change inoutput. This isbeaucase some
    steps can lead to change in other steps (e.g, remove level, then you don have enough
    levels, etc"""

    # res_check_before=pd.DataFrame([0])
    # res_check_after=pd.DataFrame([1])
    ct = 1
    did_change = True
    while did_change:
        # Then previuos run made change. try again
        PA, res_check_before, res_check_after, vars_remove, reason_vars_remove, did_change \
            = _preprocess_prune_pa_enough_data(PA, EFFECT_VARS)
        if PA is None:
            # Then no data at all
            break
        # if did_change:
        #     print(PA)
        #     print(reason_vars_remove)
        ct+=1

        assert ct<20, "bug, recursion??"
    return PA, res_check_before, res_check_after, vars_remove, reason_vars_remove




def _preprocess_prune_pa_enough_data(PA, EFFECT_VARS,
                                     n_min_lev_per_var = 2,
                                     n_min_rows_per_lev = 5,
                                     n_min_rows_per_conjunction_of_var_othervar = 2,
                                     n_min_per_conj_var = 4,
                                     DEBUG=False):
    """
    Prunes PA. For each var in EFFECT_VARS, checks each level, and removes all trials of that level
    if it fails any of the cheks (see PARAMS for the checks).
    PARAMS:
    - n_min_lev_per_var, int, each var must have at least this many levels. Prunes ENTIRE
    PA if any var fails (i.e, returns None)
    - n_min_rows_per_lev, int, each level (of each var) must have at least this many rows
    (where whether rows are trials or mean_over_trials depends on what PA passed in)
    - n_min_rows_per_conjunction_of_var_othervar, int, for each level, check how many
    rows it has for each level of (all other vars conjunction), and must be more than this.
    RETURNS:
        - PA, pruned. None if entire PA is empty
        - res_check_before, df
        - res_check_after, df
    """

    # assert False, "doesnt make sense to run this here becuase whether have enough data depends on var. if try all var, then will throw out data if one var isnt' good..."
    from pythonlib.tools.pandastools import grouping_count_n_samples

    res_check_before = _preprocess_pa_check_how_much_data(PA, EFFECT_VARS)
    did_change = False
    vars_remove = []
    vars_keep = []
    reason_vars_remove = {}
    for var in EFFECT_VARS:
        dfres_check = res_check_before[var]
        var_levs = dfres_check["lev"].unique().tolist()

        ############## REMOVE THIS ENTIRE VAR (i.e. all data...) N levels too low?
        tmp = dfres_check["n_levs_this_var"].unique().tolist()
        assert len(tmp)==1
        n_levs_this_var = tmp[0]
        if n_levs_this_var<n_min_lev_per_var:
            vars_remove.append(var)
            reason_vars_remove[var] = "not_enough_levels"
            continue

        #################### CHECK EACH LEVEL WITHIN THIS VAR
        # For each level,
        levs_remove = []
        levs_keep = []
        for lev in var_levs:

            dfres_check_thislev = dfres_check[dfres_check["lev"]==lev]
            assert len(dfres_check_thislev)==1

            # N rows for this level
            if dfres_check_thislev["n_rows_this_lev"].values[0]<n_min_rows_per_lev:
                levs_remove.append(lev)
                continue

            if "n_rows_for_each_othervar_lev" in dfres_check_thislev:
                # N othervars that this level spans
                n_each_other_var = dfres_check_thislev["n_rows_for_each_othervar_lev"].values[0]
                n_other_var_levs_with_enough_rows = sum([n >= n_min_rows_per_conjunction_of_var_othervar for n in n_each_other_var])
                if n_other_var_levs_with_enough_rows<2:
                    levs_remove.append(lev)
                    continue

            # got this far. keep it
            levs_keep.append(lev)
        if DEBUG:
            print(f"Keeping/removing these levs for {var}:", levs_keep, levs_remove)
        assert sort_mixed_type(levs_keep+levs_remove) == sort_mixed_type(var_levs)

        #################### MODIFY PA given results of check of levels.
        # Prune PA (levels, this var)
        if DEBUG:
            print(var, var_levs, levs_keep, levs_remove)
        if len(levs_keep)==0:
            # Then remove this entire dataset...
            vars_remove.append(var)
            reason_vars_remove[var] = "all_levels_fail_somehow"
            continue
        elif len(levs_keep)<len(var_levs):
            # Then just remove the levels...
            PA = PA.slice_by_labels("trials", var, levs_keep)
            did_change = True
        else:
            # keeping all var levs...
            pass

        # got this far, keep it
        vars_keep.append(var)

    ############### REMOVE SPECIFIC CONJUCNTIONS THAT DONT HAVE ENOHGH DATA
    # Remove conj groups with not enough data.
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
    df = PA.Xlabels["trials"]
    groupdict = grouping_append_and_return_inner_items(df, vars_keep)
    rows_remove = []
    for grp, inds in groupdict.items():
        if len(inds)<n_min_per_conj_var:
            # Then remove these rows
            rows_remove.extend(inds)
    if len(rows_remove)>0:
        print("[CHECK CONJ] Removing these rows (inds):", rows_remove)
        rows_keep = [i for i in range(len(df)) if i not in rows_remove]
        PA = PA.slice_by_dim_indices_wrapper("trials", rows_keep)
    else:
        print("[CHECK CONJ] NOTHING TO REMOVE!")

    ################# FINISH UP
    assert sort_mixed_type(vars_keep+vars_remove) == sort_mixed_type(EFFECT_VARS)

    # If any var is to be removed, then this means you would remove all rows. This means
    # you can't study these effects with this PA.
    if len(vars_remove)>0:
        did_change = True
        return None, res_check_before, None, vars_remove, reason_vars_remove, did_change
    else:
        # Assess how much data is available.
        res_check_after = _preprocess_pa_check_how_much_data(PA, EFFECT_VARS)

        # how many levs must be thrown out for each var?
        return PA, res_check_before, res_check_after, vars_remove, reason_vars_remove, did_change

def _preprocess_pa_check_how_much_data(PA, EFFECT_VARS):
    """ Ideally pass in the PA _before_ normalization, so that the
    rows here are trials.
    """
    from pythonlib.tools.pandastools import grouping_count_n_samples

    for var in EFFECT_VARS:
        if var not in PA.Xlabels["trials"].columns:
            print(var)
            print(PA.Xlabels["trials"].columns)
            assert False

    # - n levels for each var
    res = {}
    for var in EFFECT_VARS:
        var_levs = PA.Xlabels["trials"][var].unique().tolist()

        # for each level, how many levels of other var
        resthis = []
        for lev in var_levs:
            ######## COUNT N SAMPLES
            dfthis = PA.Xlabels["trials"][PA.Xlabels["trials"][var]==lev]

            if len(dfthis)==0:
                print(var, lev)
                print(PA.Xlabels["trials"][var].value_counts())
                assert False

            # Count n trials for this level
            # True if greater than <n_min_cases_per_lev> in this level.
            # n = len(dfthis) # num rows that has this level

            # Count n trials across each conjunction of othervars
            if len(EFFECT_VARS)>1:
                n_each_other_var = tuple(grouping_count_n_samples(dfthis, [v for v in EFFECT_VARS if not v==var]))
            else:
                n_each_other_var = []

            # min, median, and max
            # resthis[lev] = (
            #     len(dfthis), # num rows that has this level
            #     len(n_each_other_var), # num othervar conjunctive levels spanned
            #     min(n_each_other_var), # min n rows across other-levels
            #     int(np.median(n_each_other_var)),
            #     max(n_each_other_var)
            # )

            resthis.append({
                "var":var,
                "n_levs_this_var":len(var_levs),
                "lev":lev,
                "n_rows_this_lev":len(dfthis), # num rows that has this level
                })

            if len(n_each_other_var)>0:
                resthis[-1]["n_othervar_levs_spanned"] = len(n_each_other_var) # num othervar conjunctive levels spanned
                resthis[-1]["min_n_rows_across_othervar_levs"] = min(n_each_other_var) # min n rows across other-levels
                resthis[-1]["median_n_rows_across_othervar_levs"] = int(np.median(n_each_other_var))
                resthis[-1]["max_n_rows_across_othervar_levs"] = max(n_each_other_var)
                resthis[-1]["n_othervar_levs_with_only_one_row"] = sum([n==1 for n in n_each_other_var])
                resthis[-1]["n_rows_for_each_othervar_lev"] = n_each_other_var

        res[var] = pd.DataFrame(resthis)

    # also include the conj vars
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items
    groupdict = grouping_append_and_return_inner_items(PA.Xlabels["trials"], EFFECT_VARS)
    tmp = []
    for grp, inds in groupdict.items():
        tmp.append({
            "grp":grp,
            "n":len(inds)
        })
    res["CONJ_VARS"] = pd.DataFrame(tmp)

    return res


# def rsa_plot_raw_matrix(PAagg, grouping_vars, sort_order, version_distance="pearson",
#                         doplot=True):
#     """ Plot the input data (i.e. not sim matrix), which must be
#     already converted to scalar represntation, with options for sorting labels
#     to allow visualziation of interesting patterns
#     INPUT:
#     """
#     from pythonlib.tools.listtools import argsort_list_of_tuples
#     from pythonlib.cluster.clustclass import Clusters
#
#     # Pull out data in correct format, and return as clusters.
#     X = PAagg.X.squeeze().T # (ndat, nchans)
#     labels_rows = PAagg.Xlabels["trials"].loc[:, grouping_vars].values.tolist()
#     labels_rows = [tuple(x) for x in labels_rows] # list of tuples
#     labels_cols = PAagg.Chans # list of ints
#
#     # Sort labels if needed
#     # key =lambda x:(x[1], x[2], x[0])
#     if sort_order is not None:
#         key =lambda x:tuple([x[i] for i in sort_order])
#         inds_sort = argsort_list_of_tuples(labels_rows, key)
#     else:
#         inds_sort = list(range(len(labels_rows)))
#
#     labels_sorted = [labels_rows[i] for i in inds_sort]
#     Xsort = X[inds_sort] # sort the X also
#
#     Clraw = Clusters(Xsort, labels_sorted, labels_cols)
#
#     # Plot
#     if doplot:
#         figraw, X, labels_col, labels_row, ax = Clraw._plot_heatmap_data(Xsort, labels_sorted,
#                                                                    labels_cols, diverge=True,
#                                                                       rotation=90, rotation_y=0)
#     else:
#         figraw = None
#
#     #### Also plot the correlation matrix
#     if version_distance=="euclidian":
#         from scipy.spatial import distance_matrix
#         D = distance_matrix(Xsort, Xsort)
#         diverge = False
#
#         inds = np.where(~np.eye(D.shape[0],dtype=bool))
#         zmin = np.min(D[inds])
#         zlims = (zmin, None)
#     elif version_distance=="pearson":
#         # correlation matrix
#         D = np.corrcoef(Xsort)
#         diverge=True
#         zlims = (None, None)
#     else:
#         assert False
#
#     # plot heatmap
#     Clsim = Clusters(D, labels_sorted, labels_sorted)
#     if doplot:
#         figsim, X, labels_col, labels_row, ax = Clsim._plot_heatmap_data(D, labels_sorted,
#                                                                    labels_sorted, diverge=diverge,
#                                                                    zlims=zlims,
#                                                                       rotation=90, rotation_y=0)
#     else:
#         figsim = None
#
#     return Clraw, Clsim, figraw, figsim


########################## WORKING WITH PROCESSED DATA
############################################ AFTER SAVING PAs, Then run this.

def OBS_load_single_data(RES, bregion, twind, which_level, recompute_sim_mats=False):
    """ Helper to load a single dataset...
    :param
    - recompute_sim_mats, then uses the params in the loaded data to recompute the
    similarity matrix for neural data (i.e., Clsim), if it is not found in the saved
    data (e.g., was too large so excluded).
    """

    tmp = [res for res in RES if res["which_level"]==which_level]
    if len(tmp)!=1:
        print(tmp)
        assert False
    res = tmp[0]

    # Extract specifics
    key = (bregion, twind)
    PA = res["DictBregionTwindPA"][key]
    if "DictBregionTwindClraw" in res.keys():
        Clraw = res["DictBregionTwindClraw"][key]
    else:
        Clraw = None
    if "DictBregionTwindClsim" in res.keys():
        Clsim = res["DictBregionTwindClsim"][key]
    else:
        Clsim = None

    if Clsim is None and recompute_sim_mats:
        # Generate the Clsim for this data
        EFFECT_VARS = res["EFFECT_VARS"]
        subtract_mean_each_level_of_var = res["subtract_mean_each_level_of_var"]
        version_distance = res["version_distance"]
        DO_AGG_TRIALS = res["DO_AGG_TRIALS"]
        _, _, PAagg, fig, axes, groupdict = popanal_preprocess_scalar_normalization(PA, EFFECT_VARS,
                                                                                      subtract_mean_each_level_of_var,
                                                                                      plot_example_chan=None,
                                                                                      DO_AGG_TRIALS=DO_AGG_TRIALS)
        Clraw, Clsim = _rsagood_convert_PA_to_Cl(PAagg, EFFECT_VARS, version_distance, DO_AGG_TRIALS)

    return res, PA, Clraw, Clsim

def load_single_data_wrapper(animal, DATE, which_level, version_distance,
                             event):
    """ Helper to load the orginal data BEFORE splitting into questions
    I.e.,
    """

    DO_AGG_TRIALS = True #
    question = None
    RES, SAVEDIR_MULT, params, REGIONS_IN_ORDER = load_mult_data_helper(animal, DATE,
                                                                        version_distance,
                                                                        [which_level],
                                                                        question,
                                                                        DO_AGG_TRIALS,
                                                                        event=event)

    tmp = [res for res in RES if res["which_level"]==which_level]
    if len(tmp)!=1:
        print(tmp)
        assert False
    res = tmp[0]

    print(res.keys())
    # Extract specifics
    EFFECT_VARS = ["EFFECT_VARS"]
    list_twind = res["list_time_windows"]
    # res["subtract_mean_each_level_of_var"]
    DictBregionTwindPA = res["DictBregionTwindPA"]
    # DictEvBrTw_to_PA = res["DictEvBrTw_to_PA"]

    # PA = res["DictBregionTwindPA"][key]
    # Clraw = res["DictBregionTwindClraw"][key]
    # Clsim = res["DictBregionTwindClsim"][key]

    # sanity
    for bregion, twind in zip(REGIONS_IN_ORDER, list_twind):
        assert (bregion, twind) in DictBregionTwindPA.keys()

    assert len(DictBregionTwindPA)>0, "empty data!"
    return DictBregionTwindPA, EFFECT_VARS, list_twind, REGIONS_IN_ORDER


def OBS_load_mult_data_helper(animal, DATE, version_distance, list_which_level,
                          question, DO_AGG_TRIALS, event):
    """ Load all data across (i) timw widopws (ii) which levels
    RETURNS:
        - RES, list of dicts, each a single specific dataset, across bregions and
        time windows, for a specific Snippets (e.g., which_level), with the following
        keys:
        dict_keys(['version_distance', 'which_level', 'DFRES_SAMEDIFF', 'DFRES_THEOR', 'DictBregionTwindPA', 'DictBregionTwindClraw', 'DictBregionTwindClsim', 'EFFECT_VARS', 'list_time_windows', 'SAVEDIR', 'subtract_mean_each_level_of_var'])
    """

    if list_which_level is None:
        list_which_level = ["stroke", "stroke_off"]

    if DO_AGG_TRIALS:
        SAVEDIR_LOAD = "/gorilla1/analyses/recordings/main/RSA"
    else:
        SAVEDIR_LOAD = "/gorilla1/analyses/recordings/main/RSA_trial_datapt"

    params = {
        "DATE":DATE,
        "animal":animal,
        "list_which_level":list_which_level,
        "version_distance":version_distance,
        "SAVEDIR":SAVEDIR_LOAD,
        "question":question
    }

    def convert_dict_pa_to_only_bregion_twind_keys(DictEvBrTw_to_PA, event):
        """
        """
        # collect all that have this event
        DictBregionTwindPA = {}
        for key, pa in DictEvBrTw_to_PA.items():
            ev, br, tw = key
            if ev==event:
                bregion_twind = (br, tw)
                assert bregion_twind not in DictBregionTwindPA.keys()
                DictBregionTwindPA[bregion_twind] = pa
        return DictBregionTwindPA

    ####### LOAD
    RES = []
    for which_level in list_which_level:
        print("Getting: ", which_level)
        if question is None:
            # Then is not split by question
            savedir = f"{SAVEDIR_LOAD}/{animal}/{DATE}/{which_level}/{version_distance}"
            SAVEDIR_MULT = f"{SAVEDIR_LOAD}/{animal}/MULT/{DATE}/{version_distance}"
        else:
            # split
            savedir = f"{SAVEDIR_LOAD}/{animal}/SPLIT_BY_QUESTIONS/{DATE}/{which_level}/{version_distance}/{question}"
            SAVEDIR_MULT = f"{SAVEDIR_LOAD}/{animal}/SPLIT_BY_QUESTIONS/MULT/{DATE}/{version_distance}/{question}"

        path = f"{savedir}/resthis.pkl"
        print("Loading res from: ", path)
        try:
            with open(path, "rb") as f:
                res = pickle.load(f)

            # Clean up sometig
            if "DictEvBrTw_to_PA" in res.keys():
                assert event is not None, "to pull out specific evnet"
                # HACKy, convret it to DictBregionTwindPA
                DictBregionTwindPA = convert_dict_pa_to_only_bregion_twind_keys(
                    res["DictEvBrTw_to_PA"], event)
                res["DictBregionTwindPA"] = DictBregionTwindPA
            # if "DictBregionTwindPA" in res.keys():
            #     assert "DictEvBrTw_to_PA" not in res.keys()
            #     DictEvBrTw_to_PA = []
            #     for k, pa in res["DictBregionTwindPA"].items():
            #         k = tuple([f"00_{which_level}", k[0], k[1]])
            #         assert k not in DictEvBrTw_to_PA.keys(), "i assume only one event/wl..."
            #         DictEvBrTw_to_PA[k] = pa
            #     res["DictEvBrTw_to_PA"] = DictEvBrTw_to_PA

            RES.append(res)
        except FileNotFoundError as err:
            print(path)
            print("Couldnt load this data! *******************", version_distance, animal, DATE, which_level)
            assert False

    import os
    os.makedirs(SAVEDIR_MULT, exist_ok=True)
    path = f"{SAVEDIR_MULT}/params.yaml"
    writeDictToYaml(params, path)
    # bregions
    from neuralmonkey.neuralplots.brainschematic import REGIONS_IN_ORDER

    return RES, SAVEDIR_MULT, params, REGIONS_IN_ORDER


def OBS_pipeline_rsa_scalar_population_MULT(animal, DATE, version_distance, yvar, list_which_level,
                                        question=None, DO_AGG_TRIALS=True, event=None):
    """ Wrapper for pipeline_rsa_scalar_population_MULT_PLOTS...
    Load data saved in pipeline_rsa_scalar_population acrooss
    different (time windows, bregions, effects, which_levels[from SP extraction],
    version_distance, ...) and make single plots combining all of them
    """

    ########## LOAD ALL DATA
    RES, SAVEDIR_MULT, params, REGIONS_IN_ORDER = load_mult_data_helper(animal, DATE, version_distance,
                                                                        list_which_level=list_which_level,
                                                                        question=question,
                                                                        DO_AGG_TRIALS=DO_AGG_TRIALS,
                                                                        event=event)

    ## COLLECT
    # Compare to theoetical simmat
    list_df = []
    # list_which_level = []
    for res in RES:
        which_level = res["which_level"]
        version_distance = res["version_distance"]
        df = res["DFRES_THEOR"]
        df["which_level"] = which_level
        df["version_distance"] = version_distance
        list_df.append(df)
    #     list_which_level.append(which_level)
    # list_which_level = sorted(set(list_which_level))
    DFMULT_THEOR = pd.concat(list_df).reset_index(drop=True)

    return rsagood_pa_vs_theor_plot_results(
        DFMULT_THEOR, SAVEDIR_MULT, yvar)
    # DFMULT_THEOR, DictBregionToDf2d, DictVarToDf2d, dfres_kernels_2d, PARAMS

def rsagood_pa_effectsize_plot_summary(DFRES_THEOR, DFRES_EFFECT_MARG, DFRES_EFFECT_CONJ,
                                       Params, yvar="cc"):
    """
    Plot summary plots of effect sizes (i/e, norm of fr vector), and also its realtion to
    rsa (i.e,, pop activity vs. theoretical for each var).
    PARAMS:
    - inputs are outputs from rsagood_pa_vs_theor_wrapper
    :return:
    - Saves figures...
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.plottools import savefig
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    DFRES_EFFECT_CONJ = append_col_with_grp_index(DFRES_EFFECT_CONJ, ["which_level", "event"], "wl_ev", use_strings=False)
    DFRES_EFFECT_MARG = append_col_with_grp_index(DFRES_EFFECT_MARG, ["which_level", "event"], "wl_ev", use_strings=False)
    DFRES_EFFECT_MARG = append_col_with_grp_index(DFRES_EFFECT_MARG, ["var", "lev"], "var_lev", use_strings=False)
    list_wl = DFRES_EFFECT_CONJ["which_level"].unique().tolist()
    list_ev = DFRES_EFFECT_CONJ["event"].unique().tolist()
    list_tw = DFRES_EFFECT_CONJ["twind"].unique().tolist()
    question_params = Params["question_params"]
    effect_vars = question_params["effect_vars"]

    SAVEDIR = Params["savedir"]
    assert SAVEDIR is not None
    savedir = f"{SAVEDIR}/effect_size"
    os.makedirs(savedir, exist_ok=True)
    print("[rsagood_pa_effectsize_plot_summary] Saving figures at: ", savedir)

    ################ HEATMAPS, MARGINAL LEVELS
    for wl in list_wl:
        for ev in list_ev:
            b = DFRES_EFFECT_MARG["event"]==ev
            c = DFRES_EFFECT_MARG["which_level"]==wl
            dfthis = DFRES_EFFECT_MARG[b & c]

            fig, axes = plot_subplots_heatmap(dfthis, "bregion", "twind", "norm", "var_lev", share_zlim=True)
            savefig(fig, f"{savedir}/heatmaps-MARG-bregion_vs_twind-splots_var_lev-{wl}-{ev}.pdf")

            fig, axes = plot_subplots_heatmap(dfthis, "bregion", "twind", "norm", "var", share_zlim=True)
            savefig(fig, f"{savedir}/heatmaps-MARG-bregion_vs_twind-splots_var-{wl}-{ev}.pdf")

            fig, axes = plot_subplots_heatmap(dfthis, "bregion", "var_lev", "norm", "twind", share_zlim=True)
            savefig(fig, f"{savedir}/heatmaps-MARG-bregion_vs_var_lev-splots_twind-{wl}-{ev}.pdf")

            plt.close("all")

    fig, axes = plot_subplots_heatmap(DFRES_EFFECT_MARG, "bregion", "twind", "norm", "wl_ev", share_zlim=True)
    savefig(fig, f"{savedir}/heatmaps-MARG-bregion_vs_twind-splots_wl_ev.pdf")
    plt.close("all")

    ################ HEATMAPS, CONJ LEVELS
    for i in range(len(effect_vars)-1):
        var1 = effect_vars[i]
        var2 = effect_vars[i+1]
        for share_zlim in [False, True]:
            fig, axes = plot_subplots_heatmap(DFRES_EFFECT_CONJ, var1, var2, "norm", "bregion",
                                              share_zlim=share_zlim)
            savefig(fig, f"{savedir}/heatmaps-CONJ-{var1}_vs_{var2}-splots_bregion-sharez_{share_zlim}.pdf")
        plt.close("all")

    for wl in list_wl:
        for ev in list_ev:
            b = DFRES_EFFECT_CONJ["event"]==ev
            c = DFRES_EFFECT_CONJ["which_level"]==wl
            dfthis = DFRES_EFFECT_CONJ[b & c]

            for var in effect_vars:
                fig, axes = plot_subplots_heatmap(dfthis, "bregion", "twind", "norm", var,
                                                  share_zlim=True)
                savefig(fig, f"{savedir}/heatmaps-CONJ-bregion_vs_twind-splots_{var}-{wl}-{ev}.pdf")
            plt.close("all")
    fig, axes = plot_subplots_heatmap(DFRES_EFFECT_CONJ, "bregion", "twind",
                                      "norm", "wl_ev", share_zlim=True)
    savefig(fig, f"{savedir}/heatmaps-CONJ-bregion_vs_twind-splots_wl_ev.pdf")
    plt.close("all")

    ######################### Scatter plot, rsa vs. effect size
    # Make temporary combined df for scatter plotting
    dftheor = DFRES_THEOR.copy()
    dftheor["metric"] = "rsa"
    dftheor["value"] = dftheor[yvar]
    for ver in ["effect_for_each_var", "effect_global"]:
        # Different ways of defining effect sizes.
        if ver=="effect_for_each_var":
            # Each var, using different effect sizes, based on marginalizing over othervar BEFORE computing efect size.
            dfeff = DFRES_EFFECT_MARG.copy()
            dfeff["metric"] = "effect_size"
            dfeff["value"] = dfeff["norm"]
            list_df = [dftheor, dfeff]
        elif ver=="effect_global":
            # using mean over all conj for effect sizes (same for all var)
            list_df = [dftheor]
            for var in effect_vars:
                dfeff = DFRES_EFFECT_CONJ.loc[:, ["bregion", "twind", "event", "which_level", "norm"]].groupby(["bregion", "twind", "event", "which_level"], as_index=False).mean().reset_index(drop=True)
                dfeff["metric"] = "effect_size"
                dfeff["value"] = dfeff["norm"]
                dfeff["var"] = var
                list_df.append(dfeff)
        else:
            assert False
        dfthis = pd.concat(list_df)

        # Make plots
        for wl in list_wl:
            for ev in list_ev:
                for tw in list_tw:
                    a = dfthis["which_level"] == wl
                    b = dfthis["event"] == ev
                    c = dfthis["twind"] == tw
                    dfthisthis = dfthis[a & b & c]
                    dfres, fig = plot_45scatter_means_flexible_grouping(dfthisthis, "metric",
                                                                        "effect_size", "rsa",
                                                                        "var", "value",
                                                                        "bregion", shareaxes=True, alpha=0.6,
                                                                        SIZE=5)
                    savefig(fig, f"{savedir}/scatter-rsa_vs_effectsize-{ver}-{wl}_{ev}_{tw}.pdf")
                    plt.close("all")


def rsagood_pa_vs_theor_plot_pairwise_distmats(DFallpa, version_distance, SAVEDIR, variables_plot,
                                               list_wl_ev_tw_plot, vars_test_invariance_over_dict,
                                               DO_AGG_TRIALS_PLOT=True):
    """
    Plot pairwise distance matrices and raw activity data.
    PARAMS
    - DFallpa, holds each pa
    - Params, params, for the DFallpa
    - SAVEDIR, base
    - variables_plot, list of str, will plot all pairs from this
    - list_wl_ev_tw_plot, list of tuples, each a (wl, ev, twind). Will make plots froe ach of
    these tuples
    - DO_AGG_TRIALS_PLOT, bool
    NOTE: load the inputs:
    DFRES_THEOR, DFallpa, Params, savedir = rsagood_pa_vs_theor_wrapper_loadresults(animal, date, question, version_distance, DO_AGG_TRIALS, subtract_mean_each_level_of_var)
    """
    savedir = f"{SAVEDIR}/replotting_pairwise_vars"
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    # for wl in Params["list_which_level"]:
    #     for ev in Params["list_event"]:
    #             # for tw in Params["list_twind"]:
    #             for tw in twinds_plot:

    list_bregion = DFallpa["bregion"].unique().tolist()

    DictVarToClsimtheor = {}
    for wl, ev, tw in list_wl_ev_tw_plot:
        # Norm of activity for each level of the variable. Then average these norms over the levels.
        for ivar, var1 in enumerate(variables_plot):
            for jvar, var2 in enumerate(variables_plot):
                if jvar>ivar:
                    for br in list_bregion:

                        # Slice this df
                        a = DFallpa["which_level"]==wl
                        b = DFallpa["event"]==ev
                        c = DFallpa["bregion"]==br
                        d = DFallpa["twind"]==tw
                        dfthis = DFallpa[a & b & c & d]
                        if len(dfthis)!=1:
                            print(len(dfthis))
                            print(dfthis["which_level"].value_counts())
                            print(dfthis["event"].value_counts())
                            print(dfthis["twind"].value_counts())
                            print(dfthis["bregion"].value_counts())
                            print(sum(a))
                            print(sum(b))
                            print(sum(c))
                            print(sum(d))
                            print(wl, ev, tw, br)
                            assert False
                        pa = dfthis["pa"].values[0]
                        print(wl, ev, br, tw)

                        for subtrmean in [None, var1, var2]:
                            sdir = f"{savedir}/{wl}-{ev}-{br}-{tw}/{var1}--{var2}--subtr_{subtrmean}"
                            os.makedirs(sdir, exist_ok=True)

                            # Extract raw data
                            _, _, Clraw, _, PAagg, DictVarToClsimtheor = rsagood_pa_vs_theor_single(pa, [var1, var2],
                                                                                                    version_distance,
                                                                                                    subtrmean,
                                                                                                    vars_test_invariance_over_dict=vars_test_invariance_over_dict,
                                                                                                    PLOT=True,
                                                                                                    sdir=sdir,
                                                                                                    PLOT_THEORETICAL_SIMMATS=True,
                                                                                                    COMPUTE_VS_THEOR_MAT=True,
                                                                                                    DO_AGG_TRIALS=DO_AGG_TRIALS_PLOT,
                                                                                                    DictVarToClsimtheor=DictVarToClsimtheor)

                            plt.close("all")

def rsagood_pa_vs_theor_plot_pairwise_distmats_singlevar(DFallpa, version_distance, SAVEDIR, var_effect,
                                               list_wl_ev_tw_plot, vars_test_invariance_over_dict,
                                               DO_AGG_TRIALS_PLOT=True):
    """
    Plot pairwise distance matrices and raw activity data. This works
    for cases with just a single var (e.g., shape), which doesnt work for other
    which requires 2 vars.
    PARAMS
    - DFallpa, holds each pa
    - Params, params, for the DFallpa
    - SAVEDIR, base
    - variables_plot, list of str, will plot all pairs from this
    - list_wl_ev_tw_plot, list of tuples, each a (wl, ev, twind). Will make plots froe ach of
    these tuples
    - DO_AGG_TRIALS_PLOT, bool
    NOTE: load the inputs:
    DFRES_THEOR, DFallpa, Params, savedir = rsagood_pa_vs_theor_wrapper_loadresults(animal, date, question, version_distance, DO_AGG_TRIALS, subtract_mean_each_level_of_var)
    """
    savedir = f"{SAVEDIR}/replotting_pairwise_vars"
    os.makedirs(savedir, exist_ok=True)
    print(savedir)

    # for wl in Params["list_which_level"]:
    #     for ev in Params["list_event"]:
    #             # for tw in Params["list_twind"]:
    #             for tw in twinds_plot:

    list_bregion = DFallpa["bregion"].unique().tolist()

    DictVarToClsimtheor = {}
    for wl, ev, tw in list_wl_ev_tw_plot:
        # Norm of activity for each level of the variable. Then average these norms over the levels.
        for br in list_bregion:

            # Slice this df
            a = DFallpa["which_level"]==wl
            b = DFallpa["event"]==ev
            c = DFallpa["bregion"]==br
            d = DFallpa["twind"]==tw
            dfthis = DFallpa[a & b & c & d]
            if len(dfthis)!=1:
                print(len(dfthis))
                print(dfthis["which_level"].value_counts())
                print(dfthis["event"].value_counts())
                print(dfthis["twind"].value_counts())
                print(dfthis["bregion"].value_counts())
                print(sum(a))
                print(sum(b))
                print(sum(c))
                print(sum(d))
                print(wl, ev, tw, br)
                assert False
            pa = dfthis["pa"].values[0]
            print(wl, ev, br, tw)

            for subtrmean in [None]:
                sdir = f"{savedir}/{wl}-{ev}-{br}-{tw}/{var_effect}"
                os.makedirs(sdir, exist_ok=True)

                # Extract raw data
                _, _, Clraw, _, PAagg, DictVarToClsimtheor = rsagood_pa_vs_theor_single(pa, [var_effect],
                                                                                        version_distance,
                                                                                        subtrmean,
                                                                                        vars_test_invariance_over_dict=vars_test_invariance_over_dict,
                                                                                        PLOT=True,
                                                                                        sdir=sdir,
                                                                                        PLOT_THEORETICAL_SIMMATS=True,
                                                                                        COMPUTE_VS_THEOR_MAT=True,
                                                                                        DO_AGG_TRIALS=DO_AGG_TRIALS_PLOT,
                                                                                        DictVarToClsimtheor=DictVarToClsimtheor)

                plt.close("all")

def rsagood_pa_vs_theor_samecontextctrl(DFMULT_THEOR, SAVEDIR_MULT):
    """
    :param DFMULT_THEOR:
    :param SAVEDIR_MULT:
    :return:
    """
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    # assert "cc_same_context" in DFMULT_THEOR.columns
    # assert "EffD_CtxS" in DFMULT_THEOR.columns

    DFMULT_THEOR["twind_str"] = ["_to_".join([str(tt) for tt in t]) for t in DFMULT_THEOR["twind"].tolist()]
    DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["which_level", "twind_str"], "wl_tw", strings_compact=True)
    DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["which_level", "event"], "wl_ev", strings_compact=True)

    # Derived metrics
    DFMULT_THEOR["score_CtxS"] = DFMULT_THEOR["EffD_CtxS"] - DFMULT_THEOR["EffS_CtxS"]
    DFMULT_THEOR["score_CtxD"] = DFMULT_THEOR["EffD_CtxD"] - DFMULT_THEOR["EffS_CtxD"]
    # DFMULT_THEOR["score_ratio"] = DFMULT_THEOR["score_CtxD"]/DFMULT_THEOR["score_CtxS"]

    # First, plot all the normal plots
    rsagood_pa_vs_theor_plot_results(DFMULT_THEOR, SAVEDIR_MULT, yvar="cc_same_context")
    rsagood_pa_vs_theor_plot_results(DFMULT_THEOR, SAVEDIR_MULT, yvar="score_CtxS")
    rsagood_pa_vs_theor_plot_results(DFMULT_THEOR, SAVEDIR_MULT, yvar="score_CtxD")
    # rsagood_pa_vs_theor_plot_results(DFMULT_THEOR, SAVEDIR_MULT, yvar="score_ratio")

    ################
    savedir = f"{SAVEDIR_MULT}/samecontextctrl_OLD_cc"
    os.makedirs(savedir, exist_ok=True)

    # Second, compare cc (diff ) vs same contxt.
    # - Stack, to use scatterplot.
    DFRES_THEOR_1 = DFMULT_THEOR.copy()
    DFRES_THEOR_2 = DFMULT_THEOR.copy()
    DFRES_THEOR_1["cc_ver"] = "diff_ctxt"
    DFRES_THEOR_2["cc_ver"] = "same_ctxt"
    DFRES_THEOR_2["cc"] = DFRES_THEOR_2["cc_same_context"]
    DFRES_THEOR_SCAT = pd.concat([DFRES_THEOR_1, DFRES_THEOR_2]).reset_index(drop=True)

    # Make plots
    list_wl = DFRES_THEOR_SCAT["which_level"].unique().tolist()
    list_ev = DFRES_THEOR_SCAT["event"].unique().tolist()
    list_tw = DFRES_THEOR_SCAT["twind"].unique().tolist()
    for wl in list_wl:
        for ev in list_ev:
            for tw in list_tw:
                a = DFRES_THEOR_SCAT["which_level"] == wl
                b = DFRES_THEOR_SCAT["event"] == ev
                c = DFRES_THEOR_SCAT["twind"] == tw
                dfthisthis = DFRES_THEOR_SCAT[a & b & c]
                dfres, fig = plot_45scatter_means_flexible_grouping(dfthisthis, "cc_ver",
                                                                    "same_ctxt", "diff_ctxt",
                                                                    "var", "cc",
                                                                    "bregion", shareaxes=True, alpha=0.6,
                                                                    SIZE=5)
                savefig(fig, f"{savedir}/scatter-rsa_vs_effectsize-{wl}_{ev}_{tw}.pdf")

    ################
    savedir = f"{SAVEDIR_MULT}/samediff_bycontext_NEW"
    os.makedirs(savedir, exist_ok=True)

    try:
        for yvar in ["EffD_CtxS", "EffS_CtxS", "EffD_CtxD", "EffS_CtxD"]:
            fig = sns.catplot(data=DFMULT_THEOR, x="twind", y=yvar, col="bregion", hue="var",
                              kind="point", row="wl_ev")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/pointplot-{yvar}-bregions.pdf")
            plt.close("all")
    except Exception as err:
        fig, ax = plt.subplots()
        savefig(fig, f"{savedir}/FAILED.pdf")
        plt.close("all")

    # Second, compare cc (diff ) vs same contxt.
    # - Stack, to use scatterplot.
    DFRES_THEOR_1 = DFMULT_THEOR.copy()
    DFRES_THEOR_2 = DFMULT_THEOR.copy()

    DFRES_THEOR_1["score_ver"] = "diff_ctxt"
    DFRES_THEOR_1["score"] = DFRES_THEOR_1["score_CtxD"]

    DFRES_THEOR_2["score_ver"] = "same_ctxt"
    DFRES_THEOR_2["score"] = DFRES_THEOR_2["score_CtxS"]

    DFRES_THEOR_SCAT = pd.concat([DFRES_THEOR_1, DFRES_THEOR_2]).reset_index(drop=True)

    # Make plots
    list_wl = DFRES_THEOR_SCAT["which_level"].unique().tolist()
    list_ev = DFRES_THEOR_SCAT["event"].unique().tolist()
    list_tw = DFRES_THEOR_SCAT["twind"].unique().tolist()
    for wl in list_wl:
        for ev in list_ev:
            for tw in list_tw:
                a = DFRES_THEOR_SCAT["which_level"] == wl
                b = DFRES_THEOR_SCAT["event"] == ev
                c = DFRES_THEOR_SCAT["twind"] == tw
                dfthisthis = DFRES_THEOR_SCAT[a & b & c]
                dfres, fig = plot_45scatter_means_flexible_grouping(dfthisthis, "score_ver",
                                                                    "same_ctxt", "diff_ctxt",
                                                                    "var", "score",
                                                                    "bregion", shareaxes=True, alpha=0.6,
                                                                    SIZE=5)
                savefig(fig, f"{savedir}/scatter-rsa_vs_effectsize-{wl}_{ev}_{tw}.pdf")


# OBS_pipeline_rsa_scalar_population_MULT_PLOTS
def rsagood_pa_vs_theor_plot_results(DFMULT_THEOR, SAVEDIR_MULT, yvar="cc"):
    """ [GOOD - final plots] Given already computed correaltions between
    data distmat and theoretical distmats (DFMULT_THEOR), make all plots of
    effects.
    """
    from neuralmonkey.neuralplots.brainschematic import plot_df_from_longform, plot_df_from_wideform

    list_which_level = sorted(DFMULT_THEOR["which_level"].unique().tolist())
    list_event = sorted(DFMULT_THEOR["event"].unique().tolist())

    ##### Preprocess
    DFMULT_THEOR["twind_str"] = ["_to_".join([str(tt) for tt in t]) for t in DFMULT_THEOR["twind"].tolist()]
    DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["which_level", "twind_str"], "wl_tw", strings_compact=True)
    DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["which_level", "event"], "wl_ev", strings_compact=True)
    EFFECT_VARS = DFMULT_THEOR["var"].unique().tolist()
    list_bregion = DFMULT_THEOR["bregion"].unique().tolist()

    ##### Plots
    savedir = f"{SAVEDIR_MULT}/overview"
    os.makedirs(savedir, exist_ok=True)

    if False: # too busy, I never look at it
        assert False, "takes too long - split into diff events"
        # (1) Pointplot, showing all results, but hard to read.
        fig = sns.catplot(data=DFMULT_THEOR, x="twind", y=yvar, col="bregion", hue="var",
                          kind="point", row="wl_ev")
        rotateLabel(fig)
        savefig(fig, f"{savedir}/pointplot-{yvar}-bregions.pdf")
        plt.close("all")

    # IN PROGRESS - subtracting global mean within each level of (effect var). Decided it wasnt needed
    # conjucntion of twind and which_level
    if False:
        from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping
        # datamod_normalize_row_after_grouping(DFMULT_THEOR, "bregion",
        grp = ["bregion"]
        DFMULT_THEOR.groupby(grp).transform(lambda x: (x - x.mean()) / x.std())

        # Normalize by subtracting mean effect within each bregion, to allow comparison of each bregion's "signature"
        from pythonlib.tools.pandastools import aggregGeneral
        aggregGeneral(DFMULT_THEOR, ["var" , ""])

    if False:
        # IN PROGRESS - ONE VECTOR FOR EACH BREGION (across var, which_level, and time window).
        # new variable, conjunction of var and time window
        DFMULT_THEOR = append_col_with_grp_index(DFMULT_THEOR, ["var", "wl_tw"], "var_wl_tw", strings_compact=True)

        # Heatmap
        ncols = 3
        W = 4
        H = 4
        nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
        dfthis = DFMULT_THEOR
        for norm_method in [None, "row_sub", "col_sub"]:
            _, fig, _, _ = convert_to_2d_dataframe(dfthis, "bregion", "var_wl_tw", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                    norm_method=norm_method)
            savefig(fig, f"{savedir}/heatmap-")


    ########################################### HEATMAPS
    for ev in list_event:
        DFMULT_THEOR_THIS = DFMULT_THEOR[DFMULT_THEOR["event"]==ev]

        # (2) Heatmaps, easier to parse (Concatting all which_levels)
        W = 4
        H = 4
        ncols = 3
        # for norm_method in [None, "row_sub", "col_sub"]:
        for norm_method in [None]:
            # Heatmap - one subplot for each var
            nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
            for var, ax in zip(EFFECT_VARS, axes.flatten()):
                print(var)
                dfthis = DFMULT_THEOR_THIS[DFMULT_THEOR_THIS["var"]==var]

                convert_to_2d_dataframe(dfthis, "bregion", "wl_tw", True,
                                        "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                        norm_method=norm_method, ax=ax)
                ax.set_title(var)
            savefig(fig, f"{savedir}/heatmap-subplot_by_var-norm_{norm_method}-ccvar_{yvar}-ev_{ev}.pdf")

            # Heatmap - one subplot for each bregion
            nrows = int(np.ceil(len(list_bregion)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
            for bregion, ax in zip(list_bregion, axes.flatten()):
                print(norm_method, bregion)
                dfthis = DFMULT_THEOR_THIS[DFMULT_THEOR_THIS["bregion"]==bregion]
                convert_to_2d_dataframe(dfthis, "var", "wl_tw", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                        norm_method=norm_method, ax=ax)
                ax.set_title(bregion)
            savefig(fig, f"{savedir}/heatmap_concat-subplot_by_bregion-norm_{norm_method}-ccvar_{yvar}-ev_{ev}.pdf")

            plt.close("all")

    # (3) Heatmaps, easier to parse (Separate plots for each which_levels)
    if False: # Skip, not really needed, and takes a while
        W = 4
        H = 4
        ncols = 3
        for which_level in list_which_level:
            for norm_method in [None, "all_sub", "row_sub", "col_sub"]:
                # Heatmap - one subplot for each var
                nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
                for var, ax in zip(EFFECT_VARS, axes.flatten()):
                    print(var)
                    dfthis = DFMULT_THEOR[(DFMULT_THEOR["var"]==var) & (DFMULT_THEOR["which_level"]==which_level)]
                    convert_to_2d_dataframe(dfthis, "bregion", "twind", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                            norm_method=norm_method, ax=ax)
                    ax.set_title(var)
                savefig(fig, f"{savedir}/heatmap_whichlevel_{which_level}-subplot_by_var-norm_{norm_method}-ccvar_{yvar}.pdf")


                # Heatmap - one subplot for each bregion
                nrows = int(np.ceil(len(list_bregion)/ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*W, nrows*H))
                for bregion, ax in zip(list_bregion, axes.flatten()):
                    print(norm_method, bregion)
                    dfthis = DFMULT_THEOR[(DFMULT_THEOR["bregion"]==bregion) & (DFMULT_THEOR["which_level"]==which_level)]
                    convert_to_2d_dataframe(dfthis, "var", "twind", True, "mean", yvar, annotate_heatmap=False, dosort_colnames=False,
                                            norm_method=norm_method, ax=ax)
                    ax.set_title(bregion)
                savefig(fig, f"{savedir}/heatmap_whichlevel_{which_level}-subplot_by_bregion-norm_{norm_method}-ccvar_{yvar}.pdf")

                plt.close("all")

    ############# Brain schematics
    savedir = f"{SAVEDIR_MULT}/brain_schematic"
    os.makedirs(savedir, exist_ok=True)
    # for norm_method in [None, "col_sub", "row_sub"]:
    for norm_method in [None]:
        for ev in list_event:
            DFMULT_THEOR_THIS = DFMULT_THEOR[DFMULT_THEOR["event"]==ev]

            # Over time, plot for a given variable
            sdir = f"{savedir}/split_by_var/"
            os.makedirs(sdir, exist_ok=True)
            for var in EFFECT_VARS:
                dfthis = DFMULT_THEOR_THIS[DFMULT_THEOR_THIS["var"]==var]
                suff = f"ev_{ev}-var_{var}-yvar_{yvar}"
                plot_df_from_longform(dfthis, yvar, "wl_tw", savedir = sdir, savesuffix=suff, norm_method=norm_method)
            plt.close("all")

            sdir = f"{savedir}/split_by_which_level_twind"
            os.makedirs(sdir, exist_ok=True)
            for wl_tw in DFMULT_THEOR_THIS["wl_tw"].unique().tolist():
                dfthis = DFMULT_THEOR_THIS[DFMULT_THEOR_THIS["wl_tw"]==wl_tw]
                suff = f"ev_{ev}-wltw_{wl_tw}-yvar_{yvar}"
                plot_df_from_longform(dfthis, yvar, "var", savedir = sdir, savesuffix=suff, norm_method=norm_method)
            plt.close("all")

    ################################################
    ##################### Kernels - each bregions "signature" by concatting across all times and var
    DictBregionToDf2d = None
    DictVarToDf2d = None
    dfres_kernels_2d = None
    KERNELS = None

    if len(list_event)==1:
        # Otherwise kernel;s are not yet defined across events, beyond stroke aligned to onset or offset

        savedir = f"{SAVEDIR_MULT}/kernels"
        os.makedirs(savedir, exist_ok=True)

        def kernel_convert_from_combined_to_split(kernel_combined):
            """ Convert from list of tuple (event string, weight) to
            two lists; events and weights.
            """
            kernel = [k[0] for k in kernel_combined]
            weights = [k[1] for k in kernel_combined]

            return kernel, weights

        def kernel_extract_which_var(kernel_name):

            map_name_to_var = {
                "reach_prev_durstk":"gap_from_prev_angle_binned",
                "reach_next_durstk":"gap_to_next_angle_binned",
                "reach_prev":"gap_from_prev_angle_binned",
                "reach_next":"gap_to_next_angle_binned",
                "stkidx_stroke":"stroke_index_fromlast_tskstks",
                "shape":"shape_oriented",
                "loc_future":"CTXT_loc_next",
                "shape_future":"CTXT_shape_next",
                "angle_future":"gap_to_next_angle_binned",
                "stkidx_entire":"stroke_index_fromlast_tskstks"}

            if kernel_name not in map_name_to_var.keys():
                # The name is the var
                return kernel_name
            else:
                return map_name_to_var[kernel_name]

        def _check_twind_center_within(twind, tmin, tmax):
            """ Return True if the center of the window (twind)
            is greater than tmin and les sthan tmax"""
            cen = np.mean(twind)
            return cen>tmin and cen<tmax

        # Construct kernels for each (i.e., events in order and weights in order)
        list_twind = DFMULT_THEOR["twind"].unique().tolist()

        kernel_reach_prev = []
        kernel_reach_next = []
        kernel_reach_prev_durstk = []
        kernel_reach_next_durstk = []
        kernel_gridloc = []
        kernel_shape = []
        kernel_future_loc = []
        kernel_future_angle = []
        kernel_future_shape = []
        kernel_stkidx_during = []
        kernel_taskkind = []
        # kernel_stkidx_entire = []
        for wl in list_which_level:
            for twind in list_twind:
                ev = f"{wl}|{twind[0]}_to_{twind[1]}"

                ########### DURING GAPS (related to the gap -- i.e., motor)
                # Previous gap (it's delayed back, since that is causal to the gap)
                if wl=="stroke" and _check_twind_center_within(twind, -0.5, -0.25):
                    # then window ends before align time
                    k = 1
                else:
                    k = 0
                kernel_reach_prev.append((ev, k))

                # Next gap (it's delayed back, since that is causal to the gap)
                if wl=="stroke_off" and _check_twind_center_within(twind, 0.1, 0.3):
                    # then window ends before align time
                    k = 1
                else:
                    k = 0
                kernel_reach_next.append((ev, k))

                ############# CURRENT STROKE
                # vars related to current stroke, during current stroke.
                if wl=="stroke" and _check_twind_center_within(twind, -0.1, 0.3):
                    # then window ends before align time
                    k = 1
                else:
                    k = 0
                kernel_gridloc.append((ev, k))
                kernel_shape.append((ev, k))
                kernel_stkidx_during.append((ev, k))
                kernel_taskkind.append((ev, k))
                kernel_reach_prev_durstk.append((ev, k))
                kernel_reach_next_durstk.append((ev, k))

                ############ FUTURE (during current stroke)
                # predicting future, aligned to current stroke.
                if wl=="stroke_off" and _check_twind_center_within(twind, -0.5, -0.1):
                    # then window ends before align time
                    k = 1
                else:
                    k = 0
                kernel_future_loc.append((ev, k))
                kernel_future_shape.append((ev, k))
                kernel_future_angle.append((ev, k))

                # Stroke index (entire)
                k=1
                # kernel_stkidx_entire.append((ev, k))

        KERNELS = {
            "reach_prev":kernel_reach_prev,
            "reach_next":kernel_reach_next,
            "reach_prev_durstk":kernel_reach_prev_durstk,
            "reach_next_durstk":kernel_reach_next_durstk,
            "gridloc":kernel_gridloc,
            "stkidx_stroke":kernel_stkidx_during,
            "shape":kernel_shape,
            "task_kind":kernel_taskkind,
            "loc_future":kernel_future_loc,
            "shape_future":kernel_future_shape,
            "angle_future":kernel_future_angle,
            # "stkidx_entire":kernel_stkidx_entire,
        }

        ############### Plot kernels
        # Plot kernel templates
        fig, ax = plt.subplots()

        for i, (name, kernel) in enumerate(KERNELS.items()):
            k, w= kernel_convert_from_combined_to_split(kernel)

            var = kernel_extract_which_var(name)

            ax.scatter(k, np.ones(len(k))*i, c=[1-x for x in w], alpha=0.5, label=name)
            ax.text(0, i, var, color="r", alpha=0.5)

        ax.set_yticks(list(range(len(KERNELS.keys()))), labels=KERNELS.keys())
        ax.set_ylabel("kernel name")
        ax.set_xlabel("time window")
        rotate_x_labels(ax, 90)
        rotate_y_labels(ax, 0)
        ax.set_title("Kernels (dark dot = 1); red: var it operates on")
        savefig(fig, f"{savedir}/kernels_weights.pdf")

        # PREPROCESS - # for each variable, get 2d df (bregion x twinds)
        from neuralmonkey.analyses.event_temporal_modulation import kernel_compute, _kernel_compute_scores
        DictBregionToDf2d = {}
        for bregion in list_bregion:
            dfthis = DFMULT_THEOR[DFMULT_THEOR["bregion"]==bregion]
            dftmp, fig, ax, rgba_values = convert_to_2d_dataframe(dfthis, "var", "wl_tw", False, "mean", yvar, annotate_heatmap=False, dosort_colnames=False)
            DictBregionToDf2d[bregion] = dftmp
        DictVarToDf2d = {}
        for var in EFFECT_VARS:
            dfthis = DFMULT_THEOR[DFMULT_THEOR["var"]==var]
            dftmp, fig, ax, rgba_values = convert_to_2d_dataframe(dfthis, "bregion", "wl_tw", False, "mean", yvar, annotate_heatmap=False, dosort_colnames=False)
            DictVarToDf2d[var] = dftmp

        ##### Score data using kernesl
        res = []

        for i, (name, kernel) in enumerate(KERNELS.items()):
            k, w= kernel_convert_from_combined_to_split(kernel)
            var = kernel_extract_which_var(name)

            if var in EFFECT_VARS:
                # Ie if not, then this kernel is not defined, this dataset doesnt include this effect

                dfthis = DictVarToDf2d[var]

                # apply kernel
                scores = _kernel_compute_scores(dfthis, k, w)

                # Distribute scores across bregions
                assert list_bregion==dfthis.index.tolist()

                # Collect
                for s, br in zip(scores, list_bregion):

                    res.append({
                        "bregion":br,
                        "score":s,
                        "kernel_name":name,
                        "kernel_events":k,
                        "kernel_weights":w,
                        "var":var
                    })
        dfres_kernels = pd.DataFrame(res)

        if len(dfres_kernels)>0:
            # Second-order kernels, whose dimensions are the first-order kernels
            # NOTE: First-order kernels operate over time windows (for a specific var)
            # --> Therefore, second-order kernels are 2d-kernels, operating first over time, then over variables.
            # NOTE: many first order kernels are perfectly fine to keep as second-order..

            # first, get reshaped df (bregion, first order kernel name)
            # dfres_kernels_2d = convert_to_2d_dataframe(dfres_kernels, "bregion", "kernel_name", False, "mean", "score", dosort_colnames=False, list_cat_1=REGIONS_IN_ORDER)[0]
            dfres_kernels_2d = convert_to_2d_dataframe(dfres_kernels, "bregion", "kernel_name", False, "mean", "score", dosort_colnames=False)[0]

            k = ("reach_next", "reach_prev")
            w = (1,1)
            name = "reachdir"
            if all([_k in dfres_kernels_2d.columns for _k in k]):
                scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
                dfres_kernels_2d[name] = scores

            k = ("reach_next_durstk", "reach_prev_durstk")
            w = (1,1)
            name = "reachdir_durstk"
            if all([_k in dfres_kernels_2d.columns for _k in k]):
                scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
                dfres_kernels_2d[name] = scores

            k = ("gridloc", "reachdir_durstk")
            w = (1,-1)
            name = "gridloc_abstract"
            if all([_k in dfres_kernels_2d.columns for _k in k]):
                scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
                dfres_kernels_2d[name] = scores

            # k = ("gridloc", "reachdir")
            # w = (1,-1)
            # name = "gridloc_abstract"
            # if all([_k in dfres_kernels_2d.columns for _k in k]):
            #     scores = _kernel_compute_scores(dfres_kernels_2d, k, w)
            #     dfres_kernels_2d[name] = scores

            from pythonlib.tools.snstools import heatmap
            fig, axes = plt.subplots(2,2, figsize=(10,10))
            # for ax, norm_method in zip(axes.flatten(), [None, "all_sub", "col_sub", "row_sub"]):
            for ax, norm_method in zip(axes.flatten(), [None]):
                heatmap(dfres_kernels_2d, ax, False, (None, None), norm_method=norm_method)
                # convert_to_2d_dataframe(dfres_kernels, "bregion", "kernel_name", True, "mean", "score", annotate_heatmap=False, norm_method="col_sub", dosort_colnames=False, list_cat_1=REGIONS_IN_ORDER)
                ax.set_title(f"norm_{norm_method}")
            savefig(fig, f"{savedir}/heatmap-kernel_scores-ccvar_{yvar}.pdf")

            plt.close("all")
        else:
            dfres_kernels_2d = None



        ########### OVERLAY ON BRAIN SCHEMATIC
        # for norm_method in [None, "col_sub", "all_sub", "row_sub"]:
        for norm_method in [None]:
            if norm_method is None:
                zlims = (0, None)
            else:
                zlims = (None, None)
            if dfres_kernels_2d is not None:
                sdir = f"{savedir}/brain_schematic/kernels"
                os.makedirs(sdir, exist_ok=True)
                plot_df_from_wideform(dfres_kernels_2d, sdir, "bregion", "kernel",
                                      "score", norm_method=norm_method, zlims=zlims)
                # # one plot for each kernel.
                # plot_df_from_longform(dfres_kernels, "score", "kernel_name", savedir = sdir, savesuffix=kernel,
                #     DEBUG=False, diverge=False)
                plt.close("all")

                # One plot for each kernel - so that it has its own zlim
                sdir = f"{savedir}/brain_schematic/kernels/each_kernel"
                os.makedirs(sdir, exist_ok=True)
                for kernel in dfres_kernels_2d.columns:
                    suff = f"{kernel}-{yvar}"
                    plot_df_from_wideform(dfres_kernels_2d.loc[:,[kernel]], sdir, "bregion",
                                          "kernel", "score", savesuffix=suff,
                                          norm_method=norm_method, zlims=zlims)
                plt.close("all")

            # # Over time, plot for a given variable
            # sdir = f"{savedir}/brain_schematic/split_by_var"
            # os.makedirs(sdir, exist_ok=True)
            # for var in PARAMS["EFFECT_VARS"]:
            #     dfthis = DFMULT_THEOR[DFMULT_THEOR["var"]==var]
            #     plot_df_from_longform(dfthis, yvar, "wl_tw", savedir = sdir, savesuffix=var, norm_method=norm_method)
            # plt.close("all")
            #
            # sdir = f"{savedir}/brain_schematic/split_by_which_level_twind"
            # os.makedirs(sdir, exist_ok=True)
            # for wl_tw in DFMULT_THEOR["wl_tw"].unique().tolist():
            #     dfthis = DFMULT_THEOR[DFMULT_THEOR["wl_tw"]==wl_tw]
            #     plot_df_from_longform(dfthis, yvar, "var", savedir = sdir, savesuffix=wl_tw, norm_method=norm_method)
            # plt.close("all")
            #

    PARAMS = {
        "EFFECT_VARS":EFFECT_VARS,
        "list_bregion":list_bregion,
        "KERNELS":KERNELS,
    }

    return DFMULT_THEOR, DictBregionToDf2d, DictVarToDf2d, dfres_kernels_2d, PARAMS


def OBS_pipeline_rsa_scalar_population_MULT_PLOT_DETAILED(animal, DATE, version_distance,
                                                      list_which_level, question=None,
                                                      DO_AGG_TRIALS=True, event=None):
    """ [GOOD] Detialed plots, means that plots distance scores for each level of
    each var. Plots "same" i.e, dist between the
    same level across different levels of othervar, and diff... Does NOT care about
    theoretical simmats.
    """
    from pythonlib.tools.pandastools import grouping_count_n_samples
    from pythonlib.tools.listtools import stringify_list
    from pythonlib.tools.pandastools import plot_subplots_heatmap

    # Is similarity dominated by a specific level of var?
    # -- e.g, "stroke_index_semantic" is high... is this just becuase "first_stroke" is similar?
    RES, SAVEDIR_MULT, params, list_bregion = load_mult_data_helper(animal, DATE, version_distance,
                                                                            list_which_level=list_which_level,
                                                                            question=question,
                                                                    DO_AGG_TRIALS=DO_AGG_TRIALS,
                                                                    event=event)
    list_twind = RES[0]["list_time_windows"]
    savedir = f"{SAVEDIR_MULT}/details_each_level"
    os.makedirs(savedir, exist_ok=True)

    ###### COLLECT DATA
    resthis = []
    for which_level in list_which_level:
        for bregion in list_bregion:
            for twind in list_twind:
                res, PA, Clraw, Clsim = _load_single_data(RES, bregion, twind, which_level,
                                                          recompute_sim_mats=True)
                EFFECT_VARS = res["EFFECT_VARS"]
                for var in EFFECT_VARS:
                    var_levs = Clsim.rsa_labels_extract_var_levels()[var]

                    # ONlye consider the categorical ones
                    EFFECT_VARS = _preprocess_check_which_vars_categorical(PA, EFFECT_VARS)

                    for lev in var_levs:

                        ######## COUNT N SAMPLES
                        dfthis = PA.Xlabels["trials"][PA.Xlabels["trials"][var]==lev] # trial, not agged.

                        # Count n trials for this level
                        n = len(dfthis)

                        # Count n trials across each conjunction of othervars
                        n_each_other_var = tuple(grouping_count_n_samples(dfthis, [v for v in EFFECT_VARS if not v==var]))
                        # groupdict = grouping_append_and_return_inner_items(dfthis, [v for v in EFFECT_VARS if not v==var])
                        # # - collect distribution of n across othervars
                        # n_each_other_var = []
                        # for lev_other, inds in groupdict.items():
                        #     n_each_other_var.append(len(inds))
                        #

                        if n>1:
                            # get indices that are same, and diff
                            ma_same, ma_diff = Clsim.rsa_matindex_same_diff_this_level(var, lev)

                            # upper triangular
                            ma_ut = Clsim._rsa_matindex_generate_upper_triangular()

                            assert sum(sum(ma_same & ma_ut))>0
                            assert sum(sum(ma_diff & ma_ut))>0

                            # get values
                            d_same = Clsim.Xinput[ma_same & ma_ut].mean()
                            d_diff = Clsim.Xinput[ma_diff & ma_ut].mean()

                            assert not np.isnan(d_same)
                            assert not np.isnan(d_diff)
                        else:
                            d_same = np.nan
                            d_diff = np.nan

                        ### SAVE
                        resthis.append({
                            "which_level": which_level,
                            "bregion": bregion,
                            "twind":twind,
                            "var":var,
                            "lev":lev,
                            "lev_str":stringify_list(lev, return_as_str=True),
                            "n":n,
                            "n_each_other_var":n_each_other_var,
                            "dist_same":d_same,
                            "dist_diff":d_diff
                        })
    # Save
    dfres = pd.DataFrame(resthis)

    # how many had only n of 1?
    if False:
        dfres["n"].hist(bins=20)
        dfres["n"]==1

    #### Distribution of counts for each level of each var (nrows, i.e,, trials)

    # Pull out just a single bregion for counts analysis, they are identical
    dfresthis = dfres[dfres["bregion"]==list_bregion[0]]

    ncols = 3
    nrows = int(np.ceil(len(EFFECT_VARS)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    list_lev = dfresthis["lev"].unique().tolist()
    twind = dfresthis["twind"].values[0]
    which_level = dfresthis["which_level"].values[0]
    for ax, var in zip(axes.flatten(), EFFECT_VARS):
        ax.set_title(var)
        ax.set_xlabel("n counts, across all conjunctions of other var")
        for i, lev in enumerate(list_lev):
            dftmp = dfresthis[(dfresthis["which_level"]==which_level) & (dfresthis["twind"]==twind) & (dfresthis["var"]==var) & (dfresthis["lev"]==lev)]
            if len(dftmp)>0:
                assert len(dftmp)==1
                n_each_other_var = dftmp.iloc[0]["n_each_other_var"]
                ax.plot(n_each_other_var, np.ones(len(n_each_other_var))*i, "o", label=lev, alpha=0.5)
                ax.text(min(n_each_other_var), i, f"n={n_each_other_var}", fontsize=5)
            else:
                # This is expected, since im looping through levs which are colelcted acorss all
                # var, not just this var
                pass

        ax.legend()
    path = f"{savedir}/n_rows_per_level.pdf"
    savefig(fig, path)
    plt.close("all")

    #### Plot distances between datapts that are "same" and "diff" for each level
    for twind in list_twind:
        for which_level in list_which_level:
            dfresthis = dfres[(dfres["which_level"]==which_level) & (dfres["twind"]==twind)].dropna().reset_index(drop=True)
            if len(dfresthis)==0:
                print(which_level)
                print(twind)
                print(dfres["which_level"].unique())
                print(dfres["twind"].unique())
                assert False

            # Between pairs of datapts with same values for each level
            for val in ["dist_same", "dist_diff"]:
                fig, axes = plot_subplots_heatmap(dfresthis, "bregion", "lev_str", val,
                                                  "var", annotate_heatmap=False, share_zlim=True)
                savefig(fig, f"{savedir}/{val}-wl_{which_level}-twind_{twind}.pdf")
            plt.close("all")


def rsagood_questions_dict(animal, date, question=None):
    """ Return dict of questions for this animal/date
    PARAMS:
    - question, if None, then get all for this animal. else get this specific inputed one.
    """

    if question is not None:
        questions = [question]
    else:
        ## Single prims
        if (animal, date) in [
                ("Pancho", 220717)]:
            # (location, size)
            questions = ["SP_loc_size"]
        elif (animal, date) in [
                ("Pancho", 220716),
                ("Diego", 230618),
                ("Diego", 230619)]:
            questions = ["SP_shape_size", "SS_shape"]
        ## Single prims
        elif (animal, date) in [
                ("Diego", 230614),
                ("Diego", 230615)]:
            questions = ["SP_shape_loc_TIME", "SP_shape_loc", "SS_shape"]
        elif (animal, date) in [
                ("Pancho", 220715)]:
            # JUST TESTING STROEKS = TRIALS>.. (resutsl).
            questions = ["SP_shape_loc_STROKES", "SP_shape_loc_TIME", "SP_shape_loc"]
        ## Single prims
        elif (animal, date) in [
                ("Pancho", 220606),
                ("Pancho", 220608),
                ("Pancho", 220609),
                ("Pancho", 220610),
                ("Pancho", 220918)]:
            questions = ["SP_shape_loc_size"]
        ## PIG
        elif (animal, date) in [
                ("Diego", 230628),
                ("Diego", 230630),
                ("Pancho", 230623),
                ("Pancho", 230626)]:
            questions = ["shape_loc", "seq_pred", "pig_vs_sp", "seq_ctxt"]
        elif (animal, date) in [
                ("Pancho", 230612),
                ("Pancho", 230613)
                ]:
            questions = ["CV_shape", "CV_shape_2", "CV_loc", "CV_loc_2"][::-1]
        elif (animal, date) in [
                ("Diego", 231201),
                ("Diego", 231204),
                ("Diego", 231219),
                ("Pancho", 230120),
                ("Pancho", 230122),
                ("Pancho", 230125),
                ("Pancho", 230126),
                ("Pancho", 230127),
                ]:
            questions = ["CHAR_shape_2", "CHAR_shape"]
        else:
            print(animal, date)
            assert False

    DictParamsEachQuestion = {q:rsagood_questions_params(q) for q in questions}

    return DictParamsEachQuestion


def rsagood_questions_params(question):
    """
    Return the params_dict for this question
    :param question:
    :return:
    """

    distmat_animal = None
    distmat_date = None
    distmat_distance_ver = None
    THRESH_clust_sim_max = None
    list_subtract_mean_each_level_of_var = [None]
    list_vars_test_invariance_over_dict = [None]
    dict_vars_levels_prune = None

    # slice_agg_slices = None
    # slice_agg_vars_to_split = None
    # events_keep = None
    list_time_windows = [
        (-0.5, -0.3),
        (-0.3, -0.1),
        (-0.1, 0.1),
        (0.1, 0.3),
        (0.3, 0.5),
        ]

    ##### Use specific datasets for specific questions.
    if question=="seq_pred":
        ################ sequence prediction
        # Predicting next stroke's features.
        effect_vars = ["shape_oriented", "gridloc", "CTXT_loc_next", "CTXT_shape_next", "gap_to_next_angle_binned", "stroke_index"]
        list_which_level = ["stroke", "stroke_off"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        # - include all strokes within sequence
        exclude_last_stroke=True
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 2
        max_taskstrokes = 5

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "seqcontext"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None]

        # Which variables to plot all the pairwise distmats for
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["gridloc", "CTXT_loc_next"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="shape_loc":
        # Logic - try to get all strokes, but account for stroke-index dependence with stroke_index_semantic,
        # since main effects are usually differences for first and last (reaching movements), and
        # gridloc is strongly correlated with stroke index... (onset and offset)


        HACK = False # temporary, just to compare trial to stroke based analy
        # THIS requires up[daing code in analy_rsa_script to rename variables.
        # and update events_keep to include the stroke for trial data

        ## Params which apply across all which_level
        if HACK:
            effect_vars = ["shape_oriented", "gridloc"]
            list_which_level = ["stroke", "stroke_off"] # Whihc which_level to keep

            ## For "stroke" and "stroke_off" which_levels
            # - include all strokes within sequence
            exclude_last_stroke=False
            exclude_first_stroke=False
            keep_only_first_stroke=True
            min_taskstrokes = 1
            max_taskstrokes = 6
        else:
            effect_vars = ["shape_oriented", "gridloc", "gap_from_prev_angle_binned", "stroke_index_semantic"]
            list_which_level = ["stroke"] # Whihc which_level to keep

            ## For "stroke" and "stroke_off" which_levels
            # - include all strokes within sequence
            exclude_last_stroke=False
            exclude_first_stroke=False
            keep_only_first_stroke=False
            min_taskstrokes = 2
            max_taskstrokes = 6

        ## Optionally, rename variables for speicifc which_level, so that variable
        # names match across which_level --> Helps since the analy requires all datapts
        # to use same variable names.
        map_varname_to_new_varname = {
            "trial":{"seqc_0_shape":"shape_oriented", "seqc_0_loc":"shape_oriented"}
        }

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "seqcontext"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None]

        # Which variables to plot all the pairwise distmats for
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["shape_oriented", "gridloc"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="seq_ctxt":
        # Logic - care about each index, but exclude first stroke since it has large movement. Conceptualize
        # this as sequence context within sequence.
        # Account for location and previous location, as these are correlated with index.

        effect_vars = ["shape_oriented", "gridloc", "stroke_index_fromlast_tskstks", "gap_from_prev_angle_binned"]
        list_which_level = ["stroke"] # Whihc which_level to keep

        # ## For "stroke" and "stroke_off" which_levels
        # # - include all strokes within sequence
        exclude_last_stroke=False
        exclude_first_stroke=True
        keep_only_first_stroke=False
        min_taskstrokes = 3
        max_taskstrokes = 5

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "seqcontext"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None]

        # Which variables to plot all the pairwise distmats for
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["stroke_index_fromlast_tskstks", "shape_oriented"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="pig_vs_sp":
        effect_vars = ["shape_oriented", "gridloc", "task_kind"]
        list_which_level = ["stroke"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        # - include all strokes within sequence
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=True
        min_taskstrokes = 1
        max_taskstrokes = 5
        #
        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "seqcontext"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None]

        # Which variables to plot all the pairwise distmats for
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["shape_oriented", "gridloc", "task_kind"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="CHAR_shape":
        # Consistent modulation by shape across stroke indices (which are a proxy for sequential
        # context

        ## Params which apply across all which_level
        # effect_vars = ["shape_label", "stroke_index_fromlast", "velmean_thbin"]
        effect_vars = ["shape_label", "stroke_index"]
        list_which_level = ["stroke"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        # - include all strokes within sequence
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 2
        max_taskstrokes = 6
        # THRESH_clust_sim_max = 2

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "charstrokes"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None, "stroke_index"]

        # Which variables to plot all the pairwise distmats for
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["shape_label", "stroke_index"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="CHAR_shape_2":
        # Consistent modulation by shape across stroke indices (which are a proxy for sequential
        # context

        ## Params which apply across all which_level
        effect_vars = ["shape_label", "stroke_index_fromlast", "velmean_thbin"]
        list_which_level = ["stroke"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        # - include all strokes within sequence
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 2
        max_taskstrokes = 6
        # THRESH_clust_sim_max = 2

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "charstrokes"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None, "stroke_index_fromlast"]

        # Which variables to plot all the pairwise distmats for
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["shape_label", "stroke_index_fromlast"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="SP_shape_size":
        ## Params which apply across all which_level
        effect_vars = ["seqc_0_shape", "gridsize"]
        list_which_level = ["trial"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 1
        max_taskstrokes = 5
        THRESH_clust_sim_max = None

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = [
            '03_samp',
            '06_on_strokeidx_0']
        ANALY_VER = "singleprim"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = ["gridsize", None]

        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["seqc_0_shape", "gridsize"]
        plot_pairwise_distmats_twinds = [
            ("trial", "06_on_strokeidx_0", (-0.3, -0.1)),
            ("trial", "03_samp", (0.3, 0.5)),
        ]

    elif question=="SP_shape_loc":
        ## Params which apply across all which_level
        effect_vars = ["seqc_0_shape", "seqc_0_loc"]
        list_which_level = ["trial"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 1
        max_taskstrokes = 5
        THRESH_clust_sim_max = None

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = [
            '03_samp',
            '06_on_strokeidx_0']
        ANALY_VER = "singleprim"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = ["seqc_0_loc", None]

        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["seqc_0_shape", "seqc_0_loc"]
        plot_pairwise_distmats_twinds = [
            ("trial", "06_on_strokeidx_0", (-0.3, -0.1)),
            ("trial", "03_samp", (0.3, 0.5)),
        ]

    elif question=="SP_shape_loc_STROKES":
        # TEMP, just to compare to trial version, saniuyt cehcek they are same.

        effect_vars = ["shape_oriented", "gridloc"]
        list_which_level = ["stroke"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        # - include all strokes within sequence
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=True
        min_taskstrokes = 1
        max_taskstrokes = 5

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ["00_stroke"]
        ANALY_VER = "seqcontext"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None, "gridloc"]

        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["shape_oriented", "gridloc"]
        plot_pairwise_distmats_twinds = [
            ("stroke", "00_stroke", (-0.3, -0.1)),
        ]

    elif question=="SP_shape_loc_size":
        ## Params which apply across all which_level
        effect_vars = ["seqc_0_shape", "seqc_0_loc", "gridsize"]
        list_which_level = ["trial"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 1
        max_taskstrokes = 5
        THRESH_clust_sim_max = None

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        # events_keep = [
        #     '03_samp',
        #     '04_go_cue',
        #     '05_first_raise',
        #     '06_on_strokeidx_0',
        #     '07_off_stroke_last',
        #     '08_doneb']
        events_keep = [
            '03_samp',
            '06_on_strokeidx_0']
        ANALY_VER = "singleprim"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = ["gridsize", None]
        list_vars_test_invariance_over_dict = [
            {"same":["seqc_0_loc"], "diff":["gridsize"]}
        ]

        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["seqc_0_shape", "seqc_0_loc", "gridsize"]
        plot_pairwise_distmats_twinds = [
            ("trial", "06_on_strokeidx_0", (-0.3, -0.1)),
            ("trial", "03_samp", (0.3, 0.5)),
        ]

    elif question=="CV_shape":
        # Complexvar, shape, before see samp, comapred to draw.

        ## Params which apply across all which_level
        # effect_vars = ["seqc_0_shape", "seqc_0_loc", "epochset", "epoch"]
        effect_vars = ["seqc_0_shape", "seqc_0_loc", "event"]
        list_which_level = ["trial"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 1
        max_taskstrokes = 5
        THRESH_clust_sim_max = None

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = [
            '03_samp',
            '06_on_strokeidx_0']
        ANALY_VER = "singleprimvar" # FOr preprocessing

        # If this requires slicing and agging DFallpa
        slice_agg_slices = [
            ("trial", "03_samp", (-0.3, -0.1)),
            ("trial", "06_on_strokeidx_0", (-0.3, -0.1))
        ]
        slice_agg_vars_to_split = ["bregion"]

        list_subtract_mean_each_level_of_var = ["event"]
        list_vars_test_invariance_over_dict = [
            {"same":["seqc_0_loc"], "diff":["event"]}
        ]

        # Only keep data with these levels
        dict_vars_levels_prune = {
            "epoch":["oneshp_varyloc"]
        }

        ## Params for analysis plots
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["seqc_0_shape", "event"]
        plot_pairwise_distmats_twinds = [
            ("dummy", "dummy", "dummy")
        ]

    elif question=="CV_shape_2":
        # Complexvar, shape, before see samp, comapred to draw.
        # Here, compare pre samp to post samp.

        q_params = rsagood_questions_params("CV_shape")
        slice_agg_slices = [
            ("trial", "03_samp", (-0.3, -0.1)),
            ("trial", "03_samp", (0.3, 0.5)),
        ]
        q_params["slice_agg_slices"] = slice_agg_slices

        return q_params

    elif question=="CV_loc":
        # Complexvar, location, before see samp, comapred to draw.

        q_params = rsagood_questions_params("CV_shape")

        # Update params
        list_vars_test_invariance_over_dict = [
            {"same":["seqc_0_shape"], "diff":["event"]}
        ]
        dict_vars_levels_prune = {
            "epoch":["oneloc_varyshp"]
        }
        plot_pairwise_distmats_variables = ["seqc_0_loc", "event"]

        q_params["list_vars_test_invariance_over_dict"] = list_vars_test_invariance_over_dict
        q_params["dict_vars_levels_prune"] = dict_vars_levels_prune
        q_params["plot_pairwise_distmats_variables"] = plot_pairwise_distmats_variables

        return q_params

    elif question=="CV_loc_2":
        # Complexvar, location, before see samp, comapred to draw.

        q_params = rsagood_questions_params("CV_loc")
        slice_agg_slices = [
            ("trial", "03_samp", (-0.3, -0.1)),
            ("trial", "03_samp", (0.3, 0.5)),
        ]
        q_params["slice_agg_slices"] = slice_agg_slices

        return q_params

    elif question=="SP_shape_loc_TIME":
        ## Params which apply across all which_level
        effect_vars = ["seqc_0_shape", "seqc_0_loc", "event"]
        list_which_level = ["trial"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 1
        max_taskstrokes = 5
        THRESH_clust_sim_max = None

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = [
            '03_samp',
            '06_on_strokeidx_0']
        ANALY_VER = "singleprim"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = [
            ("trial", "03_samp", (0.3, 0.5)),
            ("trial", "06_on_strokeidx_0", (-0.3, -0.1))
        ]
        slice_agg_vars_to_split = ["bregion"]

        list_subtract_mean_each_level_of_var = ["event"]
        list_vars_test_invariance_over_dict = [
            {"same":["seqc_0_loc"], "diff":["event"]}
        ]

        ## Params for analysis plots
        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["seqc_0_shape", "seqc_0_loc", "event"]
        plot_pairwise_distmats_twinds = [
            ("dummy", "dummy", "dummy")
        ]

    elif question=="SS_shape":
        # Substrokes, shapes (strokes) vs. substroke features (category)

        ## Params which apply across all which_level
        effect_vars = ["shape", "dist_angle", "index_within_stroke"]
        list_which_level = ["substroke"] # Whihc which_level to keep

        ## For "stroke" and "stroke_off" which_levels
        exclude_last_stroke=False
        exclude_first_stroke=False
        keep_only_first_stroke=False
        min_taskstrokes = 1
        max_taskstrokes = 5
        THRESH_clust_sim_max = None

        ## Params which apply AFTER you have concated across which_level
        # Which events to prune to
        events_keep = ['00_substrk']
        ANALY_VER = "singleprim"

        # If this requires slicing and agging DFallpa
        slice_agg_slices = None
        slice_agg_vars_to_split = None

        list_subtract_mean_each_level_of_var = [None]
        list_vars_test_invariance_over_dict = [
            {"same":["index_within_stroke"], "diff":["shape"]},
            {"same":["index_within_stroke"], "diff":["dist_angle"]},
            None,
        ]

        # Which variables to plot all the pairwise distmats for
        plot_pairwise_distmats_variables = ["shape", "dist_angle"]
        plot_pairwise_distmats_twinds = [
            ("substroke", "00_substrk", (-0.1, 0)),
        ]

        list_time_windows = [
            (-0.3, -0.2),
            (-0.2, -0.1),
            (-0.1, 0.),
            (0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            ]
    elif question=="SS_shape_firstss":
        # The first substroke only, which removes influence of
        # within-stroke context of sequence of substrokes.

        q_params = rsagood_questions_params("SS_shape")

        q_params["keep_only_first_stroke"] = True
        # Altenratiebly, prun it.
        # dict_vars_levels_prune = {
        #     "epoch":["oneloc_varyshp"]
        # }

        q_params["effect_vars"] = ["shape", "dist_angle"]
        q_params["list_vars_test_invariance_over_dict"] = [None]

        return q_params

    else:
        print(question)
        assert False

    # DictParamsEachQuestion = {}
    #
    # # Store
    # DictParamsEachQuestion["seq_pred"] = {
    #     "effect_vars":effect_vars,
    #     "exclude_last_stroke":exclude_last_stroke,
    #     "exclude_first_stroke":exclude_first_stroke,
    #     "keep_only_first_stroke":keep_only_first_stroke,
    #     "min_taskstrokes":min_taskstrokes,
    #     "max_taskstrokes":max_taskstrokes}

    if False:
        if len(effect_vars)>2:
            assert len(list_vars_test_invariance_over_dict)>0, "maybe you need? if want to be sure testing invar over specific vars"

    # make sure things that ened to be in "effect_vars" are there
    for vars_test_invariance_over_dict in list_vars_test_invariance_over_dict:
        if vars_test_invariance_over_dict is not None:
            for var in vars_test_invariance_over_dict["same"]:
                assert var in effect_vars
            for var in vars_test_invariance_over_dict["diff"]:
                assert var in effect_vars
    for var in plot_pairwise_distmats_variables:
        assert var in effect_vars

    tmp = []
    for x in list_vars_test_invariance_over_dict:
        if x is not None and len(x["same"])==0 and len(x["diff"])==0:
            tmp.append(None)
        else:
            tmp.append(x)
    list_vars_test_invariance_over_dict = tmp

    q_params = {
        "effect_vars":effect_vars,
        "exclude_last_stroke":exclude_last_stroke,
        "exclude_first_stroke":exclude_first_stroke,
        "keep_only_first_stroke":keep_only_first_stroke,
        "min_taskstrokes":min_taskstrokes,
        "max_taskstrokes":max_taskstrokes,
        "THRESH_clust_sim_max":THRESH_clust_sim_max,
        "distmat_animal":distmat_animal,
        "distmat_date":distmat_date,
        "distmat_distance_ver":distmat_distance_ver,
        "list_which_level":list_which_level,
        "events_keep":events_keep,
        "plot_pairwise_distmats_variables":plot_pairwise_distmats_variables,
        "plot_pairwise_distmats_twinds":plot_pairwise_distmats_twinds,
        "slice_agg_slices":slice_agg_slices,
        "slice_agg_vars_to_split":slice_agg_vars_to_split,
        # "subtract_mean_each_level_of_var":subtract_mean_each_level_of_var,
        "list_subtract_mean_each_level_of_var":list_subtract_mean_each_level_of_var,
        "list_vars_test_invariance_over_dict":list_vars_test_invariance_over_dict,
        "dict_vars_levels_prune":dict_vars_levels_prune,
        "ANALY_VER":ANALY_VER,
        "list_time_windows":list_time_windows
    }

    return q_params

