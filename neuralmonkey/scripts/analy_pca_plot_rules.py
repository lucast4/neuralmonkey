from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
import numpy as np
import sys
import pickle

animal = "Pancho"

DATE = int(sys.argv[1])

# for DATE in [220715, 220716, 220717]:
# # DATE = 221020
# DATE = 221020
# # DATE = 220805
# DATE = 220717
# # DATE = 220901
# DATE = 220715

# # DATE = 221020
# DATE = 230603
# # DATE = 220805
# # DATE = 220715
# # DATE = 220901
# animal = "Diego"

# %matplotlib inline
# to help debug if times are misaligned.
MINIMAL_LOADING = True
MS = load_mult_session_helper(DATE, animal, MINIMAL_LOADING=MINIMAL_LOADING)
    


from neuralmonkey.classes.snippets import load_and_concat_mult_snippets
# if False:
#     # stroke level, old
#     SAVEDIR = "/gorilla1/analyses/recordings/main/chunks_modulation"
# else:
#     # trial level, new
#     SAVEDIR = "/gorilla1/analyses/recordings/main/anova/bytrial"
which_level="trial"    
SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level=which_level)





ANALY_VER = "rulesw"
# ANALY_VER = "singleprim"
anova_interaction = False





assert False, "replabe below with function that preprocesses Dataset"

# ANALY_VER = "singleprim"

from neuralmonkey.metadat.analy.anova_params import params_getter_plots, params_getter_extraction
from pythonlib.dataset.analy_dlist import concatDatasets
from neuralmonkey.classes.snippets import _dataset_extract_prune_general, _dataset_extract_prune_general_dataset
params = params_getter_plots(animal, DATE, which_level, ANALY_VER, anova_interaction=anova_interaction)
params_extraction = params_getter_extraction(animal, DATE, which_level, ANALY_VER)


#### COPIED From analy_anova_plot

list_dataset = []
for i, sn in enumerate(MS.SessionsList):
    # if which_level=="trial":
    # use the dataset here, since it is not saved
    D = sn.Datasetbeh
    # else:
    #     # use the dataset linked to DS, since it is saved
    #     D = SP.DSmult[i].Dataset
    #     assert len(D.Dat)==len(sn.Datasetbeh.Dat), "a sanity check. desnt have to be, but I am curious why it is not..."

    # THINGs that must be done by each individual D
    D.behclass_preprocess_wrapper()

    # Second, do preprocessing to concatted D
    if params_extraction["DO_SCORE_SEQUENCE_VER"]=="parses":
        D.grammar_successbinary_score_parses()
    elif params_extraction["DO_SCORE_SEQUENCE_VER"]=="matlab":
        D.grammar_successbinary_score_matlab()
    else:
        # dont score
        assert params_extraction["DO_SCORE_SEQUENCE_VER"] is None

    if params_extraction["taskgroup_reassign_simple_neural"]:
        # do here, so the new taskgroup can be used as a feature.
        D.taskgroup_reassign_ignoring_whether_is_probe(CLASSIFY_PROBE_DETAILED=False)                
        print("Resulting taskgroup/probe combo, after taskgroup_reassign_simple_neural...")
        D.grouping_print_n_samples(["taskgroup", "probe"])

    if params_extraction["DO_CHARSEQ_VER"] is not None:
        D.sequence_char_taskclass_assign_char_seq(ver=params_extraction["DO_CHARSEQ_VER"])

    list_dataset.append(D.copy())
# concat the datasets 
Dall = concatDatasets(list_dataset)

################ DO SAME THING AS IN EXTRACTION (these dont fail, when use concatted)
if params_extraction["DO_EXTRACT_CONTEXT"]:
    Dall.seqcontext_preprocess()

for this in params_extraction["list_epoch_merge"]:
    # D.supervision_epochs_merge_these(["rndstr", "AnBmTR|1", "TR|1"], "rank|1")
    Dall.supervision_epochs_merge_these(this[0], this[1], key=params_extraction["epoch_merge_key"],
        assert_list_epochs_exist=False)


if params_extraction["EXTRACT_EPOCHSETS"]:
    Dall.epochset_extract_common_epoch_sets(
        trial_label=params_extraction["EXTRACT_EPOCHSETS_trial_label"],
        n_max_epochs=params_extraction["EXTRACT_EPOCHSETS_n_max_epochs"],
        merge_sets_with_only_single_epoch=params_extraction["EXTRACT_EPOCHSETS_merge_sets"],
        merge_sets_with_only_single_epoch_name = ("LEFTOVER",))

if params_extraction["DO_EXTRACT_EPOCHKIND"]:
    Dall.supervision_epochs_extract_epochkind()

# Sanity check that didn't remove too much data.
if False:
    if "wrong_sequencing_binary_score" not in params["preprocess_steps_append"]:
        # Skip if is error trials.
        npre = len(D.Dat)
        npost = len(dat_pruned.Dat)
        if npost/npre<0.25 and len(sn.Datasetbeh.Dat)>200: # ie ignore this if it is a small session...
            print(params)
            print("THis has no wrong_sequencing_binary_score: ",  params['preprocess_steps_append'])
            assert False, "dataset pruning removed >0.75 of data. Are you sure correct? Maybe removing a supervisiuon stage that is actually important?"


####################### REEXTRACT VARIABLES
list_var_reextract = []
for var, vars_conjuction in zip(params["LIST_VAR"], params["LIST_VARS_CONJUNCTION"]):
    list_var_reextract.append(var)
    for v in vars_conjuction:
        list_var_reextract.append(v)
list_var_reextract = list(set(list_var_reextract))
print("* REEXTRACTING THESE VARS to SP.DfScalar:")
print(list_var_reextract)


if which_level=="trial":
    # First collect all variables that you might need (before deleting dataset).
    # By default, recompute them, since concatting datasets might change some variables.
    def _reextract_var(SP, varthis):
        """ Repopulate SP.DfScalar[varthis] with the new values in
        dataset.
        Modifies SP in place
        """
        from pythonlib.tools.pandastools import slice_by_row_label

        trialcodesthis = SP.DfScalar["trialcode"].tolist()

        # Get the sliced dataframe
        dfslice = slice_by_row_label(Dall.Dat, "trialcode", trialcodesthis,
            reset_index=True, assert_exactly_one_each=True)

        # Assign the values to SP
        print("Updating this column of SP.DfScalar with Dataset beh:")
        print(varthis)
        SP.DfScalar[varthis] = dfslice[varthis].tolist()

    # vars_already_extracted =[]
    for var in list_var_reextract:
    # for var, vars_conjuction in zip(params["LIST_VAR"], params["LIST_VARS_CONJUNCTION"]):

        # If any of these vars dont exist, try to extract them again from dataset
        # if var not in SP.DfScalar.columns:
        #     valuesthis = _reextract_var(SP, var)
        #     SP.DfScalar[var] = valuesthis
        # for v in vars_conjuction:
        #     if v not in SP.DfScalar.columns:
        #         valuesthis = _reextract_var(SP, v)
        #         SP.DfScalar[v] = valuesthis
        _reextract_var(SP, var)

    # For deletion code later, make dummys
    list_ds_dat = None
    DS = None
    dfstrokes = None
    dfstrokes_slice = None

elif which_level=="stroke":

    # For each DS (session) extract the column from Dall
    list_ds_dat = []
    for DS in SP.DSmult:

        # Update dataset
        DS.dataset_replace_dataset(Dall)

        # get fields from Dall
        for var in list_var_reextract:
            if var in DS.Dataset.Dat.columns:
                # 1) append a new column in DS that has the desired variable from Dataset
                DS.dataset_append_column(var)
            else:
                print("SHOULD REGENERATE DS  with this var!!")

        # concat 
        list_ds_dat.append(DS.Dat)

    # Extract the concatenated df strokes            
    dfstrokes = pd.concat(list_ds_dat).reset_index(drop=True)

    # 2) extract a slice of DS with desired trialcodes and strokeindices.
    list_trialcode = SP.DfScalar["trialcode"].tolist()
    list_stroke_index = SP.DfScalar["stroke_index"].tolist()
    dfstrokes_slice = SP.DSmult[0].dataset_slice_by_trialcode_strokeindex(list_trialcode, list_stroke_index, 
        df=dfstrokes)

    # Sanity checks
    assert np.all(SP.DfScalar["trialcode"] == dfstrokes_slice["trialcode"])
    assert np.all(SP.DfScalar["stroke_index"] == dfstrokes_slice["stroke_index"])
    # if not np.all(SP.DfScalar["gridloc"] == dfstrokes_slice["gridloc"]):
    #     for i in range(len(dfstrokes_slice)):
    #         a = SP.DfScalar.iloc[i]["gridloc"]
    #         b = dfstrokes_slice.iloc[i]["gridloc"]
    #         if a == b:
    #             print(i, a, b)
    #         if a != b:
    #             print("**", i, a, b)
    #     assert False, "why different?"

    # Debug, if rows are not one to one
    if False:
        dfstrokes[dfstrokes["trialcode"]=="221020-2-211"]

        sp.DS.Dat[sp.DS.Dat["trialcode"]=="221020-2-211"]

        import numpy as np
        assert np.all(sp.DfScalar["trialcode"] == dfstrokes_slice["trialcode"])
        assert np.all(sp.DfScalar["stroke_index"] == dfstrokes_slice["stroke_index"])
        assert np.all(sp.DfScalar["gridloc"] == dfstrokes_slice["gridloc"])

    # merge with sp
    for var in list_var_reextract:
        print("Updating this column of SP.DfScalar with DatStrokes beh:")
        print(var)
        SP.DfScalar[var] = dfstrokes_slice[var]
else:
    print(which_level)
    assert False

###### PRUNE DATASET TO GET SUBSET TRIALCODES
# Only keep subset these trialcodes
dataset_pruned_for_trial_analysis = _dataset_extract_prune_general_dataset(Dall, 
    list_superv_keep=params["list_superv_keep"], 
    preprocess_steps_append=params["preprocess_steps_append"],
    remove_aborts=params["remove_aborts"],
    list_superv_keep_full=params["list_superv_keep_full"], 
    )    

TRIALCODES_KEEP = dataset_pruned_for_trial_analysis.Dat["trialcode"].tolist()

###### SANITY CHECK, the type of each item for each var, must be the same across levels.
# or else errors, e.g., seaborn fails.
for var, vars_conjuction in zip(params["LIST_VAR"], params["LIST_VARS_CONJUNCTION"]):
    tmp = SP.DfScalar[var].unique().tolist()
    if len(set([type(x) for x in tmp]))>1:
        print(tmp)
        print([type(x) for x in tmp])
        print(var)
        assert False, "levels are not all same type..."

    for v in vars_conjuction:
        tmp = SP.DfScalar[v].unique().tolist()
        if len(set([type(x) for x in tmp]))>1:
            print(tmp)
            print([type(x) for x in tmp])
            print(v)
            assert False, "levels are not all same type..."

#################### OUTPUT
# Keep only the good trialcodes
SP.DfScalar = SP.DfScalar[SP.DfScalar["trialcode"].isin(TRIALCODES_KEEP)].reset_index(drop=True)





SP.datamod_append_unique_indexdatapt()
SP.DfScalar = SP.datamod_append_bregion(SP.DfScalar)




from pythonlib.tools.expttools import load_yaml_config



def _load_pca_space(pca_trial_agg_grouping):
    SAVEDIR = f"/gorilla1/analyses/recordings/main/pca/{animal}/{DATE}/aggby_{pca_trial_agg_grouping}"

    path = f"{SAVEDIR}/DF_PA_SPACES.pkl"
    with open(path, "rb") as f:
        DF_PA_SPACES = pickle.load(f)

    path = f"{SAVEDIR}/params_pca_space.yaml"
    params_pca_space = load_yaml_config(path)

    print("Loaded this already computed PCA space, with these params:")
    print(params_pca_space)
    
    return DF_PA_SPACES, params_pca_space, SAVEDIR



if False:
    # Scalar
    time_windows = [
        (0.2, 0.6),
    ]
else:
    PRE = -0.6
    POST = 0.6
    DUR = 0.2
    SLIDE = 0.05
    n = (POST-PRE)/DUR
    times1 = np.arange(PRE, POST-DUR, SLIDE)
    times2 = times1+DUR
    time_windows = np.stack([times1, times2], axis=1)

print(time_windows)    









from neuralmonkey.analyses.state_space_good import plotwrapper_statespace_single_event_bregion, _preprocess_extract_plot_params, _plot_statespace_dfmult_on_ax, plot_statespace_grpbyevent, _plot_statespace_df_on_ax
# vars_others = None
from neuralmonkey.analyses.state_space_good import plot_statespace_grpbyevent, plot_statespace_grpbyvarsothers, _preprocess_extract_PApca, plot_statespace_grpbyevent_overlay_othervars
from pythonlib.tools.pandastools import append_col_with_grp_index

list_vars = [
    "epoch", 
]
list_vars_others = [
    ["epochset"],
]

for pca_trial_agg_grouping in list_vars:
    ## 1) Load PCA space
    # pca_trial_agg_grouping = "epoch"
    # pca_trial_agg_grouping = "seqc_0_loc"
    # pca_trial_agg_grouping = "seqc_0_shape"

    ## 2) Load and process data
    # event_wind_pca = "03_samp_0.04_0.6"
    # bregion = "PMd_p"
    # event_dat = "03_samp"
    # var_dat = "epoch"
    # vars_others_dat = ["epochset"]
    # # vars_others_dat = None

    try:        
        DF_PA_SPACES, params_pca_space, SAVEDIR = _load_pca_space(pca_trial_agg_grouping)
    except FileNotFoundError:
        continue

    for var_dat, vars_others_dat in zip(list_vars, list_vars_others):
        # var_dat = "seqc_0_loc"
        # vars_others_dat = ["seqc_0_shape"]

        # var_dat = "seqc_0_shape"
        # vars_others_dat = ["seqc_0_loc"]

        if not var_dat == pca_trial_agg_grouping:
            continue

        if len(SP.DfScalar[var_dat].unique())==1:
            continue

        # Prepping
        SP.DfScalar = append_col_with_grp_index(SP.DfScalar, vars_others_dat, "vars_others_tmp", use_strings=False)

        if vars_others_dat is not None:
            vars_others_dat_str = "-".join(vars_others_dat)
        else:
            vars_others_dat_str = "None"

        SDIR = f"{SAVEDIR}/FIGS/var_{var_dat}__varothers_{vars_others_dat_str}"
        print(SDIR)

        list_event_wind_pca = DF_PA_SPACES["event_wind"].unique().tolist()
        list_bregion = DF_PA_SPACES["bregion"].unique().tolist()
        if True:
            # - just plot a specific event_dat
            list_event_data = ["03_samp", "04_go_cue", "06_on_strokeidx_0"]
        else:
            list_event_data = SP.Params["list_events_uniqnames"]

        for event_dat in list_event_data:
            assert event_dat in SP.Params["list_events_uniqnames"]


        for event_wind_pca in list_event_wind_pca:
            for bregion in list_bregion:
                PApca = _preprocess_extract_PApca(DF_PA_SPACES, event_wind_pca, bregion)

                # plot dimensionlaity.
                from neuralmonkey.analyses.state_space_good import _plot_pca_results
                fig = _plot_pca_results(PApca)
                fig.savefig(f"{SAVEDIR}/pca_results-{event_wind_pca}-{bregion}.pdf")


        ######################
        savedir = f"{SDIR}/each_event_trajs"
        import os
        os.makedirs(savedir, exist_ok=True)

        # One plot for each event
        for event_wind_pca in list_event_wind_pca:
            for bregion in list_bregion:
                PApca = _preprocess_extract_PApca(DF_PA_SPACES, event_wind_pca, bregion)

                for event_dat in list_event_data:
                    print(event_wind_pca, bregion, event_dat)
                    plotwrapper_statespace_single_event_bregion(DF_PA_SPACES, SP, event_wind_pca, bregion,
                                                                event_dat, var_dat, vars_others_dat,
                                                                time_windows_traj=time_windows,
                                                                savedir=savedir)
                    plt.close("all")


        ##########################
        time_windows = [
            (0.2, 0.6),
        ]
        n_trials_rand = 20

        # One plot for each event
        for event_wind_pca in list_event_wind_pca:
            for bregion in list_bregion:
                PApca = _preprocess_extract_PApca(DF_PA_SPACES, event_wind_pca, bregion)
                for event_dat in list_event_data:
                    print(event_wind_pca, bregion, event_dat)
                    for PLOT_TRIALS in [True, False]:
                        sdirthis = f"{SDIR}/each_event_scalar/plot_trials_{PLOT_TRIALS}"
                        import os
                        os.makedirs(sdirthis, exist_ok=True)
                        plotwrapper_statespace_single_event_bregion(DF_PA_SPACES, SP, event_wind_pca, bregion,
                                                                    event_dat, var_dat, vars_others_dat,
                                                                    time_windows_traj=time_windows,
                                                                    savedir=sdirthis, n_trials_rand=n_trials_rand,
                                                                    PLOT_TRIALS=PLOT_TRIALS)
                        plt.close("all")            