from neuralmonkey.classes.session import Session
import matplotlib.pyplot as plt
from neuralmonkey.scripts.load_and_save_locally import load_and_preprocess_single_session
import neuralmonkey.utils.monkeylogic as mkl
from neuralmonkey.classes.session import load_session_helper, load_mult_session_helper
import os
from neuralmonkey.metadat.analy.anova_params import dataset_apply_params

which_level="trial"    
ANALY_VER = "singleprim"

animal = "Diego"
LIST_DATE = [230615, 230619, 230621, 230614, 230616, 230618]

# animal = "Pancho"
# LIST_DATE =  [220606, 220608, 220609, 220610, 220718, 220719, 220724, 220918, 
#     221217, 221218, 221220, 230103, 230104]

for DATE in LIST_DATE:

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
    SP, SAVEDIR_ALL = load_and_concat_mult_snippets(MS, which_level=which_level)

    ListD = [sn.Datasetbeh for sn in MS.SessionsList]
    Dall, dataset_pruned_for_trial_analysis, TRIALCODES_KEEP, params, params_extraction = \
        dataset_apply_params(ListD, animal, DATE, which_level, ANALY_VER)
    SP.DfScalar = SP.DfScalar[SP.DfScalar["trialcode"].isin(TRIALCODES_KEEP)].reset_index(drop=True)            

    ####################### REEXTRACT VARIABLES into SP.DfScalar
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

        # def _reextract_var(SP, varthis):
        #     """ Repopulate SP.DfScalar[varthis] with the new values in
        #     dataset.
        #     Modifies SP in place
        #     """
        #     from pythonlib.tools.pandastools import slice_by_row_label
            
        #     trialcodesthis = SP.DfScalar["trialcode"].tolist()

        #     # Get the sliced dataframe
        #     dfslice = slice_by_row_label(Dall.Dat, "trialcode", trialcodesthis,
        #         reset_index=True, assert_exactly_one_each=True)

        #     # Assign the values to SP
        #     print("Updating this column of SP.DfScalar with Dataset beh:")
        #     print(varthis)
        #     SP.DfScalar[varthis] = dfslice[varthis].tolist()

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
            SP.datasetbeh_append_column(var, Dataset=Dall) 
            # _reextract_var(SP, var)

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


    #############################
    SP.datamod_append_unique_indexdatapt()
    SP.DfScalar = SP.datamod_append_bregion(SP.DfScalar)


    ##################

    from neuralmonkey.analyses.state_space_good import _compute_PCA_space
    # pca_trial_agg_grouping = "epoch"
    # list_vars = ["seqc_0_shape", "seqc_0_loc"]
    # list_vars_others = [
    #     ["seqc_0_loc"],
    #     ["seqc_0_shape"]
    # ]
    list_vars = ["seqc_0_shape", "seqc_0_loc", "gridsize"]
    # list_vars_others = [
    #     ["gridsize"],
    #     ["seqc_0_shape"]
    # ]

    pca_time_agg_method=None
    pca_norm_subtract_condition_invariant = True
        

    for pca_trial_agg_grouping in list_vars:
    # #     pca_trial_agg_grouping = "seqc_0_shape"
    #     # vars_others = ["epochset"]
    #     vars_others = ["seqc_0_loc"]


        list_event_window = [
            ('03_samp', -0.6, -0.04),
            ('03_samp', 0.04, 0.6),
            ('04_go_cue', -0.6, -0.04),
            ('05_first_raise', -0.6, -0.05),
            ('06_on_strokeidx_0', -0.45, 0.35),
        #     ('08_doneb', -0.5, 0.3),
        #     ('09_post', 0.05, 0.6),
        #     ('10_reward_all', 0.05, 0.6)
        ]

        # Save params
        params_pca_space = {
            "pca_trial_agg_grouping":pca_trial_agg_grouping,
            "pca_time_agg_method":pca_time_agg_method,
            "pca_norm_subtract_condition_invariant":pca_norm_subtract_condition_invariant,
            "list_event_window":list_event_window
        }


        SAVEDIR = f"/gorilla1/analyses/recordings/main/pca/{animal}/{DATE}/aggby_{pca_trial_agg_grouping}"
        print(SAVEDIR)
        os.makedirs(SAVEDIR, exist_ok=True)
        from pythonlib.tools.expttools import writeDictToYaml
        writeDictToYaml(params_pca_space, f"{SAVEDIR}/params_pca_space.yaml")

        DF_PA_SPACES = _compute_PCA_space(SP, pca_trial_agg_grouping, list_event_window=list_event_window,
                          pca_norm_subtract_condition_invariant=pca_norm_subtract_condition_invariant)

        # Save 
        import pickle
        path = f"{SAVEDIR}/DF_PA_SPACES.pkl"
        with open(path, "wb") as f:
            pickle.dump(DF_PA_SPACES, f)
