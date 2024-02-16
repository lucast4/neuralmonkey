"""
Decoding features for prims.
Currently using static features (fr vector)
See notebook: 220713_prims_state_space
"""

import matplotlib.pyplot as plt
import numpy as np

assert False, "OLD. instead, use state_space_good. See analy_pca_extract.py"

def get_traintest_split_conjunction(DfTrials, list_varnames, list_varlevels, verbose=False):
    """ 
    Helper to get train/test split, where test is holding out all trials for a specific
    combination of params. Train defined as all others. 
    Does not yet do balancing of trials.
    Hold out all trials that have conjunction of one specific combiantion fo 
    variables - e..g, specific shape x location.
    RETURNS:
    - inds_train, inds_test, index into DfTrials.
    """
    from pythonlib.tools.pandastools import filterPandasMultOrs
    inds_test = filterPandasMultOrs(DfTrials, list_varnames, [list_varlevels])

    # 2) Get train
    # -- Method 1: anything that isnt test
    inds_train = [i for i in range(len(DfTrials)) if i not in inds_test]
    
    if verbose:
        # 3) print what got
        print("TESTING: ", len(inds_test))
        print(DfTrials.iloc[inds_test]["grp"].value_counts())

        print('   ')


        print("TRAINING: ", len(inds_train))
        print(DfTrials.iloc[inds_train]["grp"].value_counts())
    return inds_train, inds_test

def dftrials_slice_for_model(DfTrials, inds, yfeat, map_var_to_cat, 
    ndims=None, verbose=False, which_x="default"):
    """ Slice and format data for modeling
    PARAMS;
    - map_var_to_cat, dict, varname --> integer.
    - dftrials_slice_for_model, str, which fr represntation to use
    RETURNS:
    - X, array (ntrials, nfeats)
    - y, array (ntrials, )
    """ 
    
    if which_x=="default":
        xname = "x"
    elif which_x=="x_centered":
        xname = "x_centered"
    else:
        assert False
    X = np.stack(DfTrials.iloc[inds][xname].tolist(), axis=1).T
    if ndims:

        X = X[:, :ndims]
    assert X.shape[0] == len(inds)

    y = DfTrials.iloc[inds][yfeat].tolist()
    y = [map_var_to_cat[ythis] for ythis in y]
    if verbose:
        print("Shape of X (ntrials, nfeats/chans): ", X.shape)
        print("Map from var to labels: ", map_var_to_cat)

    return X, y, ndims

def get_test_data_and_models(DfTrials, dfmodels, yfeat, VER):
    """ Helper function to pull out specific models and associated test data, under
    different methods
    PARAMS:
    - DfTrials, dataframe with each row a trials
    - dfmodels, dataframe holding modeling results
    - yfeat, str, feature being predicted
    - VER, str, what set of models to get (see within)
    """
    
    if VER=="new_common":
        # 1) Generate new test dataset. Ignores what was held out for each model.
        # Can use all models.
        assert False, "fill in"
        # -- Generate test dataset
        shape_test = "squiggle3-2-1"
        loc_test = None
        indsthis = filterPandasMultOrs(DfTrials, list_varnames, [[shape_test, loc_test]], True)
        assert False, "next fn needs which_x"
        Xtest, ytest = dftrials_slice_for_model(DfTrials, indsthis, yfeat, ndims)
        def test_data_getter(dfthis, ind):
            """ Returns Xtest, ytest"""
            return Xtest, ytest
        # -- Use all models
        dfmodels_this = dfmodels

    elif VER=="held_out_model_specific":
        # 2) For each model, use the specific held out test set. In this case, choose specific models to plot
        # (usually those that have all levels for test category)
        
        # -- Use only specific models
        # By default pulls out the models tyring to predict within all levels for the predicted variable.
        # (e.g., all shapes)
        if yfeat=="shape_oriented":
            dfmodels_this = dfmodels[dfmodels["shape_test"]=="all"].reset_index(drop=True)
        elif yfeat=="gridloc":
            dfmodels_this = dfmodels[dfmodels["loc_test"]==(99,99)].reset_index(drop=True)
        elif yfeat=="gridsize":
            dfmodels_this = dfmodels[dfmodels["size_test"]=="all"].reset_index(drop=True)
        else:
            print(yfeat)
            print(dfmodels)
            assert False, "code it for this yfeat?"
        
        def test_data_getter(dfthis, ind):
            """ Returns Xtest, ytest"""
            Xtest = dfthis.iloc[ind]["Xtest"]
            ytest = dfthis.iloc[ind]["ytest"]
            return Xtest, ytest
            
    return dfmodels_this, test_data_getter



def plotsummary_confusion_matrix(dfmodels_this, test_data_getter, indmod, mapper_label_code_to_name=None):
    """ For each model, plot its confusion matrix
    PARAMS:
    - dfmodels_this, each row a model, usually a subset of all the models in dfmodels, which are good for plotting
    - test_data_getter, function with signature test_data_getter(dfmodels_this, index_row) --> Xtest, ytest
    - mapper_label_code_to_name, dict, mapping codenum-->string, where codenum are the ylabels (intiger cats). Used for labeling
    plot axes.
    """

    from pythonlib.tools.listtools import tabulate_list
    from sklearn.metrics import confusion_matrix
    from pythonlib.tools.snstools import rotateLabel 

    def _convert_labelcode_to_name(code):
        if mapper_label_code_to_name is None:
            return code
        else:
            return mapper_label_code_to_name[code]

    # 1) collect data, for this model
    Xtest, ytest = test_data_getter(dfmodels_this, indmod)
    mod = dfmodels_this.iloc[indmod]["pipe"]["lin_svm"]

    # 2) Make confusion matrix fior this model
    # Ground truths
    codes_true = ytest
    codes_true_names = [_convert_labelcode_to_name(code) for code in ytest]
    print("True labels: ", tabulate_list(codes_true_names))

    # Predictions
    score_test = mod.score(Xtest, ytest)
    codes_predicted = mod.predict(Xtest)
    codes_predicted_names = [_convert_labelcode_to_name(code) for code in codes_predicted]
    print("Test results: score: ", score_test, "| pred labels: ", tabulate_list(codes_predicted_names))
    
    # PLOT CONFUSION MATRIX
    labels = sorted(set(codes_true))
    labels_names = [_convert_labelcode_to_name(code) for code in labels]
    fig, axes = plt.subplots(2,2, figsize=(15, 15))
    for norm, ax in zip([None, "true", "pred", "all"], axes.flatten()):
        # 1) un nomrlzized
        C = confusion_matrix(codes_true, codes_predicted, labels=labels, normalize=norm)
        ax.imshow(C)
        ax.set_title(f"normalization over: {norm}")
        ax.set_ylabel("true label")
        ax.set_xlabel("predicted label")
        ax.set_yticks(range(len(labels_names)))
        ax.set_yticklabels(labels_names)
        ax.set_xticks(range(len(labels_names)))
        ax.set_xticklabels(ax.get_xticks(), rotation = 45)
        ax.set_xticklabels(labels_names)
    return fig



def decode_and_plot_iter_hyperparams(MS, DS, LIST_REGIONS, LIST_ALIGN_TO, LIST_YFEAT, 
    LIST_NDIMS, SAVEDIR_DECODE, VERSION, DATAPLOT_GROUPING_VARS, pca_trial_agg_grouping,
    pca_trial_agg_method, pca_time_agg_method):
    """ Decode variables, varying all hyperparameters, each time making plots and saving
    PARAMS:
    - See notebook: 220713_prims_state_space

    """
    from neuralmonkey.analyses.prims_decode import get_traintest_split_conjunction, dftrials_slice_for_model, get_test_data_and_models, plotsummary_confusion_matrix
    from neuralmonkey.population.classify import _model_fit, _model_score_OBSOLETE
    import pandas as pd
    from pythonlib.tools.pandastools import replaceNone
    import seaborn as sns
    from neuralmonkey.classes.population import datgrp_flatten_to_dattrials, extract_neural_snippets_aligned_to, compute_data_projections, dftrials_centerize_by_group_mean
    

    ### PARAMS
    # # single
    # align_to = "go_cue"
    # yfeat = "shape_oriented"
    # REGIONS = ["preSMA_p", "preSMA_a"]
    # # yfeat = "gridloc"
    # ndims = 50 # takes top n feature dimensions
    # # list
    LIST_RECENTER_BY_GROUP_MEAN = [False, True]

    for align_to in LIST_ALIGN_TO:
        
        # 1) 
        PAall = extract_neural_snippets_aligned_to(MS, DS, align_to=align_to)

        for REGIONS in LIST_REGIONS:
            
            # 1) get subset of dataset
            DatGrp, groupdict, DatGrpDf = compute_data_projections(PAall, DS.Dat, MS, VERSION, REGIONS, 
                                                                   DATAPLOT_GROUPING_VARS, 
                                                    pca_trial_agg_grouping, pca_trial_agg_method, 
                                                    pca_time_agg_method, ploton=False)

            DfTrials = datgrp_flatten_to_dattrials(DatGrp, DATAPLOT_GROUPING_VARS)

            # map from categorical to numerical variable
            MapVarToCat = {}
            for var in DATAPLOT_GROUPING_VARS:
                map_var_to_cat = {}
                variables = sorted(set(DfTrials[var].tolist()))
                for i, varname in enumerate(variables):
                    map_var_to_cat[varname] = i
                print(map_var_to_cat)
                MapVarToCat[var] = map_var_to_cat

            # 1) get test
            DfTrials = DfTrials.reset_index(drop=True)


            for yfeat in LIST_YFEAT: # feature to predict (e..g, shape)
                
                # 2) Add a new column "x_centered" which is centered for the groups NOT being predicted.
                # This adds new column.
                DATAPLOT_GROUPING_VARS_NOT_YFEAT = [var for var in DATAPLOT_GROUPING_VARS if var!=yfeat]
                DfTrials = dftrials_centerize_by_group_mean(DfTrials, DATAPLOT_GROUPING_VARS_NOT_YFEAT)

                for ndims in LIST_NDIMS:

                    #### PARAMS
                    map_var_to_cat = MapVarToCat[yfeat]

                    ########### AUTO PARAMS
                    list_varnames = DATAPLOT_GROUPING_VARS
                    LIST_SHAPES = DfTrials["shape_oriented"].unique().tolist()
                    LIST_LOCS = DfTrials["gridloc"].unique().tolist()
                    LIST_SIZES = DfTrials["gridsize"].unique().tolist()

                    if yfeat=="shape_oriented":
                        LIST_SHAPES_THIS = LIST_SHAPES + [None]
                        LIST_LOCS_THIS = LIST_LOCS
                        LIST_SIZES_THIS = LIST_SIZES    
                    elif yfeat=="gridloc":
                        LIST_SHAPES_THIS = LIST_SHAPES
                        LIST_LOCS_THIS = LIST_LOCS + [None]
                        LIST_SIZES_THIS = LIST_SIZES    
                    elif yfeat=="gridsize":
                        LIST_SHAPES_THIS = LIST_SHAPES
                        LIST_LOCS_THIS = LIST_LOCS
                        LIST_SIZES_THIS = LIST_SIZES + [None]
                    else:
                        assert False

                    for do_recenter in LIST_RECENTER_BY_GROUP_MEAN:

                        if do_recenter:
                            which_x = "x_centered"
                        else:
                            which_x = "default"

                        out = []
                        for shape_test in LIST_SHAPES_THIS:
                        #     shape_test = "Lcentered-4-0"
                            for loc_test in LIST_LOCS_THIS:
                                
                                for size_test in LIST_SIZES_THIS:

                                    ############### RUN
                            #     for loc_test in [(-1,-1), (-1,1), (1,-1), (1,1), None]:
                                #     loc_test = (1,-1)
                                    inds_train, inds_test = get_traintest_split_conjunction(DfTrials, list_varnames, 
                                                                                            [shape_test, loc_test, size_test])
                                    
                                    
                                    print("Train N: ", len(inds_train))
                                    print("Test N: ", len(inds_test))
                                    if len(inds_test)==0:
                                        # Skip this
                                        print("Skipping (because n test=0): [shape_test, loc_test, size_test]", shape_test, loc_test, size_test)
                                        continue

                                    # Sanity check
                                    for i in inds_train:
                                        assert i not in inds_test

                                    # Convert to X, y
                                    Xtrain, ytrain, ndims = dftrials_slice_for_model(DfTrials, inds_train, yfeat, 
                                        map_var_to_cat, ndims, which_x=which_x)
                                    Xtest, ytest, ndims = dftrials_slice_for_model(DfTrials, inds_test, yfeat, 
                                        map_var_to_cat, ndims, which_x=which_x)

                                    # Fit /test model
                                    model_params_optimal = {"C":0.01} # optimized regularization params
                                    pipe, score = _model_fit(Xtrain, ytrain, model_params=model_params_optimal)

                                    print("*** Held out test data score: ", pipe.score(Xtest, ytest), " - ", loc_test)
                                    pipe.get_params()
                                    mod = pipe.get_params()["lin_svm"]

                                    # SAVE THINGS
                                    out.append({
                                        "shape_test":shape_test,
                                        "loc_test":loc_test,
                                        "size_test":size_test,
                                        "Xtrain":Xtrain,
                                        "ytrain":ytrain,
                                        "Xtest":Xtest,
                                        "ytest":ytest,
                                        "inds_train":inds_train,
                                        "inds_test":inds_test,
                                        "ntrain":len(inds_train),
                                        "ntest":len(inds_test),
                                        "ndims":ndims,
                                        "yfeat":yfeat,
                                        "pipe":pipe,
                                        "score_train":score,
                                        "score_test":pipe.score(Xtest, ytest),
                                        "do_recenter_to_grp":do_recenter
                                    })

                            dfmodels = pd.DataFrame(out)

                            # Names of variables, make them intepretable for plotting. Convert Nones to something else.
                            dfmodels = replaceNone(dfmodels, "size_test", "all")
                            dfmodels = replaceNone(dfmodels, "shape_test", "all")
                            dfmodels = replaceNone(dfmodels, "loc_test", (99,99))

                            ##### Model plots
                            if REGIONS is None:
                                regions = ["ALL"]
                            else:
                                regions = REGIONS
                            sdir = f"{SAVEDIR_DECODE}/alignto_{align_to}-yfeat_{yfeat}-region_{'__'.join(regions)}-ndims_{ndims}-do_recenter_to_grp_{do_recenter}"
                            print("Saving model/figs at: ", sdir)
                            import os
                            os.makedirs(sdir, exist_ok=True)

                            # save model results and params
                            dfmodels.to_pickle(f"{sdir}/dfmodels.pkl")

                            # prediction accuracy
                            if yfeat=="shape_oriented":
                                x = "shape_test"
                    #             hue = "loc_test"
                                hue = ("loc_test", "size_test")
                            elif yfeat=="gridloc":
                                x = "loc_test"
                    #             hue = "shape_test"
                                hue = ("shape_test", "size_test")
                            elif yfeat=="gridsize":
                                x = "size_test"
                    #             hue = "shape_test"
                                hue = ("shape_test", "loc_test")
                            else:
                                assert False
                            
                            if isinstance(hue, tuple):
                                from pythonlib.tools.pandastools import append_col_with_grp_index
                                dfmodels = append_col_with_grp_index(dfmodels, hue, "-".join(hue))
                                hue = "-".join(hue)
                                
                            fig = sns.catplot(data=dfmodels, x = x, y="score_test", hue=hue, kind="bar", height=7, aspect=1.5)
                            fig.savefig(f"{sdir}/score_test_summarybars-x_vartopredict-hue_heldoutlevel.pdf")


                            ##### Generate confusion matrix for any given model

                            # ALSO: Given a specific model, return its prediction for any pt


                            ### Test dataset. Two options:

                            VER = "held_out_model_specific"
                            dfmodels_this, test_data_getter = get_test_data_and_models(DfTrials, dfmodels, yfeat, VER)


                            ##### Plot - confusion matrix

                            # Make mapper: code --> name
                            mapper_label_code_to_name = {}
                            for name, num in MapVarToCat[yfeat].items():
                                assert num not in mapper_label_code_to_name.keys()
                                mapper_label_code_to_name[num] = name
                            print(mapper_label_code_to_name)

                            def get_model_string_name(dfmodels, indrow):
                                name = f"shapetest_{dfmodels.iloc[indrow]['shape_test']}-loctest_{dfmodels.iloc[indrow]['loc_test']}-yfeat_{dfmodels.iloc[indrow]['yfeat']}-testsc_{dfmodels.iloc[indrow]['score_test']:.2f}"
                                return name

                            for indmod in range(len(dfmodels_this)):
                                modelstr = get_model_string_name(dfmodels_this, indmod)
                                fig = plotsummary_confusion_matrix(dfmodels_this, test_data_getter, indmod, mapper_label_code_to_name);
                                fig.savefig(f"{sdir}/confusionmat_{modelstr}.pdf")


############################ [12/8/22] GOOD PLOTS using PA in sn
# from pythonlib.tools.pandastools import append_col_with_grp_index


# # 1) in df, first prepare by getting, for each var, the conjucntion of the other vars.
# # (currently only works for n=2 vars)
# map_var_to_othervars = {}
# for i in range(len(list_var)):
#     for j in range(i+1, len(list_var)):
#         grp = [list_var[i], list_var[j]]
        
#         print(grp)
#         df, new_col_name = append_col_with_grp_index(df, grp, "-".join(grp), 
#                                                      strings_compact=True, return_col_name=True)
        
#         # Save mapper 
#         assert len(list_var)==3, "assumes this is 3 when finding the left out (k)"
#         k = [k for k in range(len(list_var)) if k not in [i, j]][0]
#         map_var_to_othervars[list_var[k]] = new_col_name
# print(map_var_to_othervars)

# list_var = ["gridloc", "gridsize", "shape_oriented"]
# nrows = len(list_events)+1
# ncols = len(list_var)
# fig, axes = plt.subplots(nrows, ncols,  sharey=True, figsize=(ncols*4, nrows*4))

# for j, var in enumerate(list_var):
#     for i, ev in enumerate(list_events):
            
#         ax = axes[i][j]
#         dfthisthis = dfthis[dfthis["event_aligned"]==ev]
#         other_vars = map_var_to_othervars[var] # conjucntion of other varss
#         g = sns.pointplot(ax=ax, data=dfthisthis, x=var, y="fr_scalar", hue=other_vars)
#         g.legend().remove()
        
#         if i==len(list_events)-1:
#             # also plot all combined
#             ax = axes[i+1][j]
#             sns.pointplot(ax=ax, data=dfthis, x=var, y = "fr_scalar", hue="event_aligned")
# # sns.catplot(data=dfthis, x=xvar, y = "fr_scalar", hue="event_aligned")