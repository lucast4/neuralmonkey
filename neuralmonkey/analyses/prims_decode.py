"""
Decoding features for prims.
Currently using static features (fr vector)
See notebook: 220713_prims_state_space
"""

import matplotlib.pyplot as plt
import numpy as np

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
    ndims=None, verbose=False):
    """ Slice and format data for modeling
    PARAMS;
    - map_var_to_cat, dict, varname --> integer.
    RETURNS:
    - X, array (ntrials, nfeats)
    - y, array (ntrials, )
    """ 
    X = np.stack(DfTrials.iloc[inds]["x"].tolist(), axis=1).T
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
        else:
            print(yfeat)
            print(dfmodels)
            assert False
        
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



