"""
About encoding and decoding motor kinematics.

See notebook: /home/lucas/code/drawmonkey/drawmonkey/notebooks_datasets/240912_MANUSCRIPT_FIGURES_1_revision.ipynb

Was developed for revision for MS action symbols, showing that PMv doesnt encoding kinematics that genrealize.
"""

from neuralmonkey.classes.population_mult import dfallpa_extraction_load_wrapper
import sys
import numpy as np
from pythonlib.tools.plottools import savefig
from pythonlib.tools.pandastools import append_col_with_grp_index
from pythonlib.globals import PATH_ANALYSIS_OUTCOMES
import os
import sys
import pandas as pd
from pythonlib.tools.expttools import writeDictToTxt
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import extract_single_pa, load_handsaved_wrapper
import seaborn as sns
from neuralmonkey.analyses.decode_good import preprocess_extract_X_and_labels
from pythonlib.tools.pandastools import append_col_with_grp_index
from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper
from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
import seaborn as sns


def preprocess_pa_to_alignedneuralmotor(PA, lag_neural_vs_beh):
    """
    For each trial, get neural and stroke data, aligned, and placed into a new dataframe, wjhere eacjh
    row is a trial. Do this for both position and velocity data.
    """
    from pythonlib.tools.nptools import isnear

    # First, extract all data, before doing any train-test splits
    res = []
    for i in range(len(PA.Trials)):
        pa1, strok1 = PA.behavior_extract_neural_stroke_aligned(i, lag_neural_vs_beh, var_strok="strok_beh", PLOT=False)
        pa2, strok2 = PA.behavior_extract_neural_stroke_aligned(i, lag_neural_vs_beh, var_strok="strok_beh_vel", PLOT=False)
        try:
            assert isnear(pa1.X, pa2.X)
        except Exception as err:
            print(" ---- ")
            print(pa1.X)
            print(pa2.X)
            print(type(pa1.X))
            print(type(pa2.X))
            raise err
        res.append({
            "ind_trial":i,
            "x_neural":pa1.X[:, 0, :],
            "strok_beh":strok1,
            "strok_beh_vel":strok2,
        })
    DFRES = pd.DataFrame(res)
    DFLAB = PA.Xlabels["trials"]
    return DFRES, DFLAB

def preprocess_alignedneuralmotor_to_flattened(dfres, dflab_orig, variables_take = None):
    """
    Convert from aligned-neural-motor data (dfres) to flattened data that can then pass into regression
    analyses. Also helps extract categorical variables to match shape of raw (timebin x trial) neural and 
    motor data.

    PARAMS:
    - dflab_orig, this is the original, with rows matching the indices in 
    dfres["ind_trial"]. Therefore, if dfres is a subset of data (e.g,, for train-test split)
    you should still pass in the original dflab.
    Returns:
    - X, (ndat, nchans)
    - Ypos, (ndat, 2)
    - Yvel, (ndat, 2)
    """
    import numpy as np
    X = np.concatenate(dfres["x_neural"], axis=1).T # (ndat, nchans)
    Ypos = np.concatenate(dfres["strok_beh"], axis=0)[:, :2] # ndat, 2
    Yvel = np.concatenate(dfres["strok_beh_vel"], axis=0)[:, :2] # ndat, 2
    assert X.shape[0] == Ypos.shape[0] == Yvel.shape[0]

    # Also optionally get varialbes, repeated to same length
    if variables_take is not None:

        # Get indices
        # assert len(dfres)==len(dflab)
        trials = []
        for _, row in dfres.iterrows():
            ind_trial = row["ind_trial"]
            x_neural = row["x_neural"]
            trials.extend([ind_trial for _ in range(x_neural.shape[1])])

        # Get variables
        variables = {}
        for var_take in variables_take:
            variables[var_take] = dflab_orig.iloc[trials][var_take].tolist()
            assert len(variables[var_take]) == X.shape[0]
    else:
        variables = None

    return X, Ypos, Yvel, variables

def preprocess_convert_flattened_to_dataregress(neural, pos, vel, variables_dict):
    """
    Convert flattened data to data structure useful as input to regression, with
    formatting for statsmodel
    INPUT DATA are all shape (ntrials*timebins, ndims), ie is concatenated across
    all trials, along the timebin dimension

    """

    data = pd.DataFrame({"vel_x":vel[:, 0], "vel_y":vel[:, 1], "pos_x":pos[:,0], "pos_y":pos[:, 1]})

    # Get each neural chan
    nchans = neural.shape[1]
    for i in range(nchans):
        data[f"neural_{i}"] = neural[:, i]

    # Add other variables
    if variables_dict is not None:
        for var in variables_dict:
            data[var] = variables_dict[var]
            
    return data

def score_wrapper(DFRES, DFLAB, inds_train, inds_test, method, beh_variables, PRINT=False, DEBUG=False):
    """
    WRapper to compute regression score across train-test splits, with particular params, returning 
    r2, both train and test, and agged across all dimensions (if doing decoding) the proper way, using the
    outputed values and predictions.

    PARAMS:
    - inds_train and inds_test, row indices into DFRES/DFLAB.
    - method, either "encoding" or "decoding"
    - beh_variables, list of str, which variables to use
    RETURNS:
    - resthis, list of dict, holding results across dimensions of Y.
    - r2_train_all, r2_test_all, agging across dimesnions of Y (the proper way, using vals and predicted)
    """
    from neuralmonkey.analyses.regression_good import fit_and_score_regression
    from pythonlib.tools.statstools import coeff_determination_R2
    from neuralmonkey.analyses.regression_good import fit_and_score_regression_with_categorical_predictor
    from neuralmonkey.scripts.analy_motor_decode_encode import preprocess_alignedneuralmotor_to_flattened, preprocess_convert_flattened_to_dataregress

    assert len(DFRES)==len(DFLAB)
    assert beh_variables is not None, "for now, shouild pass in."

    variables_categorical = ["seqc_0_shape", "seqc_0_loc"]
    variables_motor = ["pos_x", "pos_y", "vel_x", "vel_y"]
    for v in beh_variables:
        assert v in variables_categorical + variables_motor, "you want to inlucde a variable in regression which will not be part of data"

    ## Checks
    if method=="decoding":
        for v in beh_variables:
            assert v not in variables_categorical, "decoding only predicts continuosu variables."
            
    ### Extract data in arrays
    # Extract train/test splits
    dfres_train = DFRES.iloc[inds_train].reset_index(drop=True)
    dfres_test = DFRES.iloc[inds_test].reset_index(drop=True)
    neural_train, pos_train, vel_train, variables_dict_train = preprocess_alignedneuralmotor_to_flattened(dfres_train, DFLAB, variables_take=variables_categorical)
    neural_test, pos_test, vel_test, variables_dict_test = preprocess_alignedneuralmotor_to_flattened(dfres_test, DFLAB, variables_take=variables_categorical)
    
    ### Formatting
    data_train = preprocess_convert_flattened_to_dataregress(neural_train, pos_train, vel_train, variables_dict_train)
    data_test = preprocess_convert_flattened_to_dataregress(neural_test, pos_test, vel_test, variables_dict_test)

    ### COLLECT DATA
    resthis = []
    y_train_all = np.empty((0,))
    y_train_pred_all = np.empty((0,))
    y_test_all = np.empty((0,))
    y_test_pred_all = np.empty((0,))

    list_ss_resid_train = []
    list_ss_tot_train = []
    list_ss_resid_test = []
    list_ss_tot_test = []

    if method=="decoding":
        # Decoding model
        for dim_y in range(len(beh_variables)):
            # Method: statsmodel
            y_var = beh_variables[dim_y]
            nchans = neural_train.shape[1]
            x_vars = [f"neural_{_i}" for _i in range(nchans)]
            x_vars_is_cat = [False for _ in range(nchans)]
            dict_coeff, model, original_feature_mapping, results = fit_and_score_regression_with_categorical_predictor(
                            data_train, y_var, x_vars, x_vars_is_cat, data_test=data_test, PRINT=False)
            if DEBUG:
                print(y_var)
                print(x_vars)
                display(results)
                print(model.summary())
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                ax = axes.flatten()[0]
                ax.plot(y_train, y_train_pred, "xk", alpha=0.05)
                
                ax = axes.flatten()[1]
                ax.plot(y_test, y_test_pred, "xk", alpha=0.05)
                assert False

            r2_train = results["r2_train"]
            r2_test = results["r2_test"]
            y_train = results["y_train"] 
            y_test = results["y_test"]
            y_train_pred = results["y_train_pred"] 
            y_test_pred = results["y_test_pred"] 

            resthis.append({
                "inds_train":inds_train,
                "inds_test":inds_test,
                "dim_y":dim_y,
                "r2_train":r2_train,
                "r2_test":r2_test,
                "ss_resid_train":results["ss_resid_train"],
                "ss_tot_train":results["ss_tot_train"],
                "ss_resid_test":results["ss_resid_test"],
                "ss_tot_test":results["ss_tot_test"],
            })

            y_train_all = np.concatenate([y_train_all, y_train])
            y_train_pred_all = np.concatenate([y_train_pred_all, y_train_pred])
            y_test_all = np.concatenate([y_test_all, y_test])
            y_test_pred_all = np.concatenate([y_test_pred_all, y_test_pred])

            list_ss_resid_train.append(results["ss_resid_train"])
            list_ss_tot_train.append(results["ss_tot_train"])
            list_ss_resid_test.append(results["ss_resid_test"])
            list_ss_tot_test.append(results["ss_tot_test"])

    elif method=="encoding":
        # Encoding model

        x_vars = beh_variables
        x_vars_is_cat = []
        for xv in x_vars:
            if xv in variables_categorical:
                x_vars_is_cat.append(True)
            else:
                x_vars_is_cat.append(False)

        nchans = neural_train.shape[1]
        for dim_y in range(nchans):
            # Method: statsmodel
            y_var = f"neural_{dim_y}"
            # x_vars = ["vel_x", "vel_y", "pos_x", "pos_y", "seqc_0_shape"]
            # x_vars_is_cat = [False, False, False, False, True]
            # x_vars = ["vel_x", "vel_y", "pos_x", "pos_y"]
            # x_vars_is_cat = [False, False, False, False]

            dict_coeff, model, original_feature_mapping, results = fit_and_score_regression_with_categorical_predictor(
                            data_train, y_var, x_vars, x_vars_is_cat, data_test=data_test, PRINT=False)
            
                
            if DEBUG:
                print(x_vars)
                print(x_vars_is_cat)
                display(results)
                print(model.summary())
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                ax = axes.flatten()[0]
                ax.plot(y_train, y_train_pred, "xk")
                
                ax = axes.flatten()[1]
                ax.plot(y_test, y_test_pred, "xk")
                assert False

            r2_train = results["r2_train"]
            r2_test = results["r2_test"]
            y_train = results["y_train"] 
            y_test = results["y_test"]
            y_train_pred = results["y_train_pred"] 
            y_test_pred = results["y_test_pred"] 


            # # Demean before collecting across dimensions, to be fair, since predicted values are allowed to have
            # # different intercepts per dimension.
            # y_train_mean = np.mean(y_train)
            # y_train -= y_train_mean
            # y_train_pred -= y_train_mean

            # y_test_mean = np.mean(y_test)
            # y_test -= y_test_mean
            # y_test_pred -= y_test_mean

            if False:
                # Compare results to sklearn  method,. This assumes that x are the four kinematic variable.
                posvel_train = data_train.loc[:, variables_motor].values
                posvel_test = data_test.loc[:, variables_motor].values
                reg, _, _, y_train2, y_test2, y_train_pred2, y_test_pred2 = fit_and_score_regression(
                                                    posvel_train, neural_train[:, dim_y], posvel_test, neural_test[:, dim_y], 
                                                    version="ols", PRINT=PRINT, also_return_predictions=True)
                # Plot residuals.
                fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
                ax = axes.flatten()[0]
                ax.plot(y_train, y_train_pred, "xk", alpha=0.05)
                ax.set_title(f"means: {np.mean(y_train):.2f}, {np.mean(y_train_pred):.2f}")
                ax = axes.flatten()[1]
                ax.plot(y_train2, y_train_pred2, "xk", alpha=0.05)
                ax.set_title(f"means: {np.mean(y_train2):.2f}, {np.mean(y_train_pred2):.2f}")
                assert False

            y_train_all = np.concatenate([y_train_all, y_train])
            y_train_pred_all = np.concatenate([y_train_pred_all, y_train_pred])
            y_test_all = np.concatenate([y_test_all, y_test])
            y_test_pred_all = np.concatenate([y_test_pred_all, y_test_pred])

            list_ss_resid_train.append(results["ss_resid_train"])
            list_ss_tot_train.append(results["ss_tot_train"])
            list_ss_resid_test.append(results["ss_resid_test"])
            list_ss_tot_test.append(results["ss_tot_test"])

            resthis.append({
                "inds_train":inds_train,
                "inds_test":inds_test,
                "dim_y":dim_y,
                "r2_train":r2_train,
                "r2_test":r2_test,
                "ss_resid_train":results["ss_resid_train"],
                "ss_tot_train":results["ss_tot_train"],
                "ss_resid_test":results["ss_resid_test"],
                "ss_tot_test":results["ss_tot_test"],
            })  

            # To compare scale (i.e. variance) of residuals, uncomment this.
            if False:
                print("---- ", dim_y)
                # print(np.sum((neural_train[:, dim_y] - np.mean(neural_train[:, dim_y]))**2))
                # print(np.sum((y_train - np.mean(y_train))**2))
                # print(np.sum((neural_test[:, dim_y] - np.mean(neural_test[:, dim_y]))**2))
                # print(np.sum((y_train_pred - np.mean(y_train_pred))**2))
                # print(np.sum((y_test_pred - np.mean(y_test_pred))**2))
                residuals = y_test_pred - y_test
                print(np.sum((y_test - np.mean(y_test))**2))
                print(np.sum((residuals - np.mean(residuals))**2))
                
                # fig, ax = plt.subplots()
                # ax.plot(y_test, y_test_pred, "ok", alpha=0.2)
                # assert False
    else:
        print(method)
        assert False
    
    # Method 1
    r2_train_all, _, _ = coeff_determination_R2(y_train_all, y_train_pred_all, doplot=False)
    r2_test_all, _, _ = coeff_determination_R2(y_test_all, y_test_pred_all, doplot=False)
    
    # Method 2 -- run this just to sanity check the computation
    if DEBUG:
        print(" -- here")
        print(list_ss_resid_train)
        print(list_ss_tot_train)
        print(list_ss_resid_test)
        print(list_ss_tot_test)
    r2_train_all_2 = 1 - sum(list_ss_resid_train)/sum(list_ss_tot_train)
    r2_test_all_2 = 1 - sum(list_ss_resid_test)/sum(list_ss_tot_test)
    if np.abs(r2_train_all - r2_train_all_2)>0.01:
        print(r2_train_all, r2_train_all_2)
        print(list_ss_resid_train)
        print(list_ss_tot_train)
        assert False
    if np.abs(r2_test_all - r2_test_all_2)>0.01:
        print(r2_test_all, r2_test_all_2)
        print(list_ss_resid_test)
        print(list_ss_tot_test)
        assert False

    return resthis, r2_train_all, r2_test_all

def _score_obsolete(DFRES, DFLAB, inds_train, inds_test, method, PRINT=False, beh_variables=None):
    """
    obsolete -- this is old sklearn method, whic works, but doesnt handle cartegorical varialbe.s
    
    Get decoding r2
    """
    from neuralmonkey.analyses.regression_good import fit_and_score_regression
    from pythonlib.tools.statstools import coeff_determination_R2
    from neuralmonkey.analyses.regression_good import fit_and_score_regression_with_categorical_predictor
    
    assert False, "this is obsolete"

    assert len(DFRES)==len(DFLAB)
    variables_take = ["seqc_0_shape", "seqc_0_loc"]
    variables_motor = ["pos_x", "pos_y", "vel_x", "vel_y"]
    for v in beh_variables:
        assert v in variables_take + variables_motor, "you want to inlucde a variable in regression which will not be part of data"

    # Extract train/test splits
    dfres_train = DFRES.iloc[inds_train].reset_index(drop=True)
    dfres_test = DFRES.iloc[inds_test].reset_index(drop=True)

    ### Extract data in arrays
    neural_train, pos_train, vel_train, variables_dict_train = preprocess_alignedneuralmotor_to_flattenedraw(dfres_train, DFLAB, variables_take=variables_take)
    neural_test, pos_test, vel_test, variables_dict_test = preprocess_alignedneuralmotor_to_flattenedraw(dfres_test, DFLAB, variables_take=variables_take)
    # neural, pos, vel, variables_dict = _dfres_to_raw(DFRES, DFLAB, variables_take=variables_take)

    ### Formatting
    data_train = _convert_raw_to_df(neural_train, pos_train, vel_train, variables_dict_train)
    data_test = _convert_raw_to_df(neural_test, pos_test, vel_test, variables_dict_test)

    # (2) for sklearn
    posvel_train = np.concatenate([pos_train, vel_train], axis=1) # (n, 4)
    posvel_test = np.concatenate([pos_test, vel_test], axis=1) # (n, 4)

    resthis = []
    y_train_all = np.empty((0,))
    y_train_pred_all = np.empty((0,))
    y_test_all = np.empty((0,))
    y_test_pred_all = np.empty((0,))
    if method=="decoding":
        # Decoding model
        for dim_y in [0, 1, 2, 3]:
            if True:
                reg, r2_train, r2_test, y_train, y_test, y_train_pred, y_test_pred = fit_and_score_regression(
                                                    neural_train, posvel_train[:,dim_y], neural_test, posvel_test[:,dim_y],
                                                    version="ridge", PRINT=PRINT, also_return_predictions=True)
                # print(r2_train, r2_test)
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # ax = axes.flatten()[0]
                # ax.plot(y_train, y_train_pred, "xk", alpha=0.05)
                
                # ax = axes.flatten()[1]
                # ax.plot(y_test, y_test_pred, "xk", alpha=0.05)
            else:
                # Method: statsmodel
                y_vars = ["pos_x", "pos_y", "vel_x", "vel_y"]
                y_var = y_vars[dim_y]
                nchans = neural_train.shape[1]
                x_vars = [f"neural_{_i}" for _i in range(nchans)]
                x_vars_is_cat = [False for _ in range(nchans)]
                dict_coeff, model, original_feature_mapping, results = fit_and_score_regression_with_categorical_predictor(
                                data_train, y_var, x_vars, x_vars_is_cat, data_test=data_test, PRINT=False)
                # display(results)
                # print(model.summary())

                r2_train = results["r2_train"]
                r2_test = results["r2_test"]
                y_train = results["y_train"] 
                y_test = results["y_test"]
                y_train_pred = results["y_train_pred"] 
                y_test_pred = results["y_test_pred"] 
                # print("LEFT OFF HERE -- THESE two methods lead to different r2. Check details for what returned to solve this problem.")
                # print("Then run thru all dates, and diff variable, including shape")

                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # ax = axes.flatten()[0]
                # ax.plot(y_train, y_train_pred, "xk", alpha=0.05)
                
                # ax = axes.flatten()[1]
                # ax.plot(y_test, y_test_pred, "xk", alpha=0.05)
                # assert False

            resthis.append({
                "inds_train":inds_train,
                "inds_test":inds_test,
                "dim_y":dim_y,
                "r2_train":r2_train,
                "r2_test":r2_test,
                "bregion":bregion,
            })

            y_train_all = np.concatenate([y_train_all, y_train])
            y_train_pred_all = np.concatenate([y_train_pred_all, y_train_pred])
            y_test_all = np.concatenate([y_test_all, y_test])
            y_test_pred_all = np.concatenate([y_test_pred_all, y_test_pred])

    elif method=="encoding":
        # Encoding model
        nchans = neural_train.shape[1]
        for dim_y in range(nchans):
            if True:
                # Method: sklearn
                reg, r2_train, r2_test, y_train, y_test, y_train_pred, y_test_pred = fit_and_score_regression(
                                                    posvel_train, neural_train[:, dim_y], posvel_test, neural_test[:, dim_y], 
                                                    version="ols", PRINT=PRINT, also_return_predictions=True)
                # print(r2_train, r2_test)
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # ax = axes.flatten()[0]
                # ax.plot(y_train, y_train_pred, "xk")
                
                # ax = axes.flatten()[1]
                # ax.plot(y_test, y_test_pred, "xk")
                
            else:
                # Method: statsmodel
                y_var = f"neural_{dim_y}"
                # x_vars = ["vel_x", "vel_y", "pos_x", "pos_y", "seqc_0_shape"]
                # x_vars_is_cat = [False, False, False, False, True]
                x_vars = ["vel_x", "vel_y", "pos_x", "pos_y"]
                x_vars_is_cat = [False, False, False, False]
                dict_coeff, model, original_feature_mapping, results = fit_and_score_regression_with_categorical_predictor(
                                data_train, y_var, x_vars, x_vars_is_cat, data_test=data_test, PRINT=False)
                # display(results)
                # print(model.summary())

                r2_train = results["r2_train"]
                r2_test = results["r2_test"]
                y_train = results["y_train"] 
                y_test = results["y_test"]
                y_train_pred = results["y_train_pred"] 
                y_test_pred = results["y_test_pred"] 

                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # ax = axes.flatten()[0]
                # ax.plot(y_train, y_train_pred, "xk")
                
                # ax = axes.flatten()[1]
                # ax.plot(y_test, y_test_pred, "xk")

                # assert False, "check it"

            resthis.append({
                "inds_train":inds_train,
                "inds_test":inds_test,
                "dim_y":dim_y,
                "r2_train":r2_train,
                "r2_test":r2_test,
                "bregion":bregion,
            })  

            y_train_all = np.concatenate([y_train_all, y_train])
            y_train_pred_all = np.concatenate([y_train_pred_all, y_train_pred])
            y_test_all = np.concatenate([y_test_all, y_test])
            y_test_pred_all = np.concatenate([y_test_pred_all, y_test_pred])

            # To compare scale (i.e. variance) of residuals, uncomment this.
            if False:
                print("---- ", dim_y)
                # print(np.sum((neural_train[:, dim_y] - np.mean(neural_train[:, dim_y]))**2))
                # print(np.sum((y_train - np.mean(y_train))**2))
                # print(np.sum((neural_test[:, dim_y] - np.mean(neural_test[:, dim_y]))**2))
                # print(np.sum((y_train_pred - np.mean(y_train_pred))**2))
                # print(np.sum((y_test_pred - np.mean(y_test_pred))**2))
                residuals = y_test_pred - y_test
                print(np.sum((y_test - np.mean(y_test))**2))
                print(np.sum((residuals - np.mean(residuals))**2))
                
                # fig, ax = plt.subplots()
                # ax.plot(y_test, y_test_pred, "ok", alpha=0.2)
                # assert False
    else:
        print(method)
        assert False
        
    r2_train_all, _, _ = coeff_determination_R2(y_train_all, y_train_pred_all, doplot=False)
    r2_test_all, _, _ = coeff_determination_R2(y_test_all, y_test_pred_all, doplot=False)

    return resthis, r2_train_all, r2_test_all


def plot_all(DFDECODE, DFDECODE_COMB, savedir):
    """
    GEneral wrapper for all plots using the results from 
    """
    from pythonlib.tools.pandastools import append_col_with_grp_index, stringify_values
    DFDECODE = append_col_with_grp_index(DFDECODE, ["traintest_method", "score_method", "beh_variables_code"], "method")
    DFDECODE_COMB = append_col_with_grp_index(DFDECODE_COMB, ["traintest_method", "score_method", "beh_variables_code"], "method")
    DFDECODE = stringify_values(DFDECODE)
    DFDECODE_COMB = stringify_values(DFDECODE_COMB)
    
    from pythonlib.tools.plottools import savefig
    for var_score in ["r2_train", "r2_test"]:
        for dfdecode_kind in ["DFDECODE", "DFDECODECOMB"]:
                if dfdecode_kind=="DFDECODE":
                    dfdecode = DFDECODE
                elif dfdecode_kind=="DFDECODECOMB":
                    dfdecode = DFDECODE_COMB
                else:
                    assert False

                if "dim_y" in dfdecode:
                    fig = sns.relplot(data=dfdecode, x="lag_neural_vs_beh", y=var_score, hue="bregion", kind="line", errorbar="se", col="dim_y",
                                row="method")
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/df={dfdecode_kind}-var={var_score}-relplot-1.pdf")

                    fig = sns.catplot(data=dfdecode, x="bregion", y=var_score, hue="dim_y", kind="bar", col="method")
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/df={dfdecode_kind}-var={var_score}-catplot-2.pdf")

                    if "grp_test" in dfdecode:
                        fig = sns.catplot(data=dfdecode, x="bregion", y=var_score, hue="dim_y", kind="bar", col="grp_test", row="method")
                        for ax in fig.axes.flatten():
                                        ax.axhline(0, color="k", alpha=0.5)
                        savefig(fig, f"{savedir}/df={dfdecode_kind}-var={var_score}-catplot-4.pdf")
                        
                fig = sns.relplot(data=dfdecode, x="lag_neural_vs_beh", y=var_score, hue="bregion", kind="line", errorbar="se", col="method")
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/df={dfdecode_kind}-var={var_score}-relplot-2.pdf")

                fig = sns.catplot(data=dfdecode, x="bregion", y=var_score, kind="bar", col="method")
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                savefig(fig, f"{savedir}/df={dfdecode_kind}-var={var_score}-catplot-3.pdf")

                plt.close("all")

def analy_decode_encode_wrapper(DFallpa, SAVEDIR):
    """
    Overall wrapper to perform all decoding and encoding analyses, saving results for each set of metaprams.
    """
    # Split into train and test based on shapes
    from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
    from pythonlib.tools.nptools import isnear
    import numpy as np
    from neuralmonkey.scripts.analy_motor_decode_encode import preprocess_pa_to_alignedneuralmotor, score_wrapper, plot_all
    from pythonlib.tools.expttools import writeDictToTxtFlattened

    ### Extract motor behavior
    # # Get velocities
    # PA.behavior_strokes_kinematics_stats()
    # Convert strokes to instantaneous velocities

    # from pythonlib.tools.stroketools import strokes_bin_velocity_wrapper
    # strokes_bin_velocity_wrapper(strokes)
    # Get strokes as velocities
    from pythonlib.tools.stroketools import strokesVelocity, sample_rate_equalize_across_strokes
    for PA in DFallpa["pa"]:
        PA.behavior_extract_strokes_to_dflab(trial_take_first_stroke=True)
        dflab = PA.Xlabels["trials"]
        strokes = dflab["strok_beh"].tolist()
        # First equalize their fs
        strokes = sample_rate_equalize_across_strokes(strokes)
        strokes_vels, _ = strokesVelocity(strokes, None)
        dflab["strok_beh"] = strokes # Replace, so that matches lengths of each strok in strokes_vels
        dflab["strok_beh_vel"] = strokes_vels
        PA.Xlabels["trials"] = dflab

    ### RUN
    NPCS_KEEP = 10                    
    nsplits = 5
    test_size = 0.2
    assert nsplits==5 and test_size==0.2, "this works, in that there is enough data to have at least one of each shape for each split."

    # beh_variables = ["vel_x", "vel_y", "pos_x", "pos_y", "seqc_0_shape"]
    # beh_variables = ["vel_x", "vel_y", "pos_x", "pos_y"]

    # traintest_method = 2
    # score_method = "encoding"
    # bregions_get = ["M1", "PMv"]
    bregions_get = None
    list_time_lag = np.arange(-0.3, 0.3, 0.05)
    # list_time_lag = [0.]
    for _, row in DFallpa.iterrows():
        PA = row["pa"]
        bregion = row["bregion"]
        event = row["event"]
        assert event in ["00_stroke", "06_on_strokeidx_0"]

        if bregions_get is not None and bregion not in bregions_get:
            continue

        ### Do dim reduction first, esp for encoding model
        scalar_or_traj = "traj"
        dim_red_method = "pca"
        savedir_this = f"{SAVEDIR}/pca-{bregion}-{event}"
        os.makedirs(savedir_this, exist_ok=True)
        twind_pca = (PA.Times[0]-0.01, PA.Times[-1]+0.01)
        tbin_slide = 0.01
        _, PA = PA.dataextract_dimred_wrapper(scalar_or_traj, dim_red_method, savedir_this, 
                                        twind_pca, tbin_dur="default", tbin_slide=tbin_slide, 
                                        NPCS_KEEP = NPCS_KEEP,
                                        n_min_per_lev_lev_others = 2,
                                        return_pca_components=False)
        plt.close("all")

        # Prune to just shapes that have data
        from pythonlib.tools.pandastools import extract_with_levels_of_var_good, grouping_print_n_samples
        dflab = PA.Xlabels["trials"]
        
        grouping_print_n_samples(dflab, ["seqc_0_shape", "seqc_0_loc", "gridsize"], savepath=f"{savedir_this}/counts-1-pre.txt")
        grouping_print_n_samples(dflab, ["seqc_0_shape"], savepath=f"{savedir_this}/counts-2-pre.txt")
        
        # _, inds_keep_1 = extract_with_levels_of_var_good(dflab, ["seqc_0_shape", "seqc_0_loc", "gridsize"], n_min_per_var=3)
        # _, inds_keep_2 = extract_with_levels_of_var_good(dflab, ["seqc_0_shape"], n_min_per_var=5)
        
        # inds_keep = sorted([i for i in inds_keep_1 if i in inds_keep_2])

        _, inds_keep = extract_with_levels_of_var_good(dflab, ["seqc_0_shape"], n_min_per_var=nsplits)
        PA = PA.slice_by_dim_indices_wrapper("trials", inds_keep)

        grouping_print_n_samples(PA.Xlabels["trials"], ["seqc_0_shape", "seqc_0_loc", "gridsize"], savepath=f"{savedir_this}/counts-1-clean.txt")
        grouping_print_n_samples(PA.Xlabels["trials"], ["seqc_0_shape"], savepath=f"{savedir_this}/counts-2-clean.txt")

        # print(len(inds_keep_1), inds_keep_1)
        # print(len(inds_keep_2), inds_keep_2)
        # print(len(inds_keep), inds_keep)
        # assert False

        ### ITERATE over all metaparams
        for traintest_method in [1, 2]:
            for score_method in ["encoding", "decoding"]:
                for beh_variables, beh_variables_code in [
                    [("vel_x", "vel_y"), "vels"],
                    [("pos_x", "pos_y", "vel_x", "vel_y"), "motor"],
                    [("seqc_0_shape",), "shape"],
                    [("pos_x", "pos_y", "vel_x", "vel_y", "seqc_0_shape"), "motor_shape"],
                    ]:
        # for traintest_method in [2]:
        #     for score_method in ["encoding"]:
        #         for beh_variables, beh_variables_code in [
        #             [("seqc_0_shape",), "shape"],
        #             ]:

                    # Ignore certain combinations
                    # - Cannot decode categorical
                    if score_method=="decoding" and beh_variables_code in ["shape", "motor_shape"]:
                        continue
                    # - Cannot do cross-shape generalization if shape is included in variables
                    if traintest_method==1 and beh_variables_code in ["shape", "motor_shape"]:
                        continue

                    savedir = f"{SAVEDIR}/ttsplitmeth={traintest_method}-scoremeth={score_method}-behvar={beh_variables_code}/{bregion}-{event}"
                    os.makedirs(savedir, exist_ok=True)
                    print(savedir)
                    writeDictToTxtFlattened({
                        "NPCS_KEEP":NPCS_KEEP,
                        "list_time_lag":list_time_lag,
                        "beh_variables":beh_variables,
                        "beh_variables_code":beh_variables_code,
                    }, f"{savedir}/params.txt")

                    res_all = []
                    res_all_combdim = []
                    for lag_neural_vs_beh in list_time_lag:

                        ### RUN for this bregion
                        # First, extract all data, before doing any train-test splits
                        DFRES, DFLAB = preprocess_pa_to_alignedneuralmotor(PA, lag_neural_vs_beh)
                        # res = []
                        # for i in range(len(PA.Trials)):
                        #     pa1, strok1 = PA.behavior_extract_neural_stroke_aligned(i, lag_neural_vs_beh, var_strok="strok_beh", PLOT=False)
                        #     pa2, strok2 = PA.behavior_extract_neural_stroke_aligned(i, lag_neural_vs_beh, var_strok="strok_beh_vel", PLOT=False)
                        #     assert isnear(pa1.X, pa2.X)
                        #     res.append({
                        #         "ind_trial":i,
                        #         "x_neural":pa1.X[:, 0, :],
                        #         "strok_beh":strok1,
                        #         "strok_beh_vel":strok2,
                        #     })
                        # DFRES = pd.DataFrame(res)
                        # DFLAB = PA.Xlabels["trials"]

                        ########## TRAIN-TEST SPLITS
                        if traintest_method == 1:
                            ### METHOD 1 -- split by variable (e.g, shape, and test generalization to other shapes)
                            # Second, do train-test splits
                            dflab = PA.Xlabels["trials"]
                            grpdict = grouping_append_and_return_inner_items_good(dflab, ["seqc_0_shape"])
                            for grp, inds in grpdict.items():
                                inds_test = inds
                                inds_train = [i for i in range(len(dflab)) if i not in inds_test]
                                # print("n train/test: ", len(inds_train), len(inds_test)) 

                                resthis, r2_train_all, r2_test_all = score_wrapper(DFRES, DFLAB, inds_train, inds_test, method=score_method, beh_variables=beh_variables)
                                for r in resthis:
                                    r["grp_test"] = grp
                                    r["bregion"] = bregion
                                    r["event"] = event
                                    r["lag_neural_vs_beh"] = lag_neural_vs_beh
                                    r["traintest_method"] = traintest_method
                                    r["score_method"] = score_method
                                    r["beh_variables_code"] = beh_variables_code
                                res_all.extend(resthis)

                                res_all_combdim.append({
                                    "grp_test":grp,
                                    "bregion":bregion,
                                    "event":event,
                                    "lag_neural_vs_beh":lag_neural_vs_beh,
                                    "traintest_method":traintest_method,
                                    "score_method":score_method,
                                    "beh_variables_code":beh_variables_code,
                                    "r2_train":r2_train_all,
                                    "r2_test":r2_test_all,
                                })

                        elif traintest_method==2:
                            ### METHOD 2 -- train-test splits at level of trials, stratified across levels of given variable.
                            label_grp_vars = ["seqc_0_shape", "seqc_0_loc", "gridsize"]
                            plot_train_test_counts = True
                            min_frac_datapts_unconstrained = 0.01 
                            n_constrained = 1 #
                            folds, fig_unc, fig_con = PA.split_stratified_constrained_grp_var(nsplits, label_grp_vars, 
                                                                                              n_constrained=n_constrained,
                                                                                            fraction_constrained_set=test_size, 
                                                                                            min_frac_datapts_unconstrained=min_frac_datapts_unconstrained,
                                                                                            plot_train_test_counts=plot_train_test_counts)
                            savefig(fig_unc, f"{savedir}/ttsplitmeth=2-counts-unc.pdf")
                            savefig(fig_con, f"{savedir}/ttsplitmeth=2-counts-con.pdf")
                            print("Saving ttsplit counts to: ", f"{savedir}/ttsplitmeth=2-counts-con.pdf")
                            plt.close("all")

                            for inds_train, inds_test in folds:               
                                # print("n train/test: ", len(inds_train), len(inds_test)) 

                                resthis, r2_train_all, r2_test_all = score_wrapper(DFRES, DFLAB, inds_train, inds_test, method=score_method, beh_variables=beh_variables)
                                for r in resthis:
                                    r["bregion"] = bregion
                                    r["event"] = event
                                    r["lag_neural_vs_beh"] = lag_neural_vs_beh
                                    r["traintest_method"] = traintest_method
                                    r["score_method"] = score_method
                                    r["beh_variables_code"] = beh_variables_code
                                res_all.extend(resthis)

                                res_all_combdim.append({
                                    "bregion":bregion,
                                    "event":event,
                                    "lag_neural_vs_beh":lag_neural_vs_beh,
                                    "traintest_method":traintest_method,
                                    "score_method":score_method,
                                    "beh_variables_code":beh_variables_code,
                                    "r2_train":r2_train_all,
                                    "r2_test":r2_test_all,
                                })
                        else:
                            assert False

                    # SAVE INTERIM
                    DFDECODE = pd.DataFrame(res_all)
                    DFDECODE_COMB = pd.DataFrame(res_all_combdim)

                    DFDECODE.to_pickle(f"{savedir}/DFDECODE.pkl")
                    DFDECODE_COMB.to_pickle(f"{savedir}/DFDECODE_COMB.pkl")
                    plt.close("all")

                    # Make plots
                    savedir_plots = f"{savedir}/plots"
                    os.makedirs(savedir_plots, exist_ok=True)
                    plot_all(DFDECODE, DFDECODE_COMB, savedir_plots)

if __name__=="__main__":

    from neuralmonkey.scripts.analy_dfallpa_extract import extract_dfallpa_helper
    from neuralmonkey.classes.population_mult import load_handsaved_wrapper, dfpa_match_chans_across_pa_each_bregion
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.metadat.analy.anova_params import params_getter_euclidian_vars
    from neuralmonkey.classes.population_mult import dfpa_concatbregion_preprocess_clean_bad_channels, dfpa_concatbregion_preprocess_wrapper
    from pythonlib.tools.pandastools import append_col_with_grp_index
    import seaborn as sns
    from pythonlib.tools.plottools import savefig
    import os
    from neuralmonkey.classes.population_mult import extract_single_pa
    from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single, euclidian_distance_compute_trajectories

    animal = sys.argv[1]
    date = int(sys.argv[2])

    # animal = "Diego"
    # date = 230615
    combine = True
    question = "SP_BASE_trial"
    version = "trial"
    
    # Load a single DFallPA

    ### Load and preprocess
    DFallpa = load_handsaved_wrapper(animal, date, version=version, combine_areas=combine, question=question)

    DFallpa = DFallpa[DFallpa["event"] == "06_on_strokeidx_0"].reset_index(drop=True)

    dfpa_concatbregion_preprocess_wrapper(DFallpa, animal, date)

    ### RUN
    SAVEDIR_ANALYSIS = f"/lemur2/lucas/analyses/recordings/main/MOTOR_DECODE_ENCODE/{animal}-{date}-combine={combine}-wl={version}"
    os.makedirs(SAVEDIR_ANALYSIS, exist_ok=True)
    print(SAVEDIR_ANALYSIS)
    
    analy_decode_encode_wrapper(DFallpa, SAVEDIR_ANALYSIS)
