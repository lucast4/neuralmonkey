""" Moment by moment decoding  -- e.g.,, for each tgime bin, get probabiltiy of each label, can loook at
within indiv trail, dynamics of activation
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pythonlib.tools.plottools import savefig
from pythonlib.tools.listtools import sort_mixed_type
from pythonlib.tools.pandastools import append_col_with_grp_index
import seaborn as sns
from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap

# TBIN_DUR = 0.15
# TBIN_SLIDE = 0.02

TBIN_DUR = 0.2
TBIN_SLIDE = 0.01

class Decoder():
    def __init__(self, PAtrain, var_decode, twind_train):
        """
        Holds a single PA and decoder instance, which comprises variable,
        twindow for decoder training, and all other aspects of preprocessing 
        relevant for training a single deceoer istance. 
        For decoer variations, just make multiple Decoder() instances.
        """
        self.PAtrain = PAtrain
        assert PAtrain.X.shape[2]==1, "must pass in (nchans, ndatapts, 1) -- i.e. already time-averaged"
        
        self.VarDecode = var_decode
        self.Params = {
            "twind_train":twind_train
        }

        # Store some params
        self.LabelsUnique = sort_mixed_type(self.PAtrain.Xlabels["trials"][self.VarDecode].unique().tolist())
        # Initialize
        self.LabelsDecoderGood = None

    def train_decoder(self, PLOT=False, do_upsample_balance=True, 
                      do_upsample_balance_fig_path_nosuff=None, classifier_version = "logistic"):
        """ Train a decoder and store it in self
        PARAMS:
        - do_upsample_balance, bool, whether to upsample lower N classes.
        """

        from neuralmonkey.analyses.decode_good import decode_train_model
        import numpy as np
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import MultiLabelBinarizer


        var_decode = self.VarDecode
        twind_train = self.Params["twind_train"]
        self.ClassifierVer = classifier_version

        ### Prepare training data
        if self.PAtrain.X.shape[2]>1:
            # Then you want to average over time
            # Avg over time --> vector
            pathis = self.PAtrain.slice_by_dim_values_wrapper("times", twind_train) 
            pathis = pathis.agg_wrapper("times")
        else:
            # you have already passed in data shaped (chans, trials, 1)
            pathis = self.PAtrain

        X = pathis.X.squeeze(axis=2).T # (ntrials, nchans)
        # times = pathis.Times
        dflab = pathis.Xlabels["trials"]
        labels = dflab[var_decode].tolist() 
        assert X.shape[0] == len(labels)

        if PLOT == False:
            do_upsample_balance_fig_path_nosuff = None
        
        if do_upsample_balance:
            from pythonlib.tools.listtools import tabulate_list
            from neuralmonkey.analyses.decode_good import decode_upsample_dataset, decode_resample_balance_dataset
            print("Upsampling dataset...")
            print("... starting distribution: ", tabulate_list(labels), X.shape)
            X, labels = decode_resample_balance_dataset(X, labels, "upsample", do_upsample_balance_fig_path_nosuff)
            # X, labels = decode_upsample_dataset(X, labels, do_upsample_balance_fig_path_nosuff)
            print("... ending distribution: ", tabulate_list(labels), X.shape)

        if len(X.shape)==1:
            print(X.shape)
            print(labels)
            print(len(labels))
            print(pathis.X.shape)
            assert False, "not sure why. will run into error"

        if False:
            # Stack presamp (label="presamp") and postsamp (label="shape X")
            twind = [-0.8, -0.1]
            pathis_presamp = pa.slice_by_dim_values_wrapper("times", twind)
            pathis_presamp = pathis_presamp.agg_wrapper("times")

            X_presamp = pathis_presamp.X.squeeze().T

            # And add a label (pre-samp) by concatenating to post-samp data
            X = np.concatenate([X, X_presamp], axis=0)
            _labels = ["presamp" for _ in range(X_presamp.shape[0])]
            labels += _labels

        # print(X.shape)
        # print(len(labels))

        # Plot data
        if PLOT and False:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(X.T)

        if False:
            # Train decoders

            mod = decode_train_model(X, labels, do_center=True)

            clf = LogisticRegression().fit(X, labels)


            clf.predict(X[:2, :])

            clf.predict_log_proba(X).shape
            clf.predict_proba(X).shape
            clf.classes_
            np.sum(clf.predict_proba(X), axis=1)

            clf = OneVsRestClassifier(LogisticRegression()).fit(X, labels)

        ##### GOOD - multi-label classificatin
        ### Convert labels (shape strings) into one-hot labels
        mlb = MultiLabelBinarizer()

        # convert to list of tuples
        labels_tuples = [tuple([x]) for x in labels]

        # Return array of one-hots 
        labels_mlb = mlb.fit_transform(labels_tuples)

        if False:
            if PLOT:
                for x, y in zip(labels, labels_mlb):
                    print(x, y)

        # These are the classes
        print("Classes, in order: ", mlb.classes_)

        # (2) Fit classifier
        if classifier_version == "logistic":
            from sklearn.linear_model import LogisticRegression
            classifier = LogisticRegression
        elif classifier_version == "naive_bayes":
            from sklearn.naive_bayes import GaussianNB
            classifier = GaussianNB
        elif classifier_version == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            classifier = LinearDiscriminantAnalysis
        else:
            assert False
        clf = OneVsRestClassifier(classifier()).fit(X, labels_mlb)
        # np.max(clf.predict_proba(X), axis=1)

        if PLOT:
            # Plot probabilites
            probs = clf.predict_proba(X)
            from pythonlib.tools.snstools import heatmap_mat
            fig, ax = plt.subplots(figsize=(5, 10))
            heatmap_mat(probs, ax = ax, annotate_heatmap=False, zlims=(0,1));

            fig, ax = plt.subplots()
            probs_max = np.max(probs, axis=1)
            ax.hist(probs_max, bins=20);

        # SAVE RESULTS
        self.Classifier = clf
        self.MultiLabelBinarizer = mlb
        
        self.MapLabelToIdx = {}
        self.MapIdxToLabel = {}
        for i, lab in enumerate(self.MultiLabelBinarizer.classes_):
            self.MapLabelToIdx[lab] = i
            self.MapIdxToLabel[i] = lab
    
    def _plot_single_trial_helper_color_by(self, color_by, params):
        """
        Helper to return mapping coloring each label, baseed on information for this trial.
        Useful if you want all trials to have same color map, e.g, for single-trial timecourse plots.
        Can take ouput and apply as colormap for all plots across trials.
        PARAMS:
        - color_by, str, how to determine color. 
        - params, list, flex, depends on color_by
        RETURN:
        - map_shape_to_col_this, map from shape (str) to 4-d color array.
        """
        from pythonlib.tools.plottools import color_make_map_discrete_labels
        map_shape_to_col_this = {}

        
        # Get for this trial
        if color_by=="stroke_index_seqc":
            # Color by the sequence of strokes actually drawn on this trial (0,1,2...), 
            # so that first stroke is always same color, etc
            # Converts color for all undrawn shapes to 
            
            shapes_drawn = params[0] # list of shape str
            max_stroke_num = params[1] # across all trials, n strokes maximum
            shapes_to_plot = params[2] # what shapes exist across trials, including those not shown on this trial.
            # This should be the same input across trials, to ensure same coloring.
            
            map_shape_to_color_orig, _, _ = color_make_map_discrete_labels(sorted(shapes_to_plot))
            MAP_INDEX_TO_COL, _, _ = color_make_map_discrete_labels(range(max_stroke_num+1))


            for i, sh in enumerate(shapes_drawn):
                col = MAP_INDEX_TO_COL[i]
                map_shape_to_col_this[sh] = col

            for sh in shapes_to_plot:
                if sh not in map_shape_to_col_this:
                    # Then give it a low-alpah version of its color
                    col = map_shape_to_color_orig[sh].copy()
                    col[3] = 0.25
                    # col[:3] = 0.25*col[:3]
                    col[:3] = 0.5 + 0.5*col[:3]
                    map_shape_to_col_this[sh] = col

            # ct=0
            # for sh in shapes_to_plot:
            #     if sh not in map_shape_to_col_this:
            #         # Then give it a low-alpah version of another color.
            #         # Here the color is meaningless wrt stroke order.
            #         idx_fake = len(shapes_drawn) + ct
            #         col = MAP_INDEX_TO_COL[idx_fake].copy()
            #         col[3] = 0.25
            #         map_shape_to_col_this[sh] = col
            #         ct+=1

        elif color_by=="shape_order_global":
            # Color by global index in sequence for Shape sequence rule, e.g. a rule for shape sequence
            # Also good if there is no suqence, but you just want consistent color for each shape across trials.

            shapes_drawn = params[0] # list of shape str
            shape_sequence = params[1] # Ground truth sequence of shapes.

            MAP_SHAPE_TO_INDEX = {sh:i for i, sh in enumerate(shape_sequence)}
            MAP_INDEX_TO_COL, _, _ = color_make_map_discrete_labels(range(len(shape_sequence)))
            
            for sh in shapes_drawn:
                if sh not in shape_sequence: # ground truth sequeqnce
                    map_shape_to_col_this[sh] = np.array([0.8, 0.8, 0.8, 0.8])
                else:
                    idx = MAP_SHAPE_TO_INDEX[sh]
                    col = MAP_INDEX_TO_COL[idx]
                    map_shape_to_col_this[sh] = col

            # - get the shapes that are not drawn
            for sh in shape_sequence:
                if sh not in map_shape_to_col_this:
                    # Then give it a low-alpah version of its color
                    idx = MAP_SHAPE_TO_INDEX[sh]
                    col = MAP_INDEX_TO_COL[idx].copy()
                    col[3] = 0.3
                    map_shape_to_col_this[sh] = col
        else:
            assert False

        return map_shape_to_col_this



    def _plot_single_trial(self, probs_mat, times, labels=None, map_lab_to_col=None, title=None, 
                           shift_time_dur=None, ax=None, plot_legend=True,
                           labels_to_plot=None, alpha=1.):
        """
        Low-level code for plot timecourse of decode for an example trial.
        PARAMS:
        - probs_mat, array of probs, (ntimes, nclasses)
        - tbin_dur, in sec, window duration for smoothing. Note: use >=0.1 to avoid high noisiness
        - tbin_slide, in sec, for smoothing 
        - labels, matches probs_mat.shape[1]. Needed if you pass in map_lab... or labels_to_plot.
        - map_lab_to_col, dict, from lab -->4d-shape array
        - labels_to_plot, list of labels, plots only these, if not None.
        """
        from pythonlib.tools.plottools import legend_add_manual, makeColors
        
        if labels is None:
            labels = self.LabelsUnique
        assert len(labels) == probs_mat.shape[1]
        assert len(times) == probs_mat.shape[0]

        if map_lab_to_col is not None:
            for lab in labels:
                assert lab in map_lab_to_col
        if labels_to_plot is not None:
            for lab in labels_to_plot:
                assert lab in labels

        # Prepare plot colors, for each label
        if map_lab_to_col is None:
            pcols = makeColors(len(labels))
            map_lab_to_col = {}
            for cl, col in zip(labels, pcols):
                map_lab_to_col[cl] = col

        if shift_time_dur is not None:
            times = times + shift_time_dur

        if ax is None:
            # fig, axes = plt.subplots(2,1, figsize=(12, 6))
            fig, ax = plt.subplots(1,1, figsize=(12, 3))
        else:
            fig = None

        # ax = axes.flatten()[0]
        for i, lab in enumerate(labels):
            if labels_to_plot is not None and lab not in labels_to_plot:    
                continue
            probs = probs_mat[:, i]
            col = map_lab_to_col[lab]
            # ax.plot(times, probs, label=lab, color=col, linewidth=2)
            if len(col)==4:
                alphathis=None
            else:
                alphathis = alpha
            ax.plot(times, probs, label=lab, color=col, alpha=alphathis)
        ax.axvline(0, color="k")

        if title is not None:
            ax.set_title(title)

        if plot_legend:
            ax.legend(loc="best")
            # ax = axes.flatten()[1]
            # legend_add_manual(ax, map_lab_to_col.keys(), map_lab_to_col.values())

        return fig, ax, map_lab_to_col


    def plot_single_trial(self, indtrial, PA=None, tbin_dur=None, tbin_slide=None,
                          map_lab_to_col=None, title=None, shape_var="seqc_0_shape",
                          shift_time_dur=None, ax=None, plot_legend=True,
                          labels_to_plot=None, return_probs_mat=False):
        """
        Plot timecourse of decode for an example trial. Wrapper of _plot_single_trial()
        PARAMS:
        - tbin_dur, in sec, window duration for smoothing. Note: use >=0.1 to avoid high noisiness
        - tbin_slide, in sec, for smoothing
        """
        from pythonlib.tools.plottools import legend_add_manual, makeColors

        PA = self.prepare_pa_dataset(PA, None, tbin_dur, tbin_slide)

        # if tbin_dur is None:
        #     tbin_dur=TBIN_DUR
        # if tbin_slide is None:
        #     tbin_slide=TBIN_SLIDE            

        # if PA is None:
        #     PA = self.PAtrain

        # clf = self.Classifier
        # mlb = self.MultiLabelBinarizer

        # # Smooth the data
        # pathis = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        
        # # Plot timecourse of decode
        # x = pathis.X[:, indtrial, :].T # (ntimes, nchans)
        # probs_mat = clf.predict_proba(x) # (ntimes, nlabels)
        # times = pathis.Times
        probs_mat, times, labels = self.timeseries_score_single(indtrial, PA, labels_to_plot)

        fig, ax, map_lab_to_col = self._plot_single_trial(probs_mat, times, labels, map_lab_to_col, title, 
                           shift_time_dur, ax, plot_legend, labels_to_plot)

        if return_probs_mat:
            return fig, ax, map_lab_to_col, probs_mat, times, labels
        else:
            return fig, ax, map_lab_to_col
        
        # # Prepare plot colors, for each label
        # if map_lab_to_col is None:
        #     pcols = makeColors(len(mlb.classes_))
        #     map_lab_to_col = {}
        #     for cl, col in zip(mlb.classes_, pcols):
        #         map_lab_to_col[cl] = col
        # if False:
        #     # Color by novelty
        #     dflab[dflab["shape_is_novel_all"]==True].index.tolist()
        #     pcols = makeColors(3)

        #     map_lab_to_col = {}
        #     for sh, nov in dflab.loc[:, ["seqc_0_shape", "shape_is_novel_all"]].values:
        #         if nov:
        #             pcolthis =  pcols[0]
        #         else:
        #             pcolthis = pcols[1]
                
        #         if sh not in map_lab_to_col:
        #             map_lab_to_col[sh] = pcolthis
        #         else:
        #             assert np.all(map_lab_to_col[sh] == pcolthis)

        #     map_lab_to_col["presamp"] = pcols[2]

        # if shift_time_dur is not None:
        #     times = times + shift_time_dur

        # if ax is None:
        #     # fig, axes = plt.subplots(2,1, figsize=(12, 6))
        #     fig, ax = plt.subplots(1,1, figsize=(12, 3))
        # else:
        #     fig = None

        # # ax = axes.flatten()[0]
        # for i, lab in enumerate(mlb.classes_):
        #     if labels_to_plot is not None and lab not in labels_to_plot:
        #         continue
        #     probs = probs_mat[:, i]
        #     col = map_lab_to_col[lab]
        #     # ax.plot(times, probs, label=lab, color=col, linewidth=2)
        #     ax.plot(times, probs, label=lab, color=col)
        # ax.axvline(0, color="k")

        # if title is not None:
        #     ax.set_title(title)

        # if plot_legend:
        #     ax.legend(loc="best")
        #     # ax = axes.flatten()[1]
        #     # legend_add_manual(ax, map_lab_to_col.keys(), map_lab_to_col.values())

        # return fig, ax, map_lab_to_col

    def timeseries_score_single(self, indtrial, PA, labels_in_order_keep=None):
        """ 
        Return time-series decode prob for this trial.
        PARAMS:
        - indtrial, index into PA.
        - twind, (t1, t2)
        - labels_in_order_keep, keeps only these lables, and reetursn in this order.
        RETURNS;
        - probs_mat, array of probs, (ntimes, nclasses)
        - times, timestamps.
        - labels_in_order_keep, keeps only these lables, and reetursn in this order.
        """ 

        # if tbin_dur is None:
        #     tbin_dur=TBIN_DUR
        # if tbin_slide is None:
        #     tbin_slide=TBIN_SLIDE            

        # if PA is None:
        #     PA = self.PAtrain

        clf = self.Classifier
        # mlb = self.MultiLabelBinarizer
        
        # if twind is not None:
        #     PA = PA.slice_by_dim_values_wrapper("times", twind)

        # Smooth the data
        # PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        
        # Get timecourse of decode
        x = PA.X[:, indtrial, :].T # (ntimes, nchans)
        probs_mat = clf.predict_proba(x) # (ntimes, nlabels)
        times = PA.Times

        if labels_in_order_keep is not None:
            # Then sort columns of probs_mat to match desired order
            inds_sort = [self.MapLabelToIdx[lab] for lab in labels_in_order_keep]
            probs_mat = probs_mat[:, inds_sort]
            labels = labels_in_order_keep
        else:
            labels = self.LabelsUnique

        return probs_mat, times, labels


    def prepare_pa_dataset(self, PA, twind, tbin_dur, tbin_slide):
        """
        Helper to preprocess PA. Place here since this is done frequenctly, 
        so hould always call this from higher-level (not base, singel trail) code
        """
        
        if PA is None:
            PA = self.PAtrain

        # Slice out a smaller twind.
        if twind is not None:
            PA = PA.slice_by_dim_values_wrapper("times", twind)

        # Smooth the data
        if tbin_dur is None:
            tbin_dur=TBIN_DUR
        if tbin_slide is None:
            tbin_slide=TBIN_SLIDE            
        PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)

        return PA


    def timeseries_score_wrapper(self, PA, twind=None, indtrials=None, 
                                 labels_in_order_keep=None, tbin_dur=None, tbin_slide=None):
        """
        Retrun all activations across these trials, in a matrix of probabilties.
        RETURNS:
        - probs_mat_all, (labels, trials, times)
        - times
        - labels
        """ 

        PA = self.prepare_pa_dataset(PA, twind, tbin_dur, tbin_slide)
        
        if indtrials is None:
            indtrials = list(range(len(PA.Trials)))
                      
        list_probs_mat = []
        times = None
        labels = None
        for ind in indtrials:
            # Collect probs mat across all trials
            probs_mat, _times, _labels = self.timeseries_score_single(ind, PA=PA, labels_in_order_keep=labels_in_order_keep)

            if labels is None:
                labels = _labels
            else:
                assert labels == _labels
            
            if times is None:
                times = _times
            else:
                from pythonlib.tools.checktools import check_objects_identical
                assert check_objects_identical(times, _times)

            # Center probs, because this is concatting across trials, this reduces effect of trial.

            # probs_mat = probs_mat - np.mean(probs_mat, axis=1, keepdims=True)
            list_probs_mat.append(probs_mat)
            
            # # res, lags = _crosscorr_compute_allpairs(probs_mat)
            # # # list_res.append(res)

        # Concatenate all probs mats
        probs_mat_all = np.stack(list_probs_mat) # (trials, times, labels)
        probs_mat_all = np.transpose(probs_mat_all, (2, 0, 1))

        # Return as PA (labels, trials, times)
        from neuralmonkey.classes.population import PopAnal
        dflab = PA.Xlabels["trials"]
        PAprobs = PopAnal(probs_mat_all, times=times, chans=labels, trials=indtrials)
        PAprobs.Xlabels["trials"] = dflab.iloc[indtrials].copy().reset_index(drop=True)
        assert len(PAprobs.Trials) == len(PAprobs.Xlabels["trials"])

        return PAprobs, probs_mat_all, times, labels

    def timeseries_extract_by_inputed_reorder(self, PAprobs, list_labelidx_ordered,
                                              append_leftover_indices=False):
        """
        Extract probs_mat, by flexibly reordering the trials based on user inputed specific orders,
        i.e.,, explciit indices for each trial.

        PARAMS:
        - list_labelidx_ordered, list of list of ints, each inner list a new ordering for that trial.
        e..g, list_labelidx_ordered[12] = [0,3,2] means on trial 12 reorder so that labels 0, 3, and 2 
        are now indices 0, 1,2 in label oreder.
        - append_leftover_indices, if True, then takes mean over all leftover labels, and appends as 
        last index, so output is (nlabels+1, trials, times)
        RETURNS:
        - probs_mat_all_reordered, array of probs, (nlabels, trials, times), where nlabels is the length 
        of inner lists.
        """

        assert len(set([len(x) for x in list_labelidx_ordered]))==1, "all trials must same same length indices, or else cant make rectagle outoput"
        assert len(list_labelidx_ordered) == len(PAprobs.Trials)

        # extract probs_mat in correct order.
        probs_mat_all = PAprobs.X # labels, trials, times
        # dflab_this = PAprobs.Xlabels["trials"]
        labels = PAprobs.Chans

        list_probs_mat = []
        for trial in range(probs_mat_all.shape[1]):
            
            # Extract data
            probs_mat = probs_mat_all[:, trial, :]
            labelidx_ordered = list_labelidx_ordered[trial]
            
            probs_mat_reordered = probs_mat[labelidx_ordered, :] # (ninds, ntimes)

            labelidx_remain = [i for i in range(len(labels)) if i not in labelidx_ordered]
            probs_mat_remain = probs_mat[labelidx_remain, :]
            probs_mat_remain_mean = np.mean(probs_mat[labelidx_remain, :], axis=0)[None, :]

            if append_leftover_indices:
                probs_mat_reordered = np.concatenate([probs_mat_reordered, probs_mat_remain_mean], axis=0) # (ninds+1, ntimes)

            list_probs_mat.append(probs_mat_reordered)

        probs_mat_all_reordered = np.stack(list_probs_mat) # (trials, ninds, times)
        probs_mat_all_reordered = np.transpose(probs_mat_all_reordered, (1, 0, 2))
        return probs_mat_all_reordered
    
    def timeseries_extract_by_shape_drawn_order(self, PAprobs, keep_first_n_strokes=None):
        """
        Helper to plot one curve for each storke index (0, 1, 2,...) -- i.e,. for each trial,
        what was the decode for sahpe that was done first (this is "0") and second ("1") and so on.
        For each trial this may be different shape, but only conisder its stroke index. 

        Used for shape sequence plots, showing the timecourse of decode for the different strokes that will 
        be drawn on that trial.

        PARAMS:
        - PAprobs, returned from self.timeseries_score_wrapper()
        --- Must have thse columns: shapes_drawn, locs_drawn
        --- All trials must have same n strokes drawn, or else cannot concat probs mat and will fail.
        - MAP_INDEX_TO_COL, dict, from index 0, 1, 2... --> 4-d color array
        RETURNS:
        - probs_mat_all_strokeindex, array hodling probs, (stroke_index, trials, times), where the last index in 
        stroke_index dimension is mean over all those not drawn.
        - labels_this, labels for strokes dimension, (0,1, ..., <max_strokes>, "not_drawn")
        - MAP_INDEX_TO_COL, dict: stroke_index --> 4-d array for coloring.
        """
        from pythonlib.tools.plottools import color_make_map_discrete_labels

        # ---- Slice
        probs_mat_all = PAprobs.X # labels, trials, times
        dflab_this = PAprobs.Xlabels["trials"]
        labels = PAprobs.Chans

        if keep_first_n_strokes is None:
            # Auto count n strokes drawn
            tmp = list(set([len(x) for x in dflab_this["shapes_drawn"]]))
            assert len(tmp)==1, "only can run if all trials have same n strokes drawn. see docs for reason"
            nstrokes = tmp[0]
        else:
            assert isinstance(keep_first_n_strokes, int)
            nstrokes = keep_first_n_strokes

        # Get color for stroke indices
        MAP_INDEX_TO_COL, _, _ = color_make_map_discrete_labels(range(nstrokes))
        MAP_INDEX_TO_COL["not_drawn"] = np.array([0.7, 0.7, 0.7, 0.8])

        ### Reorder based on each trial's drawn shape order.
        indtrials = list(range(len(dflab_this)))
        list_probs_mat = []
        PLOT=False
        for trial in indtrials:
            
            # Extract data
            probs_mat = probs_mat_all[:, trial, :]
            shapes_drawn = dflab_this.iloc[trial]["shapes_drawn"]
            locs_drawn = dflab_this.iloc[trial]["locs_drawn"]

            if self.VarDecode == "seqc_0_loc":

                idxs_in_locs_drawn = [labels.index(loc) for loc in locs_drawn]
                idxs_in_locs_not_drawn = [i for i, loc in enumerate(labels) if loc not in locs_drawn]
                assert sorted(idxs_in_locs_drawn + idxs_in_locs_not_drawn) == list(range(len(labels)))

                _probs_mat_drawn = probs_mat[idxs_in_locs_drawn, :]
                # TODO: only keep "not drawn" if it exists. but downstream code expects rectangle array. will fail.
                # if len(idxs_in_locs_not_drawn)>0:
                #     _probs_mat_not_drawn = np.mean(probs_mat[idxs_in_locs_not_drawn, :], axis=0)[None, :]
                # else:
                #     _probs_mat_not_drawn = None
                _probs_mat_not_drawn = np.mean(probs_mat[idxs_in_locs_not_drawn, :], axis=0)[None, :]

                assert len(idxs_in_locs_drawn)==nstrokes
            else:
                # Assume you care about hsapes
            # elif self.VarDecode == "seqc_0_shape":
                ##### Collect probs
                # 1. Shapes that were drawn
                # Pull out indices from probs_mat, in order of shapes drawn
                idxs_in_shapes_drawn = [labels.index(sh) for sh in shapes_drawn]
                # include labels for "not drawn"
                idxs_in_shapes_not_drawn = [i for i, sh in enumerate(labels) if sh not in shapes_drawn]

                # print("---")
                # print(idxs_in_shapes_drawn, idxs_in_shapes_not_drawn)
                # print(list(range(len(labels))))
                if len(shapes_drawn)==len(set(shapes_drawn)): # Then each shape on this trial is unique
                    assert sorted(idxs_in_shapes_drawn + idxs_in_shapes_not_drawn) == list(range(len(labels)))

                if keep_first_n_strokes is None:
                    assert len(idxs_in_shapes_drawn)==nstrokes
                
                _probs_mat_drawn = probs_mat[idxs_in_shapes_drawn, :]
                _probs_mat_not_drawn = np.mean(probs_mat[idxs_in_shapes_not_drawn, :], axis=0)[None, :]
            # else:
            #     print(self.VarDecode)
            #     assert False

            if keep_first_n_strokes:
                # Then take first n strokes
                # print(_probs_mat_drawn.shape)
                # assert False
                _probs_mat_drawn = _probs_mat_drawn[:keep_first_n_strokes, :]

            if _probs_mat_not_drawn is not None:
                probs_mat_reordered = np.concatenate([_probs_mat_drawn, _probs_mat_not_drawn], axis=0) # (ndrawn+1, ntimes)
            else:
                probs_mat_reordered = _probs_mat_drawn # (ndrawn, ntimes)
            list_probs_mat.append(probs_mat_reordered)
            
        probs_mat_all_strokeindex = np.stack(list_probs_mat) # (trials, stroke_index, times)
        probs_mat_all_strokeindex = np.transpose(probs_mat_all_strokeindex, (1, 0, 2))
        
        if probs_mat_all_strokeindex.shape[0]>nstrokes:
            labels_this = list(range(nstrokes)) + ["not_drawn"]
        else:
            labels_this = list(range(nstrokes))

        # probs_mat_all_strokeindex = np.transpose(probs_mat_all_strokeindex, (0, 2, 1))

        return probs_mat_all_strokeindex, labels_this, MAP_INDEX_TO_COL
    
    def _timeseries_plot_by_shape_drawn_order(self, probs_mat_all_strokeindex, times, labels, MAP_INDEX_TO_COL,
                                              ax, ylims=None):
        """
        [Legacy name for this function, it was not accurate, better to be _timeseries_plot_flex()]
        """
        return self._timeseries_plot_flex(self, probs_mat_all_strokeindex, times, labels, MAP_INDEX_TO_COL,
                                              ax, ylims)
    
    def _timeseries_plot_flex(self, probs_mat_all_strokeindex, times, labels, MAP_INDEX_TO_COL,
                                              ax, ylims=None, plot_legend=True):
        """
        [Low-level] Helper to plot one curve for each label (usually, the decoder, but could be anything, as long
        as aligned with dimensions of probs_mat).

        Use for plotting, how well each decoder does, across time.
        
        PARAMS:
        - probs_mat_all_strokeindex, array of probs, (nlables, ntrials, ntimes)
        - times, the within-trial time
        - labels, list of str (usualyl decoder names)
        - MAP_INDEX_TO_COL, dict, to map from label to 4-d color array. If None, then decides within.
        """
        from pythonlib.tools.plottools import color_make_map_discrete_labels

        if MAP_INDEX_TO_COL is None:
            MAP_INDEX_TO_COL, _, _ = color_make_map_discrete_labels(labels)

        print("PLOTTING... (nstrokes, ntrials, ntimes) ", probs_mat_all_strokeindex.shape)
        from neuralmonkey.neuralplots.population import plot_smoothed_fr
        for i in range(probs_mat_all_strokeindex.shape[0]):
            lab = labels[i]
            probs_mat = probs_mat_all_strokeindex[i, :, :]
            plot_smoothed_fr(probs_mat, times, ax=ax, color=MAP_INDEX_TO_COL[lab])

        probs_mat_strokeindex = np.mean(probs_mat_all_strokeindex, axis=1).T
        self._plot_single_trial(probs_mat_strokeindex, times, labels=labels, map_lab_to_col=MAP_INDEX_TO_COL, ax=ax, 
                            plot_legend=plot_legend, alpha=1)
        
        ax.axhline(0, color="k", alpha=0.5)

        if ylims is not None:
            ax.set_ylim(ylims)
        
    def timeseries_plot_wrapper(self, PAprobs, list_title_filtdict, list_n_strokes=None,
                                SIZE=4, ylims=None, filter_n_strokes_using="both"):
        """
        Make a single figure with timecoures of deocde, where each subplot are all trials with one specific n strokes drawn,
        and further optionally filtered by conjunctiontimeseries_plot_wrapper of that an a filtdict.

        NOTE: n strokes = x means trials where n strokes drawn and task are both x

        PARAMS:
        - PAprobs, output of self.timeseries_score_wrapper()
        - list_title_filtdict, list of tuples (str, filtdict), where each tuple will be a subplot, and filtereing data in
        speciifc way defined by filtdict, and str is title of ax. Leave filtdict as {} or None to ignore (thereby plotting all
        trials of this n-strokes). Note: input {}, then just a single plot averaging over all data
        - n_strokes_using, whether to use "beh", "tasks", or "both" to classify each trail's nstrokes.
        """
  
        if list_n_strokes is None:
            list_n_strokes = sorted(PAprobs.Xlabels["trials"]["FEAT_num_strokes_beh"].unique().tolist())
        if list_title_filtdict is None or len(list_title_filtdict)==0:
            list_title_filtdict = [("all", {})]

        ncols = len(list_title_filtdict)
        nrows = len(list_n_strokes)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True, squeeze=False)

        ### FILTER
        ct = 0
        for nstrokes in list_n_strokes:
            for title, filtdict in list_title_filtdict:
                ax = axes.flatten()[ct]
                # ax.set_title(title)
                ax.set_title(f"nstrokes={nstrokes}--{title}")

                ct+=1

                if filter_n_strokes_using=="both":
                    filtdict_this = {"FEAT_num_strokes_task":[nstrokes], "FEAT_num_strokes_beh":[nstrokes]}
                elif filter_n_strokes_using=="beh":
                    filtdict_this = {"FEAT_num_strokes_beh":[nstrokes]}
                # elif filter_n_strokes_using=="task": # THis doesnt make sense -- will have error, as looks for beh shapes drawn.
                #     filtdict_this = {"FEAT_num_strokes_task":[nstrokes]}
                elif filter_n_strokes_using == "beh_at_least_n":
                    # take trials with AT LEAST this many strokes. 
                    # Plot will include thte first n strokes, so some trials might have more than N, but just
                    # take first n.
                    filtdict_this = {"FEAT_num_strokes_beh":list(range(nstrokes, nstrokes+20))}
                else:
                    assert False
                    
                if filtdict is not None:
                    for k, v in filtdict.items():
                        filtdict_this[k] = v

                pathis = PAprobs.slice_by_labels_filtdict(filtdict_this)
                if len(pathis.Trials)==0:
                    print("SKIPPING!! this filtdict led to all data lost:")
                    print(filtdict_this)
                    continue

                self.timeseries_plot_by_shape_drawn_order(pathis, ax, ylims=ylims, keep_first_n_strokes=nstrokes)
        return fig

    def timeseries_plot_by_shape_drawn_order(self, PAprobs, ax, ylims=None, keep_first_n_strokes=None):
        """
        Helper to plot one curve for each storke index (0, 1, 2,...) -- i.e,. for each trial,
        what was the decode for sahpe that was done first (this is "0") and second ("1") and so on.
        For each trial this may be different shape, but only conisder its stroke index. 

        Used for shape sequence plots, showing the timecourse of decode for the different strokes that will 
        be drawn on that trial.

        PARAMS:
        - PAprobs, returned from self.timeseries_score_wrapper()
        --- Must have thse columns: shapes_drawn, locs_drawn
        --- All trials must have same n strokes drawn, or else cannot concat probs mat and will fail.
        - MAP_INDEX_TO_COL, dict, from index 0, 1, 2... --> 4-d color array
        """
        from pythonlib.tools.plottools import color_make_map_discrete_labels

        probs_mat_all_strokeindex, labels, MAP_INDEX_TO_COL = self.timeseries_extract_by_shape_drawn_order(PAprobs,
                                                                                                           keep_first_n_strokes=keep_first_n_strokes) 
        times = PAprobs.Times
        self._timeseries_plot_by_shape_drawn_order(probs_mat_all_strokeindex, times, labels, MAP_INDEX_TO_COL,
                                              ax, ylims=ylims)
            

    def scalar_score_twinds_trials(self, list_twind, PA=None, tbin_dur=None, tbin_slide=None, 
                                   PLOT=True, return_fig=False, height=12, zmax=1, score_ver="mean",
                                   min_expected_score = 0.):
        """
        Get mean decode within each trial and each twind in list_twind, and return as
        array (ntrials, nclasses, ntwinds)

        PARAMS:
            list_twind = [
                [-0.7, -0.1],
                [0.1, 0.7],
            ]
        """

        # if PA is None:
        #     PA = self.PAtrain

        # # Smooth the data
        # PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)

        ntrials = len(PA.Trials)
        assert ntrials>0
        nclass = len(self.LabelsUnique)
        ntwind = len(list_twind)
        scores = np.zeros((ntrials, nclass, ntwind)) + min_expected_score -1 # -1, so can sanity check that all filled

        for ind_twind, twind in enumerate(list_twind):
            PAthis = self.prepare_pa_dataset(PA, twind, tbin_dur, tbin_slide)            
            # PAthis = PA.slice_by_dim_values_wrapper("times", twind)

            for ind_trial in range(ntrials):
                probs_mat, _, _ = self.timeseries_score_single(ind_trial, PAthis)

                if score_ver=="mean":
                    probs_vec = np.mean(probs_mat, axis=0)
                elif score_ver=="median":
                    probs_vec = np.median(probs_mat, axis=0)
                elif score_ver=="max":
                    probs_vec = np.max(probs_mat, axis=0)
                else:
                    assert False

                assert np.all(scores[ind_trial, :, ind_twind]<0), "already filled..."
                scores[ind_trial, :, ind_twind] = probs_vec

        assert np.all(scores>=min_expected_score), "did not fill up scores..."     

        if PLOT:
            # Plot results

            from pythonlib.tools.snstools import heatmap_mat

            ind_twind = 0

            ncols = 3
            nrows = int(np.ceil(len(list_twind)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*height))

            for ax, ind_twind in zip(axes.flatten(), range(len(list_twind))):
                heatmap_mat(scores[:, :, ind_twind], ax, False, zlims=(0,zmax)) 
                ax.set_xlabel("class")
                ax.set_ylabel("trial")
                ax.set_title(f"twind: {list_twind[ind_twind]}")
        else:
            fig, axes = None, None

        if return_fig:
            return scores, fig, axes
        else:
            return scores

    # OLD VERSION - above is better, it uses the same code as timecourse extraction. This is helpful for
    # DecoderEnsemble, as it has to modify just a single base function.
    # def scalar_score_twinds_trials(self, list_twind, PA=None, tbin_dur=None, tbin_slide=None, 
    #                                PLOT=True, return_fig=False, height=12, zmax=1, score_ver="mean"):
    #     """
    #     Get mean decode within each trial and each twind in list_twind, and return as
    #     array (ntrials, nclasses, ntwinds)

    #     PARAMS:
    #         list_twind = [
    #             [-0.7, -0.1],
    #             [0.1, 0.7],
    #         ]
    #     """
    #     if PA is None:
    #         PA = self.PAtrain

    #     if tbin_dur is None:
    #         tbin_dur = TBIN_DUR
    #     if tbin_slide is None:
    #         tbin_slide=TBIN_SLIDE
    #     pathis = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
    #     clf = self.Classifier

    #     ntrials = len(pathis.Trials)
    #     assert ntrials>0
    #     nclass = clf.n_classes_
    #     ntwind = len(list_twind)
    #     scores = np.zeros((ntrials, nclass, ntwind))-1 # -1, so can sanity check that all filled
    #     for ind_trial in range(len(pathis.Trials)):

    #         x = pathis.X[:, ind_trial, :].T # (ntimes, nchans)
    #         probs_mat = clf.predict_proba(x) # (ntimes, nlabels)
    #         times = pathis.Times

    #         for ind_twind, twind in enumerate(list_twind):
    #             inds = (times>=twind[0]) & (times<=twind[1])
    #             if score_ver=="mean":
    #                 probs_vec = np.mean(probs_mat[inds, :], axis=0)
    #             elif score_ver=="median":
    #                 probs_vec = np.median(probs_mat[inds, :], axis=0)
    #             elif score_ver=="max":
    #                 probs_vec = np.max(probs_mat[inds, :], axis=0)
    #             else:
    #                 assert False

    #             assert np.all(scores[ind_trial, :, ind_twind]<0), "already filled..."
    #             scores[ind_trial, :, ind_twind] = probs_vec
    #             # res.append(probs_vec)
    #     assert np.all(scores>0), "did not fill up scores..."     

    #     if PLOT:
    #         # Plot results

    #         from pythonlib.tools.snstools import heatmap_mat

    #         ind_twind = 0

    #         ncols = 3
    #         nrows = int(np.ceil(len(list_twind)/ncols))
    #         fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*height))

    #         for ax, ind_twind in zip(axes.flatten(), range(len(list_twind))):
    #             heatmap_mat(scores[:, :, ind_twind], ax, False, zlims=(0,zmax)) 
    #             ax.set_xlabel("class")
    #             ax.set_ylabel("trial")
    #             ax.set_title(f"twind: {list_twind[ind_twind]}")
    #     else:
    #         fig, axes = None, None

    #     if return_fig:
    #         return scores, fig, axes
    #     else:
    #         return scores
    
    def labels_sort_according_to_decoder_indices(self, labels):
        """
        Return labels sorted to match the labels used in decoder.
        
        Any items in labels that are not in self.LabelsUnique, will append to end in alpha order.
        
        Any items in self.LabelsUnique not in labels, will not include
        """

        labels_sorted = []
        for lab in self.LabelsUnique:
            if lab in labels:
                labels_sorted.append(lab)
        
        for lab in sorted(labels):
            if lab not in labels_sorted:
                labels_sorted.append(lab)
        
        return labels_sorted

    def scalar_score_twinds_trialgroupings(self, vars_trial, list_twind, PA=None, tbin_dur=None, 
                                           tbin_slide=None, PLOT=True, 
                                           vars_trial_levels_sorted=None, zlims=None):
        """
        Get mean decode within each trial grouping (which may require averaging over multipel trials)
        and each twind in list_twind, and return as
        array (ntrials, nclasses, ntwinds)

        PARAMS:
        - vars_trial = ["seqc_0_shape"]
        - list_twind = [
            [-0.7, -0.1],
            [0.1, 0.7],
        ]
        - vars_trial_levels_sorted, either list of items (if vars_trial is str) or list of tuples (if vars_trial is list-like)
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good

        # PA = self.prepare_pa_dataset(PA, twind, tbin_dur, tbin_slide)            

        # if tbin_dur is None:
        #     tbin_dur=TBIN_DUR
        # if tbin_slide is None:
        #     tbin_slide=TBIN_SLIDE            

        # if PA is None:
        #     PA = self.PAtrain

        # First, get score for each trial
        scores = self.scalar_score_twinds_trials(list_twind, PA, tbin_dur, tbin_slide, PLOT)

        # Second, group trials
        dflab = PA.Xlabels["trials"]
        grpdict = grouping_append_and_return_inner_items_good(dflab, vars_trial, sort_keys=True)


        # Sort grpdict to match input labels, if exist
        if vars_trial_levels_sorted is not None:
            grpdict = {(lev,):grpdict[(lev,)] for lev in vars_trial_levels_sorted if (lev,) in grpdict}
        
        n_trial_classes = len(grpdict)
        n_decode_classes = scores.shape[1]
        n_twind = scores.shape[2]

        # Iterate over each trialgroup and twind
        resthis = np.zeros((n_trial_classes, n_decode_classes, n_twind)) - 1
        for ind_twind in range(n_twind):
            for ind_trialclass, (grp, inds) in enumerate(grpdict.items()):
                print(ind_twind, grp)

                prob_vec = np.mean(scores[inds, :, ind_twind], axis=0)

                resthis[ind_trialclass, :, ind_twind] = prob_vec

        assert np.all(resthis>=0)

        if PLOT:
            # Plot results
            from pythonlib.tools.snstools import heatmap_mat

            if zlims is None:
                zlims = (0, 1)
            ncols = 3
            nrows = int(np.ceil(len(list_twind)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))

            for ax, ind_twind in zip(axes.flatten(), range(len(list_twind))):
                heatmap_mat(resthis[:, :, ind_twind], ax, False, zlims=zlims);
                ax.set_xlabel("decoded class label")
                ax.set_ylabel("trial group label")
                ax.set_title(f"twind: {list_twind[ind_twind]}")
            
                ax.set_xticks(np.array((range(len(self.LabelsUnique)))) + 0.5, labels=self.LabelsUnique)
                ax.set_yticks(np.array(range(len(grpdict.keys())))+0.5, labels=list(grpdict.keys()))
                # ax.set_xticks(dict_plot_vals["xticks"], labels=dict_plot_vals["xtick_labels"])

            if len(list_twind)==2:
                ax = axes.flatten()[2]
                heatmap_mat(resthis[:, :, 1] - resthis[:, :, 0], ax, False, diverge=True);
                ax.set_xlabel("decoded class label")
                ax.set_ylabel("trial group label")
                ax.set_title(f"twind (difference)")
            
                ax.set_xticks(np.array((range(len(self.LabelsUnique)))) + 0.5, labels=self.LabelsUnique)
                ax.set_yticks(np.array(range(len(grpdict.keys())))+0.5, labels=list(grpdict.keys()))
                # ax.set_xticks(dict_plot_vals["xticks"], labels=dict_plot_vals["xtick_labels"])

        return resthis

    def scalar_score_extract_df(self, PA, twind, tbin_dur=None, score_ver="mean", cols_append=None, 
                                labels_decoder_good=None, prune_labels_exist_in_train_and_test=False,
                                var_decode=None):
        """
        Return a dataframe summarizing all scores (each taking a single scalar mean in this
        time window, trial, decoder label)

        PARAMS:
        - labels_decoder_good, list of decoder labels to flag as decoder_class_good=True. if None,
        then all classes are good. Does not prune data.
        - prune_labels_exist_in_train_and_test, bool, if True, then prunes dfscores...
        """

        assert PA.X.shape[1]>0

        if var_decode is None:
            var_decode = self.VarDecode

        import pandas as pd
        if cols_append is None:
            cols_append = []

        if labels_decoder_good is None:
            # assume all labels are good
            labels_decoder_good = self.LabelsUnique

        # Trials, during samp
        scores = self.scalar_score_twinds_trials([twind], PA=PA, tbin_dur=tbin_dur, score_ver=score_ver,
                                                 PLOT=False)
        ntrials, ndecoderlab = scores.shape[:2]
        
        dflab = PA.Xlabels["trials"]
        pa_classes_unique = dflab[var_decode].unique().tolist()
        assert len(dflab)==ntrials

        # Collect into dataframe
        res = []
        for i in range(ntrials):
            for j in range(ndecoderlab):
                
                decoder_class = self.MapIdxToLabel[j] # shape 

                # # What this decoder class (shape) actually drawn?
                # shapes_drawn = dflab.iloc[i]["shapes_drawn"]
                # shape_drawn_first = dflab.iloc[i]["seqc_0_shapesemgrp"]

                # # What index (in shape sequence) is this decoder's shape?
                # # decoder_class = self.MultiLabelBinarizer.classes_[j]
                # assert decoder_class in shape_sequence
                # decoder_class_idx_in_syntax = shape_sequence.index(decoder_class)
                
                # Syntax role: Is this shape visible, outside squence (TI) or within (TI)
                # What is syntax concrete (i.e., shapes shown)
                trialcode = dflab.iloc[i]["trialcode"]
                epoch = dflab.iloc[i]["epoch"]
                pa_class = dflab.iloc[i][var_decode]

                # syntax_concrete = map_tc_to_syntaxconcrete[trialcode]
                # if epoch in ["base", "baseline"]:
                #     syntax_role = None
                # else:
                #     if syntax_concrete[decoder_class_idx_in_syntax] == 1:
                #         syntax_role = "visible"
                #     else:
                #         a = syntax_concrete[decoder_class_idx_in_syntax] == 0
                #         b = sum(syntax_concrete[:decoder_class_idx_in_syntax])>0
                #         c = sum(syntax_concrete[decoder_class_idx_in_syntax+1:])>0
                #         if a & b & c:
                #             syntax_role = "within"
                #         else:
                #             syntax_role = "outside"

                # # Sanity check
                # if decoder_class in shapes_drawn and (epoch not in ["base", "baseline"]):
                #     assert syntax_concrete[decoder_class_idx_in_syntax] == 1

                    

                res.append({
                    "score":scores[i, j][0],
                    "decoder_class":decoder_class,
                    "decoder_class_good":decoder_class in labels_decoder_good,
                    "pa_class":pa_class,
                    "same_class":decoder_class==pa_class,
                    "pa_class_is_in_decoder":pa_class in self.LabelsUnique,
                    "decoder_class_is_in_pa":decoder_class in pa_classes_unique,
                    # "decoder_class_idx_in_syntax":decoder_class_idx_in_syntax,
                    # "decoder_class_was_drawn":decoder_class in shapes_drawn,
                    # "decoder_class_was_first_drawn":decoder_class == shape_drawn_first,
                    # "row":i,
                    # "col":j,
                    "pa_idx":i,
                    "decoder_idx":j,
                    # "syntax_concrete":syntax_concrete,
                    # "syntax_role":syntax_role,
                    "trialcode":trialcode,
                    "twind":twind,
                    "epoch":epoch
                })

                # Append any column of interest
                for col in cols_append:
                    res[-1][col] = dflab.iloc[i][col]

        dfscores = pd.DataFrame(res)

        # A semantic label for the "quality" of this decoder label
        dfscores["decoder_class_semantic"] = [(row["same_class"], row["decoder_class_good"], row["decoder_class_is_in_pa"]) for i, row in dfscores.iterrows()]

        def F(x):
            return str(int(x[0])) + str(int(x[1])) + str(int(x[2]))
        dfscores["decoder_class_semantic_str"] = [F(row["decoder_class_semantic"]) for i, row in dfscores.iterrows()]

        # Optionally, prune dfscores to just good labels
        if prune_labels_exist_in_train_and_test:
            dfscores = dfscores[(dfscores["pa_class_is_in_decoder"]==True) & (dfscores["decoder_class_is_in_pa"]==True)].reset_index(drop=True)

        assert dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]]

        return dfscores
    
    def scalar_score_compute_metric(self, dfscores, var_score = "score", version="overall_mean"):
        """
        Good -- get a single score for how well this decoder this doing, is like average over classes of class-specific
        accuracies. for each decoder, get its score for "match" trials minus non-match trials. Then average this over decoders.
        NOTE: do this instad of conditioning on each trail label, in order to effectively normalize and account 
        for baseline variation in decoder scores.
        RETURNS:
        - map_twind_to_score, mapping each twind to score, scalar, which is average devaition of decoder from its avfeage.
        - dfsummary, df, holding twind and score, labeled as <yvar>
        - yvar, str.
        """

        # Sort by decoder_class and pa_class
        from pythonlib.tools.pandastools import sort_by_two_columns_separate_keys
        dfscores = sort_by_two_columns_separate_keys(dfscores, "pa_class", "decoder_class")

        # (1) First, agg, so that each pa_class x decoder has single value.
        from pythonlib.tools.pandastools import aggregGeneral, summarize_featurediff
        dfscores_agg = aggregGeneral(dfscores, ["twind", "pa_class", "decoder_class"], [var_score], ["same_class"])

        # (2) Get diff of match vs. not match
        if version=="overall_mean":
            dfsummary, _, _, _, _ = summarize_featurediff(dfscores_agg, "same_class", [False, True], [var_score], 
                                ["twind"])
            
            # (3) Return dict summary per twind
            map_twind_to_score = {row["twind"]:row[yvar] for i, row in dfsummary.iterrows()}
        elif version=="decoder_class":
            # one score for each decoder class
            dfsummary, _, _, _, _ = summarize_featurediff(dfscores_agg, "same_class", [False, True], [var_score], 
                                ["twind", "decoder_class"])
            map_twind_to_score = None
        elif version=="pa_class":
            # one score for each decoder class
            dfsummary, _, _, _, _ = summarize_featurediff(dfscores_agg, "same_class", [False, True], [var_score], 
                                ["twind", "pa_class"])
            map_twind_to_score = None            
        elif isinstance(version, (list, tuple)):
            dfsummary, _, _, _, _ = summarize_featurediff(dfscores_agg, "same_class", [False, True], [var_score], 
                                version)
            map_twind_to_score = None            
        else:
            assert False
        yvar = f"{var_score}-TrueminFalse"

        return map_twind_to_score, dfsummary, yvar

    def scalar_score_df_plot_summary(self, dfscores, savedir, var_score="score"):
        """
        Many many kidns of summary plots of scores in dfscores, generally asking how well decoder scores across 
        classes (trials) and classes (decoder).
        
        PARAMS:
        - dfscores, which are extracted using scalar_score_extract_df.
        """
        from neuralmonkey.classes.population_mult import extract_single_pa
        import seaborn as sns
        from pythonlib.tools.pandastools import plot_subplots_heatmap
        from pythonlib.tools.snstools import rotateLabel
        from pythonlib.tools.pandastools import extract_with_levels_of_var_good, stringify_values, sort_by_two_columns_separate_keys
        from pythonlib.tools.pandastools import summarize_featurediff

        print("Saving plots at ... ", savedir)
        dfscores = sort_by_two_columns_separate_keys(dfscores, "pa_class", "decoder_class")
        dfscores = stringify_values(dfscores)

        # # Plot n trials for training
        # from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        # fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain.Xlabels["trials"], var_train, "task_kind", None)
        # savefig(fig, f"{savedir}/counts-var_train={var_train}.pdf")

        list_twind = dfscores["twind"].unique().tolist()

        ###
        row_values = sorted(dfscores["pa_class"].unique())
        col_values = sorted(dfscores["decoder_class"].unique())
        if False:
            fig, axes = plot_subplots_heatmap(dfscores, "pa_class", "decoder_class", var_score, "twind", False, True,
                                row_values=row_values, col_values=col_values)
            savefig(fig, f"{savedir}/heatmap-pa_class-vs-decoder_class.pdf")

        ###
        for twind in list_twind:
            df = dfscores[dfscores["twind"] == twind].reset_index(drop=True)
            fig, axes = plot_subplots_heatmap(df, "pa_class", "decoder_class", var_score, "pa_class_is_in_decoder", False, True,
                                row_values=row_values, col_values=col_values)
            savefig(fig, f"{savedir}/heatmap-pa_class-vs-decoder_class-twind={twind}-sub=pa_class_is_in_decoder.pdf")
            
            fig, axes = plot_subplots_heatmap(df, "pa_class", "decoder_class", var_score, "decoder_class_good", False, True,
                                row_values=row_values, col_values=col_values)
            savefig(fig, f"{savedir}/heatmap-pa_class-vs-decoder_class-twind={twind}-sub=decoder_class_good.pdf")
            
            if len(df["decoder_class"].unique()) < 20 and len(df["pa_class"].unique()) < 20: # Otherwise is too slow. huge plots. 
                # if "decoder_class_semantic_str" in df.columns:
                fig = sns.catplot(data=df, x="decoder_class", y=var_score, col="pa_class", col_wrap=6, alpha=0.2, 
                            jitter=True, hue="decoder_class_semantic_str")
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-score-vs-decoder_class-twind={twind}-1.pdf")

                fig = sns.catplot(data=df, x="decoder_class", y=var_score, col="pa_class", col_wrap=6,
                        kind="bar", hue="decoder_class_semantic_str")
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-score-vs-decoder_class-twind={twind}-2.pdf")

                fig = sns.catplot(data=df, x="pa_class", y=var_score, col="decoder_class", col_wrap=6, alpha=0.2, 
                            jitter=True, hue="decoder_class_semantic_str")
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-score-vs-pa_class-twind={twind}-1.pdf")

                fig = sns.catplot(data=df, x="pa_class", y=var_score, col="decoder_class", col_wrap=6,
                        kind="bar", hue="decoder_class_semantic_str")
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-score-vs-pa_class-twind={twind}-2.pdf")

            if False:
                fig = sns.catplot(data=df, x="decoder_class", y=var_score, hue="decoder_class_good", alpha=0.2, 
                            jitter=True)
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-decoder_class-twind={twind}-1.pdf")

                fig = sns.catplot(data=df, x="decoder_class", y=var_score, hue="decoder_class_good", kind="bar")
                rotateLabel(fig)
                savefig(fig, f"{savedir}/catplot-decoder_class-twind={twind}-2.pdf")

            plt.close("all")

        ###
        if "pa_idx" in dfscores.columns: # Otherwise this is an agged dataset...
            # Single scalar to summarize decoder for each class of data
            # - (same label) - (diff label)

            dfsummary, _, _, _, _ = summarize_featurediff(dfscores, "same_class", [False, True], [var_score], 
                                ["decoder_class_good", "decoder_class_is_in_pa", "pa_class", "pa_class_is_in_decoder", "pa_idx", "trialcode", "twind"])

            # dfsummary, _, _, _, _ = summarize_featurediff(dfscores, "same_class", [False, True], [var_score], 
            #                     ["decoder_class_good", "decoder_class_is_in_pa", "decoder_class", "pa_class_is_in_decoder", "pa_idx", "trialcode", "twind"])

            yvar = f"{var_score}-TrueminFalse"
            fig = sns.catplot(data=dfsummary, x="pa_class", y=yvar, hue="decoder_class_good", alpha=0.2, jitter=True, col="twind")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/summary_same_min_diff-1.pdf")

            fig = sns.catplot(data=dfsummary, x="pa_class", y=yvar, hue="decoder_class_good", kind="bar", col="twind")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/summary_same_min_diff-2.pdf")

            # A single score summarizing, across all pa labels
            fig = sns.catplot(data=dfsummary, x="twind", y=yvar, kind="bar", col="decoder_class_good")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/overallsummary_same_min_diff.pdf")


        ########### For each decoder class, ask how it does on trials matching it, compared to trials not matching.
        # First, get a single datapt for each (paclass, decoderclass). This solves problem where imbalance across trials.
        from pythonlib.tools.pandastools import aggregGeneral
        # dfscores_agg = aggregGeneral(dfscores, ["pa_class", "decoder_class"], ["score", "score_norm"], ["decoder_class_good", 
        #                                                                         "decoder_class_is_in_pa", "same_class", "decoder_idx", "decoder_class_semantic",
        #                                                                             "vars_others_grp", "var_test", "decoder_class_was_fixated",
        #                                                                                 "decoder_class_was_first_drawn", "pa_class_is_in_decoder", "twind"])
        dfscores_agg = aggregGeneral(dfscores, ["twind", "pa_class", "decoder_class"], [var_score], ["decoder_class_is_in_pa", "same_class", "pa_class_is_in_decoder"])

        for var_condition in ["decoder_class", "pa_class"]:
            # dfsummary, _, _, _, _ = summarize_featurediff(dfscores_agg, "same_class", [False, True], [var_score], 
            #                     ["decoder_class_good", "decoder_class_is_in_pa", var_condition, "twind"])
            dfsummary, _, _, _, _ = summarize_featurediff(dfscores_agg, "same_class", [False, True], [var_score], 
                                ["decoder_class_is_in_pa", "pa_class_is_in_decoder", var_condition, "twind"])

            yvar = f"{var_score}-TrueminFalse"
            fig = sns.catplot(data=dfsummary, x=var_condition, y=yvar, hue="pa_class_is_in_decoder", alpha=0.2, jitter=True, col="twind")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/aggsummary-cond={var_condition}-same_min_diff-1.pdf")

            fig = sns.catplot(data=dfsummary, x=var_condition, y=yvar, hue="pa_class_is_in_decoder", kind="bar", col="twind")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/aggsummary-cond={var_condition}-same_min_diff-2.pdf")

            # A single score summarizing, across all pa labels
            fig = sns.catplot(data=dfsummary, x="twind", y=yvar, kind="bar", col="pa_class_is_in_decoder")
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/aggoverallsummary-cond={var_condition}-same_min_diff.pdf")

            plt.close("all")

        #### Also plot classifcation accuracy
        score, score_adjusted, dfclasses, dfaccuracy = self.scalar_score_convert_to_classification_accuracy(dfscores, plot_savedir=savedir)

        plt.close("all")

    def scalar_score_convert_to_classification_accuracy(self, dfscores, var_score="score",
                                                        plot_savedir=None):
        """
        Discretize decoder output to return classfiicaiton accuracy - ie., for each pa_class, 
        determine fraction of trails that it is correctly classified.
        RETURNS:
        - score, score_adjusted, dfclasses, dfaccuracy
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        from sklearn.metrics import balanced_accuracy_score
        
        if "vars_others_grp" in dfscores and len(dfscores["vars_others_grp"].unique())>1:
            # Then this is a concated dfscores acorss different vars otheres. You should
            # instead run each one independent.
            return None, None, None, None
        
        grpdict = grouping_append_and_return_inner_items_good(dfscores, ["twind", "pa_idx"])
        # list_pa_class = dfscores["pa_class"].unique().tolist()
        # list_pa_idx = dfscores["pa_idx"].unique().tolist()

        labels_test = [] # Correct label
        labels_predicted = [] # which decoder won?
        # labels_test = {}
        # labels_predicted = {}
        # Collect labels across each test datapt (i.e., row in dflab in pa)
        res = []
        for grp, inds in grpdict.items():

            dfthis = dfscores.iloc[inds]
            # dfthis = dfscores[dfscores["pa_idx"] == pa_idx]

            decoder_class_max = dfthis.iloc[np.argmax(dfthis[var_score])]["decoder_class"]
            labels_predicted.append(decoder_class_max)
            
            tmp = dfthis["pa_class"].unique().tolist()
            if not len(tmp)==1:
                print(tmp)
                assert False, "each datapt (pa_idx) is assumed to have its only duplication be due to time windows. "
            label_actual = tmp[0]
            labels_test.append(label_actual)

            # Collect, to get scores for diff slices of data
            res.append({
                "twind":grp[0],
                "pa_idx":grp[1],
                "label_predicted":decoder_class_max,
                "label_actual":label_actual,
            })
        
        score = balanced_accuracy_score(labels_test, labels_predicted, adjusted=False)
        score_adjusted = balanced_accuracy_score(labels_test, labels_predicted, adjusted=True)

        # One row for each (twind, datapt)
        dfclasses = pd.DataFrame(res)

        ### Also get scores for each pa label
        # score separately for each ground truth label
        grpdict = grouping_append_and_return_inner_items_good(dfclasses, ["twind", "label_actual"])

        accuracy_each = []
        for (twind, label_actual), inds in grpdict.items():
            labels_predicted = dfclasses.iloc[inds]["label_predicted"]

            n_correct = sum([lab==label_actual for lab in labels_predicted])
            n_tot = len(labels_predicted)
            accuracy = n_correct/n_tot

            accuracy_each.append({
                "twind":twind,
                "label_actual":label_actual,
                "accuracy":accuracy
            })

        dfaccuracy = pd.DataFrame(accuracy_each)
        annotate_heatmap = len(self.LabelsUnique)<16
        if plot_savedir is not None:
            # Plot summaries of accuracy
            from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            for norm_method in [None, "row_sub", "col_sub"]:
                fig = grouping_plot_n_samples_conjunction_heatmap(dfclasses, "label_actual", 
                                                                  "label_predicted", ["twind"], 
                                                                  norm_method=norm_method, annotate_heatmap=annotate_heatmap)            
                savefig(fig, f"{plot_savedir}/accuracy_heatmap-norm={norm_method}.pdf")

        return score, score_adjusted, dfclasses, dfaccuracy

class DecoderEnsemble(Decoder):
    """
    Wrapper for multiple decoders, each applied to identical dataset, but result aggregating (flexible methods) across decoders.
    
    Can call methods just as if this were a single decoder.

    """

    def __init__(self, PAtrain, var_decode, twind_train, list_classifier_ver = None):
        """
        """

        if list_classifier_ver is None:
            self.ListClassifierVer = ["logistic", "naive_bayes"]
        else:
            self.ListClassifierVer = list_classifier_ver

        
        self.PAtrain = PAtrain
        assert PAtrain.X.shape[2]==1, "must pass in (nchans, ndatapts, 1) -- i.e. already time-averaged"
        
        self.VarDecode = var_decode
        self.Params = {
            "twind_train":twind_train
        }

        # Store some params
        self.LabelsUnique = sort_mixed_type(self.PAtrain.Xlabels["trials"][self.VarDecode].unique().tolist())
        # Initialize
        self.LabelsDecoderGood = None

        self.DecodersDict = {}
        for classifier_ver in self.ListClassifierVer:
            Dc = Decoder(PAtrain, var_decode, twind_train)
            self.DecodersDict[classifier_ver] = Dc

        # Sanity checks
        # _LabelsUnqiue = None
        # for classifier_ver, Dc in self.DecodersDict.items():
        #     assert Dc.LabelsUnqiue
        assert len(set([tuple(Dc.LabelsUnique) for Dc in self.DecodersDict.values()]))==1


    def train_decoder(self, PLOT=False, do_upsample_balance=True, 
                      do_upsample_balance_fig_path_nosuff=None):
        """
        Train all decoders in ensemble.
        """
        for classifier_ver, Dc in self.DecodersDict.items():
            Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff, classifier_ver)


    def timeseries_score_single(self, indtrial, PA=None, labels_in_order_keep=None,
                                # prob_thresh=0.15):
                                prob_thresh=0.0):
        """ 
        Return time-series decode prob for this trial.
        PARAMS:
        - indtrial, index into PA.
        - twind, (t1, t2)
        - labels_in_order_keep, keeps only these lables, and reetursn in this order.
        - prob_thresh, scalar [0,1], to clamp all probabilties before this number to 0. May help 
        reduce noise. But I keep at 0 so it doesnt do antyhgin.
        RETURNS;
        - probs_mat, array of probs, (ntimes, nclasses)
        - times, timestamps.
        - labels_in_order_keep, keeps only these lables, and reetursn in this order.
        """ 

        list_probs_mat = []
        times = None
        labels = None
        MapLabelToIdx = None
        MapIdxToLabel = None
        for class_ver, Dc in self.DecodersDict.items():
            probs_mat, _times, _labels = Dc.timeseries_score_single(indtrial, PA, labels_in_order_keep)
            list_probs_mat.append(probs_mat)
            if times is None:
                times = _times
                labels = _labels
                MapLabelToIdx = Dc.MapLabelToIdx
                MapIdxToLabel = Dc.MapIdxToLabel
            else:
                from pythonlib.tools.checktools import check_objects_identical
                assert check_objects_identical(times, _times)
                assert check_objects_identical(labels, _labels)
                assert check_objects_identical(MapLabelToIdx, Dc.MapLabelToIdx)
                assert check_objects_identical(MapIdxToLabel, Dc.MapIdxToLabel)
        
        # Take mean
        probs_mat = np.stack(list_probs_mat, axis=0).mean(axis=0)

        if False:
            # threshold each of the decoders and in the ensemble, and use AND
            # NOTE: this is not great -- can lead to large discontinuities in the final output...
            # But maybe halps reduce noise..
            assert len(self.ListClassifierVer)==2, "this masking is hacky. recode it."
            mask = (list_probs_mat[0]>prob_thresh) & (list_probs_mat[1] >prob_thresh)
        else:
            # Apply threshold to the final average decode...
            # 7/8/24 -- changed to this, after realized saw many discontinuities using the above.
            mask = probs_mat>=prob_thresh
        probs_mat[~mask] = 0.
            
        # For these, just trak take the first decoder        
        self.MapLabelToIdx = MapLabelToIdx
        self.MapIdxToLabel = MapIdxToLabel

        return probs_mat, times, labels



######## PLOTS THAT COMBINE MULTIPLE DC
def plot_single_trial_combine_signals_wrapper(DFallpa, bregion, TRIALCODES_PLOT, train_dataset_name, 
                                              SAVEDIR, MS, var_train = "seqc_0_shape", 
                                              color_by="stroke_index_seqc", syntax_shape_sequence=None, 
                                              TBIN_DUR=None, TBIN_SLIDE=None
                                              ):
    """
    Make single trial plots of many different signals for each trial, inclding decode, eye tracking,
    average FR over population, PCs.

    Wraps all steps, including training decoder.

    PARAMS;
    - TRIALCODES_PLOT, list of trialcodes, makes one plot per.
    - train_dataset_name, code to help get params for training decoder.
    - MS, MultSessions object for this days data
    - color_by, string, how to color decoders, e.g, "stroke_index_seqc", "stroke_index_seqc"
    - syntax_shape_sequence, list of str, optional, depends on color_by,
    - var_train, str, decoder. e.g, "seqc_0_shape",
    - TBIN_DUR, TBIN_SLIDE, for smoothing decode. e.g., TBIN_DUR = 0.2, TBIN_SLIDE = 0.01

    """
    from neuralmonkey.analyses.decode_moment import pipeline_train_test_scalar_score, pipeline_get_dataset_params_from_codeword
    from pythonlib.tools.plottools import color_make_map_discrete_labels
    import numpy as np
    from pythonlib.tools.plottools import legend_add_manual
    from neuralmonkey.classes.population_mult import extract_single_pa

    if color_by=="shape_order_global":
        assert syntax_shape_sequence is not None

    ### PLOT PARAMS
    PLOT_DECODER = True
    events_plot = ["03_samp", "05_first_raise"]

    dims = [0,1,2,3]
    map_pc_to_col = color_make_map_discrete_labels(dims)[0]

    ### Training params
    # var_test = var_train
    # savedir = "/tmp"
    n_min_per_var = 6

    ###### Methods to map from shape to index (e.g., for consistent ordering across trials and plots)
    #####
    # test_dataset_name = "pig_samp_post"

    ####
    savedir = f"{SAVEDIR}/trial_plots-{bregion}-TRAIN_DATA={train_dataset_name}-tbin_dur={TBIN_DUR}-color_shape_by={color_by}"
    os.makedirs(savedir, exist_ok=True)
    print("SAVING PLOTS AT: ", savedir)
    # Given bregion, get decoder and test dataset
    event_train, twind_train, filterdict_train, _, which_level_train = pipeline_get_dataset_params_from_codeword(train_dataset_name)

    # event_test, _, filterdict_test, list_twind_test, which_level_test = pipeline_get_dataset_params_from_codeword(test_dataset_name)
    # dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, 
    #                                     var_train, event_train, twind_train, filterdict_train,
    #                                     var_test, event_test, list_twind_test, filterdict_test,
    #                                     savedir, include_null_data=False, decoder_method_index=None,
    #                                     prune_labels_exist_in_train_and_test=True, PLOT=PLOT_DECODER,
    #                                     which_level_train=which_level_train, which_level_test=which_level_test, 
    #                                     n_min_per_var=n_min_per_var,
    #                                     subtract_baseline=False, subtract_baseline_twind=None,
    #                                     do_upsample_balance=True,
    #                                     downsample_trials=False,
    #                                     allow_multiple_twind_test=False,
    #                                     classifier_version="logistic")
    savedir_this = f"{savedir}/decoder_training"
    os.makedirs(savedir_this, exist_ok=True)
    _, Dc = train_decoder_helper(DFallpa, bregion, var_train, event_train, 
                            twind_train, PLOT=PLOT_DECODER, include_null_data=False,
                            n_min_per_var=n_min_per_var, filterdict_train=filterdict_train,
                            which_level=which_level_train, decoder_method_index=None,
                            savedir=savedir_this, do_upsample_balance=True, do_upsample_balance_fig_path_nosuff=None,
                            downsample_trials=False)

    # Extract a single pa jsut to get list of trialcode
    _pa = extract_single_pa(DFallpa, bregion, None, "trial", "03_samp")
    if TRIALCODES_PLOT is None:
        list_trialcode = sorted(_pa.Xlabels["trials"]["trialcode"].unique())
    else:
        list_trialcode = TRIALCODES_PLOT

    ### PLOT - One figure for each trial
    for trialcode in list_trialcode:
        print("Trialcode: ", trialcode)
        # shapes_to_plot = [sh for sh in Dc.LabelsUnique if sh!="null"]

        fig, axes = plt.subplots(4,1, figsize=(16,12), sharex=True, sharey=False)

        # Task features
        sn, trial_sn, _ = MS.index_convert_trial_trialcode_flex(trialcode)
        D = sn.Datasetbeh
        task_kind = D.Dat[D.Dat["trialcode"] == trialcode]["task_kind"].values[0]

        assert "03_samp" in events_plot, "need this to determine alignemnet time, nbelow."

        ########### Trials
        which_level = "trial"
        time_trial_onset = 0.
        for _i, event in enumerate(events_plot):
            pa = extract_single_pa(DFallpa, bregion, None, which_level, event)
            pa_pca = extract_single_pa(DFallpa, bregion, None, which_level, event, pa_field="pa_pca")

            # get the trial for this trialcode
            dflab = pa.Xlabels["trials"]
            tmp = dflab[dflab["trialcode"] == trialcode].index.tolist()
            assert len(tmp)==1
            trial = tmp[0]

            shapes_drawn = dflab.iloc[trial]["shapes_drawn"]
            taskconfig_shp = dflab.iloc[trial]["taskconfig_shp"]
            event_time = dflab.iloc[trial]["event_time"]
            twind = dflab.iloc[trial]["twind"]

            if event=="03_samp":
                time_trial_onset = event_time+twind[0]

            # Get for this trial
            if color_by=="stroke_index_seqc":
                max_stroke_num = max(dflab["FEAT_num_strokes_beh"])
                # for i in range(10):
                #     if all(dflab[f"seqc_{i}_shape"] =="IGN"):
                #         break
                # max_stroke_num = i-1
                shapes_to_plot = Dc.LabelsUnique
                params = [shapes_drawn, max_stroke_num, shapes_to_plot]
            elif color_by=="shape_order_global":
                # Two ways to map shape to stroke index
                # 1. global (rule)
                params = [shapes_drawn, syntax_shape_sequence]
                shapes_to_plot = syntax_shape_sequence
                # assert isinstance(shape_sequence, list)
                # MAP_INDEX_TO_COL, _, _ = color_make_map_discrete_labels(range(len(shape_sequence)))
                # MAP_SHAPE_TO_INDEX = {sh:i for i, sh in enumerate(shape_sequence)}
            else:
                assert False

            map_shape_to_col_this = Dc._plot_single_trial_helper_color_by(color_by, params)            
            # print(color_by, params)
            # print(map_shape_to_col_this)
            # assert False

            # Plot this event
            ax = axes.flatten()[0]
            if _i>0:
                plot_legend=False
                title = None
            else:
                plot_legend = True
                title = f"shapes_drawn={shapes_drawn} | taskconfig_shp={taskconfig_shp}"
            Dc.plot_single_trial(trial, pa, title=title, shift_time_dur=event_time, ax=ax,
                                                        map_lab_to_col=map_shape_to_col_this, labels_to_plot=shapes_to_plot,
                                                        plot_legend=plot_legend, tbin_dur=TBIN_DUR, 
                                                        tbin_slide=TBIN_SLIDE) 
            legend_add_manual(ax, labels=map_shape_to_col_this.keys(), colors=map_shape_to_col_this.values())

            # 3 - Plot raw neural data (average FR across channels)
            ax = axes.flatten()[1]
            ax.set_title("Mean FR across all chans")
            pathis = pa.slice_by_dim_indices_wrapper("trials", [trial])
            pathis.plotwrapper_smoothed_fr(ax=ax, plot_indiv=False, plot_summary=True, time_shift_dur=event_time)

            # 4 - Plot PCA
            # Get the order of shapes drawn, and map to colors
            ax = axes.flatten()[2]
            ax.set_title("PCA (within events)")
            for d in dims:
                x = pa_pca.X[d, trial, :]
                t = pa_pca.Times
                t = t+event_time
                col = map_pc_to_col[d]
                ax.plot(t, x, color=col, label=f"dim {d}")
            if _i==0:
                # legend_add_manual(ax, labels=map_shape_to_col_this.keys(), colors=map_shape_to_col_this.values())
                ax.legend()

        ########### Strokes 
        which_level = "stroke"
        event="00_stroke"
        pa = extract_single_pa(DFallpa, bregion, twind, which_level, event)
        pa_pca = extract_single_pa(DFallpa, bregion, twind, which_level, event, pa_field="pa_pca")

        # get the trial for this trialcode
        dflab = pa.Xlabels["trials"]
        inds = dflab[dflab["trialcode"] == trialcode].index.tolist()

        ct=0
        time_trial_offset = -10000
        for trial in inds:
            stroke_index = dflab.iloc[trial]["stroke_index"]
            
            if False: #Instad, overlay storke index below
                assert stroke_index==ct, f"skipped a stroke? {dflab.iloc[inds]['stroke_index'].tolist()}"

            # shape = dflab.iloc[trial]["shape"]
            event_time = dflab.iloc[trial]["event_time"]

            ax = axes.flatten()[0]
            title = None
            Dc.plot_single_trial(trial, pa, title=title, shift_time_dur=event_time, ax=ax, plot_legend=False,
                            map_lab_to_col=map_shape_to_col_this, labels_to_plot=shapes_to_plot, tbin_dur=TBIN_DUR, 
                                                        tbin_slide=TBIN_SLIDE)

            # Overlay stroke index
            ax.text(event_time, ax.get_ylim()[1], f"stk#{stroke_index}")

            # 3 - Plot neural data
            ax = axes.flatten()[1]
            ax.set_title("Mean FR across all chans")
            pathis = pa.slice_by_dim_indices_wrapper("trials", [trial])
            pathis.plotwrapper_smoothed_fr(ax=ax, plot_indiv=False, plot_summary=True, time_shift_dur=event_time)

            # 4 - Plot PCA
            ax = axes.flatten()[2]
            for d in dims:
                x = pa_pca.X[d, trial, :]
                t = pa_pca.Times
                t = t+event_time
                col = map_pc_to_col[d]
                ax.plot(t, x, color=col, label=f"dim {d}")
            if event_time+twind[1]>time_trial_offset:
                time_trial_offset = event_time+twind[1]

            ct+=1

        ###### TRIAL STUFF, NOT NEURAL DATA
        sn, trial_sn, _ = MS.index_convert_trial_trialcode_flex(trialcode)
        
        ### FIXATIONS
        # 1 - overlay closest shape
        for ax in axes.flatten()[:3]:
            ylim = ax.get_ylim()
            map_shape_to_col_this["FAR_FROM_ALL_SHAPES"] = np.array([0.8, 0.8, 0.8, 1.])
            dffix, map_shape_to_y, map_shape_to_col = sn.beh_eye_fixation_task_shape_overlay_plot(trial_sn, ax, 
                                                                                            map_shape_to_col=map_shape_to_col_this, 
                                                                                            yplot=ylim[0]-0.1, 
                                                                                            plot_vlines=True, vlines_alpha=0.4)
            # As sanity check, also save the fixations.
            dffix.to_csv(f"{savedir}/{trialcode}-{task_kind}-fixations.csv")

        # 2 - plot raw traces
        ax = axes.flatten()[3]
        sn.beh_plot_eye_raw_overlay_good(trial_sn, ax)
        if True:
            for t in dffix["time_global"]:
                ax.axvline(t, color="k", alpha=0.5)

        # Window
        for ax in axes.flatten():
            ax.set_xlim([time_trial_onset, time_trial_offset])
            
        # Overlay events
        for ax in axes.flatten()[:4]:
            sn.plotmod_overlay_trial_events(ax, trial_sn, alpha=0.1, xmin=time_trial_onset, xmax=time_trial_offset)

        # Plot 0 lines
        for ax in axes.flatten()[1:3]:
            ax.axhline(0, color="k", alpha=0.2)


        ###### Plot drawing
        fig_draw, axes = plt.subplots(1, 2, figsize=(6,3), sharex=True, sharey=True)

        ax = axes.flatten()[0]
        sn.plot_taskimage(ax, trial_sn)

        ax = axes.flatten()[1]
        sn.plot_final_drawing(ax, trial_sn, strokes_only=True)


        ## SAVE
        from pythonlib.tools.plottools import savefig
        savefig(fig, f"{savedir}/{trialcode}-{task_kind}-decode.pdf")
        savefig(fig_draw, f"{savedir}/{trialcode}-{task_kind}-draw.pdf")

        plt.close("all")


######## HELPER FUNCTIONS
def train_decoder_helper_extract_train_dataset_slice(PAtrain, var_train, twind_train, 
                                                     twind_train_null=None, decoder_method_index=None):
    """
    Extract dataset that is preprocessed and ready to pass into decoding anallyses.
    PARAMS;
    - twind_train_null, (t1, t2), if not None, then will also append datapts that are using this 
    twindow, hwich are all given the "null" label (e.g., baseline, presamp)
    """
    from neuralmonkey.classes.population import concatenate_popanals_flexible

    if decoder_method_index is None:
        decoder_method_index = 2

    if twind_train_null is not None:
        # what type is it. must match
        if isinstance(PAtrain.Xlabels["trials"].iloc[0][var_train], str):
            null_label = "null"
        elif isinstance(PAtrain.Xlabels["trials"].iloc[0][var_train], int):
            null_label = -999
        assert sum(PAtrain.Xlabels["trials"][var_train]==null_label)==0, "use a diff null label"

    if decoder_method_index==1:
        ##### Method 1 -- use entire window (time mean)

        # post-samp
        pa_train_postsamp = PAtrain.slice_by_dim_values_wrapper("times", twind_train) 
        pa_train_postsamp = pa_train_postsamp.agg_wrapper("times")

        # pre-samp
        if twind_train_null is not None:
            pa_train_presamp = PAtrain.slice_by_dim_values_wrapper("times", twind_train_null) 
            pa_train_presamp = pa_train_presamp.agg_wrapper("times")
            # relabel all as presamp
            pa_train_presamp.Xlabels["trials"][var_train] = null_label

            # Concatenate pre and post samp
            pa_train_all, _= concatenate_popanals_flexible([pa_train_postsamp, pa_train_presamp])
        else:
            pa_train_all = pa_train_postsamp

        # Update the params
        pa_train_all.Times = [0]
        _twind_train = [-1, 1]

    elif decoder_method_index==2:
        ##### Method 2 -- use each time bin
        # PAtrain = PAtrain.slice_by_dim_values_wrapper("times", twind)
        # PAtrain = PAtrain.agg_by_time_windows_binned(dur, slide)
        # reshape PA to 

        dur = 0.3
        slide = 0.1
        # dur = 0.3
        # slide = 0.02    

        # Get post-samp time bins
        reshape_method = "chans_x_trialstimes"
        X, PAfinal, PAslice, pca, X_before_dimred = PAtrain.dataextract_state_space_decode_flex(twind_overall=twind_train, 
                                                                                                tbin_dur=dur, tbin_slide=slide,
                                                    reshape_method=reshape_method,
                                                    norm_subtract_single_mean_each_chan=False)
        pa_train_postsamp = PAfinal

        if np.any(np.isnan(pa_train_postsamp.X)):
            print(pa_train_postsamp.X)
            print(DFallpa, bregion, None, which_level, event_train)
            assert False

        # Get pre-samp time bins
        if twind_train_null is not None:
            if True:
                reshape_method = "chans_x_trialstimes"
                # dur = 0.9
                # slide = 0.9
                X, PAfinal, PAslice, pca, X_before_dimred = PAtrain.dataextract_state_space_decode_flex(twind_overall=twind_train_null, 
                                                                                                        tbin_dur=dur, tbin_slide=slide,
                                                            reshape_method=reshape_method,
                                                            norm_subtract_single_mean_each_chan=False)
                pa_train_presamp = PAfinal
                # relabel all as presamp
                pa_train_presamp.Xlabels["trials"][var_train] = null_label

                if np.any(np.isnan(pa_train_presamp.X)):
                    print(pa_train_presamp.X)
                    print(DFallpa, bregion, None, which_level, event_train)
                    assert False

            else:
                pa_train_presamp = PAtrain.slice_by_dim_values_wrapper("times", twind_train_null) 
                pa_train_presamp = pa_train_presamp.agg_wrapper("times")
                # relabel all as presamp
                pa_train_presamp.Xlabels["trials"][var_train] = "null"
            # Concatenate pre and post samp
            pa_train_all, _= concatenate_popanals_flexible([pa_train_presamp, pa_train_postsamp])
        else:
            pa_train_all = pa_train_postsamp
    
        # Update the params
        _twind_train = [-1, 1]

    else:
        assert False

    return pa_train_all, _twind_train, PAtrain


def train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, event_train, twind_train, 
                                               include_null_data=False, n_min_per_var=5, filterdict_train=None,
                                               which_level="trial", decoder_method_index=None, PLOT=False,
                                               downsample_trials=False,
                                               do_train_splits=False, do_train_splits_nsplits=5,
                                               split_method = "train_test"):
    """
    Wrapper to extract training data given some codeword inputs, Returns data where each datapt is (nchans, 1) vector, 
    and each has associated label baseed on var_train

    Has  many methods to do resampling and train-test splits.

    Useful calls:
    i. downsample_trials=True, do_train_splits=True, split_method = "train_only"
    -- get resampled datasets, each is balanced for the variable. Useful if you want to test
    on a different dataset (e.g., train on one loc, test on other)
    ii. downsample_trials=True, do_train_splits=True, split_method = "train_test"
    -- get myultiple train-test splits, each trial contirnbuting to test once only,
    and balancing the varaible.    

    RETURNS:
    - pa_train_all, holding training data, shape (chans, datapts, 1)
    - _twind_train, becuase twind has changed, pass this into testing
    - PAtrain, training trials, before split into vecotrs for pa_train_all

    """
    
    from neuralmonkey.classes.population_mult import extract_single_pa
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    from neuralmonkey.classes.population import concatenate_popanals_flexible
    from pythonlib.tools.pandastools import extract_resample_balance_by_var
    from pythonlib.tools.statstools import balanced_stratified_kfold
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter
    from pythonlib.tools.statstools import balanced_stratified_resample_kfold

    if include_null_data:
        twind_train_null = (-0.8, -0.1)
    else:
        twind_train_null = None

    PAtrain = extract_single_pa(DFallpa, bregion, None, which_level, event_train)
    if PLOT:
        PAtrain.plotNeurHeat(trial=100)
    
    if np.any(np.isnan(PAtrain.X)):
        print(PAtrain.X)
        print(DFallpa, bregion, None, which_level, event_train)
        assert False

    ###### FILTER trials, if required.
    # dflab = PAtrain.Xlabels["trials"]
    PAtrain = PAtrain.slice_by_labels_filtdict(filterdict_train)

    if np.any(np.isnan(PAtrain.X)):
        print(PAtrain.X)
        print(DFallpa, bregion, None, which_level, event_train)
        assert False

    # Prune to keep only cases with at least n trials per label
    _, inds_keep = extract_with_levels_of_var_good(PAtrain.Xlabels["trials"], [var_train], n_min_per_var=n_min_per_var)
    print("Keeping n trials / total: ", len(inds_keep), "/", len(PAtrain.Trials))
    PAtrain = PAtrain.slice_by_dim_indices_wrapper("trials", inds_keep)
    if len(inds_keep)==0 and do_train_splits:
        return None
    elif len(inds_keep)==0 and not do_train_splits:
        return None, None, None

    if np.any(np.isnan(PAtrain.X)):
        print(PAtrain.X)
        print(DFallpa, bregion, None, which_level, event_train)
        assert False

    ### Methods to get training datasets, conjunctions of (splits, split_method, and downsample)
    if do_train_splits==False:
        # Simplest, a single train dataset
        # Return a single training set

        if downsample_trials==True:
            # Return balanced dataset, a single random sample.
            # NOTE: shoudl do here, isntead of on output, since this keeps trials as the manipulated data level.
            # NOTE: this can be a bad method if some classes have few trials. Pulls every other class down.
            print("do_train_splits==False -- downsample_trials==True")
            dflab = PAtrain.Xlabels["trials"]
            _dflab = extract_resample_balance_by_var(dflab, var_train, "min", "replacement", assert_all_rows_unique=True)
            inds_keep = sorted(_dflab.index.tolist())
            print(f"Downsampling (balance) PAtrain from {len(dflab)} rows to {len(inds_keep)} rows.")
            PAtrain = PAtrain.slice_by_dim_indices_wrapper("trials", inds_keep)
        else:
            print("do_train_splits==False -- downsample_trials==False")

        pa_train_all, _twind_train, PAtrain = train_decoder_helper_extract_train_dataset_slice(PAtrain, var_train, twind_train, 
                                                        twind_train_null=twind_train_null, decoder_method_index=decoder_method_index)
        return pa_train_all, _twind_train, PAtrain
    else:
        # Get multiple training sets... resampled in particular way.
        dflab = PAtrain.Xlabels["trials"]
        labels = dflab[var_train].tolist()        

        # (1) # On each fold, get (traininds, testinds).
        if split_method == "train_test":
            do_train_splits_nsplits = min([do_train_splits_nsplits, min(Counter(labels).values())]) # so that each test trial is done once and only once.
            if downsample_trials:
                # Then do balancing of training trials.
                print(f"split_method={split_method}, downsample_trials={downsample_trials}, nsplits={do_train_splits_nsplits}")
                folds = balanced_stratified_kfold(np.zeros(len(labels)), labels, n_splits=do_train_splits_nsplits)
            else:
                # Else just get splits, ignoring whether is balanced.
                # print(f"[not balanced] Doing {do_train_splits_nsplits} splits")
                print(f"split_method={split_method}, downsample_trials={downsample_trials}, nsplits={do_train_splits_nsplits}")
                skf = StratifiedKFold(n_splits=do_train_splits_nsplits, shuffle=True)
                folds = list(skf.split(np.zeros(len(labels)), labels)) # folds[0], 2-tuple of arays of ints

            # Construct trainsets
            trainsets = []
            for _, (train_index, test_index) in enumerate(folds):
                #TODO: This is waste of time to collect for both 
                # Datasets for training data
                PAtrain_train = PAtrain.slice_by_dim_indices_wrapper("trials", train_index, reset_trial_indices=True)
                pa_train_all, _twind_train, _ = train_decoder_helper_extract_train_dataset_slice(PAtrain_train, var_train, twind_train, 
                                                    twind_train_null=twind_train_null, decoder_method_index=decoder_method_index)
                
                # # also get one that combines trian and test indices
                # both_indices = [int(i) for i in train_index] + [int(i) for i in test_index]
                # PAtrain_traintest = PAtrain.slice_by_dim_indices_wrapper("trials", both_indices, reset_trial_indices=True)
                # pa_traintest_all, _twind_traintest, _ = train_decoder_helper_extract_train_dataset_slice(PAtrain_traintest, var_train, twind_train, 
                #                                     twind_train_null=twind_train_null, decoder_method_index=decoder_method_index)
                
                # assert _twind_traintest == _twind_train
                
                trainsets.append({
                    "pa_train_all":pa_train_all, # Training indice, chopping time window up in to pieces.
                    "PAtrain_train":PAtrain_train,
                    # "pa_traintest_all":pa_traintest_all, # training + testing indices
                    # "PAtrain_traintest":PAtrain_traintest,
                    "_twind_train":_twind_train,
                    "train_index_PAtrain_orig":train_index,
                    "test_index_PAtrain_orig":test_index,
                    "PAtrain_orig":PAtrain
                })

        # (2) Resample, on each fold getting a single set of inds (training inds)
        elif split_method == "train_only":
            if downsample_trials:
                # This is the only reason you wuld use split_method == "train_only".
                print(f"split_method={split_method}, downsample_trials={downsample_trials}, nsplits={do_train_splits_nsplits}")
                folds = balanced_stratified_resample_kfold(None, labels, n_splits=do_train_splits_nsplits) # list of list of ints
            else:
                assert False, "no reason to do this..."

            # Construct trainsets
            trainsets = []
            for indices in folds:

                # Datasets for training data
                PAtrain_train = PAtrain.slice_by_dim_indices_wrapper("trials", indices, reset_trial_indices=True)
                pa_train_all, _twind_train, _ = train_decoder_helper_extract_train_dataset_slice(PAtrain_train, var_train, twind_train, 
                                                    twind_train_null=twind_train_null, decoder_method_index=decoder_method_index)
                
                trainsets.append({
                    "pa_train_all":pa_train_all, # Training indice, chopping time window up in to pieces.
                    "PAtrain_train":PAtrain_train,
                    "_twind_train":_twind_train,
                    "index_PAtrain_orig":indices,
                    "PAtrain_orig":PAtrain
                })
        else:
            assert False

        return trainsets

    # if decoder_method_index==1:
    #     ##### Method 1 -- use entire window (time mean)

    #     # post-samp
    #     pa_train_postsamp = PAtrain.slice_by_dim_values_wrapper("times", twind_train) 
    #     pa_train_postsamp = pa_train_postsamp.agg_wrapper("times")

    #     # pre-samp
    #     if twind_train_null is not None:
    #         pa_train_presamp = PAtrain.slice_by_dim_values_wrapper("times", twind_train_null) 
    #         pa_train_presamp = pa_train_presamp.agg_wrapper("times")
    #         # relabel all as presamp
    #         pa_train_presamp.Xlabels["trials"][var_train] = null_label

    #         # Concatenate pre and post samp
    #         pa_train_all, _= concatenate_popanals_flexible([pa_train_postsamp, pa_train_presamp])
    #     else:
    #         pa_train_all = pa_train_postsamp

    #     # Update the params
    #     pa_train_all.Times = [0]
    #     _twind_train = [-1, 1]

    # elif decoder_method_index==2:
    #     ##### Method 2 -- use each time bin
    #     # PAtrain = PAtrain.slice_by_dim_values_wrapper("times", twind)
    #     # PAtrain = PAtrain.agg_by_time_windows_binned(dur, slide)
    #     # reshape PA to 

    #     dur = 0.3
    #     slide = 0.1
    #     # dur = 0.3
    #     # slide = 0.02    

    #     # Get post-samp time bins
    #     reshape_method = "chans_x_trialstimes"
    #     X, PAfinal, PAslice, pca, X_before_dimred = PAtrain.dataextract_state_space_decode_flex(twind_overall=twind_train, 
    #                                                                                             tbin_dur=dur, tbin_slide=slide,
    #                                                 reshape_method=reshape_method,
    #                                                 norm_subtract_single_mean_each_chan=False)
    #     pa_train_postsamp = PAfinal

    #     if np.any(np.isnan(pa_train_postsamp.X)):
    #         print(pa_train_postsamp.X)
    #         print(DFallpa, bregion, None, which_level, event_train)
    #         assert False

    #     # Get pre-samp time bins
    #     if twind_train_null is not None:
    #         if True:
    #             reshape_method = "chans_x_trialstimes"
    #             # dur = 0.9
    #             # slide = 0.9
    #             X, PAfinal, PAslice, pca, X_before_dimred = PAtrain.dataextract_state_space_decode_flex(twind_overall=twind_train_null, 
    #                                                                                                     tbin_dur=dur, tbin_slide=slide,
    #                                                         reshape_method=reshape_method,
    #                                                         norm_subtract_single_mean_each_chan=False)
    #             pa_train_presamp = PAfinal
    #             # relabel all as presamp
    #             pa_train_presamp.Xlabels["trials"][var_train] = null_label

    #             if np.any(np.isnan(pa_train_presamp.X)):
    #                 print(pa_train_presamp.X)
    #                 print(DFallpa, bregion, None, which_level, event_train)
    #                 assert False

    #         else:
    #             pa_train_presamp = PAtrain.slice_by_dim_values_wrapper("times", twind_train_null) 
    #             pa_train_presamp = pa_train_presamp.agg_wrapper("times")
    #             # relabel all as presamp
    #             pa_train_presamp.Xlabels["trials"][var_train] = "null"
    #         # Concatenate pre and post samp
    #         pa_train_all, _= concatenate_popanals_flexible([pa_train_presamp, pa_train_postsamp])
    #     else:
    #         pa_train_all = pa_train_postsamp
    
    #     # Update the params
    #     _twind_train = [-1, 1]

    # else:
    #     assert False

    # return pa_train_all, _twind_train, PAtrain

def train_decoder_helper(DFallpa, bregion, var_train="seqc_0_shape", event_train=None, 
                            twind_train = None,
                            PLOT=True, include_null_data=False,
                            n_min_per_var=5, filterdict_train=None,
                            which_level="trial", decoder_method_index=None,
                            savedir=None, n_min_per_var_good = 10,
                            do_upsample_balance=False, do_upsample_balance_fig_path_nosuff=None,
                            downsample_trials=False, classifier_version="logistic",
                            return_dfscores_train = False):
    """
    Train a decoder for this bregion and train variables, and make relevant plots, and with variaous 
    preprocessing methods optional.
    PARAMS:
    - include_null_data, bool, if True, then includes "pre-samp" data as a "null" label. Tends to
    make decoding post-samp worse.
    - n_min_per_var, keeps only those levels of var_train which have at least this many trials.
    NOTE: Methods for up and downsampling.
    If downsampling, best to use downsample_trials, becuase this works at level fo trials, instead of
    after snipping into vectors, which would be incorrect.
    If upsampling, is fine to use do_upsample_balance at level of snipped vectors.
    """

    from neuralmonkey.analyses.decode_moment import Decoder
    # var = "seqc_0_shapesemgrp" # Decoded variable
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    from neuralmonkey.classes.population import concatenate_popanals_flexible
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good

    if decoder_method_index is None:
        decoder_method_index = 2

    if event_train is None:
        event_train = "03_samp"
    if twind_train is None:
        twind_train = (0.1, 0.9)

    # trainsets, PAtrain = train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, 
    #                                                                                  event_train, twind_train,  
    #                                                                                  include_null_data, n_min_per_var, filterdict_train,
    #                                                                                  which_level, decoder_method_index, PLOT,
    #                                                                                  downsample_trials=downsample_trials,
    #                                                                                  do_train_splits=True, do_train_splits_nsplits=5)
    # for t in trainsets:
    #     print(t)
    # print(PAtrain.X.shape)
    # assert False

    pa_train_all, _twind_train, PAtrain = train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, 
                                                                                     event_train, twind_train,  
                                                                                     include_null_data, n_min_per_var, filterdict_train,
                                                                                     which_level, decoder_method_index, PLOT,
                                                                                     downsample_trials=downsample_trials)

    print("Training this classifier version: ", classifier_version)
    if classifier_version == "ensemble":
        # Train
        Dc = DecoderEnsemble(pa_train_all, var_train, _twind_train)
        Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)
    else:
        # Train
        Dc = Decoder(pa_train_all, var_train, _twind_train)
        Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff, classifier_version=classifier_version)

    # A flag for "good" labels --> i.e. those in decoder that enough trials. 
    _df, _ = extract_with_levels_of_var_good(PAtrain.Xlabels["trials"], [Dc.VarDecode], n_min_per_var_good)
    labels_decoder_good = _df[Dc.VarDecode].unique().tolist()
    Dc.LabelsDecoderGood = labels_decoder_good

    if PLOT:
        # Plot n trials for training
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain.Xlabels["trials"], var_train, "task_kind", None)
        savefig(fig, f"{savedir}/counts-var_train={var_train}.pdf")

        # Plot scores on TRAIN set
        labels_decoder_good = Dc.LabelsDecoderGood
        dfscores_train = Dc.scalar_score_extract_df(PAtrain, twind_train, labels_decoder_good=labels_decoder_good, 
                                                    var_decode=var_train)
        sdir = f"{savedir}/train_set"
        os.makedirs(sdir, exist_ok=True)
        Dc.scalar_score_df_plot_summary(dfscores_train, sdir)      

    if return_dfscores_train:
        if not PLOT:
            labels_decoder_good = Dc.LabelsDecoderGood
            dfscores_train = Dc.scalar_score_extract_df(PAtrain, twind_train, labels_decoder_good=labels_decoder_good, 
                                                        var_decode=var_train)
        return PAtrain, Dc, dfscores_train
    else:
        return PAtrain, Dc

def train_decoder_helper_with_splits(DFallpa, bregion, var_train="seqc_0_shape", event_train=None, 
                            twind_train = None,
                            PLOT=True, include_null_data=False,
                            n_min_per_var=5, filterdict_train=None,
                            which_level="trial", decoder_method_index=None,
                            savedir=None, n_min_per_var_good = 10,
                            do_upsample_balance=False, do_upsample_balance_fig_path_nosuff=None,
                            downsample_trials=False, classifier_version="logistic",
                            return_dfscores_train = False,
                            do_train_splits_nsplits=5, split_method="train_test"):
    """
    IDentical to train_decoder_helper, but do multiple times, each time splitting trials for training data, 
    and return each decoder, already trained, and do plots for each.
    PARAMS:
    - do_train_splits_nsplits, int, how many stratified train-test splits...
    - combine_train_test_indices, bool (FAlse).
    --- True, then combines train and test indices to get training setes. This is useful if
    you will then take Decodersa nd apply to held-out data. i.e. you dont care about the
    train-test split.
    --- False, ...

    """

    from neuralmonkey.analyses.decode_moment import Decoder
    # var = "seqc_0_shapesemgrp" # Decoded variable
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    from neuralmonkey.classes.population import concatenate_popanals_flexible
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good

    if decoder_method_index is None:
        decoder_method_index = 2

    if event_train is None:
        event_train = "03_samp"
    if twind_train is None:
        twind_train = (0.1, 0.9)

    trainsets = train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, 
                                                                                     event_train, twind_train,  
                                                                                     include_null_data, n_min_per_var, filterdict_train,
                                                                                     which_level, decoder_method_index, PLOT,
                                                                                     downsample_trials=downsample_trials,
                                                                                     do_train_splits=True, 
                                                                                     do_train_splits_nsplits=do_train_splits_nsplits,
                                                                                     split_method=split_method)
    if trainsets is None:
        return [], []
    
    decoders = []
    # trainsets_success = []
    for i_ts, ts in enumerate(trainsets):
        # -- Train on train set indices only.
        print("Training this classifier version: ", classifier_version, "trainset(split num)=", i_ts)
        print("... info for this train-test split: ", ts)
        # Default. Here is training on subset of indices. 
        pa_train_all = ts["pa_train_all"]
        _twind_train = ts["_twind_train"]
        PAtrain_train = ts["PAtrain_train"] # Just those used for training trials, here.
        if classifier_version == "ensemble":
            # Train
            Dc = DecoderEnsemble(pa_train_all, var_train, _twind_train)
            Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)
            # success = Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)
        else:
            # Train
            Dc = Decoder(pa_train_all, var_train, _twind_train)
            Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff, classifier_version=classifier_version)
        decoders.append(Dc)

        #################### POST-STUFF
        # A flag for "good" labels --> i.e. those in decoder that enough trials. 
        _df, _ = extract_with_levels_of_var_good(PAtrain_train.Xlabels["trials"], [Dc.VarDecode], n_min_per_var_good)
        labels_decoder_good = _df[Dc.VarDecode].unique().tolist()
        Dc.LabelsDecoderGood = labels_decoder_good

        # Always make this plot of sample size
        # Plot n trials for training
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        savedir_this = f"{savedir}/trainsplit={i_ts}"
        os.makedirs(savedir_this, exist_ok=True)
        fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain_train.Xlabels["trials"], var_train, "task_kind", None)
        savefig(fig, f"{savedir_this}/counts-var_train={var_train}.pdf")
        plt.close("all")

        # Note down the exact trialcodes used. This useful for sanity check downsampling, etc.
        from pythonlib.tools.pandastools import grouping_print_n_samples
        grouping_print_n_samples(PAtrain_train.Xlabels["trials"], [var_train, "trialcode"], savepath=f"{savedir_this}/datapts-{var_train}-vs-trialcode.txt")

        if PLOT:
            # Plot scores on TRAIN set
            labels_decoder_good = Dc.LabelsDecoderGood
            dfscores_train = Dc.scalar_score_extract_df(PAtrain_train, twind_train, labels_decoder_good=labels_decoder_good, 
                                                        var_decode=var_train)
            sdir = f"{savedir_this}/train_set"
            os.makedirs(sdir, exist_ok=True)
            Dc.scalar_score_df_plot_summary(dfscores_train, sdir)      
    
    assert len(trainsets)==len(decoders)
    return trainsets, decoders

def _test_decoder_helper(Dc, PAtest, var_test, list_twind_test, subtract_baseline=False, PLOT=False, savedir=None,
                         prune_labels_exist_in_train_and_test=True, filterdict_test=None):
    """
    Get scores, using this decoder (Dc) against this dataset (PAtest), and make vaeity of plots, and return
    dataframe of scroes.
    Extract test dataest, and then score decoder Dc aginst this test dataset.
    PARAMS:
    - list_twind_test, multiple time windows to run tests over.
    RETURNS:
    - dfscores, dataframe holding score, where each row is a trial x decoder class.
    - PAtest, the testing trials used to construct dfscores.
    """

    if filterdict_test is not None and len(filterdict_test)>0:
        PAtest = PAtest.slice_by_labels_filtdict(filterdict_test)

    ### Extract df summarizing all scalar scores
    labels_decoder_good = Dc.LabelsDecoderGood

    # Collect scores across all twinds for testing
    list_df = []
    for twind in list_twind_test:
        _dfscores = Dc.scalar_score_extract_df(PAtest, twind, labels_decoder_good=labels_decoder_good, 
                                            prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test,
                                            var_decode=var_test)
        list_df.append(_dfscores)

    # Also get baseline?
    if subtract_baseline:
        _dfscores_base = Dc.scalar_score_extract_df(PAtest, subtract_baseline_twind, labels_decoder_good=labels_decoder_good, 
                                            prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test,
                                            var_decode=var_test)
        
        # DO subtraction
        for _dfscores in list_df:
            assert np.all(_dfscores.loc[:, ["pa_idx", "decoder_idx"]] == _dfscores_base.loc[:, ["pa_idx", "decoder_idx"]]), "data not aligned between data and base"
            _dfscores["score_min_base"] = _dfscores["score"] - _dfscores_base["score"]
            _dfscores["score_base"] = _dfscores_base["score"]

    # Finalize by concatting.
    dfscores = pd.concat(list_df).reset_index(drop=True)
    dflab = PAtest.Xlabels["trials"]
    assert dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]]

    ### Plots
    if PLOT:
        sdir = f"{savedir}/test-var_score=score"
        os.makedirs(sdir, exist_ok=True)
        Dc.scalar_score_df_plot_summary(dfscores, sdir)

    return dfscores, PAtest

def test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                        which_level_test, savedir, prune_labels_exist_in_train_and_test=True, PLOT=True,
                        subtract_baseline=False, subtract_baseline_twind=None,
                        allow_multiple_twind_test=False):
    """
    Test decode for a dataset, here helsp extract that dataset and runs testing using decoder in Dc.
    PARAMS:
    - list_twind_test, list of twinds, tests each of those. Useful if one twind is pre-event and one is post.
    - prune_labels_exist_in_train_and_test, bool, useful for cleaning up results.
    - subtract_baseline, bool, to subtract pre-event (Usually) from each datapt, where the time of the event is
    given by subtract_baseline_twind
    RETURNS:
    - dfscores, dataframe holding score, where each row is a trial x decoder class.
    - PAtest, the testing trials used to construct dfscores.
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.snstools import rotateLabel
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good

    if allow_multiple_twind_test:
        assert len(list_twind_test)>0
    else:
        assert len(list_twind_test)==1

    ### Get testing data
    PAtest = extract_single_pa(DFallpa, bregion, None, which_level=which_level_test, event=event_test)
    # if filterdict_test is not None:
    #     for col, vals in filterdict_test.items():
    #         print(f"filtering with {col}, starting len...", len(dflab))
    #         dflab = dflab[dflab[col].isin(vals)]
    #         print("... ending len: ", len(dflab))
    #     inds = dflab.index.tolist()
    #     PAtest = PAtest.slice_by_dim_indices_wrapper("trials", inds)
    # assert len(dflab)>0, "All data pruned!!!" 

    dfscores, PAtest = _test_decoder_helper(Dc, PAtest, var_test, list_twind_test, subtract_baseline, 
                                    PLOT, savedir, prune_labels_exist_in_train_and_test,
                                    filterdict_test=filterdict_test)
    dflab = PAtest.Xlabels["trials"]
    assert dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]]

    # ### Extract df summarizing all scalar scores
    # labels_decoder_good = Dc.LabelsDecoderGood

    # # Collect scores across all twinds for testing
    # list_df = []
    # for twind in list_twind_test:
    #     _dfscores = Dc.scalar_score_extract_df(PAtest, twind, labels_decoder_good=labels_decoder_good, 
    #                                         prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test,
    #                                         var_decode=var_test)
    #     list_df.append(_dfscores)

    # # Also get baseline?
    # if subtract_baseline:
    #     _dfscores_base = Dc.scalar_score_extract_df(PAtest, subtract_baseline_twind, labels_decoder_good=labels_decoder_good, 
    #                                         prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test,
    #                                         var_decode=var_test)
        
    #     # DO subtraction
    #     for _dfscores in list_df:
    #         assert np.all(_dfscores.loc[:, ["pa_idx", "decoder_idx"]] == _dfscores_base.loc[:, ["pa_idx", "decoder_idx"]]), "data not aligned between data and base"
    #         _dfscores["score_min_base"] = _dfscores["score"] - _dfscores_base["score"]
    #         _dfscores["score_base"] = _dfscores_base["score"]

    # # Finalize by concatting.
    # dfscores = pd.concat(list_df).reset_index(drop=True)

    # ### Plots
    # if PLOT:
    #     sdir = f"{savedir}/test-var_score=score"
    #     os.makedirs(sdir, exist_ok=True)
    #     Dc.scalar_score_df_plot_summary(dfscores, sdir)
    return dfscores, PAtest

def pipeline_train_test_scalar_score_with_splits(DFallpa, bregion, 
                                     var_train, event_train, twind_train, filterdict_train,
                                     var_test, event_test, list_twind_test, filterdict_test,
                                     savedir, include_null_data=False, decoder_method_index=None,
                                     prune_labels_exist_in_train_and_test=True, 
                                     PLOT_TRAIN=True, PLOT_TEST_SPLIT=True, PLOT_TEST_CONCATTED=True,
                                     which_level_train="trial", which_level_test="trial", n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=True,
                                     downsample_trials=False,
                                     allow_multiple_twind_test=False,
                                     classifier_version="logistic", # 7/11/24 - decided this, since it seems to do better for chars (same image, dif beh)
                                    #  classifier_version="ensemble",
                                     do_train_splits_nsplits=5, score_user_test_data=True,
                                     do_agg_of_user_test_data=True, split_method="train_test"):
    """
    Helper to extract dataframe holding decode score for each row of this bregion's test dataset (PA)
    
    WRitten when doing the Shape seuqence TI stuff.

    This differs from pipeline_train_test_scalar_score, in that here does train-test splits on training dataset, and returns 
    all of the split models

    PARAMS:
    - subtract_baseline, bool, if True, then extracts data using twind in subtract_baseline_twind, and subtracts 
    the resulting scores from each twind in list_twind_test (ie each datapt) -- new column: score_min_base
    RETURNS:
    - (Mainly goal is saves plots and data)
    - dfscores, Dc, PAtrain, PAtest
    - allow_multiple_twind_test, bool, as a switch so that doesnt accidentally forget that is multkpel twind.
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.snstools import rotateLabel

    twind_train = tuple(twind_train)
    list_twind_test = [tuple(twind) for twind in list_twind_test]

    if not allow_multiple_twind_test:
        assert len(list_twind_test)==1, "you must turn on flag allow_multiple_twind_test"

    ### Train decoder
    if n_min_per_var is None:
        n_min_per_var=3
    n_min_per_var_good = 10

    do_upsample_balance_fig_path_nosuff = f"{savedir}/upsample_pcs"
    trainsets, decoders = train_decoder_helper_with_splits(DFallpa, bregion, var_train, event_train, twind_train,
                                    PLOT_TRAIN, include_null_data, n_min_per_var, filterdict_train,
                                    which_level_train, decoder_method_index, savedir, n_min_per_var_good,
                                    do_upsample_balance, do_upsample_balance_fig_path_nosuff,
                                    downsample_trials=downsample_trials, classifier_version=classifier_version,
                                    return_dfscores_train=True, do_train_splits_nsplits=do_train_splits_nsplits,
                                    split_method=split_method)
    if len(trainsets)==0:
        # Then no data after filtering...'
        print("!Skipping, len(trainsets)==0")
        return None, None, None, None, None, None
        
    ################ EVALUATION
    # For each training set, evaluate (i) the test set and (ii) the held-out part of the training set.
    list_dfscores_testsplit = []
    list_dfscores_usertest = []

    for i_ts, (Dc, ts) in enumerate(zip(decoders, trainsets)):    
        savedir_this = f"{savedir}/trainsplit={i_ts}-vs-testsplit"
        
        if split_method =="train_test":
            ### (1) Test on held-out data
            test_index = [int(i) for i in ts["test_index_PAtrain_orig"]] # array of ints
            PAtrain_orig = ts["PAtrain_orig"]

            # - slice out test dataset
            patest = PAtrain_orig.slice_by_dim_indices_wrapper("trials", test_index, reset_trial_indices=True)
            dfscores, patest = _test_decoder_helper(Dc, patest, var_test, list_twind_test, subtract_baseline, PLOT_TEST_SPLIT, savedir_this, 
                                            prune_labels_exist_in_train_and_test)
            dfscores["train_split_idx"] = i_ts

            # Modify indices that are specific to this decoder and test set
            dfscores["pa_idx"] = [(i_ts, pa_idx) for pa_idx in dfscores["pa_idx"]]
            dfscores["decoder_idx"] = [(i_ts, decoder_idx) for decoder_idx in dfscores["decoder_idx"]]

            list_dfscores_testsplit.append(dfscores)

        ### (2) Test on user-inputted test set.
        if score_user_test_data:
            savedir_this = f"{savedir}/trainsplit={i_ts}-vs-user_inputed_test"
            dfscores, PAtest = test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                                which_level_test, savedir_this, prune_labels_exist_in_train_and_test, PLOT_TEST_SPLIT,
                                subtract_baseline, subtract_baseline_twind, allow_multiple_twind_test=allow_multiple_twind_test)
            dfscores["train_split_idx"] = i_ts
        else:
            dfscores = pd.DataFrame([])
            PAtest = None
        list_dfscores_usertest.append(dfscores)

    # Concat the results
    dfscores_usertest = pd.concat(list_dfscores_usertest).reset_index(drop=True)

    ################ COMBINING RESULTS 
    # Some colums which might differ across train-test splits, give them common fake value here. THse are not important
    if split_method =="train_test":
        dfscores_testsplit = pd.concat(list_dfscores_testsplit).reset_index(drop=True)
        dfscores_testsplit["decoder_class_good"] = True
        dfscores_testsplit["decoder_class_semantic_str"] = "IGN" # igbnore this.
    else:
        dfscores_testsplit = None
    # del dfscores["decoder_class_semantic_str"]
    
    # dfscores_testsplit["decoder_class_semantic_str"] = True
    # dfscores_testsplit["decoder_class_good"] = True
    
    # Combine test results from train-test and from uset test.
    dfscores_both = None
    if do_agg_of_user_test_data:
        # This averages over all train-test splits, so that each test datapt has a single row. Can do this for
        # user data (but not for split train data) becuyase user data has identical test datapts on each split.
        from pythonlib.tools.pandastools import aggregGeneral
        dfscores_usertest = aggregGeneral(dfscores_usertest, 
                                          ["decoder_idx", "pa_idx", "twind"], 
                                          ["score"], 
                                          ["decoder_class", "pa_class", "trialcode", "epoch", "same_class"] # code does sanity check that these are all unqiue values.
                                          )

        # Optioanlly, concat the two dfscores. This is useful if train-test-split version is testing on one subset of data
        # and the usertest is testing on other subset. (e.g., I did this for morphset, different susbet of indices)
        # assert do_agg_of_user_test_data==True, "otherwise rows will be duplicated for teh user-test"

        # 1. santiyc check that they have same columns except train_split_idx
        if False: # no need, since I am explicitly asking for the columns I need in aggregGeneral.
            if [col for col in dfscores_testsplit.columns if col not in dfscores_usertest] != ['train_split_idx']:
                print([col for col in dfscores_testsplit.columns if col not in dfscores_usertest])
                print([col for col in dfscores_usertest.columns if col not in dfscores_testsplit])
                assert False, "did you mistakenly throw out columns for dfscores_usertest when doing aggGeneral(dfscores_usertest)?"

        if split_method =="train_test":
            # Combine split and user test data into a single dataframe
            dfscores_both = pd.concat([dfscores_testsplit, dfscores_usertest]).reset_index(drop=True)
            dfscores_both = dfscores_both.drop("train_split_idx", axis=1)

    # Make a final plot of concated test data
    if PLOT_TEST_CONCATTED:
        sdir = f"{savedir}/test-concatted-testsplit"
        os.makedirs(sdir, exist_ok=True)
        Dc.scalar_score_df_plot_summary(dfscores_testsplit, sdir)

    # Save params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({
        "bregion":bregion, 
        "var_train":Dc.VarDecode, 
        "event_train":event_train, 
        "twind_train":twind_train, 
        "filterdict_train":filterdict_train,
        "var_test": var_test, 
        "event_test": event_test, 
        "list_twind_test": list_twind_test, 
        "filterdict_test":filterdict_test,
        "which_level_train":which_level_train,
        "which_level_test":which_level_test,
        "include_null_data":include_null_data,
        "decoder_method_index":decoder_method_index,
        "prune_labels_exist_in_train_and_test":prune_labels_exist_in_train_and_test,
        "n_min_per_var":n_min_per_var,
        "subtract_baseline":subtract_baseline,
        "subtract_baseline_twind":subtract_baseline_twind,
        "do_upsample_balance":do_upsample_balance,
        "downsample_trials":downsample_trials,
        "allow_multiple_twind_test":allow_multiple_twind_test,
        "classifier_version":classifier_version,
        "do_train_splits_nsplits":do_train_splits_nsplits}, 
        path=f"{savedir}/params.txt")

    return dfscores_testsplit, dfscores_usertest, dfscores_both, decoders, trainsets, PAtest


def pipeline_get_dataset_params_from_codeword(dataset_name):
    """
    Helper to take codeword (dataset_name) and output params that can 
    be passed into pipeline_train_test_scalar_score and related
    functions.

    Each dataset_name is some slice of trials and times (e.g., post-sampe for single prims),
    that is meaningful as a train or test dataset that is commonly used.

    """

    which_level = "trial"

    SP_HOLD_DUR = 1.2
    # PIG_HOLD_DUR = 1.2
    PIG_HOLD_DUR = 1.8

    if dataset_name == "sp_samp":
        event = "03_samp"
        twind = (0.1, SP_HOLD_DUR)
        list_twind = [(-0.9, -0.1), twind]
        filterdict = {"FEAT_num_strokes_task":[1]}

    elif dataset_name == "pig_samp":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR)
        list_twind = [(-0.9, -0.1), twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8], 
                    "task_kind":["prims_on_grid"]
                    }

    elif dataset_name == "char_samp_post":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR)
        list_twind = [twind]
        filterdict = {
            "FEAT_num_strokes_task":[2,3,4,5,6,7,8], 
            "task_kind":["character"]
            }

    elif dataset_name == "pig_samp_post":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR)
        list_twind = [twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8], 
                    "task_kind":["prims_on_grid"]
                    }

    elif dataset_name == "pig_samp_post_early":
        event = "03_samp"
        twind = (0.1, PIG_HOLD_DUR/2)
        list_twind = [twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8], 
                    "task_kind":["prims_on_grid"]
                    }

    elif dataset_name == "pig_samp_post_late":
        event = "03_samp"
        twind = (PIG_HOLD_DUR/2, PIG_HOLD_DUR)
        list_twind = [twind]
        filterdict = {"FEAT_num_strokes_task":[2,3,4,5,6,7,8], 
                    "task_kind":["prims_on_grid"]
                    }

    elif dataset_name == "sp_prestroke":
        event = "06_on_strokeidx_0"
        twind = (-0.6, 0)
        list_twind = [twind, (0, 0.6)]
        filterdict = {"FEAT_num_strokes_task":[1]}

    elif dataset_name == "pig_prestroke":
        event = "06_on_strokeidx_0"
        twind = (-0.6, 0)
        list_twind = [twind, (0, 0.6)]
        filterdict = {
            "FEAT_num_strokes_task":[2,3,4,5,6,7,8],
            "task_kind":["prims_on_grid"]}

    # elif dataset_name == "test":
    #     event = "06_on_strokeidx_0"
    #     twind = (-0.6, -0.1)
    #     list_twind = [twind]
    #     filterdict = {
    #         "task_kind":["prims_single", "prims_on_grid"]}

    elif dataset_name == "sp_pig_samp_post":
        event = "03_samp"
        twind = (0.1, SP_HOLD_DUR)
        list_twind = [twind]
        filterdict = {
            "task_kind":["prims_single", "prims_on_grid"]}

    elif dataset_name == "sp_pig_pre_stroke_all":
        event = "00_stroke"
        # twind = (-0.6, -0.1)
        twind = (-0.5, -0.15)
        list_twind = [twind]
        filterdict = {"task_kind":["prims_single", "prims_on_grid"]}
        which_level = "stroke"

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


def pipeline_train_test_scalar_score_split_gridloc(list_loc, savedir_base,
                                                   DFallpa, bregion, 
                                     var_train, event_train, twind_train, filterdict_train,
                                     var_test, event_test, list_twind_test, filterdict_test,
                                     include_null_data=False, prune_labels_exist_in_train_and_test=True, PLOT=False, PLOT_TEST=True,
                                     which_level_train="trial", which_level_test="trial", n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=True, downsample_trials=False,
                                     allow_multiple_twind_test=False,
                                     auto_prune_locations=False, do_train_splits_nsplits=10):
    """
    Good wrapper for train and testing, but here splitting so that training and testing on all separate pairs of 
    gridloc (I.e., testing generalization across gridloc).

    Does not do train-test split of trails, since splits by loc.

    PARAMS:
    - DFSCORES, one row per (pa_class, decoder_class, twind)
    - list_loc, list of 2-tuple locations. Each loc gets turn as train and test.
    """
    from pythonlib.tools.pandastools import aggregGeneral


    if filterdict_train is None:
        filterdict_train = {}
    if filterdict_test is None:
        filterdict_test = {}


    # [Optioanlly], Prune to keep only locations which have lots of prims. This wilkl fail if it ends up pruning too much.
    if auto_prune_locations:
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        # pick the first dflab, assuming this is general
        dflab = DFallpa["pa"].values[0].Xlabels["trials"]
        # n_min_per_var_this = 3 # shouldnt be the general version. here is just pruning locations.
        n_min_per_var_this = n_min_per_var # shouldnt be the general version. here is just pruning locations.
        dflab_tmp, _ = extract_with_levels_of_conjunction_vars(dflab, "seqc_0_loc", [var_train], None, 
                                                            n_min_per_var_this, lenient_allow_data_if_has_n_levels=2, 
                                                            prune_levels_with_low_n=True, 
                                                            balance_no_missed_conjunctions=True, balance_force_to_drop_which=1, 
                                                            plot_counts_heatmap_savepath=f"{savedir_base}/counts_auto_prune_location_post.pdf",
                                                            plot_counts_heatmap_savepath_pre=f"{savedir_base}/counts_auto_prune_location_pre.pdf")
        n_shapes_clean = len(dflab_tmp[var_train].unique())
        n_shapes_before = len(dflab[var_train].unique())
        print(n_shapes_clean, n_shapes_before)

        n_locs_clean = len(dflab_tmp["seqc_0_loc"].unique())
        n_locs_before = len(dflab["seqc_0_loc"].unique())

        assert n_locs_before>1, "only one location, even before clean... shoudl not do held-out lcoation expts."
        assert n_locs_clean>1, "only one location remaining.."
        if False: # to allow pruning to just high n trials cases.
            assert n_shapes_clean/n_shapes_before>0.8, "lost too many shapes based on pruning..."
            assert len(dflab_tmp)/len(dflab)>0.8, "threw out too many trials..."

        trialcodes_keep = dflab_tmp["trialcode"].unique().tolist()

        assert "trialcode" not in filterdict_train
        assert "trialcode" not in filterdict_test

        filterdict_train["trialcode"] = trialcodes_keep
        filterdict_test["trialcode"] = trialcodes_keep
        # list_loc = [loc for loc in list_loc if loc in dflab_tmp["seqc_0_loc"].unique().tolist()]
        list_loc = dflab_tmp["seqc_0_loc"].unique().tolist()

    ### Do decoding
    list_dfscores_across_locs = []
    decoders = []
    list_pa_train = []
    list_pa_test = []
    for train_loc in list_loc:
        for test_loc in list_loc:
            if train_loc != test_loc:
                
                filterdict_train["seqc_0_loc"] = [train_loc]
                filterdict_test["seqc_0_loc"] = [test_loc]

                print("filterdict_train:", filterdict_train)
                print("filterdict_test:", filterdict_test)
                
                # Other params
                savedir = f"{savedir_base}/decoder_training-train_loc={train_loc}-test_loc={test_loc}"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)
                    
                if downsample_trials:
                    # Then best way is to do multiple-folds, so that you aren't throwing away useful data.
                    PLOT_TRAIN = False
                    PLOT_TEST_SPLIT = False
                    PLOT_TEST_CONCATTED = PLOT
                    score_user_test_data = True
                    do_agg_of_user_test_data = True
                    combine_train_test_indices = True # This allows using ALL data (train and test) in location 1 to train.
                    _, dfscores_usertest, _, decoders, _, PAtest = pipeline_train_test_scalar_score_with_splits(DFallpa, 
                                                                                    bregion, var_train, event_train, 
                                                                                    twind_train, filterdict_train,
                                                        var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                        include_null_data=include_null_data, 
                                                        prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, 
                                                        PLOT_TRAIN=PLOT_TRAIN, PLOT_TEST_SPLIT=PLOT_TEST_SPLIT, PLOT_TEST_CONCATTED=PLOT_TEST_CONCATTED,
                                                        which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                        subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                        do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials,
                                                        do_train_splits_nsplits=do_train_splits_nsplits, 
                                                        score_user_test_data=score_user_test_data, allow_multiple_twind_test=allow_multiple_twind_test,
                                                        do_agg_of_user_test_data=do_agg_of_user_test_data,
                                                        split_method="train_only",
                                                        )
                    dfscores = dfscores_usertest
                    Dc = decoders[0]
                    PAtrain = None
                else:     
                    # No need to do splits., Keep all data in training, since train and test are already
                    # different trials (locations).
                    dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, 
                                                                                    twind_train, filterdict_train,
                                                        var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                        include_null_data=include_null_data, prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, 
                                                        PLOT=PLOT, allow_multiple_twind_test=allow_multiple_twind_test,
                                                        which_level_train=which_level_train, which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                        subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                        do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials)
                
                
                
                
                fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "pa_class", "decoder_class", ["twind"]);
                savefig(fig, f"{savedir}/counts_trials-dfscores.pdf")

                # Store dfscores across locations sets
                dfscores["train_loc"] = [train_loc for _ in range(len(dfscores))]
                dfscores["test_loc"] = [test_loc for _ in range(len(dfscores))]
                dfscores["pa_idx"] = dfscores["trialcode"] # Otherwise pa_idx is not unique across test sets.

                list_dfscores_across_locs.append(dfscores)
                decoders.append(Dc)
                list_pa_train.append(PAtrain)
                list_pa_test.append(PAtest)

    # Concat across location sets, so that final dataframe has one datapt per trialcode.
    if len(list_dfscores_across_locs)==0:
        print("! skipping! len(list_dfscores_across_locs)==0")
        return None
    
    dfscores_tmp = pd.concat(list_dfscores_across_locs).reset_index(drop=True)
    DFSCORES = aggregGeneral(dfscores_tmp, 
                                        ["decoder_class", "twind", "trialcode"], 
                                        ["score", "score_norm"], 
                                        ["pa_class", "epoch", "same_class"] # code does sanity check that these are all unqiue values.
                                        )

    if PLOT_TEST:
        # Also plot, for normed score
        sdir = f"{savedir_base}/test-final"
        os.makedirs(sdir, exist_ok=True)
        Dc.scalar_score_df_plot_summary(DFSCORES, sdir, var_score="score")

    return DFSCORES, decoders, list_pa_train, list_pa_test

def pipeline_train_test_scalar_score(DFallpa, bregion, 
                                     var_train, event_train, twind_train, filterdict_train,
                                     var_test, event_test, list_twind_test, filterdict_test,
                                     savedir, include_null_data=False, decoder_method_index=None,
                                     prune_labels_exist_in_train_and_test=True, PLOT=True,
                                     which_level_train="trial", which_level_test="trial", n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=True,
                                     downsample_trials=False,
                                     allow_multiple_twind_test=False,
                                     classifier_version="logistic", 
                                     PLOT_TEST=None):
                                    #  classifier_version="ensemble"):
    """
    Helper to extract dataframe holding decode score for each row of this bregion's test dataset (PA)
    
    WRitten when doing the Shape seuqence TI stuff.

    PARAMS:
    - subtract_baseline, bool, if True, then extracts data using twind in subtract_baseline_twind, and subtracts 
    the resulting scores from each twind in list_twind_test (ie each datapt) -- new column: score_min_base
    RETURNS:
    - (Mainly goal is saves plots and data)
    - dfscores, Dc, PAtrain, PAtest
    - allow_multiple_twind_test, bool, as a switch so that doesnt accidentally forget that is multkpel twind.
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.snstools import rotateLabel

    twind_train = tuple(twind_train)
    list_twind_test = [tuple(twind) for twind in list_twind_test]

    if PLOT_TEST is None:
        PLOT_TEST = PLOT

    ### Train decoder
    # include_null_data=True
    if n_min_per_var is None:
        n_min_per_var=3
    n_min_per_var_good = 10

    do_upsample_balance_fig_path_nosuff = f"{savedir}/upsample_pcs"
    PAtrain, Dc, dfscores_train = train_decoder_helper(DFallpa, bregion, var_train, event_train, twind_train,
                                    PLOT, include_null_data, n_min_per_var, filterdict_train,
                                    which_level_train, decoder_method_index, savedir, n_min_per_var_good,
                                    do_upsample_balance, do_upsample_balance_fig_path_nosuff,
                                    downsample_trials=downsample_trials, classifier_version=classifier_version,
                                    return_dfscores_train=True)
    
    dfscores, PAtest = test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                        which_level_test, savedir, prune_labels_exist_in_train_and_test, PLOT,
                        subtract_baseline, subtract_baseline_twind, allow_multiple_twind_test=allow_multiple_twind_test)
    dflab = PAtest.Xlabels["trials"]
    assert dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]]

    # Normalize scores to min and max, where for a given decoder, min is its score on data that is not same class, whiel
    # max is same class (on training dawta)
    dfmeans = dfscores_train.groupby(["decoder_class", "same_class"])["score"].mean().reset_index()
    # - max, for each decoder
    dfmeans_this = dfmeans[dfmeans["same_class"] == True]
    dfmeans_this = dfmeans_this.drop(["same_class"], axis=1)
    dfmeans_this = dfmeans_this.rename(columns={"score": "score_max"})
    dfscores = pd.merge(dfscores, dfmeans_this, on="decoder_class", how='left')
    # - min, for each decoder
    dfmeans_this = dfmeans[dfmeans["same_class"] == False]
    dfmeans_this = dfmeans_this.drop(["same_class"], axis=1)
    dfmeans_this = dfmeans_this.rename(columns={"score": "score_min"})
    dfscores = pd.merge(dfscores, dfmeans_this, on="decoder_class", how='left')
    # - norm to 0(min) to 1 (max).
    dfscores["score_norm_minmax"] = (dfscores["score"] - dfscores["score_min"])/(dfscores["score_max"] - dfscores["score_min"])

    dflab = PAtest.Xlabels["trials"]
    assert dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]]

    # Subtract base, if that also exists
    if subtract_baseline:
        # - for base data, norm it to (0,1)
        dfscores["score_base_norm_minmax"] = (dfscores["score_base"] - dfscores["score_min"])/(dfscores["score_max"] - dfscores["score_min"])
        # - actual data - base data (all in the 0,1 space)
        dfscores["score_norm_minmax_min_base"] = dfscores["score_norm_minmax"] - dfscores["score_base_norm_minmax"]

    if PLOT:
        # Also plot, for normed score
        sdir = f"{savedir}/test-var_score=score_norm_minmax"
        os.makedirs(sdir, exist_ok=True)
        Dc.scalar_score_df_plot_summary(dfscores, sdir, var_score="score_norm_minmax")

    dflab = PAtest.Xlabels["trials"]
    assert dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]]

    # Save params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({
        "bregion":bregion, 
        "var_train":Dc.VarDecode, 
        "event_train":event_train, 
        "twind_train":twind_train, 
        "filterdict_train":filterdict_train,
        "var_test": var_test, 
        "event_test": event_test, 
        "list_twind_test": list_twind_test, 
        "filterdict_test":filterdict_test,
        "which_level_train":which_level_train,
        "which_level_test":which_level_test,
        "include_null_data":include_null_data,
        "decoder_method_index":decoder_method_index,
        "prune_labels_exist_in_train_and_test":prune_labels_exist_in_train_and_test,
        "n_min_per_var":n_min_per_var,
        "subtract_baseline":subtract_baseline,
        "subtract_baseline_twind":subtract_baseline_twind,
        "do_upsample_balance":do_upsample_balance,
        "downsample_trials":downsample_trials,
        "allow_multiple_twind_test":allow_multiple_twind_test,
        "classifier_version":classifier_version}, 
        path=f"{savedir}/params.txt")

    return dfscores, Dc, PAtrain, PAtest


def pipeline_train_test_scalar_score_mult_train_dataset(DFallpa, bregion, 
                                     list_train_dataset, list_var_train, 
                                     var_test, event_test, list_twind_test, filterdict_test, 
                                     which_level_test="trial",
                                     savedir=None, include_null_data=False, decoder_method_index=None,
                                     prune_labels_exist_in_train_and_test=True, PLOT=True, n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=True, n_min_per_var_good=10,
                                     allow_multiple_twind_test=False, classifier_version="logistic"):
    """
    Helper to extract dataframe holding decode score for each row of this bregion's test dataset (PA)
    
    WRitten when doing the Shape seuqence TI stuff.

    This differs from pipeline_train_test_scalar_score, in that here can concatenate multiple datasets intoa  single 
    dataset before training.

    PARAMS:
    list_train_dataset = ["sp_samp", "pig_samp", "pre_stroke"]
    list_var_train = ["seqc_0_shapesemgrp", "seqc_0_shapesemgrp", "shape_semantic_grp"]
    list_twind_pa = [(-1, 1), (-1, 1), (-0.8, 1.2)]
    - subtract_baseline, bool, if True, then extracts data using twind in subtract_baseline_twind, and subtracts 
    the resulting scores from each twind in list_twind_test (ie each datapt) -- new column: score_min_base
    - n_min_per_var, n mins per var, applies independelt to EACH dataset in list_train_dataset, therefore this can be low.
    RETURNS:
    - (Mainly goal is saves plots and data)
    - dfscores, Dc, PAtrain, PAtest
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.snstools import rotateLabel
    from neuralmonkey.classes.population import concatenate_popanals_flexible
    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import get_dataset_params

    assert len(list_var_train)==len(list_train_dataset), "mistake in entry"
    # New method --> concatenate multiple data events to train decoder.
    # list_train_params = []
    # for train_dataset in list_train_dataset:
    #     event_train, twind_train, filterdict_train, _, _ = get_dataset_params(train_dataset)

    list_train_params = [get_dataset_params(train_dataset) for train_dataset in list_train_dataset]

    list_pa_train = []
    list_dflab_beforesplit = [] # collect list of labels, to then evaluate which has enough data
    for i, (params, var_train) in enumerate(zip(list_train_params, list_var_train)):
        pa_train, _twind_train, PAtrain = train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, params[0], 
                                                params[1], include_null_data, 
                                                n_min_per_var, params[2], params[4], decoder_method_index, PLOT=False)
        

        if PLOT:
            # Plot n trials for training
            from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain.Xlabels["trials"], var_train, "task_kind", None)
            savefig(fig, f"{savedir}/counts-var_train=train_dataset_idx={i}-var_train={var_train}.pdf")

        # Assing a common variable for var_train
        pa_train.Xlabels["trials"]["var_train"] = pa_train.Xlabels["trials"][var_train]
        PAtrain.Xlabels["trials"]["var_train"] = PAtrain.Xlabels["trials"][var_train]
        
        list_dflab_beforesplit.append(PAtrain.Xlabels["trials"])

        # Sanity
        if len(list_pa_train)>0:
            assert list_pa_train[-1].X.shape[0] == pa_train.X.shape[0]
            assert list_pa_train[-1].X.shape[2] == pa_train.X.shape[2]
        
        list_pa_train.append(pa_train)
        print("Extracted a single pa_train: ", pa_train.X.shape)
        # print(PAtrain.X.shape)

    # Concatenate all the training data
    pa_train_all, _= concatenate_popanals_flexible(list_pa_train)

    # Train
    print("Input data to train decoder: ", pa_train_all.X.shape)
    do_upsample_balance_fig_path_nosuff = f"{savedir}/upsample_pcs"

    if classifier_version == "ensemble":
        # Train
        Dc = DecoderEnsemble(pa_train_all, "var_train", _twind_train)
        Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)
    else:
        Dc = Decoder(pa_train_all, "var_train", _twind_train)
        Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)

    # A flag for "good" labels --> i.e. those in decoder that enough trials. 
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    df = pd.concat(list_dflab_beforesplit).reset_index(drop=True)
    _df, _ = extract_with_levels_of_var_good(df, ["var_train"], n_min_per_var_good)
    labels_decoder_good = _df[Dc.VarDecode].unique().tolist()
    Dc.LabelsDecoderGood = labels_decoder_good

    # Run test
    dfscores, PAtest = test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                            which_level_test, savedir, prune_labels_exist_in_train_and_test, PLOT, subtract_baseline, 
                            subtract_baseline_twind, allow_multiple_twind_test=allow_multiple_twind_test)

    # Save params
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    writeDictToTxtFlattened({
        "bregion":bregion, 
        "list_train_dataset":list_train_dataset,
        "list_var_train":list_var_train,
        "var_test": var_test, 
        "event_test": event_test, 
        "list_twind_test": list_twind_test, 
        "filterdict_test":filterdict_test,
        "which_level_test":which_level_test,
        "include_null_data":include_null_data,
        "decoder_method_index":decoder_method_index,
        "prune_labels_exist_in_train_and_test":prune_labels_exist_in_train_and_test,
        "n_min_per_var":n_min_per_var,
        "subtract_baseline":subtract_baseline,
        "subtract_baseline_twind":subtract_baseline_twind,
        "do_upsample_balance":do_upsample_balance, 
        "n_min_per_var_good":n_min_per_var_good,
        "allow_multiple_twind_test":allow_multiple_twind_test, 
        "classifier_version":classifier_version}, 
        path=f"{savedir}/params.txt")

    return dfscores, Dc, PAtrain, PAtest

def trainsplit_timeseries_score_wrapper(decoders, trainsets, twind):
    """
    Operates on output of train-test split pipeline.
    
    Helper to compute timeseries (probs) over all train-test splits, and then
    combine results
    PARAMS:
    - decoder, trainsets, the output of pipeline_train_test_scalar_score_with_splits
    RETURNS:
    - PAprobs, holding probs across classes.
    """

    list_probs_mat_all = []
    # list_patest = []
    list_dflab = []
    labels = None
    times = None
    for idx_trainsplit in range(len(decoders)):
        Dc = decoders[idx_trainsplit]
        PAtrain_orig = trainsets[idx_trainsplit]["PAtrain_orig"]
        test_index_PAtrain_orig = trainsets[idx_trainsplit]["test_index_PAtrain_orig"]

        # Slice out the test dataset
        patest = PAtrain_orig.slice_by_dim_indices_wrapper("trials", test_index_PAtrain_orig, reset_trial_indices=True)
        dflab = patest.Xlabels["trials"]

        # Get these trials
        _, probs_mat_all, _times, _labels = Dc.timeseries_score_wrapper(patest, twind, indtrials=None)
        if labels is None:
            labels = _labels
            times = _times
        else:
            from pythonlib.tools.checktools import check_objects_identical
            assert check_objects_identical(labels, _labels)
            assert check_objects_identical(times, _times)

        # Collect
        list_probs_mat_all.append(probs_mat_all)
        # list_patest.append(patest)
        dflab = patest.Xlabels["trials"].copy()
        list_dflab.append(dflab)

    # Combine across heldout sets
    probs_mat_all = np.concatenate(list_probs_mat_all, axis=1) # (labels, trials, times)
    # dflab = pd.concat([patest.Xlabels["trials"] for patest in list_patest]).reset_index(drop=True)
    dflab = pd.concat([dflab for dflab in list_dflab]).reset_index(drop=True)

    from neuralmonkey.classes.population import PopAnal
    PAprobs = PopAnal(probs_mat_all, times=times, chans=labels, trials=list(range(probs_mat_all.shape[1])))
    PAprobs.Xlabels["trials"] = dflab.copy().reset_index(drop=True)

    return PAprobs, labels, times

def analy_eyefixation_dfscores_condition(dfscores, dflab, var_test):
    """
    Wrapper for conditioning dfscores from analyses looking at actiivtiyg alinge dto eye fixations, and computing 
    decoding of shape fixation vs. shape draw.
    """
    assert var_test == "shape-fixation", "assuming this for below."
    
    # Normalize decode by subtracting mean within each decoder class, when it is not fixated
    dfscores["decoder_class_was_fixated"] = [row["decoder_class"] == row["pa_class"] for i, row in dfscores.iterrows()]

    # if var_test == "shape-fixation":
    #     # Then each decoder normalize to when 
    from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping_return_same_len_df
    dfscores, _, _ = datamod_normalize_row_after_grouping_return_same_len_df(dfscores, "decoder_class_was_fixated", 
                                                                            ["decoder_class"], "score", False, True, True)
    
    # Add variables
    def f(smi):
        if smi == -1:
            return "withinfixation"
        elif smi in [0,1]:
            return "early"
        elif smi>1:
            return "late"
        else:
            assert False
    if "shape-macrosaccade-index" in dflab.columns:
        dflab["early_late_by_smi"] = [f(smi) for smi in dflab["shape-macrosaccade-index"]]

    def f(event_idx):
        if event_idx<4:
            return "early"
        else:
            return "late"
    if "event_idx_within_trial" in dflab.columns:
        dflab["early_late_by_eidx"] = [f(event_idx) for event_idx in dflab["event_idx_within_trial"]]

    # Add some columns
    assert np.all(dfscores["trialcode"].tolist() == [dflab.iloc[pa_idx]["trialcode"] for pa_idx in dfscores["pa_idx"]])
    for var in ["event_idx_within_trial", "is-first-macrosaccade", "early_late_by_eidx", "early_late_by_smi", 
                "seqc_0_shape", "shape-fixation", "shape-macrosaccade-index"]:
        dfscores[var] = [dflab.iloc[pa_idx][var] for pa_idx in dfscores["pa_idx"]]

    # Note whether each decoder was the shape drawn on this trial
    dfscores["decoder_class_was_first_drawn"] = [row["decoder_class"] == row["seqc_0_shape"] for i, row in dfscores.iterrows()]

    from pythonlib.tools.pandastools import append_col_with_grp_index
    # - column, conjunction of draw and fix
    dfscores = append_col_with_grp_index(dfscores, ["seqc_0_shape", "shape-fixation"], "shape_draw_fix")

    return dfscores


def analy_chars_dfscores_condition(dfscores, dflab):
    """
    Conditions dfscores to then run in analysis pipeline for characters, asking, generally, about parsing, e.g.g,
    activation of shapes that will be drwan.
    """
    # APPEND info related to trials.
    list_decoder_class_idx_in_shapes_drawn = []
    list_decoder_class_was_drawn = []
    list_decoder_class_was_seen = []
    list_decoder_class_was_first_drawn = []

    for _i, row in dfscores.iterrows():

        decoder_class = row["decoder_class"]
        pa_idx = row["pa_idx"]
        trialcode = row["trialcode"]
        epoch = row["epoch"]

        shapes_drawn = dflab.iloc[pa_idx]["shapes_drawn"]
        FEAT_num_strokes_beh = dflab.iloc[pa_idx]["FEAT_num_strokes_beh"]
        # shapes_visible = dflab.iloc[pa_idx]["taskconfig_shp"]
        if decoder_class in shapes_drawn:
            decoder_class_idx_in_shapes_drawn = shapes_drawn.index(decoder_class)
        else:
            decoder_class_idx_in_shapes_drawn = -1
        
        assert FEAT_num_strokes_beh==len(shapes_drawn)
        assert decoder_class_idx_in_shapes_drawn<FEAT_num_strokes_beh
        
        list_decoder_class_idx_in_shapes_drawn.append(decoder_class_idx_in_shapes_drawn)
        list_decoder_class_was_drawn.append(decoder_class in shapes_drawn)
        # list_decoder_class_was_seen.append(decoder_class in shapes_visible)
        list_decoder_class_was_first_drawn.append(decoder_class == shapes_drawn[0])
        
    dfscores["decoder_class_idx_in_shapes_drawn"] = list_decoder_class_idx_in_shapes_drawn
    dfscores["decoder_class_was_drawn"] = list_decoder_class_was_drawn
    # dfscores["decoder_class_was_seen"] = list_decoder_class_was_seen
    dfscores["decoder_class_was_first_drawn"] = list_decoder_class_was_first_drawn

    from pythonlib.tools.pandastools import append_col_with_grp_index
    dfscores = append_col_with_grp_index(dfscores, ["decoder_class_was_drawn", "decoder_class_was_first_drawn"], "decoder_class_was_drawn|firstdrawn")
    dfscores["FEAT_num_strokes_beh"] = [dflab.iloc[pa_idx]["FEAT_num_strokes_beh"] for pa_idx in dfscores["pa_idx"]]

    # Normalize decode by subtracting mean within each decoder class
    from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping_return_same_len_df
    dfscores, _, _ = datamod_normalize_row_after_grouping_return_same_len_df(dfscores, "decoder_class_was_drawn", 
                                                                            ["decoder_class"], "score", False, True, True)
    
    # Pull out character information
    dfscores["character"] = [dflab.iloc[pa_idx]["character"] for pa_idx in dfscores["pa_idx"]]

    return dfscores

def _analy_chars_score_postsamp_plot_timecourse(Dc, PAtest, savedir, twind_test=None):
    """
    For chars analysis, make set of plots of timecourse of probabilties for shapes drawn, splitting into subplots based on 
    n strokes drawn.

    Normalizes timecourse to trails where the decoder class was not drawn.
    
    Goal is to ask if early on in trial, see decode of all the shapes that are being parsed.
    """

    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import probs_timecourse_normalize

    # Plot, coloring by shape 1, 2, 3, ...
    if twind_test is None:
        twind_test = (-0.5, 1.8)

    PAprobs, probs_mat_all, times, labels = Dc.timeseries_score_wrapper(PAtest, twind_test, indtrials=None)

    # Keep only trials where all the shapes drawn are part of decoder
    # This is sort of hacky, becuase if not, then downstream code will fail
    PAprobs.Xlabels["trials"]["shapes_drawn_all_exist_in_decoder"] = [all([sh in labels for sh in shapes_drawn]) for shapes_drawn in PAprobs.Xlabels["trials"]["shapes_drawn"]]

    # Normalize probs
    dflab = PAprobs.Xlabels["trials"]
    for NORM_METHOD in ["minus_not_visible_and_base", "minus_not_visible_timecourse", None]:
        if NORM_METHOD is not None:
            map_decoder_class_to_baseline_trials ={}
            for decoder_class in labels:
                inds_not_drawn = [i for i, shapes_drawn in enumerate(dflab["shapes_drawn"]) if decoder_class not in shapes_drawn]
                map_decoder_class_to_baseline_trials[decoder_class] = inds_not_drawn
            PAprobsThis = probs_timecourse_normalize(Dc, PAprobs, NORM_METHOD, None, map_decoder_class_to_baseline_trials=map_decoder_class_to_baseline_trials)
            YLIMS = (-0.15, 0.25)
        else:
            PAprobsThis = PAprobs
            YLIMS = (-0.2, 0.4)

        for ylims in [None, YLIMS]:
            for filter_n_strokes_using in ["beh", "beh_at_least_n"]:
                list_title_filtdict = [("all", {"shapes_drawn_all_exist_in_decoder":[True]})]
                fig = Dc.timeseries_plot_wrapper(PAprobsThis, list_title_filtdict, None, SIZE=4, ylims=ylims, filter_n_strokes_using=filter_n_strokes_using)
                savefig(fig, f"{savedir}/timecourse-NORM_METHOD=filternstrokes={filter_n_strokes_using}-NORM_METHOD={NORM_METHOD}-ylims={ylims}.pdf")
    plt.close("all")

def _analy_chars_score_postsamp_same_image_diff_parse(dfscores, savedir):
    """
    Plots to ask if, for same image (char), trials with diff parses, focus on the decoders
    for shapes that were drawn in one parse but not another, fidn that those decoders have
    diff probabilities acros sthe trials.
    """
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    from pythonlib.tools.pandastools import aggregGeneral
    import seaborn as sns
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
    from pythonlib.tools.statstools import ttest_paired, signrank_wilcoxon
    from pythonlib.tools.pandastools import pivot_table

    # Get (char, decodershape) that has cases both drawn and not drawn
    dfscores_samechar_diffbeh, _ = extract_with_levels_of_conjunction_vars_helper(dfscores, "decoder_class_was_drawn", 
                                                                                        ["character", "decoder_class"], 1, None, 2)
    

    if len(dfscores_samechar_diffbeh)==0:
        return
    
    # Aggregate, so each (character, decoder_class) contributes exactly 1 datapt to (False, True) for 
    # decoder_class_was_drawn
    dfscores_samechar_diffbeh = aggregGeneral(dfscores_samechar_diffbeh, ["character", "decoder_class", "decoder_class_was_drawn"], 
                                            ["score", "score_norm"], nonnumercols=["decoder_class_good"])
    dfscores_samechar_diffbeh = append_col_with_grp_index(dfscores_samechar_diffbeh, ["character", "decoder_class"], "char_class")

    if False:
        from pythonlib.tools.pandastools import grouping_print_n_samples
        grouping_print_n_samples(dfscores_samechar_diffbeh, ["character", "decoder_class", "decoder_class_was_drawn"])
    
    if False:
        from pythonlib.tools.pandastools import summarize_featurediff
        summarize_featurediff(dfscores_samechar_diffbeh)

    ### Plots
    stats_strings = []
    for use_good_decoder in [False, True]:
        if use_good_decoder:
            dfthis = dfscores_samechar_diffbeh[dfscores_samechar_diffbeh["decoder_class_good"]==True].reset_index(drop=True)
        else:
            dfthis = dfscores_samechar_diffbeh

        for var_score in ["score", "score_norm"]:

            try:
                fig = sns.catplot(data=dfthis, x = "decoder_class_was_drawn", y=var_score, kind="point", errorbar=("ci", 68),
                                hue="decoder_class_good")
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/catplot-decoder_class_was_drawn-good_decoder={use_good_decoder}-var_score={var_score}.pdf")
                    plt.close("all")
            except ValueError as err:
                pass
            except Exception as err:
                raise err

            # Scatter
            for var_datapt in ["character", "char_class"]:
                _, fig, = plot_45scatter_means_flexible_grouping(dfthis, "decoder_class_was_drawn", 
                                                                False, True, "decoder_class_good", var_score, var_datapt,
                                                                plot_text=False, shareaxes=True, SIZE=6, alpha=0.3)
                savefig(fig, f"{savedir}/scatter-xy=decoder_class_was_drawn-good_decoder={use_good_decoder}-var_datapt={var_datapt}-var_score={var_score}.pdf")
                
            ### Stats
            dfscores_samechar_diffbeh_pivot = pivot_table(dfthis, ["character", "decoder_class", "decoder_class_good"], 
                                                          ["decoder_class_was_drawn"], ["score", "score_norm"], flatten_col_names=True)
            dat = dfscores_samechar_diffbeh_pivot.loc[:, [f"{var_score}-False", f"{var_score}-True"]].values
            x_false = dat[:,0]
            x_true = dat[:, 1]
            # ttest_paired(x_false, x_true)
            p = signrank_wilcoxon(x_false, x_true)
            stats_strings.append(
                f"use_good_decoder={use_good_decoder} -- var_score={var_score} -- p={p}"
            )
    
    from pythonlib.tools.expttools import writeStringsToFile
    writeStringsToFile(f"{savedir}/stats_signrank.txt", stats_strings)
    plt.close("all")


def _analy_chars_score_postsamp_timecourse_load_image_distances(animal, date):
    """
    Helper to load image-distances between all pairs of characters in this dataset

    Must have previously been run in drawmonkey.
    
    Returns ether:
    - list_char, list of str, unique
    - distance_mat, (nchar, nchar) array of dist

    or

    None, None if this doesnt eixst

    """
    import glob
    import pickle
    SDIR_BASE = f"/lemur2/lucas/analyses/main/char_psycho/{animal}_{date}_*"
    print(SDIR_BASE)
    list_dir = glob.glob(SDIR_BASE)
    if len(list_dir)==1:
        with open(f"{list_dir[0]}/list_char.pkl", "rb") as f:
            list_char = pickle.load(f)
        with open(f"{list_dir[0]}/distances_mat.pkl", "rb") as f:
            distance_mat = pickle.load(f)
        assert len(list_char) == distance_mat.shape[0] == distance_mat.shape[1]        
        assert len(list_char)==len(set(list_char)), "I asusme unique chars"
        return list_char, distance_mat
    else:
        return None, None

def _analy_chars_score_postsamp_timecourse_splitby_image_distance(Dc, PAtest, animal, date, savedir, twind_test=None):
    """
    Analyses, where use image distance (psychometric chars), find pairs of trials with close (or far) distance,
    and for those two groups, plot their timecourse of dcoding, labeling decoder class by whether it
    is present in one,other, or both, or neither, group of trials.

    Question is whether, if image close, then see early on that there is confusion between decoder classes.
    """

    
    ### Load previously saved visual (psychometric) distance between chars
    list_char, distance_mat = _analy_chars_score_postsamp_timecourse_load_image_distances(animal, date)
    if list_char is None:
        print("SKIPPING timecourse_splitby_image_distance -- did not find image distancews")
        return 
    assert len(list_char)==len(set(list_char))

    if False:
        # Distnaces, this was hard-coded for Diego, 231211...
        # THRESH_SIMILAR = 0.06
        THRESH_SIMILAR = 0.1
        THRESH_DIFF = 0.42
    else:
        # Use percentiles to define thresholds
        THRESH_SIMILAR, THRESH_DIFF = np.percentile(distance_mat.flatten(), [10, 90])

    # - Same image distnaces
    fig, ax = plt.subplots()
    ax.set_xlabel("image distnace")
    ax.set_title("thresholds, lower than black=same ... higher than red = diff")

    ax.hist(distance_mat.flatten(), bins=50)
    ax.axvline(THRESH_SIMILAR, color="k")
    ax.axvline(THRESH_DIFF, color="r")
    savefig(fig, f"{savedir}/hist_image_distances.pdf")
    print("You should make sure distance threshold are good for this day! See saved histogram of distances in:", f"{savedir}/hist_image_distances.pdf")

    ### Get PAprobs
    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import probs_timecourse_normalize
    if twind_test is None:
        twind_test = (-0.5, 1.8)
    PAprobs, probs_mat_all, times, labels = Dc.timeseries_score_wrapper(PAtest, twind_test, indtrials=None)
    # Normalize probs
    dflab = PAprobs.Xlabels["trials"]
    NORM_METHOD = "minus_not_visible_and_base"
    map_decoder_class_to_baseline_trials ={}
    for decoder_class in labels:
        inds_not_drawn = [i for i, shapes_drawn in enumerate(dflab["shapes_drawn"]) if decoder_class not in shapes_drawn]
        map_decoder_class_to_baseline_trials[decoder_class] = inds_not_drawn
    PAprobs = probs_timecourse_normalize(Dc, PAprobs, NORM_METHOD, None, map_decoder_class_to_baseline_trials=map_decoder_class_to_baseline_trials)
    
    ### PLOTS
    for image_similarity_mode in ["similar", "diff"]:
        
        # Find pairs of characters that are distance below a threshold
        list_char_pairs = []
        for i, char1 in enumerate(list_char):
            for j, char2 in enumerate(list_char):
                if j>i and not (char1==char2):
                    if image_similarity_mode=="similar":
                        if distance_mat[i, j]<=THRESH_SIMILAR:
                            # Store this pair of char
                            list_char_pairs.append((char1, char2))
                    elif image_similarity_mode=="diff":
                        if distance_mat[i, j]>=THRESH_DIFF:
                            # Store this pair of char
                            list_char_pairs.append((char1, char2))
                    else:
                        assert False
        
        if len(list_char_pairs)==0:
            print(THRESH_SIMILAR, THRESH_DIFF)
            print(np.percentile(distance_mat.flatten(), [10, 90]))
            assert False, "this shoud not be possible, as they are percentiles."

        # Make sure no double dipping (each pair of char is exist only once, with one set order.)
        for (char1, char2) in list_char_pairs:
            assert (char2, char1) not in list_char_pairs, "this means inputed list chars are notunqiue/.."
        from itertools import product

        # # Now get all pairs of trials that match this pair of char
        # NOTE: each pair in this list will be unique (so dont worry about getting flipped)
        list_indtrial_pairs = []
        for (char1, char2) in list_char_pairs:
            inds1 = dflab[dflab["character"] == char1].index.tolist()
            inds2 = dflab[dflab["character"] == char2].index.tolist()

            # get all pairs of trials
            list_indtrial_pairs.extend(list(product(inds1, inds2)))

        print("Found this many pairs of trials:", len(list_indtrial_pairs))
        # Group trials based on visual and shape sim

        ### Given pair of trials, find these groups of decoders: 
        dflab = PAprobs.Xlabels["trials"]
        var_shapes_exist = "shapes_drawn"
        assert labels == Dc.LabelsUnique

        # drawn in A (not B)
        # drawn in B (not A)
        # drawn in both
        # drawn in neither

        trial_labels_all = []
        list_probs_mat = []
        # for indtrial1, indtrial2 in zip(list_indtrial_1, list_indtrial_2):
        # for indtrial1, indtrial2 in product(range(len(dflab)), range(len(dflab))):
        # for indtrial1, indtrial2 in product(range(300), range(300)):
        decoder_labels = ["in trial 1", "in trial 2", "in trial 1 & 2", "in neither"]
        res = []
        for indtrial1, indtrial2 in list_indtrial_pairs:
            
            # indtrial1 = 100
            # indtrial2 = 200

            shapes_exist_1 = dflab.iloc[indtrial1][var_shapes_exist]
            shapes_exist_2 = dflab.iloc[indtrial2][var_shapes_exist]

            # print(shapes_exist_1)
            # print(shapes_exist_2)

            inds_1 = [i for i, sh in enumerate(Dc.LabelsUnique) if (sh in shapes_exist_1) and (sh not in shapes_exist_2)]
            inds_2 = [i for i, sh in enumerate(Dc.LabelsUnique) if (sh not in shapes_exist_1) and (sh in shapes_exist_2)]
            inds_12 = [i for i, sh in enumerate(Dc.LabelsUnique) if (sh in shapes_exist_1) and (sh in shapes_exist_2)]
            inds_none = [i for i, sh in enumerate(Dc.LabelsUnique) if (sh not in shapes_exist_1) and (sh not in shapes_exist_2)]

            if len(inds_1)==0 or len(inds_2)==0 or len(inds_12)==0 or len(inds_none)==0:
                continue

            # Collect mean prob_vec for each class 
            _probmat = np.zeros((4, 2, len(times))) # (n_decoder_groups, 2 trials, times)
            for i, i_trial in enumerate([indtrial1, indtrial2]):
                _probmat[0, i, :] = np.mean(PAprobs.X[inds_1, i_trial, :], axis=0) # (ntimes,) [decoder_classes_drawn]
                _probmat[1, i, :] = np.mean(PAprobs.X[inds_2, i_trial, :], axis=0) # (ntimes,) []
                _probmat[2, i, :] = np.mean(PAprobs.X[inds_12, i_trial, :], axis=0) # (ntimes,)
                _probmat[3, i, :] = np.mean(PAprobs.X[inds_none, i_trial, :], axis=0) # (ntimes,)

                for j, dlab in enumerate(decoder_labels):
                    res.append({
                        "trial_group":f"trial_{i+1}",
                        "decoder_group":dlab,
                        "prob_vec":_probmat[j, i, :],
                    })

            # Save for this pair of trials
            trial_labels_all.append("trial1")
            trial_labels_all.append("trial2")
            list_probs_mat.append(_probmat)
        probs_mat_all = np.concatenate(list_probs_mat, axis=1)

        # Plot timecourse
        fig, axes = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
        # separate axes for trial splits
        trial_labels = ["trial1", "trial2"]
        for tl, ax in zip(trial_labels, axes.flatten()):
            ax.set_title(tl)
            inds_trials = [i for i,lab in enumerate(trial_labels_all) if lab==tl]
            probs_mat_all_this = probs_mat_all[:, inds_trials, :]
            Dc._timeseries_plot_by_shape_drawn_order(probs_mat_all_this, times, decoder_labels, None, ax=ax)
        savefig(fig, f"{savedir}/imsim={image_similarity_mode}-timecourse.pdf")

        ### Recode
        # plot after recoding, to either same or diff trial (i.e., trail1 and 2 are arbitrary)
        dfres = pd.DataFrame(res)
        tmp = []
        for i, row in dfres.iterrows():
            if (row["trial_group"], row["decoder_group"]) in [("trial_1", "in trial 1"), ("trial_2", "in trial 2")]:
                tmp.append("drawn_in_this_trial")
            elif (row["trial_group"], row["decoder_group"]) in [("trial_1", "in trial 2"), ("trial_2", "in trial 1")]:
                tmp.append("drawn_in_other_trial")
            elif row["decoder_group"] == "in trial 1 & 2":
                tmp.append("drawn_in_both_trials")
            elif row["decoder_group"] == "in neither":
                tmp.append("drawn_in_neither_trial")
            else:
                assert False
        dfres["decoder_group_recoded"] = tmp
        dfres["trial_group_recoded"] = "combined"

        # Plot
        fig, axes = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
        decoder_labels = ["drawn_in_this_trial", "drawn_in_other_trial", "drawn_in_both_trials", "drawn_in_neither_trial"]
        trial_labels = ["combined"]
        for tl, ax in zip(trial_labels, axes.flatten()):
            ax.set_title(tl)
            
            # Pull out this subplot (trial group)
            dfres_this = dfres[dfres["trial_group_recoded"]==tl]

            # Get data in shape (ndecoder labs, ntrials, ntimes)
            list_probmat = []
            for lab in decoder_labels:
                probmat = np.stack(dfres_this[dfres_this["decoder_group_recoded"] == lab]["prob_vec"].tolist())[None, :, :]
                list_probmat.append(probmat)
            probs_mat_all_this = np.concatenate(list_probmat, axis=0) # (ndecoder lab, ntrials, ntimes)

            Dc._timeseries_plot_by_shape_drawn_order(probs_mat_all_this, times, decoder_labels, None, ax=ax)
        savefig(fig, f"{savedir}/imsim={image_similarity_mode}-timecourse-recoded.pdf")
    plt.close("all")



def _analy_chars_score_postsamp_image_distance_neural_distance(PAtest, var_test, animal, date, savedir):
    """
    Ask if image distance or shape distance is a better predictor of neural distance, where neural distance 
    is euclidian betwene trajectories -- i.e., this has nothign to do with decoder, but put here since it complements
    ongoign decoder analyses.

    Image distance -- psychometric chars
    Shape distance -- categorical, based on whether has matching shapes... see within.

    # Compute shape distance between chars (n shapes different)/(n shapes same)
    # Get average Cldist

    # For all pairs of chars, relate their image distance to neural distance
    # For all pairs of chars, relate their shape_parse distance to neural distance

    # Show distribution of neural distance for all combos of (image dist, shape_parse)

    """

    ### LOAD IMAGE DISTANCES
    list_char, distance_mat = _analy_chars_score_postsamp_timecourse_load_image_distances(animal, date)
    if list_char is None:
        print("SKIPPING image_distance_neural_distance -- did not find image distancews")
        return 

    ### COMPUTE EUCLIDIAN DISTANCES
    var_pca = var_test
    pca_twind = (0.1, 1.8)
    pca_tbindur = 0.2
    pca_tbin_slice = 0.02
    npcs = 8
    Xredu, PAredu, stats_redu, Xfinal_before_redu, pca = PAtest.dataextract_pca_demixed_subspace(var_pca, None,
                                                    pca_twind, pca_tbindur, # -- PCA params start
                                                    raw_subtract_mean_each_timepoint=False,
                                                    pca_subtract_mean_each_level_grouping=False,
                                                    n_min_per_lev_lev_others=3, prune_min_n_levs = 2,
                                                    n_pcs_subspace_max = npcs, 
                                                    reshape_method="chans_x_trials_x_times",
                                                    pca_tbin_slice=pca_tbin_slice)
    
    # Get pairwise distance between trials.
    # from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single
    twind_eucl = pca_twind
    if False:
        pa = PAtest.slice_by_dim_values_wrapper("times", twind_eucl)
        pa = pa.agg_by_time_windows_binned(0.2, 0.02)
    else:
        pa = PAredu.slice_by_dim_values_wrapper("times", twind_eucl)
    Cl = pa.dataextract_as_distance_matrix_clusters_flex(["character"], version_distance="euclidian",
                                                                            agg_before_distance=False,
                                                                            return_as_single_mean_over_time=True)
    # Sanity check that matches order of input labels
    assert [(char,) for char in PAtest.Xlabels["trials"]["character"]] == Cl.Labels
    fig, X, labels_col, labels_row, ax = Cl.plot_heatmap_data()
    savefig(fig, "eucldist-heatmap.pdf")

    ### HELPER FUNCTIONS
    dflab = PAtest.Xlabels["trials"]

    def image_distance_compute(distance_mat, char1, char2):
        i = list_char.index(char1)
        j = list_char.index(char2)
        return distance_mat[i, j]

    def shapes_drawn_count(dflab, char):
        """ Return mean n strokes for this char across trials
        """
        return np.mean([len(s) for s in dflab[dflab["character"] == char]["shapes_drawn"]])

    def shape_distance_compute(dflab, char1, char2):
        """
        Return scalar distnace between chars in shape space,
        using lsit of strings that are shapes.

        Takes average between all pairs of trails for char1 and char2

        Depends only on the set of shapes, not on drawn order.
        """

        indtrials1 = dflab[dflab["character"] == char1].index.tolist()
        indtrials2 = dflab[dflab["character"] == char2].index.tolist()

        distances = []
        for ind1 in indtrials1:
            for ind2 in indtrials2:
                distances.append(shape_distance_compute_trials(dflab, ind1, ind2))
        
        return np.mean(distances)
    
    def shape_distance_compute_trials(dflab, ind1, ind2):
        """ compute shape distance between two trials
        """
        shapes_drawn_1 = dflab.iloc[ind1]["shapes_drawn"]
        shapes_drawn_2 = dflab.iloc[ind2]["shapes_drawn"]

        n_only_1 = len([sh for sh in shapes_drawn_1 if sh not in shapes_drawn_2])
        n_only_2 = len([sh for sh in shapes_drawn_2 if sh not in shapes_drawn_1])
        n_both_1 = len([sh for sh in shapes_drawn_1 if sh in shapes_drawn_2])
        n_both_2 = len([sh for sh in shapes_drawn_2 if sh in shapes_drawn_1]) # can be different from n_both_1 (i.e, repeated hsapes)

        # shape_distance = (n_only_1 + n_only_2)/(n_only_1 + n_only_2 + n_both_1 + n_both_2)
        shape_distance = (n_only_1 + n_only_2) - (n_both_1 + n_both_2)
        
        if False:
            print(n_only_1, n_only_2, n_both_1, n_both_2, " --> ", shape_distance)

        return shape_distance
    
    map_char_to_n_drawn_strokes = {}
    for char in list_char:
        map_char_to_n_drawn_strokes[char] = shapes_drawn_count(dflab, char)

    ##### Method 1 - datapt = trial 
    # This has advantagne in that can ask for pairs of trials with same shapes drawn
    # Datapt = trial
    dflab = PAtest.Xlabels["trials"]
    res = []
    for i in range(len(dflab)):
        print(i)
        for j in range(len(dflab)):
            # if i>100:
            #     continue
            if j>i:
                char1 = dflab.iloc[i]["character"]
                char2 = dflab.iloc[j]["character"]

                # Get neural distance
                dist_neural = Cl.Xinput[i, j]
                # dist_neural = Cl.index_find_dat_by_label((char1,), (char2,))
                assert Cl.Labels[i] == (char1,)
                assert Cl.Labels[j] == (char2,)

                # Get shape distance
                dist_shape = shape_distance_compute_trials(dflab, i, j)

                # Get image distance
                dist_image = image_distance_compute(distance_mat, char1, char2)

                sd1 = dflab.iloc[i]["shapes_drawn"]
                sd2 = dflab.iloc[j]["shapes_drawn"]
                drew_same_shapes = set(sd1) == set(sd2)
                drew_same_first_shape = sd1[0] == sd2[0]
                drew_same_secondplus_shapes = (set(sd1[1:]) == set(sd2[1:])) and (len(sd1[1:])>0) and ((len(sd2[1:])>0))
                drew_all_diff_shapes = len([sh for sh in sd1 if sh in sd2])==0

                res.append({
                    "dist_neural":dist_neural,
                    "dist_shape":dist_shape,
                    "dist_image":dist_image,
                    "char1":char1,
                    "char2":char2,
                    "indtrial1":i,
                    "indtrial2":j,
                    "n_strokes_char1":len(dflab.iloc[i]["shapes_drawn"]),
                    "n_strokes_char2":len(dflab.iloc[j]["shapes_drawn"]),
                    "drew_same_shapes":drew_same_shapes,
                    "drew_same_secondplus_shapes":drew_same_secondplus_shapes,
                    "drew_same_first_shape":drew_same_first_shape,
                    "drew_all_diff_shapes":drew_all_diff_shapes
                })
    dfres = pd.DataFrame(res)

    # Condition dfres
    dfres["n_strokes_char1_binned"] = pd.cut(dfres["n_strokes_char1"], 3, labels=False)
    dfres["n_strokes_char2_binned"] = pd.cut(dfres["n_strokes_char2"], 3, labels=False)
    dfres["dist_shape_binned"] = pd.cut(dfres["dist_shape"], 4, labels=False)
    dfres["dist_image_binned"] = pd.cut(dfres["dist_image"], 4, labels=False)

    from pythonlib.tools.pandastools import append_col_with_grp_index
    dfres = append_col_with_grp_index(dfres, ["drew_all_diff_shapes", "drew_same_shapes", "drew_same_first_shape"], "drew_diff_same_samefirst")
    dfres = append_col_with_grp_index(dfres, ["drew_all_diff_shapes", "drew_same_secondplus_shapes", "drew_same_first_shape"], "drew_diff_samesecondplus_samefirst")
    dfres = append_col_with_grp_index(dfres, ["n_strokes_char1_binned", "n_strokes_char2_binned"], "n_strokes_char1_char2_binned")

    dfres["drew_diff_samesecondplus_samefirst"].value_counts()
    dfres["image_same"] = [char1==char2 for char1, char2 in dfres.loc[:, ["char1", "char2"]].values.tolist()]
    dfres["image_same"].value_counts()

    ### PLOTS
    from pythonlib.tools.pandastools import stringify_values
    dfres = stringify_values(dfres)

    # Distributions, and bin labels
    fig, axes = plt.subplots(2,3, figsize=(14,10))
    dfres["dist_neural_binned"] = "ignore"
    for ax, k in zip(axes.flatten(), ["dist_neural", "dist_shape", "dist_image", "n_strokes_char1", "n_strokes_char2"]):
        # ax.hist(dfres[k], bins=50, label=dfres[f"{k}_binned"])
        # ax.plot(dfres[f"{k}_binned"], dfres[k], "o")
        sns.histplot(x=dfres[k], hue=[str(l) for l in dfres[f"{k}_binned"]], ax=ax, element="step")
        ax.set_title(k)
    savefig(fig, f"{savedir}/histograms_distance_and_bin_labels.pdf")

    # CatPlots
    for var1, var2 in [
        ("drew_diff_samesecondplus_samefirst", "image_same"),
        ("drew_diff_samesecondplus_samefirst", "dist_image_binned"),
        ]:
        fig = sns.catplot(data=dfres, x=var1, y="dist_neural", jitter=True, alpha=0.2, hue=var2)
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-bulk-1.pdf")

        fig = sns.catplot(data=dfres, x=var1, y="dist_neural", kind="point", hue=var2)
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-bulk-2.pdf")

        fig = sns.catplot(data=dfres, x=var2, y="dist_neural", jitter=True, alpha=0.2, hue=var1)
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-bulk-3.pdf")
        
        fig = sns.catplot(data=dfres, x=var2, y="dist_neural", kind="point", hue=var1)
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-bulk-4.pdf")

        fig = sns.catplot(data=dfres, x=var1, y="dist_neural", jitter=True, alpha=0.2, hue=var2, col="n_strokes_char1_binned", row="n_strokes_char2_binned")
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-splitby-nstrokes-1.pdf")

        fig = sns.catplot(data=dfres, x=var1, y="dist_neural", kind="point", hue=var2, col="n_strokes_char1_binned", row="n_strokes_char2_binned")
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-splitby-nstrokes-2.pdf")

        fig = sns.catplot(data=dfres, x=var2, y="dist_neural", jitter=True, alpha=0.2, hue=var1, col="n_strokes_char1_binned", row="n_strokes_char2_binned")
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-splitby-nstrokes-3.pdf")

        fig = sns.catplot(data=dfres, x=var2, y="dist_neural", kind="point", hue=var1, col="n_strokes_char1_binned", row="n_strokes_char2_binned")
        savefig(fig, f"{savedir}/catplot-var1={var1}-var2={var2}-splitby-nstrokes-4.pdf")
        plt.close("all")

    # Heatmaps
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
    n_min_trials = 5

    col_var = "drew_diff_samesecondplus_samefirst"
    # col_var = "drew_diff_same_samefirst"

    col_levels = ["1|0|0", "0|0|0", "0|0|1", "0|1|1"]
    # col_levels = ["0|0|0", "0|0|1", "0|1|1"]
    dfresthis, dict_dfthis = extract_with_levels_of_conjunction_vars(dfres, col_var, ["n_strokes_char1_char2_binned", "dist_image_binned"], 
                                                                    col_levels, n_min_trials,
                                                                    lenient_allow_data_if_has_n_levels=len(col_levels)-2, 
                                                                    prune_levels_with_low_n=True, 
                                                                    plot_counts_heatmap_savepath=f"{savedir}/counts1.pdf")

    row_levels = sorted(dfresthis["dist_image_binned"].unique().tolist())
    dfresthis, dict_dfthis = extract_with_levels_of_conjunction_vars(dfresthis, "dist_image_binned", ["n_strokes_char1_char2_binned", col_var], 
                                                                    row_levels, n_min_trials,
                                                                    lenient_allow_data_if_has_n_levels=len(row_levels)-2, 
                                                                    prune_levels_with_low_n=True, 
                                                                    plot_counts_heatmap_savepath=f"{savedir}/counts2.pdf")

    # Heatmap
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    norm_method = None
    for zlims in [None, (0.85, 1.3)]:
        fig, axes = plot_subplots_heatmap(dfresthis, "dist_image_binned", col_var, "dist_neural", 
                            "n_strokes_char1_char2_binned", False, True, norm_method, False, False, ZLIMS=zlims,
                            col_values=col_levels, row_values=row_levels)
        savefig(fig, f"{savedir}/heatmaps-zlims={zlims}.pdf")

        fig, axes = plot_subplots_heatmap(dfresthis, "dist_image_binned", col_var, "dist_neural", None,
                                          share_zlim=True, norm_method=norm_method, ZLIMS=zlims,
                                          col_values=col_levels, row_values=row_levels)
        savefig(fig, f"{savedir}/heatmaps-zlims={zlims}-combined.pdf")

    plt.close("all")

    # KEEP ALL (just prune to keep only conjucntiosn with n trials)
    dfresthis, dict_dfthis = extract_with_levels_of_conjunction_vars(dfres, col_var, ["n_strokes_char1_char2_binned", "dist_image_binned"], 
                                                                     n_min_across_all_levs_var=n_min_trials, lenient_allow_data_if_has_n_levels=1, 
                                                                    prune_levels_with_low_n=True, 
                                                                    plot_counts_heatmap_savepath=f"{savedir}/counts3.pdf")
    # Heatmap
    for zlims in [None, (0.85, 1.3)]:
        fig, axes = plot_subplots_heatmap(dfresthis, "dist_image_binned", col_var, "dist_neural", 
                            "n_strokes_char1_char2_binned", False, True, norm_method, False, False, ZLIMS=zlims,
                            col_values=col_levels, row_values=row_levels)
        savefig(fig, f"{savedir}/heatmaps-zlims={zlims}-include_all.pdf")
        
        fig, axes = plot_subplots_heatmap(dfresthis, "dist_image_binned", col_var, "dist_neural", None,
                                          share_zlim=True, norm_method=norm_method, ZLIMS=zlims,
                                          col_values=col_levels, row_values=row_levels)
        savefig(fig, f"{savedir}/heatmaps-zlims={zlims}-include_all-combined.pdf")

    plt.close("all")

def analy_chars_score_postsamp(DFallpa, SAVEDIR, animal, date):
    """
    Wrapper for all analyses for characters.
    
    Speicific analysis script for characters, 
    Ask if shape represetnations are activatied during char planning (post-samp), using variety of 
    methods for training decoder.

    For each run, trains decoder for shape (e..g, using sp and pig samp).

    Then tests on post-samp for char.

    Plots assess whether activation for 1st, 2nd ... strokes are > chance.

    """
    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import get_dataset_params
    import os

    # Hard coded params:
    include_null_data = False
    n_min_per_var = 3
    assert n_min_per_var<5, "this is PER datsaet, so the sum will be large..."
    subtract_baseline=False
    subtract_baseline_twind=(-0.45, -0.05)
    PLOT = True

    ### Test params
    # - post-samp
    # test_dataset = "char_samp_post"
    # var_test = "seqc_1_shapesemgrp"
    test_dataset = "char_samp_post"
    var_test = "seqc_0_shapesemgrp"

    ######### TRAINING PARAMS
    LIST_TRAIN_DATASET = []
    LIST_VAR_TRAIN = []
    LIST_SAVE_SUFF = []

    ### Train params
    list_train_dataset = ["sp_samp", "pig_samp", "sp_pig_pre_stroke_all"]
    list_var_train = ["seqc_0_shapesemgrp", "seqc_0_shapesemgrp", "shape_semantic_grp"]
    save_suff = "|".join(list_train_dataset)
    LIST_TRAIN_DATASET.append(list_train_dataset)
    LIST_VAR_TRAIN.append(list_var_train)
    LIST_SAVE_SUFF.append(save_suff)

    list_train_dataset = ["sp_samp", "pig_samp"]
    list_var_train = ["seqc_0_shapesemgrp", "seqc_0_shapesemgrp"]
    save_suff = "|".join(list_train_dataset)
    LIST_TRAIN_DATASET.append(list_train_dataset)
    LIST_VAR_TRAIN.append(list_var_train)
    LIST_SAVE_SUFF.append(save_suff)

    list_train_dataset = ["sp_samp"]
    list_var_train = ["seqc_0_shapesemgrp"]
    save_suff = "|".join(list_train_dataset)
    LIST_TRAIN_DATASET.append(list_train_dataset)
    LIST_VAR_TRAIN.append(list_var_train)
    LIST_SAVE_SUFF.append(save_suff)

    list_train_dataset = ["pig_samp"]
    list_var_train = ["seqc_0_shapesemgrp"]
    save_suff = "|".join(list_train_dataset)
    LIST_TRAIN_DATASET.append(list_train_dataset)
    LIST_VAR_TRAIN.append(list_var_train)
    LIST_SAVE_SUFF.append(save_suff)

    list_train_dataset = ["sp_pig_pre_stroke_all"]
    list_var_train = ["shape_semantic_grp"]
    save_suff = "|".join(list_train_dataset)
    LIST_TRAIN_DATASET.append(list_train_dataset)
    LIST_VAR_TRAIN.append(list_var_train)
    LIST_SAVE_SUFF.append(save_suff)

    # Extract some params
    list_bregion = DFallpa["bregion"].unique().tolist()
    event_test, _, filterdict_test, list_twind_test, which_level_test = get_dataset_params(test_dataset)

    for list_train_dataset, list_var_train, save_suff in zip(LIST_TRAIN_DATASET, LIST_VAR_TRAIN, LIST_SAVE_SUFF):
        for bregion in list_bregion:
            # Other params
            savedir = f"{SAVEDIR}/traindata={save_suff}-testdata={test_dataset}/{bregion}/decoder_training_mult"
            os.makedirs(savedir, exist_ok=True)
            print(savedir)

            dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score_mult_train_dataset(DFallpa, bregion, 
                                                list_train_dataset, list_var_train, 
                                                var_test, event_test, list_twind_test, filterdict_test, 
                                                which_level_test, savedir, include_null_data, 
                                                prune_labels_exist_in_train_and_test=True, PLOT=PLOT, n_min_per_var=n_min_per_var,
                                                subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind)
            
            dflab = PAtest.Xlabels["trials"]
            dfscores = analy_chars_dfscores_condition(dfscores, dflab)
            dfscores["bregion"] = bregion

            # (1) keep only successful trials.
            # Keep good characters
            trialcodes_success = dflab[dflab["FEAT_num_strokes_beh"]>1]["trialcode"].tolist()
            print("dfscores, before prune to good trialcodes... ", len(dfscores))
            dfscores_success = dfscores[dfscores["trialcode"].isin(trialcodes_success)].reset_index(drop=True)
            print("... after ", len(dfscores_success))


            from pythonlib.tools.pandastools import stringify_values
            dfscores_str = stringify_values(dfscores)
            dfscores_str_success = stringify_values(dfscores_success)

            ############# PLOTS
            savedir = f"{SAVEDIR}/traindata={save_suff}-testdata={test_dataset}/{bregion}/summary_plots"
            os.makedirs(savedir, exist_ok=True)

            dfthis = dfscores_str_success
            for var_score in ["score", "score_norm"]:
                # from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import plot_scalar_all
                # plot_scalar_all(dfscores_str_success, savedir, var_score="score")
                import seaborn as sns

                for xvar in ["decoder_class_was_drawn", "decoder_class_was_drawn|firstdrawn", "decoder_class_idx_in_shapes_drawn"]:
                    for hue in [None, "FEAT_num_strokes_beh", "decoder_class", "decoder_class_good"]:
                        fig = sns.catplot(data=dfthis, x = xvar, y=var_score, 
                                        col="decoder_class_good", col_wrap=6,
                                            kind="point", errorbar=("ci", 68), hue = hue)
                        for ax in fig.axes.flatten():
                            ax.axhline(0, color="k", alpha=0.5)
                        savefig(fig, f"{savedir}/catplot-{xvar}-hue={hue}-var_score={var_score}.pdf")
                        plt.close("all")

                from pythonlib.tools.pandastools import plot_subplots_heatmap
                if var_score == "score":
                    norm_method = None
                    zlims = [0, 0.45]
                    diverge = False
                elif var_score == "score_norm":
                    norm_method = None
                    zlims = [-0.28, 0.28]
                    diverge = True
                elif var_score == "score_min_base":
                    norm_method = None
                    zlims = [-0.3, 0.3]
                    diverge = True
                else:
                    assert False

                for row_var in ["decoder_class_was_drawn|firstdrawn", "decoder_class_idx_in_shapes_drawn", "decoder_class_was_drawn"]:
                    for col_var in ["FEAT_num_strokes_beh", "decoder_class"]:
                        sub_var = "decoder_class_good"
                        row_values = sorted(dfthis[row_var].unique())[::-1]
                        col_values = sorted(dfthis[col_var].unique())
                        fig, axes = plot_subplots_heatmap(dfthis, row_var, col_var, var_score, sub_var,
                                            annotate_heatmap=False, norm_method=norm_method, 
                                            row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                        savefig(fig, f"{savedir}/heatmap-{row_var}-vs-{col_var}-var_score={var_score}.pdf")

                    # row_var = "decoder_class_idx_in_shapes_drawn"
                    # col_var = "FEAT_num_strokes_beh"
                    # sub_var = "decoder_class_good"
                    # row_values = sorted(dfthis[row_var].unique())[::-1]
                    # col_values = sorted(dfthis[col_var].unique())
                    # fig, axes = plot_subplots_heatmap(dfthis, row_var, col_var, var_score, sub_var,
                    #                     annotate_heatmap=False, norm_method=norm_method, 
                    #                     row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                    # savefig(fig, f"{savedir}/heatmap-{row_var}-vs-{col_var}-var_score={var_score}.pdf")


                # row_var = "decoder_class_was_drawn"
                # col_var = "decoder_class"
                # sub_var = "decoder_class_good"
                # row_values = sorted(dfthis[row_var].unique())[::-1]
                # col_values = sorted(dfthis[col_var].unique())
                # fig, axes = plot_subplots_heatmap(dfthis, row_var, col_var, var_score, sub_var,
                #                     annotate_heatmap=False, norm_method=norm_method, 
                #                     row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                # savefig(fig, f"{savedir}/heatmap-{row_var}-vs-{col_var}-var_score={var_score}.pdf")

                # plt.close("all")

            ### Plots (same image, diff parse)
            savedir = f"{SAVEDIR}/traindata={save_suff}-testdata={test_dataset}/{bregion}/same_char_diff_parse"
            os.makedirs(savedir, exist_ok=True)
            _analy_chars_score_postsamp_same_image_diff_parse(dfscores_success, savedir)

            ### Plots (timecourse)
            savedir = f"{SAVEDIR}/traindata={save_suff}-testdata={test_dataset}/{bregion}/timecourse"
            os.makedirs(savedir, exist_ok=True)
            _analy_chars_score_postsamp_plot_timecourse(Dc, PAtest, savedir, twind_test=(-0.5, 1.8))

            ### Plots (image distance --> timecourse)
            savedir = f"{SAVEDIR}/traindata={save_suff}-testdata={test_dataset}/{bregion}/image_distance-timecourse"
            os.makedirs(savedir, exist_ok=True)
            _analy_chars_score_postsamp_timecourse_splitby_image_distance(Dc, PAtest, animal, date, savedir, twind_test=(-0.5, 1.8))

            ### Plots (image distance vs. shape distance)
            savedir = f"{SAVEDIR}/traindata={save_suff}-testdata={test_dataset}/{bregion}/image_distance-vs-shape_distance"
            os.makedirs(savedir, exist_ok=True)
            _analy_chars_score_postsamp_image_distance_neural_distance(PAtest, var_test, animal, date, savedir)

def analy_psychoprim_dfscores_condition(dfscores, morphset_this_dfscores, DSmorphsets, 
                                        map_morphsetidx_to_assignedbase_or_ambig,
                                        map_tcmorphset_to_info):
    """
    Condition dfscores for psycho prim analysis.
    
    """
    dfscores = dfscores.copy()
    morphset = morphset_this_dfscores # The morphset for this dfscores

    ##### Store features related to psycho
    dfscores["morph_set_idx"] = morphset
    list_idx = sorted(DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == morphset]["morph_idxcode_within_set"].unique().tolist())
    dfscores["pa_morph_assigned_baseorambig"] = [map_morphsetidx_to_assignedbase_or_ambig[(morphset, idx_morph_temp)] for idx_morph_temp in dfscores["pa_class"]]

    #### Recoding classes, to allow averaging over all data.

    # PA class --> already done, "pa_morph_assigned_baseorambig"
    # Decoder class
        # For each pa, group the decoder classes into semantic groupings, to allow averaging over all data.
        # (base1, interm1, same, interm2, base2)

    tmp = []
    for i, row in dfscores.iterrows():

        if row["decoder_class"] == 0:
            decoder_class_semantic_good = "base1"
        elif row["decoder_class"] == 99:
            decoder_class_semantic_good = "base2"
        elif row["decoder_class"] == row["pa_class"]:
            assert row["same_class"]
            decoder_class_semantic_good = "same"
        else:
            # Is this to left or right of pa class
            if row["decoder_class"]<row["pa_class"]:
                decoder_class_semantic_good = "interm1"
            elif row["decoder_class"]>row["pa_class"]:
                decoder_class_semantic_good = "interm2"
            else:
                assert False
        tmp.append(decoder_class_semantic_good)
    dfscores["decoder_class_semantic_good"] = tmp


    ##### Normalize each decoder, to range between min and max across the pa_classes (their means)
    y_var = "score"

    group_max = dfscores.groupby(["decoder_class"])[y_var].max().reset_index().rename(columns={y_var: f'{y_var}_max'})
    group_min = dfscores.groupby(["decoder_class"])[y_var].min().reset_index().rename(columns={y_var: f'{y_var}_max'})

    group_max = dfscores.groupby(["decoder_class", "pa_class"])[y_var].mean().reset_index().groupby(["decoder_class"]).max().reset_index()
    group_max = group_max.rename(columns={"pa_class":"pa_class_max", y_var: f'{y_var}_max_paclass'})

    group_min = dfscores.groupby(["decoder_class", "pa_class"])[y_var].mean().reset_index().groupby(["decoder_class"]).min().reset_index()
    group_min = group_min.rename(columns={"pa_class":"pa_class_min", y_var: f'{y_var}_min_paclass'})

    dfscores = pd.merge(dfscores, group_max, on="decoder_class", how='left')
    dfscores = pd.merge(dfscores, group_min, on="decoder_class", how='left')

    if "score_min_paclass" not in dfscores.columns:
        print(group_min)
        print(dfscores.columns)
        print(len(dfscores))
        assert False
    dfscores["score_norm"] = (dfscores["score"] - dfscores["score_min_paclass"])/(dfscores["score_max_paclass"] - dfscores["score_min_paclass"])


    ######### Additional preprocessing of dscores
    # -----------------
    dfscores["trial_morph_assigned_to_which_base"] = [map_tcmorphset_to_info[(row["trialcode"], row["morph_set_idx"])][0] for i, row in dfscores.iterrows()]

    # Only keep cases with enough trials
    n1 = sum(dfscores["trial_morph_assigned_to_which_base"] == "not_enough_trials")
    n2 = len(dfscores)
    assert n1/n2<0.3, f"throwing away >0.3 of data. is this expected? {n1}/{n2}"
    dfscores = dfscores[dfscores["trial_morph_assigned_to_which_base"] != "not_enough_trials"].reset_index(drop=True)

    # -----------------
    # New column -- was the decoder for base1 or base2 drawn on this trial.
    # Each trial must get a match to one decoder, even if its ambiguous
    tmp = []
    for i, row in dfscores.iterrows():
        
        if row["trial_morph_assigned_to_which_base"] in ["base1", "ambig_base1", "not_ambig_base1"]:
            drew = "base1"
        elif row["trial_morph_assigned_to_which_base"] in ["base2", "ambig_base2", "not_ambig_base2"]:
            drew = "base2"
        else:
            print(row["trial_morph_assigned_to_which_base"])
            assert False

        if row["decoder_class"] == 0 and drew == "base1":
            decoder_is_drawn_base_prim = True
        elif row["decoder_class"] == 99 and drew == "base2":
            decoder_is_drawn_base_prim = True
        else:
            decoder_is_drawn_base_prim = False    

        # if i==1:
        #     print(row)
        #     print(drew)
        #     print(decoder_is_drawn_base_prim)
        #     assert False
        # assert False
        tmp.append(decoder_is_drawn_base_prim)

    dfscores["decoder_class_was_drawn"] = tmp

    # Assign a new variable, which is (idxwithin, which_assigned), which allows grouping by what actually drawn
    from pythonlib.tools.pandastools import append_col_with_grp_index
    dfscores = append_col_with_grp_index(dfscores, ["pa_class", "trial_morph_assigned_to_which_base"], "idx_within|assigned")
    dfscores = append_col_with_grp_index(dfscores, ["morph_set_idx", "pa_class"], "morph_set_idx|idx_within")

    ### RECODE, to combine data across the two directions (base1 or base2 facing)
    def F(x):
        assign = x["trial_morph_assigned_to_which_base"]
        decoder = x["decoder_class_semantic_good"]

        if decoder == "same":
            return "same"
        elif assign in ['not_ambig_base1', 'base1', 'ambig_base1']:
            if decoder == "base1":
                return "basethis"
            elif decoder == "interm1":
                return "intermthis"
            if decoder == "base2":
                return "baseother"
            elif decoder == "interm2":
                return "intermother"
            else:
                print(decoder)
                assert False
        elif assign in ['not_ambig_base2', 'base2', 'ambig_base2']:
            if decoder == "base1":
                return "baseother"
            elif decoder == "interm1":
                return "intermother"
            if decoder == "base2":
                return "basethis"
            elif decoder == "interm2":
                return "intermthis"
            else:
                assert False
        
    dfscores["recoded_decoder"] = [F(row) for i, row in dfscores.iterrows()]

    def F(x):
        assign = x["trial_morph_assigned_to_which_base"]
        if assign in ['not_ambig_base1', 'not_ambig_base2']:
            return "not_ambig"
        elif assign in ['ambig_base1', 'ambig_base2']:
            return "ambig"
        elif assign in ['base1', 'base2']:
            return "base"
        else:
            print(decoder)
            assert False
        
    dfscores["recoded_trial_morph"] = [F(row) for i, row in dfscores.iterrows()]

    # Sanity check -- each trialcode has a single case that is True
    if not np.all(dfscores[dfscores["decoder_class_was_drawn"]==True].groupby(["morph_set_idx", "trialcode"]).size().reset_index(drop=True)==1):
        print(dfscores[dfscores["decoder_class_was_drawn"]==True].groupby(["morph_set_idx", "trialcode"]).size().reset_index(drop=True))
        assert np.all(dfscores[dfscores["decoder_class_was_drawn"]==True].groupby(["morph_set_idx", "trialcode"]).size().reset_index(drop=True)==1), "expect each (morph_set_idx, trialcode) to have 1 decoder (i.e, varying by idx within morpht) that matches it"

    return dfscores

def analy_psychoprim_prepare_beh_dataset(animal, date, cetegory_expt_version, 
                                         savedir="/tmp", D=None, plot_drawings=True):
    """
    Extract beh related to psychometric prims, behavior.
    """
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper, psychogood_plot_drawings_morphsets, params_remap_angle_to_idx_within_morphset

    date = int(date)

    if D is None:
        from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import prepare_beh_dataset
        _, D, _, _, _ = prepare_beh_dataset(animal, date, do_syntax_rule_stuff=False)        

    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper_GOOD
    return psychogood_preprocess_wrapper_GOOD(D, NEURAL_VERSION=True, NEURAL_SAVEDIR=savedir, NEURAL_PLOT_DRAWINGS=plot_drawings,
                                              cetegory_expt_version=cetegory_expt_version)
    
def _analy_psychoprim_score_postsamp_plot_scores(dfscores, savedir, do_agg_over_trials=True):
    """
    Plot results for psychoprim analyses.

    dfscores shoudl have trial-level data. Within here, will first agg so that each datapt is (morphidx, idx_within).
    """
    from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap, stringify_values, aggregGeneral

    assert len(dfscores["twind"].unique())==1, "assuming this below"
    
    ### First, agg data, so each datapt is single morph_set_idx|idx_within
    dfscores_orig = dfscores.copy()
    if do_agg_over_trials:
        dfscores = aggregGeneral(dfscores, 
                                 ["idx_within|assigned", "morph_set_idx|idx_within", "pa_class", "decoder_class", "twind", "trial_morph_assigned_to_which_base"], 
                                 ["score", "score_norm"], nonnumercols="all")

    if "morph_set_idx" not in dfscores.columns:
        assert "morph_set_idx" in dfscores_orig.columns
        print("This type: ", type(dfscores_orig["morph_set_idx"].tolist()[0]))
        assert False, "fix this."
        
    ### 
    fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "trial_morph_assigned_to_which_base", "pa_class", ["morph_set_idx"])
    savefig(fig, f"{savedir}/counts_idx_assigned-1.pdf")
    
    fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "trial_morph_assigned_to_which_base", "pa_class", ["morph_set_idx"])
    savefig(fig, f"{savedir}/counts_idx_assigned-2.pdf")


    for var_score in ["score"]:
        # var_score = "score_norm"

        for x_var in ["decoder_class_semantic_good", "decoder_class", "recoded_decoder"]:

            if x_var == "decoder_class_semantic_good":
                x_order = ("base1", "interm1", "same", "interm2", "base2")
            else:
                x_order = None

            fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue="pa_class", kind="point", errorbar=("ci", 68), 
                            order=x_order)
            for ax in fig.axes.flatten():
                ax.axhline(0, color="k", alpha=0.3)
            savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-1.pdf")

            # for col in [None, "pa_morph_assigned_baseorambig", "trial_morph_assigned_to_which_base", "recoded_trial_morph"]:
            for col in [None, "trial_morph_assigned_to_which_base", "recoded_trial_morph"]:
                fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue="pa_class", kind="point", errorbar=("ci", 68), 
                                col=col, order=x_order)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.3)
                savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-col={col}-2.pdf")

                fig = sns.catplot(data=dfscores, x=x_var, y=var_score, kind="bar", errorbar=("ci", 68), 
                                col=col, order=x_order)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.3)
                savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-col={col}-3.pdf")

                fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue = "morph_set_idx", kind="point", errorbar=("ci", 68), 
                                col=col, order=x_order)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.3)
                savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-col={col}-4.pdf")

                fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue = "morph_set_idx|idx_within", kind="point", errorbar=("ci", 68), 
                                col=col, order=x_order)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.3)
                savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-col={col}-4b.pdf")

                fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue=col, kind="point", 
                                errorbar=("ci", 68), col="pa_class", order=x_order)
                for ax in fig.axes.flatten():
                    ax.axhline(0, color="k", alpha=0.3)
                savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-hue={col}-5.pdf")

                if col is not None:
                    fig = sns.catplot(data=dfscores, x=col, y=var_score, hue = "morph_set_idx", kind="point", errorbar=("ci", 68), 
                                    col=x_var)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.3)
                    savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-col={col}-6.pdf")

            if False: # not usefiule
                sns.catplot(data=dfscores, x="pa_class", y=var_score, hue="decoder_class", kind="point", errorbar=("ci", 68))
                sns.catplot(data=dfscores, x="pa_class", y=var_score, hue="decoder_class", kind="point", errorbar=("ci", 68), col="pa_morph_assigned_baseorambig")

            from pythonlib.tools.pandastools import plot_subplots_heatmap

            for var_subplot in [None, "morph_set_idx"]:
                for row_var in ["idx_within|assigned", "trial_morph_assigned_to_which_base", "recoded_trial_morph"]:
                    for col_var in [x_var]:
                        if row_var == "trial_morph_assigned_to_which_base":
                            row_values = ['base1', 'not_ambig_base1', 'ambig_base1', 'ambig_base2', 'not_ambig_base2', 'base2']
                        else:
                            row_values = sorted(dfscores[row_var].unique().tolist())
                        if x_order is None:
                            x_order = sorted(dfscores[col_var].unique().tolist())

                        # zlims = [0,1]
                        zlims = None
                        fig, axes = plot_subplots_heatmap(dfscores, row_var, col_var, var_score, var_subplot, 
                                                        share_zlim=True, row_values=row_values,
                                            col_values=x_order, ZLIMS=zlims)
                        savefig(fig, f"{savedir}/heatmap-varsubplot={var_subplot}-rowvar={row_var}-colvar={col_var}-varscore={var_score}.pdf")

                        plt.close("all")
    

def analy_psychoprim_score_postsamp(DFallpa, DSmorphsets, 
                                    map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                    map_tcmorphset_to_info,
                                    SAVEDIR_BASE, animal, date,
                                    list_bregion=None,
                                    version=1):
    """
    Wrapper to run main scoring and plots for psychoprim.
    One decoder for each morphset.
    """

    LIST_TWIND_TEST = [
        (0.6, 1.2), # second half better for PMv?
        (0.05, 1.2),
        (0.05, 0.6),
    ]

    TWIND_TRAIN = (0.05, 1.2)

    subtract_baseline=False
    subtract_baseline_twind=None
    include_null_data = False
    do_upsample_balance=True

    # Plots for each specific morphset.
    PLOT_DECODER = False
    PLOT_EACH_MORPHSET = False
    PLOT_TEST_SPLIT = PLOT_DECODER
    PLOT_TEST_CONCATTED = PLOT_DECODER

    if list_bregion is None:
        list_bregion = DFallpa["bregion"].unique().tolist() 

    # Which version of train-test split?
    if version==0:
        # Previous version (ok)
        # Decoders (0, 1, ..., 99)
        # Test data: (0, 1, ..., 99), all using train-test split
        USE_TRAIN_TEST_SPLIT = True # This leads to more accurate esimate for same-condition.
        prune_labels_exist_in_train_and_test = True # important, if want to have mismatch between train adn test labels
        train_on_which_prims = "all" # just the 2 base prims
        score_user_test_data = False
        do_agg_of_user_test_data = False
        list_downsample_trials = [False, True] # important to balance across base and morphsets.
        split_by_gridloc = True
        test_on_which_prims = "not_trained"
        assert False, "split_by_gridloc will not work -- as this applies only to user test data. Solution: dont need to do train-test split."
    elif version == 1:
        # Good version.
        # Decoders: (0, 99)
        # Test data 1: (0, 99) using train-test split
        # Test data 2: (1, ..., ) morphsetes.
        # (Then concat test data 1 and 2)
        # And train-test are different gridloc.
        USE_TRAIN_TEST_SPLIT = False # This leads to more accurate esimate for same-condition.
        prune_labels_exist_in_train_and_test = False # important, if want to have mismatch between train adn test labels
        train_on_which_prims = "base" # just the 2 base prims
        list_downsample_trials = [False] # no need to balance, as this is comparing two base prims and testing generalization to morphed.
        split_by_gridloc = True
        test_on_which_prims = "all" # this is fine, since here is doing generalization across locations.
    elif version == 2:
        # Like version 1, but dont split by gridloc. Therefore you need to do train-test split for base prims.
        # Decoders: (0, 99)
        # Test data 1: (0, 99) using train-test split
        # Test data 2: (1, ..., ) morphsetes.
        # (Then concat test data 1 and 2)
        USE_TRAIN_TEST_SPLIT = True # This leads to more accurate esimate for same-condition.
        prune_labels_exist_in_train_and_test = False # important, if want to have mismatch between train adn test labels
        train_on_which_prims = "base" # just the 2 base prims
        score_user_test_data = True
        do_agg_of_user_test_data = True
        list_downsample_trials = [False] # no need to balance, as this is comparing two base prims and testing generalization to morphed.
        split_by_gridloc = False
        test_on_which_prims = "not_trained"
    else:
        assert False

    assert split_by_gridloc == True, "currently this is hardcoded, in that there are for loops for gridlocs... shold make that an option instead."

    list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())
    for TWIND_TEST in LIST_TWIND_TEST:
        for downsample_trials in list_downsample_trials:
            if downsample_trials:
                # So that the lowest N doesnt pull all other categories down.
                n_min_per_var = 9
            else:
                n_min_per_var = 6

            if version==1:
                n_min_per_var = 5 # because its generalizing across loc.
                
            SAVEDIR = f"{SAVEDIR_BASE}/downsample_trials={downsample_trials}-TWIND_TEST={TWIND_TEST}-version={version}"
            
            for bregion in list_bregion:



                ### Collect data across all morphsets.
                list_df = []
                list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())
                for morphset in list_morphset:
                    # Given morphset, assign new column which is the trial's role in that morphset.

                    for pa in DFallpa["pa"].values:
                        dflab = pa.Xlabels["trials"]
                        dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

                    # Train on all the base_prims data
                    idx_exist = sorted(list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"])))

                    event_train = "03_samp"
                    twind_train = TWIND_TRAIN
                    var_train = "idx_morph_temp"
                    if train_on_which_prims == "all": # all within morpjset
                        idx_train = idx_exist
                    elif train_on_which_prims == "base": # just the 2 base prims
                        idx_train = [0, 99]
                    else:
                        print(train_on_which_prims)
                        assert False

                    # Test on morphed data - get all here
                    # - Test on indices that are not included in training
                    var_test = "idx_morph_temp"
                    event_test = "03_samp"
                    list_twind_test = [TWIND_TEST]
                    which_level_test = "trial"

                    # Test on which prims? This is "user" inputed test data
                    if test_on_which_prims == "all":
                        idx_test = idx_exist
                    elif test_on_which_prims == "not_trained":
                        idx_test = [i for i in idx_exist if i not in idx_train]
                    else:
                        assert False

                    list_loc = dflab["seqc_0_loc"].unique().tolist()
                    list_dfscores_across_locs = []

                    savedir = f"{SAVEDIR}/{bregion}/morphset={morphset}"
                    os.makedirs(savedir, exist_ok=True)
                    print(savedir)

                    filterdict_train = {"idx_morph_temp":idx_train}
                    filterdict_test = {"idx_morph_temp":idx_test}
                    auto_prune_locations=True

                    dfscores, decoders, list_pa_train, list_pa_test = pipeline_train_test_scalar_score_split_gridloc(list_loc, savedir,
                                                                                                                    DFallpa, 
                                                                                    bregion, var_train, event_train, 
                                                                                    twind_train, filterdict_train,
                                                        var_test, event_test, list_twind_test, filterdict_test, 
                                                        include_null_data=include_null_data, 
                                                        prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, 
                                                        PLOT=PLOT_DECODER, PLOT_TEST=PLOT_TEST_CONCATTED,
                                                        which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                        subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                        do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials,
                                                        auto_prune_locations=auto_prune_locations, do_train_splits_nsplits=USE_TRAIN_TEST_SPLIT)
                    Dc = decoders[0]

                    dfscores["run_morph_set_idx"] = morphset
                    dfscores = analy_psychoprim_dfscores_condition(dfscores, morphset, DSmorphsets, map_morphsetidx_to_assignedbase_or_ambig, map_tcmorphset_to_info)

                    fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "pa_class", "decoder_class", ["run_morph_set_idx", "twind"]);
                    savefig(fig, f"{savedir}/counts_trials-dfscores.pdf")

                    # # TODO replace with pipeline_train_test_scalar_score_split_gridloc 
                    # for train_loc in list_loc:
                    #     for test_loc in list_loc:
                    #         if train_loc != test_loc:

                    #             filterdict_train = {"idx_morph_temp":idx_train, "seqc_0_loc":[train_loc]}
                    #             filterdict_test = {"idx_morph_temp":idx_test, "seqc_0_loc":[test_loc]}
                    #             print("filterdict_train:", filterdict_train)
                    #             print("filterdict_test:", filterdict_test)
                                
                    #             # Other params
                    #             savedir = f"{SAVEDIR}/{bregion}/morphset={morphset}/decoder_training-train_loc={train_loc}-test_loc={test_loc}"
                    #             os.makedirs(savedir, exist_ok=True)
                    #             print(savedir)
                                
                    #             if USE_TRAIN_TEST_SPLIT:
                    #                 do_train_splits_nsplits=10
                    #                 dfscores_testsplit, dfscores_usertest, dfscores_both, decoders, trainsets, PAtest = pipeline_train_test_scalar_score_with_splits(DFallpa, 
                    #                                                                                 bregion, var_train, event_train, 
                    #                                                                                 twind_train, filterdict_train,
                    #                                                     var_test, event_test, list_twind_test, filterdict_test, savedir,
                    #                                                     include_null_data=include_null_data, decoder_method_index=None,
                    #                                                     prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, 
                    #                                                     PLOT_TRAIN=PLOT_DECODER, PLOT_TEST_SPLIT=PLOT_TEST_SPLIT, PLOT_TEST_CONCATTED=PLOT_TEST_CONCATTED,
                    #                                                     which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                    #                                                     subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                    #                                                     do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials,
                    #                                                     do_train_splits_nsplits=do_train_splits_nsplits, 
                    #                                                     score_user_test_data=score_user_test_data, 
                    #                                                     do_agg_of_user_test_data=do_agg_of_user_test_data)

                    #                 if dfscores_testsplit is None:
                    #                     # Then no data after filtering.
                    #                     print("! skipping! dfscores_testsplit is None")
                    #                     continue

                    #                 dfscores_testsplit["run_morph_set_idx"] = morphset
                    #                 dfscores_testsplit = analy_psychoprim_dfscores_condition(dfscores_testsplit, morphset, DSmorphsets, map_morphsetidx_to_assignedbase_or_ambig, map_tcmorphset_to_info)
                    #                 fig = grouping_plot_n_samples_conjunction_heatmap(dfscores_testsplit, "pa_class", "decoder_class", ["run_morph_set_idx", "twind"]);
                    #                 savefig(fig, f"{savedir}/counts_trials-dfscores_traintestsplit.pdf")

                    #                 # Which dfscores is the final?
                    #                 if version == 0:
                    #                     # Use dfscores_testsplit.
                    #                     dfscores = dfscores_testsplit
                    #                 elif version == 2:
                    #                     dfscores_usertest["run_morph_set_idx"] = morphset
                    #                     dfscores_both["run_morph_set_idx"] = morphset

                    #                     dfscores_usertest = analy_psychoprim_dfscores_condition(dfscores_usertest, morphset, DSmorphsets, map_morphsetidx_to_assignedbase_or_ambig, map_tcmorphset_to_info)
                    #                     dfscores_both = analy_psychoprim_dfscores_condition(dfscores_both, morphset, DSmorphsets, map_morphsetidx_to_assignedbase_or_ambig, map_tcmorphset_to_info)

                    #                     fig = grouping_plot_n_samples_conjunction_heatmap(dfscores_usertest, "pa_class", "decoder_class", ["run_morph_set_idx", "twind"]);
                    #                     savefig(fig, f"{savedir}/counts_trials-dfscores_usertest.pdf")
                    #                     fig = grouping_plot_n_samples_conjunction_heatmap(dfscores_both, "pa_class", "decoder_class", ["run_morph_set_idx", "twind"]);
                    #                     savefig(fig, f"{savedir}/counts_trials-dfscores_both.pdf")
                    #                     plt.close("all")

                    #                     # Finally, keep just the combined
                    #                     dfscores = dfscores_both
                    #             else:
                    #                 dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, 
                    #                                                                                 twind_train, filterdict_train,
                    #                                                     var_test, event_test, list_twind_test, filterdict_test, savedir,
                    #                                                     include_null_data=include_null_data, decoder_method_index=None,
                    #                                                     prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, PLOT=PLOT_DECODER,
                    #                                                     which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                    #                                                     subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                    #                                                     do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials)
                                    
                    #                 dfscores["run_morph_set_idx"] = morphset
                    #                 dfscores = analy_psychoprim_dfscores_condition(dfscores, morphset, DSmorphsets, map_morphsetidx_to_assignedbase_or_ambig, map_tcmorphset_to_info)

                    #                 fig = grouping_plot_n_samples_conjunction_heatmap(dfscores, "pa_class", "decoder_class", ["run_morph_set_idx", "twind"]);
                    #                 savefig(fig, f"{savedir}/counts_trials-dfscores.pdf")

                    #             # Store dfscores across locations sets
                    #             dfscores["train_loc"] = [train_loc for _ in range(len(dfscores))]
                    #             dfscores["test_loc"] = [test_loc for _ in range(len(dfscores))]
                    #             list_dfscores_across_locs.append(dfscores)

                    # # Concat across location sets, so that final dataframe has one datapt per trialcode.
                    # if len(list_dfscores_across_locs)==0:
                    #     print("! skipping! len(list_dfscores_across_locs)==0")
                    #     continue
                    
                    # from pythonlib.tools.pandastools import aggregGeneral
                    # dfscores_tmp = pd.concat(list_dfscores_across_locs).reset_index(drop=True)
                    # dfscores = aggregGeneral(dfscores_tmp, 
                    #                                     ["decoder_class", "twind", "trialcode"], 
                    #                                     ["score", "score_norm"], 
                    #                                     ["pa_class", "epoch", "same_class"] # code does sanity check that these are all unqiue values.
                    #                                     )

                    ##### PLOTS
                    if PLOT_EACH_MORPHSET:
                        savedir = f"{SAVEDIR}/{bregion}/morphset={morphset}/plots"

                        os.makedirs(savedir, exist_ok=True)
                        print("Saving plots at... ", savedir)
                        _analy_psychoprim_score_postsamp_plot_scores(dfscores, savedir)

                    ### Collect
                    list_df.append(dfscores)

                ### Plot summary
                DFSCORES_ALL = pd.concat(list_df).reset_index(drop=True)



                # Save data
                import pickle
                savedir = f"{SAVEDIR}/{bregion}"
                with open(f"{savedir}/DFSCORES.pkl", "wb") as f:
                    pickle.dump(DFSCORES_ALL, f)

                ############## PRUNE MORPHSETS
                for morphset_get in [None]:
                # for morphset_get in [None, "good_ones"]:



                    # print("morphset_get:", morphset_get)
                    # if morphset_get == "good_ones":
                    #     # Hand modified
                    #     if (animal, date) == ("Diego", 240515):
                    #         # Angle rotation
                    #         morphsets_ignore = [2] # Did most with two strokes...
                    #     if (animal, date) == ("Diego", 240523):
                    #         # THis is a garbage morphset, is not actually morphing.
                    #         morphsets_ignore = [0]
                    #     elif (animal, date) == ("Pancho", 240521):
                    #         morphsets_ignore = [0] # one base prim is garbage
                    #     elif (animal, date) == ("Pancho", 240524):
                    #         morphsets_ignore = [4] # doesnt actually vaciallte across tirals
                    #     else:
                    #         morphsets_ignore = []
                    # elif morphset_get is None:
                    #     # Get all morphsets
                    #     morphsets_ignore = []
                    # elif isinstance(morphset_get, int):
                    #     # get just this one morphset (exclude the others)
                    #     morphsets_ignore = [ms for ms in DF_TCRES["run_morphset"].unique().tolist() if ms!=morphset_get]
                    # else:
                    #     print(morphset_get)
                    #     assert False
                    #     # morphsets_ignore = []
                    # morphsets_keep = [ms for ms in DFSCORES_ALL["run_morph_set_idx"].unique().tolist() if ms not in morphsets_ignore]
                    # print("morphsets_ignore: ", morphsets_ignore)
                    # DFSCORES = DFSCORES_ALL[DFSCORES_ALL["run_morph_set_idx"].isin(morphsets_keep)].reset_index(drop=True)

                    DFSCORES = DFSCORES_ALL

                    

                    ################ Plot
                    savedir = f"{SAVEDIR}/{bregion}/combine_morphsets-all_scalar_summary-morphset_get={morphset_get}"
                    os.makedirs(savedir, exist_ok=True)
                    print("Saving plots at... ", savedir)

                    _analy_psychoprim_score_postsamp_plot_scores(DFSCORES, savedir)

                    ######## PRUNE TO BALANCE DATASET< then plot. E.g.,, want to keep only those (morphset, idx) which
                    # have data across all decoders (same, intermediate, diff). Otehrwise acorss morphsets/idcs there is
                    # big difference.
                    n_min = 1
                    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars

                    ### Q: is the intermediate lower than same and base?
                    # levels_get = ["base1", "interm1", "same"]
                    # levels_get = ["base2", "interm2", "same"]
                    # levels_get = ["base2", "interm2", "same"]
                    levels_get = ["same", "intermthis", "basethis"]
                    savedir = f"{SAVEDIR}/{bregion}/combine_morphsets-questions_interm-morphset_get={morphset_get}"
                    os.makedirs(savedir, exist_ok=True)

                    lenient_allow_data_if_has_n_levels = len(levels_get)
                    dfscores_this, dict_dfthis = extract_with_levels_of_conjunction_vars(DFSCORES, "recoded_decoder", 
                                                                                ["morph_set_idx|idx_within"], levels_get, n_min,  
                                                                                lenient_allow_data_if_has_n_levels=lenient_allow_data_if_has_n_levels,
                                                                                prune_levels_with_low_n=True, 
                                                                                plot_counts_heatmap_savepath=f"{savedir}/counts_extract.png")
                    if len(dfscores_this)>0:
                        _analy_psychoprim_score_postsamp_plot_scores(dfscores_this, savedir)


                    ### Q: match the morphset to ask whether ambig is diff from not-ambig
                    levels_get = ["not_ambig_base1", "ambig_base1"]
                    savedir = f"{SAVEDIR}/{bregion}/combine_morphsets-questions_ambig_vs_notambig-morphset_get={morphset_get}"
                    os.makedirs(savedir, exist_ok=True)
                    dfscores_this, dict_dfthis = extract_with_levels_of_conjunction_vars(DFSCORES, "trial_morph_assigned_to_which_base", 
                                                                                ["morph_set_idx", "decoder_class_semantic_good"], levels_get, n_min,  
                                                                                lenient_allow_data_if_has_n_levels=len(levels_get), 
                                                                                prune_levels_with_low_n=True, 
                                                                                plot_counts_heatmap_savepath=f"{savedir}/counts_extract.png")
                    if len(dfscores_this)>0:
                        _analy_psychoprim_score_postsamp_plot_scores(dfscores_this, savedir)


                    # Trial by trial variation in what drawn, given same stimulus
                    ##### Prep dataset
                    savedir = f"{SAVEDIR}/{bregion}/combine_morphsets-ambig_trials-morphset_get={morphset_get}"
                    os.makedirs(savedir, exist_ok=True)
                    print("Saving plots at... ", savedir)

                    # TODO Agg -- for each decoder, get diff between trials where draw that vs. where not.

                    from pythonlib.tools.pandastools import aggregGeneral

                    # task_vars = ["character"]
                    task_vars = ["morph_set_idx", "pa_class"]

                    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
                    if True:
                        # Version 1 - pull out each task ("morph_set_idx", "pa_class") that is ambiguous, along with all decoder classes
                        dfscores_ambig, dict_df = extract_with_levels_of_conjunction_vars_helper(DFSCORES, "trial_morph_assigned_to_which_base", 
                                                                                                task_vars, 1, None, 2)
                        dfscores_ambig_agg = aggregGeneral(dfscores_ambig, task_vars + ["decoder_class", "decoder_class_was_drawn", "trial_morph_assigned_to_which_base"], 
                                                                ["score", "score_norm"], nonnumercols=["decoder_class_semantic_good"])
                    else:
                        # Version 2 - pull out each task ("morph_set_idx", "pa_class") that is ambiguous, and also just the datapts for the base1 and base2 decoders
                        # i.e, this is subset of above.
                        dfscores_ambig, dict_df = extract_with_levels_of_conjunction_vars_helper(DFSCORES, "decoder_class_was_drawn", 
                                                                                            task_vars + ["decoder_class"], 1, None, 2)

                        # Aggregate, so each task contributes exactly 1 datapt to (False, True) for 
                        # decoder_class_was_drawn
                        dfscores_ambig_agg = aggregGeneral(dfscores_ambig, task_vars + ["decoder_class", "decoder_class_was_drawn", "trial_morph_assigned_to_which_base"], 
                                                                ["score", "score_norm"], nonnumercols=["decoder_class_semantic_good"])
                        
                    # USeful, for printing conjunctions
                    if False:
                        grouping_print_n_samples(dfscores_ambig, task_vars + ["decoder_class", "trial_morph_assigned_to_which_base", "decoder_class_was_drawn"])

                    ##### Plots
                    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping
                    for dfthis, dfthis_str in [
                        (dfscores_ambig, "dfscores_ambig"), 
                        (dfscores_ambig_agg, "dfscores_ambig_agg")]:

                        for var_score in ["score", "score_norm"]:

                            # Catplot
                            fig = sns.catplot(data=dfthis, x="trial_morph_assigned_to_which_base", y=var_score, hue="decoder_class_semantic_good", kind="point")
                            for ax in fig.axes.flatten():
                                ax.axhline(0, color="k", alpha=0.3)
                            savefig(fig, f"{savedir}/catplot-dfthis_str={dfthis_str}-var_score={var_score}-1.pdf")

                            fig = sns.catplot(data=dfthis, x="decoder_class_semantic_good", y=var_score, hue="trial_morph_assigned_to_which_base", kind="point")
                            for ax in fig.axes.flatten():
                                ax.axhline(0, color="k", alpha=0.3)
                            savefig(fig, f"{savedir}/catplot-dfthis_str={dfthis_str}-var_score={var_score}-2.pdf")

                            fig = sns.catplot(data=dfthis, x="trial_morph_assigned_to_which_base", y=var_score, hue="decoder_class_semantic_good", kind="point",
                                            col="morph_set_idx")
                            for ax in fig.axes.flatten():
                                ax.axhline(0, color="k", alpha=0.3)
                            savefig(fig, f"{savedir}/catplot-dfthis_str={dfthis_str}-var_score={var_score}-morph_set_idx-1.pdf")

                            fig = sns.catplot(data=dfthis, x="decoder_class_semantic_good", y=var_score, hue="trial_morph_assigned_to_which_base", kind="point",
                                            col="morph_set_idx")
                            for ax in fig.axes.flatten():
                                ax.axhline(0, color="k", alpha=0.3)
                            savefig(fig, f"{savedir}/catplot-dfthis_str={dfthis_str}-var_score={var_score}-morph_set_idx-2.pdf")

                            # 45-scatter
                            _, fig = plot_45scatter_means_flexible_grouping(dfthis, "decoder_class_semantic_good", "base1", "base2",
                                                                "trial_morph_assigned_to_which_base", var_score, task_vars,
                                                                plot_text=False, shareaxes=True, SIZE=6, alpha=0.3);
                            savefig(fig, f"{savedir}/scatter-x=decoder_class_semantic_good-y=trial_morph_assigned_to_which_base-dfthis_str={dfthis_str}-var_score={var_score}.pdf")

                            _, fig = plot_45scatter_means_flexible_grouping(dfthis, "trial_morph_assigned_to_which_base", "ambig_base1", "ambig_base2",
                                                                "decoder_class_semantic_good", var_score, task_vars,
                                                                plot_text=False, shareaxes=True, SIZE=6, alpha=0.3);
                            savefig(fig, f"{savedir}/scatter-x=trial_morph_assigned_to_which_base-y=decoder_class_semantic_good-dfthis_str={dfthis_str}-var_score={var_score}.pdf")

                            plt.close("all")


def analy_psychoprim_score_postsamp_better(DFallpa, DSmorphsets, 
                                    map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                    map_tcmorphset_to_info,
                                    SAVEDIR_BASE, 
                                    animal, date,
                                    PLOT_EACH_IDX=True,
                                    ):
    """
    newer, which does two thigns:
    (1) takes each set of (base prim, morph prim, base prim 2) and trains and tests. ie does this instead 
    of each morph set as in previous vresion. This cleans things up...
    (2) Uses train/test split, so can ask if base prim decodes better than morph

    TODO: Morph this and old coade by adding to here: scoring (PAtest) also for all of the idxs in the morphset, 
    instead of just (base1, morph, base2) as in here (just 3 indices).
    """

    list_morphset = DSmorphsets.Dat["morph_set_idx"].unique()

    include_null_data = False
    only_train_on_base_prims=True
    TWIND_TRAIN = (0.05, 1.5)
    TWIND_TEST = (0.05, 1.5)
    downsample_trials = True
    do_train_splits_nsplits=10
    prune_labels_exist_in_train_and_test = False
    score_user_test_data = False
    PLOT_DECODER=True
    n_min_per_var = 7
    subtract_baseline = False
    subtract_baseline_twind = None
    do_upsample_balance = True

    HACK = False

    LIST_NORM_METHOD = ["none", "minus_base_twind", "minus_not_visible_and_base"]

    # PLOT_EACH_IDX = True

    if PLOT_EACH_IDX==False:
        PLOT_DECODER = False
    PLOT_TEST_SPLIT=PLOT_DECODER

    # PARAMS = {}
    list_bregion = sorted(DFallpa["bregion"].unique().tolist())
    for bregion in list_bregion:

        SAVEDIR = f"{SAVEDIR_BASE}/{bregion}"

        LIST_DFSCORES = []
        LIST_TC_RES = []
        for morphset in list_morphset:
            MORPHSET = morphset

            if HACK:
                morphset = 3

            for pa in DFallpa["pa"].values:
                dflab = pa.Xlabels["trials"]
                dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

            # Train on all the base_prims data
            idx_exist = sorted(list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"])))
            idx_exist = [i for i in idx_exist if i not in [0, 99]] # Exclude conditioning on base prims... 
            for idx_within in idx_exist:
                IDX_WITHIN = idx_within
                if HACK:
                    idx_within = 3

                if only_train_on_base_prims:
                    idx_exist_train = [0, idx_within, 99]
                else:
                    idx_exist_train = idx_exist

                event_train = "03_samp"
                twind_train = TWIND_TRAIN
                filterdict_train = {"idx_morph_temp":idx_exist_train}
                var_train = "idx_morph_temp"

                # Test on morphed data - get all here
                var_test = "idx_morph_temp"
                event_test = "03_samp"
                filterdict_test = {"idx_morph_temp":idx_exist}
                list_twind_test = [TWIND_TEST]
                which_level_test = "trial"

                # Other params
                savedir = f"{SAVEDIR}/run_morphset={morphset}-run_idx_within={idx_within}"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)

                dfscores_testsplit, dfscores_usertest, dfscores_both, decoders, trainsets, PAtest = pipeline_train_test_scalar_score_with_splits(DFallpa, bregion, var_train, event_train, 
                                                                                twind_train, filterdict_train,
                                                    var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                    include_null_data=include_null_data, decoder_method_index=None,
                                                    prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, PLOT_TRAIN=PLOT_DECODER, PLOT_TEST_SPLIT=PLOT_TEST_SPLIT,
                                                    which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                    subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                    do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials,
                                                    do_train_splits_nsplits=do_train_splits_nsplits, 
                                                    score_user_test_data=score_user_test_data)
                Dc = decoders[0]
                dfscores_testsplit["run_morph_set_idx"] = morphset
                dfscores_testsplit["run_idx_within"] = idx_within
                # dfscores_testsplit["run_idx_exist_train"] = [tuple(idx_exist_train) 

                from neuralmonkey.analyses.decode_moment import analy_psychoprim_dfscores_condition
                dfscores_testsplit = analy_psychoprim_dfscores_condition(dfscores_testsplit, morphset, DSmorphsets, 
                                                        map_morphsetidx_to_assignedbase_or_ambig,
                                                        map_tcmorphset_to_info)
                
                if score_user_test_data:
                    dfscores_usertest["morph_set_idx"] = morphset

                LIST_DFSCORES.append(dfscores_testsplit)

                if HACK:
                    assert False

                ######## PLOT TIMECOURSES FOR THIS (MORPHSET, IDX_WITHIN)
                # Plot all trials for a single morph trial (specific morphset, idxwithin)
                # Get just trials where he drew ambig_base1
                twind_time = (-0.5, 1.8)

                from neuralmonkey.analyses.decode_moment import trainsplit_timeseries_score_wrapper
                PAprobs, labels, times = trainsplit_timeseries_score_wrapper(decoders, trainsets, twind_time)
                dflab = PAprobs.Xlabels["trials"]
                dflab["_assigned"] = [map_tcmorphset_to_info[(tc, morphset)][0] for tc in dflab["trialcode"]]

                # Normalize time-series
                from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import probs_timecourse_normalize
                # -- for each decode class, input which indtrials are its "baseline"
                # for each decoder class, its baseline trials are when the image is the other base prim
                map_decoder_class_to_baseline_trials = {}
                for decoder_class in labels:
                    if decoder_class==0:
                        inds = dflab[dflab["idx_morph_temp"] == 99].index.tolist()
                    elif decoder_class==99:
                        inds = dflab[dflab["idx_morph_temp"] == 0].index.tolist()
                    else:
                        inds = dflab[dflab["idx_morph_temp"] != decoder_class].index.tolist()
                    map_decoder_class_to_baseline_trials[decoder_class] = inds

                # probs_timecourse_normalize(None, None)
                for norm_method in LIST_NORM_METHOD:
                    if norm_method == "none":
                        PAprobsNorm = PAprobs
                    else:
                        PAprobsNorm = probs_timecourse_normalize(None, PAprobs, norm_method, None, 
                                                                None, map_decoder_class_to_baseline_trials=map_decoder_class_to_baseline_trials)

                    # probs_mat_all = PAprobs.X
                    probs_mat_all = PAprobsNorm.X
                    dflab = PAprobsNorm.Xlabels["trials"]

                    # Get these trials
                    if PLOT_EACH_IDX:
                        fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)

                    # Collect data
                    ct = 0
                    for idx_within in dflab["idx_morph_temp"].unique():
                        for _assigned in dflab["_assigned"].unique():
                            indtrials = dflab[
                                (dflab["idx_morph_temp"] == idx_within) & (dflab["_assigned"] == _assigned)
                                ].index.tolist()
                            print("--- For ", _assigned, ", idx_within=, ", idx_within, " plotting these trials:", indtrials)

                            if len(indtrials)>0:
                                MAP_INDEX_TO_COL=None
                        
                                probs_mat_all_this = probs_mat_all[:, indtrials, :]
                                if PLOT_EACH_IDX:
                                    ax = axes.flatten()[ct]
                                    ct+=1

                                    Dc._timeseries_plot_by_shape_drawn_order(probs_mat_all_this, times, labels, MAP_INDEX_TO_COL, ax)
                                    # ax.set_title(f"{idx_within}, {assigned}")
                                    ax.set_title(f"{idx_within}, {_assigned}")

                                LIST_TC_RES.append({
                                    "run_morphset":MORPHSET,
                                    "run_idx_within":IDX_WITHIN,
                                    # "run_idx_exist_train":tuple(idx_exist_train),
                                    "dat_trials_idx_within":idx_within,
                                    "dat_trials_assigned":_assigned,
                                    "norm_method":norm_method,
                                    "indtrials":indtrials,
                                    "probs_mat_all_this":probs_mat_all_this,
                                    "times":times, 
                                    "labels":labels
                                })
                    if PLOT_EACH_IDX:
                        savefig(fig, f"{savedir}/timecourse_testtrials-norm={norm_method}.pdf")
                    
                    plt.close("all")
                        

        ####################### COMBINE ACROSS MORPHSET AND MORPHINDEX
        # Save data
        import pickle

        with open(f"{SAVEDIR}/LIST_TC_RES.pkl", "wb") as f:
            pickle.dump(LIST_TC_RES, f)

        with open(f"{SAVEDIR}/LIST_DFSCORES.pkl", "wb") as f:
            pickle.dump(LIST_DFSCORES, f)

        # Timecourse plots

        # Three sets of plots:
        # 1. ambig
        # 2. notambig - base1
        # 3. notambig - base2

        # for each (morphset, idx), get mean over trials
        # -- (4 trial conditions)
        # -- (3 decoder classes)
        # --> 12 mean traces.

        # save as (labels, 12conditions, times)
        DF_TCRES = pd.DataFrame(LIST_TC_RES)
        import numpy as np
        Dc = decoders[0]

        for norm_method in LIST_NORM_METHOD:
            savedir = f"{SAVEDIR}/SUMMARY-norm={norm_method}"
            os.makedirs(savedir, exist_ok=True)
            for trial_ver, do_recode in [
                ("ambig", False), 
                ("ambig", True), 
                ("notambig1", False), 
                ("notambig2", False)
                ]:
                
                if trial_ver == "ambig":
                    dat_trials_assigned_get = ["base1", "ambig_base1", "ambig_base2", "base2"]
                elif trial_ver == "notambig1":
                    dat_trials_assigned_get = ["base1", "not_ambig_base1", "base2"]
                elif trial_ver == "notambig2":
                    dat_trials_assigned_get = ["base1", "not_ambig_base2", "base2"]
                else:
                    print(trial_ver)
                    assert False

                from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
                dfout, grpdict = extract_with_levels_of_conjunction_vars(DF_TCRES, "dat_trials_assigned", ["run_morphset", "run_idx_within"],
                                                        dat_trials_assigned_get,
                                                        n_min_across_all_levs_var=1, lenient_allow_data_if_has_n_levels=len(dat_trials_assigned_get), 
                                                        plot_counts_heatmap_savepath="/tmp/counts.pdf")

                for morphset_get in ["good_ones", None] + DF_TCRES["run_morphset"].unique().tolist():
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
                        # morphsets_ignore = []

                    print("morphsets_ignore: ", morphsets_ignore)

                    ### COLLECT DATA across all morphsets.
                    list_probs_mat_all_this = []
                    labels_all_means = []
                    for morphset, idx_within in grpdict.keys():

                        # if morphset_get is None:
                        #     # Then get all morphsetes
                        #     pass
                        # else:
                        #     if not morphset == morphset_get:
                        #         continue
                        
                        if morphset in morphsets_ignore:
                            print("... skipping this morphset: ", morphset)
                            continue
                        else:
                            print("... not skipping this morphset: ", morphset)

                        for dat_trials_assigned in dat_trials_assigned_get:
                            a = DF_TCRES["run_morphset"]==morphset
                            b = DF_TCRES["run_idx_within"]==idx_within
                            c = DF_TCRES["dat_trials_assigned"]==dat_trials_assigned
                            d = DF_TCRES["norm_method"]==norm_method
                            dfthis = DF_TCRES[(a & b & c & d)]

                            if len(dfthis)==0:
                                print("... skipping this morphset: ", morphset , " (empty)")

                                print(morphset, idx_within, dat_trials_assigned, norm_method)
                                print(a,b,c,d)
                                print(DF_TCRES["run_morphset"].unique())
                                print(DF_TCRES["run_idx_within"].unique())
                                print(DF_TCRES["dat_trials_assigned"].unique())
                                print(DF_TCRES["norm_method"].unique())
                                assert False, "this shouldnt be possible?? possibly variable changed somewhere.."
                                # continue

                            assert len(dfthis)==1

                            probs_mat_all = dfthis["probs_mat_all_this"].values[0]
                            labels = dfthis["labels"].values[0]
                            times = dfthis["times"].values[0]

                            # Get the mean prob_vec
                            decoder_labels_get = [0, idx_within, 99]

                            probs_mat_this = np.zeros((len(decoder_labels_get), 1, len(times)))-1
                            for i_lab, lab in enumerate(decoder_labels_get):
                                idx_lab = labels.index(lab)
                                prob_vec = np.mean(probs_mat_all[idx_lab, :, :], axis=0) # (ntimes,)
                                probs_mat_this[i_lab, 0, :] = prob_vec
                            assert np.all(probs_mat_this>-1)

                            list_probs_mat_all_this.append(probs_mat_this)
                            labels_all_means.append(dat_trials_assigned)

                    if len(list_probs_mat_all_this)==0:
                        continue

                    probs_mat_all_means = np.concatenate(list_probs_mat_all_this, axis=1) # (nlabels, n_condition_means, ntimes)
                    print("shape of probs_mat_all_means:", probs_mat_all_means.shape)

                    # Recode ambig to (baseother, ambig, basechosen)
                    if do_recode:
                        labels_all_means_grouped = []
                        for i, lab in enumerate(labels_all_means):
                            if lab == "base1":
                                labels_all_means_grouped.append("base")
                            elif lab == "ambig_base1":
                                labels_all_means_grouped.append("ambig")
                            elif lab == "base2":
                                probs_mat_all_means[:, i, :] = probs_mat_all_means[::-1, i, :]
                                labels_all_means_grouped.append("base")
                            elif lab == "ambig_base2":
                                probs_mat_all_means[:, i, :] = probs_mat_all_means[::-1, i, :]
                                labels_all_means_grouped.append("ambig")
                            else:
                                assert False
                        labels_all_means  = labels_all_means_grouped
                        dat_trials_assigned_get_plot = ["base", "ambig"]
                        decoder_labels_plot = ["basethis", "morphed", "baseother"]
                    else:
                        dat_trials_assigned_get_plot = dat_trials_assigned_get
                        decoder_labels_plot = [0, "morphed", 99]
                    

                    fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)

                    for i_ax, triallab in enumerate(dat_trials_assigned_get_plot):
                        ax = axes.flatten()[i_ax]
                        ax.set_title(triallab)
                        ax.set_xlabel("time")
                        ax.set_title(triallab)

                        inds = [i for i, lab in enumerate(labels_all_means) if lab==triallab]

                        probs_mat_all_this = probs_mat_all_means[:, inds, :]
                        Dc._timeseries_plot_by_shape_drawn_order(probs_mat_all_this, times, decoder_labels_plot, None, ax)

                    savefig(fig, f"{savedir}/timecourse-trialver={trial_ver}-dorecode={do_recode}-norm={norm_method}-morphsetget={morphset_get}.pdf")            
                    plt.close("all")

                    print("Final")
                    print(f"{savedir}/timecourse-trialver={trial_ver}-dorecode={do_recode}-norm={norm_method}-morphsetget={morphset_get}.pdf")
                    print(grpdict.keys())
                    print(len(grpdict))
                    print(probs_mat_all_means.shape)
                    print(labels_all_means)
                    print(dat_trials_assigned_get)
                    print(dat_trials_assigned_get_plot)
                    # assert False

            ########################################
            ### Not-ambiguous --> (i) combine across true and false, and (ii) recode
            for do_recode in [False, True]:
                
                list_probs_mat_all_means = []
                list_labels_all_means = []
                # for trial_ver in ["notambig1", "notambig2", "notambig1_hasambig", "notambig2_hasambig"]:
                for trial_ver in ["notambig1", "notambig2"]:

                    if trial_ver == "ambig":
                        dat_trials_assigned_get = ["base1", "ambig_base1", "ambig_base2", "base2"]
                        dat_trials_assigned_get_filter = ["base1", "ambig_base1", "ambig_base2", "base2"]
                    elif trial_ver == "notambig1":
                        dat_trials_assigned_get = ["base1", "not_ambig_base1", "base2"]
                        dat_trials_assigned_get_filter = ["base1", "not_ambig_base1", "base2"]
                    elif trial_ver == "notambig2":
                        dat_trials_assigned_get = ["base1", "not_ambig_base2", "base2"]
                        dat_trials_assigned_get_filter = ["base1", "not_ambig_base2", "base2"]
                    elif trial_ver == "notambig1_hasambig":
                        dat_trials_assigned_get = ["base1", "not_ambig_base1", "base2"]
                        dat_trials_assigned_get_filter = ["base1", "not_ambig_base1", "ambig_base1", "ambig_base2", "base2"]
                    elif trial_ver == "notambig2_hasambig":
                        dat_trials_assigned_get = ["base1", "not_ambig_base2", "base2"]
                        dat_trials_assigned_get_filter = ["base1", "not_ambig_base2", "ambig_base1", "ambig_base2", "base2"]
                    else:
                        print(trial_ver)
                        assert False


                    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
                    dfout, grpdict = extract_with_levels_of_conjunction_vars(DF_TCRES, "dat_trials_assigned", ["run_morphset", "run_idx_within"],
                                                            dat_trials_assigned_get_filter,
                                                            n_min_across_all_levs_var=1, lenient_allow_data_if_has_n_levels=len(dat_trials_assigned_get_filter), 
                                                            plot_counts_heatmap_savepath="/tmp/counts.pdf")

                    probs_mat_all_means = []
                    labels_all_means = []
                    for morphset, idx_within in grpdict.keys():
                        for dat_trials_assigned in dat_trials_assigned_get:
                            a = DF_TCRES["run_morphset"]==morphset
                            b = DF_TCRES["run_idx_within"]==idx_within
                            c = DF_TCRES["dat_trials_assigned"]==dat_trials_assigned
                            d = DF_TCRES["norm_method"]==norm_method
                            dfthis = DF_TCRES[(a & b & c & d)]
                            assert len(dfthis)==1

                            probs_mat_all = dfthis["probs_mat_all_this"].values[0]
                            labels = dfthis["labels"].values[0]
                            times = dfthis["times"].values[0]

                            # Get the mean prob_vec
                            if trial_ver=="ambig":
                                decoder_labels_get = [0, idx_within, 99]
                            elif "notambig" in trial_ver:
                                decoder_labels_get = [0, idx_within, 99]
                            else:
                                assert False

                            probs_mat_this = np.zeros((len(decoder_labels_get), 1, len(times)))-1
                            for i_lab, lab in enumerate(decoder_labels_get):
                                idx_lab = labels.index(lab)
                                prob_vec = np.mean(probs_mat_all[idx_lab, :, :], axis=0) # (ntimes,)
                                probs_mat_this[i_lab, 0, :] = prob_vec
                            assert np.all(probs_mat_this>-1)

                            probs_mat_all_means.append(probs_mat_this)
                            labels_all_means.append(dat_trials_assigned)

                    probs_mat_all_means = np.concatenate(probs_mat_all_means, axis=1) # (nlabels, n_condition_means, ntimes)
                    decoder_labels_plot = [0, "morphed", 99]

                    list_probs_mat_all_means.append(probs_mat_all_means)
                    list_labels_all_means.append(labels_all_means)

                probs_mat_all_means = np.concatenate(list_probs_mat_all_means, axis=1)
                labels_all_means = []
                for tmp in list_labels_all_means:
                    labels_all_means.extend(tmp)
                dat_trials_assigned_get = ["base1", "not_ambig_base1", "not_ambig_base2", "base2"]

                # Recode ambig to (baseother, ambig, basechosen)
                if do_recode:
                    labels_all_means_grouped = []
                    for i, lab in enumerate(labels_all_means):
                        if lab == "base1":
                            labels_all_means_grouped.append("base")
                        elif lab == "ambig_base1":
                            labels_all_means_grouped.append("ambig")
                        elif lab == "not_ambig_base1":
                            labels_all_means_grouped.append("not_ambig")
                        elif lab == "base2":
                            probs_mat_all_means[:, i, :] = probs_mat_all_means[::-1, i, :]
                            labels_all_means_grouped.append("base")
                        elif lab == "ambig_base2":
                            probs_mat_all_means[:, i, :] = probs_mat_all_means[::-1, i, :]
                            labels_all_means_grouped.append("ambig")
                        elif lab == "not_ambig_base2":
                            probs_mat_all_means[:, i, :] = probs_mat_all_means[::-1, i, :]
                            labels_all_means_grouped.append("not_ambig")
                        else:
                            assert False

                    labels_all_means  = labels_all_means_grouped
                    dat_trials_assigned_get = ["base", "not_ambig"]
                    decoder_labels_plot = ["basethis", "morphed", "baseother"]

                fig, axes = plt.subplots(2, 3, figsize=(15,10), sharex=True, sharey=True)

                for i_ax, triallab in enumerate(dat_trials_assigned_get):
                    ax = axes.flatten()[i_ax]
                    ax.set_title(triallab)
                    ax.set_xlabel("time")
                    ax.set_title(triallab)

                    inds = [i for i, lab in enumerate(labels_all_means) if lab==triallab]

                    probs_mat_all_this = probs_mat_all_means[:, inds, :]
                    Dc._timeseries_plot_by_shape_drawn_order(probs_mat_all_this, times, decoder_labels_plot, None, ax)

                savefig(fig, f"{savedir}/timecourse-trialver=notambigcombined-dorecode={do_recode}-norm={norm_method}.pdf")
                plt.close("all")
