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

        X = pathis.X.squeeze().T # (ntrials, nchans)
        # times = pathis.Times
        dflab = pathis.Xlabels["trials"]
        labels = dflab[var_decode].tolist()
        
        if do_upsample_balance:
            from pythonlib.tools.listtools import tabulate_list
            from neuralmonkey.analyses.decode_good import decode_upsample_dataset, decode_resample_balance_dataset
            print("Upsampling dataset...")
            print("... starting distribution: ", tabulate_list(labels))
            X, labels = decode_resample_balance_dataset(X, labels, "upsample", do_upsample_balance_fig_path_nosuff)
            # X, labels = decode_upsample_dataset(X, labels, do_upsample_balance_fig_path_nosuff)
            print("... ending distribution: ", tabulate_list(labels))

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
        if PLOT:
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
        PARAMS:
        - color_by, str, how to determine color. 
        - params, list, flex, depends on color_by
        RETURN:
        - map_shape_to_col_this, map from shape (str) to 4-d color array.
        """

        map_shape_to_col_this = {}

        # Get for this trial
        if color_by=="stroke_index_seqc":
            # Color by the sequence of strokes actually drawn (0,1,2...), so that first stroke is always same color, etc
            
            shapes_drawn = params[0] # list of shape str
            MAP_INDEX_TO_COL = params[1] # index --> color
            shapes_to_plot = params[2]
            map_shape_to_color_orig = params[3]

            for i, sh in enumerate(shapes_drawn):
                col = MAP_INDEX_TO_COL[i]
                map_shape_to_col_this[sh] = col
            for sh in shapes_to_plot:
                if sh not in map_shape_to_col_this:
                    # Then give it a low-alpah version of its color
                    # map_shape_to_col_this[sh] = MAP_INDEX_TO_COL["NOT_DRAWN"]
                    col = map_shape_to_color_orig[sh].copy()
                    col[3] = 0.25
                    map_shape_to_col_this[sh] = col
        elif color_by=="shape_order_global":
            # Color by global index in sequence for Shape sequence rule, e.g. a rule for shape sequence

            shapes_drawn = params[0] # list of shape str
            MAP_INDEX_TO_COL = params[1] # index --> color
            shape_sequence = params[2] # 
            MAP_SHAPE_TO_INDEX = {sh:i for i, sh in enumerate(shape_sequence)}
            
            for sh in shapes_drawn:
                if sh not in shape_sequence: # ground truth sequeqnce
                    map_shape_to_col_this[sh] = np.array([0.8, 0.8, 0.8, 0.8])
                else:
                    idx = MAP_SHAPE_TO_INDEX[sh]
                    col = MAP_INDEX_TO_COL[idx]
                    map_shape_to_col_this[sh] = col
            # - get the shapes that are not drawn
            for sh in shapes_to_plot:
                if sh not in map_shape_to_col_this:
                    # Then give it a low-alpah version of its color
                    idx = MAP_SHAPE_TO_INDEX[sh]
                    col = MAP_INDEX_TO_COL[idx].copy()
                    col[3] = 0.35
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
            ax.plot(times, probs, label=lab, color=col, alpha=alpha)
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
        Helper to plot one curve for each storke index (0, 1, 2,...) -- i.e,. for each trial,
        what was the decode for sahpe that was done first (this is "0") and second ("1") and so on.
        For each trial this may be different shape, but only conisder its stroke index. 

        Used for shape sequence plots, showing the timecourse of decode for the different strokes that will 
        be drawn on that trial.

        PARAMS:
        - probs_mat_all_strokeindex, array of probs, (nlables, ntrials, ntimes)
        - MAP_INDEX_TO_COL, dict, from index 0, 1, 2... --> 4-d color array
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
                            plot_legend=True, alpha=1)
        
        ax.axhline(0, color="k", alpha=0.5)

        if ylims is not None:
            ax.set_ylim(ylims)
        
    def timeseries_plot_wrapper(self, PAprobs, list_title_filtdict, list_n_strokes=None,
                                SIZE=4, ylims=None, filter_n_strokes_using="both"):
        """
        Make a single figure with timecoures of deocde, where each subplot are all trials with one specific n strokes drawn,
        and further optionally filtered by conjunction of that an a filtdict.

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
        if len(list_title_filtdict)==0 or list_title_filtdict is None:
            list_title_filtdict = [("all", {})]

        ncols = len(list_title_filtdict)
        nrows = len(list_n_strokes)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*SIZE, nrows*SIZE), sharex=True, sharey=True)

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

        return dfscores
    
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
        from pythonlib.tools.pandastools import extract_with_levels_of_var_good, stringify_values

        print("Saving plots at ... ", savedir)

        dfscores = stringify_values(dfscores)
        # # Plot n trials for training
        # from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        # fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain.Xlabels["trials"], var_train, "task_kind", None)
        # savefig(fig, f"{savedir}/counts-var_train={var_train}.pdf")

        list_twind = dfscores["twind"].unique().tolist()

        ###
        row_values = sorted(dfscores["pa_class"].unique())
        col_values = sorted(dfscores["decoder_class"].unique())
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
            
            fig = sns.catplot(data=df, x="decoder_class", y=var_score, col="pa_class", col_wrap=6, alpha=0.2, 
                        jitter=True, hue="decoder_class_semantic_str")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-score-vs-decoder_class-twind={twind}-1.pdf")

            fig = sns.catplot(data=df, x="decoder_class", y=var_score, col="pa_class", col_wrap=6,
                    kind="bar", hue="decoder_class_semantic_str")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-score-vs-decoder_class-twind={twind}-2.pdf")

            fig = sns.catplot(data=df, x="decoder_class", y=var_score, hue="decoder_class_good", alpha=0.2, 
                        jitter=True)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-decoder_class-twind={twind}-1.pdf")

            fig = sns.catplot(data=df, x="decoder_class", y=var_score, hue="decoder_class_good", kind="bar")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-decoder_class-twind={twind}-2.pdf")

            plt.close("all")

        ###
        # Single scalar to summarize decoder for each class of data
        # - (same label) - (diff label)
        from pythonlib.tools.pandastools import summarize_featurediff
        dfsummary, _, _, _, _ = summarize_featurediff(dfscores, "same_class", [False, True], [var_score], 
                            ["decoder_class_good", "decoder_class_is_in_pa", "pa_class", "pa_class_is_in_decoder", "pa_idx", "trialcode", "twind"])
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

        plt.close("all")


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
                                               do_train_splits=False, do_train_splits_nsplits=5):
    """
    Wrapper to extract training data given some codeword inputs, Returns data where each datapt is (nchans, 1) vector, and each has associated label baseed on var_train
    RETURNS:
    - pa_train_all, holding training data, shape (chans, datapts, 1)
    - _twind_train, becuase twind has changed, pass this into testing
    - PAtrain, training trials, before split into vecotrs for pa_train_all

    """
    
    from neuralmonkey.classes.population_mult import extract_single_pa
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good
    from neuralmonkey.classes.population import concatenate_popanals_flexible

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
    
    if np.any(np.isnan(PAtrain.X)):
        print(PAtrain.X)
        print(DFallpa, bregion, None, which_level, event_train)
        assert False

    # Balance by resampling, optionally
    if downsample_trials==True and do_train_splits==False:
        # NOTE: shoudl do here, isntead of on output, since this keeps trials as the manipulated data level.
        from pythonlib.tools.pandastools import extract_resample_balance_by_var
        dflab = PAtrain.Xlabels["trials"]
        _dflab = extract_resample_balance_by_var(dflab, var_train, "min", "replacement", assert_all_rows_unique=True)
        inds_keep = sorted(_dflab.index.tolist())
        print(f"Downsampling (balance) PAtrain from {len(dflab)} rows to {len(inds_keep)} rows.")
        PAtrain = PAtrain.slice_by_dim_indices_wrapper("trials", inds_keep)
    
    if do_train_splits:
        # Given PAtrain, get train/test splits
        from sklearn.model_selection import StratifiedKFold
        dflab = PAtrain.Xlabels["trials"]
        labels = dflab[var_train].tolist()        

        if downsample_trials:
            # Then do balancing of training trials.
            from pythonlib.tools.statstools import balanced_stratified_kfold
            folds = balanced_stratified_kfold(np.zeros(len(labels)), labels, n_splits=do_train_splits_nsplits)
        else:
            from collections import Counter
            do_train_splits_nsplits = min([do_train_splits_nsplits, min(Counter(labels).values())])
            print(f"[not balanced] Doing {do_train_splits_nsplits} splits")
            skf = StratifiedKFold(n_splits=do_train_splits_nsplits, shuffle=True)
            folds = list(skf.split(np.zeros(len(labels)), labels)) # folds[0], 2-tuple of arays of ints

        trainsets = []
        for i, (train_index, test_index) in enumerate(folds):
            # print("-----", i)
            # print(train_index)    
            # print(test_index)
            PAtrain_train = PAtrain.slice_by_dim_indices_wrapper("trials", train_index, reset_trial_indices=True)
            pa_train_all, _twind_train, _ = train_decoder_helper_extract_train_dataset_slice(PAtrain_train, var_train, twind_train, 
                                                twind_train_null=twind_train_null, decoder_method_index=decoder_method_index)
            trainsets.append({
                "pa_train_all":pa_train_all,
                "_twind_train":_twind_train,
                "PAtrain_train":PAtrain_train,
                "train_index_PAtrain_orig":train_index,
                "test_index_PAtrain_orig":test_index,
                "PAtrain_orig":PAtrain
            })
        return trainsets
    else:
        # Return a single training set
        pa_train_all, _twind_train, PAtrain = train_decoder_helper_extract_train_dataset_slice(PAtrain, var_train, twind_train, 
                                                        twind_train_null=twind_train_null, decoder_method_index=decoder_method_index)
        return pa_train_all, _twind_train, PAtrain

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
                            do_train_splits_nsplits=5):
    """
    IDentical to train_decoder_helper, but do multiple times, each time splitting trials for training data, 
    and return each decoder, already trained, and do plots for each.
    PARAMS:
    - do_train_splits_nsplits, int, how many stratified train-test splits...
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
                                                                                     do_train_splits_nsplits=do_train_splits_nsplits)
    decoders = []
    for i_ts, ts in enumerate(trainsets):
        print("Training this classifier version: ", classifier_version, "trainset(splits)=", i_ts)
        pa_train_all = ts["pa_train_all"]
        _twind_train = ts["_twind_train"]
        PAtrain_train = ts["PAtrain_train"] # Just those used for training trials, here.
        if classifier_version == "ensemble":
            # Train
            Dc = DecoderEnsemble(pa_train_all, var_train, _twind_train)
            Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)
        else:
            # Train
            Dc = Decoder(pa_train_all, var_train, _twind_train)
            Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff, classifier_version=classifier_version)

        # A flag for "good" labels --> i.e. those in decoder that enough trials. 
        _df, _ = extract_with_levels_of_var_good(PAtrain_train.Xlabels["trials"], [Dc.VarDecode], n_min_per_var_good)
        labels_decoder_good = _df[Dc.VarDecode].unique().tolist()
        Dc.LabelsDecoderGood = labels_decoder_good

        # Save the decoder
        decoders.append(Dc)

        if PLOT:
            # Plot n trials for training
            from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
            savedir_this = f"{savedir}/trainsplit={i_ts}"
            os.makedirs(savedir_this, exist_ok=True)

            fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain_train.Xlabels["trials"], var_train, "task_kind", None)
            savefig(fig, f"{savedir_this}/counts-var_train={var_train}.pdf")

            # Plot scores on TRAIN set
            labels_decoder_good = Dc.LabelsDecoderGood
            dfscores_train = Dc.scalar_score_extract_df(PAtrain_train, twind_train, labels_decoder_good=labels_decoder_good, 
                                                        var_decode=var_train)
            sdir = f"{savedir_this}/train_set"
            os.makedirs(sdir, exist_ok=True)
            Dc.scalar_score_df_plot_summary(dfscores_train, sdir)      
        
    return trainsets, decoders

# def test_decoder_helper_extract_test_dataset(DFallpa, bregion, which_level, event, filterdict):
#     from neuralmonkey.classes.population_mult import extract_single_pa
#     PAtest = extract_single_pa(DFallpa, bregion, None, which_level=which_level, event=event)
#     PAtest = PAtest.slice_by_labels_filtdict(filterdict)
#     return PAtest

def _test_decoder_helper(Dc, PAtest, var_test, list_twind_test, subtract_baseline=False, PLOT=False, savedir=None,
                         prune_labels_exist_in_train_and_test=True):
    """
    Extract test dataest, and then score decoder Dc aginst this test dataset.
    PARAMS:
    - list_twind_test, multiple time windows to run tests over.
    """

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

    ### Plots
    if PLOT:
        sdir = f"{savedir}/test-var_score=score"
        os.makedirs(sdir, exist_ok=True)
        Dc.scalar_score_df_plot_summary(dfscores, sdir)

    return dfscores

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
    PAtest = PAtest.slice_by_labels_filtdict(filterdict_test)
    # if filterdict_test is not None:
    #     for col, vals in filterdict_test.items():
    #         print(f"filtering with {col}, starting len...", len(dflab))
    #         dflab = dflab[dflab[col].isin(vals)]
    #         print("... ending len: ", len(dflab))
    #     inds = dflab.index.tolist()
    #     PAtest = PAtest.slice_by_dim_indices_wrapper("trials", inds)
    # assert len(dflab)>0, "All data pruned!!!" 

    dfscores = _test_decoder_helper(Dc, PAtest, var_test, list_twind_test, subtract_baseline, PLOT, savedir, prune_labels_exist_in_train_and_test)

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
                                     prune_labels_exist_in_train_and_test=True, PLOT=True,
                                     which_level_train="trial", which_level_test="trial", n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=True,
                                     downsample_trials=False,
                                     allow_multiple_twind_test=False,
                                     classifier_version="logistic", # 7/11/24 - decided this, since it seems to do better for chars (same image, dif beh)
                                    #  classifier_version="ensemble",
                                     do_train_splits_nsplits=5, score_user_test_data=True):
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

    ### Train decoder
    # include_null_data=True
    if n_min_per_var is None:
        n_min_per_var=3
    n_min_per_var_good = 10

    do_upsample_balance_fig_path_nosuff = f"{savedir}/upsample_pcs"
    trainsets, decoders = train_decoder_helper_with_splits(DFallpa, bregion, var_train, event_train, twind_train,
                                    PLOT, include_null_data, n_min_per_var, filterdict_train,
                                    which_level_train, decoder_method_index, savedir, n_min_per_var_good,
                                    do_upsample_balance, do_upsample_balance_fig_path_nosuff,
                                    downsample_trials=downsample_trials, classifier_version=classifier_version,
                                    return_dfscores_train=True, do_train_splits_nsplits=do_train_splits_nsplits)
    
    # For each training set, evaluate (i) the test set and (ii) the held-out part of the training set.
    list_dfscores_testsplit = []
    list_dfscores_usertest = []

    for i_ts, (Dc, ts) in enumerate(zip(decoders, trainsets)):    
        savedir_this = f"{savedir}/trainsplit={i_ts}-vs-testsplit"
        
        # (1) Test on held-out data
        test_index = [int(i) for i in ts["test_index_PAtrain_orig"]] # array of ints
        PAtrain_orig = ts["PAtrain_orig"]

        # - slice out test dataset
        patest = PAtrain_orig.slice_by_dim_indices_wrapper("trials", test_index, reset_trial_indices=True)

        dfscores = _test_decoder_helper(Dc, patest, var_test, list_twind_test, subtract_baseline, PLOT, savedir_this, 
                                        prune_labels_exist_in_train_and_test)
        dfscores["train_split_idx"] = i_ts
        list_dfscores_testsplit.append(dfscores)

        if score_user_test_data:
            # (2) Test on user-inputted test set.
            savedir_this = f"{savedir}/trainsplit={i_ts}-vs-user_inputed_test"
            dfscores, PAtest = test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                                which_level_test, savedir_this, prune_labels_exist_in_train_and_test, PLOT,
                                subtract_baseline, subtract_baseline_twind, allow_multiple_twind_test=allow_multiple_twind_test)
            dfscores["train_split_idx"] = i_ts
        else:
            dfscores = pd.DataFrame([])
            PAtest = None
        list_dfscores_usertest.append(dfscores)

    # Concat the results
    dfscores_testsplit = pd.concat(list_dfscores_testsplit).reset_index(drop=True)
    dfscores_usertest = pd.concat(list_dfscores_usertest).reset_index(drop=True)

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

    return dfscores_testsplit, dfscores_usertest, decoders, trainsets, PAtest


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
                                     classifier_version="logistic"):
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
    from neuralmonkey.analyses.state_space_good import euclidian_distance_compute_trajectories_single
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

    # Sanity check -- each trialcode has a single case that is True
    assert np.all(dfscores[dfscores["decoder_class_was_drawn"]==True].groupby(["morph_set_idx", "trialcode"]).size().reset_index(drop=True)==1), "expect each (morph_set_idx, trialcode) to have 1 decoder (i.e, varying by idx within morpht) that matches it"

    return dfscores

def analy_psychoprim_prepare_beh_dataset(animal, date, savedir="/tmp"):
    """
    Extract beh related to psychometric prims, behavior.
    """
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_preprocess_wrapper, psychogood_plot_drawings_morphsets
    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_decide_if_tasks_are_ambiguous

    from neuralmonkey.scripts.analy_pig_decode_moment_syntaxTI import prepare_beh_dataset
    MS, D, _, _, _ = prepare_beh_dataset(animal, date, do_syntax_rule_stuff=False)        

    # Clean up first --> only trials with one beh stroke
    D.preprocessGood(params=["beh_strokes_one"])

    ### Extract all psycho prims stuff
    # How to map back to neural data?
    # -- trialcode

    # Code: given a (morphset, idx), get all sets of trialcodes
    # def find_morphset_morphidx(DSmorphsets, morphset, idx_in_morphset, return_as_trialcodes=True):
    #     """ Return indices in DSmorphsets that match morphset and idx_in_morphset
    #     """
    #     from pythonlib.tools.pandastools import _check_index_reseted
    #     # _check_index_reseted(DSmorphsets.Dat)
    #     inds = DSmorphsets.Dat[
    #         (DSmorphsets.Dat["morph_set_idx"] == morphset) & 
    #         (DSmorphsets.Dat["morph_idxcode_within_set"] == idx_in_morphset)].index.tolist()
        
    #     if return_as_trialcodes:
    #         return DSmorphsets.Dat.iloc[inds]["trialcode"].tolist()
    #     else:
    #         return inds

    if (animal, date) in [("Diego", 240523), ("Pancho", 240524)]:
        # Structured morphs -- e.g., modify angle of one arm gradaully
        DFRES, DSmorphsets, PARAMS, los_allowed_to_miss = psychogood_preprocess_wrapper(D, PLOT_DRAWINGS=False, 
                                                                                    PLOT_EACH_TRIAL=False, PLOT_SCORES=False,
                                                                                    clean_ver="singleprim_psycho_noabort")
    elif (animal, date) in [("Diego", 240522), ("Pancho", 240523)]:
        # Continuous morphs -- i.e., take onset and offset of two base prims to form one single middle prim.
        # Option: continuous morph between two base sets (e.g. take onset of one, and offset of other)
        from pythonlib.dataset.dataset_analy.psychometric_singleprims import preprocess_cont_morph
        _, DSmorphsets, _, _, _, _ = preprocess_cont_morph(D, clean_ver="singleprim_psycho_noabort")        
        # Recode to follow convention.
        map_old_to_new = {
            0:0, # base1
            1:99, # base2
            -1:1, # morph
        }
        DSmorphsets.Dat["morph_idxcode_within_set"] = [map_old_to_new[morph_idxcode_within_set] for morph_idxcode_within_set in DSmorphsets.Dat["morph_idxcode_within_set"]]
    elif (animal, date) in [
        ("Diego", 240515), ("Diego", 240517), ("Diego", 240521), 
        ("Pancho", 240516), ("Pancho", 240521),
        ]:
        # Angle morphs
        from pythonlib.dataset.dataset_analy.psychometric_singleprims import preprocess, plot_overview, preprocess_angle_to_morphsets, psychogood_decide_if_tasks_are_ambiguous, psychogood_prepare_for_neural_analy
        _, DSlenient, _ = preprocess(D, clean_ver="singleprim_psycho_noabort")

        if False:
            savedir = "/tmp/pshychjo"
            os.makedirs(savedir, exist_ok=True)
            plot_overview(DS, D, savedir)

        DSmorphsets = preprocess_angle_to_morphsets(DSlenient)
    else:
        print(animal, date)
        assert False

    # Plot the morphsets
    savedir_this = f"{savedir}/{animal}-{date}"
    os.makedirs(savedir_this, exist_ok=True)

    psychogood_plot_drawings_morphsets(DSmorphsets, savedir=savedir_this)

    from pythonlib.dataset.dataset_analy.psychometric_singleprims import psychogood_prepare_for_neural_analy
    DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, \
        map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status = psychogood_prepare_for_neural_analy(D, DSmorphsets)

    # Save things
    from pythonlib.tools.expttools import writeDictToTxtFlattened
    path = f"{savedir}/map_tc_to_morph_info.txt"
    writeDictToTxtFlattened(map_tc_to_morph_info, path, sorted_by_keys=True)

    path = f"{savedir}/map_morphset_to_basemorphinfo.txt"
    writeDictToTxtFlattened(map_morphset_to_basemorphinfo, path, sorted_by_keys=True)

    path = f"{savedir}/map_tcmorphset_to_idxmorph.txt"
    writeDictToTxtFlattened(map_tcmorphset_to_idxmorph, path, sorted_by_keys=True)

    path = f"{savedir}/map_tcmorphset_to_info.txt"
    writeDictToTxtFlattened(map_tcmorphset_to_info, path, sorted_by_keys=True)

    path = f"{savedir}/map_morphsetidx_to_assignedbase_or_ambig.txt"
    writeDictToTxtFlattened(map_morphsetidx_to_assignedbase_or_ambig, path, sorted_by_keys=True)

    path = f"{savedir}/map_tc_to_morph_status.txt"
    writeDictToTxtFlattened(map_tc_to_morph_status, path, sorted_by_keys=True)

    # from pythonlib.tools.pandastools import _check_index_reseted
    # _check_index_reseted(DSmorphsets.Dat)

    # # PLOT_SAVEDIR = "/tmp"
    # PLOT_SAVEDIR = None
    # assignments = psychogood_decide_if_tasks_are_ambiguous(DSmorphsets, PLOT_SAVEDIR)

    # from pythonlib.tools.pandastools import grouping_print_n_samples


    # # for morphset in sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist()):
    # #     df = DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == morphset]

    # #     from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
    # #     grouping_plot_n_samples_conjunction_heatmap(df, "los_info", "morph_idxcode_within_set", ["morph_set_idx"])

    # ##### Condition DSmorphsets
    # # Each base shape is defined by (morphset, idx_within)  -- i.e.e the "shape" variable is wrong - its the task component.
    # from pythonlib.tools.pandastools import append_col_with_grp_index
    # DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_set_idx", "morph_idxcode_within_set"], "morph_id_both", use_strings=False)
    # DSmorphsets.Dat = append_col_with_grp_index(DSmorphsets.Dat, ["morph_set_idx", "morph_idxcode_within_set"], "morph_id_both_str", use_strings=True)
    # ##### Generate various mappings (beh features)
    # # Generate maps of this kind -- MAP: tc --> stuff [GOOD]

    # import numpy as np

    # # Map from trialcode to whether is base or morph
    # map_tc_to_morph_info = {}
    # for tc in D.Dat["trialcode"]:
    #     if tc not in DSmorphsets.Dat["trialcode"].tolist():
    #         map_tc_to_morph_info[tc] = "no_exist"
    #         print("This tc in D.Dat but not in DSmorphsets...", tc)
    #     else:
    #         tmp = DSmorphsets.Dat[DSmorphsets.Dat["trialcode"] == tc]

    #         morph_is_morphed_list = tmp["morph_is_morphed"].unique().tolist()
    #         assert len(morph_is_morphed_list)==1

    #         if morph_is_morphed_list[0]:
    #             # Is morph -- Then this is not base prim.
    #             map_tc_to_morph_info[tc] = "morphed"
    #         else:
    #             # THis is base prim. But could be participant in multiple morhph sets, so 
    #             # Just give it the first id
    #             mid = tmp["morph_id_both_str"].unique().tolist()
    #             map_tc_to_morph_info[tc] = mid[0]

    # # Generate maps of this kind -- morphset --> stuff

    # # For base prims, map from (morphset) --> (base1, base2) where base1 and base2 are the codes used in decoder.\

    # # {0: ('0|0', '0|99'),
    # #  1: ('0|0', '0|99'),
    # #  2: ('2|0', '2|99'),
    # #  3: ('2|0', '3|99'),
    # #  4: ('4|0', '0|99'),
    # #  5: ('0|99', '5|99'),
    # #  6: ('6|0', '4|0')}

    # list_morphset = DSmorphsets.Dat["morph_set_idx"].unique()
    # map_morphset_to_basemorphinfo = {}
    # for morphset in list_morphset:

    #     trialcodes = find_morphset_morphidx(DSmorphsets, morphset, 0)
    #     mis = [map_tc_to_morph_info[tc] for tc in trialcodes]
    #     assert len(set(mis))==1
    #     base1_mi = mis[0]
        
    #     trialcodes = find_morphset_morphidx(DSmorphsets, morphset, 99)
    #     mis = [map_tc_to_morph_info[tc] for tc in trialcodes]
    #     assert len(set(mis))==1
    #     base2_mi = mis[0]
        
    #     map_morphset_to_basemorphinfo[morphset] = (base1_mi, base2_mi)
    # # Generate maps of this kind -- MAP: from (tc, morphset) --> stuff

    # map_tcmorphset_to_idxmorph = {} # (tc, morphset) --> idx_in_morphset | "not_in_set"
    # map_tcmorphset_to_info = {} # (tc, morphset) --> (amibig, base1, base2)

    # for i, row in DSmorphsets.Dat.iterrows():
    #     tc = row["trialcode"]
    #     morphset = row["morph_set_idx"]
    #     morph_idxcode_within_set = row["morph_idxcode_within_set"]

    #     # (1) 
    #     assert (tc, morphset) not in map_tcmorphset_to_idxmorph,  "probably multiple strokes on this trial..."
    #     map_tcmorphset_to_idxmorph[(tc, morphset)] = morph_idxcode_within_set

    #     # (2)
    #     if (tc, morphset) in map_tcmorphset_to_info:
    #         print(tc, morphset, row["morph_idxcode_within_set"], row["stroke_index"])
    #         assert False, "probably multiple strokes on this trial..."
    #     else:
    #         if False:
    #             # Get its base prims
    #             _inds = find_morphset_morphidx(DSmorphsets, morphset, 0, False)
    #             _tmp = DSmorphsets.Dat.iloc[_inds]["morph_id_both_str"].unique()
    #             assert len(_tmp)==1
    #             base1_mi = _tmp[0]

    #             _inds = find_morphset_morphidx(DSmorphsets, morphset, 99, False)
    #             _tmp = DSmorphsets.Dat.iloc[_inds]["morph_id_both_str"].unique()
    #             assert len(_tmp)==1
    #             base2_mi = _tmp[0]
    #         else:
    #             base1_mi = map_morphset_to_basemorphinfo[morphset][0]
    #             base2_mi = map_morphset_to_basemorphinfo[morphset][1]

    #         map_tcmorphset_to_info[(tc, morphset)] = (row["morph_assigned_to_which_base"], base1_mi, base2_mi)
            
    # # Fill in the missing ones
    # list_morphset = DSmorphsets.Dat["morph_set_idx"].unique().tolist()
    # list_tc = D.Dat["trialcode"].tolist()
    # for morphset in list_morphset:
    #     for tc in list_tc:
    #         if (tc, morphset) not in map_tcmorphset_to_idxmorph:
    #             map_tcmorphset_to_idxmorph[(tc, morphset)] = "not_in_set"
    # # Generate maps of this kind -- (morphset, idx within) --> stuff

    # map_morphsetidx_to_assignedbase_or_ambig = {}
    # # map_morphsetidx_to_assignedbase = {}
    # for i, row in DSmorphsets.Dat.iterrows():
    #     morphset = row["morph_set_idx"]
    #     morph_idxcode_within_set = row["morph_idxcode_within_set"]  

    #     key = (morphset, morph_idxcode_within_set)

    #     # Convert to avalue that is same across trials.
    #     if row["morph_assigned_to_which_base"] in ["ambig_base2", "ambig_base1"]:
    #         value = "is_ambig"
    #     else:
    #         value = row["morph_assigned_to_which_base"]

    #     if key in map_morphsetidx_to_assignedbase_or_ambig:
    #         assert map_morphsetidx_to_assignedbase_or_ambig[key] == value
    #     else:
    #         map_morphsetidx_to_assignedbase_or_ambig[key] = value

    #     # if key in map_morphsetidx_to_assignedbase:
    #     #     assert map_morphsetidx_to_assignedbase[key] == row["morph_assigned_to_which_base"]
    #     # else:
    #     #     map_morphsetidx_to_assignedbase[key] = row["morph_assigned_to_which_base"]

    # map_morphsetidx_to_assignedbase_or_ambig = {k:map_morphsetidx_to_assignedbase_or_ambig[k] for k in sorted(map_morphsetidx_to_assignedbase_or_ambig.keys())}
    # # map_morphsetidx_to_assignedbase = {k:map_morphsetidx_to_assignedbase[k] for k in sorted(map_morphsetidx_to_assignedbase.keys())}


    # # list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())    
    # # map_morphsetidx_to_assignedbase = {}
    # # for morphset in list_morphset:
    # #     has_switched = False
    # #     list_idx = sorted(DSmorphsets.Dat[DSmorphsets.Dat["morph_set_idx"] == morphset]["morph_idxcode_within_set"].unique().tolist())    
    # #     for idx in list_idx:
    # #         k = (morphset, idx)

    # #         if map_morphsetidx_to_isambig[k]:
    # #             has_switched = True
    # #             assigned_base = "is_ambig"
    # #         else:
    # #             if has_switched or idx==99:
    # #                 assigned_base = "base2"
    # #             else:
    # #                 assigned_base = "base1"
            
    # #         map_morphsetidx_to_assignedbase[k] = assigned_base

    # #         print(idx, map_morphsetidx_to_isambig[k], assigned_base)
        
    # # Map from trialcode to morph information
    # # NOTE - this doesnt work, since a given TC can be in multiple morph sets
    # map_tc_to_morph_status = {}
    # ct_missing = 0
    # ct_present = 0
    # for tc in D.Dat["trialcode"]:
    #     if tc not in DSmorphsets.Dat["trialcode"].tolist():
    #         print("This tc in D.Dat but not in DSmorphsets...", tc)
    #         ct_missing += 1
    #     else:
    #         tmp = DSmorphsets.Dat[DSmorphsets.Dat["trialcode"] == tc]["morph_is_morphed"].unique().tolist()
    #         assert len(tmp)==1
    #         ct_present += 1

    #         # map_tc_to_morph_info[tc] = (tmp["morph_set_idx"].values[0], tmp["morph_idxcode_within_set"].values[0], tmp["morph_assigned_to_which_base"].values[0], tmp["morph_is_morphed"].values[0])

    # print("Missing / got:", ct_missing, ct_present)
    # from pythonlib.tools.pandastools import grouping_print_n_samples
    # grouping_print_n_samples(DSmorphsets.Dat, ["shape", "morph_idxcode_within_set", "morph_is_morphed"])

    return DSmorphsets, map_tc_to_morph_info, map_morphset_to_basemorphinfo, map_tcmorphset_to_idxmorph, map_tcmorphset_to_info, map_morphsetidx_to_assignedbase_or_ambig, map_tc_to_morph_status


def analy_psychoprim_score_postsamp(DFallpa, DSmorphsets, 
                                    map_tcmorphset_to_idxmorph, map_morphsetidx_to_assignedbase_or_ambig,
                                    map_tcmorphset_to_info,
                                    SAVEDIR_BASE, list_bregion=None):
    """
    Wrapper to run main scoring and plots for psychoprim
    """

    TWIND_TEST = (0.05, 1.2)
    # TWIND_TEST = (0.1, 0.9)
    # TWIND_TEST = (0.05, 0.6)
    # TWIND_TEST = (0.05, 1.2)
    TWIND_TRAIN = (0.05, 1.2)
    # Subtrract baseline?
    subtract_baseline=False
    subtract_baseline_twind=None
    include_null_data = False
    do_upsample_balance=True
    PLOT_DECODER = True
    n_min_per_var = 6
    prune_labels_exist_in_train_and_test = True

    if list_bregion is None:
        list_bregion = DFallpa["bregion"].unique().tolist() 
    
    list_morphset = sorted(DSmorphsets.Dat["morph_set_idx"].unique().tolist())
    for downsample_trials in [False, True]:
    # for downsample_trials in [False]:
        SAVEDIR = f"{SAVEDIR_BASE}/downsample_trials={downsample_trials}-TWIND_TEST={TWIND_TEST}"
        for bregion in list_bregion:
            list_df = []
            for morphset in list_morphset:
                # Given morphset, assign new column which is the trial's role in that morphset.

                for pa in DFallpa["pa"].values:
                    dflab = pa.Xlabels["trials"]
                    dflab["idx_morph_temp"] = [map_tcmorphset_to_idxmorph[(tc, morphset)] for tc in dflab["trialcode"]]

                # Train on all the base_prims data
                idx_exist = sorted(list(set([x for x in dflab["idx_morph_temp"] if x!="not_in_set"])))

                event_train = "03_samp"
                twind_train = TWIND_TRAIN
                filterdict_train = {"idx_morph_temp":idx_exist}
                var_train = "idx_morph_temp"

                # Test on morphed data - get all here
                var_test = "idx_morph_temp"
                event_test = "03_samp"
                filterdict_test = {"idx_morph_temp":idx_exist}
                list_twind_test = [TWIND_TEST]
                which_level_test = "trial"

                # Other params
                savedir = f"{SAVEDIR}/{bregion}/morphset={morphset}/decoder_training"
                os.makedirs(savedir, exist_ok=True)
                print(savedir)
                
                dfscores, Dc, PAtrain, PAtest = pipeline_train_test_scalar_score(DFallpa, bregion, var_train, event_train, 
                                                                                twind_train, filterdict_train,
                                                    var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                    include_null_data=include_null_data, decoder_method_index=None,
                                                    prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, PLOT=PLOT_DECODER,
                                                    which_level_test=which_level_test, n_min_per_var=n_min_per_var,
                                                    subtract_baseline=subtract_baseline, subtract_baseline_twind=subtract_baseline_twind,
                                                    do_upsample_balance=do_upsample_balance, downsample_trials=downsample_trials)
                
                dfscores = analy_psychoprim_dfscores_condition(dfscores, morphset, DSmorphsets, map_morphsetidx_to_assignedbase_or_ambig, map_tcmorphset_to_info)

                ##### PLOTS
                savedir = f"{SAVEDIR}/{bregion}/morphset={morphset}/plots"
                os.makedirs(savedir, exist_ok=True)
                print("Saving plots at... ", savedir)

                for var_score in ["score", "score_norm"]:
                    # var_score = "score_norm"

                    for x_var in ["decoder_class_semantic_good", "decoder_class"]:

                        if x_var == "decoder_class_semantic_good":
                            order = ("base1", "interm1", "same", "interm2", "base2")
                        else:
                            order = None

                        fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue="pa_class", kind="point", errorbar=("ci", 68), 
                                        order=order)
                        for ax in fig.axes.flatten():
                            ax.axhline(0, color="k", alpha=0.3)
                        savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-1.pdf")

                        fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue="pa_class", kind="point", errorbar=("ci", 68), 
                                        col="pa_morph_assigned_baseorambig", order=order)
                        for ax in fig.axes.flatten():
                            ax.axhline(0, color="k", alpha=0.3)
                        savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-2.pdf")

                        fig = sns.catplot(data=dfscores, x=x_var, y=var_score, hue="pa_morph_assigned_baseorambig", kind="point", 
                                        errorbar=("ci", 68), col="pa_class", order=order)
                        for ax in fig.axes.flatten():
                            ax.axhline(0, color="k", alpha=0.3)
                        savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-3.pdf")

                        if False: # not usefiule
                            sns.catplot(data=dfscores, x="pa_class", y=var_score, hue="decoder_class", kind="point", errorbar=("ci", 68))
                            sns.catplot(data=dfscores, x="pa_class", y=var_score, hue="decoder_class", kind="point", errorbar=("ci", 68), col="pa_morph_assigned_baseorambig")

                        from pythonlib.tools.pandastools import plot_subplots_heatmap

                        for var_subplot in [None, "pa_morph_assigned_baseorambig", x_var]:
                            row_values = sorted(dfscores["pa_class"].unique().tolist())
                            col_values = sorted(dfscores["decoder_class"].unique().tolist())

                            zlims = [0,1]
                            fig, axes = plot_subplots_heatmap(dfscores, "pa_class", "decoder_class", var_score, var_subplot, 
                                                            share_zlim=True, row_values=row_values,
                                                col_values=col_values, ZLIMS=zlims)
                            savefig(fig, f"{savedir}/heatmap-subplot={var_subplot}-{var_score}.pdf")

                            plt.close("all")

                ### Collect
                list_df.append(dfscores)

            ### Plot summary
            DFSCORES = pd.concat(list_df).reset_index(drop=True)

            if False:
                list_morphset_keep = [1,3,4,5,6] # Just those with ambig cases
                print(list_morphset_keep)

                DFSCORES = DFSCORES[DFSCORES["morph_set_idx"].isin(list_morphset_keep)].reset_index(drop=True)
        
            # Save data
            import pickle
            savedir = f"{SAVEDIR}/{bregion}"
            with open(f"{savedir}/DFSCORES.pkl", "wb") as f:
                pickle.dump(DFSCORES, f)

            # Plot
            savedir = f"{SAVEDIR}/{bregion}/combine_morphsets-all_scalar_summary"
            os.makedirs(savedir, exist_ok=True)
            print("Saving plots at... ", savedir)

            for var_score in ["score", "score_norm"]:
                # var_score = "score_norm"

                for x_var in ["decoder_class_semantic_good", "decoder_class"]:

                    if x_var == "decoder_class_semantic_good":
                        order = ("base1", "interm1", "same", "interm2", "base2")
                    else:
                        order = None


                    fig = sns.catplot(data=DFSCORES, x=x_var, y=var_score, kind="bar", errorbar=("ci", 68), 
                                      col = "pa_morph_assigned_baseorambig", order=order)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.3)
                    savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-1.pdf")

                    fig = sns.catplot(data=DFSCORES, x=x_var, y=var_score, hue="pa_morph_assigned_baseorambig", 
                                      kind="point", errorbar=("ci", 68), 
                                    order=order)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.3)
                    savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-2.pdf")

                    fig = sns.catplot(data=DFSCORES, x=x_var, y=var_score, hue="morph_set_idx", kind="point",
                                      col="pa_morph_assigned_baseorambig", errorbar=("ci", 68), order=order)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.3)
                    savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-3.pdf")

                    fig = sns.catplot(data=DFSCORES, x=x_var, y=var_score, hue="pa_morph_assigned_baseorambig", kind="point",
                                      col="morph_set_idx", errorbar=("ci", 68), order=order)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.3)
                    savefig(fig, f"{savedir}/decoder_class-x_var={x_var}-var_score={var_score}-4.pdf")

                    plt.close("all")

                for var_subplot in [None, "morph_set_idx"]:
                    row_values = sorted(DFSCORES["pa_class"].unique().tolist())
                    col_values = sorted(DFSCORES["decoder_class"].unique().tolist())

                    zlims = [0,1]
                    fig, axes = plot_subplots_heatmap(DFSCORES, "pa_morph_assigned_baseorambig", 
                                                        "decoder_class_semantic_good", var_score, var_subplot, 
                                                        share_zlim=True, row_values=row_values,
                                        col_values=col_values, ZLIMS=zlims)
                    savefig(fig, f"{savedir}/heatmap-subplot={var_subplot}-{var_score}.pdf")

                    plt.close("all")

            
            # Trial by trial variation in what drawn, given same stimulus
            ##### Prep dataset
            savedir = f"{SAVEDIR}/{bregion}/combine_morphsets-ambig_trials"
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
                                                        ["score", "score_norm"], nonnumercols=["decoder_class_good", "decoder_class_semantic_good"])
            else:
                # Version 2 - pull out each task ("morph_set_idx", "pa_class") that is ambiguous, and also just the datapts for the base1 and base2 decoders
                # i.e, this is subset of above.
                dfscores_ambig, dict_df = extract_with_levels_of_conjunction_vars_helper(DFSCORES, "decoder_class_was_drawn", 
                                                                                    task_vars + ["decoder_class"], 1, None, 2)

                # Aggregate, so each task contributes exactly 1 datapt to (False, True) for 
                # decoder_class_was_drawn
                dfscores_ambig_agg = aggregGeneral(dfscores_ambig, task_vars + ["decoder_class", "decoder_class_was_drawn", "trial_morph_assigned_to_which_base"], 
                                                        ["score", "score_norm"], nonnumercols=["decoder_class_good", "decoder_class_semantic_good"])
                
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
    if False:
        morphsets_ignore = [0]
    else:
        morphsets_ignore = []

    # PLOT_EACH_IDX = True

    if PLOT_EACH_IDX==False:
        PLOT_DECODER = False

    # PARAMS = {}
    list_bregion = sorted(DFallpa["bregion"].unique().tolist())
    for bregion in list_bregion:

        SAVEDIR = f"{SAVEDIR_BASE}/{bregion}"

        LIST_DFSCORES = []
        LIST_TC_RES = []
        for morphset in list_morphset:
            MORPHSET = morphset

            if morphset in morphsets_ignore:
                continue

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

                dfscores_testsplit, dfscores_usertest, decoders, trainsets, PAtest = pipeline_train_test_scalar_score_with_splits(DFallpa, bregion, var_train, event_train, 
                                                                                twind_train, filterdict_train,
                                                    var_test, event_test, list_twind_test, filterdict_test, savedir,
                                                    include_null_data=include_null_data, decoder_method_index=None,
                                                    prune_labels_exist_in_train_and_test=prune_labels_exist_in_train_and_test, PLOT=PLOT_DECODER,
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
