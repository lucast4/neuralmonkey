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


TBIN_DUR = 0.15
TBIN_SLIDE = 0.02

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
        self.LabelsUnique = sorted(self.PAtrain.Xlabels["trials"][self.VarDecode].unique().tolist())
        # Initialize
        self.LabelsDecoderGood = None

    def train_decoder(self, PLOT=False, do_upsample_balance=True, do_upsample_balance_fig_path_nosuff=None):
        """ Train a decoder and store it in self
        """

        from neuralmonkey.analyses.decode_good import decode_train_model
        from sklearn.linear_model import LogisticRegression
        from neuralmonkey.analyses.decode_good import decode_train_model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC
        from sklearn.preprocessing import MultiLabelBinarizer


        var_decode = self.VarDecode
        twind_train = self.Params["twind_train"]

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
            print("Upsampling dataset...")
            from neuralmonkey.analyses.decode_good import decode_upsample_dataset
            X, labels = decode_upsample_dataset(X, labels, do_upsample_balance_fig_path_nosuff)

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

        if PLOT:
            for x, y in zip(labels, labels_mlb):
                print(x, y)

        # These are the classes
        print("Classes, in order: ", mlb.classes_)

        # (2) Fit classifier
        clf = OneVsRestClassifier(LogisticRegression()).fit(X, labels_mlb)
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
                          labels_to_plot=None):
        """
        Plot timecourse of decode for an example trial. Wrapper of _plot_single_trial()
        PARAMS:
        - tbin_dur, in sec, window duration for smoothing. Note: use >=0.1 to avoid high noisiness
        - tbin_slide, in sec, for smoothing
        """
        from pythonlib.tools.plottools import legend_add_manual, makeColors

        if tbin_dur is None:
            tbin_dur=TBIN_DUR
        if tbin_slide is None:
            tbin_slide=TBIN_SLIDE            

        if PA is None:
            PA = self.PAtrain

        # clf = self.Classifier
        # mlb = self.MultiLabelBinarizer

        # # Smooth the data
        # pathis = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        
        # # Plot timecourse of decode
        # x = pathis.X[:, indtrial, :].T # (ntimes, nchans)
        # probs_mat = clf.predict_proba(x) # (ntimes, nlabels)
        # times = pathis.Times
        probs_mat, times, labels = self.timeseries_score_single(indtrial, None, PA, labels_to_plot, tbin_dur, tbin_slide)

        fig, ax, map_lab_to_col = self._plot_single_trial(probs_mat, times, labels, map_lab_to_col, title, 
                           shift_time_dur, ax, plot_legend, labels_to_plot)
        
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

    def timeseries_score_single(self, indtrial, twind=None, PA=None, labels_in_order_keep=None,
                                tbin_dur=None, tbin_slide=None):
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

        if tbin_dur is None:
            tbin_dur=TBIN_DUR
        if tbin_slide is None:
            tbin_slide=TBIN_SLIDE            

        if PA is None:
            PA = self.PAtrain

        clf = self.Classifier
        # mlb = self.MultiLabelBinarizer

        if twind is not None:
            PA = PA.slice_by_dim_values_wrapper("times", twind)

        # Smooth the data
        PA = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)

        
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

    def timeseries_score_wrapper(self, PA, twind=None, indtrials=None, 
                                 labels_in_order_keep=None, tbin_dur=None, tbin_slide=None):
        """
        Retrun all activations across these trials, in a matrix of probabilties.
        RETURNS:
        - probs_mat_all, (trials, times, labels)
        - times
        - labels
        """

        if indtrials is None:
            indtrials = list(range(len(PA.Trials)))
                      
        list_probs_mat = []
        times = None
        labels = None
        for ind in indtrials:
            print(ind)
            # Collect probs mat across all trials
            probs_mat, _times, _labels = self.timeseries_score_single(ind, PA=PA, twind=twind, labels_in_order_keep=labels_in_order_keep,
                                                        tbin_dur=tbin_dur, tbin_slide=tbin_slide)

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

        # Return as PA (labels, trials, times)
        from neuralmonkey.classes.population import PopAnal
        dflab = PA.Xlabels["trials"]
        X = np.transpose(probs_mat_all, (2, 0, 1))
        PAprobs = PopAnal(X, times=times, chans=labels, trials=indtrials)
        PAprobs.Xlabels["trials"] = dflab.iloc[indtrials].copy().reset_index(drop=True)
        assert len(PAprobs.Trials) == len(PAprobs.Xlabels["trials"])

        return PAprobs, probs_mat_all, times, labels


    def timeseries_plot_by_shape_drawn_order(self, PAprobs, ax, ylims=None):
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

        # ---- Slice
        probs_mat_all = PAprobs.X # labels, trials, times
        dflab_this = PAprobs.Xlabels["trials"]
        labels = PAprobs.Chans
        times = PAprobs.Times

        # Auto count n strokes drawn
        tmp = list(set([len(x) for x in dflab_this["shapes_drawn"]]))
        assert len(tmp)==1, "only can run if all trials have same n strokes drawn. see docs for reason"
        nstrokes = tmp[0]

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
                _probs_mat_not_drawn = np.mean(probs_mat[idxs_in_locs_not_drawn, :], axis=0)[None, :]

                assert len(idxs_in_locs_drawn)==nstrokes

            elif self.VarDecode == "seqc_0_shape":
                ##### Collect probs
                # 1. Shapes that were drawn
                # Pull out indices from probs_mat, in order of shapes drawn
                idxs_in_shapes_drawn = [labels.index(sh) for sh in shapes_drawn]
                # include labels for "not drawn"
                idxs_in_shapes_not_drawn = [i for i, sh in enumerate(labels) if sh not in shapes_drawn]
                assert sorted(idxs_in_shapes_drawn + idxs_in_shapes_not_drawn) == list(range(len(labels)))

                assert len(idxs_in_shapes_drawn)==nstrokes
                
                _probs_mat_drawn = probs_mat[idxs_in_shapes_drawn, :]
                _probs_mat_not_drawn = np.mean(probs_mat[idxs_in_shapes_not_drawn, :], axis=0)[None, :]
            else:
                print(self.VarDecode)
                assert False
                
            probs_mat_reordered = np.concatenate([_probs_mat_drawn, _probs_mat_not_drawn], axis=0) # (ndrawn+1, ntimes)
            list_probs_mat.append(probs_mat_reordered)

            
        probs_mat_all_strokeindex = np.stack(list_probs_mat) # (trials, stroke_index, times)
        probs_mat_strokeindex = np.mean(probs_mat_all_strokeindex, axis=0)
        labels_this = list(range(nstrokes)) + ["not_drawn"]

        print("PLOTTING... () ", probs_mat_all_strokeindex.shape)
        from neuralmonkey.neuralplots.population import plot_smoothed_fr
        for i in range(probs_mat_all_strokeindex.shape[1]):
            lab = labels_this[i]
            probs_mat = probs_mat_all_strokeindex[:, i, :]
            plot_smoothed_fr(probs_mat, times, ax=ax, color=MAP_INDEX_TO_COL[lab])

        self._plot_single_trial(probs_mat_strokeindex.T, times, labels=labels_this, map_lab_to_col=MAP_INDEX_TO_COL, ax=ax, 
                            plot_legend=True, alpha=1)
        
        ax.axhline(0, color="k", alpha=0.5)

        if ylims is not None:
            ax.set_ylim(ylims)


    def scalar_score_twinds_trials(self, list_twind, PA=None, tbin_dur=None, tbin_slide=None, 
                                   PLOT=True, return_fig=False, height=12, zmax=1, score_ver="mean"):
        """
        Get mean decode within each trial and each twind in list_twind, and return as
        array (ntrials, nclasses, ntwinds)

        PARAMS:
            list_twind = [
                [-0.7, -0.1],
                [0.1, 0.7],
            ]
        """
        if PA is None:
            PA = self.PAtrain

        if tbin_dur is None:
            tbin_dur = TBIN_DUR
        if tbin_slide is None:
            tbin_slide=TBIN_SLIDE
        pathis = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        clf = self.Classifier

        ntrials = len(pathis.Trials)
        assert ntrials>0
        nclass = clf.n_classes_
        ntwind = len(list_twind)
        scores = np.zeros((ntrials, nclass, ntwind))-1 # -1, so can sanity check that all filled
        for ind_trial in range(len(pathis.Trials)):

            x = pathis.X[:, ind_trial, :].T # (ntimes, nchans)
            probs_mat = clf.predict_proba(x) # (ntimes, nlabels)
            times = pathis.Times

            times.shape
            probs_mat.shape

            for ind_twind, twind in enumerate(list_twind):
                inds = (times>=twind[0]) & (times<=twind[1])
                if score_ver=="mean":
                    probs_vec = np.mean(probs_mat[inds, :], axis=0)
                elif score_ver=="median":
                    probs_vec = np.median(probs_mat[inds, :], axis=0)
                elif score_ver=="max":
                    probs_vec = np.max(probs_mat[inds, :], axis=0)
                else:
                    assert False

                assert np.all(scores[ind_trial, :, ind_twind]<0), "already filled..."
                scores[ind_trial, :, ind_twind] = probs_vec
                # res.append(probs_vec)
        assert np.all(scores>0), "did not fill up scores..."     

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

        if tbin_dur is None:
            tbin_dur=TBIN_DUR
        if tbin_slide is None:
            tbin_slide=TBIN_SLIDE            

        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        if PA is None:
            PA = self.PAtrain

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
    

######## HELPER FUNCTIONS
def train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, event_train, twind_train, 
                                               include_null_data=False, n_min_per_var=5, filterdict_train=None,
                                               which_level="trial", decoder_method_index=None, PLOT=False):
    """
    Extract training data given some codeword inputs, Returns data where each datapt is (nchans, 1) vector, and each has associated label baseed on var_train
    RETURNS:
    - pa_train_all, holding training data, shape (chans, datapts, 1)
    - _twind_train, becuase twind has changed, pass this into testing
    - PAtrain, training trials, before split into vecotrs for pa_train_all
    """

    from neuralmonkey.classes.population_mult import extract_single_pa
    from pythonlib.tools.pandastools import extract_with_levels_of_var_good

    if decoder_method_index is None:
        decoder_method_index = 2
    if include_null_data:
        twind_train_null = (-0.8, -0.1)
    else:
        twind_train_null = None

    PAtrain = extract_single_pa(DFallpa, bregion, None, which_level, event_train)
    if PLOT:
        PAtrain.plotNeurHeat(trial=100)

    ###### FILTER trials, if required.
    # dflab = PAtrain.Xlabels["trials"]
    PAtrain = PAtrain.slice_by_labels_filtdict(filterdict_train)

    # Prune to keep only cases with at least n trials per label
    _, inds_keep = extract_with_levels_of_var_good(PAtrain.Xlabels["trials"], [var_train], n_min_per_var=n_min_per_var)
    print("Keeping n trials / total: ", len(inds_keep), "/", len(PAtrain.Trials))
    PAtrain = PAtrain.slice_by_dim_indices_wrapper("trials", inds_keep)
    
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
            pa_train_presamp.Xlabels["trials"][var_train] = "null"

            # Concatenate pre and post samp
            from neuralmonkey.classes.population import concatenate_popanals_flexible
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
                pa_train_presamp.Xlabels["trials"][var_train] = "null"
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


def train_decoder_helper(DFallpa, bregion, var_train="seqc_0_shape", event_train=None, 
                            twind_train = None,
                            PLOT=True, include_null_data=False,
                            n_min_per_var=5, filterdict_train=None,
                            which_level="trial", decoder_method_index=None,
                            savedir=None, n_min_per_var_good = 10,
                            do_upsample_balance=False, do_upsample_balance_fig_path_nosuff=None):
    """
    Train a decoder for this bregion and train variables, and make relevant plots, and with variaous 
    preprocessing methods optional.
    PARAMS:
    - include_null_data, bool, if True, then includes "pre-samp" data as a "null" label. Tends to
    make decoding post-samp worse.
    - n_min_per_var, keeps only those levels of var_train which have at least this many trials.
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

    pa_train_all, _twind_train, PAtrain = train_decoder_helper_extract_train_dataset(DFallpa, bregion, var_train, 
                                                                                     event_train, twind_train,  
                                                                                     include_null_data, n_min_per_var, filterdict_train,
                                                                                     which_level, decoder_method_index, PLOT)

    # Train
    Dc = Decoder(pa_train_all, var_train, _twind_train)
    Dc.train_decoder(PLOT, do_upsample_balance, do_upsample_balance_fig_path_nosuff)

    # A flag for "good" labels --> i.e. those in decoder that enough trials. 
    _df, _ = extract_with_levels_of_var_good(PAtrain.Xlabels["trials"], [Dc.VarDecode], n_min_per_var_good)
    labels_decoder_good = _df[Dc.VarDecode].unique().tolist()
    Dc.LabelsDecoderGood = labels_decoder_good

    if PLOT:
        # Plot n trials for training
        from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain.Xlabels["trials"], var_train, "task_kind", None)
        savefig(fig, f"{savedir}/counts-var_train={var_train}.pdf")

    return PAtrain, Dc

def test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                        which_level_test, savedir, prune_labels_exist_in_train_and_test=True, PLOT=True,
                        subtract_baseline=False, subtract_baseline_twind=None):
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

    assert len(list_twind_test)>0

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

    # Finalize by concatting.
    dfscores = pd.concat(list_df).reset_index(drop=True)

    ### Plots
    if PLOT:
        # # Plot n trials for training
        # from pythonlib.tools.pandastools import grouping_plot_n_samples_conjunction_heatmap
        # fig = grouping_plot_n_samples_conjunction_heatmap(PAtrain.Xlabels["trials"], var_train, "task_kind", None)
        # savefig(fig, f"{savedir}/counts-var_train={var_train}.pdf")

        row_values = sorted(dfscores["pa_class"].unique())
        col_values = sorted(dfscores["decoder_class"].unique())
        fig, axes = plot_subplots_heatmap(dfscores, "pa_class", "decoder_class", "score", "twind", False, True,
                            row_values=row_values, col_values=col_values)
        print("Saving plots at ... ", savedir)
        savefig(fig, f"{savedir}/heatmap-pa_class-vs-decoder_class.pdf")

        for twind in list_twind_test:
            df = dfscores[dfscores["twind"] == twind].reset_index(drop=True)
            fig, axes = plot_subplots_heatmap(df, "pa_class", "decoder_class", "score", "pa_class_is_in_decoder", False, True,
                                row_values=row_values, col_values=col_values)
            savefig(fig, f"{savedir}/heatmap-pa_class-vs-decoder_class-twind={twind}-sub=pa_class_is_in_decoder.pdf")
            
            fig, axes = plot_subplots_heatmap(df, "pa_class", "decoder_class", "score", "decoder_class_good", False, True,
                                row_values=row_values, col_values=col_values)
            savefig(fig, f"{savedir}/heatmap-pa_class-vs-decoder_class-twind={twind}-sub=decoder_class_good.pdf")
            
            fig = sns.catplot(data=df, x="decoder_class", y="score", col="pa_class", col_wrap=6, alpha=0.2, 
                        jitter=True, hue="decoder_class_semantic_str")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-score-vs-decoder_class-twind={twind}-1.pdf")

            fig = sns.catplot(data=df, x="decoder_class", y="score", col="pa_class", col_wrap=6,
                    kind="bar", hue="decoder_class_semantic_str")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-score-vs-decoder_class-twind={twind}-2.pdf")

            fig = sns.catplot(data=df, x="decoder_class", y="score", hue="decoder_class_good", alpha=0.2, 
                        jitter=True)
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-decoder_class-twind={twind}-1.pdf")

            fig = sns.catplot(data=df, x="decoder_class", y="score", hue="decoder_class_good", kind="bar")
            rotateLabel(fig)
            savefig(fig, f"{savedir}/catplot-decoder_class-twind={twind}-2.pdf")

            plt.close("all")

        # Single scalar to summarize decoder for each class of data
        # - (same label) - (diff label)
        from pythonlib.tools.pandastools import summarize_featurediff
        dfsummary, _, _, _, _ = summarize_featurediff(dfscores, "same_class", [False, True], ["score"], 
                            ["decoder_class_good", "decoder_class_is_in_pa", "pa_class", "pa_class_is_in_decoder", "pa_idx", "trialcode", "twind"])

        fig = sns.catplot(data=dfsummary, x="pa_class", y="score-TrueminFalse", hue="decoder_class_good", alpha=0.2, jitter=True, col="twind")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.3)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/summary_same_min_diff-1.pdf")

        fig = sns.catplot(data=dfsummary, x="pa_class", y="score-TrueminFalse", hue="decoder_class_good", kind="bar", col="twind")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.3)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/summary_same_min_diff-2.pdf")

        # A single score summarizing, across all pa labels
        fig = sns.catplot(data=dfsummary, x="twind", y="score-TrueminFalse", kind="bar", col="decoder_class_good")
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.3)
        rotateLabel(fig)
        savefig(fig, f"{savedir}/overallsummary_same_min_diff.pdf")

        plt.close("all")

    return dfscores, PAtest

def pipeline_train_test_scalar_score(DFallpa, bregion, 
                                     var_train, event_train, twind_train, filterdict_train,
                                     var_test, event_test, list_twind_test, filterdict_test,
                                     savedir, include_null_data=False, decoder_method_index=None,
                                     prune_labels_exist_in_train_and_test=True, PLOT=True,
                                     which_level_train="trial", which_level_test="trial", n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=False):
    """
    Helper to extract dataframe holding decode score for each row of this bregion's test dataset (PA)
    
    WRitten when doing the Shape seuqence TI stuff.

    PARAMS:
    - subtract_baseline, bool, if True, then extracts data using twind in subtract_baseline_twind, and subtracts 
    the resulting scores from each twind in list_twind_test (ie each datapt) -- new column: score_min_base
    RETURNS:
    - (Mainly goal is saves plots and data)
    - dfscores, Dc, PAtrain, PAtest
    """
    from neuralmonkey.classes.population_mult import extract_single_pa
    import seaborn as sns
    from pythonlib.tools.pandastools import plot_subplots_heatmap
    from pythonlib.tools.snstools import rotateLabel

    ### Train decoder
    # include_null_data=True
    if n_min_per_var is None:
        n_min_per_var=3
    n_min_per_var_good = 10

    do_upsample_balance_fig_path_nosuff = f"{savedir}/upsample_pcs"
    PAtrain, Dc = train_decoder_helper(DFallpa, bregion, var_train, event_train, twind_train,
                                       PLOT, include_null_data, n_min_per_var, filterdict_train,
                                       which_level_train, decoder_method_index, savedir, n_min_per_var_good,
                                       do_upsample_balance, do_upsample_balance_fig_path_nosuff)
    
    dfscores, PAtest = test_decoder_helper(Dc, DFallpa, bregion, var_test, event_test, list_twind_test, filterdict_test,
                        which_level_test, savedir, prune_labels_exist_in_train_and_test, PLOT,
                        subtract_baseline, subtract_baseline_twind)
    
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
        "subtract_baseline_twind":subtract_baseline_twind
    }, path=f"{savedir}/params.txt")

    return dfscores, Dc, PAtrain, PAtest


def pipeline_train_test_scalar_score_mult_train_dataset(DFallpa, bregion, 
                                     list_train_dataset, list_var_train, 
                                     var_test, event_test, list_twind_test, filterdict_test, 
                                     which_level_test="trial",
                                     savedir=None, include_null_data=False, decoder_method_index=None,
                                     prune_labels_exist_in_train_and_test=True, PLOT=True, n_min_per_var=None,
                                     subtract_baseline=False, subtract_baseline_twind=None,
                                     do_upsample_balance=True, n_min_per_var_good=10):
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

    Dc = Decoder(pa_train_all, "var_train", _twind_train)
    do_upsample_balance_fig_path_nosuff = f"{savedir}/upsample_pcs"
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
                            subtract_baseline_twind)

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
        "subtract_baseline_twind":subtract_baseline_twind
    }, path=f"{savedir}/params.txt")

    return dfscores, Dc, PAtrain, PAtest


def analy_chars_score_postsamp(DFallpa, SAVEDIR):
    """
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
            
            # APPEND info related to trials.
            dflab = PAtest.Xlabels["trials"]
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

            dfscores["FEAT_num_strokes_beh"] = [dflab.iloc[pa_idx]["FEAT_num_strokes_beh"] for pa_idx in dfscores["pa_idx"]]
            dfscores["bregion"] = bregion


            # Normalize decode by subtracting mean within each decoder class
            from pythonlib.tools.pandastools import datamod_normalize_row_after_grouping_return_same_len_df
            dfscores, _, _ = datamod_normalize_row_after_grouping_return_same_len_df(dfscores, "decoder_class_was_drawn", 
                                                                                    ["decoder_class"], "score", False, True, True)


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

                for hue in [None, "FEAT_num_strokes_beh", "decoder_class", "decoder_class_good"]:
                    fig = sns.catplot(data=dfthis, x = "decoder_class_idx_in_shapes_drawn", y=var_score, 
                                    col="bregion", col_wrap=6,
                                        kind="point", errorbar=("ci", 68), hue = hue)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.5)
                    savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-hue={hue}-var_score={var_score}.pdf")
                    plt.close("all")

                import seaborn as sns

                for hue in [None, "FEAT_num_strokes_beh", "decoder_class", "decoder_class_good"]:
                    fig = sns.catplot(data=dfthis, x = "decoder_class_idx_in_shapes_drawn", y=var_score, 
                                    col="bregion", col_wrap=6,
                                        kind="point", errorbar=("ci", 68), hue = hue)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.5)
                    # savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-hue={hue}-var_score={var_score}.pdf")
                    # plt.close("all")

                import seaborn as sns
                for hue in [None, "FEAT_num_strokes_beh", "decoder_class"]:
                    fig = sns.catplot(data=dfthis, x = "decoder_class_was_drawn", y=var_score, 
                                    col="decoder_class_good", col_wrap=6,
                                        kind="point", errorbar=("ci", 68), hue = hue)
                    for ax in fig.axes.flatten():
                        ax.axhline(0, color="k", alpha=0.5)
                    # savefig(fig, f"{savedir}/catplot-decoder_class_idx_in_shapes_drawn-hue={hue}-var_score={var_score}.pdf")
                    # plt.close("all")

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

                row_var = "decoder_class_idx_in_shapes_drawn"
                col_var = "FEAT_num_strokes_beh"
                sub_var = "bregion"
                row_values = sorted(dfthis[row_var].unique())[::-1]
                col_values = sorted(dfthis[col_var].unique())
                fig, axes = plot_subplots_heatmap(dfthis, row_var, col_var, var_score, sub_var,
                                    annotate_heatmap=False, norm_method=norm_method, 
                                    row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                savefig(fig, f"{savedir}/heatmap-{row_var}-vs-{col_var}-var_score={var_score}.pdf")

                row_var = "decoder_class"
                col_var = "decoder_class_was_drawn"
                sub_var = "decoder_class_good"
                row_values = sorted(dfthis[row_var].unique())[::-1]
                col_values = sorted(dfthis[col_var].unique())
                fig, axes = plot_subplots_heatmap(dfthis, row_var, col_var, var_score, sub_var,
                                    annotate_heatmap=False, norm_method=norm_method, 
                                    row_values=row_values, col_values=col_values, ZLIMS=zlims, share_zlim=True, W=6, diverge=diverge)
                savefig(fig, f"{savedir}/heatmap-{row_var}-vs-{col_var}-var_score={var_score}.pdf")

                plt.close("all")
