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


class Decoder():
    def __init__(self, PAtrain, var_decode, twind_train):
        """
        Holds a single PA and decoder instance, which comprises variable,
        twindow for decoder training, and all other aspects of preprocessing 
        relevant for training a single deceoer istance. 
        For decoer variations, just make multiple Decoder() instances.
        """
        self.PAtrain = PAtrain
        
        self.VarDecode = var_decode
        self.Params = {
            "twind_train":twind_train
        }

    def train_decoder(self, PLOT=False):
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


        pa = self.PAtrain
        var_decode = self.VarDecode
        twind_train = self.Params["twind_train"]

        ### Prepare training data
        # Avg over time --> vector
        pathis = pa.slice_by_dim_values_wrapper("times", twind_train) 
        pathis = pathis.agg_wrapper("times")

        X = pathis.X.squeeze().T # (ntrials, nchans)
        times = pathis.Times
        dflab = pathis.Xlabels["trials"]
        labels = dflab[var_decode].tolist()

        if False:
            # Stack presamp (label="presamp") and postsamp (label="shape X")
            twind = [-0.7, -0.1]
            pathis_presamp = pa.slice_by_dim_values_wrapper("times", twind)
            pathis_presamp = pathis_presamp.agg_wrapper("times")

            X_presamp = pathis_presamp.X.squeeze().T

            # And add a label (pre-samp) by concatenating to post-samp data
            X = np.concatenate([X, X_presamp], axis=0)
            _labels = ["presamp" for _ in range(X_presamp.shape[0])]
            labels += _labels

        print(X.shape)
        print(len(labels))

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
            heatmap_mat(probs, ax = ax, annotate_heatmap=False);

            fig, ax = plt.subplots()
            probs_max = np.max(probs, axis=1)
            ax.hist(probs_max, bins=20);

        # SAVE RESULTS
        self.Classifier = clf
        self.MultiLabelBinarizer = mlb

        # for i, lab in enumerate(self.MultiLabelBinarizer):
        #     [i] = lab
        
    def plot_single_trial(self, indtrial, PA=None, tbin_dur=0.15, tbin_slide=0.01):
        """
        Plot timecourse of decode for an example trial
        PARAMS:
        - tbin_dur, in sec, window duration for smoothing. Note: use >=0.1 to avoid high noisiness
        - tbin_slide, in sec, for smoothing
        """
        from pythonlib.tools.plottools import legend_add_manual, makeColors

        if PA is None:
            PA = self.PAtrain

        clf = self.Classifier
        mlb = self.MultiLabelBinarizer

        # Smooth the data
        pathis = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)

        # Prepare plot colors, for each label.
        pcols = makeColors(len(mlb.classes_))
        map_lab_to_col = {}
        for cl, col in zip(mlb.classes_, pcols):
            map_lab_to_col[cl] = col
        if False:
            # Color by novelty
            dflab[dflab["shape_is_novel_all"]==True].index.tolist()
            pcols = makeColors(3)

            map_lab_to_col = {}
            for sh, nov in dflab.loc[:, ["seqc_0_shape", "shape_is_novel_all"]].values:
                if nov:
                    pcolthis =  pcols[0]
                else:
                    pcolthis = pcols[1]
                
                if sh not in map_lab_to_col:
                    map_lab_to_col[sh] = pcolthis
                else:
                    assert np.all(map_lab_to_col[sh] == pcolthis)

            map_lab_to_col["presamp"] = pcols[2]
        
        # Plot timecourse of decode
        x = pathis.X[:, indtrial, :].T # (ntimes, nchans)
        probs_mat = clf.predict_proba(x) # (ntimes, nlabels)
        times = pathis.Times

        fig, axes = plt.subplots(2,1, figsize=(12, 6))

        ax = axes.flatten()[0]
        for i, lab in enumerate(mlb.classes_):
            probs = probs_mat[:, i]
            col = map_lab_to_col[lab]
            ax.plot(times, probs, label=lab, color=col)
        ax.axvline(0, color="k")

        title = pathis.Xlabels["trials"].iloc[indtrial]["seqc_0_shape"]
        ax.set_title(f"seqc_0_shape: {title}")

        # ax.legend(loc="best")

        ax = axes.flatten()[1]
        legend_add_manual(ax, map_lab_to_col.keys(), map_lab_to_col.values())

    def scalar_score_twinds_trials(self, list_twind, PA=None, tbin_dur=0.15, tbin_slide=0.01, PLOT=True):
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

        pathis = PA.agg_by_time_windows_binned(tbin_dur, tbin_slide)
        clf = self.Classifier

        ntrials = len(pathis.Trials)
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
                probs_vec = np.mean(probs_mat[inds, :], axis=0)

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
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*12))

            for ax, ind_twind in zip(axes.flatten(), range(len(list_twind))):
                heatmap_mat(scores[:, :, ind_twind], ax, False)    
                ax.set_xlabel("class")
                ax.set_ylabel("trial")
                ax.set_title(f"twind: {list_twind[ind_twind]}")

        return scores

    def scalar_score_twinds_trialgroupings(self, vars_trial, list_twind, PA=None, tbin_dur=0.15, tbin_slide=0.01, PLOT=True):
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
        """
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items_good
        if PA is None:
            PA = self.PAtrain

        # First, get score for each trial
        scores = self.scalar_score_twinds_trials(list_twind, PA, tbin_dur, tbin_slide, PLOT)

        # Second, group trials
        dflab = PA.Xlabels["trials"]
        grpdict = grouping_append_and_return_inner_items_good(dflab, vars_trial, sort_keys=True)

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

            ncols = 3
            nrows = int(np.ceil(len(list_twind)/ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))

            for ax, ind_twind in zip(axes.flatten(), range(len(list_twind))):
                heatmap_mat(resthis[:, :, ind_twind], ax, False)    
                ax.set_xlabel("decoded class label")
                ax.set_ylabel("trial group label")
                ax.set_title(f"twind: {list_twind[ind_twind]}")

        return resthis
