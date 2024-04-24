import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralmonkey.neuralplots.population import plotNeurHeat, plotNeurTimecourse


class PopAnal():
    """ for analysis of population state spaces
    """

    def __init__(self, X, times=None, chans=None, dim_units=0,
        stack_trials_ver="append_nan", 
        feature_list = None, spike_trains=None,
        print_shape_confirmation = False, trials=None):
        """ 
        Options for X:
        - array, where one dimensions is nunits. by dfefualt that should 
        be dim 0. if not, then tell me in dim_units which it is.
        - list, length trials, each element (nunits, timebins).
        can have diff timebine, but shoudl have same nunits. 
        Useful if, e.g., each trial different length, and want to 
        preprocess in standard ways to make them all one stack.
        - pandas dataframe, where each row is a trial. must have a column called
        "neur" which is (nunits, timebins)
        - times,  array of time values for each time bin. same for all trials.
        --------
        - axislabels = list of strings
        - chans, list of ids for each chan, same len as nunits. if None, then labels them [0, 1, ..]
        - dim_units, dimeision holding units. if is not
        0 then will transpose to do all analyses.
        - feature_list = list of strings, each identifying one column for X if 
        X is datafarme, where these are features already extracted in X, useful for 
        downstream analysis.
        - spike_trains, list of list of spike trains, where the outer list is 
        trials, and inner list is one for each chan.
        ATTRIBUTES:
        - X, (nunits, cond1, cond2, ...). NOTE, can enter different
        shape, but tell me in dim_units which dimension is units. will then
        reorder. *(nunits, ntrials, time)

        """
        import copy

        self.Params = {}
        self.Xdataframe = None
        self.Xz = None

        self.dim_units = dim_units
        if isinstance(X, list):
            from ..tools.timeseriestools import stackTrials
            self.X = stackTrials(X, ver=stack_trials_ver)
            assert dim_units==1, "are you sure? usually output of stackTrials puts trials as dim0."
        elif isinstance(X, pd.core.frame.DataFrame):
            self.Xdataframe = X
            try:
                self.X = np.stack(X["neur"].values) # (ntrials, nunits, time)
                self.dim_units = 1
            except:
                print("assuming you gave me dataframe where each trial has diff length neural.")
                print("running 'stackTrials ...")
                X = [x for x in X["neur"].values] # list, where each elemnent is (nunits, tbins)
                self.X = stackTrials(X, ver=stack_trials_ver)
                assert dim_units==1, "are you sure? usually output of stackTrials puts trials as dim0."

            self.Featurelist = feature_list
        else:
            if len(X.shape)==2:
                # assume it is (nunits, timebins). unsqueeze so is (nunits, 1, timebins)
                self.X = np.expand_dims(X, 1)
            else:
                assert len(X.shape)==3
                self.X = X
        
        self.Saved = {}
        if times is None:
            # just use dummy indices
            self.Times = np.arange(self.X.shape[2])
        else:
            # assert isinstance(times, list), "or something list-like? change code?"
            if isinstance(times, (list, tuple)) and not isinstance(times[0], str):
                times = np.array(times)
            if isinstance(times, np.ndarray) and len(times.shape)>1:
                times = times.squeeze(axis=1)
                assert len(times.shape)==1
            if not len(times)==self.X.shape[2]:
                assert False
            self.Times = times
        # print("HERERE", times, len(times))

        if chans is None:
            self.Chans = range(self.X.shape[0])
        else:
            self.Chans = copy.copy(chans)

        # Spike trains
        self.SpikeTrains = spike_trains
        if spike_trains is not None:    
            assert len(spike_trains)==self.X.shape[1], "doesnt match num trials"
            for st in spike_trains:
                assert len(st)==self.X.shape[0], "doesnt match number chans"

        self.preprocess()
        if print_shape_confirmation:
            print("Final shape of self.X; confirm that is (nunits, ntrials, time)")
            print(self.X.shape)

        if trials is None:
            self.Trials = list(range(self.X.shape[1]))
        else:
            self.Trials = copy.copy(trials)

        # Initialize dataframe holding dimension labels
        self.Xlabels = {}
        self.Xlabels["trials"] = pd.DataFrame()
        self.Xlabels["chans"] = pd.DataFrame()
        self.Xlabels["times"] = pd.DataFrame()
        
        # Final sanity check
        assert len(self.Chans)==self.X.shape[0]
        assert len(self.Trials)==self.X.shape[1]
        assert len(self.Times)==self.X.shape[2]


    def preprocess(self):
        """ preprocess X, mainly so units dimension is axis 0 
        """

        if self.dim_units!=0:
            self.X = np.swapaxes(self.X, 0, self.dim_units)

    def unpreprocess(self, Xin):
        """ undoes preprocess, useful if return values,
        does not chagne anythings in this object, just Xin."""
        if self.dim_units!=0:
            Xin = np.swapaxes(Xin, 0, self.dim_units)

        return Xin

    def sortPop(self, dim, ver="trial_length"):
        """ sort self.X --> self.Xsorted.
        - dim, which dimension to sort by. 
        0, neurons; 1, trials
        - ver, is shortcut for filter functions.
        """
        from ..tools.nptools import sortPop

        def getFilt(ver="trial_length"):
            """ returns lambda function that can be used as filt in "sortPop"
            """
            if ver=="trial_length":
                # sort by incresauibng trial duration. 
                # assumes trials are concatenated ysing "append_nan" method
                # in PopAnal. returns index of first occurance of nan.
        #         return lambda x: print(x.shape)
                # assume (<>, <>, time), where time is 
                def filt(x):
                    idx = np.where(np.isnan(x[0]))[0]
                    if len(idx)>0:
                        # then nans exist
                        return idx[0]
                    else:
                        # then return number which is one larger than what 
                        # wuld return if last index was nan.
                        return len(x[0])
                return filt
            else:
                assert False, "not coded"

        filt = getFilt(ver)
        self.Xsorted = sortPop(self.X, dim=dim, filt=filt)
        print(f"shape of self.X: {self.X.shape}")
        print(f"shape of self.Xsorted: {self.Xsorted.shape}")        


    def centerAndStack(self):
        """ convert to (nunits, -1), 
        and center each row.
        """
        X = self.X.copy()
        # - reshape to N x tiembins
        X = np.reshape(X, (X.shape[0],-1))
        # - demean
        means = np.mean(X, axis=1)[:, None]
        X = X - means
        
        self.Xcentered = X
        self.Saved["centerAndStack_means"] = means # if want to apply same trnasformation in future data.

    def pca(self, ver="svd", ploton=False):
        """ perform pca
        - saves in cache the axes, self.Saved["pca"]
        """
        
        self.centerAndStack() # (nchans, ..)

        if ver=="svd":

            if self.Xcentered.shape[1]>self.Xcentered.shape[0]:
                # "dim 1 is time x trials..., shoudl be larger"
                full_matrices=False
            else:
                full_matrices=True

            u, s, v = np.linalg.svd(self.Xcentered, full_matrices=full_matrices)
            nunits = self.X.shape[0]
            w = s**2/(nunits-1)
            
        elif ver=="eig":
            Xcov = np.cov(self.Xcentered)
            w, u = np.linalg.eig(Xcov)

            # - plot
            if False:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1,2, figsize=(10,5))
                axes[0].imshow(Xcov);
                axes[1].hist(Xcov[:]);
                axes[1].set_title("elements in cov mat")

                # - plot 
                fig, axes = plt.subplots(1, 2, figsize=(15,5))

                axes[0].plot(w, '-ok')
                wsum = np.cumsum(w)/np.sum(w)
                axes[0].plot(wsum, '-or')
                axes[0].set_title('eigenvals')
                axes[1].imshow(v)
                axes[1].set_title('eigen vects')
                axes[1].set_xlabel('vect')
        else:
            assert False

        w = w/np.sum(w)

        if ploton:
            import matplotlib.pyplot as plt
            # - plot 
            fig, axes = plt.subplots(1, 2, figsize=(15,5))

            axes[0].plot(w, '-ok')
            axes[1].plot(np.cumsum(w)/np.sum(w), '-or')
            axes[1].set_title('cumulative variance explained')
            axes[1].hlines(0.9, 0, len(w))

            axes[0].set_title('s vals')
            # axes[1].imshow(v)
            # axes[1].set_title('eigen vects')
            # axes[1].set_xlabel('vect')
        else:
            fig = None
        
        # === save
        self.Saved["pca"]={"w":w, "u":u}
        
        return fig


    def pca_make_space(self, trial_agg_method, trial_agg_grouping,
        time_agg_method=None,
        norm_subtract_condition_invariant=False,
        ploton=True):
        """ Prperocess data (e.g,, grouping by trial and time) and then
        Make a PopAnal object holding (i) data for PCA and (ii) the results of
        PCA.
        PARAMS:
        - PA, popanal object, holds all data.
        - DF, dataframe, with one column for each categorical variable you care about (in DATAPLOT_GROUPING_VARS).
        The len(DF) must equal num trials in PA (asserts this)
        - trial_agg_grouping, list of str defining how to group trials, e.g,
        ["shape_oriented", "gridloc"]
        - norm_subtract_condition_invariant, bool, if True, then at each timepoint subtracts
        mean FR across trials
        RETURNS:
        - PApca, a popanal holding the data that went into PCA, and the results of PCA,
        and methods to project any new data to this space.
        """

        # assert DF==None, "instead, put this in self.Xlabels"

        # First, decide whether to take mean over some way of grouping trials
        if trial_agg_method==None:
            # Then dont aggregate by trials
            PApca = self.copy()
        elif trial_agg_method=="grouptrials":
            # Then take mean over trials, after grouping, so shape
            # output is (nchans, ngrps, time), where ngrps < ntrials
            DF = self.Xlabels["trials"]
            if False:
                groupdict = grouping_append_and_return_inner_items(DF, trial_agg_grouping)
                # groupdict = DS.grouping_append_and_return_inner_items(trial_agg_grouping)
                PApca = self.agg_by_trialgrouping(groupdict)
            else:
                # Better, since it retains Xlabels
                PApca = self.slice_and_agg_wrapper("trials", trial_agg_grouping)
        else:
            print(trial_agg_method)
            assert False

        # First, whether to subtract mean FR at each timepoint
        if norm_subtract_condition_invariant:
            PApca = PApca.norm_subtract_condition_invariant()

        # second, whether to agg by time (optional). e..g, take mean over time
        if time_agg_method=="mean":
            PApca = PApca.agg_wrapper("times")
            # PApca = PApca.mean_over_time(return_as_popanal=True)
        else:
            assert time_agg_method==None

        print("Shape of data going into PCA (chans, trials, times):", PApca.X.shape)
        fig = PApca.pca("svd", ploton=ploton)

        return PApca, fig


    # def reproject1(self, Ndim=3):
    #     """ reprojects neural pop onto subspace.
    #     uses axes defined by ver. check if saved if 
    #     not then autoamitcalyl extracst those axes
    #     - Ndim is num axes to take."""
        
    #     maxdim = self.X.shape[0] # max number of neurons
    #     # if Ndim>maxdim:
    #     #     print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
    #     #     print(f"reducing Ndim to {maxdim}")
    #     #     Ndim = min((maxdim, Ndim))

    #     # # - get old saved
    #     # if "pca" not in self.Saved:
    #     #     print(f"- running {ver} for first time")
    #     #     self.pca(ver="eig")

    #     if True:
    #         w = self.Saved["pca"]["w"]
    #         u = self.Saved["pca"]["u"]
                
    #         # - project data onto eigen
    #         usub = u[:,:Ndim]
    #         Xsub = usub.T @ self.Xcentered

    #         # - reshape back to (nunits, ..., ...)
    #         sh = list(self.X.shape)
    #         sh[0] = Ndim
    #         # print(self.X.shape)
    #         # print(Ndim)
    #         # print(Xsub.shape)
    #         # print(self.Xcentered.shape)
    #         # print(usub.T.shape)
    #         # print(u.shape)
    #         # print(u.)
    #         Xsub = np.reshape(Xsub, sh)
    #         # Ysub = Ysub.transpose((1, 0, 2))
    #     else:
    #         Xsub = self.reprojectInput(self.X, Ndim)


    #     # -- return with units in the correct axis
    #     return self.unpreprocess(Xsub)


    def reproject(self, Ndim=3):
        """ reprojects neural pop onto subspace.
        uses axes defined by ver. check if saved if 
        not then autoamitcalyl extracst those axes
        - Ndim is num axes to take."""
        
        # maxdim = self.X.shape[0] # max number of neurons
        # if Ndim>maxdim:
        #     print(f"not enough actual neurons ({maxdim}) to match desired Ndim ({Ndim})")
        #     print(f"reducing Ndim to {maxdim}")
        #     Ndim = min((maxdim, Ndim))

        # # - get old saved
        # if "pca" not in self.Saved:
        #     print(f"- running {ver} for first time")
        #     self.pca(ver="eig")

        Xsub = self.reprojectInput(self.X, Ndim)

        # -- return with units in the correct axis
        return self.unpreprocess(Xsub)

    def reprojectInput(self, X, Ndim=3, Dimslist = None):
        """ same as reproject, but here project activity passed in 
        X.
        - X, (nunits, *), as many dim as wanted, as long as first dim is nunits (see reproject
        for how to deal if first dim is not nunits)
        - Dimslist, insteast of Ndim, can give me list of dims. Must leave Ndim None.
        RETURNS:
        - Xsub, (Ndim, *) [X not modified]
        NOTE: 
        - This also applis centering tranformation to X using the saved mean of data
        used to compute the PCA space.
        """

        assert "pca" in self.Saved, "need to first run self.pca()"
        nunits = X.shape[0]
        assert nunits==self.X.shape[0]
        sh = list(X.shape) # save original shape.

        if Dimslist is not None:
            assert Ndim is None, "choose wiether to take the top N dim (Ndim) or to take specific dims (Dimslist)"
            numdims = len(Dimslist)
        else:
            assert Ndim is not None
            numdims = Ndim

        if numdims>nunits:
            print(f"not enough actual neurons ({nunits}) to match desired num dims ({numdims})")
            assert False
            # print(f"reducing Ndim to {nunits}")
            # Ndim = min((nunits, Ndim))

        # - get old saved
        # if "pca" not in self.Saved:
        #     print(f"- running {ver} for first time")
        #     self.pca(ver="eig")

        # 1) center and stack
        # X = X.copy()
        # - reshape to N x tiembins
        X = np.reshape(X, (nunits,-1)) # (nunits, else)
        # - demean
        X = X - self.Saved["centerAndStack_means"] # demean

        # 2) 
        # w = self.Saved["pca"]["w"]
        u = self.Saved["pca"]["u"]
        if Ndim is None:
            usub = u[:,Dimslist]
        else:
            usub = u[:,:Ndim]
        Xsub = usub.T @ X

        # 3) reshape back to origuinal
        sh[0] = numdims
        Xsub = np.reshape(Xsub, sh)

        return Xsub

    ### ANALYSIS
    def dataframeByTrial(self, dim_trial = 1, columns_to_add = None):
        """ useful preprocess before do analyses.
        converts X to dataframe, where row is trial.
        - columns_to_add, dict, where each item is new column.
        entry must be same length as num trials.
        - dim_trial, which dim is trialnum?
        """
        assert False, "[DEPRECATED] - isntead, pass in list of dicts to PopAnal directly, this ensures keeps all fields in the dicts"
        ntrials = self.X.shape[dim_trial]


        assert dim_trial==1, "below code assumes 1. indexing is not as flexible"
        dat = []
        for i in range(ntrials):
            dat.append({
                "x":self.X[:, i, ...],
                "trial":i
                })
        df = pd.DataFrame(dat)

        if columns_to_add is not None:
            for k, v in columns_to_add.items():
                assert len(v)==ntrials
                df[k] = v

        return df

    # Data Transformations
    def zscoreFr(self, groupby=None):
        """ z-score firing rates using across trial mean and std.
        - groupby, what mean and std to use. if [], then all trials
        combined (so all use same mean and std). if ["circ_binned"], 
        then asks for each trial what its bin (for circuiliatry) and then
        uses mean and std within that bin. Must have binned data first 
        before this (see binFeatures)
        ==== RETURN:
        modifies self.Xdataframe, adds column "neur_z"
        """
        from pythonlib.tools.pandastools import applyFunctionToAllRows
        if groupby is None:
            groupby = []
        # 1. get mean and std.
        _, colname_std = self.aggregate(groupby, "trial", "std", "std", return_new_col_name=True)
        _, colname_mean = self.aggregate(groupby, "trial", "mean", "mean", return_new_col_name=True)

        # 2. take zscore
        def F(x):
            # returns (neur, time) fr in z-score units
            return (x["neur"] - x[colname_mean])/x[colname_std]

        self.Xdataframe = applyFunctionToAllRows(self.Xdataframe, F, newcolname="neur_z")

    def zscoreFrNotDataframe(self):
        """ z-score across trials and time bins, separately for each chan
        RETURNS:
        - modifies self.Xz
        - return self.Xz
        """

        X = self.X

        # reshape to (nchans, trials*time)
        x = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
        xstd = np.std(x, axis=1)
        xmean = np.mean(x, axis=1)
        xstd = xstd.reshape(xstd.shape[0], 1, 1)
        xmean = xmean.reshape(xmean.shape[0], 1, 1)

        self.Xz = (X - xmean)/xstd
        return self.Xz


    def binFeatures(self, nbins, feature_list=None):
        """ assign bin to each trial, based on its value for feature
        in feature_list.
        - nbins, int, will bin by percentile (can modify to do uniform)
        - feature_list, if None, then uses self.Featurelist. otherwise give
        list of strings, but these must already be columnes in self.Xdataframe
        === RETURNS
        self.Xdataframe modifed to have new columns, e..g, <feat>_bin
        """
        if feature_list is None:
            feature_list = self.Featurelist

        for k in feature_list:
            kbin = self.binColumn(k, nbins)


    def binColumn(self, col_to_bin, nbins):
        """ assign bin value to each trial, by binning based on
        scalar variable.
        - modifies in place P.Xdataframe, appends new column.
        """
        from ..tools.pandastools import binColumn
        new_col_name = binColumn(self.Xdataframe, col_to_bin=col_to_bin, nbins=nbins)
        return new_col_name

    def aggregate(self, groupby, axis, agg_method="mean", new_col_suffix="agg", 
        force_redo=False, return_new_col_name = False, fr_use_this = "raw"):
        """ get aggregate pop activity in flexible ways.
        - groupby is how to group trials (rows of self.Xdataframe). 
        e.g., if groupby = [], then combines all trials into one aggreg.
        e.g., if groupby = "circ_binned", then groups by bin.
        e.g., if groupby = ["circ_binned", "angle_binned"], then conjunction
        - axis, how to take mean (after stacking all trials after grouping)
        e.g., if string, then is shortcut.
        e.g., if number then is axis.
        - new_col_suffix, for new col name.
        - agg_method, how to agg. could be string, or could be function.
        - fr_use_this, whether to use raw or z-scored (already done).
        RETURNS:
        - modifies self.Xdataframe, with one new column, means repopulated.
        - also the aggregated datagframe XdataframeAgg
        """
        from ..tools.pandastools import aggregThenReassignToNewColumn
        if isinstance(axis, str):
            # shortcuts
            if axis=="trial":
                # average over trials.
                axis = 0
            else:
                assert False, "not coded"

        if isinstance(agg_method, str):
            def F(x):
                if fr_use_this=="raw":
                    X = np.stack(x["neur"].values)
                elif fr_use_this=="z":
                    X = np.stack(x["neur_z"].values)
                # expect X to be (ntrials, nunits, time)
                # make axis accordinyl.
                if agg_method=="mean":
                    Xagg = np.mean(X, axis=axis)
                elif agg_method=="std":
                    Xagg = np.std(X, axis=axis)
                elif agg_method=="sem":
                    Xagg = np.std(X, axis=axis)/np.sqrt(X.shape[0])
                else:
                    assert False, "not coded"
                return Xagg
        else:
            F = agg_method
            assert callable(F)

        if len(groupby)==0:
            new_col_name = f"alltrials_{new_col_suffix}"
        elif isinstance(groupby, str):
            new_col_name = f"{groupby}_{new_col_suffix}"
        else:
            new_col_name = f"{'_'.join(groupby)}_{new_col_suffix}"

        # if already done, don't run
        if new_col_name in self.Xdataframe.columns:
            if not force_redo:
                print(new_col_name)
                assert False, "this colname already exists, force overwrite if actualyl want to run again."

        [self.Xdataframe, XdataframeAgg] = aggregThenReassignToNewColumn(self.Xdataframe, F, 
            groupby, new_col_name, return_grouped_df=True)

        if return_new_col_name:
            return XdataframeAgg, new_col_name
        else:
            return XdataframeAgg

    ####################### SLICING
    def slice_by_dim_values_wrapper(self, dim, values, time_keep_only_within_window=True):
        """ Slice based on values (not indices), works for dim =
        times, trials, or chans.
        PARAMS:
        - dim, str (see slice_by_dim_indices_wrapper)
        - values, list of values into self.Trials or Chans (depending on dim).
        Asserts that self.Trials or Chans doesnt contain any Nones
        -- if dim=="times", then values are the min and max of the window
        """
        if dim in ["chans", "trials"]:
            # 1) Map the values to indices
            # dim, dim_str = self.help_get_dimensions(dim)
            indices = self.index_find_these_values(dim, values)
        elif dim=="times":
            # values are [t1, t2]
            assert len(values)==2
            assert values[1]>values[0]
            indices = self.index_find_this_time_window(values, time_keep_only_within_window=time_keep_only_within_window)
            # indices = self.index_find_these_values(dim, values)
            assert len(indices)==2
        else:
            assert False

        # 2) Call indices version
        return self.slice_by_dim_indices_wrapper(dim, indices)

    # def slice_time_by_indices(self, ind1, ind2):
    #     """ This is actually doing what slice_by_dim_indices_wrapper is supposed to do
    #     for time, gets indices in time dimension, inclusive of ind1 and ind2. i.e
    #     is like ind1:ind2 (inclusive). Can use -1 for last index, etc.
    #     RETURNS:
    #     - PopAnal
    #     """
    #
    #     # convert from indices to times
    #     time_wind = [self.Times[ind1], self.Times[ind2]]
    #     return self.slice_by_dim_indices_wrapper("times", time_wind)
    #
    # def bin_time(self):


    def slice_by_dim_indices_wrapper(self, dim, inds, reset_trial_indices=False):
        """ Helper that is standard across the three dims (trials, times, chans),
        to use indices (i.e., not chans, but indices into chans)
        PARAMS:
        - dim, either int or str
        - inds: indices into self.Chans or self.Trials, or self.Times.
        (is index, NOT the value itself). if dim=="times", then inds must be len 2,
        the start and ending times, for window, inclusive.
        - reset_trial_indices, bool, if True, then new PA will have trials 0, 1, 2, ...
        regardless of initial trials. Useful if trials are just playibg role of index, with
        no meaning, and meaningful trial indiices are stored in PA.Xlabels["trials"]["trialcode"]
        RETURNS:
        - PopAnal object, , copy
        - OR frmat array (chans, trials, times) if return_only_X==True
        """

        # convert to int
        dim, dim_str = self.help_get_dimensions(dim)

        if dim_str=="times":
            assert len(inds)==2
            t1 = self.Times[inds[0]]
            t2 = self.Times[inds[1]]

            # make times slgitly wider, to ensure get inclusive indices. This solves
            # problems if numerical imprecision leading to variable output sizes.
            if inds[0]==0:
                t1 = t1-1
            else:
                t1_prev = self.Times[inds[0]-1]
                t1 -= (t1-t1_prev)/2

            if inds[1]==-1 or inds[1]==len(self.Times)-1:
                t2 = t2+1
            else:
                t2_next = self.Times[inds[1]+1]
                t2 += (t2_next - t2)/2

            pa = self._slice_by_time_window(t1, t2, True, False)

            if len(self.Xlabels["times"])>0:
                # then slice it
                assert False, "code it"
                # dfnew = self.Xlabels["trials"].iloc[inds].reset_index(True)
            else:
                dfnew = self.Xlabels["times"].copy()

            # assert not isinstance(inds[0], int), "num, not int"
            # t1 = inds[0]
            # t2 = inds[1]
            # pa = self._slice_by_time_window(t1, t2, True, True)
            # if len(self.Xlabels["times"])>0:
            #     # then slice it
            #     assert False, "code it"
            #     # dfnew = self.Xlabels["trials"].iloc[inds].reset_index(True)
            # else:
            #     dfnew = self.Xlabels["times"].copy()
        elif dim_str=="chans":
            pa = self._slice_by_chan(inds, return_as_popanal=True, 
                chan_inputed_row_index=True)
            if len(self.Xlabels["chans"])>0:
                # then slice it
                dfnew = self.Xlabels["chans"].iloc[inds].reset_index(drop=True).copy()
            else:
                dfnew = pd.DataFrame()
        elif dim_str=="trials":
            pa = self._slice_by_trial(inds, return_as_popanal=True)
            if len(self.Xlabels["trials"])>0:
                # then slice it
                dfnew = self.Xlabels["trials"].iloc[inds].reset_index(drop=True).copy()
            else:
                dfnew = pd.DataFrame()
        else:
            print(dim_str)
            assert False

        # if True:
        # Retain the labels for the dimensions that are untouched
        # pa.Xlabels = self.Xlabels.copy()
        pa.Xlabels = {dim:df.copy() for dim, df in self.Xlabels.items()}
        # subsample the dimensions you have sliced
        pa.Xlabels[dim_str] = dfnew
        # else:
        #     # copy all dimensions
        #     pa.Xlabels = {dim:df.copy() for dim, df in self.Xlabels.items()}

        if reset_trial_indices:
            assert dim_str=="trials", "this doesnt make sense otherwise. mistake?"
            pa.Trials = list(range(pa.X.shape[1]))

        return pa


    def _slice_by_time_window(self, t1, t2, return_as_popanal=False,
            fail_if_times_outside_existing=True, version="raw", 
            subtract_this_from_times = None,
            method_if_not_enough_time="keep_and_prune_time"):
        """ Slice population by time window, where
        time is based on self.Times
        PARAMS;
        - t1, t2, start and end time for slicing, inclusive
        - fail_if_times_outside_existing, bool, if True, then self.Times must have times
        before t1 and after t2 (i.e., t1 and t2 are within range of data), otherwise raoises
        NotEnoughDataException. if False, returns whatever exists within time window..
        - subtract_this_from_times, scalar, will subtract from times (to recenter). or None
        does nothing.
        RETURNS:
        - np array, (nchans, ntrials, timesliced)
        - OR None, if not enough data, and fail_if_times_outside_existing==True
        """

        if sum(self.Times<=t1)==0 or sum(self.Times>=t2)==0:
            # Not enough time data.
            if fail_if_times_outside_existing:
                # Then throw error
                print("asking for times outside data range; (min, max that exists, t1, t2):", min(self.Times), max(self.Times), t1, t2)
                from pythonlib.tools.exceptions import NotEnoughDataException
                raise NotEnoughDataException
            else:
                # Then silently deal with it.
                if method_if_not_enough_time=="keep_and_prune_time":
                    # Then keep data, and just purne the time window
                    pass
                elif method_if_not_enough_time=="return_none":
                    # Then abort, and return Nones
                    if return_as_popanal:
                        return None
                    else:
                        return None, None
                else:
                    print(method_if_not_enough_time)
                    assert False, "code it"

        if not isinstance(self.Times, list):
            self.Times = np.array(self.Times)
        X = self.extract_activity_copy(version=version)
        inds = (self.Times>=t1) & (self.Times<=t2)
        # print(sum(inds))

        if sum(inds)==0:
            print(inds)
            print(self.Times)
            print(t1, t2)
            print("must give times that have data within them!!")
            assert False

        try:
            x_windowed = X[:, :, inds]
        except Exception as err:
            print(inds)
            print(X.shape)
            raise err
        times = np.array(self.Times[inds])

        if subtract_this_from_times:
            times = times - subtract_this_from_times

        if return_as_popanal:
            PA = PopAnal(x_windowed, times, chans=self.Chans, 
                trials = self.Trials, print_shape_confirmation=False)
            return PA
        else:
            return x_windowed, times

    def _slice_by_trial(self, inds, version="raw", return_as_popanal=False):
        """ Slice activity to only get these trials, returned as popanal
        if return_as_popanal is True
        PARAMS:
        - inds, list of ints, indices into dim 1 of self.X
        NOTE: inds are not the trials themselves, rather indices into Trials.
        """

        # Collect data
        X = self.extract_activity_copy(version=version)
        X = X[:, inds, :]
        trials_actual = [self.Trials[i] for i in inds]

        # Generate new popanal
        if return_as_popanal:
            PA = PopAnal(X, times=self.Times, chans=self.Chans,
                trials = trials_actual, print_shape_confirmation=False)
            return PA
        else:
            return X

    def _slice_by_chan(self, chans, version="raw", return_as_popanal=True, 
            chan_inputed_row_index=False):
        """ Slice data to keep only subset of channels
        PARAMS;
        - chans, list of chan labels, These are NOT the row indices, but are instead
        the chan labels in self.Chans. To use row indices (0,1,2, ...), make
        chan_inputed_row_index=True. 
        (NOTE: will be sorted as in chans)
        RETURNS:
        - EIther:
        --- X, np array, shape
        --- PopAnal object (if return_as_popanal)
        """

        # convert from channel labels to row indices
        if chan_inputed_row_index:
            inds = chans
            del chans
        else:
            inds = [self.index_find_this_chan(ch) for ch in chans]
            del chans

        # Slice
        X = self.extract_activity_copy(version=version)
        X = X[inds, :, :]

        if return_as_popanal:
            chans_actual = [self.Chans[i] for i in inds]
            PA = PopAnal(X, times=self.Times, trials=self.Trials, chans=chans_actual)
            return PA
        else:
            return X

    def copy_replacing_X(self, X, times=None, chans=None):
        """
        Retyrns a copy, with a new X that has same n trials (so that can copy the labels from self,
        self.Xlabels), but potentialyl diff n chans and times.
        NOTE: if X has any dimensions matching self.X, then assumes the labels are the same for that dim,
        (if that input (e.g., times) is NOne)
        :return:
        """

        assert X.shape[1]==self.X.shape[1], "assumes trials are matches, to get labels"
        trials = self.Trials

        if chans is None and (X.shape[0]==self.X.shape[0]):
            chans = self.Chans

        if times is None and (X.shape[2]==self.X.shape[2]):
            times = self.Times

        pa = PopAnal(X, times=times, chans=chans, trials=trials)

        # pa.Xlabels = {dim:df.copy() for dim, df in self.Xlabels.items()}
        pa.Xlabels = {}
        pa.Xlabels["trials"] = self.Xlabels["trials"].copy()
        pa.Xlabels["chans"] = pd.DataFrame()
        pa.Xlabels["times"] = pd.DataFrame()

        return pa

    def copy(self):
        """ Returns a copy.
        """
        # trials = range(self.X.shape[1])
        inds = list(range(len(self.Chans)))
        return self.slice_by_dim_indices_wrapper("chans", inds)
        # return self._slice_by_trial(trials, return_as_popanal=True)

    def mean_over_time(self, version="raw", return_as_popanal=False):
        """ Return X, but mean over time,
        shape (nchans, ntrials)
        """
        assert False, "OBSOLETE. use agg_wrapper, it retains Xlabels"
        X = self.extract_activity_copy(version = version)
        if return_as_popanal:
            Xnew = np.mean(X, axis=2, keepdims=True)
            return PopAnal(Xnew, times=np.array([0.]), chans=self.Chans, print_shape_confirmation=False)
        else:
            Xnew = np.mean(X, axis=2, keepdims=False)
            return Xnew

    def mean_over_trials(self, version="raw"):
        """ Return X, but mean over trials,
        out shape (nchans, 1, time)
        """
        assert False, "OBSOLETE. use agg_wrapper, it retains Xlabels"
        X = self.extract_activity_copy(version=version)
        return np.mean(X, axis=1, keepdims=True)

    def agg_by_time_windows_binned(self, DUR, SLIDE):
        """
        REturna  new PA that has times binned in sliding windows.
        PARAMS:
        - DUR, wiodth of window, in sec
        - SLIDE, dur to slide window, in sec. if slide is DUR then is perfect coverage.
        NOTES:
            - windows designed so at least each window, at least half of it inlcudes data
            - possible to exclude some data at end, if the dur is small and slide>dur

        """

        if SLIDE is None:
            SLIDE = DUR

        # MAke new times iwndows
        PRE = self.Times[0]
        POST = self.Times[-1]
        if False:
            # Failed soemtimes, if dur > amount of data. then time_wind would be []
            # n = (POST-PRE)/DUR
            times1 = np.arange(PRE, POST-DUR, SLIDE)
            times2 = times1+DUR
            time_windows = np.stack([times1, times2], axis=1)
        else:
            times1 = np.arange(PRE, POST-DUR/2, SLIDE) # each window, at least half of it inlcudes data
            times2 = times1 + DUR
            time_windows = np.stack([times1, times2], axis=1)
            assert np.isclose(time_windows[0,0], PRE)
            # assert time_windows[-1,1]>=post
            assert time_windows[-1,0]<POST

        # print(DUR, SLIDE)
        # print(time_windows)
        # print(self.Times)
        # assert False
        X, times = self.agg_by_time_windows(time_windows)

        PA = PopAnal(X, times=times, chans=self.Chans,
                    trials = self.Trials)
        PA.Xlabels["trials"] = self.Xlabels["trials"].copy()

        return PA


    def agg_by_time_windows(self, time_windows):
        """ Take mean within multkple time windows, and use those as new time bins.
        - time_windows_mean, list of 2-tuples, each a (pre_dur, post_dur), where negative
        pre_dur means before. Converts fr from (nchans, times) to (nchans, len(times_windows_mean))        
        RETURNS:
        - frmat, shape  (nchans, ntrials, len(time_windows)), the first 2 dims retained from self.X,
        the last dim being the time windowed data.
        - times, array with times, each the mean time in the window, (ntimes, 1)
        """

        assert len(time_windows)>0
        list_xthis = []
        list_time_mean = []
        for wind in time_windows:
            # pathis = self.slice_by_dim_indices_wrapper("times", wind)
            # xthis = pathis.X
            pathis = self.slice_by_dim_values_wrapper("times", wind)
            xthis = pathis.X
            # times = pathis.Times
            # assert False, "replace with wrapper for slicing by time"
            # xthis, times = self._slice_by_time_window(wind[0], wind[1])
            xthis = np.mean(xthis, axis=2, keepdims=True)
            list_xthis.append(xthis)
            list_time_mean.append(np.mean(wind))

        X = np.concatenate(list_xthis, axis=2) # (nchans, ntrials, len(time_windows))
        times=np.array(list_time_mean)[:, None]
        return X, times

    def agg_wrapper(self, along_dim, agg_method="mean", 
            rename_values_agged_dim=True):
        """ aggregate ALL data along a given dimenisions. if you want to first slice,
        do self.slice_and_agg_wrapper()
        PARAMS:
        - along_dim, int or str
        - agg_method, str, how to agg, e.g, mean
        - rename_values_agged_dim, if True, then renames it as a string: f"{agg_method}-{val1}_{val2}"
        otherwise, give it [None]
        RETURNS:
        - PopAnal object
        """

        along_dim, along_dim_str = self.help_get_dimensions(along_dim)

        X = self.extract_activity_copy(version="raw")
        if agg_method=="mean":
            Xagg = np.mean(X, axis=along_dim, keepdims=True)
        elif agg_method=="median":
            Xagg = np.median(X, axis=along_dim, keepdims=True)
        elif agg_method=="sem":
            from scipy import stats
            Xsem = stats.sem(X, axis=along_dim)
            if along_dim==0:
                Xagg = Xsem.reshape(1, Xsem.shape[0], Xsem.shape[1])
            elif along_dim==1:
                Xagg = Xsem.reshape(Xsem.shape[0], 1, Xsem.shape[1])
            elif along_dim==2:
                Xagg = Xsem.reshape(Xsem.shape[0], Xsem.shape[1], 1)
            else:
                assert False
        else:
            print(agg_method)
            assert False, "not coded"

        chans = self.Chans
        trials = self.Trials
        times = self.Times
        # Create new popanal
        if along_dim_str=="times":
            val1 = self.Times[0]
            val2 = self.Times[-1]
            if rename_values_agged_dim:
                times = [f"{agg_method}-{val1}_{val2}"]
            else:
                times = [None]
        elif along_dim_str=="trials":
            val1 = self.Trials[0]
            val2 = self.Trials[-1]
            if rename_values_agged_dim:
                trials = [f"{agg_method}-{val1}_{val2}"]
            else:
                trials = [None]
        elif along_dim_str=="chans":
            val1 = self.Chans[0]
            val2 = self.Chans[-1]
            if rename_values_agged_dim:
                chans = [f"{agg_method}-{val1}_{val2}"]
            else:
                chans = [None]

        PA = PopAnal(Xagg, times=times, chans=chans, trials=trials)
        
        # Retain the lables
        PA.Xlabels = {k:v for k, v in self.Xlabels.items()} # self.Xlabels.copy()
        PA.Xlabels[along_dim] = pd.DataFrame()

        return PA

    def slice_by_labels_range(self, dim_str, dim_variable, valmin, valmax):
        """ Returns data with variable within valmin and
        valmax (inclusive)
        """
        dfthis = self.Xlabels[dim_str]
        assert np.all(np.diff(dfthis.index))==1
        inds = dfthis[(dfthis[dim_variable]>=valmin) & (dfthis[dim_variable]<=valmax)].index.tolist()
        pa = self.slice_by_dim_indices_wrapper(dim_str, inds)
        return pa

    def slice_extract_with_levels_of_conjunction_vars(self, var, vars_others,
                                                      prune_min_n_trials=5, prune_min_n_levs=2,
                                                      plot_counts_heatmap_savepath=None):
        """
        Keep only levels of vars_others, which have at least <prune_min_n_trials> across
        <prune_min_n_levs> many levels of var. Remove all levels of var and vars_others which
        fail this test.
        :return: pa, copy of self, with pruned trials.
        """
        from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars
        pa = self.copy()
        dflab = pa.Xlabels["trials"]
        dfout, dict_dfthis = extract_with_levels_of_conjunction_vars(dflab, var, vars_others,
                                                                 n_min_across_all_levs_var=prune_min_n_trials,
                                                                 lenient_allow_data_if_has_n_levels=prune_min_n_levs,
                                                                 prune_levels_with_low_n=True,
                                                                 ignore_values_called_ignore=True,
                                                                 plot_counts_heatmap_savepath=plot_counts_heatmap_savepath)

        if len(dfout)>0:
            # Only keep the indices in dfout
            pa = pa.slice_by_dim_indices_wrapper("trials", dfout["_index"].tolist(), True)
            return pa, dfout, dict_dfthis
        else:
            return None, None, None

    def slice_by_labels_filtdict(self, filtdict):
        """
        Filter based on self.Xlabels["trials"]
        :param filtdict: variable:list of levels to keep.
        :return: pa, a copy of self, with trials pruned
        """
        pa = self.copy()
        if filtdict is not None:
            for _var, _levs in filtdict.items():
                assert isinstance(_var, str)
                assert isinstance(_levs, (list, tuple))
                pa.Xlabels["trials"] = pa.Xlabels["trials"][pa.Xlabels["trials"][_var].isin(_levs)].reset_index(drop=True)
        return pa

    def slice_by_labels(self, dim_str, dim_variable, list_values, verbose=False):
        """
        SLice to return PA that is subset, where you
        filtering to keep if value is in list_values (filtering).
        PARAMS: 
        - dim_str, string name in {trials, chans, ...}
        - dim_variable, name of column in self.Xlabels[dim_str], or tuple of columns, ion which case considers
        this a conjunctive variabl. List values must thne be list of tuples
        - list_values, list of vales, will keep indices only those with values in list_values
        EG:
        - dim_str = "trials"
        - dim_variable = "epoch"
        - dim_value = "L|0"
        """
        from pythonlib.tools.pandastools import filterPandas

        assert isinstance(list_values, list)
        if isinstance(dim_variable, tuple):
            # Then this is grouping variable.
            from pythonlib.tools.pandastools import append_col_with_grp_index
            self.Xlabels[dim_str] = append_col_with_grp_index(self.Xlabels[dim_str], dim_variable, "_tmp", use_strings=False)
            for v in list_values:
                assert isinstance(v, tuple) and len(v)==len(dim_variable)
            dim_variable = "_tmp"

        if True:
            dfthis = self.Xlabels[dim_str]
            inds = dfthis[dfthis[dim_variable].isin(list_values)].index.tolist()
            # print(len(inds))
            # print(len(self.Xlabels[dim_str]))
        else:
            dfthis = self.Xlabels[dim_str]
            inds = filterPandas(dfthis, {dim_variable:list_values}, return_indices=True)
        if verbose:
            print(f"pa size, before slicing with variable={dim_variable}", self.X.shape)

        pa = self.slice_by_dim_indices_wrapper(dim_str, inds)

        if verbose:
            print(f"... pa size, after slicing with variable={dim_variable}", pa.X.shape)

        return pa

    # def slice_by_label(self, dim_str, dim_variable, dim_value):
    #     """
    #
    #     """
    #     return self.slice_by_labels(dim_str, dim_variable, [dim_value])
    #     # dfthis = self.Xlabels[dim_str]
    #     # inds = dfthis[dfthis[dim_variable]==dim_value].index.tolist()
    #     # pa = self.slice_by_dim_indices_wrapper(dim_str, inds)
    #     # return pa

    def norm_subtract_mean_each_chan(self):
        """
        For each channel, subtract its mean fr (single scalar fr mean across all time bins and trials) from
        all (time bin, trials). i.e, subract from X a vector that is (nchans, 1, 1).
        :return: PA, a copyu of self.
        """

        pamean = self.agg_wrapper("trials")
        pamean = pamean.agg_wrapper("times")

        assert pamean.X.shape[0]==self.X.shape[0]
        assert pamean.X.shape[1]==1
        assert pamean.X.shape[2]==1

        PA = self.copy()
        PA.X = PA.X - pamean.X

        assert PA.X.shape == self.X.shape

        return PA

    def norm_subtract_trial_mean_each_timepoint(self, dim="trials"):
        """ Take mean over one of the dims, and return data subtracting
        that out. e..g, if dim=="trials", for each (chan, timepoint), subtract the mean
        across all trials within that (chan, timepoint).
        RETURNS:
        - PA, a copy of self, with normalized X.
        """
        pamean = self.agg_wrapper(dim, "mean") # (chans, 1, times)
        PA = self.copy()
        PA.X = PA.X - pamean.X

        if False: # NOTE: this succeeds.
            for i in range(PA.X.shape[0]):
                for j in range(PA.X.shape[2]):
                    assert(np.isclose(np.mean(PA.X[i, :, j]), 0))

        return PA

    def norm_subtract_condition_invariant(self, dim="trials"):
        """ at each timepoint, subtract the component of fr that 
        is due to condition-invariant (time), while leaving intact
        the different mean fr for each chan x trial.
        i.e, decompose fr to
        mean_scalar (for a neuron) + (mean at this timporint) +
        noise.
        RETURNS:
        - PA, a copy of self, with normalized X.
        """
        assert dim=="trials", "not sure makes sense otherwise."
        print("TODO: first, subsample so that there are equal num trials across all levels of var.")
        pamean = self.agg_wrapper(dim, "mean") # (chans, 1, times)
        pameanscal = self.agg_wrapper("times", "mean") # (chans, trials, 1)

        # 2) for each trial, subtract the mean timecourse for its level.
        PA = self.copy()
        PA.X = PA.X - pamean.X + pameanscal.X # i.e., subtract (meanscal + conditin_inv)
        # and then add back (meanscal).

        return PA

    def norm_by_label_subtract_mean(self, dim_str, dim_variable_grp):
        """ Returns PA of same size as self, but subtracting the mean
        fr for each level of dim_variable. I.e., gets the noise after taking
        into account the level.
        Optionally subtract the mean at each time point, or a single scalar mean
        across time.
        PARAMS:
        - dim_variable_grp, list of str
        RETURNS:
            - copy of PA, same size as self, guarandeed to have same ordering of
            chans and times, but NOT guaranteed to have same ordering of trials.
        """

        # First, split into separate pa, one for each level of <dim_variable>
        list_pa = self.split_by_label(dim_str, dim_variable_grp)[0]

        # Subtract mean for each, then concat.
        list_pa_norm = []
        for pa in list_pa:
            # list_pa_norm.append(pa.norm_subtract_trial_mean_each_timepoint()) # wrong -- subtracts each time point.,
            list_pa_norm.append(pa.norm_subtract_mean_each_chan())
        PA = concatenate_popanals(list_pa_norm, "trials")

        # Check match between input and output.
        assert PA.Chans == self.Chans
        assert check_identical_times([self, PA])==True
        assert len(PA.Trials)==len(self.Trials)

        return PA

    def split_by_label(self, dim_str, dim_variable_grp):
        """ Splits self into multiple smaller PA, each with a single level for
        dim_variable_grp. Uses dataframe self.Xlabels[dim_str].
        PARAMS:
        - dim_str, which dim in self.X to check. e.g,., "trials"
        - dim_variable_grp, list of string, e.g., "gridsize"
        RETURNS:
        - ListPA, e.g., each PA has only trials with a single level of gridsize
        - list_levels, list of str, maches ListPA
        """

        if isinstance(dim_variable_grp, str):
            # Legacy code.
            dim_variable_grp = [dim_variable_grp]
            IS_STRING = True
        else:
            IS_STRING = False

        # make dummy variable
        from pythonlib.tools.pandastools import append_col_with_grp_index
        self.Xlabels[dim_str] = append_col_with_grp_index(self.Xlabels[dim_str],
                                                          dim_variable_grp,
                                                          "_dummy", use_strings=False)

        # 1) Get list of levels
        list_levels = self.Xlabels[dim_str]["_dummy"].unique().tolist()

        # 2) For each level, return a single PA
        ListPA = []
        for lev in list_levels:
            # slice
            pathis = self.slice_by_labels(dim_str, "_dummy", [lev])
            ListPA.append(pathis)

        if IS_STRING:
            # convert back to string (legacy code)
            list_levels = [x[0] for x in list_levels]

        return ListPA, list_levels

    #######################
    # def dataextract_as_clusters_after_conj_grouping(self, vars_grp_and_extract, do_agg_by_grouping=False):
    #     """
    #     Extract clusters represntation of PA.X, which is (
    #     :param vars_grp_and_extract:
    #     :param do_agg_by_grouping:
    #     :return:
    #     """
    #     from pythonlib.cluster.clustclass import Clusters
    #
    #     if agg_by_grouping:
    #         pa = self.slice_and_agg_wrapper("trials", vars_grp_and_extract)
    #     else:
    #         pa = self.copy()
    #
    #     label_vars = vars_grp_and_extract
    #     dflab = pa.Xlabels["trials"]
    #     labels_rows = dflab.loc[:, label_vars].values.tolist()
    #     labels_rows = [tuple(x) for x in labels_rows] # list of tuples
    #     params = {
    #         "label_vars":label_vars,
    #     }
    #
    #     # If >1 time dimension, take mean over time.
    #     if pa.X.shape[2]>1:
    #         pa = pa.agg_wrapper("times")
    #
    #     Cl = Clusters(pa.X, labels_rows, ver="rsa", params=params)
    #
    #     return Cl

    def dataextract_state_space_decode_flex(self, twind_overall=None,
                                            tbin_dur=None, tbin_slide=None,
                                            reshape_method = "chans_x_trials_x_times",
                                            pca_reduce=False,
                                            how_decide_npcs_keep = "cumvar",
                                            pca_frac_var_keep = 0.9, pca_frac_min_keep=0.01,
                                            plot_pca_explained_var_path=None, plot_loadings_path=None,
                                            pca_method="svd",
                                            norm_subtract_single_mean_each_chan=True,
                                            npcs_keep_force=None,
                                            extra_dimred_method=None,
                                            extra_dimred_method_n_components=2, umap_n_neighbors = 30,
                                            PLOT_EXAMPLE_X_BEFORE_GO_INTO_PCA=False):
        """
        Fleixble methods for extract data for use in population analyses, slicing out a specific time window,
        and binning by time, and ootionally reshaping to (ntrials, ...), where you can optionally
        combine the higher dimensions with various methods for reshaping data output.

        In general, will demean within each channel (subtract single scalar fr across time bins), so maintaining temporal
        structure within each channel, and then doing PCA on that (nchans x ntimes vector).

        Keeps top N dimensions by criteriion either based on cumvar or minvar.

        PARAMS:
        - twind_overall, only keep data within this window (e.g, [0.3, 0.6])
        - tbin_dur, optional, for binning data (sec)
        - tbin_slide, optional, if binning, how slide bin
        - reshape_method, str, defines shape of output.
        - pca_method, str, either
        --- "sklearn" : centers each time bin!
        --- "svd" : same method as sklearn, but does not do any centering.
        - extra_dimred_method, str, methdo to apply after PCA.
        --- "umap"

        RETURNS:
        - X, final data, reshaped as desired, and dim reduction applied.
        - PAfinal, PA holding X. (always dims, trials, times).
        - PAslice, PA holding data sliced, but before reshape or dim reduction.
        """

        PAslice = self.copy()

        # Slice to desired window
        if twind_overall is not None:
            PAslice = PAslice.slice_by_dim_values_wrapper("times", twind_overall)

        # Bin
        if tbin_dur is None:
            # Then take mean
            PAslice = PAslice.agg_wrapper("times") # (chans, trials, 1)
        else:
            PAslice = PAslice.agg_by_time_windows_binned(tbin_dur, tbin_slide)

        if norm_subtract_single_mean_each_chan:
            # Normalize activity before doing pca?
            # - at very least, always subtract mean within each channel (not going as far as subtracting mean
            # within eahc time point of each channel).
            PAslice = PAslice.norm_subtract_mean_each_chan()

        if PLOT_EXAMPLE_X_BEFORE_GO_INTO_PCA:
            PAslice.plotNeurHeat(0)

        ## DIM REDUCTION
        nchans, ntrials, ntimes = PAslice.X.shape
        if reshape_method=="trials_x_chanstimes":
            # Reshape to (ntrials, nchans*ntimes)
            tmp = np.transpose(PAslice.X, (1, 0, 2)) # (trials, chans, times)
            X = np.reshape(tmp, [ntrials, nchans * ntimes]) # (ntrials, nchans*timebins)
            X_before_dimred = X.copy()

            # Sanitych check
            if False: # no need to check. know it works.
                trial = 0
                tmp = np.concatenate([PAslice.X[:, trial, i] for i in range(ntimes)])
                if not np.isclose(np.std(X[trial]), np.std(tmp)):
                    print(np.std(X[trial]))
                    print(np.std(tmp))
                    assert False, "bug in reshaping"

            if pca_reduce:
                print("Running PCA")
                print(how_decide_npcs_keep, pca_frac_var_keep, pca_frac_min_keep)
                from neuralmonkey.analyses.state_space_good import dimredgood_pca
                # Make labels (chans x timebins)
                ntimes = len(PAslice.Times)
                col_labels = []
                for ch in PAslice.Chans:
                    for t in range(ntimes):
                        col_labels.append((ch, t))
                X, _, pca = dimredgood_pca(X,
                                           how_decide_npcs_keep = how_decide_npcs_keep,
                                           pca_frac_var_keep=pca_frac_var_keep, pca_frac_min_keep=pca_frac_min_keep,
                                           plot_pca_explained_var_path=plot_pca_explained_var_path,
                                           plot_loadings_path=plot_loadings_path,
                                           plot_loadings_feature_labels=col_labels,
                                           method=pca_method,
                                           npcs_keep_force=npcs_keep_force) # (ntrials, nchans) --> (ntrials, ndims)

                # Represent X in PopAnal
                # PAfinal = PopAnal(X.T[:, :, None].copy(), [0])  # (ndimskeep, ntrials, 1)
                # PAfinal.Xlabels = {dim:df.copy() for dim, df in PAslice.Xlabels.items()}
                # assert len(PAfinal.Xlabels["trials"])==PAfinal.X.shape[1]

            # assert X.shape[0] == PAslice.X.shape[1]
            # if pca_reduce:
            #     assert X.shape[1] == PAfinal.X.shape[0]
            #     assert X.shape[0] == PAfinal.X.shape[1]

            # Extra dimreduction step?
            if extra_dimred_method in ["umap", "mds"]:
                from neuralmonkey.analyses.state_space_good import dimredgood_nonlinear_embed_data
                X, _ = dimredgood_nonlinear_embed_data(X, METHOD=extra_dimred_method, n_components=extra_dimred_method_n_components,
                                                           umap_n_neighbors=umap_n_neighbors) # (ntrials, ndims)
            else:
                assert extra_dimred_method is None

            # Represent X in PopAnal
            PAfinal = PopAnal(X.T[:, :, None].copy(), [0])  # (ndimskeep, ntrials, 1)
            PAfinal.Xlabels = {dim:df.copy() for dim, df in PAslice.Xlabels.items()}
            assert len(PAfinal.Xlabels["trials"])==PAfinal.X.shape[1]

            # Sanity check
            assert X.shape[0] == PAslice.X.shape[1]
            assert X.shape[1] == PAfinal.X.shape[0]
            assert X.shape[0] == PAfinal.X.shape[1]

            # print(X.shape)
            # print(PAfinal.X.shape)
            # print(PAslice.X.shape)
            # assert False
            # assert False
        elif reshape_method=="chans_x_trials_x_times":
            # Default.
            # PCA --> first combines trials x timebins (i.e.,
            # (ntrials*ntimes, nchans) --> (ntrials*ntimes, ndims)

            if False: # Check passes
                # Check that reshapes (skipping PCA) dont affect data
                X = np.reshape(PAslice.X, [nchans, ntrials * ntimes]).T
                # <PCA would be here>
                X1 = X.T
                X1 = np.reshape(X1, [nchans, ntrials, ntimes])
                assert np.all(PAslice.X==X1)

            # No need to reshape
            X = PAslice.X
            X_before_dimred = X.copy()

            if pca_reduce:
                # Reshape to pass into PCA
                X = np.reshape(PAslice.X, [nchans, ntrials * ntimes]).T # (ntrials*ntimes, nchans)

                from neuralmonkey.analyses.state_space_good import dimredgood_pca
                # Make labels (chans x timebins)
                ntimes = len(PAslice.Times)
                col_labels = []
                for ch in PAslice.Chans:
                    for t in range(ntimes):
                        col_labels.append((ch, t))
                X, _, pca = dimredgood_pca(X,
                                           how_decide_npcs_keep = how_decide_npcs_keep,
                                           pca_frac_var_keep=pca_frac_var_keep, pca_frac_min_keep=pca_frac_min_keep,
                                           plot_pca_explained_var_path=plot_pca_explained_var_path,
                                           plot_loadings_path=plot_loadings_path,
                                           plot_loadings_feature_labels=col_labels,
                                           method=pca_method,
                                           npcs_keep_force=npcs_keep_force) # (ntrials, nchans) --> (ntrials, ndims)

                # Reshape back to original
                npcs_keep = X.shape[1]
                X = X.T # (npcs_keep, ntrials*ntimes)
                X = np.reshape(X, [npcs_keep, ntrials, ntimes]) # (npcs_keep, ntrials*ntimes)

            # Represent X in PopAnal
            PAfinal = PopAnal(X.copy(), PAslice.Times)  # (ndimskeep, ntrials, 1)
            PAfinal.Xlabels = {dim:df.copy() for dim, df in PAslice.Xlabels.items()}
            assert len(PAfinal.Xlabels["trials"])==PAfinal.X.shape[1]

            # Sanity check
            assert X.shape[1:] == PAslice.X.shape[1:]
            if pca_reduce:
                assert X.shape == PAfinal.X.shape

            # Extra dimreduction step?
            assert extra_dimred_method is None, "not yet coded.. a bit tricky?"

        else:
            print(reshape_method)
            assert False

            # X = PAslice.X
            #
            # if pca_reduce:
            #     assert False, "not coded..."
            #
            # assert extra_dimred_method is None, "not yet coded"

        if not pca_reduce:
            pca = None
            # PAfinal = None

        # print(X.shape)
        # print(PAfinal.X.shape)
        # print(PAslice.X.shape)
        # assert False
        return X, PAfinal, PAslice, pca, X_before_dimred

    def _dataextract_split_by_label_grp_for_statespace(self, grpvars):
        """
        Return a dataframe, where each row is a single level of conjunctive grpvars,
        holding all trials. Useful for state space traj plotting etc, e.g.,
        trajgood_plot_colorby_splotby().

        RETURNS:
        - df, standard form for holding trajectories, each row holds; one condition (e.g., shape,location):
        --- "z", activity (ndims, ntrials, ntimes),
        --- "z_scalar", scalarized version (ndims, ntrials, 1) in "z_scalar".
        --- "times", matching ntimes
        """
        from neuralmonkey.analyses.state_space_good import trajgood_construct_df_from_raw
        labels = self.Xlabels["trials"].loc[:, grpvars]
        labelvars = grpvars
        df = trajgood_construct_df_from_raw(self.X, self.Times, labels, labelvars)
        return df

    #######################
    def reshape_by_splitting(self):
        """
        Reshape self into new PA with PA.X.shape = (1, m*k, t), instead of
        orignal shape (m, k, t), i.e, all data are along the trials dimension.
        Makes sure that PA.Xlabels["trials"] is correctly matchins the m*k rows.
        Will add a new column to PA.Xlabels["trials"] called chan, which is the
        original chan in self.Chans. Throws out PA.Chans and PA.Trials.
        
        RETURNS:
        - PA, (without modifyin self)
        """
        
        assert len(self.Xlabels["trials"])>0, "then will not retain information about trials and chans..."

        list_pa = []
        list_labels = []
        for i, chan in enumerate(self.Chans):
            # get the sliced pa
            pathis = self.slice_by_dim_indices_wrapper("chans", [i])
            # collect infoextract_snippets_trials
            list_pa.append(pathis)
            list_labels.append(chan)
        PA = concatenate_popanals(list_pa, dim="trials", 
                                        map_idxpa_to_value=list_labels, 
                                        map_idxpa_to_value_colname="chan",
                                        assert_otherdims_have_same_values=False)
        
        assert PA.X.shape[0]==1 and PA.X.shape[1]==(self.X.shape[0]*self.X.shape[1])
        return PA


    def slice_by_label_grouping(self, dim_str, grouping_variables, grouping_values):
        """ 
        Return sliced PA, where first constructs grouping varible (can be conjunctive)
        then only keep desired subsets 
        PARAMS;
        - dim_str, e.g,., "trials"
        - grouping_variables, list of str
        - grouping_values, values to keep
        """
        assert False, "code it, see slice_and_agg_wrapper"



    def slice_and_agg_wrapper(self, along_dim, grouping_variables, grouping_values=None,
            agg_method = "mean", return_group_dict=False, return_list_pa=False):
        """ Flexibly aggregate neural data along any of the three dimensions, using
        any variable, etc. Returns PA where each level of the grouping
        variable is a single "trial" (after averaging over all trials with that level).
        PARAMS:
        - along_dim, eitehr int {0,1,2} or str {'chans', 'trials', 'times'}, euivalent, and
        ordered as such. which diemsions to agg (will collapse this dim)
        - grouping_variables, list of str, cross-prod of thee variables will define the goruping
        - grouping_values, list of str, levels of the given grouping. Leave None to use
        all levels
        """

        assert len(grouping_variables)==len(set(grouping_variables))
        from pythonlib.tools.pandastools import grouping_append_and_return_inner_items

        along_dim, along_dim_str = self.help_get_dimensions(along_dim)

        if isinstance(grouping_variables, str):
            grouping_variables = [grouping_variables]
        else:
            assert isinstance(grouping_variables, list)

        # Collect indices for each level of the grouping
        df = self.Xlabels[along_dim_str]
        # - sanity check
        for var in grouping_variables:
            assert var in df.columns

        # Get indices for each grouping level
        groupdict = grouping_append_and_return_inner_items(df, 
            grouping_variables, groupinner='index', 
            groupouter_levels=grouping_values,
                                           sort_keys=True)

        # Get sliced pa for each grouping level
        list_pa = []
        list_grplevel = []
        for grp in groupdict:
            inds = groupdict[grp]

            # slice
            pathis = self.slice_by_dim_indices_wrapper(dim=along_dim, inds=inds)

            # agg
            pathis = pathis.agg_wrapper(along_dim=along_dim, agg_method=agg_method)

            # collect
            list_pa.append(pathis)
            list_grplevel.append(grp)

        # Concatenate all pa into a larger pa
        pa_all = concatenate_popanals(list_pa, dim=along_dim, values_for_concatted_dim=list_grplevel)

        # Pass in the label dataframes
        # (best to pass in here, and not agg, or slice, beucase this is the lowest level code
        # where there is consistently meaningfull labels.
        # - initialize with the current labels
        pa_all.Xlabels = self.Xlabels.copy()
        
        # - then replace the slice-agged dimension
        dat = {}
        for i, var in enumerate(grouping_variables):
            vals = [x[i] for x in list_grplevel]
            dat[var] = vals
        dflab = pd.DataFrame(dat)
        pa_all.Xlabels[along_dim_str] = dflab

        if return_list_pa:
            return list_pa, list_grplevel

        if return_group_dict:
            return pa_all, groupdict
        else:
            return pa_all

    def agg_by_trialgrouping(self, groupdict, version="raw", return_as_popanal=True):
        """ aggreagate so that trials dimension is reduced,
        by collecting multiple trials into single groups
        by taking mean over trials. 
        PARAMS:
        - groupdict, {groupname:trials_list}, output from
        pandatools...
        RETURNS:
        - eitehr PopAnal or np array, with data shape = (nchans, ngroups, time),
        with ngroups replacing ntrials after agging.
        """
        assert False, "OBSOLETE - use slice_and_agg_wrapper"

        # 1) Collect X (trial-averaged) for each group
        list_x = []
        for grp, inds in groupdict.items():
            PAthis = self._slice_by_trial(inds, version=version, return_as_popanal=True)
            x = PAthis.mean_over_trials()
            list_x.append(x)

        # 2) Concatenate all groups
        X = np.concatenate(list_x, axis=1)

        # 3) Convert to PA
        if return_as_popanal:
            # assert PAnew.shape[1]==len(groupdict)
            PA = PopAnal(X, PAthis.Times, PAthis.Chans)
            # save the grp labels
            PA.Xlabels
            return PA
        else:
            return X

    ############### LABELS, each value ina given dimension of X 
    def labels_features_input(self, name, values, dim="trials"):
        """ Just synonym for self.labels_input(), easier to search for.
        Append values, stored in self.Xlabels[dim]
        PARAMS:
        - name, str
        - values, list-like values, must match the size of this dim exactly.
        - dim, str, whether labels match {trials, times, chans}
        NOTE: opverwrites name if it has been p[reviusly entered]
        """
        return self.labels_input(name, values, dim)

    def labels_features_input_from_dataframe_merge_append(self, dflab):
        """ Good method to merge all labeles in dflab into self.Xtrials["trials"]
        PARAMS;
        - dflab, rows are trialcodes, and must have at least all trialcodes that
        exist in self.Xtrials["trials"]; can have more
        Guarantees that trialcodes are matched.
        If any columns already exist in self,
        RTEURNS:
        - appends to self.Xtrials["trials"]
        """
        # add the new labels
        from pythonlib.tools.pandastools import slice_by_row_label

        # Get slice of dflab matching trialcodes in self.
        tcs = self.Xlabels["trials"]["trialcode"].tolist()
        dflab_this = slice_by_row_label(dflab, "trialcode", tcs, assert_exactly_one_each=True,
                           prune_to_values_that_exist_in_df=False)

        # Only append the columns that are new. Also, if not new check that values are identical
        # in self and dflab
        cols_keep = []
        for col in dflab_this:
            if col in self.Xlabels["trials"] and not col=="trialcode":
                assert np.all(self.Xlabels["trials"][col] == dflab_this[col])
            else:
                cols_keep.append(col)

        # Merge
        dftmp = self.Xlabels["trials"].merge(dflab_this[cols_keep], "outer", on="trialcode")
        assert np.all(dftmp["trialcode"] == self.Xlabels["trials"]["trialcode"]), "merge error"
        self.Xlabels["trials"] = dftmp.reset_index(drop=True)

    def labels_features_input_from_dataframe(self, df, list_cols, dim, overwrite=True):
        """ Assign batchwise, labels to self.Xlabels
        PARAMS:
        - df, dataframe from which values will be extarcted
        - list_cols, feature names, strings, columns in df. if "index" then gets that
        - dim, which dimension in self to modify.
        - overwrite, if False, then throws error if that var already exists in self.Xlabels[dim]
        NOTE: assumes that df is ordereed idetncail to the data in dim.
        """

        assert len(list_cols)>0, 'or else will not retrain self.Xlabels[trials]; ie will be empty...'

        # store each val
        for col in list_cols:
            if col=="index":
                # confirm that this is nto exist as a column
                assert "index" not in df.columns, "problem, ambiguos you want df.index or df[index]?"
                values = df.index.tolist()
            else:
                values = df[col].tolist()
            self.labels_input(col, values, dim=dim, overwrite=overwrite)          

    def labels_features_input_conjunction_other_vars(self, dim, list_var):
        """ Assingn new columns to self.Xlabels[dim], one for each conjunction
        of vars in list_vars, where the number of conjunctions is len(list_vars),
        i.e,., for each var, the conjunction of others. Useful for then computing the 
        effect of a given var condiitioned on some value of the conjucntion of 
        other vars.
        PARAMS
        - dim, str
        - list_var, list of str, each a column in self.Xlabels[dim]
        RETURNS:
        - map_var_to_othervars, dict mapping from variable to column name of conjuction
        of its other vartiables (or None, if none exist).
        """        
        from pythonlib.tools.pandastools import append_col_with_grp_index
        from pythonlib.tools.checktools import check_is_categorical

        df = self.Xlabels[dim]
        for var in list_var:
            assert var in df.columns

        # Only take ocnjucntions of categorical variables.
        list_var_categorical = [var for var in list_var if check_is_categorical(df[var].tolist()[0])]

        # Prepare columns indicating value of conjucntion of other faetures
        # list_var = ["gridloc", "gridsize", "shape_oriented"]

        # 1) in df, first prepare by getting, for each var, the conjucntion of the other vars.
        map_var_to_othervars = {}
        for var in list_var:
            # get conjuction of the other vars
            # other_vars = [v for v in list_var if not v==var]
            other_vars = [v for v in list_var_categorical if not v==var]

            if len(other_vars)>1:
                # Then good, get conjunction
                df, new_col_name = append_col_with_grp_index(df, other_vars, "-".join(other_vars), 
                                                             strings_compact=True, return_col_name=True)            
                print("added column: ", new_col_name)
            elif len(other_vars)==1:
                # if len 1, then this col already exists..
                new_col_name = other_vars[0]
            else:
                # nothing, add a dummy variable, so downstream code works.
                df["dummy"] = "dummy"
                new_col_name = "dummy"

            # Save mapper 
            map_var_to_othervars[var] = new_col_name

        self.Xlabels[dim] = df

        return map_var_to_othervars



    def labels_input(self, name, values, dim="trials", overwrite=True):
        """ Append values, stored in self.Xlabels[dim]. Does 
        sanity check that lenghts are correct
        PARAMS:
        - name, str
        - values, list-like values, must match the size of this dim exactly.
        - dim, str, whether labels match {trials, times, chans}
        - overwrite, bool, if true, opverwrites name if it has been p[reviusly entered]
        """

        # Extract the desired dataframe
        # and sanity check matching of sizes
        if dim=="trials":
            df = self.Xlabels["trials"]
            assert len(values)==len(self.Trials)
        elif dim=="chans":
            df = self.Xlabels["chans"]
            assert len(values)==len(self.Chans)
        elif dim=="times":
            df = self.Xlabels["times"]
            assert len(values)==len(self.Times)
        else:
            print(dim)
            assert False

        if overwrite==False:
            # confirm that it doesnt aklready exist
            assert name not in df.columns
        df[name] = values


    ################ INDICES
    def index_find_this_chan(self, chan):
        """ Returns the index (into self.X[index, :, :]) for this
        chan, looking into self.Chans, tyhe lables.
        PARAMS:
        - chan, label, as in self.Chans
        RETURNS;
        - index, see above
        """
        assert isinstance(chan, int)
        return self.Chans.index(chan)
        # for i, ch in enumerate(self.Chans):
        #     if ch==chan:
        #         return i
        # assert False, "this chan doesnt exist in self.Chans"

    def index_find_this_trial(self, trial):
        """ Returns the index (into self.X[:, index, :]) for this
        trial, looking into self.Trials, tyhe lables.
        PARAMS:
        - trial, value to serach for in self.Trials
        RETURNS;
        - index, see above
        """
        assert isinstance(trial, int)
        return self.Trials.index(trial)

    def index_find_this_time(self, time):
        """ Given time (sec) find index into self.X[:, :, index] that is
        closest to this time, in absolute terms
        """
        ind = np.argmin(np.abs(self.Times - time))
        return ind

    def index_find_this_time_window(self, twind, time_keep_only_within_window=True):
        """
        Get min and max indices into self.Times, such that all values in self.Times[indices]
        are contained within twind (ie.,a ll less than twind).
        PARAMS:
        - time_keep_only_within_window, bool, if True, then the time of the indices must be
        within twind. If False, then the times are the ones CLOSEST to twind, but they could
        be larger.
        """

        inds = self.index_find_these_values("times", twind)
        assert len(inds)==2

        if time_keep_only_within_window:
            # inclusive, deal with numerical imprecision..
            # ensuring that indices are entirely contained within twind.
            while self.Times[inds[0]]<=twind[0]:
                inds[0]+=1
            while self.Times[inds[1]]>=twind[1]:
                inds[1]-=1
        assert inds[0]<=inds[1]

        return inds

    def index_find_these_values(self, dim, values):
        """ return the indices into self.Trials or self.Chans for these values
        PARAMS;
        - dim, string in {'chans', 'trials'}
        RETURNS:
        - list of indices, matching order of values exaclty.
        """

        dim, dim_str = self.help_get_dimensions(dim)

        if dim_str == "trials":
            return [self.index_find_this_trial(x) for x in values]
        elif dim_str == "chans":
            return [self.index_find_this_chan(x) for x in values]
        elif dim_str == "times":
            return [self.index_find_this_time(x) for x in values]
        else:
            print(dim)
            assert False

    def shuffle_by_trials(self, inds_rows):
        """ Shuffle rows, given input indices, with many sanityc checeks,
        and ensuring that both data and labels shuffled
        PARAMS:
        - inds_rows, list of ints, into self.X[:, inds_rows, :] and labels,
        the new order.
        RETURNS:
        - copy of pa, shuiffled.
        """

        assert False, "In progress!!"

        n = self.X.shape[1]
        assert len(inds_rows) == n
        assert(max(inds_rows)==n-1)
        assert(min(inds_rows)==0)


        inds_rows = df_shuff["index"].tolist()
        pa_shuff = self.copy()

        list(range(self.X.shape[1]))

        pa_shuff.X = self.X[:, inds_rows, :]
        pa_shuff.Xlabels["trials"] = self.Xlabels["trials"].iloc[inds_rows].reset_index(drop=True)
        # sanity check
        pa_shuff.Xlabels["trials"][var] == self.Xlabels["trials"][var]
        pa_shuff.Xlabels["trials"][var]
        self.Xlabels["trials"][var]

    #### EXTRACT ACTIVITY
    def extract_activity_copy(self, trial=None, version="raw"):
        """ REturn activity for this trial, a copy
        PARAMS:
        - trial, int, if None, then returns over all trials
        - version, string, ee.g. {'raw', 'z'}
        RETURNS:
        - X, shape (nchans, 1, ntime), or (nchans, ntrials, ntime, if trial is None)
        """

        if version=="raw":
            if trial is None:
                return self.X.copy()
            else:
                return self.X[:, trial, :].copy()
        elif version=="z":
            assert self.Xz is not None, "need to do zscore first"
            if trial is None:
                return self.Xz.copy()
            else:
                return self.Xz[:, trial, :].copy()
        else:
            print(version)
            assert False
    def extract_activity_copy_all(self):
        """
        Copy and return all data from self.
        :return: data, dict.
        """
        data = {
            "X":self.X.copy(),
            "dflab":self.Xlabels["trials"].copy(),
            "Times":self.Times,
            "Chans":self.Chans,
            "Trials":self.Trials,
        }

        return data


    def help_get_dimensions(self, dim):
        """ Hleper to convert between types for dim (int or str)
        PARAMS;
        - dim, eitehr int {0,1,2} or str {'chans', 'trials', 'times'}, euivalent
        RETURNS:
        - dim(int), dim_str(string)
        """
        return help_get_dimensions(dim)


    ### PLOTTING
    def plotNeurHeat(self, trial, version="raw", **kwargs):
        X = self.extract_activity_copy(trial, version)
        return plotNeurHeat(X, **kwargs)

    def plotNeurTimecourse(self, trial, version="raw", **kwargs):
        X = self.extract_activity_copy(trial, version)
        return plotNeurTimecourse(X, **kwargs)

    def plotwrapper_smoothed_fr(self, inds=None, axis_for_inds="site", ax=None, 
                     plot_indiv=True, plot_summary=False, error_ver="sem",
                     pcol_indiv = "k", pcol_summary="r", summary_method="mean",
                     event_bounds=(None, None, None), alpha=0.6):
        """ Wrapper for different ways of plotting multiple smoothed fr traces, where
        multiple could be trials or sites. Also to plot summaries (means, etc). 
        PARAMS:
        - PA, popanal object
        - inds, list of ints, indices into PA. interpretation dpeends on axis_for_inds.
        Note, these are the labels in PA.Chans, not the indices into them. If None, then 
        plots all data
        - axis_for_inds, str, which axis inds indexes, either {'site', 'trial'}
        - ax, axis to plot on    
        - plot_indiv, bool, whether to plot individual traces.
        - plot_summary, bool, whether to plot summary (method defined by error_ver)
        - error_ver, string or None, methods for overlaying mean+error
        """
        from neuralmonkey.neuralplots.population import plotNeurTimecourse, plotNeurTimecourseErrorbar
        import numpy as np

        # extract data
        if inds is not None:
            PAthis = self.slice_by_dim_values_wrapper(axis_for_inds, inds)
            # if axis_for_inds in ["site", "sites"]:
            # #         idxs = [PA.Chans.index(site) for site in inds]
            #     PAthis = self._slice_by_chan(inds, return_as_popanal=True)
            # elif axis_for_inds in ["trial","trials"]:
            #     PAthis = self._slice_by_trial(inds, return_as_popanal=True)
            # else:
            #     assert False
        else:
            PAthis = self
        X = PAthis.X
        times = PAthis.Times

        if X.shape[1]==1:
            plot_indiv = True
            plot_summary = False

        # Reshape to (nsamp, times), combining all chans and trials.
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        n_time_bins = X.shape[1]

        # 1) Plot indiividual traces?
        if plot_indiv:
            fig1, ax1 = plotNeurTimecourse(X, times, ax=ax, color = pcol_indiv,
                alpha=alpha)
        else:
            fig1, ax1 = None, None

        # 2) Plot summary too?
        if plot_summary:
            if error_ver=="sem":
                from scipy import stats
                if summary_method=="mean":
                    Xmean = np.mean(X, axis=0)
                elif summary_method=="median":
                    Xmean = np.median(X, axis=0)
                else:
                    print(summary_method)
                    assert False
                Xsem = stats.sem(X, axis=0)
            else:
                assert error_ver is None, "not coded"

            fig2, ax2 = plotNeurTimecourseErrorbar(Xmean, Xerror=Xsem, times=times,ax=ax, color=pcol_summary)
        else:
            fig2, ax2 = None, None

        # 3) Overlay event boundaries?
        colors = ['r', 'k', 'b']
        for evtime, pcol in zip(event_bounds, colors):
            if evtime is not None:
                ax.axvline(evtime, color=pcol, linestyle="--", alpha=0.4)

        return fig1, ax1, fig2, ax2

    def plotwrapper_smoothed_fr_split_by_label(self, dim_str, dim_variable, ax=None,
                                              plot_indiv=False, plot_summary=True,
                                              event_bounds=[None, None, None],
                                              add_legend=True, legend_levels=None,
                                               chan=None):
        """ Plot separate smoothed fr traces, overlaid on single plot, each a different
        level of an inputted variable
        PARAMS:
        - dim_str, str, in {times, chans}
        - dim_variable, column in dataframe in self.Xlabels, to look for levels of
        - event_bounds, [num, num, num] to plot pre, alignemnt, and post times. any that
        are None will be skipped.
        - legend_levels, list of values that will be used for legend, where the order
        defines a globally true mapping between level and color, useful if you want to 
        dictate the colors for levels that are not in this partiucla plot.
        """
        from pythonlib.tools.plottools import makeColors

        if chan is not None:
            # Then pull out specific PA that is just this chan
            PA = self.slice_by_dim_values_wrapper("chans", [chan])
        else:
            assert False, "did you really not want to input chan?"
            PA = self

        # Split into each pa for each level
        list_pa, list_levels_matching_pa = PA.split_by_label(dim_str, dim_variable)
        
        # make dict mapping from level to col
        if legend_levels is None:
            # then use the levels within here
            legend_levels = list_levels_matching_pa
        pcols = makeColors(len(legend_levels))
        dict_lev_color = {}

        for pc, lev in zip(pcols, legend_levels):
            dict_lev_color[lev] = pc

        for pa, lev in zip(list_pa, list_levels_matching_pa):
            pcol = dict_lev_color[lev]
            pa.plotwrapper_smoothed_fr(ax=ax, plot_indiv=plot_indiv, plot_summary=plot_summary,
                                          pcol_indiv = pcol, pcol_summary=pcol,
                                          event_bounds=event_bounds)

        # add legend
        if add_legend:
            from pythonlib.tools.plottools import legend_add_manual
            legend_add_manual(ax, legend_levels, pcols, 0.2)

        return pcols

    def plotwrapper_smoothed_fr_split_by_label_and_subplots(self, chan, var, vars_subplots):
        """
        Helper to plot smoothed fr, multkiple supblots, each varying by var
        :param var: str, to splot and color within subplot
        :param vars_subplots: list of str, each is a supblot
        :param chan:
        :return:
        """
        list_pa, levels = self.split_by_label("trials", vars_subplots)
        ncols = 8
        nrows = int(np.ceil(len(levels)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3), sharex=True, sharey=True)
        for ax, lev, pa in zip(axes.flatten(), levels, list_pa):
            pa.plotwrapper_smoothed_fr_split_by_label("trials", var, ax, chan=chan)
            ax.set_title(lev)

    ############################
    def convert_to_dataframe_long(self):
        """ Convert to dataframe, where each trial*chan is a row. each row
        will have a column "sm_fr" holding the smoothed fr. Does not modify self.
        RETURNS:
        - dataframe (see above).
        """

        # 1) Reshape, so that X[:,i,:] holds a trial*chan combo
        pathis = self.reshape_by_splitting()

        # 2) extract fr for each row
        frlist = [pathis.X[:, i, :] for i in range(pathis.X.shape[1])]
        pathis.Xlabels["trials"]["fr_sm"] = frlist    

        return pathis.Xlabels["trials"]    
                    
    ########################### STATE SPACE
    def statespace_pca_plot_projection(self, frmat, ax, dims_pc=(0, 1), 
        color_for_trajectory="k", alpha=0.2,
        times=None, times_to_mark=None, times_to_mark_markers=None,
        time_windows_mean=None, markersize=3, marker="o"):
        """ plots data in frmat in the space defined by PApca, which holds pca space
        computed from the data in PApca
        PARAMS:
        - PApca, PopAnal object
        - frmat, array shape (chans, times), data to plot, chans must match chans
        in PApca. len(times) can be 1.
        - dims_pc, 2-integers, which pc dimensions to plot (x and y axes)
        - times, array-like, len matches frmat.shape[1], used for placing markers
        - times_to_mark, list of times, to place markers on them
        - times_to_mark_markers, list of strings, markers to use. If None, then uses 
        0,1, 2,.../
        [OBS] If None, then uses the times (as strings)
        """    
        from neuralmonkey.population.dimreduction import plotStateSpace

        assert len(dims_pc)==2
        NDIM = max(dims_pc)+1
        is_traj = frmat.shape[1]>1

        if not is_traj:
            times_to_mark = None
            times_to_mark_inds = None

        if times_to_mark is not None:
            assert is_traj
            assert times is not None
            assert len(times)==frmat.shape[1]
            if isinstance(times, list):
                times = np.array(times)
            # find indices matching these times
            def _find_ind(t):
                ind = np.argmin(np.abs(times - t))
                return ind
            times_to_mark_inds = [_find_ind(t) for t in times_to_mark]
            if False: # is better to just use 0,1, ..
                if times_to_mark_markers is None:
                    # then use the times themselves
                    times_to_mark_markers = [f"${t}$" for t in times_to_mark]
        else:
            times_to_mark_inds = None

        X = self.reprojectInput(frmat, Ndim=NDIM)
        x1, x2 = plotStateSpace(X, dims_neural=dims_pc, plotndim=len(dims_pc), ax=ax, 
            color_for_trajectory=color_for_trajectory, is_traj=is_traj, alpha=alpha,
            traj_mark_times_inds = times_to_mark_inds, 
            traj_mark_times_markers = times_to_mark_markers,
            markersize=markersize, marker=marker)
        
        # grid on, for easy comparisons
        ax.grid()


def concatenate_popanals_flexible(list_pa, concat_dim="trials"):
    """ Concatenates popanals (along trial dim) which may have different time bases (but
    the same n time bins.
    If differnet, then replaces time with 0,1, 2... (index), otherwise uses actual time.
    RETURNS:
        - PA, new popanal
        - twind, (tmin, tmax) from new PA.Times.
    """
    assert len(list_pa)>0, "didnt get any data"

    if concat_dim=="trials":
        # if you are combining multiple times, then replace times iwth a
        # dummy variable
        times_identical = check_identical_times(list_pa)
        replace_times_with_dummy_variable = not times_identical

        PA = concatenate_popanals(list_pa, "trials",
                                    replace_times_with_dummy_variable=replace_times_with_dummy_variable)

    elif concat_dim=="times":
        # ALl will copy the trials df from the first pa.
        PA = concatenate_popanals(list_pa, "times",
                                     all_pa_inherit_trials_of_pa_at_this_index=0,
                                     replace_times_with_dummy_variable=False)

    elif concat_dim=="chans":
        PA = concatenate_popanals(list_pa, "chans",
                                  all_pa_inherit_trials_of_pa_at_this_index=0)
    else:
        print(concat_dim)
        assert False

    # if times were replaced, what is new fake time window?
    twind = (PA.Times[0], PA.Times[-1])

    return PA, twind

def concatenate_popanals(list_pa, dim, values_for_concatted_dim=None, 
    map_idxpa_to_value=None, map_idxpa_to_value_colname = None, 
    assert_otherdims_have_same_values=True, 
    assert_otherdims_restrict_to_these=("chans", "trials", "times"),
    all_pa_inherit_times_of_pa_at_this_index=None,
     replace_times_with_dummy_variable=False,
     all_pa_inherit_trials_of_pa_at_this_index=None):
    """ Concatenate multiple popanals. They must have same shape except
    for the one dim concatted along.
    PARAMS:
    - list_pa, list of PopAnal objects
    - dim, int, which dimensiion to concat along
    - values_for_concatted_dim, list of items which are labels for
    each value in the new concatted dimension. Must be apporopriate length.
    - map_idxpa_to_value, dict or list, mapping from index in list_pa toa value, 
    used to populate a new column named map_idxpa_to_value_colname
    - map_idxpa_to_value_colname, str, name of new col, mapping (se ablve0)
    - assert_otherdims_have_same_values, if True, then makes sure that the values in 
    other dims are identical across pa in list_pa. e.g., pa.Chans is same, if you
    are concatting across trials. Useful as a sanity cehck. Otherwise takes values if
    they are the same across pa, otherwise lsit of Nones
    the first pa.
    - all_pa_inherit_times_of_pa_at_this_index, either None (does nothign) or int, which
    is index into list_pa. all pa will be forced to use pa.Times from this pa. Useful if thye
    have differnet time bases, but you really just care about realtive time to alignment.
    replace_times_with_dummy_variable, bool, if True, then reaplces all times with indices
    0, 1.,,,
    RETURNS:
    - PopAnal object,
    --- or None, if inputed list_pa is empty.
    NOTE:
    - for the non-concatted dim, will use the values for the first pa. assumes
    this is same across pa.
    """

    if len(list_pa)==0:
        return None

    # always copy
    list_pa = [pa.copy() for pa in list_pa]

    dim, dim_str = help_get_dimensions(dim)

    # Sometimes times are len 1 off from each other. Here is quick fix.
    # If any trials are off from other by one time bin (possible, round error), then remove the last sample
    if not dim_str=="times":
        list_n = [pa.X.shape[2] for pa in list_pa]
        # n_median = int(np.round(np.mean((list_n))))
        n_min = int(min(list_n)) # use min since can only prune not append. 
        for i, pa in enumerate(list_pa):
            if pa.X.shape[2]==n_min+1:
                # then too long by one. prune it.
                list_pa[i] = pa.slice_by_dim_indices_wrapper("times", [0, -2]) # takes inclusive from self.Times[0] to self.Times[-2]
                # list_pa[i] = pa.slice_time_by_indices(0, -2)
            # elif pa.X.shape[2]==n_min+2:
            #     # then too long by two, e.g, onset and offset? (assumed so, but checks later). prune it.
            #     print(pa.Times)
            #     list_pa[i] = pa.slice_by_dim_indices_wrapper("times", [1, -2])
            #     # list_pa[i] = pa.slice_time_by_indices(0, -2)
            elif not pa.X.shape[2]==n_min:
                print(list_n)
                assert False, "time bins are not smae length acrfoss all pa..."
            else:
                pass

    if all_pa_inherit_times_of_pa_at_this_index is not None:
        assert replace_times_with_dummy_variable==False
        pa_base = list_pa[all_pa_inherit_times_of_pa_at_this_index]
        for pa in list_pa:
            assert len(pa.Times)==len(pa_base.Times)
            pa.Times = pa_base.Times

    if replace_times_with_dummy_variable:
        assert all_pa_inherit_times_of_pa_at_this_index is None
        for pa in list_pa:
            pa.Times = np.arange(len(pa.Times))

    # 1) Concat the data
    list_x = [pa.X for pa in list_pa]
    # for x in list_x:
    #     print(x.shape)
    X = np.concatenate(list_x, axis=dim)

    # 2) Create new PA
    if False:
        # [OLD METHOD] values for the non-concatted dimensions.
        pa1 = list_pa[0]
        chans = pa1.Chans
        trials = pa1.Trials
        times = pa1.Times

    # Extract values to populate the other dimensions
    # - decide whether to enforce same values across all list_pa.
    tmp = {}

    for d in ["times", "chans", "trials"]:
        tmp[d] = (assert_otherdims_have_same_values) and (d in assert_otherdims_restrict_to_these)
    # - extract values
    if dim_str=="times":
        times = values_for_concatted_dim
        if times is None:
            # Make times = [(i, t) ...] where i is PA index, and t is the time iun PA.times.
            times =[]
            for i, patmp in enumerate(list_pa):
                # times.extend([(i, t) for t in patmp.Times])
                times.extend([f"{i}|{t:.4f}" for t in patmp.Times])
        chans = check_get_common_values_this_dim(list_pa,"chans", tmp["chans"])
        trials = check_get_common_values_this_dim(list_pa, "trials", tmp["trials"])
    elif dim_str=="chans":
        times = check_get_common_values_this_dim(list_pa, "times", tmp["times"])
        chans = values_for_concatted_dim
        trials = check_get_common_values_this_dim(list_pa, "trials", tmp["trials"])
    elif dim_str=="trials":
        times = check_get_common_values_this_dim(list_pa, "times", tmp["times"])
        chans = check_get_common_values_this_dim(list_pa, "chans", tmp["chans"])
        trials = values_for_concatted_dim   
    else:
        print(dim_str)
        assert False

    # Generate new popanal
    PA = PopAnal(X, times=times, trials = trials,
        chans = chans)

    # Concatenate Xlabels dataframe
    # - concat the dimension chosen
    from pythonlib.tools.pandastools import concat
    #
    # PA.Xlabels = self.Xlabels.copy()
    list_df = [pa.Xlabels[dim_str] for pa in list_pa]
    PA.Xlabels[dim_str] = concat(list_df)

    # Inherit the df from a specific idx.
    if all_pa_inherit_trials_of_pa_at_this_index is not None:
        # Sanity chcek that all dfs have same trialcodes.
        tcs = list_pa[0].Xlabels["trials"]["trialcode"].tolist()
        for patmp in list_pa[1:]:
            assert patmp.Xlabels["trials"]["trialcode"].tolist() == tcs

        # Replace
        dftmp = list_pa[all_pa_inherit_trials_of_pa_at_this_index].Xlabels["trials"].copy()
        PA.Xlabels["trials"] = dftmp

    # convert index_to_old_dataframe to meaningful value
    if map_idxpa_to_value is not None:
        assert map_idxpa_to_value_colname is not None
        inds = PA.Xlabels[dim_str]["idx_df_orig"].tolist()
        new_val = [map_idxpa_to_value[i] for i in inds]
        PA.Xlabels[dim_str][map_idxpa_to_value_colname] = new_val

    return PA

def help_get_dimensions(dim):
    """ Hleper to convert between types for dim (int or str)
    PARAMS;
    - dim, eitehr int {0,1,2} or str {'chans', 'trials', 'times'}, euivalent
    RETURNS:
    - dim(int), dim_str(string)
    """
    
    MapDimensionsStrToInt = {
        "chans":0,
        "trials":1,
        "times":2
    }
    MapDimensionsIntToStr = {
        0:"chans",
        1:"trials",
        2:"times"
    }


    if isinstance(dim, str):
        dim_str = dim
        dim = MapDimensionsStrToInt[dim]
    elif isinstance(dim, int):
        dim_str = MapDimensionsIntToStr[dim]
    else:
        assert False
    return dim, dim_str


def check_identical_times(list_pa):
    """ Returns True if all pa in list_pa have
    smae time base (in pa.Times), wihtin numerical precision
    """
    from pythonlib.tools.nptools import isin_array

    times_prev = None
    times_identical = True
    for pa in list_pa:
        if times_prev is not None:
            if len(pa.Times) != len(times_prev):
                times_identical = False
                break
            if not isin_array(pa.Times, [times_prev]):
                times_identical = False
                break
        times_prev = pa.Times
    return times_identical


def check_get_common_values_this_dim(list_pa, thisdim, assert_all_pa_have_same_values,
                                     dims_are_columns_in_xlabels=False):
    """ collect the values for this dim, across all pa,.
    returns as a list of lists, each inner list unique.
    PARAMS:
    - thisdim, e.g., "trials"
    - assert_all_pa_have_same_values, bool, if True, then fails if not unique values across
    all pa.
    RETURNS:
        - list of values, the unique values. if no unique, thje returns list ofNones.
    """

    def _isin(vals, list_vals):
        # Returns True if vals is in list_vals.

        # solve bug if this is list of np arraus,.
        # vals = list(vals)
        # list_vals = list(list_vals)

        if isinstance(vals, list):
            return vals in list_vals
        elif isinstance(vals, np.ndarray) and not isinstance(vals[0], str):
            from pythonlib.tools.nptools import isin_array
            return isin_array(vals, list_vals)
            # vals_in = [np.all(vals == vals_check) for vals_check in list_vals]
            # return any(vals_in)
        else:
            print(type(vals))
            assert False

    # Collect vals
    vals_all = []
    for pa in list_pa:
        if dims_are_columns_in_xlabels:
            vals_this = pa.Xlabels["trials"][thisdim].tolist()
        else:
            if thisdim=="chans":
                vals_this = pa.Chans
            elif thisdim=="trials":
                vals_this = pa.Trials
            elif thisdim=="times":
                vals_this = pa.Times
            else:
                assert False

        if isinstance(vals_this, np.ndarray) and isinstance(vals_this[0], str):
            # Make this list of str, not array, or else will have bug downstream in _isin
            vals_this = ([v for v in vals_this])

        if not _isin(vals_this, vals_all):
            # This is new. collect it.
            vals_all.append(vals_this)

    # Decide whether they have the same vals.
    if len(vals_all)==1:
        # then good, all pa have same vals. keep iot
        vals_good = vals_all[0]
    elif len(vals_all)==0:
        assert False, "how is this possible../."
    else:
        # then pas have different vals.
        if assert_all_pa_have_same_values:
            print("this dimensions has diff values across pa...")
            print(thisdim)
            for vals in vals_all:
                print(len(vals), vals)
            assert False

        # replace with list of None, so dont get confused.
        n_all = list(set([len(vals) for vals in vals_all]))
        assert len(n_all)==1, "not supposed to be able to try to concat pa that have diff sizes for this dim"
        n = n_all[0]
        vals_good = [None for _ in range(n)]

    return vals_good

